"""Sparse feasible active-set projection with equality-basis bookkeeping.

Solve

    min 1/2 ||y - t*b||^2   subject to B.T@y = d, y >= 0.

This is the SBK-QP-2 prototype from assets/24.  It deliberately has no fallback
to the original LP: every PROJECTED result carries an original-data KKT check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.optimize import linprog
from scipy.sparse import bmat, csc_matrix, diags, eye
from scipy.sparse.linalg import splu


TOL = 1e-9


def _emit(trace, event, **fields):
    if trace is None:
        return
    record = {"event": event, **fields}
    trace(record) if callable(trace) else trace.append(record)


def _original_errors(B, d, y):
    residual = B.T @ y - d
    scale = 1.0 + np.abs(d) + np.abs(B).T @ np.abs(y)
    equality = float(np.max(np.abs(residual) / scale)) if residual.size else 0.0
    yscale = max(1.0, float(np.abs(y).max()) if y.size else 1.0)
    nonnegative = max(0.0, float(-y.min()) if y.size else 0.0) / yscale
    return equality, nonnegative


def reduce_equalities_sparse(B, d):
    """Select independent scaled rows of A=B.T without densifying the result."""
    A = csc_matrix(np.asarray(B, dtype=float).T)
    d = np.asarray(d, dtype=float)
    row_norm = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    row_scale = np.where(row_norm > 0.0, row_norm, 1.0)
    As = diags(1.0 / row_scale) @ A
    fs = d / row_scale
    if As.shape[0] == 0:
        return As, fs, np.zeros(0, dtype=int), 0.0

    # The one-time RRQR is only for row selection.  The selected system remains
    # sparse for every face factorization.
    _, R, pivot = linalg.qr(As.T.toarray(), mode="economic", pivoting=True,
                            check_finite=True)
    diagonal = np.abs(np.diag(R))
    cutoff = ((diagonal[0] if diagonal.size else 0.0)
              * max(As.shape) * np.finfo(float).eps)
    rank = int(np.sum(diagonal > cutoff))
    selected = np.asarray(pivot[:rank], dtype=int)
    E = csc_matrix(As[selected, :])
    f = fs[selected]

    # Check that the discarded equations are represented consistently.
    if rank:
        coefficients, *_ = np.linalg.lstsq(E.toarray().T, As.toarray().T,
                                            rcond=None)
        represented_rhs = coefficients.T @ f
    else:
        represented_rhs = np.zeros_like(fs)
    consistency = (float(np.linalg.norm(represented_rhs - fs, np.inf))
                   / max(1.0, float(np.linalg.norm(fs, np.inf))))
    return E, f, selected, consistency


class EqualityBasis:
    """Maintain a nonsingular r-column submatrix of E through bound pivots."""

    def __init__(self, E, free, pivot_tol=1e-11):
        self.E = csc_matrix(E)
        self.r, self.n = self.E.shape
        self.pivot_tol = pivot_tol
        self.indices = []
        self.lu = None
        if self.r == 0:
            return
        candidates = np.where(free)[0]
        _, R, pivot = linalg.qr(self.E[:, candidates].toarray(),
                                mode="economic", pivoting=True,
                                check_finite=True)
        diagonal = np.abs(np.diag(R))
        cutoff = ((diagonal[0] if diagonal.size else 0.0)
                  * max(self.E[:, candidates].shape) * np.finfo(float).eps)
        rank = int(np.sum(diagonal > cutoff))
        if rank != self.r:
            raise np.linalg.LinAlgError("free set does not span equality rows")
        self.indices = [int(candidates[j]) for j in pivot[:self.r]]
        self._refactor()

    def _refactor(self):
        if self.r:
            self.lu = splu(csc_matrix(self.E[:, self.indices]),
                           permc_spec="COLAMD", diag_pivot_thresh=0.1)

    def _eligible_replacement(self, position, free, leaving):
        in_basis = set(self.indices)
        # Prefer an already-free column; releasing a bound is a last resort.
        pools = (np.where(free)[0], np.where(~free)[0])
        for pool in pools:
            for candidate in pool:
                candidate = int(candidate)
                if candidate == leaving or candidate in in_basis:
                    continue
                column = self.E[:, candidate].toarray().ravel()
                tableau = self.lu.solve(column)
                scale = max(1.0, float(np.linalg.norm(tableau, np.inf)))
                if abs(float(tableau[position])) > self.pivot_tol * scale:
                    return candidate
        return None

    def activate_bound(self, leaving, free):
        """Set leaving nonfree, exchanging its basis column if necessary."""
        leaving = int(leaving)
        if leaving not in self.indices:
            free[leaving] = False
            return True, None
        position = self.indices.index(leaving)
        replacement = self._eligible_replacement(position, free, leaving)
        if replacement is None:
            return False, None
        was_active = not bool(free[replacement])
        trial = list(self.indices)
        trial[position] = replacement
        old_indices, old_lu = self.indices, self.lu
        try:
            self.indices = trial
            self._refactor()
        except RuntimeError:
            self.indices, self.lu = old_indices, old_lu
            return False, None
        free[leaving] = False
        if was_active:
            free[replacement] = True
        return True, replacement if was_active else None


@dataclass
class FaceFactor:
    K: object
    lu: object
    free_indices: np.ndarray
    equality_rank: int

    def solve(self, rhs):
        return self.lu.solve(rhs)

    def equality_sensitivity(self):
        nf = self.free_indices.size
        r = self.equality_rank
        if r == 0:
            return np.zeros((nf, 0))
        rhs = np.zeros((nf + r, r))
        rhs[nf:, :] = np.eye(r)
        return self.solve(rhs)[:nf, :]


def factor_face(E, f, c, free, refinement_steps=3,
                equilibrate_rows=False):
    original_E = csc_matrix(E)
    original_f = np.asarray(f, dtype=float)
    indices = np.where(free)[0]
    original_EF = csc_matrix(original_E[:, indices])
    nf, r = indices.size, original_E.shape[0]
    if equilibrate_rows:
        row_norm = np.sqrt(np.asarray(
            original_EF.multiply(original_EF).sum(axis=1)).ravel())
        row_scale = 1.0 / np.maximum(
            row_norm, np.sqrt(np.finfo(float).tiny))
        row_scale = np.clip(row_scale, 1e-12, 1e12)
    else:
        row_scale = np.ones(r)
    EF = diags(row_scale, format="csc") @ original_EF
    scaled_f = row_scale * original_f
    K = bmat([[eye(nf, format="csc"), EF.T],
              [EF, csc_matrix((r, r))]], format="csc")
    rhs = np.concatenate([c[indices], scaled_f])
    rhs_scale = max(1.0, float(np.linalg.norm(rhs, np.inf)))
    best = None
    # This solve is already the exceptional repair for an unusable incremental
    # reduced Gram factor.  Sparse LU quality can vary sharply with ordering on
    # degenerate faces, so try a small deterministic portfolio and retain the
    # solution with the smallest checked residual.  Ordinary pivots still use
    # product-form/rank-one updates; this does not reintroduce dense SVD work.
    for ordering, pivot_threshold in (
            ("COLAMD", 0.1), ("COLAMD", 1.0),
            ("MMD_AT_PLUS_A", 0.1), ("NATURAL", 0.1)):
        try:
            lu = splu(K, permc_spec=ordering,
                      diag_pivot_thresh=pivot_threshold)
            solution = lu.solve(rhs)
            for _ in range(refinement_steps):
                residual = rhs - K @ solution
                if (float(np.linalg.norm(residual, np.inf))
                        <= 20.0 * np.finfo(float).eps * rhs_scale):
                    break
                solution += lu.solve(residual)
            residual = rhs - K @ solution
            error = float(np.linalg.norm(residual, np.inf)) / rhs_scale
        except (RuntimeError, ValueError):
            continue
        if best is None or error < best[0]:
            best = error, solution, lu
        if error <= 1e-10:
            break
    if best is None:
        raise np.linalg.LinAlgError("all sparse KKT factorizations failed")
    error, solution, lu = best
    y = np.zeros(original_E.shape[1])
    y[indices] = solution[:nf]
    multiplier = row_scale * solution[nf:]
    return y, multiplier, FaceFactor(K, lu, indices, r), error


def factor_face_relative(E, f, delta_c, free, anchor_z,
                         anchor_multiplier, anchor_stationarity,
                         refinement_steps=6):
    """Solve one identical projection face in anchor-relative coordinates.

    For ``z=anchor_z+dz`` and ``lambda=anchor_multiplier+dlambda``, solve

        dz_F + E_F' dlambda = delta_c_F - anchor_stationarity_F
        E_F dz_F = f - E_F anchor_z_F,

    with inactive coordinates of ``z`` fixed to zero.  Equality rows are
    diagonally equilibrated only inside this linear solve.  The returned
    ``z,lambda`` remain in original coordinates and require an independent
    original-data certificate from the caller.
    """
    E = csc_matrix(E)
    f = np.asarray(f, dtype=float)
    delta_c = np.asarray(delta_c, dtype=float)
    free = np.asarray(free, dtype=bool)
    anchor_z = np.asarray(anchor_z, dtype=float)
    anchor_multiplier = np.asarray(anchor_multiplier, dtype=float)
    anchor_stationarity = np.asarray(anchor_stationarity, dtype=float)
    r, n = E.shape
    if (f.shape != (r,) or delta_c.shape != (n,) or free.shape != (n,)
            or anchor_z.shape != (n,)
            or anchor_multiplier.shape != (r,)
            or anchor_stationarity.shape != (n,)):
        raise ValueError("relative face solve has inconsistent shapes")
    indices = np.where(free)[0]
    EF = csc_matrix(E[:, indices])
    nf = indices.size
    row_norm = np.sqrt(np.asarray(EF.multiply(EF).sum(axis=1)).ravel())
    row_scale = 1.0 / np.maximum(row_norm, np.sqrt(np.finfo(float).tiny))
    row_scale = np.clip(row_scale, 1e-12, 1e12)
    R = diags(row_scale, format="csc")
    scaled_E = R @ EF
    rhs_stationarity = (delta_c[indices]
                        - anchor_stationarity[indices])
    rhs_equality = f - np.asarray(EF @ anchor_z[indices]).ravel()
    K = bmat([[eye(nf, format="csc"), scaled_E.T],
              [scaled_E, csc_matrix((r, r))]], format="csc")
    rhs = np.concatenate([rhs_stationarity, row_scale * rhs_equality])
    rhs_scale = max(1.0, float(np.linalg.norm(rhs, np.inf)))
    best = None
    for ordering, pivot_threshold in (
            ("COLAMD", 0.1), ("COLAMD", 1.0),
            ("MMD_AT_PLUS_A", 0.1), ("NATURAL", 0.1)):
        try:
            lu = splu(K, permc_spec=ordering,
                      diag_pivot_thresh=pivot_threshold)
            solution = lu.solve(rhs)
            for _ in range(refinement_steps):
                residual = rhs - K @ solution
                if (float(np.linalg.norm(residual, np.inf))
                        <= 20.0 * np.finfo(float).eps * rhs_scale):
                    break
                solution += lu.solve(residual)
            residual = rhs - K @ solution
            error = float(np.linalg.norm(residual, np.inf)) / rhs_scale
        except (RuntimeError, ValueError):
            continue
        if best is None or error < best[0]:
            best = error, solution, lu
        if error <= 1e-12:
            break
    if best is None:
        raise np.linalg.LinAlgError(
            "all relative sparse KKT factorizations failed")
    error, solution, lu = best
    z = np.zeros(n)
    z[indices] = anchor_z[indices] + solution[:nf]
    multiplier = anchor_multiplier + row_scale * solution[nf:]
    return z, multiplier, FaceFactor(K, lu, indices, r), error


def _lexicographic_order(tied, ratios, direction, factor):
    """Order primary-ratio ties by equality-RHS symbolic sensitivities."""
    tied = [int(i) for i in tied]
    if len(tied) <= 1 or factor.equality_rank == 0:
        return tied
    sensitivity = factor.equality_sensitivity()
    local = {int(g): k for k, g in enumerate(factor.free_indices)}

    def key(index):
        denominator = max(np.finfo(float).tiny, float(-direction[index]))
        return sensitivity[local[index], :] / denominator

    ordered = []
    for candidate in sorted(tied):
        candidate_key = key(candidate)
        position = len(ordered)
        for j, incumbent in enumerate(ordered):
            incumbent_key = key(incumbent)
            difference = candidate_key - incumbent_key
            decisive = np.where(np.abs(difference) > 64.0 * np.finfo(float).eps
                                * (1.0 + np.maximum(np.abs(candidate_key),
                                                   np.abs(incumbent_key))))[0]
            if decisive.size and difference[int(decisive[0])] < 0.0:
                position = j
                break
        ordered.insert(position, candidate)
    return ordered


def projection_kkt_certificate(B, b, d, y, t=1.0, tol=1e-7):
    """Independent original-data KKT certificate with multiplier LPs.

    The first LP minimizes one relaxation for bound-dual feasibility and scaled
    complementarity without choosing a support.  On badly scaled instances its
    single relaxation can stop just above tolerance even when an accurate
    multiplier exists.  In that case, try several support hypotheses and accept
    one only after an original-data residual check; the hypothesis selects a
    candidate multiplier but is never itself evidence of correctness.
    """
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = B.shape
    c = t * b
    equality, nonnegative = _original_errors(B, d, y)
    # v = y-c+B*u is the bound multiplier.  Require v>=0 and y*v=0,
    # both with componentwise residual scaling.
    vscale = 1.0 + np.abs(y) + np.abs(c)
    yp = np.maximum(y, 0.0)
    comp_scale = 1.0 + yp * vscale
    base = y - c
    rows = [np.column_stack([-B, -vscale]),
            np.column_stack([yp[:, None] * B, -comp_scale]),
            np.column_stack([-yp[:, None] * B, -comp_scale])]
    rhs = [base, -yp * base, yp * base]
    A_ub = np.vstack(rows)
    b_ub = np.concatenate(rhs)
    objective = np.zeros(m + 1)
    objective[-1] = 1.0
    result = linprog(objective, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(None, None)] * m + [(0.0, None)],
                     method="highs-ds")
    if not result.success:
        return {"passed": False, "equality_error": equality,
                "nonnegative_error": nonnegative,
                "stationarity_error": np.inf,
                "certificate_status": int(result.status)}
    stationarity = float(result.x[-1])
    if max(equality, nonnegative, stationarity) <= tol:
        return {"passed": True, "equality_error": equality,
                "nonnegative_error": nonnegative,
                "stationarity_error": stationarity,
                "multiplier": result.x[:-1],
                "certificate_backend": "relaxed multiplier LP"}

    yscale = max(1.0, float(np.abs(y).max()) if y.size else 1.0)
    best = stationarity
    best_u = result.x[:-1]
    for relative_threshold in (1e-12, 1e-10, 1e-8, 1e-6):
        positive = y > relative_threshold * yscale
        zero = ~positive
        face = linprog(
            np.zeros(m),
            A_ub=-B[zero] if zero.any() else None,
            b_ub=(y - c)[zero] if zero.any() else None,
            A_eq=B[positive] if positive.any() else None,
            b_eq=(c - y)[positive] if positive.any() else None,
            bounds=[(None, None)] * m,
            method="highs-ds")
        if not face.success:
            continue
        u = np.asarray(face.x)
        bound_multiplier = y - c + B @ u
        product_scale = (1.0 + np.abs(y) *
                         (1.0 + np.abs(y) + np.abs(c)
                          + np.abs(B) @ np.abs(u)
                          + np.abs(bound_multiplier)))
        dual_scale = (1.0 + np.abs(y) + np.abs(c)
                      + np.abs(B) @ np.abs(u))
        bound_error = float(np.max(
            np.maximum(-bound_multiplier, 0.0) / dual_scale))
        complementarity_error = float(np.max(
            np.abs(y * bound_multiplier) / product_scale))
        checked = max(bound_error, complementarity_error)
        if checked < best:
            best, best_u = checked, u
        if max(equality, nonnegative, checked) <= tol:
            return {"passed": True, "equality_error": equality,
                    "nonnegative_error": nonnegative,
                    "stationarity_error": checked,
                    "bound_dual_error": bound_error,
                    "complementarity_error": complementarity_error,
                    "multiplier": u,
                    "support_threshold": relative_threshold,
                    "certificate_backend": "verified face multiplier LP"}
    return {"passed": False, "equality_error": equality,
            "nonnegative_error": nonnegative,
            "stationarity_error": best,
            "multiplier": best_u,
            "certificate_backend": "multiplier LPs failed"}


def project_sparse_basis(B, b, d, t=1.0, tol=TOL, maxiter=5000, trace=None):
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    n, m = B.shape
    if b.shape != (n,) or d.shape != (m,) or not np.all(np.isfinite(B)):
        raise ValueError("invalid projection dimensions or nonfinite matrix")
    E, f, selected, consistency = reduce_equalities_sparse(B, d)
    if consistency > 100.0 * tol:
        return {"status": "EQUALITY INCONSISTENT", "iterations": 0,
                "consistency_error": consistency}

    phase = linprog(np.zeros(n), A_eq=E if E.shape[0] else None,
                    b_eq=f if E.shape[0] else None, bounds=(0.0, None),
                    method="highs-ds")
    if not phase.success:
        return {"status": "C INFEASIBLE", "iterations": 0,
                "phase_status": int(phase.status)}
    y = np.asarray(phase.x, dtype=float)
    phase_scale = max(1.0, float(np.abs(y).max()))
    y[np.abs(y) <= tol * phase_scale] = 0.0
    free = np.ones(n, dtype=bool)
    try:
        basis = EqualityBasis(E, free)
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as error:
        return {"status": "BASIS FAILURE", "iterations": 0,
                "detail": str(error)}

    c = t * b
    seen = set()
    exchanges = lex_ties = refinements = 0
    for iteration in range(maxiter):
        state = np.packbits(free).tobytes()
        if state in seen:
            certificate = projection_kkt_certificate(B, b, d, y, t=t)
            return {"status": "CYCLE", "iterations": iteration, "y": y,
                    "working_set": free, "basis": np.asarray(basis.indices),
                    "basis_exchanges": exchanges, "lex_ties": lex_ties,
                    **certificate}
        seen.add(state)
        try:
            z, multiplier, factor, factor_error = factor_face(E, f, c, free)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            return {"status": "FACTORIZATION FAILURE", "iterations": iteration,
                    "y": y, "working_set": free, "detail": str(error),
                    "basis_exchanges": exchanges, "lex_ties": lex_ties}
        refinements += 1
        if factor_error > 1e-8:
            return {"status": "KKT SOLVE RESIDUAL", "iterations": iteration,
                    "factor_error": factor_error, "y": y,
                    "working_set": free, "basis_exchanges": exchanges,
                    "lex_ties": lex_ties}

        face_scale = max(1.0, float(np.abs(z).max()))
        negative = free & (z < -100.0 * tol * face_scale)
        if negative.any():
            direction = z - y
            blockers = np.where(negative & (direction < 0.0))[0]
            if not blockers.size:
                return {"status": "NO BLOCKER", "iterations": iteration,
                        "y": y, "working_set": free}
            ratios = np.maximum(y[blockers], 0.0) / (-direction[blockers])
            minimum = float(ratios.min())
            tie_window = 64.0 * np.finfo(float).eps * max(1.0, abs(minimum))
            tied = blockers[ratios <= minimum + tie_window]
            if tied.size > 1:
                lex_ties += 1
            order = _lexicographic_order(tied, ratios, direction, factor)
            leaving = None
            replacement = None
            for candidate in order:
                accepted, proposed = basis.activate_bound(candidate, free)
                if accepted:
                    leaving, replacement = int(candidate), proposed
                    break
            if leaving is None:
                return {"status": "RANK-ESSENTIAL BLOCKER",
                        "iterations": iteration, "y": y,
                        "working_set": free, "basis_exchanges": exchanges,
                        "lex_ties": lex_ties}
            candidate_position = int(np.where(blockers == leaving)[0][0])
            alpha = min(1.0, max(0.0, float(ratios[candidate_position])))
            y = y + alpha * direction
            y[leaving] = 0.0
            if replacement is not None:
                y[replacement] = 0.0
                exchanges += 1
            _emit(trace, "activate", iteration=iteration, index=leaving,
                  replacement=replacement, alpha=alpha,
                  free=int(free.sum()), factor_error=factor_error)
            continue

        y = z
        active = ~free
        reduced = (-c[active] + E[:, active].T @ multiplier
                   if active.any() else np.zeros(0))
        reduced = np.asarray(reduced).ravel()
        scale = (1.0 + np.abs(c[active])
                 + (np.asarray(np.abs(E[:, active]).T @ np.abs(multiplier)).ravel()
                    if active.any() else 0.0))
        scaled = reduced / scale if reduced.size else reduced
        active_indices = np.where(active)[0]
        violated = active_indices[scaled < -tol]
        if violated.size:
            entering = int(violated.min())
            free[entering] = True
            y[entering] = 0.0
            _emit(trace, "release", iteration=iteration, index=entering,
                  reduced=float(scaled[np.where(active_indices == entering)[0][0]]),
                  free=int(free.sum()))
            continue

        certificate = projection_kkt_certificate(B, b, d, y, t=t)
        status = "PROJECTED" if certificate["passed"] else "FALSE KKT"
        return {"status": status, "iterations": iteration, "y": y,
                "support": y > 1e-7 * max(1.0, float(np.abs(y).max())),
                "working_set": free, "basis": np.asarray(basis.indices),
                "selected_equalities": selected,
                "basis_exchanges": exchanges, "lex_ties": lex_ties,
                "factorizations": refinements, **certificate}

    certificate = projection_kkt_certificate(B, b, d, y, t=t)
    return {"status": "ITERATION CAP", "iterations": maxiter, "y": y,
            "working_set": free, "basis_exchanges": exchanges,
            "lex_ties": lex_ties, "factorizations": refinements,
            **certificate}
