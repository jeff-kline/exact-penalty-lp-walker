"""Factor-updating face solver: update-with-monitored-refactorization.

The walker's dominant cost on the hard models is the persistent-QR face
solve (`PathPieceSkeletonFactor.solve`): 68.8 % of wall on fit1d and 73.0 %
on ship04s (agent_reports/11 Addendum 14).  Instrumenting that call on
fit1d (m = 1026, face rows ~ 1030-1038, rank = m on 771 of 777 solves)
splits the 0.360 s mean solve into

    fresh pivoted QR rebuild          0.166 s   (193 of 389 solves)
    Q @ R factor verification         0.109 s   (195 calls)
    two GELSY complete-orthogonal     0.209 s   (777 calls, 1026x1026)
    incremental qr_insert/qr_delete   0.005 s

so the *update* primitive is already two orders of magnitude cheaper than
everything built on top of it.  Three measured facts drive this module.

1.  `PathPieceFactor._update` passes ``rcond=1e-12`` to
    ``scipy.linalg.qr_insert(..., which="row")``.  SciPy 1.13 rejects that
    ("'rcond' is unused when inserting rows and must be None"), the
    ``except (ValueError, LinAlgError)`` arm swallows it, and **every row
    ADDITION silently falls back to a full pivoted refactorization**.  On
    fit1d that is 137 of 280 supports.  This module inserts rows without
    ``rcond`` and keeps them.
2.  The incumbent verifies each update with ``Q @ R`` against the original
    face — an O(rows x rank x m) product costing 0.109 s, two thirds of the
    refactorization it is protecting against, and it never once fired
    (max factor error 9.8e-14 over 140 checks, 0 % above the 1e-9 gate).
    This module drops that check and monitors the *honest piece residual*
    on original data instead, which the incumbent already computes anyway.
3.  When the face has full column rank the maintained ``R`` is square and
    nonsingular, so every quantity `piece_y` returns is a triangular solve.
    The incumbent instead runs two fresh GELSY decompositions of that same
    ``R`` on every face — i.e. it re-factorizes the factor it just updated.

The safety discipline is the point, not the speed: a factor that has been
updated is trusted only until its own solve fails an honest residual test
against original data, and never for more than ``refactor_cadence``
updates.  Downdating (row deletion) is refused outright on an
ill-conditioned factor.

Contract: ``solve(W)`` returns ``(g, h, ua, uc, dres)`` with exactly
`exp23_path_primal_dual.piece_y` semantics --

    y_W(t) = t*g + h,  u(t) = t*ua + uc,
    g = (I - P_range(B_W)) b_W,   B_W' h = d  (min-norm),
    B_W ua = g - b_W  (min-norm),  B_W uc = h  (min-norm),
    dres = ||B_W' h - d|| / max(1, ||d||).

Any face this module cannot solve inside its trust budget raises
`FactorUpdatingFallback`; the caller then runs the unchanged incumbent
path, so no acceptance gate is relaxed and no certificate changes.
"""

from __future__ import annotations

from bisect import bisect_left
import time

import numpy as np
from scipy import linalg


class FactorUpdatingFallback(RuntimeError):
    """This face is declined; the caller must use the unchanged path."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class FactorUpdatingFaceSolver:
    """Persistent economy QR of ``B[W]`` maintained by row insert/delete.

    ``refactor_cadence`` bounds how many row updates may accumulate before a
    fresh pivoted factorization is taken regardless of measured quality
    (the "monitored refactorization" half of the classical discipline).

    ``downdate_diagonal_floor`` is the diagonal ratio below which a row
    deletion is refused and replaced by a refactorization.  Downdating is
    the numerically dangerous direction: it removes information, and on an
    ill-conditioned triangular factor the removed component cannot be
    recovered accurately.  The walker's drift guard already treats a
    diagonal ratio under 1e-5 as suspicious; 1e-7 is the tighter
    "do not even try to downdate" floor.

    ``trust_residual`` is the honest-residual budget an *updated* factor
    must meet for its solve to be returned.  It is deliberately three
    decades tighter than ``accept_residual`` (the incumbent's own
    fail-closed gate, `path_piece_skeleton.py:144`): an updated factor is
    held to a stricter standard than a fresh one, so a solve that would
    have been merely acceptable triggers a refactorization instead of
    being handed back.  ``accept_residual`` remains the final fail-closed
    gate and matches the incumbent bit for bit.

    ``max_coordinate_changes`` bounds how many support coordinates may move
    in one event and still be handled by updating.  Consecutive walker
    events move the support by one coordinate; anything larger is a repair,
    a resync, or a drift-guard rederivation, and those get a fresh factor
    with no exceptions.
    """

    def __init__(self, B, b, d, refactor_cadence=32,
                 downdate_diagonal_floor=1e-7, trust_residual=1e-10,
                 accept_residual=1e-7, max_coordinate_changes=2, stats=None):
        self.B = np.asarray(B, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.d = np.asarray(d, dtype=float)
        if self.B.ndim != 2:
            raise ValueError("B must be two-dimensional")
        self.n, self.m = self.B.shape
        if self.b.shape != (self.n,) or self.d.shape != (self.m,):
            raise ValueError("expected B.shape == (len(b), len(d))")
        self.refactor_cadence = int(refactor_cadence)
        self.downdate_diagonal_floor = float(downdate_diagonal_floor)
        self.trust_residual = float(trust_residual)
        self.accept_residual = float(accept_residual)
        self.max_coordinate_changes = int(max_coordinate_changes)
        if self.refactor_cadence < 1 or self.max_coordinate_changes < 0:
            raise ValueError("refactor cadence must be positive and the "
                             "coordinate-change bound nonnegative")
        self.d_norm = max(1.0, float(np.linalg.norm(self.d)))

        self.indices = np.zeros(0, dtype=int)
        self.permutation = np.arange(self.m)
        self.Q = None
        self.R = None
        self.rank = 0
        self.updates_since_refactor = 0
        self.last_diagonal_ratio = 1.0
        self.last_piece_residual = np.inf

        self.stats = stats if stats is not None else {}
        for key in _COUNTER_KEYS:
            self.stats.setdefault(key, 0)
        for key in _TIMER_KEYS:
            self.stats.setdefault(key, 0.0)

    # ------------------------------------------------------------ factor

    @staticmethod
    def _rank_from_r(R):
        """Numerical rank from the triangular diagonal.

        `PathPieceFactor._rank_from_r` normalizes by ``diagonal[0]``, which
        is the largest entry only while the pivoted ordering survives.  Row
        updates do not preserve that ordering, so normalize by the largest
        magnitude instead; the two agree exactly on a freshly pivoted
        factor and this one stays valid after updates.
        """
        diagonal = np.abs(np.diag(R))
        if not diagonal.size:
            return 0
        top = float(diagonal.max())
        if top == 0.0:
            return 0
        cutoff = top * max(R.shape) * np.finfo(float).eps
        return int(np.sum(diagonal > cutoff))

    def _diagonal_ratio(self):
        if self.R is None or not self.rank:
            return 1.0
        diagonal = np.abs(np.diag(self.R))[:self.rank]
        top = float(diagonal.max())
        if top == 0.0:
            return 0.0
        return float(diagonal.min() / top)

    def _refactor(self, indices, reason):
        started = time.perf_counter()
        face = self.B[indices]
        self.Q, self.R, pivot = linalg.qr(
            face, mode="economic", pivoting=True, check_finite=False)
        self.permutation = np.asarray(pivot, dtype=int)
        self.indices = np.asarray(indices, dtype=int).copy()
        self.rank = self._rank_from_r(self.R)
        self.updates_since_refactor = 0
        self.stats[reason] = self.stats.get(reason, 0) + 1
        self.stats["fup_refactor_seconds"] += time.perf_counter() - started

    def _sync(self, indices):
        """Move the factor to ``indices`` by row updates or refactorization.

        Freshness is *factor state* (``updates_since_refactor == 0``), never
        a per-call flag: a repeated solve on an unchanged support must stay
        subject to the trust check, because the factor underneath it may be
        thirty updates old.  The walker re-solves the same support often
        (settle rounds, drift-guard rederivations), so a per-call notion of
        freshness would silently exempt exactly those solves.
        """
        old = set(int(i) for i in self.indices)
        new = set(int(i) for i in indices)
        additions = sorted(new - old)
        removals = sorted(old - new, reverse=True)
        changes = len(additions) + len(removals)
        if changes == 0:
            return
        if changes > self.max_coordinate_changes:
            self._refactor(indices, "fup_refactor_multichange")
            return
        if self.updates_since_refactor + changes > self.refactor_cadence:
            self._refactor(indices, "fup_refactor_cadence")
            return
        # The economy QR of a rows x m face changes shape convention at
        # rows == m; keep every update strictly on the tall side.
        rows = len(old)
        if rows < self.m or rows + len(additions) - len(removals) < self.m:
            self._refactor(indices, "fup_refactor_shape")
            return
        if removals and self._diagonal_ratio() < self.downdate_diagonal_floor:
            # Refuse to downdate an ill-conditioned factor: the deleted
            # row's component is exactly what a small diagonal cannot
            # resolve, so the downdate would be the one operation whose
            # backward error is unbounded here.
            self._refactor(indices, "fup_refactor_degenerate")
            return

        started = time.perf_counter()
        current = list(int(i) for i in self.indices)
        old_rank = self.rank
        Q, R = self.Q, self.R
        try:
            for index in additions:
                position = bisect_left(current, index)
                row = self.B[index, self.permutation]
                # No ``rcond``: SciPy rejects it for row insertion, which is
                # precisely the argument that makes the incumbent's every
                # addition fall back to a full refactorization.
                Q, R = linalg.qr_insert(Q, R, row, position, which="row",
                                        overwrite_qru=False,
                                        check_finite=False)
                current.insert(position, index)
            for index in removals:
                position = current.index(index)
                Q, R = linalg.qr_delete(Q, R, position, which="row",
                                        overwrite_qr=False,
                                        check_finite=False)
                current.pop(position)
        except (ValueError, np.linalg.LinAlgError):
            self.stats["fup_update_seconds"] += time.perf_counter() - started
            self._refactor(indices, "fup_refactor_exception")
            return

        new_rank = self._rank_from_r(R)
        if new_rank != old_rank:
            self.stats["fup_update_seconds"] += time.perf_counter() - started
            self._refactor(indices, "fup_refactor_rank")
            return

        self.Q, self.R = Q, R
        self.indices = np.asarray(current, dtype=int)
        self.rank = new_rank
        self.updates_since_refactor += changes
        self.stats["fup_updates"] += len(additions)
        self.stats["fup_downdates"] += len(removals)
        self.stats["fup_update_seconds"] += time.perf_counter() - started

    # ------------------------------------------------------------- solve

    @staticmethod
    def _refined_triangular(matrix, rhs, lower):
        """Triangular solve with one checked step of iterative refinement.

        Mirrors the single refinement `PathPieceSkeletonFactor._cod_solve`
        applies to its GELSY solution, at O(k^2) instead of O(k^3).
        """
        solution = linalg.solve_triangular(matrix, rhs, lower=lower,
                                           check_finite=False)
        if not np.all(np.isfinite(solution)):
            raise FactorUpdatingFallback("nonfinite triangular solve")
        scale = 1.0 + float(np.linalg.norm(rhs))
        residual = rhs - matrix @ solution
        if float(np.linalg.norm(residual)) > 1e-12 * scale:
            correction = linalg.solve_triangular(matrix, residual,
                                                 lower=lower,
                                                 check_finite=False)
            candidate = solution + correction
            if (float(np.linalg.norm(rhs - matrix @ candidate))
                    < float(np.linalg.norm(residual))):
                solution = candidate
        return solution

    def _piece(self, indices):
        """`piece_y`'s five outputs from the maintained factor."""
        face_rows = indices.size
        rank = self.rank
        bW = self.b[indices]
        self.last_diagonal_ratio = self._diagonal_ratio()

        if rank:
            Qr = self.Q[:, :rank]
            Rr = self.R[:rank, :]
            d_permuted = self.d[self.permutation]
            qtb = Qr.T @ bW
            if rank == self.m:
                # Full column rank: Rr is square and nonsingular, so the
                # min-norm solutions are the unique ones and every solve is
                # triangular.  Rr' z = P'd, Rr ua_P = -Q'b, Rr uc_P = z.
                z = self._refined_triangular(Rr.T, d_permuted, True)
                ua_permuted = self._refined_triangular(Rr, -qtb, False)
                uc_permuted = self._refined_triangular(Rr, z, False)
            else:
                # Rank-deficient face: Rr is rank x m and wide.  One economy
                # QR of Rr' = Z T supplies the min-norm solves without
                # normal equations (same construction as
                # `PathPieceFactor.solve`).
                Z, T = linalg.qr(Rr.T, mode="economic", pivoting=False,
                                 check_finite=False)
                z = self._refined_triangular(T, Z.T @ d_permuted, False)
                ua_permuted = Z @ self._refined_triangular(T.T, -qtb, True)
                uc_permuted = Z @ self._refined_triangular(T.T, z, True)
            h = Qr @ z
            g = bW - Qr @ qtb
            g -= Qr @ (Qr.T @ g)
        else:
            self.last_diagonal_ratio = 1.0
            h = np.zeros(face_rows)
            g = bW.copy()
            ua_permuted = np.zeros(self.m)
            uc_permuted = np.zeros(self.m)

        ua = np.zeros(self.m)
        uc = np.zeros(self.m)
        ua[self.permutation] = ua_permuted
        uc[self.permutation] = uc_permuted

        # Honest residuals, from ORIGINAL data, in the incumbent's own
        # semantics (path_piece_skeleton.py:132-143).  This is the whole
        # trust mechanism: an updated factor is believed only because its
        # answer is checked against B, b and d themselves.
        face = self.B[indices]
        dres = (float(np.linalg.norm(face.T @ h - self.d)) / self.d_norm)
        orthogonality = (float(np.linalg.norm(face.T @ g, np.inf))
                         / max(1.0, float(np.linalg.norm(g, np.inf))))
        slope_residual = face @ ua - (g - bW)
        constant_residual = face @ uc - h
        slope_error = (float(np.linalg.norm(slope_residual, np.inf))
                       / max(1.0, float(np.linalg.norm(g - bW, np.inf))))
        constant_error = (float(np.linalg.norm(constant_residual, np.inf))
                          / max(1.0, float(np.linalg.norm(h, np.inf))))
        residual = max(orthogonality, slope_error, constant_error)
        arrays = (g, h, ua, uc)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            residual = np.inf
        return (g, h, ua, uc, float(dres)), float(residual)

    def solve(self, W):
        W = np.asarray(W, dtype=bool)
        if W.shape != (self.n,) or not W.any():
            raise ValueError("path support must be a nonempty Boolean mask")
        indices = np.where(W)[0]
        self.stats["fup_solves"] += 1

        if self.Q is None:
            self._refactor(indices, "fup_refactor_init")
        else:
            self._sync(indices)

        try:
            coefficients, residual = self._piece(indices)
        except (FactorUpdatingFallback, ValueError,
                np.linalg.LinAlgError) as exc:
            if not self.updates_since_refactor:
                raise FactorUpdatingFallback("fresh factor solve failed",
                                             str(exc)) from exc
            self._refactor(indices, "fup_refactor_residual")
            coefficients, residual = self._piece(indices)

        if self.updates_since_refactor and residual > self.trust_residual:
            # An updated factor answered outside its trust budget.  Never
            # return that answer: take a fresh factorization and re-solve.
            self._refactor(indices, "fup_refactor_residual")
            coefficients, residual = self._piece(indices)

        self.last_piece_residual = residual
        if residual > self.accept_residual:
            # Fail closed exactly as the incumbent does; the caller runs the
            # unchanged path for this face.
            raise FactorUpdatingFallback(
                "piece residual", f"{residual:.3e}")
        return coefficients

    # ------------------------------------------------------- diagnostics

    def diagnostics(self):
        result = {key: self.stats.get(key, 0) for key in _COUNTER_KEYS}
        result.update({key: round(float(self.stats.get(key, 0.0)), 4)
                       for key in _TIMER_KEYS})
        result["fup_refactors"] = sum(
            result[key] for key in _COUNTER_KEYS if key.startswith(
                "fup_refactor_"))
        result["fup_last_diagonal_ratio"] = float(self.last_diagonal_ratio)
        result["fup_last_piece_residual"] = float(self.last_piece_residual)
        result["fup_rank"] = int(self.rank)
        return result


_COUNTER_KEYS = (
    "fup_solves",
    "fup_updates",
    "fup_downdates",
    "fup_refactor_init",
    "fup_refactor_cadence",
    "fup_refactor_residual",
    "fup_refactor_degenerate",
    "fup_refactor_multichange",
    "fup_refactor_rank",
    "fup_refactor_shape",
    "fup_refactor_exception",
)

_TIMER_KEYS = (
    "fup_update_seconds",
    "fup_refactor_seconds",
)
