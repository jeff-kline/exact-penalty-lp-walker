"""Feasible active-set projection with sparse incremental face updates.

Solve

    min_y  1/2 ||y - t*b||^2    subject to B.T@y = d, y >= 0.

The equality system is reduced once to independent sparse rows.  The walk
starts with all bounds free, activates bounds only at segment hits, and keeps a
sparse nonsingular equality basis. Ordinary pivots update a reduced Cholesky
factor by rank one; only a genuine basis exchange refactors the sparse basis.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import clarabel
import numpy as np
from scipy import linalg
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, vstack

from exp11_netlib_rank_certificate import to_Bxgeb
from incremental_face import IncrementalFaceFactor
from sparse_basis_projection import (factor_face, projection_kkt_certificate,
                                     reduce_equalities_sparse)


TOL = 1e-9


def _emit(trace, event, **fields):
    if trace is None:
        return
    record = {"event": event, **fields}
    trace(record) if callable(trace) else trace.append(record)


def _rank(E, W, tol=None):
    if E.shape[0] == 0:
        return 0
    face = E[:, W]
    if hasattr(face, "toarray"):
        face = face.toarray()
    _, triangular, _ = linalg.qr(face, mode="economic", pivoting=True,
                                 check_finite=True)
    diagonal = np.abs(np.diag(triangular))
    if not diagonal.size:
        return 0
    cutoff = (diagonal[0] * max(E[:, W].shape) * np.finfo(float).eps
              if tol is None else tol)
    return int(np.sum(diagonal > cutoff))


def _complete_working_set(E, W, exclude=()):
    """Lexicographically add zero basic variables until E[:,W] has full row rank."""
    target = E.shape[0]
    current = _rank(E, W)
    added = []
    excluded = set(exclude)
    if current == target:
        return W, added
    for index in np.where(~W)[0]:
        if int(index) in excluded:
            continue
        trial = W.copy()
        trial[index] = True
        new_rank = _rank(E, trial)
        if new_rank > current:
            W = trial
            current = new_rank
            added.append(int(index))
            if current == target:
                return W, added
    raise np.linalg.LinAlgError("could not complete a full-rank equality basis")


def _original_errors(B, d, y):
    equality = B.T @ y - d
    scale = 1.0 + np.abs(d) + np.abs(B).T @ np.abs(y)
    equality_error = float(np.max(np.abs(equality) / scale))
    nonnegative_error = max(0.0, float(-y.min())) / (1.0 + float(np.abs(y).max()))
    return equality_error, nonnegative_error


def _objective(y, c):
    residual = y - c
    return 0.5 * float(residual @ residual)


def _tangent_descent_escape(E, c, y, tol):
    """Find coordinated feasible descent p=P_T(c-y) at a stalled vertex."""
    n = y.size
    yscale = max(1.0, float(np.abs(y).max()) if y.size else 1.0)
    zero = y <= 100.0 * tol * yscale
    active = np.where(zero)[0]
    selector = csc_matrix((-np.ones(active.size),
                           (np.arange(active.size), active)),
                          shape=(active.size, n))
    A = vstack([csc_matrix(E), selector], format="csc")
    rhs = np.zeros(E.shape[0] + active.size)
    cones = [clarabel.ZeroConeT(E.shape[0]),
             clarabel.NonnegativeConeT(active.size)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 500
    settings.tol_gap_abs = 1e-11
    settings.tol_gap_rel = 1e-11
    settings.tol_feas = 1e-11
    solution = clarabel.DefaultSolver(
        eye(n, format="csc"), -(c - y), A, rhs, cones, settings).solve()
    if "Solved" not in str(solution.status):
        return None, {"reason": "tangent QP", "status": str(solution.status)}
    p = np.asarray(solution.x)
    pscale = max(1.0, float(np.abs(p).max()) if p.size else 1.0)
    equality_error = (float(np.linalg.norm(E @ p, np.inf))
                      / max(1.0, float(np.linalg.norm(p, np.inf))))
    tangent_error = (max(0.0, float(-p[zero].min())) / pscale
                     if zero.any() else 0.0)
    predicted_decrease = float((c - y) @ p)
    pnorm2 = float(p @ p)
    if max(equality_error, tangent_error) > 100.0 * tol:
        return None, {"reason": "tangent residual", "equality": equality_error,
                      "tangent": tangent_error}
    if pnorm2 <= (100.0 * tol * pscale) ** 2 or predicted_decrease <= 0.0:
        return None, {"reason": "no tangent descent", "pnorm2": pnorm2,
                      "predicted_decrease": predicted_decrease}

    blockers = np.where(p < -100.0 * tol * pscale)[0]
    alpha = 1.0
    hit = None
    if blockers.size:
        ratios = np.maximum(y[blockers], 0.0) / (-p[blockers])
        position = int(np.argmin(ratios))
        if float(ratios[position]) < alpha:
            alpha = max(0.0, float(ratios[position]))
            hit = int(blockers[position])
    if alpha <= 100.0 * np.finfo(float).eps:
        return None, {"reason": "zero tangent step", "alpha": alpha}
    candidate = y + alpha * p
    if hit is not None:
        candidate[hit] = 0.0
    old_objective = _objective(y, c)
    new_objective = _objective(candidate, c)
    decrease = old_objective - new_objective
    if decrease <= 100.0 * np.finfo(float).eps * max(1.0, old_objective):
        return None, {"reason": "no measured decrease", "decrease": decrease}
    return candidate, {"reason": "descent", "alpha": alpha, "hit": hit,
                       "decrease": decrease, "pnorm": float(np.sqrt(pnorm2)),
                       "equality": equality_error, "tangent": tangent_error,
                       "zero": int(zero.sum())}


def project_lex_active_set(B, b, d, t=1.0, tol=TOL, maxiter=5000,
                           max_perturbations=4, max_tangent_escapes=20,
                           trace=None, y0=None,
                           warm_start_active_bounds=True):
    """Project ``t*b`` onto ``{y>=0:B.T@y=d}``.

    A result is labeled ``PROJECTED`` only after the *unperturbed* KKT system
    passes.  Perturbations may select a basis but can never certify a result.
    """
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    n, m = B.shape
    if b.shape != (n,) or d.shape != (m,):
        raise ValueError("expected B.shape == (len(b), len(d))")
    E, f, selected_equalities, consistency = reduce_equalities_sparse(B, d)
    equality_rank = E.shape[0]
    if consistency > 100.0 * tol:
        return {"status": "EQUALITY INCONSISTENT", "iterations": 0,
                "consistency_error": consistency}

    warm_start = y0 is not None
    if warm_start:
        y = np.asarray(y0, dtype=float).copy()
        if y.shape != (n,) or not np.all(np.isfinite(y)):
            raise ValueError("y0 must be a finite vector with len(y0) == B.shape[0]")
        equality_error, nonnegative_error = _original_errors(B, d, y)
        if max(equality_error, nonnegative_error) > 100.0 * tol:
            return {"status": "WARM START INFEASIBLE", "iterations": 0,
                    "equality_error": equality_error,
                    "nonnegative_error": nonnegative_error,
                    "warm_start": True}
    else:
        phase = linprog(np.zeros(n), A_eq=E if equality_rank else None,
                        b_eq=f if equality_rank else None, bounds=(0.0, None),
                        method="highs-ds")
        if not phase.success:
            return {"status": "C INFEASIBLE", "iterations": 0,
                    "phase_status": int(phase.status), "warm_start": False}
        y = np.asarray(phase.x, dtype=float)
    phase_scale = max(1.0, float(np.abs(y).max()))
    y[np.abs(y) <= tol * phase_scale] = 0.0
    if float(y.min()) < -100.0 * tol * phase_scale:
        return {"status": "PHASE NEGATIVE", "iterations": 0,
                "minimum": float(y.min())}
    y = np.maximum(y, 0.0)
    # A cold start begins on the full affine space: the line from the feasible
    # Phase-I point to the affine projection remains equality feasible.  A warm
    # start instead keeps its active bounds and adds only the zero coordinates
    # needed for a full-rank equality basis.  This preserves the fallback
    # vertex/face as an active-set crash; the tangent-cone escape remains the
    # guard for coordinated motion that cannot be exposed by one bound release.
    warm_completed = []
    if warm_start and warm_start_active_bounds:
        support_scale = max(1.0, float(np.abs(y).max()))
        W = y > 100.0 * tol * support_scale
        try:
            W, warm_completed = _complete_working_set(E, W)
        except np.linalg.LinAlgError as error:
            return {"status": "BASIS FAILURE", "iterations": 0,
                    "detail": str(error), "warm_start": True}
    else:
        W = np.ones(n, dtype=bool)
    try:
        face_factor = IncrementalFaceFactor(E, f, W)
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as error:
        return {"status": "BASIS FAILURE", "iterations": 0,
                "detail": str(error)}
    start_event = ("warm-active-set-start" if warm_start_active_bounds
                   else "warm-full-free-start") if warm_start else "full-free-start"
    _emit(trace, start_event,
          working=int(W.sum()), rank=equality_rank,
          phase_positive=int(np.sum(y > 0.0)), completed=warm_completed)

    original_c = t * b
    working_c = original_c.copy()
    perturbations = 0
    tangent_escapes = 0
    sparse_kkt_repairs = 0
    perturbed = False
    perturb_countdown = 0
    final_crossover = False
    stalled_iterations = 0
    seen = set()
    last_original_objective = _objective(y, original_c)
    index_pattern = ((np.arange(n, dtype=float) + 1.0) / max(1.0, float(n)))

    def activate_tangent_escape(reason, iteration):
        nonlocal y, W, working_c, perturbed, perturb_countdown, face_factor
        nonlocal stalled_iterations, seen, tangent_escapes
        nonlocal last_original_objective
        if tangent_escapes >= max_tangent_escapes:
            return False
        candidate, detail = _tangent_descent_escape(E, original_c, y, tol)
        if candidate is None:
            _emit(trace, "tangent-escape-failed", iteration=iteration,
                  trigger=reason, **detail)
            return False
        y = candidate
        support_scale = max(1.0, float(np.abs(y).max()))
        W = y > 100.0 * tol * support_scale
        try:
            W, completed = _complete_working_set(E, W)
            face_factor.reset(W)
        except np.linalg.LinAlgError:
            _emit(trace, "tangent-escape-failed", iteration=iteration,
                  trigger=reason, reason_detail="rank completion")
            return False
        tangent_escapes += 1
        working_c = original_c.copy()
        perturbed = False
        perturb_countdown = 0
        stalled_iterations = 0
        seen = set()
        last_original_objective = _objective(y, original_c)
        _emit(trace, "tangent-escape", iteration=iteration, trigger=reason,
              level=tangent_escapes, working=int(W.sum()), completed=completed,
              **detail)
        return True

    def activate_perturbation(reason, iteration):
        nonlocal working_c, perturbations, perturbed, perturb_countdown
        nonlocal stalled_iterations, seen, final_crossover
        if perturbations >= max_perturbations:
            return False
        perturbations += 1
        magnitude = ((10.0 ** (perturbations - 1)) * np.sqrt(np.finfo(float).eps)
                     * max(1.0, float(np.abs(original_c).max())))
        # Deterministic hierarchical bias: the index ordering remains stable,
        # while each escalation is large enough to dominate the detected tie.
        working_c = original_c + magnitude * index_pattern
        perturbed = True
        final_crossover = False
        perturb_countdown = max(20, 2 * equality_rank)
        stalled_iterations = 0
        seen = set()
        _emit(trace, "perturb", iteration=iteration, reason=reason,
              level=perturbations, magnitude=float(magnitude))
        return True

    def factor_payload():
        return {"sparse_kkt_repairs": sparse_kkt_repairs,
                **face_factor.diagnostics()}

    for iteration in range(maxiter):
        key = np.packbits(W).tobytes()
        if key in seen:
            if activate_tangent_escape("repeated working set", iteration):
                continue
            if not activate_perturbation("repeated working set", iteration):
                eqerr, negerr = _original_errors(B, d, y)
                return {"status": "CYCLE", "iterations": iteration, "y": y,
                        "working_set": W, "equality_error": eqerr,
                        "nonnegative_error": negerr,
                        "perturbations": perturbations,
                        "tangent_escapes": tangent_escapes,
                        **factor_payload()}
            key = np.packbits(W).tobytes()
        seen.add(key)

        try:
            try:
                z, multiplier, factor_error = face_factor.solve(working_c)
            except np.linalg.LinAlgError:
                factor_error = np.inf
            if factor_error > 1e-8:
                # Rebuild the accumulated tableau/Gram data without asking a
                # fresh numerical-rank heuristic to reject the already valid
                # sparse LU basis.
                face_factor.rebuild(W, refactor_basis=False)
                try:
                    z, multiplier, factor_error = face_factor.solve(working_c)
                except np.linalg.LinAlgError:
                    factor_error = np.inf
            if factor_error > 1e-8:
                # The reduced Gram system can square the condition of a poor
                # basis.  Preserve sparsity and the residual gate with an
                # exceptional augmented-KKT LU solve; ordinary pivots remain
                # rank-one updates and no dense SVD is reintroduced.
                z, multiplier, _repair_factor, factor_error = factor_face(
                    E, f, working_c, W, refinement_steps=5)
                sparse_kkt_repairs += 1
                if factor_error > 1e-8:
                    raise np.linalg.LinAlgError(
                        f"sparse KKT residual {factor_error:.3e}")
        except (np.linalg.LinAlgError, ValueError) as error:
            eqerr, negerr = _original_errors(B, d, y)
            return {"status": "FACTORIZATION FAILURE",
                    "iterations": iteration, "y": y,
                    "working_set": W, "equality_error": eqerr,
                    "nonnegative_error": negerr,
                    "perturbations": perturbations,
                    "tangent_escapes": tangent_escapes,
                    "detail": str(error), **factor_payload()}
        # Test primal feasibility on the normwise scale of the projection data;
        # otherwise 1e-8 roundoff in a problem whose target is 1e5 is
        # misclassified as a real blocking constraint.
        face_scale = max(1.0, float(np.abs(z).max()))
        negative = W & (z < -100.0 * tol * face_scale)
        if negative.any():
            p = z - y
            # Only materially negative face coordinates may block the step.
            # Including every tiny negative direction lets an already-zero
            # roundoff coordinate win the ratio test and creates a false
            # zero-pivot cycle while a different coordinate has the real hit.
            blockers = np.where(negative & (p < 0.0))[0]
            if blockers.size == 0:
                return {"status": "NO BLOCKER", "iterations": iteration,
                        "y": y, "working_set": W,
                        "perturbations": perturbations,
                        "tangent_escapes": tangent_escapes,
                        **factor_payload()}
            ratios = np.maximum(y[blockers], 0.0) / (-p[blockers])
            minimum = float(ratios.min())
            tie_tol = 10.0 * np.finfo(float).eps * max(1.0, abs(minimum))
            tied = blockers[ratios <= minimum + tie_tol]
            leave = int(tied.min())
            alpha = min(1.0, max(0.0, minimum))
            leave_y = float(y[leave])
            y = y + alpha * p
            y[np.abs(y) <= 100.0 * np.finfo(float).eps
              * max(1.0, float(np.abs(y).max()))] = 0.0
            y[leave] = 0.0
            try:
                accepted, replacement = face_factor.activate_bound(leave, W)
            except (np.linalg.LinAlgError, RuntimeError, ValueError) as error:
                eqerr, negerr = _original_errors(B, d, y)
                return {"status": "FACTORIZATION FAILURE",
                        "iterations": iteration, "y": y, "working_set": W,
                        "equality_error": eqerr,
                        "nonnegative_error": negerr,
                        "perturbations": perturbations,
                        "tangent_escapes": tangent_escapes,
                        "detail": str(error), **factor_payload()}
            if accepted:
                completed = ([] if replacement is None else [replacement])
                if replacement is not None:
                    y[replacement] = 0.0
            else:
                # A rank-essential zero remains basic. The repeated-face check
                # will invoke the tangent escape if no other blocker advances.
                completed = [leave]
            _emit(trace, "leave", iteration=iteration, index=leave,
                  alpha=float(alpha), y_before=leave_y,
                  z_leave=float(z[leave]), working=int(W.sum()), completed=completed,
                  perturbed=perturbed)
        else:
            y = z
            nonfree = ~W
            if equality_rank:
                reduced = (-working_c[nonfree]
                           + np.asarray(E[:, nonfree].T @ multiplier).ravel())
            else:
                reduced = -working_c[nonfree]
            reduced_scale = (1.0 + np.abs(working_c[nonfree])
                             + (np.asarray(np.abs(E[:, nonfree]).T
                                           @ np.abs(multiplier)).ravel()
                                if equality_rank else 0.0))
            scaled = reduced / reduced_scale
            violated = np.where(nonfree)[0][scaled < -tol]
            if violated.size == 0:
                if perturbed:
                    working_c = original_c.copy()
                    perturbed = False
                    perturb_countdown = 0
                    seen = set()
                    _emit(trace, "unperturb", iteration=iteration,
                          reason="perturbed KKT face found")
                    continue
                # Original-data KKT certificate.
                free_stationarity = (y[W] - original_c[W]
                                     + np.asarray(E[:, W].T
                                                  @ multiplier).ravel()
                                     if equality_rank else y[W] - original_c[W])
                free_scale = (1.0 + np.abs(y[W]) + np.abs(original_c[W])
                              + (np.asarray(np.abs(E[:, W]).T
                                            @ np.abs(multiplier)).ravel()
                                 if equality_rank else 0.0))
                stationarity_error = (float(np.max(np.abs(free_stationarity)
                                                    / free_scale))
                                      if free_stationarity.size else 0.0)
                eqerr, negerr = _original_errors(B, d, y)
                independent = projection_kkt_certificate(
                    B, b, d, y, t=t, tol=100.0 * tol)
                if (max(stationarity_error, eqerr, negerr) <= 100.0 * tol
                        and independent["passed"]):
                    return {"status": "PROJECTED", "iterations": iteration,
                            "y": y, "support": y > tol * max(1.0, float(y.max())),
                            "working_set": W, "equality_rank": equality_rank,
                            "selected_equalities": selected_equalities,
                            "warm_start": warm_start,
                            "warm_start_active_bounds": warm_start_active_bounds,
                            "warm_start_completed": warm_completed,
                            "face_stationarity_error": stationarity_error,
                            "objective": _objective(y, original_c),
                            "perturbations": perturbations,
                            "tangent_escapes": tangent_escapes,
                            **factor_payload(), **independent}
                return {"status": "FALSE KKT", "iterations": iteration,
                        "y": y, "working_set": W,
                        "equality_error": eqerr,
                        "nonnegative_error": negerr,
                        "face_stationarity_error": stationarity_error,
                        "perturbations": perturbations,
                        "tangent_escapes": tangent_escapes,
                        **factor_payload(), **independent}
            enter = int(violated.min())       # Bland entering rule
            try:
                face_factor.release_bound(enter, W)
            except (np.linalg.LinAlgError, RuntimeError, ValueError) as error:
                eqerr, negerr = _original_errors(B, d, y)
                return {"status": "FACTORIZATION FAILURE",
                        "iterations": iteration, "y": y, "working_set": W,
                        "equality_error": eqerr,
                        "nonnegative_error": negerr,
                        "perturbations": perturbations,
                        "tangent_escapes": tangent_escapes,
                        "detail": str(error), **factor_payload()}
            y[enter] = 0.0
            _emit(trace, "enter", iteration=iteration, index=enter,
                  reduced=float(scaled[np.searchsorted(np.where(nonfree)[0], enter)]),
                  working=int(W.sum()), perturbed=perturbed)

        current_original_objective = _objective(y, original_c)
        progress_tol = 1e-13 * max(1.0, abs(last_original_objective))
        if current_original_objective < last_original_objective - progress_tol:
            last_original_objective = current_original_objective
            stalled_iterations = 0
            if perturbed and perturb_countdown <= 0:
                working_c = original_c.copy()
                perturbed = False
                seen = set()
                _emit(trace, "unperturb", iteration=iteration,
                      reason="original objective decreased")
        else:
            stalled_iterations += 1
        if perturbed:
            perturb_countdown -= 1
        # A run of distinct zero-length pivots is normal in a degenerate QP and
        # is not itself a cycle.  The repeated-working-set check at the top of
        # the loop is the primary perturbation trigger.  A second trigger is a
        # rank-scaled window with no meaningful decrease in the *original*
        # objective: this catches long nonrepeating degenerate walks.
        stall_window = max(64, 2 * equality_rank)
        if stalled_iterations >= stall_window:
            if activate_tangent_escape("no original-objective progress",
                                       iteration):
                continue
            if not activate_perturbation("no original-objective progress",
                                        iteration):
                if perturbed and not final_crossover:
                    working_c = original_c.copy()
                    perturbed = False
                    final_crossover = True
                    perturb_countdown = 0
                    stalled_iterations = 0
                    seen = set()
                    _emit(trace, "unperturb", iteration=iteration,
                          reason="final certification crossover")
                    continue
                eqerr, negerr = _original_errors(B, d, y)
                return {"status": "STALLED", "iterations": iteration,
                        "y": y, "working_set": W,
                        "equality_error": eqerr,
                        "nonnegative_error": negerr,
                        "perturbations": perturbations,
                        "tangent_escapes": tangent_escapes,
                        **factor_payload()}

    eqerr, negerr = _original_errors(B, d, y)
    return {"status": "ITERATION CAP", "iterations": maxiter, "y": y,
            "working_set": W, "equality_error": eqerr,
            "nonnegative_error": negerr, "perturbations": perturbations,
            "tangent_escapes": tangent_escapes,
            **factor_payload()}


def _exact_control():
    B = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b = np.ones(3)
    d = np.array([1.0, 0.0])
    return B, b, d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=[
        "kb2", "scagr7", "recipe", "beaconfd", "israel", "boeing2",
        "share1b", "sc205", "lotfi", "brandy",
    ])
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--skip-control", action="store_true")
    args = parser.parse_args()
    passed = total = 0
    if not args.skip_control:
        B, b, d = _exact_control()
        result = project_lex_active_set(B, b, d, t=args.t)
        total += 1
        passed += result["status"] == "PROJECTED"
        print(f"{'exact':9s} {result['status']:20s} "
              f"iter={result['iterations']:4d} pert={result.get('perturbations', 0):2d} "
              f"tangent={result.get('tangent_escapes', 0):2d} "
              f"basis={result.get('basis_refactorizations', 0):3d}/"
              f"{result.get('basis_updates', 0):4d} "
              f"eq={result.get('equality_error', np.inf):.2e} "
              f"neg={result.get('nonnegative_error', np.inf):.2e} "
              f"stat={result.get('stationarity_error', np.inf):.2e}")
    for name in args.names:
        netlib_path = Path(__file__).resolve().parent.parent / "netlib" / f"{name}.mps"
        B, b, d, _ = to_Bxgeb(netlib_path)
        start = time.perf_counter()
        result = project_lex_active_set(B, b, d, t=args.t)
        total += 1
        passed += result["status"] == "PROJECTED"
        print(f"{name:9s} {result['status']:20s} "
              f"iter={result['iterations']:4d} pert={result.get('perturbations', 0):2d} "
              f"tangent={result.get('tangent_escapes', 0):2d} "
              f"basis={result.get('basis_refactorizations', 0):3d}/"
              f"{result.get('basis_updates', 0):4d} "
              f"eq={result.get('equality_error', np.inf):.2e} "
              f"neg={result.get('nonnegative_error', np.inf):.2e} "
              f"stat={result.get('stationarity_error', np.inf):.2e} "
              f"sec={time.perf_counter() - start:.2f}")
    print(f"passed={passed}/{total}")


if __name__ == "__main__":
    main()
