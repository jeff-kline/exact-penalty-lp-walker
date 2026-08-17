"""Feasible-start active-set projection onto C = {y >= 0 : B.T y = d}.

This is the bounded AS-PROJ-1 experiment from assets/22.  It deliberately uses
HiGHS only for the zero-objective Phase-I feasibility problem and, when a face
multiplier is nonunique, for the small existential multiplier LP.  It never
solves the original LP objective.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import linprog

from exp11_netlib_rank_certificate import to_Bxgeb
from exp23_path_primal_dual import piece_y


TOL = 1e-9


def _objective(y, b, t):
    r = y - t * b
    return 0.5 * float(r @ r)


def _equality_error(B, d, y):
    residual = B.T @ y - d
    scale = 1.0 + np.abs(d) + np.abs(B).T @ np.abs(y)
    return float(np.max(np.abs(residual) / scale))


def _face_minimizer(B, b, d, W, t):
    """Return the equality-constrained minimizer on the coordinate face W."""
    y = np.zeros(B.shape[0])
    if not W.any():
        return y, np.inf
    column_scale = np.linalg.norm(B, axis=0)
    column_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    Bs = B / column_scale
    ds = d / column_scale
    g, h, _, _, _ = piece_y(Bs, b, ds, W)
    y[W] = t * g + h
    return y, _equality_error(B, d, y)


def _unique_multiplier(B, b, y, W, t, tol):
    """Fast KKT test when B_W has full column rank."""
    column_scale = np.linalg.norm(B, axis=0)
    column_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    Bs = B / column_scale
    BW = Bs[W]
    if np.linalg.matrix_rank(BW) < B.shape[1]:
        return None
    rhs = y[W] - t * b[W]
    us, _, _, _ = np.linalg.lstsq(BW, rhs, rcond=None)
    u = us / column_scale
    eq_scale = 1.0 + np.abs(rhs) + np.abs(BW) @ np.abs(us)
    eq_err = float(np.max(np.abs(BW @ us - rhs) / eq_scale))
    off = ~W
    violation = t * b[off] + B[off] @ u
    off_scale = 1.0 + np.abs(t * b[off]) + np.abs(B[off]) @ np.abs(u)
    scaled = violation / off_scale
    max_viol = max(0.0, float(scaled.max()) if scaled.size else 0.0)
    if eq_err <= tol and max_viol <= tol:
        return {"passed": True, "u": u, "error": max(eq_err, max_viol),
                "rank_deficient": False}
    if eq_err > tol:
        return {"passed": False, "reason": "multiplier equality", "error": eq_err,
                "rank_deficient": False}
    off_idx = np.where(off)[0]
    vmax = float(scaled.max())
    tied = off_idx[scaled >= vmax - tol * max(1.0, abs(vmax))]
    return {"passed": False, "reason": "release", "enter": int(tied.min()),
            "error": max_viol, "u": u, "rank_deficient": False}


def _existential_multiplier(B, b, y, W, t, tol):
    """Test whether any face multiplier satisfies every off-face KKT inequality.

    If not, minimize the maximum scaled *unscaled* violation and return a
    lexicographically selected maximally violated row to release.
    """
    n, m = B.shape
    column_scale = np.linalg.norm(B, axis=0)
    column_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    Bs = B / column_scale
    off = ~W
    BW = Bs[W]
    rhs = y[W] - t * b[W]
    bounds_u = [(None, None)] * m

    feasible = linprog(
        np.zeros(m),
        A_ub=Bs[off] if off.any() else None,
        b_ub=-t * b[off] if off.any() else None,
        A_eq=BW if W.any() else None,
        b_eq=rhs if W.any() else None,
        bounds=bounds_u,
        method="highs-ds",
    )
    if feasible.success:
        us = np.asarray(feasible.x)
        u = us / column_scale
        violation = t * b[off] + Bs[off] @ us
        err = max(0.0, float(violation.max()) if violation.size else 0.0)
        return {"passed": True, "u": u, "error": err,
                "rank_deficient": True}

    # Phase-I for multiplier feasibility: minimize rho such that every
    # off-face violation is at most rho.  rho identifies a separating row.
    c = np.zeros(m + 1)
    c[-1] = 1.0
    A_ub = None
    b_ub = None
    if off.any():
        A_ub = np.column_stack([Bs[off], -np.ones(int(off.sum()))])
        b_ub = -t * b[off]
    A_eq = None
    b_eq = None
    if W.any():
        A_eq = np.column_stack([BW, np.zeros(int(W.sum()))])
        b_eq = rhs
    phase = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds_u + [(0.0, None)], method="highs-ds")
    if not phase.success:
        return {"passed": False, "reason": "multiplier phase-I failed",
                "status": int(phase.status), "rank_deficient": True}
    us = np.asarray(phase.x[:-1])
    u = us / column_scale
    off_idx = np.where(off)[0]
    violation = t * b[off] + Bs[off] @ us
    vmax = float(violation.max()) if violation.size else 0.0
    if vmax <= tol:
        return {"passed": True, "u": u, "error": max(0.0, vmax),
                "rank_deficient": True}
    tied = off_idx[violation >= vmax - tol * max(1.0, abs(vmax))]
    return {"passed": False, "reason": "release", "enter": int(tied.min()),
            "error": vmax, "rho": float(phase.x[-1]), "u": u,
            "rank_deficient": True}


def multiplier_kkt(B, b, y, W, t, tol=TOL):
    unique = _unique_multiplier(B, b, y, W, t, tol)
    if unique is not None:
        return unique
    return _existential_multiplier(B, b, y, W, t, tol)


def project_active_set(B, b, d, t=1.0, tol=TOL, maxiter=2000,
                       max_auxiliary_lps=200, trace=None,
                       initial_working_set=None):
    """Project ``t*b`` onto ``{y >= 0 : B.T@y = d}``.

    Returns a diagnostic dictionary.  ``status == 'PROJECTED'`` is issued only
    after primal face feasibility and existential multiplier KKT both pass.
    """
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    n, m = B.shape
    phase = linprog(np.zeros(n), A_eq=B.T, b_eq=d, bounds=(0.0, None),
                    method="highs-ds")
    if not phase.success:
        return {"status": "C INFEASIBLE", "phase_status": int(phase.status),
                "iterations": 0, "auxiliary_lps": 0}
    y = np.asarray(phase.x, dtype=float)
    y[np.abs(y) <= tol * max(1.0, float(np.abs(y).max()))] = 0.0
    # W is the *free working set*, not necessarily supp(y).  Under degeneracy
    # zero-valued free variables are essential: several may have to enter before
    # the equality nullspace contains any feasible descent direction.
    W = y > 0.0
    if initial_working_set is not None:
        proposed = np.asarray(initial_working_set, dtype=bool)
        if proposed.shape != (n,):
            raise ValueError("initial_working_set has the wrong shape")
        W |= proposed
    auxiliary_lps = 0
    best_by_support = {}

    def emit(event, **fields):
        if trace is None:
            return
        record = {"event": event, **fields}
        trace(record) if callable(trace) else trace.append(record)

    for iteration in range(maxiter):
        obj = _objective(y, b, t)
        key = np.packbits(W).tobytes()
        previous = best_by_support.get(key)
        progress_tol = 1e-13 * max(1.0, abs(obj))
        if previous is not None and obj >= previous - progress_tol:
            return {"status": "CYCLE", "iterations": iteration, "y": y,
                    "support": W, "objective": obj,
                    "equality_error": _equality_error(B, d, y),
                    "auxiliary_lps": auxiliary_lps}
        best_by_support[key] = obj if previous is None else min(previous, obj)

        z, face_error = _face_minimizer(B, b, d, W, t)
        if face_error > 100.0 * tol:
            return {"status": "FACE INCONSISTENT", "iterations": iteration,
                    "y": y, "support": W, "face_error": face_error,
                    "auxiliary_lps": auxiliary_lps}

        negative = W & (z < -tol * (1.0 + np.abs(t * b)))
        if negative.any():
            p = z - y
            blockers = np.where(W & (p < 0.0))[0]
            ratios = y[blockers] / (-p[blockers])
            alpha = min(1.0, float(ratios.min()))
            if alpha < -tol or not np.isfinite(alpha):
                return {"status": "BAD STEP", "iterations": iteration,
                        "alpha": alpha, "auxiliary_lps": auxiliary_lps}
            y = y + alpha * p
            rmin = float(ratios.min())
            tied = blockers[ratios <= rmin + tol * max(1.0, abs(rmin))]
            hit = int(tied.min())
            y[hit] = 0.0
            W[hit] = False
            emit("drop", iteration=iteration, index=hit, alpha=alpha,
                 support=int(W.sum()), objective=_objective(y, b, t))
            continue

        # Keep zero-valued members of W free.  They are degenerate basic
        # variables, not positive support members, and dropping them here can
        # prevent the working set from ever acquiring a feasible direction.
        y = z
        y[y < 0.0] = 0.0
        kkt = multiplier_kkt(B, b, y, W, t, tol=tol)
        if kkt.get("rank_deficient"):
            auxiliary_lps += 1
            if auxiliary_lps > max_auxiliary_lps:
                return {"status": "AUXILIARY LP CAP", "iterations": iteration,
                        "y": y, "support": W,
                        "auxiliary_lps": auxiliary_lps}
        if kkt["passed"]:
            eq_error = _equality_error(B, d, y)
            nonnegative_error = max(0.0, float(-y.min()))
            if eq_error <= 100.0 * tol and nonnegative_error <= 100.0 * tol:
                y = np.maximum(y, 0.0)
                return {"status": "PROJECTED", "iterations": iteration,
                        "y": y, "support": y > tol, "working_set": W,
                        "u": kkt["u"],
                        "objective": _objective(y, b, t),
                        "equality_error": eq_error,
                        "kkt_error": float(kkt["error"]),
                        "rank_deficient": bool(kkt["rank_deficient"]),
                        "auxiliary_lps": auxiliary_lps}
            return {"status": "FALSE KKT", "iterations": iteration,
                    "y": y, "support": W, "equality_error": eq_error,
                    "nonnegative_error": nonnegative_error,
                    "auxiliary_lps": auxiliary_lps}
        if kkt.get("reason") != "release":
            return {"status": kkt.get("reason", "KKT FAILURE").upper(),
                    "iterations": iteration, "y": y, "support": W,
                    "detail": kkt, "auxiliary_lps": auxiliary_lps}
        enter = int(kkt["enter"])
        W[enter] = True
        y[enter] = 0.0
        emit("release", iteration=iteration, index=enter,
             support=int(W.sum()), violation=float(kkt["error"]),
             rank_deficient=bool(kkt["rank_deficient"]), objective=obj)

    return {"status": "ITERATION CAP", "iterations": maxiter, "y": y,
            "support": W, "objective": _objective(y, b, t),
            "equality_error": _equality_error(B, d, y),
            "auxiliary_lps": auxiliary_lps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=[
        "kb2", "scagr7", "recipe", "beaconfd", "israel", "boeing2",
        "share1b", "sc205", "lotfi", "brandy",
    ])
    parser.add_argument("--t", type=float, default=1.0)
    args = parser.parse_args()
    passed = 0
    for name in args.names:
        B, b, d, _ = to_Bxgeb(f"../netlib/{name}.mps")
        start = time.perf_counter()
        result = project_active_set(B, b, d, t=args.t)
        passed += result["status"] == "PROJECTED"
        print(f"{name:9s} {result['status']:20s} "
              f"iter={result['iterations']:4d} "
              f"W={int(result.get('support', np.zeros(0)).sum()):4d} "
              f"eq={result.get('equality_error', np.inf):.2e} "
              f"aux={result.get('auxiliary_lps', 0):3d} "
              f"sec={time.perf_counter() - start:.2f}")
    print(f"passed={passed}/{len(args.names)}")


if __name__ == "__main__":
    main()
