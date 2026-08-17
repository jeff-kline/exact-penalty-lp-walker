"""FAILED BOUNDED PROTOTYPE: critical-cone walk for y(t)=P_C(t*b).

CCP-1 uses a projection QP only to initialize/resynchronize vertices.  Its path
direction is the projection of b onto the critical cone, and its step length is
the largest value preserving the corresponding normal cone.  It never falls
back to the original LP objective.  Exact controls pass, but scalar crossover
from an IPM point selects the wrong face on kb2; see assets/25.  Do not integrate.
"""

from __future__ import annotations

import argparse
import time

import clarabel
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, vstack

from exp11_netlib_rank_certificate import to_Bxgeb
from sparse_basis_projection import reduce_equalities_sparse


def _clarabel_settings():
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 500
    settings.tol_gap_abs = 1e-11
    settings.tol_gap_rel = 1e-11
    settings.tol_feas = 1e-11
    return settings


def project_ray_point(B, b, d, t):
    """Solve P_C(t*b), returning y and cone multipliers."""
    n, m = B.shape
    objective_scale = 1.0 / max(1.0, float(t))
    P = objective_scale * eye(n, format="csc")
    q = -objective_scale * t * b
    A = vstack([csc_matrix(B.T), -eye(n, format="csc")], format="csc")
    rhs = np.concatenate([d, np.zeros(n)])
    cones = [clarabel.ZeroConeT(m), clarabel.NonnegativeConeT(n)]
    solution = clarabel.DefaultSolver(P, q, A, rhs, cones,
                                      _clarabel_settings()).solve()
    return {"status": str(solution.status), "y": np.asarray(solution.x),
            "equality_dual": np.asarray(solution.z[:m]),
            "bound_dual": np.asarray(solution.z[m:]),
            "objective_scale": objective_scale,
            "iterations": int(solution.iterations)}


def crossover_state(B, E, f, b, t, projection, active_tol):
    """Cross an IPM point and normal onto a polyhedral face."""
    raw_y = projection["y"]
    yscale = max(1.0, float(np.abs(raw_y).max()))
    zero = raw_y <= active_tol * yscale
    free = ~zero

    # Project t*b onto the selected affine face using an SVD that permits the
    # face equality system to lose rank (routine at degenerate vertices).
    y = np.zeros_like(raw_y)
    if free.any():
        AF = E[:, free].toarray()
        U, singular, Vt = np.linalg.svd(AF, full_matrices=False)
        cutoff = ((singular[0] if singular.size else 0.0)
                  * max(AF.shape) * np.finfo(float).eps)
        keep = singular > cutoff
        Ur = U[:, keep]
        sr = singular[keep]
        Vr = Vt[keep].T
        cF = t * b[free]
        residual = f - AF @ cF
        represented = Ur @ (Ur.T @ residual) if sr.size else np.zeros_like(f)
        consistency = float(np.linalg.norm(residual - represented, np.inf))
        if consistency > 1e-7 * max(1.0, float(np.linalg.norm(f, np.inf))):
            return None
        y[free] = cF + (Vr @ ((Ur.T @ residual) / sr)
                         if sr.size else 0.0)
    elif np.linalg.norm(f, np.inf) > 1e-10:
        return None

    # Interior-point normals contain O(sqrt(tol)) artificial components on
    # weakly active bounds.  Retain multipliers only when they are materially
    # nonzero, and reconstruct q from stationarity.
    gamma = projection["objective_scale"]
    equality_dual = projection["equality_dual"] / gamma
    bound_dual = np.maximum(projection["bound_dual"] / gamma, 0.0)
    equality_normal = B @ equality_dual
    normal_scale = max(1.0, float(np.abs(equality_normal).max()),
                       float(np.abs(t * b - y).max()))
    # On an IPM central path a weakly active coordinate has y_i and mu_i of
    # comparable square-root size.  A genuinely strong bound has mu_i >> y_i.
    # The ratio test survives large global q scaling where an absolute cutoff
    # incorrectly labels small but genuine multipliers as weak.
    denominator = np.maximum(np.abs(raw_y),
                             100.0 * np.finfo(float).eps * normal_scale)
    strong = zero & (bound_dual > 100.0 * denominator)
    # Once y is crossed onto an exact affine face, the projection normal is
    # defined directly by the ray geometry.  Do not reconstruct it from IPM
    # multipliers: that would reintroduce weak/strong-bound ambiguity.
    qnormal = t * b - y
    return y, qnormal, zero, strong


def critical_derivative(E, b, qnormal, zero):
    """Compute P_K(b), K={v:E v=0, q'v=0, v_Z>=0}."""
    n = b.size
    equality_blocks = [E]
    equality_rhs = [np.zeros(E.shape[0])]
    qnorm = float(np.linalg.norm(qnormal))
    if qnorm > 1e-12 * max(1.0, float(np.linalg.norm(b))):
        equality_blocks.append(csc_matrix((qnormal / qnorm)[None, :]))
        equality_rhs.append(np.zeros(1))
    Aeq = vstack(equality_blocks, format="csc")
    active = np.where(zero)[0]
    selector = csc_matrix((-np.ones(active.size),
                           (np.arange(active.size), active)),
                          shape=(active.size, n))
    A = vstack([Aeq, selector], format="csc")
    rhs = np.concatenate(equality_rhs + [np.zeros(active.size)])
    cones = [clarabel.ZeroConeT(Aeq.shape[0]),
             clarabel.NonnegativeConeT(active.size)]
    solution = clarabel.DefaultSolver(eye(n, format="csc"), -b, A, rhs,
                                      cones, _clarabel_settings()).solve()
    return {"status": str(solution.status), "v": np.asarray(solution.x),
            "iterations": int(solution.iterations)}


def normal_cone_step(E, y, v, qnormal, b, zero, feasibility_tol=2e-8):
    """Maximize delta while q+delta(b-v) stays in the moving-face normal cone."""
    r, n = E.shape
    qdot = b - v
    yscale = max(1.0, float(np.abs(y).max()))
    vscale = max(1.0, float(np.abs(v).max()))
    moving = (~zero) | (v > 1e-8 * vscale)
    staying = ~moving

    hit = np.inf
    hitters = np.where((y > 1e-8 * yscale) & (v < -1e-10 * vscale))[0]
    if hitters.size:
        hit = float(np.min(y[hitters] / (-v[hitters])))

    rows = []
    rhs = []
    if moving.any():
        EMt = E[:, moving].T.toarray()
        qm = qnormal[moving]
        qdm = qdot[moving]
        scale = 1.0 + np.abs(qm) + np.abs(qdm) * max(1.0, min(hit, 1.0e6))
        rows.append(np.column_stack([EMt, -qdm]))
        rhs.append(qm + feasibility_tol * scale)
        rows.append(np.column_stack([-EMt, qdm]))
        rhs.append(-qm + feasibility_tol * scale)
    if staying.any():
        EZt = E[:, staying].T.toarray()
        qz = qnormal[staying]
        qdz = qdot[staying]
        scale = 1.0 + np.abs(qz) + np.abs(qdz) * max(1.0, min(hit, 1.0e6))
        rows.append(np.column_stack([-EZt, qdz]))
        rhs.append(-qz + feasibility_tol * scale)

    objective = np.zeros(r + 1)
    objective[-1] = -1.0
    upper = None if not np.isfinite(hit) else max(0.0, hit)
    result = linprog(objective,
                     A_ub=np.vstack(rows) if rows else None,
                     b_ub=np.concatenate(rhs) if rhs else None,
                     bounds=[(None, None)] * r + [(0.0, upper)],
                     method="highs-ds")
    if result.status == 3:
        return {"status": "UNBOUNDED", "delta": np.inf,
                "moving": moving, "qdot": qdot}
    if not result.success:
        return {"status": "EVENT LP FAILED", "delta": 0.0,
                "lp_status": int(result.status), "moving": moving,
                "qdot": qdot}
    return {"status": "STEP", "delta": float(result.x[-1]),
            "lambda": np.asarray(result.x[:-1]), "moving": moving,
            "qdot": qdot, "primal_hit": hit}


def normal_slope_primal(E, selected, row_scale, b, moving):
    """Find the normal-cone slope and map it to an original primal x."""
    r = E.shape[0]
    staying = ~moving
    Aeq = E[:, moving].T.toarray() if moving.any() else None
    beq = b[moving] if moving.any() else None
    Aub = -E[:, staying].T.toarray() if staying.any() else None
    bub = -b[staying] if staying.any() else None
    result = linprog(np.zeros(r), A_ub=Aub, b_ub=bub,
                     A_eq=Aeq, b_eq=beq,
                     bounds=[(None, None)] * r, method="highs-ds")
    if not result.success:
        return None, int(result.status)
    x = np.zeros(row_scale.size)
    x[selected] = result.x / row_scale[selected]
    return x, 0


def lp_certificate(B, b, d, x, y, tol=1e-7):
    primal_scale = 1.0 + np.abs(b) + np.abs(B) @ np.abs(x)
    primal = max(0.0, float(np.max((b - B @ x) / primal_scale)))
    equality_scale = 1.0 + np.abs(d) + np.abs(B).T @ np.abs(y)
    dual = float(np.max(np.abs(B.T @ y - d) / equality_scale))
    nonnegative = max(0.0, float(-y.min())) / max(1.0, float(np.abs(y).max()))
    primal_objective = float(d @ x)
    dual_objective = float(b @ y)
    gap = abs(primal_objective - dual_objective) / max(
        1.0, abs(primal_objective), abs(dual_objective))
    error = max(primal, dual, nonnegative, gap)
    return {"passed": error <= tol, "error": error,
            "primal_error": primal, "dual_error": dual,
            "nonnegative_error": nonnegative, "gap_error": gap,
            "primal_objective": primal_objective,
            "dual_objective": dual_objective}


def _prepare(B, d):
    E, f, selected, consistency = reduce_equalities_sparse(B, d)
    row_norm = np.linalg.norm(B, axis=0)
    row_scale = np.where(row_norm > 0.0, row_norm, 1.0)
    return E, f, selected, row_scale, consistency


def follow_critical_cone(B, b, d, t0=1.0, max_vertices=200,
                         active_tol=2e-5, check_derivative=False, trace=None):
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    E, f, selected, row_scale, consistency = _prepare(B, d)
    if consistency > 1e-7:
        return {"status": "EQUALITY INCONSISTENT", "vertices": 0,
                "consistency_error": consistency}
    t = float(t0)
    previous_delta = None
    projection_calls = derivative_calls = event_calls = 0
    for vertex in range(max_vertices):
        projection = project_ray_point(B, b, d, t)
        projection_calls += 1
        if "Solved" not in projection["status"]:
            return {"status": "PROJECTION FAILURE", "vertices": vertex,
                    "t": t, "detail": projection["status"]}
        crossed = crossover_state(B, E, f, b, t, projection, active_tol)
        if crossed is None:
            return {"status": "CROSSOVER FAILURE", "vertices": vertex,
                    "t": t}
        y, qnormal, zero, strong = crossed
        derivative = critical_derivative(E, b, qnormal, zero)
        derivative_calls += 1
        if "Solved" not in derivative["status"]:
            return {"status": "DERIVATIVE FAILURE", "vertices": vertex,
                    "t": t, "detail": derivative["status"]}
        v = derivative["v"]

        derivative_error = np.nan
        if check_derivative:
            epsilon = 1e-3 * max(1.0, t)
            forward = project_ray_point(B, b, d, t + epsilon)
            projection_calls += 1
            forward_crossed = crossover_state(B, E, f, b, t + epsilon,
                                              forward, active_tol)
            if forward_crossed is None:
                return {"status": "FORWARD CROSSOVER FAILURE",
                        "vertices": vertex, "t": t}
            fd = (forward_crossed[0] - y) / epsilon
            derivative_error = (float(np.linalg.norm(fd - v, np.inf))
                                / max(1.0, float(np.linalg.norm(v, np.inf))))

        event = normal_cone_step(E, y, v, qnormal, b, zero)
        event_calls += 1
        _emit = {"vertex": vertex, "t": t, "zero": int(zero.sum()),
                 "strong": int(strong.sum()),
                 "vnorm": float(np.linalg.norm(v)),
                 "delta": event["delta"], "event": event["status"],
                 "derivative_error": derivative_error}
        if trace is not None:
            trace(_emit) if callable(trace) else trace.append(_emit)

        if event["status"] == "UNBOUNDED":
            vnorm = float(np.linalg.norm(v, np.inf))
            if vnorm > 1e-7 * max(1.0, float(np.linalg.norm(b, np.inf))):
                return {"status": "UNBOUNDED AFFINE PATH", "vertices": vertex,
                        "t": t, "y": y, "v": v,
                        "derivative_error": derivative_error}
            x, slope_status = normal_slope_primal(
                E, selected, row_scale, b, event["moving"])
            if x is None:
                return {"status": "TERMINAL SLOPE FAILURE", "vertices": vertex,
                        "t": t, "y": y, "slope_status": slope_status}
            certificate = lp_certificate(B, b, d, x, y)
            return {"status": "CERTIFIED" if certificate["passed"]
                    else "FALSE TERMINAL", "vertices": vertex, "t": t,
                    "x": x, "y": y, "v": v,
                    "projection_calls": projection_calls,
                    "derivative_calls": derivative_calls,
                    "event_calls": event_calls,
                    "derivative_error": derivative_error, **certificate}
        if event["status"] != "STEP":
            return {"status": event["status"], "vertices": vertex,
                    "t": t, "y": y, "detail": event}
        delta = event["delta"]
        step_tol = 1e-9 * max(1.0, t)
        if delta <= step_tol:
            if previous_delta is not None and previous_delta <= step_tol:
                return {"status": "REPEATED ZERO EVENT", "vertices": vertex,
                        "t": t, "y": y, "v": v,
                        "derivative_error": derivative_error}
            # Cross the numerically represented vertex without changing the
            # mathematical scale of the continuation.
            delta = 10.0 * step_tol
        previous_delta = event["delta"]
        endpoint = t + delta
        # The event LP is solved with a backward-error band, so its maximizer
        # can lie infinitesimally on the old side of a multiplier-zero event.
        # Evaluate the right limit and resynchronize, as a simplex method does
        # after a ratio-test breakpoint.
        t = endpoint + 1e-6 * max(1.0, abs(endpoint))
    return {"status": "VERTEX CAP", "vertices": max_vertices, "t": t,
            "projection_calls": projection_calls,
            "derivative_calls": derivative_calls, "event_calls": event_calls}


def _controls():
    simplex_B = np.array([[1.0], [1.0]])
    simplex_b = np.array([1.0, 0.0])
    simplex_d = np.array([1.0])
    coupled_B = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    coupled_b = np.ones(3)
    coupled_d = np.array([1.0, 0.0])
    return [("simplex", simplex_B, simplex_b, simplex_d, 0.0),
            ("coupled", coupled_B, coupled_b, coupled_d, 0.0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=["kb2", "recipe", "boeing2"])
    parser.add_argument("--max-vertices", type=int, default=200)
    args = parser.parse_args()
    passed = total = 0
    for name, B, b, d, t0 in _controls():
        start = time.perf_counter()
        result = follow_critical_cone(B, b, d, t0=t0,
                                      max_vertices=args.max_vertices,
                                      check_derivative=True)
        total += 1
        passed += result["status"] == "CERTIFIED"
        print(f"{name:9s} {result['status']:24s} vertices={result['vertices']:4d} "
              f"t={result.get('t', np.nan):.4g} "
              f"derr={result.get('derivative_error', np.nan):.2e} "
              f"sec={time.perf_counter()-start:.3f}")
    for name in args.names:
        B, b, d, _ = to_Bxgeb(f"../netlib/{name}.mps")
        start = time.perf_counter()
        result = follow_critical_cone(B, b, d,
                                      max_vertices=args.max_vertices,
                                      check_derivative=name in ("kb2", "boeing2"))
        total += 1
        passed += result["status"] == "CERTIFIED"
        print(f"{name:9s} {result['status']:24s} vertices={result['vertices']:4d} "
              f"t={result.get('t', np.nan):.4g} "
              f"err={result.get('error', np.nan):.2e} "
              f"derr={result.get('derivative_error', np.nan):.2e} "
              f"sec={time.perf_counter()-start:.3f}")
    print(f"passed={passed}/{total}")


if __name__ == "__main__":
    main()
