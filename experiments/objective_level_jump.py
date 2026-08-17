"""Certified objective-level jump on the Mangasarian projection path.

At ``y=P_C(t*b)``, temporarily impose ``b'y = b'y_current + delta`` and
project the same ray point onto that slice.  If ``u_level`` is the multiplier
of the added equality, the candidate path parameter is

    t_new = t - u_level.

The candidate is accepted only when ``t_new>t`` and the existing independent
certificate verifies ``y_new=P_C(t_new*b)`` on the original, unsliced set.
"""

from __future__ import annotations

import numpy as np

from lex_active_set_projection import project_lex_active_set
from sparse_basis_projection import projection_kkt_certificate


def objective_level_jump(B, b, d, y, t, delta, tol=1e-9,
                         maxiter=2000, trace=None):
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = B.shape
    if (b.shape != (n,) or d.shape != (m,) or y.shape != (n,)
            or not np.all(np.isfinite(B)) or not np.all(np.isfinite(b))
            or not np.all(np.isfinite(d)) or not np.all(np.isfinite(y))):
        raise ValueError("invalid or nonfinite level-jump data")
    if not np.isfinite(t) or t < 0.0 or not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("t must be nonnegative and delta finite and positive")

    old_level = float(b @ y)
    target_level = old_level + float(delta)
    augmented_B = np.column_stack([B, b])
    augmented_d = np.concatenate([d, [target_level]])
    sliced = project_lex_active_set(
        augmented_B, b, augmented_d, t=t, tol=tol, maxiter=maxiter,
        max_perturbations=4, max_tangent_escapes=20, trace=trace)
    result = {
        "status": "SLICE FAILED",
        "slice_status": sliced.get("status"),
        "slice_iterations": int(sliced.get("iterations", 0)),
        "old_t": float(t),
        "old_level": old_level,
        "target_level": target_level,
        "delta": float(delta),
    }
    for key in (
            "perturbations", "tangent_escapes", "basis_refactorizations",
            "basis_updates", "sparse_kkt_repairs", "detail"):
        if sliced.get(key) is not None:
            result["slice_" + key] = sliced[key]
    if sliced.get("status") != "PROJECTED":
        return result

    multiplier = np.asarray(sliced.get("multiplier", []), dtype=float)
    if multiplier.shape != (m + 1,) or not np.all(np.isfinite(multiplier)):
        result["status"] = "NO LEVEL MULTIPLIER"
        return result
    level_multiplier = float(multiplier[-1])
    sigma = -level_multiplier
    new_t = float(t + sigma)
    candidate = np.asarray(sliced["y"], dtype=float)
    new_level = float(b @ candidate)
    level_scale = (1.0 + abs(target_level)
                   + float(np.abs(b) @ np.abs(candidate)))
    level_error = abs(new_level - target_level) / level_scale
    result.update(
        level_multiplier=level_multiplier,
        sigma=sigma,
        new_t=new_t,
        new_level=new_level,
        level_error=level_error,
        movement=float(np.linalg.norm(candidate - y)),
    )
    forward_tol = 100.0 * tol * max(1.0, abs(t), abs(new_t))
    if sigma <= forward_tol:
        result["status"] = "NO FORWARD MULTIPLIER"
        return result

    path_certificate = projection_kkt_certificate(
        B, b, d, candidate, t=new_t, tol=100.0 * tol)
    result["path_certificate"] = path_certificate
    certificate_error = max(
        float(path_certificate.get("equality_error", np.inf)),
        float(path_certificate.get("nonnegative_error", np.inf)),
        float(path_certificate.get("stationarity_error", np.inf)),
    )
    result["path_certificate_error"] = certificate_error
    if (not path_certificate.get("passed")
            or max(level_error, certificate_error) > 100.0 * tol
            or new_level <= old_level):
        result["status"] = "PATH CERTIFICATE FAILED"
        return result
    support = candidate > tol * max(1.0, float(candidate.max()))
    working_set = np.asarray(sliced.get("working_set", support),
                             dtype=bool).copy()
    result.update(status="JUMPED", y=candidate, support=support,
                  working_set=working_set)
    return result


def objective_level_jump_backtracking(B, b, d, y, t, initial_delta,
                                      attempts=16, min_delta=None, tol=1e-9,
                                      maxiter=2000, trace=None):
    """Backtrack only on a certified infeasible temporary level slice.

    ``attempts`` is a hard work cap.  ``min_delta`` is a scale-aware lower
    bound on further halvings, not a relaxation of any projection or KKT gate.
    Numerical, multiplier, and certificate failures stop immediately because
    they do not prove that the requested objective increment was too large.
    """
    y = np.asarray(y, dtype=float)
    b = np.asarray(b, dtype=float)
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if min_delta is None:
        min_delta = 1e-8 * max(1.0, abs(float(b @ y)))
    min_delta = float(min_delta)
    if not np.isfinite(min_delta) or min_delta <= 0.0:
        raise ValueError("min_delta must be finite and positive")

    history = []
    delta = float(initial_delta)
    halvings = 0
    infeasible_slices = 0
    for attempt in range(attempts):
        result = objective_level_jump(
            B, b, d, y, t, delta, tol=tol, maxiter=maxiter, trace=trace)
        history.append({key: value for key, value in result.items()
                        if key not in ("y", "support", "working_set",
                                       "path_certificate")})
        result.update(
            attempt=attempt + 1,
            history=history,
            infeasible_slices=infeasible_slices,
            infeasible_backtracks=halvings,
            minimum_delta=min_delta,
        )
        if result.get("status") == "JUMPED":
            result["backtracking_stop_reason"] = "jump accepted"
            return result

        slice_infeasible = (
            result.get("status") == "SLICE FAILED"
            and result.get("slice_status") == "C INFEASIBLE")
        if not slice_infeasible:
            result["backtracking_stop_reason"] = (
                "failure did not certify slice infeasibility")
            return result

        infeasible_slices += 1
        result["infeasible_slices"] = infeasible_slices
        if attempt + 1 >= attempts:
            result["backtracking_stop_reason"] = "attempt cap reached"
            return result
        next_delta = 0.5 * delta
        if next_delta < min_delta:
            result["backtracking_stop_reason"] = "minimum delta reached"
            return result
        delta = next_delta
        halvings += 1

    raise RuntimeError("unreachable objective-level backtracking state")
