"""Restore Mangasarian's minimum-norm dual selector after an LP fallback.

Given a certified dual optimum ``y0`` for

    max b'y  subject to B'y=d, y>=0,

solve the fixed optimal-face projection.  When the certified fallback primal
point ``x0`` is available, complementary slackness gives the better-conditioned
form

    min 1/2||y_A||^2  subject to B_A'y_A=d, y_A>=0,

where ``A={i:B_i x0=b_i}``.  Otherwise the general objective-equality form is
used:

    min 1/2||y||^2  subject to B'y=d, b'y=b'y0, y>=0.

This is a postsolve on the optimal face, not another large-t continuation.
Every successful result passes the projection QP's original-data KKT gate; if
a fallback primal point is supplied, it must also pass the original LP's
primal-dual certificate with the polished dual point.
"""

from __future__ import annotations

import numpy as np

from lex_active_set_projection import project_lex_active_set


def _scaled_equality_error(matrix, rhs, y):
    residual = matrix.T @ y - rhs
    scale = 1.0 + np.abs(rhs) + np.abs(matrix).T @ np.abs(y)
    return float(np.max(np.abs(residual) / scale)) if residual.size else 0.0


def polish_dual_optimum(B, b, d, y0, x0=None, optimal_value=None,
                        tol=1e-9, maxiter=5000, max_perturbations=4,
                        max_tangent_escapes=20, trace=None):
    """Warm-start the minimum-norm projection on a certified optimal face.

    With ``x0``, the routine first requires the supplied pair to pass the
    original LP certificate and identifies the optimal face by complementary
    primal rows.  Without ``x0``, ``optimal_value`` defaults to ``b@y0`` so the
    supplied dual point lies on the added equality to working precision.
    """
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if B.ndim != 2:
        raise ValueError("B must be two-dimensional")
    n, m = B.shape
    if b.shape != (n,) or d.shape != (m,) or y0.shape != (n,):
        raise ValueError("expected B.shape == (len(b), len(d)) and len(y0) == len(b)")
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(b))
            and np.all(np.isfinite(d)) and np.all(np.isfinite(y0))):
        raise ValueError("B, b, d, and y0 must contain only finite values")

    fstar = float(b @ y0) if optimal_value is None else float(optimal_value)
    if not np.isfinite(fstar):
        raise ValueError("optimal_value must be finite")

    lp_certificate_pass = None
    lp_certificate_detail = None
    allowed = np.ones(n, dtype=bool)
    selected_slack_error = None
    if x0 is not None:
        from exp23_path_primal_dual import certificate_pair
        x0 = np.asarray(x0, dtype=float)
        lp_certificate_pass, lp_certificate_detail = certificate_pair(
            B, b, d, x0, y0)
        if not lp_certificate_pass:
            return {
                "status": "FALLBACK LP CERTIFICATE FAILED",
                "iterations": 0,
                "optimal_value": fstar,
                "lp_certificate_pass": False,
                "lp_certificate_detail": lp_certificate_detail,
            }
        slack = B @ x0 - b
        slack_scale = 1.0 + np.abs(b) + np.abs(B) @ np.abs(x0)
        yscale = max(1.0, float(np.abs(y0).max()))
        allowed = ((slack <= 100.0 * tol * slack_scale)
                   | (y0 > 100.0 * tol * yscale))
        face_B = B[allowed]
        face_d = d
        face_y0 = y0[allowed]
        face_mode = "primal-complementary rows"
        selected_slack_error = float(np.max(
            np.abs(slack[allowed]) / slack_scale[allowed]))
    else:
        face_B = np.column_stack([B, b])
        face_d = np.concatenate([d, [fstar]])
        face_y0 = y0
        face_mode = "objective equality"

    zero_target = np.zeros(face_B.shape[0])
    initial_face_error = _scaled_equality_error(face_B, face_d, face_y0)
    initial_nonnegative_error = (max(0.0, float(-y0.min()))
                                 / (1.0 + float(np.abs(y0).max())))
    if max(initial_face_error, initial_nonnegative_error) > 100.0 * tol:
        return {
            "status": "FALLBACK POINT NOT FACE-FEASIBLE",
            "iterations": 0,
            "optimal_value": fstar,
            "initial_face_error": initial_face_error,
            "initial_nonnegative_error": initial_nonnegative_error,
        }

    projected = project_lex_active_set(
        face_B, zero_target, face_d, t=1.0, tol=tol, maxiter=maxiter,
        max_perturbations=max_perturbations,
        max_tangent_escapes=max_tangent_escapes, trace=trace, y0=face_y0)
    crash_status = projected.get("status")
    crash_iterations = int(projected.get("iterations", 0))
    backtracked = False
    # A simplex fallback vertex can force many zero variables into the equality
    # basis.  If that crash is numerically poor, backtrack to the same feasible
    # point with all bounds temporarily free.  This discards only the proposed
    # working set, not the warm point, objective, or any certificate gate.
    if crash_status in {
            "BASIS FAILURE", "FACTORIZATION FAILURE", "CYCLE", "STALLED",
            "NO BLOCKER", "FALSE KKT"}:
        projected = project_lex_active_set(
            face_B, zero_target, face_d, t=1.0, tol=tol, maxiter=maxiter,
            max_perturbations=max_perturbations,
            max_tangent_escapes=max_tangent_escapes, trace=trace, y0=face_y0,
            warm_start_active_bounds=False)
        backtracked = True
    result = dict(projected)
    result.update(
        projection_status=projected.get("status"),
        crash_status=crash_status,
        crash_iterations=crash_iterations,
        crash_backtracked=backtracked,
        face_mode=face_mode,
        face_coordinates=int(face_B.shape[0]),
        fixed_zero_coordinates=int(n - face_B.shape[0]) if x0 is not None else 0,
        selected_slack_error=selected_slack_error,
        optimal_value=fstar,
        initial_face_error=initial_face_error,
        initial_nonnegative_error=initial_nonnegative_error,
        initial_norm=float(np.linalg.norm(y0)),
    )
    if projected.get("status") != "PROJECTED":
        result["status"] = "POLISH FAILED"
        return result

    face_y = np.asarray(projected["y"], dtype=float)
    if x0 is not None:
        y = np.zeros(n)
        y[allowed] = face_y
    else:
        y = face_y
    face_error = _scaled_equality_error(face_B, face_d, face_y)
    dual_error = _scaled_equality_error(B, d, y)
    objective_error = abs(float(b @ y) - fstar) / (
        1.0 + abs(fstar) + float(np.abs(b) @ np.abs(y)))
    nonnegative_error = (max(0.0, float(-y.min()))
                         / (1.0 + float(np.abs(y).max())))
    initial_norm = float(np.linalg.norm(y0))
    polished_norm = float(np.linalg.norm(y))
    norm_increase = max(0.0, polished_norm - initial_norm) / max(1.0, initial_norm)
    secondary_kkt_error = max(
        float(projected.get("equality_error", np.inf)),
        float(projected.get("nonnegative_error", np.inf)),
        float(projected.get("stationarity_error", np.inf)),
    )

    if x0 is not None:
        lp_certificate_pass, lp_certificate_detail = certificate_pair(
            B, b, d, x0, y)

    passed = (
        max(face_error, dual_error, objective_error, nonnegative_error,
            secondary_kkt_error, norm_increase) <= 100.0 * tol
        and (lp_certificate_pass is not False)
    )
    result.update(
        status="POLISHED" if passed else "POLISH CERTIFICATE FAILED",
        y=y,
        face_error=face_error,
        dual_error=dual_error,
        objective_error=objective_error,
        nonnegative_error=nonnegative_error,
        secondary_kkt_error=secondary_kkt_error,
        polished_norm=polished_norm,
        norm_reduction=initial_norm - polished_norm,
        relative_norm_reduction=(initial_norm - polished_norm)
        / max(1.0, initial_norm),
        norm_increase=norm_increase,
        lp_certificate_pass=lp_certificate_pass,
        lp_certificate_detail=lp_certificate_detail,
    )
    return result
