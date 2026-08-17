"""Optional supporting-face crash followed by ordinary primal simplex.

This is a practical escape from a stalled projection path, not a continuation
of that path.  ``highspy`` is deliberately optional: every unavailable/error
result must fall through to the caller's independently certified direct LP
fallback.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csc_matrix


def _failure(reason, **detail):
    return {"status": "UNAVAILABLE", "reason": reason, **detail}


def supporting_face_crossover(B, b, d, projected_y, t,
                              phase_time_limit=5.0, tolerance=1e-8):
    """Return a simplex candidate from a certified projection point.

    The returned primal-dual pair is only a candidate.  The caller must apply
    its original-data LP certificate before accepting it.
    """
    try:
        import highspy
    except ImportError:
        return _failure("highspy unavailable")

    try:
        B = np.asarray(B, dtype=float)
        b = np.asarray(b, dtype=float)
        d = np.asarray(d, dtype=float)
        projected_y = np.asarray(projected_y, dtype=float)
        n, m = B.shape
        if (b.shape != (n,) or d.shape != (m,)
                or projected_y.shape != (n,)):
            return _failure("shape mismatch")
        if not (np.all(np.isfinite(B)) and np.all(np.isfinite(b))
                and np.all(np.isfinite(d))
                and np.all(np.isfinite(projected_y))
                and np.isfinite(t) and t > 0.0):
            return _failure("nonfinite input")

        matrix = csc_matrix(B.T)
        supporting_cost = projected_y - t * b
        cost_scale = max(1.0, float(np.max(np.abs(supporting_cost))))

        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = m
        lp.col_cost_ = supporting_cost / cost_scale
        lp.col_lower_ = np.zeros(n)
        lp.col_upper_ = np.full(n, highspy.kHighsInf)
        lp.row_lower_ = d.copy()
        lp.row_upper_ = d.copy()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
        lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
        lp.a_matrix_.value_ = matrix.data.astype(float)

        solver = highspy.Highs()
        solver.setOptionValue("output_flag", False)
        solver.setOptionValue("threads", 1)
        solver.setOptionValue("presolve", "off")
        solver.setOptionValue("simplex_strategy", 4)
        solver.setOptionValue("time_limit", float(phase_time_limit))
        pass_status = solver.passModel(lp)
        if pass_status != highspy.HighsStatus.kOk:
            return _failure("supporting model rejected",
                            highs_status=int(pass_status))

        started = time.perf_counter()
        solver.run()
        supporting_seconds = time.perf_counter() - started
        supporting_status = solver.getModelStatus()
        supporting_info = solver.getInfo()
        if supporting_status != highspy.HighsModelStatus.kOptimal:
            return _failure(
                "supporting solve not optimal",
                model_status=solver.modelStatusToString(supporting_status),
                supporting_seconds=supporting_seconds,
                supporting_iterations=int(
                    supporting_info.simplex_iteration_count))
        basis = solver.getBasis()
        if not basis.valid:
            return _failure("supporting basis invalid",
                            supporting_seconds=supporting_seconds)

        face_y = np.asarray(solver.getSolution().col_value)
        equality_scale = 1.0 + np.abs(d) + np.abs(B).T @ np.abs(face_y)
        equality_error = float(np.max(
            np.abs(B.T @ face_y - d) / equality_scale))
        nonnegative_error = (max(0.0, float(-face_y.min()))
                             / (1.0 + float(np.max(np.abs(face_y)))))
        projected_objective = float(supporting_cost @ projected_y)
        face_objective = float(supporting_cost @ face_y)
        supporting_gap = abs(face_objective - projected_objective) / (
            1.0 + abs(face_objective) + abs(projected_objective))
        if max(equality_error, nonnegative_error, supporting_gap) > tolerance:
            return _failure(
                "supporting face certificate failed",
                supporting_gap=supporting_gap,
                equality_error=equality_error,
                nonnegative_error=nonnegative_error,
                supporting_seconds=supporting_seconds,
                supporting_iterations=int(
                    supporting_info.simplex_iteration_count))

        indices = np.arange(n, dtype=np.int32)
        cost_status = solver.changeColsCost(n, indices, -b.astype(float))
        if cost_status != highspy.HighsStatus.kOk:
            return _failure("objective switch rejected",
                            highs_status=int(cost_status))
        solver.setOptionValue("time_limit", float(phase_time_limit))
        started = time.perf_counter()
        solver.run()
        simplex_seconds = time.perf_counter() - started
        simplex_status = solver.getModelStatus()
        simplex_info = solver.getInfo()
        if simplex_status != highspy.HighsModelStatus.kOptimal:
            return _failure(
                "objective-switch solve not optimal",
                model_status=solver.modelStatusToString(simplex_status),
                simplex_seconds=simplex_seconds,
                simplex_iterations=int(simplex_info.simplex_iteration_count),
                supporting_gap=supporting_gap)

        solution = solver.getSolution()
        dual_y = np.asarray(solution.col_value)
        primal_x = -np.asarray(solution.row_dual)
        if not (np.all(np.isfinite(primal_x))
                and np.all(np.isfinite(dual_y))):
            return _failure("nonfinite crossover solution")
        return {
            "status": "CANDIDATE",
            "x": primal_x,
            "y": dual_y,
            "detail": {
                "supporting_seconds": supporting_seconds,
                "supporting_iterations": int(
                    supporting_info.simplex_iteration_count),
                "supporting_gap": supporting_gap,
                "supporting_equality_error": equality_error,
                "supporting_nonnegative_error": nonnegative_error,
                "simplex_seconds": simplex_seconds,
                "simplex_iterations": int(
                    simplex_info.simplex_iteration_count),
                "basis_valid": True,
            },
        }
    except Exception as error:
        # Fail closed.  In particular, never let optional API drift disable the
        # direct certified LP fallback.
        return _failure("crossover exception", exception=type(error).__name__)

