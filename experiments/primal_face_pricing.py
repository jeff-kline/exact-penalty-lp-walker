"""Primal-priced, projection-KKT-gated face repair for the Mangasarian path.

The original objective cannot select among equality multipliers on one face:
``d.T @ (-ua)`` is invariant there.  This module therefore selects an affine
multiplier by forward primal/KKT feasibility first.  Original-objective value
is used only to order bounded neighboring-face proposals after that selector
fails.  No proposal is accepted without the endpoint KKT checks below.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def _increment(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def terminal_candidate_metrics(B, b, d, W, coefficients):
    """Score the terminal primal/dual candidate represented by one face."""
    g, h, ua, _uc, dres = coefficients
    x = -np.asarray(ua)
    y = np.zeros(B.shape[0])
    y[W] = h
    primal_slack = B @ x - b
    primal_scale = 1.0 + np.abs(b) + np.abs(B) @ np.abs(x)
    primal_error = max(
        0.0, float(np.max(-primal_slack / primal_scale)))
    dual_scale = 1.0 + np.abs(y)
    nonnegative_error = max(0.0, float(np.max(-y / dual_scale)))
    objective = float(d @ x)
    dual_objective = float(b @ y)
    gap_error = abs(objective - dual_objective) / max(
        1.0, abs(objective), abs(dual_objective))
    feasibility = max(primal_error, nonnegative_error, float(dres), gap_error)
    slope_scale = max(1.0, float(np.linalg.norm(b[W])))
    return {
        "feasibility": feasibility,
        "primal_error": primal_error,
        "nonnegative_error": nonnegative_error,
        "dual_error": float(dres),
        "gap_error": gap_error,
        "objective": objective,
        "dual_objective": dual_objective,
        "slope_norm": float(np.linalg.norm(g)) / slope_scale,
    }


def select_forward_affine_multiplier(
        B, b, d, W, t, coefficients, relative_probes=(1e-6, 1e-8, 1e-10),
        tol=1e-7, stats=None, column_scale=None):
    """Find an affine multiplier valid at ``t`` and one forward endpoint.

    Variables are ``u0=u(t)`` and ``ua=du/dt``.  Feasibility at both endpoints
    proves every off-face KKT inequality on the intervening interval because
    those residuals are affine in ``t``.

    ``B`` may be a SciPy sparse matrix (opt-in; the walker passes one only
    under ``sparse_b=True``).  The LP handed to HiGHS is then assembled with
    ``scipy.sparse.bmat`` instead of ``np.block`` -- the explicit zero blocks
    ``np.block`` writes are exactly the structural zeros ``bmat`` omits, and
    ``linprog(method="highs-ds")`` converts a dense ``A_eq``/``A_ub`` to CSC
    before calling HiGHS anyway, so the solver sees the same problem.  Only
    the ASSEMBLY changes; the objective, bounds, right-hand sides and the
    residual battery below are untouched.  ``column_scale`` lets the caller
    supply the DENSE ``np.linalg.norm(B, axis=0)`` it already has, so the
    scaling is bit-identical to the dense path rather than recomputed from
    the sparse structure; ``None`` keeps the original in-line computation.
    """
    sparse_input = sp.issparse(B)
    if not sparse_input:
        B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    W = np.asarray(W, dtype=bool)
    g, h, _ua_reference, _uc_reference, dres = coefficients
    n, m = B.shape
    if not W.any() or float(dres) > tol:
        return None, {"reason": "face equality", "error": float(dres)}

    y0 = t * g + h
    yscale = 1.0 + np.abs(y0)
    nonnegative0 = max(0.0, float(np.max(-y0 / yscale)))
    if nonnegative0 > tol:
        return None, {"reason": "negative face point", "error": nonnegative0}

    if column_scale is None:
        column_scale = (np.sqrt(np.asarray(B.multiply(B).sum(axis=0))).ravel()
                        if sparse_input else np.linalg.norm(B, axis=0))
    column_scale = np.asarray(column_scale, dtype=float)
    column_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    off = ~W
    if sparse_input:
        # data[k] / column_scale[indices[k]] is the same IEEE division the
        # dense B / column_scale performs on that entry.
        Bs = B.tocsr(copy=True)
        if Bs.nnz:
            Bs.data = Bs.data / column_scale[Bs.indices]
        BW = Bs[W]
        Boff = Bs[off]
        A_eq = sp.bmat([[BW, None], [None, BW]], format="csr")
        # hoisted out of the probe loop; the dense path recomputes these
        # per probe and is left exactly as it was
        absBW = abs(BW)
        Boff_raw = B.tocsr()[off]
        absBoff_raw = abs(Boff_raw)
        b_off = b[off]
    else:
        Bs = B / column_scale
        BW = Bs[W]
        Boff = Bs[off]
        zeros = np.zeros_like(BW)
        A_eq = np.block([[BW, zeros], [zeros, BW]])
    rhs0 = y0 - t * b[W]
    rhsa = g - b[W]
    b_eq = np.concatenate([rhs0, rhsa])
    bounds = [(None, None)] * (2 * m)
    objective_invariance_target = float(h @ b[W])
    last = {"reason": "no probe"}

    for relative_probe in relative_probes:
        delta = float(relative_probe) * max(1.0, abs(float(t)))
        tp = float(t) + delta
        yp = tp * g + h
        yp_scale = 1.0 + np.abs(yp)
        yp_nonnegative = max(0.0, float(np.max(-yp / yp_scale)))
        if yp_nonnegative > tol:
            last = {"reason": "forward y leaves face", "delta": delta,
                    "error": yp_nonnegative}
            continue

        if off.any():
            if sparse_input:
                A_ub = sp.bmat([[Boff, None],
                                [Boff, delta * Boff]], format="csr")
            else:
                A_ub = np.block([
                    [Boff, np.zeros_like(Boff)],
                    [Boff, delta * Boff],
                ])
            b_ub = np.concatenate([-t * b[off], -tp * b[off]])
        else:
            A_ub = None
            b_ub = None

        _increment(stats, "primal_pricing_affine_lp_calls")
        selected = linprog(
            np.zeros(2 * m), A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ds")
        if not selected.success:
            last = {"reason": "affine selector infeasible", "delta": delta,
                    "solver_status": int(selected.status)}
            continue

        u0s = np.asarray(selected.x[:m])
        uas = np.asarray(selected.x[m:])
        u0 = u0s / column_scale
        ua = uas / column_scale
        uc = u0 - t * ua

        if sparse_input:
            eq0_scale = 1.0 + np.abs(rhs0) + absBW @ np.abs(u0s)
            eqa_scale = 1.0 + np.abs(rhsa) + absBW @ np.abs(uas)
        else:
            eq0_scale = 1.0 + np.abs(rhs0) + np.abs(BW) @ np.abs(u0s)
            eqa_scale = 1.0 + np.abs(rhsa) + np.abs(BW) @ np.abs(uas)
        eq0_error = float(np.max(np.abs(BW @ u0s - rhs0) / eq0_scale))
        eqa_error = float(np.max(np.abs(BW @ uas - rhsa) / eqa_scale))
        off0_error = 0.0
        offp_error = 0.0
        if off.any():
            if sparse_input:
                up = u0 + delta * ua
                q0 = t * b_off + Boff_raw @ u0
                qp = tp * b_off + Boff_raw @ up
                scale0 = (1.0 + np.abs(t * b_off)
                          + absBoff_raw @ np.abs(u0))
                scalep = (1.0 + np.abs(tp * b_off)
                          + absBoff_raw @ np.abs(up))
            else:
                q0 = t * b[off] + B[off] @ u0
                up = u0 + delta * ua
                qp = tp * b[off] + B[off] @ up
                scale0 = 1.0 + np.abs(t * b[off]) + np.abs(B[off]) @ np.abs(u0)
                scalep = 1.0 + np.abs(tp * b[off]) + np.abs(B[off]) @ np.abs(up)
            off0_error = max(0.0, float(np.max(q0 / scale0)))
            offp_error = max(0.0, float(np.max(qp / scalep)))
        objective = float(d @ (-ua))
        invariance_error = abs(objective - objective_invariance_target) / max(
            1.0, abs(objective), abs(objective_invariance_target))
        error = max(eq0_error, eqa_error, off0_error, offp_error,
                    nonnegative0, yp_nonnegative, invariance_error, float(dres))
        detail = {
            "reason": "selected" if error <= tol else "endpoint residual",
            "delta": delta, "t_to": tp, "error": error,
            "equality_at_t": eq0_error, "equality_slope": eqa_error,
            "offface_at_t": off0_error, "offface_at_t_to": offp_error,
            "nonnegative_at_t": nonnegative0,
            "nonnegative_at_t_to": yp_nonnegative,
            "objective_invariance": invariance_error,
            "objective": objective,
        }
        if error <= tol:
            _increment(stats, "primal_pricing_affine_accepts")
            return (g, h, ua, uc), detail
        last = detail
    return None, last


def priced_settle_escape(
        B, b, d, W, t, piece_solver, max_candidates=12, rounds=30,
        relative_probes=(1e-6, 1e-8, 1e-10), tol=1e-7,
        diagnostics=None, stats=None):
    """Backtrack through primal-priced neighboring faces until KKT accepts."""
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    W = np.asarray(W, dtype=bool).copy()
    seen = set()
    _increment(stats, "primal_pricing_calls")
    evaluations = 0
    steps = 0
    objective_descent_steps = 0
    last_selector = {}

    for iteration in range(rounds):
        key = np.packbits(W).tobytes()
        if key in seen:
            if diagnostics is not None:
                diagnostics.update(
                    reason="primal-priced cycle", rounds=iteration,
                    pricing_used=True, pricing_steps=steps,
                    pricing_candidate_evaluations=evaluations,
                    pricing_objective_descent_steps=objective_descent_steps,
                    pricing_selector_reason=last_selector.get("reason"))
            _increment(stats, "primal_pricing_cycle_failures")
            return W, None, False
        seen.add(key)
        if not W.any():
            if diagnostics is not None:
                diagnostics.update(reason="primal-priced empty face",
                                   rounds=iteration, pricing_used=True)
            return W, None, False

        try:
            coefficients = piece_solver(W)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            if diagnostics is not None:
                diagnostics.update(reason="primal-priced face solve",
                                   exception=type(error).__name__,
                                   rounds=iteration, pricing_used=True)
            return W, None, False
        g, h, ua, uc, dres = coefficients
        yW = t * g + h
        y_scale = 1e-8 * max(1.0, float(np.abs(yW).max()) if yW.size else 1.0)
        nonnegative = not bool(np.any(yW < -y_scale))
        if nonnegative and float(dres) <= tol:
            selected, selector_detail = select_forward_affine_multiplier(
                B, b, d, W, t, coefficients,
                relative_probes=relative_probes, tol=tol, stats=stats)
            last_selector = selector_detail
            if selected is not None:
                if diagnostics is not None:
                    diagnostics.update(
                        reason="primal-priced settled", rounds=iteration + 1,
                        pricing_used=True, pricing_steps=steps,
                        pricing_candidate_evaluations=evaluations,
                        pricing_objective_descent_steps=objective_descent_steps,
                        pricing_forward_delta=selector_detail.get("delta"),
                        pricing_endpoint_error=selector_detail.get("error"))
                _increment(stats, "primal_pricing_settled")
                return W, selected, True

        u = t * ua + uc
        residual = -(t * b + B @ u)
        r_scale = 1e-8 * max(
            1.0, float(np.abs(residual).max()) if residual.size else 1.0)
        add_indices = np.where((~W) & (residual < -r_scale))[0]
        support_indices = np.where(W)[0]
        drop_indices = support_indices[yW < -y_scale]

        proposed = []
        for index in add_indices:
            proposed.append((float(-residual[index] / r_scale), int(index)))
        for index in drop_indices:
            position = int(np.searchsorted(support_indices, index))
            proposed.append((float(-yW[position] / y_scale), int(index)))
        proposed.sort(key=lambda item: (-item[0], item[1]))
        proposed = proposed[:max_candidates]
        if not proposed:
            if diagnostics is not None:
                diagnostics.update(
                    reason="primal-priced no proposal", rounds=iteration + 1,
                    pricing_used=True, pricing_steps=steps,
                    pricing_selector_reason=last_selector.get("reason"))
            _increment(stats, "primal_pricing_no_proposal_failures")
            return W, None, False

        current_metrics = terminal_candidate_metrics(B, b, d, W, coefficients)
        candidates = []
        for _severity, index in proposed:
            candidate_W = W.copy()
            candidate_W[index] = ~candidate_W[index]
            candidate_key = np.packbits(candidate_W).tobytes()
            if candidate_key in seen or not candidate_W.any():
                continue
            try:
                candidate_coefficients = piece_solver(candidate_W)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            metrics = terminal_candidate_metrics(
                B, b, d, candidate_W, candidate_coefficients)
            evaluations += 1
            candidates.append((metrics, index, candidate_W))
        _increment(stats, "primal_pricing_candidate_evaluations",
                   len(candidates))
        if not candidates:
            if diagnostics is not None:
                diagnostics.update(reason="primal-priced exhausted",
                                   rounds=iteration + 1,
                                   pricing_used=True, pricing_steps=steps)
            _increment(stats, "primal_pricing_exhausted_failures")
            return W, None, False

        best_feasibility = min(item[0]["feasibility"] for item in candidates)
        harris_band = 1e-6 * max(1.0, best_feasibility)
        eligible = [item for item in candidates
                    if item[0]["feasibility"] <= best_feasibility + harris_band]
        chosen_metrics, _chosen_index, chosen_W = min(
            eligible, key=lambda item: (item[0]["objective"], item[1]))
        if chosen_metrics["objective"] < current_metrics["objective"]:
            objective_descent_steps += 1
            _increment(stats, "primal_pricing_objective_descent_steps")
        W = chosen_W
        steps += 1
        _increment(stats, "primal_pricing_steps")

    if diagnostics is not None:
        diagnostics.update(
            reason="primal-priced round cap", rounds=rounds,
            pricing_used=True, pricing_steps=steps,
            pricing_candidate_evaluations=evaluations,
            pricing_objective_descent_steps=objective_descent_steps,
            pricing_selector_reason=last_selector.get("reason"))
    _increment(stats, "primal_pricing_round_cap_failures")
    return W, None, False
