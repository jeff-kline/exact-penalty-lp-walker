#!/usr/bin/env python3
"""Faithful, quarantined reference for Pinar's 1997 LPPEN algorithm.

The implementation reconstructs the exact directional-derivative line
search referred to (but not printed) in Section 3.1.  It also uses the
coherent displacement interpretation of the inconsistent Case 2 notation;
see README.md.  It intentionally does not call t-walker or an LP solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.sparse.linalg import lsmr


BACKEND = ("Python/SciPy rank-revealing SVD of B_active for m<=512; "
           "sparse LSMR otherwise")
ALGORITHM = "Pinar1997-LPPEN"


class BudgetExceeded(RuntimeError):
    pass


class NumericalFailure(RuntimeError):
    pass


@dataclass
class Options:
    kappa: float = 0.1
    beta: float = 0.1
    newton_tolerance: float = 2e-10
    solve_tolerance: float = 2e-11
    certificate_tolerance: float = 2e-7
    gap_tolerance: float = 1e-8
    max_outer: int = 200
    max_newton_total: int = 1000
    max_lsmr_iterations: int = 4000
    timeout_seconds: float = 15.0


@dataclass
class Certificate:
    certified: bool
    primal_objective: float
    dual_objective: float
    relative_gap: float
    primal_violation: float
    dual_residual: float
    dual_nonnegativity: float


@dataclass
class Result:
    model: str
    status: str
    detail: str
    backend: str = BACKEND
    algorithm: str = ALGORITHM
    exact_pinar_parameter_policy: bool = True
    exact_pinar_path_used: bool = True
    elapsed_ms: float = math.nan
    n: int = 0
    m: int = 0
    nnz: int = 0
    tau0: float = math.nan
    tau_final: float = math.nan
    outer_reductions: int = 0
    newton_iterations: int = 0
    lsmr_iterations: int = 0
    case1_steps: int = 0
    case2_steps: int = 0
    refactorizations: int = 0
    dense_rank_revealing_factorizations: int = 0
    initialization_guard_used: bool = False
    numerical_fallbacks: list[str] = field(default_factory=list)
    certificate: Optional[Certificate] = None

    def to_dict(self):
        value = asdict(self)
        return value


def _check(deadline: float):
    if time.perf_counter() > deadline:
        raise BudgetExceeded("per-model wall-clock budget exhausted")


def _normal_solve(Bactive: sp.csr_matrix, rhs: np.ndarray, options: Options,
                  deadline: float):
    """Solve C x=rhs or return the null-space projection of rhs.

    This realizes Section 3.1's three cases.  Small models obtain the
    range/null-space split directly from B_active; larger models use the
    residual of an LSMR normal solve as the null-space projection.
    """
    _check(deadline)
    m = Bactive.shape[1]
    if Bactive.shape[0] == 0:
        return rhs.copy(), False, 0
    C = (Bactive.T @ Bactive).tocsc()
    if m <= 512:
        # Pinar's AAFAC forms its factorization from active columns, not from
        # C.  Decomposing B_active itself likewise avoids squaring the
        # condition number and gives an explicit range/null-space split.
        matrix = Bactive.toarray()
        _, singular, vh = la.svd(matrix, full_matrices=True,
                                 compute_uv=True, check_finite=False,
                                 lapack_driver="gesdd")
        cutoff = (max(matrix.shape) * np.finfo(float).eps * singular[0]
                  if singular.size else 0.0)
        rank = int(np.count_nonzero(singular > cutoff))
        coefficients = vh @ rhs
        null_projection = (vh[rank:].T @ coefficients[rank:]
                           if rank < m else np.zeros_like(rhs))
        rhs_scale = max(1.0, float(np.linalg.norm(rhs)))
        consistent = float(np.linalg.norm(null_projection)) <= (
            50.0 * options.solve_tolerance * rhs_scale)
        if consistent:
            x = vh[:rank].T @ (coefficients[:rank] / singular[:rank] ** 2)
            answer = x
        else:
            answer = null_projection
        _check(deadline)
        return answer, consistent, 0
    limit = min(options.max_lsmr_iterations, max(50, 5 * m))
    answer = lsmr(C, rhs, atol=options.solve_tolerance,
                  btol=options.solve_tolerance, maxiter=limit)
    x = np.asarray(answer[0], dtype=float)
    residual = rhs - C @ x
    rhs_scale = max(1.0, float(np.linalg.norm(rhs)))
    consistent = float(np.linalg.norm(residual)) <= (
        20.0 * options.solve_tolerance * rhs_scale)
    _check(deadline)
    return (x if consistent else residual), consistent, int(answer[2])


def _path_solve(Bactive: sp.csr_matrix, rhs: np.ndarray, options: Options,
                deadline: float):
    """Minimum-norm solution of C d=rhs for Pinar's equation (25)."""
    _check(deadline)
    m = Bactive.shape[1]
    C = (Bactive.T @ Bactive).tocsc()
    if m <= 512:
        matrix = Bactive.toarray()
        _, singular, vh = la.svd(matrix, full_matrices=True,
                                 compute_uv=True, check_finite=False,
                                 lapack_driver="gesdd")
        cutoff = (max(matrix.shape) * np.finfo(float).eps * singular[0]
                  if singular.size else 0.0)
        rank = int(np.count_nonzero(singular > cutoff))
        coefficients = vh @ rhs
        null_projection = (vh[rank:].T @ coefficients[rank:]
                           if rank < m else np.zeros_like(rhs))
        scale = max(1.0, float(np.linalg.norm(rhs)))
        consistent = float(np.linalg.norm(null_projection)) <= (
            50.0 * options.solve_tolerance * scale)
        direction = vh[:rank].T @ (coefficients[:rank] / singular[:rank] ** 2)
        # The paper applies one iterative-refinement step to equation (25).
        residual = rhs - C @ direction
        correction_coefficients = vh[:rank] @ residual
        direction += vh[:rank].T @ (
            correction_coefficients / singular[:rank] ** 2)
        residual = rhs - C @ direction
        _check(deadline)
        return direction, consistent, 0, float(np.linalg.norm(residual) / scale)
    limit = min(options.max_lsmr_iterations, max(50, 5 * m))
    answer = lsmr(C, rhs, atol=options.solve_tolerance,
                  btol=options.solve_tolerance, maxiter=limit)
    direction = np.asarray(answer[0], dtype=float)
    residual = rhs - C @ direction
    scale = max(1.0, float(np.linalg.norm(rhs)))
    consistent = float(np.linalg.norm(residual)) <= (
        50.0 * options.solve_tolerance * scale)
    _check(deadline)
    return direction, consistent, int(answer[2]), float(np.linalg.norm(residual) / scale)


def _directional_derivative(tau: float, d: np.ndarray, h: np.ndarray,
                            residual: np.ndarray, ray: np.ndarray,
                            alpha: float) -> float:
    return float(tau * np.dot(d, h) +
                 np.dot(np.minimum(residual + alpha * ray, 0.0), ray))


def _exact_line_search(tau: float, d: np.ndarray, residual: np.ndarray,
                       ray: np.ndarray, h: np.ndarray) -> float:
    """Locate a zero of the piecewise-linear directional derivative.

    This is the natural exact implementation of the line search which Pinar
    delegates to Madsen and Nielsen (1990).
    """
    derivative = _directional_derivative(tau, d, h, residual, ray, 0.0)
    scale = 1.0 + abs(tau * float(np.dot(d, h))) + float(np.linalg.norm(ray))
    if derivative >= -2e-14 * scale:
        return 0.0

    # Activity immediately to the right of zero.  Pinar's extended Wbar
    # includes zero residuals; the q<0 rule resolves the one-sided slope.
    active = (residual < 0.0) | ((residual == 0.0) & (ray < 0.0))
    slope = float(np.dot(ray[active], ray[active]))
    valid = ray != 0.0
    kinks = -residual[valid] / ray[valid]
    kink_ray = ray[valid]
    keep = np.isfinite(kinks) & (kinks > 0.0)
    kinks = kinks[keep]
    kink_ray = kink_ray[keep]
    order = np.argsort(kinks, kind="mergesort")
    kinks = kinks[order]
    kink_ray = kink_ray[order]

    alpha = 0.0
    pos = 0
    while pos < len(kinks):
        next_alpha = float(kinks[pos])
        width = next_alpha - alpha
        if slope > 0.0:
            root_width = -derivative / slope
            if root_width <= width * (1.0 + 5e-14):
                return max(0.0, alpha + max(0.0, root_width))
        derivative += slope * width
        alpha = next_alpha
        # Group equal kinks before changing the one-sided slope.
        end = pos + 1
        tie = 2e-14 * max(1.0, abs(next_alpha))
        while end < len(kinks) and abs(float(kinks[end]) - next_alpha) <= tie:
            end += 1
        for q in kink_ray[pos:end]:
            if q > 0.0:
                slope -= float(q * q)
            else:
                slope += float(q * q)
        slope = max(0.0, slope)
        pos = end

    if slope > 0.0:
        return max(0.0, alpha - derivative / slope)
    if derivative < -2e-12 * scale:
        raise NumericalFailure("penalty line search is unbounded")
    return alpha


def _minimize_penalty(B: sp.csr_matrix, b: np.ndarray, d: np.ndarray,
                      z: np.ndarray, tau: float, options: Options,
                      deadline: float, remaining_iterations: int):
    lsmr_iterations = 0
    for iteration in range(remaining_iterations):
        _check(deadline)
        residual = np.asarray(B @ z - b)
        # Equation (24) uses Pinar's extended Wbar: residual <= 0.
        boundary_tolerance = 5e-12 * (1.0 + float(np.linalg.norm(b, np.inf)))
        active = residual <= boundary_tolerance
        negative = np.minimum(residual, 0.0)
        gradient = tau * d + np.asarray(B.T @ negative)
        grad_scale = 1.0 + tau * float(np.linalg.norm(d, np.inf))
        grad_scale += float(np.linalg.norm(B.T @ negative, np.inf))
        if float(np.linalg.norm(gradient, np.inf)) <= (
                options.newton_tolerance * grad_scale):
            return z, residual, active, iteration, lsmr_iterations, ""

        h, _, used = _normal_solve(B[active], -gradient, options, deadline)
        lsmr_iterations += used
        if float(np.linalg.norm(h)) <= 1e-15 * (1.0 + float(np.linalg.norm(z))):
            detail = (
                "modified Newton direction vanished before stationarity "
                f"(grad_inf={np.linalg.norm(gradient, np.inf):.3e}, "
                f"h_2={np.linalg.norm(h):.3e}, active={np.count_nonzero(active)})")
            return z, residual, active, iteration, lsmr_iterations, detail
        ray = np.asarray(B @ h)
        alpha = _exact_line_search(tau, d, residual, ray, h)
        if not np.isfinite(alpha) or alpha <= 0.0:
            derivative0 = _directional_derivative(
                tau, d, h, residual, ray, 0.0)
            detail = (
                "line search made no positive progress "
                f"(grad_inf={np.linalg.norm(gradient, np.inf):.3e}, "
                f"h_2={np.linalg.norm(h):.3e}, "
                f"psi0={derivative0:.3e}, active={np.count_nonzero(active)})")
            return z, residual, active, iteration, lsmr_iterations, detail
        z = z + alpha * h
    raise NumericalFailure("modified Newton iteration limit reached")


def certify(B: sp.csr_matrix, b: np.ndarray, d: np.ndarray,
            z: np.ndarray, y: np.ndarray, tolerance: float) -> Certificate:
    primal_objective = float(np.dot(d, z))
    dual_objective = float(np.dot(b, y))
    relative_gap = abs(primal_objective - dual_objective) / (
        1.0 + abs(primal_objective) + abs(dual_objective))
    primal_violation = float(np.max(np.maximum(b - B @ z, 0.0), initial=0.0))
    primal_violation /= 1.0 + float(np.linalg.norm(b, np.inf))
    dual_residual = float(np.linalg.norm(B.T @ y - d, np.inf))
    dual_residual /= 1.0 + float(np.linalg.norm(d, np.inf))
    dual_nonnegativity = float(np.max(np.maximum(-y, 0.0), initial=0.0))
    certified = max(relative_gap, primal_violation, dual_residual,
                    dual_nonnegativity) <= tolerance
    return Certificate(certified, primal_objective, dual_objective,
                       relative_gap, primal_violation, dual_residual,
                       dual_nonnegativity)


def solve(model: str, B: sp.spmatrix, b: np.ndarray, d: np.ndarray,
          options: Optional[Options] = None,
          progress: Optional[Callable[[dict], None]] = None) -> Result:
    options = options or Options()
    B = sp.csr_matrix(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    n, m = B.shape
    result = Result(model=model, status="RUNNING", detail="",
                    n=n, m=m, nnz=B.nnz)
    started = time.perf_counter()
    deadline = started + options.timeout_seconds

    try:
        # Paper initialization: (B'B)z0 = B'b - kappa*d.
        init_rhs = np.asarray(B.T @ b) - options.kappa * d
        z, init_consistent, used, init_relres = _path_solve(
            B, init_rhs, options, deadline)
        if m <= 512:
            result.dense_rank_revealing_factorizations += 1
        result.lsmr_iterations += used
        if not init_consistent:
            result.numerical_fallbacks.append(
                f"inexact paper initialization solve (relative residual {init_relres:.3e})")
            result.exact_pinar_path_used = False
        residual0 = np.asarray(B @ z - b)
        if not (1 <= m <= n):
            raise NumericalFailure("paper initialization requires 1 <= rank(A)=m <= n")
        mth = float(np.partition(residual0, m - 1)[m - 1])
        tau = options.beta * mth
        if not np.isfinite(tau) or tau <= 0.0:
            # The paper specifies no guard.  Keep the experiment alive but
            # make the fidelity loss machine-readable and explicit.
            positives = residual0[residual0 > 0.0]
            if positives.size == 0:
                raise NumericalFailure("paper initialization produced nonpositive tau0")
            tau = options.beta * float(np.median(positives))
            result.initialization_guard_used = True
            result.exact_pinar_parameter_policy = False
            result.exact_pinar_path_used = False
            result.numerical_fallbacks.append("nonpositive tau0 guard")
        result.tau0 = tau

        total_newton = 0
        for outer in range(options.max_outer + 1):
            remaining = options.max_newton_total - total_newton
            if remaining <= 0:
                raise NumericalFailure("global modified Newton iteration limit reached")
            z, residual, active, count, used, stall_detail = _minimize_penalty(
                B, b, d, z, tau, options, deadline, remaining)
            if m <= 512:
                result.dense_rank_revealing_factorizations += count
            total_newton += count
            result.newton_iterations = total_newton
            result.lsmr_iterations += used

            # The penalty solution is the standard-form primal in Pinar's
            # notation, i.e. the dual y of the fixture LP.
            y = -np.minimum(residual, 0.0) / tau
            boundary_tolerance = 5e-12 * (
                1.0 + float(np.linalg.norm(b, np.inf)))
            active = residual <= boundary_tolerance
            dpath, consistent, used, path_relres = _path_solve(
                B[active], d, options, deadline)
            if m <= 512:
                result.dense_rank_revealing_factorizations += 1
            result.lsmr_iterations += used
            if not consistent:
                raise NumericalFailure(
                    f"equation (25) inconsistent numerically ({path_relres:.3e})")
            endpoint = z + tau * dpath
            endpoint_residual = np.asarray(B @ endpoint - b)
            gap = float(np.dot(d, endpoint) - np.dot(b, y))
            relative_gap = abs(gap) / (
                1.0 + abs(float(np.dot(d, endpoint))) + abs(float(np.dot(b, y))))
            endpoint_feasible = float(np.min(endpoint_residual)) >= (
                -options.certificate_tolerance *
                (1.0 + float(np.linalg.norm(b, np.inf))))

            if progress:
                progress({"outer": outer, "tau": tau,
                          "newton_iterations": total_newton,
                          "relative_endpoint_gap": relative_gap,
                          "endpoint_min_residual": float(np.min(endpoint_residual)),
                          "active": int(np.count_nonzero(active))})

            if relative_gap <= options.gap_tolerance and endpoint_feasible:
                certificate = certify(B, b, d, endpoint, y,
                                      options.certificate_tolerance)
                result.certificate = certificate
                result.tau_final = tau
                result.outer_reductions = outer
                if certificate.certified:
                    result.status = "CERTIFIED"
                    result.detail = "finite Pinar endpoint certified on fixture B,b,d"
                else:
                    result.status = "NUMERICAL_FAILURE"
                    result.detail = "paper endpoint gate passed but original-data certificate failed"
                break

            if stall_detail:
                raise NumericalFailure(
                    f"{stall_detail}; stalled endpoint not certifiable "
                    f"(relative_gap={relative_gap:.3e}, "
                    f"min_residual={np.min(endpoint_residual):.3e})")

            ray = np.asarray(B @ dpath)
            if relative_gap <= options.gap_tolerance:
                # Case 1: first positive kink on z + alpha*tau*dpath.
                moving = ray != 0.0
                alphas = -residual[moving] / (tau * ray[moving])
                candidates = alphas[(alphas > 1e-13) & (alphas <= 1.0)]
                alpha = float(np.min(candidates)) if candidates.size else 0.9
                alpha = min(0.9, max(alpha, 1e-10)) if not candidates.size else alpha
                z = z + alpha * tau * dpath
                tau = (1.0 - alpha) * tau
                result.case1_steps += 1
            else:
                # Case 2: reconstructed coherent reading of the paper.  Find
                # a displacement whose active-set changes are about half the
                # changes at the zero-parameter endpoint.
                initial_active = residual <= boundary_tolerance

                def changes(delta):
                    return int(np.count_nonzero(
                        (residual + delta * ray <= boundary_tolerance) != initial_active))

                full = changes(tau * (1.0 - 2e-14))
                if full == 0:
                    delta = 0.1 * tau
                    result.numerical_fallbacks.append(
                        "Case 2 had zero predicted active-set changes; used paper lower bound")
                    result.exact_pinar_parameter_policy = False
                    result.exact_pinar_path_used = False
                else:
                    target = max(1, int(math.ceil(0.5 * full)))
                    lo, hi = 0.1 * tau, tau * (1.0 - 2e-14)
                    if changes(lo) >= target:
                        delta = lo
                    else:
                        for _ in range(50):
                            mid = 0.5 * (lo + hi)
                            if changes(mid) >= target:
                                hi = mid
                            else:
                                lo = mid
                        delta = hi
                z = z + delta * dpath
                tau = tau - delta
                result.case2_steps += 1
            result.outer_reductions = outer + 1
            if tau <= np.finfo(float).tiny or not np.isfinite(tau):
                raise NumericalFailure("penalty parameter underflowed")
        else:
            raise NumericalFailure("outer reduction limit reached")

    except BudgetExceeded as error:
        result.status = "TIME_LIMIT"
        result.detail = str(error)
    except NumericalFailure as error:
        result.status = "NUMERICAL_FAILURE"
        result.detail = str(error)
    except Exception as error:  # keep panel results machine-readable
        result.status = "ERROR"
        result.detail = f"{type(error).__name__}: {error}"
    result.elapsed_ms = 1000.0 * (time.perf_counter() - started)
    if not np.isfinite(result.tau_final):
        result.tau_final = tau if "tau" in locals() else math.nan
    return result
