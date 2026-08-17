"""SHIPPED single-RRQR terminal reconstruction for overcomplete faces.

This is the ``qr_cod`` certificate-detail route.  It left quarantine after the
model-interleaved A/B of ``bench_certificate_routes.py`` (486 guarded runs,
-3.098 s panel, -13.9 %, every accuracy component unchanged against the frozen
incumbent) and it is the route ``exp84_staged_newton`` installs by default;
``--route frozen`` restores ``exp23_path_primal_dual._certificate_detail``.

IT IS NOT A GATE AND IT CANNOT WEAKEN ONE.  ``_certificate_detail`` only
proposes a primal ``x`` for a dual ``y``; the verdict is
``exp23_path_primal_dual.certificate_pair``, componentwise, TOL_FEAS = 1e-7, on
the ORIGINAL raw (B, b, d).  This route returns ``True`` only after that frozen
pair check passes (the last block of ``qr_cod_certificate_detail``), and the
entry point additionally re-runs the frozen gate on the original data after
un-equilibrating, so the certified pair is always judged on unscaled input.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla

import exp23_path_primal_dual as exp23
import hybrid_certificate_factor as hybrid


_CACHE = None
_STATS = None


def reset_stats():
    global _CACHE, _STATS
    _CACHE = {}
    _STATS = {
        "calls": 0, "passes": 0, "factor_seconds": 0.0,
        "solve_seconds": 0.0, "null_completion_seconds": 0.0,
        "certificate_seconds": 0.0, "total_seconds": 0.0,
        "cache_hits": 0, "cache_misses": 0, "dispatches": 0,
        "failures": 0, "support_rows": [], "lstsq_ranks": [],
        "nullities": [], "rank_gap_lstsq": [], "rank_gap_null": [],
        "reasons": {},
    }


def snapshot():
    if _STATS is None:
        reset_stats()
    return {key: (list(value) if isinstance(value, list)
                  else dict(value) if isinstance(value, dict) else value)
            for key, value in _STATS.items()}


def _finish(started, result):
    stats = _STATS
    ok, _x, _err, reason = result
    stats["passes"] += int(bool(ok))
    stats["reasons"][str(reason)] = stats["reasons"].get(str(reason), 0) + 1
    stats["total_seconds"] += time.perf_counter() - started
    return result


def _rank(diagonal, cut):
    return int(np.sum(np.asarray(diagonal) > float(cut)))


def _gap(diagonal, rank, cut):
    above = (float(diagonal[rank - 1] / cut) if rank else float("inf"))
    below = (float(cut / diagonal[rank])
             if rank < len(diagonal) and diagonal[rank] > 0.0
             else float("inf"))
    return min(above, below)


def _minimum_norm(R, pivot, rhs, rank):
    """Minimum-norm solution of the rank-truncated pivoted-QR system."""
    m = R.shape[1]
    if rank == 0:
        return np.zeros(m)
    Rr = R[:rank]
    R11 = Rr[:, :rank]
    trailing = Rr[:, rank:]
    u = sla.solve_triangular(R11, rhs[:rank], lower=False,
                             check_finite=False)
    if trailing.shape[1]:
        W = sla.solve_triangular(R11, trailing, lower=False,
                                 check_finite=False)
        core = np.eye(W.shape[1]) + W.T @ W
        chol = sla.cho_factor(core, lower=True, check_finite=False)
        u = u - W @ sla.cho_solve(chol, W.T @ u, check_finite=False)
    z = sla.solve_triangular(R11.T, u, lower=True, check_finite=False)
    z = Rr.T @ z
    x = np.empty(m)
    x[np.asarray(pivot, dtype=np.intp)] = z
    return x


def _null_basis(R, pivot, rank):
    """Orthonormal null basis of the rank-truncated RRQR factor."""
    m = R.shape[1]
    deficiency = m - rank
    if deficiency <= 0:
        return np.empty((m, 0))
    if rank:
        W = sla.solve_triangular(R[:rank, :rank], R[:rank, rank:],
                                 lower=False, check_finite=False)
        raw = np.vstack((-W, np.eye(deficiency)))
    else:
        raw = np.eye(m)
    basis, _ = sla.qr(raw, mode="economic", check_finite=False)
    out = np.empty_like(basis)
    out[np.asarray(pivot, dtype=np.intp)] = basis
    return out


def qr_cod_certificate_detail(B, b, d, y):
    """Drop-in certificate detail; RRQR is used only when rows > columns."""
    global _CACHE, _STATS
    if _STATS is None:
        reset_stats()
    started = time.perf_counter()
    stats = _STATS
    stats["calls"] += 1
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    sc_d = max(1.0, float(np.abs(d).max()))
    dres = float(np.max(np.abs(B.T @ y - d)))
    if dres > exp23.TOL_FEAS * sc_d or (
            y.size and float(y.min()) < -exp23.TOL_FEAS):
        reason = ("dual equality" if dres > exp23.TOL_FEAS * sc_d
                  else "negative dual")
        return _finish(started, (False, None, dres / sc_d, reason))
    A = np.where(y > exp23.TOL_Y * max(
        1.0, float(y.max()) if y.size else 1.0))[0]
    stats["support_rows"].append(int(A.size))
    m = B.shape[1]
    if A.size <= m:
        stats["dispatches"] += 1
        return _finish(started, hybrid.single_factor_certificate_detail(
            B, b, d, y))
    key = (B.shape, A.tobytes())
    cached = _CACHE.get(key)
    sc_b = max(1.0, float(np.abs(b).max()))
    if cached is None:
        stats["cache_misses"] += 1
        BA, bA = np.asarray(B[A], float), np.asarray(b[A], float)
        factor_started = time.perf_counter()
        try:
            Q, R, pivot = sla.qr(BA, mode="economic", pivoting=True,
                                 check_finite=False)
        except Exception:
            stats["factor_seconds"] += time.perf_counter() - factor_started
            stats["failures"] += 1
            return _finish(started, (False, None, np.inf, "RRQR failure"))
        stats["factor_seconds"] += time.perf_counter() - factor_started
        solve_started = time.perf_counter()
        diagonal = np.abs(np.diag(R))
        top = float(diagonal[0]) if diagonal.size else 0.0
        lstsq_cut = np.finfo(float).eps * max(BA.shape) * top
        null_cut = 1e-10 * top
        rank = _rank(diagonal, lstsq_cut) if top else 0
        null_rank = _rank(diagonal, null_cut) if top else 0
        stats["rank_gap_lstsq"].append(_gap(diagonal, rank, lstsq_cut))
        stats["rank_gap_null"].append(_gap(diagonal, null_rank, null_cut))
        qtb = Q.T @ bA
        x = _minimum_norm(R, pivot, qtb, rank)
        consist = float(np.max(np.abs(BA @ x - bA)))
        viol = max(0.0, float(-(B @ x - b).min()))
        nullity = m - null_rank
        stats["solve_seconds"] += time.perf_counter() - solve_started
        if (consist <= exp23.TOL_FEAS * sc_b
                and viol > exp23.TOL_FEAS * sc_b and nullity > 0):
            null_started = time.perf_counter()
            N = _null_basis(R, pivot, null_rank)
            z = exp23.newton4(B @ N, b - B @ x,
                              np.zeros(N.shape[1]), 1.0)
            x2 = x + N @ z
            viol2 = max(0.0, float(-(B @ x2 - b).min()))
            if viol2 < viol:
                x, viol = x2, viol2
                consist = float(np.max(np.abs(BA @ x - bA)))
            stats["null_completion_seconds"] += (
                time.perf_counter() - null_started)
        cached = {"x": x.copy(), "consist": consist, "viol": viol,
                  "rank": rank, "nullity": nullity}
        _CACHE[key] = cached
    else:
        stats["cache_hits"] += 1
    x = np.asarray(cached["x"], float).copy()
    consist, viol = float(cached["consist"]), float(cached["viol"])
    stats["lstsq_ranks"].append(int(cached["rank"]))
    stats["nullities"].append(int(cached["nullity"]))
    if consist > exp23.TOL_FEAS * sc_b:
        return _finish(started, (False, x, consist / sc_b,
                                 "active consistency"))
    if viol > exp23.TOL_FEAS * sc_b:
        return _finish(started, (False, x, viol / sc_b, "primal violation"))
    px, dy = float(d @ x), float(b @ y)
    gap = abs(px - dy) / max(1.0, abs(px), abs(dy))
    if gap > exp23.TOL_FEAS:
        return _finish(started, (False, x, gap, "duality gap"))
    certificate_started = time.perf_counter()
    ok, detail = exp23.certificate_pair(B, b, d, x, y)
    stats["certificate_seconds"] += time.perf_counter() - certificate_started
    if not ok:
        error = max(detail.get("primal", 0.0), detail.get("dual", 0.0),
                    detail.get("nonnegative", 0.0), detail.get("gap", 0.0))
        return _finish(started, (False, x, error,
                                 "componentwise " + detail["reason"]))
    return _finish(started, (True, x, max(consist, viol) / sc_b, "passed"))


reset_stats()
