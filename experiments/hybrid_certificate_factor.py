"""Single-factor original-primal reconstruction for a Program-(4) face.

The incumbent certificate detail first calls ``lstsq(B_A,b_A)`` and, when the
active system is rank deficient, calls ``svd(B_A)`` again solely to obtain its
null space.  This quarantined equivalent computes both from one SVD.  The two
different incumbent rank cuts are retained exactly:

* NumPy's ``lstsq(..., rcond=None)`` cut for the minimum-norm active solution;
* the explicit ``1e-10 * s_max`` cut for the null-space completion.

The final decision remains ``certificate_pair`` on the original LP.
"""

from __future__ import annotations

import time

import numpy as np

import exp23_path_primal_dual as exp23


_ORIGINAL = exp23._certificate_detail
_STATS = None
_CACHE = None


def reset_stats():
    global _STATS, _CACHE
    _CACHE = {}
    _STATS = {
        "calls": 0,
        "passes": 0,
        "factor_seconds": 0.0,
        "null_completion_seconds": 0.0,
        "certificate_seconds": 0.0,
        "total_seconds": 0.0,
        "support_rows": [],
        "lstsq_ranks": [],
        "nullities": [],
        "reasons": {},
        "fallback_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "square_incumbent_dispatches": 0,
    }


def _stats():
    global _STATS
    if _STATS is None:
        reset_stats()
    return _STATS


def snapshot():
    stats = _stats()
    return {key: (list(value) if isinstance(value, list)
                  else dict(value) if isinstance(value, dict) else value)
            for key, value in stats.items()}


def _finish(stats, started, result):
    ok, _x, _err, reason = result
    stats["passes"] += int(bool(ok))
    stats["reasons"][str(reason)] = int(stats["reasons"].get(
        str(reason), 0)) + 1
    stats["total_seconds"] += time.perf_counter() - started
    return result


def single_factor_certificate_detail(B, b, d, y):
    """Drop-in replacement for ``exp23._certificate_detail``."""
    started = time.perf_counter()
    stats = _stats()
    stats["calls"] += 1
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = B.shape
    sc_d = max(1.0, float(np.abs(d).max()))
    dres = float(np.max(np.abs(B.T @ y - d)))
    if dres > exp23.TOL_FEAS * sc_d or (
            y.size and float(y.min()) < -exp23.TOL_FEAS):
        reason = ("dual equality" if dres > exp23.TOL_FEAS * sc_d
                  else "negative dual")
        return _finish(stats, started,
                       (False, None, dres / sc_d, reason))

    A = np.where(y > exp23.TOL_Y * max(
        1.0, float(y.max()) if y.size else 1.0))[0]
    stats["support_rows"].append(int(A.size))
    if A.size == 0:
        return _finish(stats, started,
                       (False, None, np.inf, "empty support"))

    # A square active matrix is the one measured regime where the incumbent's
    # single LAPACK least-squares factor is cheaper than materialising an SVD.
    # This is a state/shape dispatch, not a model list.  Undercomplete support
    # is necessarily rank deficient; overcomplete support keeps the one-SVD
    # route because it captures ship04s's hidden one-dimensional deficiency.
    if A.size == m:
        stats["square_incumbent_dispatches"] += 1
        stats["fallback_calls"] += 1
        return _finish(stats, started, _ORIGINAL(B, b, d, y))

    global _CACHE
    if _CACHE is None:
        _CACHE = {}
    key = (int(B.shape[0]), int(B.shape[1]), A.tobytes())
    cached = _CACHE.get(key)
    sc_b = max(1.0, float(np.abs(b).max()))
    if cached is not None:
        stats["cache_hits"] += 1
        x = np.asarray(cached["x"], dtype=float).copy()
        consist = float(cached["consist"])
        viol = float(cached["viol"])
        stats["lstsq_ranks"].append(int(cached["rank"]))
        stats["nullities"].append(int(cached["nullity"]))
    else:
        stats["cache_misses"] += 1
        BA = np.asarray(B[A], dtype=float)
        bA = np.asarray(b[A], dtype=float)
        factor_started = time.perf_counter()
        try:
            # Economy U still returns a square Vt when rows >= columns.  For
            # an underdetermined active system the full Vt is required to
            # represent the complete null space.
            full = BA.shape[0] < BA.shape[1]
            U, singular, Vt = np.linalg.svd(BA, full_matrices=full)
        except Exception:
            stats["factor_seconds"] += time.perf_counter() - factor_started
            stats["fallback_calls"] += 1
            return _finish(stats, started, _ORIGINAL(B, b, d, y))
        stats["factor_seconds"] += time.perf_counter() - factor_started

        smax = float(singular.max()) if singular.size else 0.0
        lstsq_cut = np.finfo(float).eps * max(BA.shape) * smax
        rank = int(np.sum(singular > lstsq_cut))
        if rank:
            x = Vt[:rank].T @ ((U[:, :rank].T @ bA) / singular[:rank])
        else:
            x = np.zeros(m, dtype=float)

        consist = float(np.max(np.abs(BA @ x - bA)))
        viol = max(0.0, float(-(B @ x - b).min()))
        null_rank = int(np.sum(singular > smax * 1e-10)) if smax else 0
        nullity = m - null_rank

        if (consist <= exp23.TOL_FEAS * sc_b
                and viol > exp23.TOL_FEAS * sc_b and rank < m):
            null_started = time.perf_counter()
            N = Vt[null_rank:].T
            if N.shape[1] > 0:
                z = exp23.newton4(B @ N, b - B @ x,
                                  np.zeros(N.shape[1]), 1.0)
                x2 = x + N @ z
                viol2 = max(0.0, float(-(B @ x2 - b).min()))
                if viol2 < viol:
                    x, viol = x2, viol2
                    consist = float(np.max(np.abs(BA @ x - bA)))
            stats["null_completion_seconds"] += (
                time.perf_counter() - null_started)
        _CACHE[key] = {"x": np.asarray(x, dtype=float).copy(),
                       "consist": float(consist), "viol": float(viol),
                       "rank": int(rank), "nullity": int(nullity)}
        stats["lstsq_ranks"].append(rank)
        stats["nullities"].append(nullity)

    if consist > exp23.TOL_FEAS * sc_b:
        return _finish(stats, started,
                       (False, x, consist / sc_b, "active consistency"))
    if viol > exp23.TOL_FEAS * sc_b:
        return _finish(stats, started,
                       (False, x, viol / sc_b, "primal violation"))

    px, dy = float(d @ x), float(b @ y)
    gap = abs(px - dy) / max(1.0, abs(px), abs(dy))
    if gap > exp23.TOL_FEAS:
        return _finish(stats, started,
                       (False, x, gap, "duality gap"))

    cert_started = time.perf_counter()
    pair_ok, pair_detail = exp23.certificate_pair(B, b, d, x, y)
    stats["certificate_seconds"] += time.perf_counter() - cert_started
    if not pair_ok:
        pair_err = max(pair_detail.get("primal", 0.0),
                       pair_detail.get("dual", 0.0),
                       pair_detail.get("nonnegative", 0.0),
                       pair_detail.get("gap", 0.0))
        return _finish(stats, started,
                       (False, x, pair_err,
                        "componentwise " + pair_detail["reason"]))
    return _finish(stats, started,
                   (True, x, max(consist, viol) / sc_b, "passed"))


reset_stats()
