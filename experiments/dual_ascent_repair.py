"""M5 dual-ascent face repair for the Mangasarian projection at fixed t.

At a hard degenerate event the unique projection y* = P_C(t*b) has
rank(B[supp(y*)]) < m, so the equality multiplier u is non-unique over a
k-dimensional affine set, and a support-thresholded working set fails the
min-norm-multiplier sign test even though y* itself is exact (lens 4 of the
pivot-20 diagnosis).  The repair: solve QP_t once with clarabel, then move u
exactly inside {u* + null(B_P)} with a ratio-test ascent that pins one
off-support slack to zero per step.  y* never changes along the motion; after
at most k steps the tight set P ∪ Z spans rank m and the face passes the
walker's ordinary fixed-t gates within 2-3 settle moves (settle itself is
the acceptance gate and may supply a final zero-valued member the sweep's
support thresholds cannot see).

The routine solves only the Mangasarian projection subproblem — no original-LP
objective is involved — and fails closed with a status string.
"""

from __future__ import annotations

import signal
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from exp23_path_primal_dual import piece_y


class RepairTimeout(Exception):
    pass


def _timeout(_signum, _frame):
    raise RepairTimeout("dual-ascent repair ceiling reached")


def _reraise_interrupt(error):
    """Alarm-style interrupts must escape the solver-error handlers.

    A swallowed harness Alarm would be recorded as a genuine M5 failure and
    set m5_last_failed_t, making near-ceiling routing wall-clock-dependent.
    Matching by class name avoids importing the harness (circular).
    """
    if isinstance(error, RepairTimeout) or type(error).__name__ == "Alarm":
        raise error


def _solve_projection(B, tb, d, tol=1e-12, max_iter=400):
    """Solve min 0.5||y - tb||^2 s.t. B'y = d, y >= 0 (clarabel, tight)."""
    import clarabel

    n, m = B.shape
    P = sp.identity(n, format="csc")
    A = sp.vstack([sp.csc_matrix(B.T), -sp.identity(n, format="csc")],
                  format="csc")
    rhs = np.concatenate([d, np.zeros(n)])
    cones = [clarabel.ZeroConeT(m), clarabel.NonnegativeConeT(n)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = max_iter
    for attribute in ("tol_gap_abs", "tol_gap_rel", "tol_feas",
                      "tol_infeas_abs", "tol_infeas_rel"):
        setattr(settings, attribute, tol)
    solution = clarabel.DefaultSolver(P, -tb, A, rhs, cones,
                                      settings).solve()
    y = np.maximum(np.asarray(solution.x), 0.0)
    u = -np.asarray(solution.z)[:m]
    return y, u, str(solution.status)


def _dual_ascent(B, tb, u0, P, maxit=None):
    """Ratio-test ascent inside {u0 + null(B_P)} pinning slacks to zero.

    The slack vector s = tb + B u satisfies s <= 0 off the support of y*
    for every multiplier in the optimal set.  Each step moves along the
    remaining null-space to drive the least-negative free slack toward zero;
    the ratio test pins whichever slack reaches zero first, and that row is
    deflated from the null-space.  y* is untouched: the motion is confined to
    null(B_P).

    Each step deflates one null direction, so the ascent needs exactly the
    nullity k of B_P to complete rank; ``maxit=None`` uses k.  (A fixed cap
    of 64 silently truncated grow7's k=97 ascent at 64 completions — the
    initialization-class diagnosis in agent_reports/11 Addendum 9.)
    """
    eps = np.finfo(float).eps
    BP = B[P]
    _, singular, Vt = np.linalg.svd(BP, full_matrices=True)
    rank = int((singular > singular[0] * max(BP.shape) * eps).sum()
               if singular.size else 0)
    M = Vt[rank:].T
    k = M.shape[1]
    if maxit is None:
        maxit = max(k, 1)
    s = tb + B @ u0
    u = u0.copy()
    Z = np.zeros(B.shape[0], dtype=bool)
    free = (~P).copy()
    steps = 0
    for _ in range(maxit):
        if M.shape[1] == 0:
            break
        BM = B @ M
        candidates = np.where(free & (np.linalg.norm(BM, axis=1) > 0))[0]
        if candidates.size == 0:
            break
        j = candidates[np.argmax(s[candidates])]
        a = BM[j]
        norm_a = float(np.linalg.norm(a))
        if norm_a <= 0.0:
            break
        w = a / norm_a
        rate = BM @ w
        active = free & (rate > 1e-12 * max(1.0,
                                            float(np.abs(rate).max())))
        if not active.any():
            break
        indices = np.where(active)[0]
        alpha = np.where(s[indices] < 0, -s[indices] / rate[indices], 0.0)
        position = int(np.argmin(alpha))
        pinned = indices[position]
        step = float(alpha[position])
        delta = M @ (step * w)
        u = u + delta
        s = s + B @ delta
        s[pinned] = 0.0
        Z[pinned] = True
        free[pinned] = False
        steps += 1
        pinned_row = M.T @ B[pinned]
        M = M @ sla.null_space(pinned_row[None, :])
    return u, Z, steps, k


def _settle_gate(B, b, d, W, t, tol=1e-7, rounds=6, max_adds=0):
    """Accept a candidate tight set through the walker's own settle.

    A boolean min-norm sign test cannot repair the one exact move these
    faces need (e.g. a zero-valued rank-completing member such as Lotfi's
    unit row 554, which sits below every support threshold), and a
    drop-only polish over-drops near-valid faces.  This gate applies
    settle's exact fixed-t moves with SINGLE-ADD discipline: drops are
    taken in batch (drops alone converge — lens 2), but only the single
    worst off-face violator is added per round (batch adds inflate the
    face and steer slower downstream trajectories — measured on Lotfi).
    Acceptance is settle's own settled verdict: dres <= tol and zero moves
    wanted, at the walker's unchanged 1e-8 relative scales.

    ``max_adds=0`` (default) reproduces the strict zero-add gate: drops are
    exact and always taken, but a face demanding an add FAILS.  Measured on
    Lotfi: allowing the add repairs 3 of 4 frozen hard events locally in
    ~60 ms, yet the repaired faces steer a materially slower downstream
    trajectory than the corrector faces they displace (t=6.6-11.4 vs a
    certificate at t=776.8 in the same ceiling).  The completing-add mode
    (max_adds>0) therefore stays available but non-default.
    """
    detail = {"gate_rounds": 0, "gate_adds": 0, "gate_drops": 0}
    W = W.copy()
    for _ in range(rounds):
        if not W.any():
            detail["settle_reason"] = "empty support"
            return None, detail
        detail["gate_rounds"] += 1
        g, h, ua, uc, dres = piece_y(B, b, d, W)
        yW = t * g + h
        r = -(t * (b + B @ ua) + B @ uc)
        y_scale = 1e-8 * max(1.0, float(np.abs(yW).max()) if yW.size
                             else 1.0)
        r_scale = 1e-8 * max(1.0, float(np.abs(r).max()) if r.size else 1.0)
        drop = np.zeros_like(W)
        drop[np.where(W)[0]] = yW < -y_scale
        violators = np.where((~W) & (r < -r_scale))[0]
        if not drop.any() and violators.size == 0:
            detail["settle_reason"] = ("settled" if float(dres) <= tol
                                       else "dual equality")
            detail["settled_support"] = int(W.sum())
            if float(dres) <= tol:
                return W, detail
            return None, detail
        if drop.any():
            W = W & ~drop
            detail["gate_drops"] += int(drop.sum())
            continue
        if detail["gate_adds"] >= max_adds:
            detail["settle_reason"] = "adds required"
            return None, detail
        worst = violators[int(np.argmin(r[violators]))]
        W[worst] = True
        detail["gate_adds"] += 1
    detail["settle_reason"] = "round cap"
    return None, detail


def repair_face(B, b, d, t, tol=1e-7,
                threshold_sweep=(1e-7, 1e-8, 1e-9, 1e-6, 1e-5, 1e-4),
                alarm_seconds=None, settle_gate_adds=0):
    """Return dict(status, working_set, wall, detail); fail-closed."""
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    started = time.perf_counter()
    old_handler = None
    if alarm_seconds is not None:
        old_handler = signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, float(alarm_seconds))
    detail = {"thresholds": []}
    try:
        tb = t * b
        try:
            y_star, u_star, status = _solve_projection(B, tb, d)
        except Exception as error:  # clarabel raises plain Exceptions
            _reraise_interrupt(error)
            return {"status": "FAILED", "working_set": None,
                    "wall": time.perf_counter() - started,
                    "detail": {"reason": "projection solve",
                               "exception": type(error).__name__}}
        detail["projection_status"] = status
        if "Solved" not in status:
            return {"status": "FAILED", "working_set": None,
                    "wall": time.perf_counter() - started, "detail": detail}
        y_max = max(1.0, float(y_star.max()) if y_star.size else 1.0)
        for threshold in threshold_sweep:
            P = y_star > threshold * y_max
            if not P.any():
                detail["thresholds"].append(
                    {"threshold": threshold, "reason": "empty support"})
                continue
            u, Z, steps, k = _dual_ascent(B, tb, u_star, P)
            W = P | Z
            row = {"threshold": threshold, "support": int(P.sum()),
                   "nullity": int(k), "steps": int(steps),
                   "working_set": int(W.sum())}
            # Ascent self-check: the motion is confined to null(B_P), so a
            # healthy ascent keeps every off-tight slack nonpositive and
            # ||u|| within a modest factor of the projection multiplier.  A
            # violation (observed at Lotfi pivot 80: ||u|| 1e8x too large,
            # off-slacks +2.8..+41.6) means the deflated null-space motion
            # lost dual feasibility at THIS threshold's P; fail it closed
            # without settling on garbage.  Per-threshold, not sweep-wide:
            # a different P can ascend cleanly (measured — a sweep-wide
            # abort cost 9 of 22 M5 successes on the certified Lotfi run).
            slack = tb + B @ u
            off = ~W
            max_off_slack = (float(slack[off].max()) if off.any()
                             else -np.inf)
            u_ratio = (float(np.linalg.norm(u))
                       / max(1.0, float(np.linalg.norm(u_star))))
            slack_tol = 1e-6 * max(1.0, float(np.abs(tb).max()))
            if max_off_slack > slack_tol or u_ratio > 1e3:
                row.update(valid=False, reason="ascent breakdown",
                           max_off_slack=max_off_slack, u_ratio=u_ratio)
                detail["thresholds"].append(row)
                detail["ascent_breakdown"] = (
                    detail.get("ascent_breakdown", 0) + 1)
                continue
            settled_W, gate_detail = _settle_gate(B, b, d, W, t, tol=tol,
                                                  max_adds=settle_gate_adds)
            row.update(valid=settled_W is not None, **gate_detail)
            detail["thresholds"].append(row)
            if settled_W is not None:
                detail["winning_threshold"] = threshold
                return {"status": "PASS", "working_set": settled_W,
                        "wall": time.perf_counter() - started,
                        "detail": detail}
        return {"status": "FAILED", "working_set": None,
                "wall": time.perf_counter() - started, "detail": detail}
    except RepairTimeout:
        return {"status": "TIMEOUT", "working_set": None,
                "wall": time.perf_counter() - started, "detail": detail}
    finally:
        if alarm_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)
