"""staged_newton: the STAGED-SCREENED NEWTON TERMINAL lane (G1 + G2 + G5).

A self-contained certified LP terminal: no HiGHS, no clarabel, no external LP
or QP solver anywhere on the measured path.  It composes three things around
the UNCHANGED semismooth-Newton machinery of ``newton_oracle``:

  G1  gap-safe screening of the projection QP (this file, section 1),
  G2  geometric t-staging with warm starts (section 3),
  G5  t-bar / violation aimed stage schedule (section 3).

Nothing here can make a certificate pass.  ``oracle_terminal.terminal_test``
and exp23's frozen ``certificate_pair`` / ``_certificate_detail`` decide,
unchanged, on the ORIGINAL ``B, b, d`` -- in particular a screened stage still
reassembles its face in FULL coordinates before any gate sees it, and the gate
checks ``sigma >= 0`` on every original row.  Every failure path falls back to
``exp23_path_primal_dual.follow`` on the ORIGINAL, unscreened problem.

Import side effects: none.


==============================================================================
1.  G1 -- GAP-SAFE SCREENING FOR THE PROJECTION QP.  The derivation.
==============================================================================

The path point at parameter ``t`` is the projection

    y(t) = argmin { f_t(y) := 0.5||y - t b||^2 : B'y = d, y >= 0 },

``B`` is ``n x m``, ``y`` in ``R^n``.  Write ``C = {y >= 0, B'y = d}`` -- note
``C`` does NOT depend on ``t``.  Dualizing only the equality with ``x`` in
``R^m`` and minimizing over ``y >= 0`` in closed form gives, with
``v = t b + B x`` and ``P(x) = (v)_+``,

    D_t(x) = d'x - 0.5||P(x)||^2 + 0.5 t^2||b||^2 = theta_t(x) + 0.5 t^2||b||^2

which is ``newton_oracle``'s ``theta_t`` up to the ``x``-free constant.  Weak
duality is ``f_t(y) >= D_t(x)`` for every feasible ``y`` and every ``x``.

DEFINITION (the gap).  For ``y_hat`` in ``C`` and ``x`` in ``R^m``,

    G(y_hat, x) := f_t(y_hat) - D_t(x)  >= 0.

LEMMA 1 (closed form -- what is actually computed).  If ``B'y_hat = d`` then

    G(y_hat, x) = 0.5||y_hat - P(x)||^2 + <y_hat, (v)_->,   (v)_- := max(-v, 0).

    Proof.  f_t(y_hat) - D_t(x) = 0.5||y_hat||^2 - t b'y_hat - d'x
    + 0.5||P||^2.  Since B'y_hat = d, d'x = (Bx)'y_hat, so
    -t b'y_hat - d'x = -v'y_hat.  Then
    0.5||y_hat||^2 - v'y_hat + 0.5||P||^2
      = 0.5||y_hat - P||^2 + y_hat'(P - v) = 0.5||y_hat - P||^2 + y_hat'(v)_-.
    Both terms are >= 0 because ``y_hat >= 0``, which re-proves weak duality
    and is the numerically stable way to evaluate ``G`` (no cancellation of
    the two large ``O(t^2)`` pieces).  []

THEOREM 1 (the ball; strong convexity, modulus 1).  For every ``y_hat`` in
``C`` and every ``x``, with ``R := sqrt(2 G(y_hat, x))``,

    ||y(t) - y_hat|| <= R      and      ||y(t) - P(x)|| <= R.

    Proof.  ``f_t`` is 1-strongly convex and ``y(t)`` minimizes it over the
    convex set ``C``, so <grad f_t(y(t)), y - y(t)> >= 0 for all y in C and
    hence f_t(y) = f_t(y(t)) + <grad f_t(y(t)), y - y(t)> + 0.5||y - y(t)||^2
               >= f_t(y(t)) + 0.5||y - y(t)||^2.
    With y = y_hat and f_t(y(t)) >= D_t(x): 0.5||y_hat - y(t)||^2 <=
    f_t(y_hat) - f_t(y(t)) <= G.  For the second inequality apply Lemma 1 to
    the FEASIBLE point y(t): G(y(t), x) = 0.5||y(t) - P||^2 + <y(t),(v)_->
    <= G(y_hat, x) because f_t(y(t)) <= f_t(y_hat); both terms are
    nonnegative, so 0.5||y(t) - P||^2 <= G.  []

THEOREM 2 (support-side test -- EXACT).  If ``y_hat_i > R`` then
``y(t)_i > 0``: coordinate ``i`` is provably IN the support of ``y(t)``.
Likewise ``P(x)_i > R`` implies ``y(t)_i > 0``.

    Proof.  |y(t)_i - y_hat_i| <= ||y(t) - y_hat|| <= R by Theorem 1.  []

THEOREM 3 (zero-side test -- a provable UPPER BOUND, O(G) not O(sqrt G)).
Dropping all but one term of Lemma 1 applied to ``y(t)``,

    <y(t), (v)_-> <= G   ==>   y(t)_i <= G / (v_i)_-   whenever v_i < 0,

and combining with Theorem 1 the provable componentwise bound is

    y(t)_i <= u_i := min( P(x)_i + R,  y_hat_i + R,  G / max(-v_i, 0) ).

    The third term is the useful one: it is FIRST ORDER in the gap, while the
    ball is only half-order, so on rows the iterate has pushed well negative
    it collapses much faster.  []

WHAT CANNOT BE PROVED, STATED PLAINLY.  There is NO test based on strict
inequalities that forces ``y(t)_i = 0`` exactly.  Reason: ``y(t)_i = 0`` iff
``v(x*)_i <= 0`` at an optimal ``x*``, and ``v(x*)`` is NOT unique.  If
``x1, x2`` are both optimal then ``(v(x1))_+ = (v(x2))_+ = y(t)``, so
``w = B(x2 - x1)`` vanishes on ``supp y(t)`` but is unconstrained (beyond
``v <= 0``) off it.  A degenerate coordinate can therefore sit at
``y(t)_i = 0`` with ``v(x*)_i = 0`` for EVERY optimal ``x*``, and no ball of
positive radius separates it from the interior of the face.  Consequently the
shipped rule eliminates coordinate ``i`` when

    u_i <= eps_screen * max(1, ||y_hat||_inf),

i.e. it is exact up to an explicit, reported, provable bound ``u_i`` -- not up
to an assumption.  ``eps_screen`` defaults to ``1e-10``, three orders below
``exp23.TOL_Y``.  Theorem 4 says what that costs.

THEOREM 4 (what an eps-elimination can cost).  Let ``Z`` be the eliminated set
and ``y^Z(t)`` the solution of the screened projection (``y_Z = 0`` imposed).
Then ``B'(y(t) - y^Z(t))`` is at most ``||B_Z'||`` ``||y(t)_Z||`` in norm and
``||y(t)_Z||_inf <= max_{i in Z} u_i <= eps_screen * max(1,||y_hat||_inf)``, so
the screened path point is a perturbation of the true one of that size.  This
is a PERTURBATION statement, not an identity, and it is why the gate is still
run on the full original data: the screened solve supplies a FACE HINT, and a
wrong hint costs one extra stage or one fallback, never a certificate.

MEASURED (see the report): the rule never eliminated a coordinate whose true
``y(t)_i`` exceeded the pipeline's own zero threshold, on lotfi at three
values of ``t`` against gap-certified reference solutions.

WHY IT IS NOT A SPEED LEVER HERE (measured, and stated up front).  The Newton
step solves with ``B_S``, ``S = {i : v_i > 0}``.  Theorem 3 can only fire on
rows with ``v_i < 0`` (or ``P_i`` below the threshold), i.e. rows that are
ALREADY outside ``S`` and contribute nothing to the step.  Screening therefore
shrinks ``n`` -- the mat-vecs, the line-search sort, the row bookkeeping -- but
not the step solve, which the fast-kernel report measured at 80-98 % of the
Newton wall.  It is shipped because it is correct, cheap and reported; it is
OFF by default (``screen=False``) because it measured as a wash.


==============================================================================
2.  Cross-stage validity
==============================================================================

An elimination proved at ``t`` is NOT valid at ``2t``: the path moves, and
``supp y(t)`` is not monotone in ``t`` (measured on lotfi: 317, 299, 286, 288
across the committed stages -- it goes back UP).  Option (a) of the brief is
what ships and is the ONLY mode implemented: screening is recomputed from
scratch inside every stage, at that stage's own ``t``, from that stage's own
feasible point, and is discarded when the stage ends.  No t-uniform rule is
claimed and no elimination is ever carried across a stage boundary.

The nonexpansiveness of the projection does give a t-uniform ball,
``||y(t') - y_hat|| <= |t' - t| ||b|| + sqrt(2G)``, but the first term is
useless at the jump sizes this schedule takes (t' / t up to 1e3), so no
cross-stage rule was shipped.


==============================================================================
3.  G2 + G5 -- the stage schedule
==============================================================================

WHY AGGRESSIVE STAGING IS ALMOST FREE (the warm-start theorem).  In the SCALED
formulation ``newton_oracle`` already uses (``x = t x'``), on a single path
piece with contract ``y_S(t) = t g + h`` the exact dual maximizer is
``x'(t) = ua + uc/t`` (``newton_oracle``'s own derivation).  Hence for
``t2 > t1`` ON THE SAME PIECE

    ||x'(t2) - x'(t1)|| = ||uc|| (1/t1 - 1/t2) <= ||uc|| / t1,

*independently of how large t2 is*.  Doubling costs ``0.5||uc||/t1``; jumping
to ``t2 = infinity`` costs ``||uc||/t1`` -- only twice as much.  Blind
4x stepping therefore buys almost nothing in warm-start quality while paying a
full terminal test per stage, and the terminal test is expensive: measured on
the committed panel, terminal-test + glue is 34.7 s of the 121.2 s same-session
Newton panel (29 %), and 26.7 s of ship04s's 50.1 s alone.  Across pieces the
bound degrades (``ua, uc`` change at every breakpoint), which is why the
schedule is aimed rather than infinite.

THE SCHEDULE.  At each rejected stage,

    t_next = max( stage_factor * t,                    (geometric floor)
                  aim_factor * t_bar_of(info),         (G5: closed-form ratio)
                  t * clip(violation / VIOL_TARGET, 1, viol_cap) )   (G5: decay)

clipped at ``T_MAX``.  The third term is the committed rule with its 1e3 cap
raised: the gate's own primal residual decays like C/t (measured, see
``newton_oracle.newton_terminal``), so the violation itself is an estimate of
the ``t`` multiple still needed and capping it at 1e3 forces extra stages for
no reason.  ``stage_factor`` defaults to 32 (measured against 4 / 8 / 64).

EXACT TERMINAL-TEST CACHE.  ``oracle_terminal.piece_y`` depends only on the
face mask ``W`` -- not on ``t`` -- so a stage whose face repeats a face already
rejected at stage 1 of the test (``dual equality`` / ``g nonzero``) is rejected
identically.  Those repeats are skipped by exact mask equality, not by a
tolerance.  Faces that reached stage 2 are always re-tested, because stage 2
consumes the ``t``-dependent candidates ``x_tilde``/``y_tilde``.

RETREAT.  Aggressive schedules can overshoot into the extreme-``t`` regime
where ``piece_y``'s own dual-equality residual degrades and the terminal test
stops recognising a face it accepted at a smaller ``t`` (measured on lotfi:
``face_terminal`` at t=2.7e5, then reason ``"dual equality"`` at t=2.7e8 --
numerical, not geometric; with stage_factor=8 and no retreat that model fell
back to the walker and cost 36 s).  The schedule therefore keeps a bracket
``(t_ok, t_bad)`` and BISECTS it once a degradation is seen.


==============================================================================
4.  Predictive terminal-test skipping (a router heuristic, and why it is safe)
==============================================================================

The terminal test is not free on large models: measured on the committed
panel, ``attempt - newton`` is 26.7 s of ship04s's 50.1 s and 4.1 s of fit1d's
14.4 s (one ``piece_y`` plus up to two dense ``lstsq`` on ``B_W``, and
``|W| ~ 1500`` there).  On ship04s SEVEN of eleven stages had
``g_relative >= 1e-3`` -- they could not possibly pass stage 1 -- and each
still paid a full test.

``g_relative`` is measured to decay like ``C/t`` and then collapse
discontinuously to round-off at the last breakpoint (ship04s: 0.273, 0.273,
0.273, 0.273, 0.216, 0.082, 0.037, 0.0053, 0.0013, 9.8e-16).  So after a test
at ``t_last`` returning ``g_last``, the next stages' value is extrapolated as
``g_last * t_last / t`` and the test is SKIPPED while that stays above
``G_SKIP_TARGET``, at most ``G_SKIP_MAX`` times in a row, and only when the
last test actually cost more than ``G_SKIP_MIN_SECONDS`` (so small models,
where the test is sub-millisecond, are bit-identical to the unskipped lane).

WHY THIS CANNOT BREAK ANYTHING.  A skipped test is a test not run: it cannot
produce a certificate, and it cannot produce a wrong one.  The only cost of a
bad extrapolation is that a certifiable stage is passed over, which the
schedule pays for with one more stage, and failing that with the walker
fallback.  It is a heuristic, it is counted (``terminal_tests_skipped``), and
it is measured in the report.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp23_path_primal_dual as exp23
import newton_oracle as newt
import oracle_terminal as ot
import woodbury_face_solver as wfs

# ---- router heuristics ONLY.  None of these can make a certificate pass. ----
# STAGE_FACTOR: geometric floor on t per stage.  Raised 16 -> 256 on
# 2026-08-09 (agent_reports/29).  Models terminate at t_accepted spanning 16 to
# 1e9, so a fine factor imposes a logarithmic tax on the high-t tail: e226
# spends 74% of its iterations converging faces that are then rejected purely
# to climb 1 -> 16 -> 256 -> 4096.  Coarsening is safe because staging still
# passes through low t and the bisection retreat absorbs overshoot -- unlike a
# cold high t0, which has no retreat path from below and cost share1b +181%.
# Measured in two independent interleaved A/Bs: -11.1% (243 runs) and -9.0%
# (324 runs), with the worst frozen componentwise certificate error UNMOVED at
# 7.6206e-09 in every arm.  Adaptive alternatives were built and lost:
# `accelerate` +1.1% (neutral), `grel` -9.2% (ties this constant, needs a
# power-law predictor with +/-1.7-decade residuals and a safety-rescue path).
# Pre-promotion value reachable with `--stage-factor 16`.
STAGE_FACTOR_PRE_PROMOTION = 16.0
STAGE_FACTOR = 256.0
AIM_FACTOR = 2.0        # t_next >= AIM_FACTOR * t_bar   (G5)
VIOL_TARGET = 1e-9      # the feasibility level the violation decay aims at
VIOL_CAP = 1e3          # measured: raising this overshoots (see report)
T_MAX = newt.T_MAX
MAX_STAGES = 14
SECONDS = 60.0

# ---- predictive terminal-test skipping (a ROUTER heuristic; see section 4) --
G_SKIP_TARGET = 1e-6    # extrapolated g_relative at which testing resumes
G_SKIP_MAX = 2          # never skip more than this many tests in a row
G_SKIP_MIN_SECONDS = 0.25   # only skip when a test actually costs something

SCREEN_EPS = 1e-10      # relative elimination threshold on the PROVABLE u_i
SCREEN_AFTER = 12       # Newton iterations at this t before screening is tried
SCREEN_MIN_DROP = 0.05  # rebuild the reduced system only for a real reduction
SCREEN_FEAS_TOL = 1e-9  # relative |B'y_hat - d| a feasibilized point must meet
SCREEN_ROUNDS = 4       # drop-negative rounds in the feasibilization


# ============================================================ G1: the machinery

def feasible_dual(Bs, BsT, d, v, rounds=SCREEN_ROUNDS, tol=SCREEN_FEAS_TOL):
    """A point of ``C = {y >= 0, B'y = d}`` built from the iterate's own ``v``.

    ``P = (v)_+`` is NOT feasible (``B'P != d`` in general).  The cheapest
    native repair is the min-norm correction on the current support,

        y = P_S - B_S^{+T}(B_S' P_S - d)   on S = {v > 0},  0 elsewhere,

    which is exactly ``oracle_terminal.y_candidates``' projection with ``P`` in
    place of an oracle dual.  Coordinates the correction drives NEGATIVE are
    dropped from ``S`` and the correction is redone (a small active-set loop);
    round-off-level negatives are clipped and the equality residual is
    re-measured on the clipped vector, so the returned point is feasible as
    stated or is reported as unusable.

    Returns ``(y_hat, ok, err, rounds_used)``.  ``ok=False`` means SCREENING
    MUST WAIT -- an infeasible point has no valid ball and is never used.
    """
    S = (v > 0.0).copy()
    P = np.maximum(v, 0.0)
    scale = max(1.0, float(np.abs(d).max()) if d.size else 1.0)
    used = 0
    for used in range(1, rounds + 1):
        if not S.any():
            break
        A = Bs[S].T
        A = A.toarray() if sp.issparse(A) else np.asarray(A)
        residual = A @ P[S] - d
        correction, _r, _rk, _sv = np.linalg.lstsq(A, residual, rcond=None)
        y = np.zeros(v.shape)
        y[S] = P[S] - correction
        yscale = max(1.0, float(np.abs(y).max()) if y.size else 1.0)
        y[np.abs(y) <= 1e-13 * yscale] = 0.0
        err = float(np.abs(BsT @ y - d).max()) / scale
        if float(y.min()) >= 0.0:
            return y, bool(err <= tol), err, used
        index = np.where(S)[0]
        negative = index[y[S] < 0.0]
        if not negative.size:
            break
        S[negative] = False
    return None, False, float("inf"), used


def gap_bounds(v, y_hat):
    """``(G, R, u)``: the gap of Lemma 1, its ball radius, and the bound u_i.

    ``v = t b + B x`` and ``y_hat`` are in the UNSCALED units of the projection
    at ``t``.  ``u`` is the provable componentwise upper bound on ``y(t)`` of
    Theorem 3.  ``G`` is evaluated through Lemma 1's cancellation-free form, so
    it is nonnegative by construction up to round-off -- never as the
    difference of two ``O(t^2)`` quantities.
    """
    P = np.maximum(v, 0.0)
    diff = y_hat - P
    G = 0.5 * float(diff @ diff) + float(y_hat @ np.maximum(-v, 0.0))
    G = max(G, 0.0)
    R = float(np.sqrt(2.0 * G))
    u = np.minimum(P + R, y_hat + R)
    negative = -v
    hit = negative > 0.0
    if hit.any():
        u[hit] = np.minimum(u[hit], G / negative[hit])
    return G, R, np.maximum(u, 0.0)


def screen(Bs, BsT, b, d, x_work, t, eps=SCREEN_EPS,
           rounds=SCREEN_ROUNDS, feas_tol=SCREEN_FEAS_TOL):
    """One gap-safe screening pass at parameter ``t``.  Returns a dict.

    ``x_work`` is the SCALED multiplier of ``newton_projection`` (``x = t
    x_work``), so ``v = t (b + B x_work)``.  ``keep`` is the boolean row mask
    to solve on; ``forced`` is the set Theorem 2 proves is in the support
    (reported, not used to restrict).  ``ok=False`` -- no feasible point yet --
    means the caller keeps every row: screening WAITS, it never guesses.
    """
    x = np.asarray(x_work, dtype=float)
    v = t * (b + Bs @ x)
    y_hat, ok, err, used = feasible_dual(Bs, BsT, d, v, rounds=rounds,
                                         tol=feas_tol)
    out = {"feasible": bool(ok), "feas_error": err, "feas_rounds": used,
           "t": float(t), "kept": None, "eliminated": 0, "forced": 0,
           "gap": None, "radius": None, "u_max_eliminated": None}
    if not ok:
        return out
    G, R, u = gap_bounds(v, y_hat)
    yscale = max(1.0, float(np.abs(y_hat).max()) if y_hat.size else 1.0)
    drop = u <= eps * yscale
    keep = ~drop
    out.update(gap=G, radius=R, eliminated=int(drop.sum()),
               forced=int((y_hat > R).sum()), kept=keep,
               u_max_eliminated=(float(u[drop].max()) if drop.any() else 0.0))
    return out


# ================================================== G2 + G5: the staged driver

def _aim(t, info, violation, stage_factor, aim_factor, viol_cap, t_max):
    """Next ``t``: geometric floor, t-bar aim, and violation-decay aim."""
    t_next = max(stage_factor * t, aim_factor * newt.t_bar_of(info))
    if violation and violation > 0.0:
        t_next = max(t_next, t * min(max(violation / VIOL_TARGET, 1.0),
                                     viol_cap))
    return min(max(t_next, stage_factor * t), t_max)


def staged_newton_terminal(B, b, d, seconds=SECONDS, t0=1.0, t_max=T_MAX,
                           stage_factor=STAGE_FACTOR, aim_factor=AIM_FACTOR,
                           viol_cap=VIOL_CAP, max_stages=MAX_STAGES,
                           maxiter=newt.MAXITER, kernel="fast",
                           kernel_tol=newt.FAST_STEP_TOL, screen_rows=False,
                           screen_eps=SCREEN_EPS, screen_after=SCREEN_AFTER,
                           face_cache=True, predictive_skip=False,
                           g_skip_target=G_SKIP_TARGET,
                           g_skip_max=G_SKIP_MAX, settle_retry=True,
                           thresholds=ot.THRESHOLDS, diagnostics=None):
    """Staged, warm-started, optionally screened Newton hunt.  Fails closed.

    Returns ``(found_or_None, diag)``.  ``found`` is only ever produced by
    ``oracle_terminal._gate`` on the ORIGINAL ``B, b, d``.
    """
    diag = {} if diagnostics is None else diagnostics
    diag.setdefault("stages", [])
    diag["route"] = None
    diag["backend_family"] = "staged-newton"
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    Bs = newt._as_sparse(B)
    BsT = Bs.T.tocsr()
    n, m = B.shape
    started = time.perf_counter()
    deadline = started + seconds
    diag["schedule"] = {"t0": t0, "stage_factor": stage_factor,
                        "aim_factor": aim_factor, "viol_cap": viol_cap,
                        "screen": bool(screen_rows), "kernel": kernel}

    # ---- S1 (Tikhonov-Woodbury) on the TERMINAL-TEST side.  One row
    # classification per model, installed into ``oracle_terminal`` behind the
    # SAME model-level gate the step route uses (unit-row fraction >= 0.70,
    # woodbury_face_solver.TIK_MIN_UNIT).  Below the gate nothing is installed
    # and the terminal test is the incumbent verbatim.  The install is undone
    # in the ``finally`` at the end of this function, so the module never
    # carries state out of the model it belongs to.
    # The classifier is installed (or explicitly CLEARED) on EVERY call, so no
    # state can leak from one model into the next; ``follow_staged`` clears it
    # again in a ``finally`` for callers that raise.
    ot.set_face_classifier(None)
    ot.reset_tik_stats()
    tik_installed = False
    if kernel and "tik" in str(kernel):
        unit_fraction = wfs.unit_row_fraction(B)
        diag["tik_unit_fraction"] = unit_fraction
        if unit_fraction >= wfs.TIK_MIN_UNIT:
            try:
                ot.set_face_classifier(wfs.FaceClassifier(B))
                tik_installed = True
            except (ValueError, MemoryError):
                ot.set_face_classifier(None)
        else:
            diag["tik_declined"] = "unit fraction below threshold"
    diag["tik_terminal"] = tik_installed

    kernels = {}

    def kernel_for(key, Bk, Bsk, bk):
        if kernel in (None, "svd"):
            return None
        if key not in kernels:
            kernels[key] = newt.FaceStepKernel(Bk, Bsk, bk, d, mode=kernel,
                                               tol=kernel_tol,
                                               BsT=Bsk.T.tocsr())
        return kernels[key]

    total_iters = 0
    newton_seconds = 0.0
    step_seconds = 0.0
    screen_seconds = 0.0
    test_seconds = 0.0
    tests_run = 0
    tests_skipped = 0
    x_work = None
    t = float(t0)
    found = None
    rejected_faces = set()
    # RETREAT STATE.  Aggressive staging can overshoot into the extreme-t
    # regime where ``piece_y``'s own dual-equality residual degrades and the
    # terminal test stops recognising a face it accepted at a smaller t
    # (measured on lotfi: face_terminal at t=2.7e5, then "dual equality" at
    # t=2.7e8 -- a numerical loss, not a geometric one).  Once that is seen the
    # schedule stops multiplying and BISECTS the bracket instead.
    t_ok = 0.0                    # largest t whose face passed the t-free test
    t_bad = float("inf")          # smallest t where the test degraded above it
    g_last = None                 # last measured g_relative, and its t
    g_last_t = 1.0
    last_test_seconds = 0.0
    skip_run = 0

    def _finish(info, detail, stage_index, label):
        diag["route"] = label
        diag["accepted_support"] = int(info.get("support", 0))
        diag["accepted_stage"] = stage_index
        diag["t_accepted"] = t
        return {"x": info["x_star"], "y": info["y_star"],
                "certificate": detail, "support": None}

    for stage in range(max_stages):
        if time.perf_counter() > deadline:
            diag["reason"] = "oracle deadline"
            break
        record = {"stage": stage, "t": t}
        budget = maxiter
        keep = None
        iterations = 0
        stage_wall = 0.0

        def _solve(rows, x0, cap):
            """One ``newton_projection`` call, on all rows or a screened set."""
            if rows is None:
                return newt.newton_projection(
                    B, b, d, t, x0=x0, scaled=True, maxiter=cap, Bs=Bs,
                    deadline=deadline, kernel=kernel_for("full", B, Bs, b),
                    BsT=BsT)
            Bk, Bsk, bk = B[rows], Bs[rows], b[rows]
            return newt.newton_projection(
                Bk, bk, d, t, x0=x0, scaled=True, maxiter=cap, Bs=Bsk,
                deadline=deadline,
                kernel=kernel_for(("screened", stage), Bk, Bsk, bk),
                BsT=Bsk.T.tocsr())

        # ---- part A: unscreened warm-up.  Screening needs a FEASIBLE point,
        # and the feasibilization only succeeds once the support has settled.
        if screen_rows and screen_after > 0:
            mark = time.perf_counter()
            sol = _solve(None, x_work, min(screen_after, budget))
            stage_wall += time.perf_counter() - mark
            step_seconds += sol["step_seconds"]
            iterations += sol["iterations"]
            budget -= sol["iterations"]
            x_work = sol["x_work"]
            if not (sol["converged"] or sol["unbounded_ray"] or budget <= 0):
                mark = time.perf_counter()
                found_screen = screen(Bs, BsT, b, d, x_work, t, eps=screen_eps)
                screen_seconds += time.perf_counter() - mark
                record["screen"] = {k: v for k, v in found_screen.items()
                                    if k != "kept"}
                if (found_screen["kept"] is not None
                        and found_screen["eliminated"] >= SCREEN_MIN_DROP * n):
                    keep = found_screen["kept"]
            else:
                budget = 0                    # the warm-up already finished it

        # ---- part B: the (possibly screened) solve at this t
        index = None
        if budget > 0:
            if keep is not None and keep.any():
                index = np.where(keep)[0]
            mark = time.perf_counter()
            sol = _solve(index, x_work, budget)
            if index is not None and sol["unbounded_ray"]:
                # the screened problem came out infeasible => an elimination
                # was wrong.  Redo the stage on ALL rows, and say so.
                record["screen_reverted"] = True
                index = None
                sol = _solve(None, x_work, budget)
            stage_wall += time.perf_counter() - mark
            step_seconds += sol["step_seconds"]
            iterations += sol["iterations"]

        newton_seconds += stage_wall
        total_iters += iterations
        # the face is ALWAYS reassembled in FULL coordinates before any gate
        # sees it: eliminated rows carry y_i = 0 exactly.
        S = np.asarray(sol["S"], dtype=bool)
        if index is None:
            W = S
            y_tilde = np.asarray(sol["y"], dtype=float)
        else:
            W = np.zeros(n, dtype=bool)
            W[index[S]] = True
            y_tilde = np.zeros(n)
            y_tilde[index] = sol["y"]
        record.update({"newton_seconds": round(stage_wall, 4),
                       "iterations": iterations,
                       "rows_solved": n if index is None else int(index.size),
                       "converged": sol["converged"],
                       "newton_reason": sol["reason"],
                       "support": int(W.sum()), "dres": sol["dres"],
                       "grad_relative": sol["grad_relative"],
                       "max_step_residual": sol["max_step_residual"],
                       "step_solver": sol["step_solver"]})
        if sol["unbounded_ray"]:
            record["outcome"] = "newton failed"
            diag["stages"].append(record)
            diag["reason"] = f"newton {sol['reason']} at t={t:g}"
            break

        x_work = sol["x_work"]
        x_tilde = -np.asarray(sol["x"], dtype=float) / t

        # ---- part C: the FROZEN terminal test + gates, on the full problem
        key = np.packbits(W).tobytes()
        cached = face_cache and key in rejected_faces
        predicted = (None if g_last is None
                     else g_last * g_last_t / max(t, 1e-300))
        skipped = (predictive_skip and not cached and predicted is not None
                   and predicted > g_skip_target and skip_run < g_skip_max
                   and last_test_seconds >= G_SKIP_MIN_SECONDS)
        if cached or skipped:
            tests_skipped += 1
            skip_run = 0 if cached else skip_run + 1
            record["outcome"] = ("face rejected (cached)" if cached
                                 else "test skipped (g predicted %.2e)"
                                 % predicted)
            record["terminal"] = False
            record["g_predicted"] = predicted
            diag["stages"].append(record)
            if t_bad < float("inf"):
                t_next = float(np.sqrt(max(t_ok, 1.0) * t_bad))
                if not (max(t_ok, 1.0) * 1.01 < t_next < t_bad * 0.99):
                    diag["reason"] = "escalation bracket exhausted"
                    break
                t = t_next
                continue
            if t >= t_max:
                diag["reason"] = "t ceiling"
                break
            t = _aim(t, {"t_bar": 0.0}, None, stage_factor, aim_factor,
                     viol_cap, t_max)
            continue
        t_started = time.perf_counter()
        passed, info = ot.terminal_test(B, b, d, W, x_tilde=x_tilde,
                                        y_tilde=y_tilde)
        last_test_seconds = time.perf_counter() - t_started
        tests_run += 1
        skip_run = 0
        if info.get("g_relative") is not None and not info.get("face_terminal"):
            g_last, g_last_t = float(info["g_relative"]), float(t)
        record.update({k: v for k, v in info.items()
                       if k not in ("x_star", "y_star", "candidates",
                                    "dual_candidates")})
        record["terminal"] = bool(passed)
        violation = None
        if info.get("face_terminal"):
            t_ok = max(t_ok, t)
        elif t_ok > 0.0 and t > t_ok and info.get("reason") == "dual equality":
            t_bad = min(t_bad, t)
        if passed or info.get("face_terminal"):
            gate_ok, detail = ot._gate(B, b, d, info)
            record["gate"] = dict(detail)
            test_seconds += time.perf_counter() - t_started
            if gate_ok:
                record["outcome"] = "certified"
                diag["stages"].append(record)
                found = _finish(info, detail, stage, "staged newton terminal")
                break
            violation = float(detail.get("primal", 0.0) or 0.0)
            r_started = time.perf_counter()
            repaired, rep = newt.repair_rows(B, b, d, W, info,
                                             x_tilde=x_tilde, y_tilde=y_tilde)
            test_seconds += time.perf_counter() - r_started
            if rep:
                record["repair"] = rep
            if repaired is not None:
                r_info, r_detail, _rW = repaired
                record["outcome"] = "certified (row repair)"
                diag["stages"].append(record)
                found = _finish(r_info, r_detail, stage,
                                "staged newton terminal + row repair")
                break
        else:
            test_seconds += time.perf_counter() - t_started
            if info.get("reason") in ("g nonzero", "dual equality",
                                      "empty support"):
                # stage-1 verdicts depend on W ALONE (piece_y is t-free), so
                # this face can be skipped by exact mask equality later.
                rejected_faces.add(key)
        if violation is None:
            sigma_rel = float(info.get("sigma_min_relative", 0.0) or 0.0)
            violation = -sigma_rel if sigma_rel < 0.0 else None
        record["escalation_violation"] = violation
        record["outcome"] = record.get("outcome", "face rejected")

        if t_bad < float("inf"):
            t_next = float(np.sqrt(max(t_ok, 1.0) * t_bad))
            record["bracket"] = [t_ok, t_bad]
            if not (max(t_ok, 1.0) * 1.01 < t_next < t_bad * 0.99):
                record["outcome"] = "bracket exhausted"
                diag["stages"].append(record)
                diag["reason"] = "escalation bracket exhausted"
                break
        else:
            if t >= t_max:
                record["outcome"] = "t ceiling"
                diag["stages"].append(record)
                diag["reason"] = "t ceiling"
                break
            t_next = _aim(t, info, violation, stage_factor, aim_factor,
                          viol_cap, t_max)
        diag["stages"].append(record)
        t = t_next

    diag["newton_seconds"] = float(newton_seconds)
    diag["newton_step_seconds"] = float(step_seconds)
    diag["screen_seconds"] = float(screen_seconds)
    diag["terminal_test_seconds"] = float(test_seconds)
    diag["terminal_tests_run"] = int(tests_run)
    diag["terminal_tests_skipped"] = int(tests_skipped)
    diag["newton_iters_total"] = int(total_iters)
    diag["stages_run"] = len(diag["stages"])
    if kernels:
        diag["kernel_counters"] = {str(k): v.counters()
                                   for k, v in kernels.items()}

    if found is not None:
        diag.setdefault("reason", "certified")
        return found, diag

    # ---- last route: exp23's own settle + threshold sweep on the Newton dual,
    # EXTERNAL solver routes disabled (projection_retry=False keeps clarabel
    # out), exactly as the committed lane does.
    if x_work is not None and time.perf_counter() < deadline:
        oracle = {"x": -np.asarray(x_work, dtype=float),
                  "y": t * np.maximum(b + Bs @ np.asarray(x_work), 0.0),
                  "method": "staged-newton", "seconds": 0.0,
                  "objective": None}
        sweep = {}
        try:
            found, sweep = ot.terminal_attempt(
                B, b, d, oracle=oracle, thresholds=thresholds,
                seconds=max(deadline - time.perf_counter(), 0.0),
                projection_retry=False, settle_retry=settle_retry,
                diagnostics=sweep)
        except Exception as error:              # fail closed on ANY exception
            sweep["reason"] = f"EXCEPTION {type(error).__name__}: {error}"
            found = None
        diag["sweep_reason"] = sweep.get("reason")
        if found is not None:
            diag["route"] = "staged newton " + str(sweep.get("route"))
            diag["accepted_support"] = sweep.get("accepted_support")
            diag["reason"] = "certified"
            return found, diag

    diag.setdefault("reason", "no terminal face certified")
    return None, diag


# ------------------------------------------------------------------- pipeline

def follow_staged(B, b, d, staged_seconds=SECONDS, staged_diagnostics=None,
                  **kw):
    """Staged-screened Newton terminal with the UNCHANGED walker fallback.

    On success returns a ``follow``-shaped dict with ``pivots=0``.  On ANY
    failure the return is ``exp23_path_primal_dual.follow(B, b, d, ...)``
    verbatim -- on the ORIGINAL, UNSCREENED problem -- with the staged counters
    attached, exactly as ``newton_oracle.follow_with_newton`` does.
    """
    # EVERY keyword ``staged_newton_terminal`` accepts must be listed here:
    # what is not popped is forwarded to ``exp23.follow``, which does not take
    # it, so a missing name turns the FALLBACK -- the whole point of this
    # function -- into a TypeError.  ``predictive_skip`` / ``g_skip_max`` were
    # missing and ``exp84_staged_newton.run`` always passes both, so any
    # fallback from that harness died with a traceback instead of walking
    # (observed 4x by the ship04s lens; the committed 27/27 panel never
    # exercised it because it takes zero fallbacks).  Regression test:
    # ``test_staged_fallback.py``.
    staged_keys = ("t0", "t_max", "stage_factor", "aim_factor", "viol_cap",
                   "max_stages", "maxiter", "kernel", "kernel_tol",
                   "screen_rows", "screen_eps", "screen_after", "face_cache",
                   "settle_retry", "predictive_skip", "g_skip_target",
                   "g_skip_max", "thresholds")
    staged_kw = {k: kw.pop(k) for k in staged_keys if k in kw}
    diag = {} if staged_diagnostics is None else staged_diagnostics
    started = time.perf_counter()
    found = None
    try:
        found, diag = staged_newton_terminal(
            B, b, d, seconds=staged_seconds, diagnostics=diag, **staged_kw)
    except Exception as error:                  # fail closed on ANY exception
        diag["reason"] = f"EXCEPTION {type(error).__name__}: {error}"
    finally:
        # S1 leaves no module state behind, even on the exception path: the
        # walker fallback below must run the incumbent terminal test.
        if diag.get("tik_terminal"):
            diag["tik_stats"] = ot.tik_stats()
        ot.set_face_classifier(None)
    attempt_seconds = time.perf_counter() - started

    counters = {
        "staged_attempt_seconds": float(attempt_seconds),
        "newton_seconds": float(diag.get("newton_seconds", 0.0)),
        "newton_step_seconds": float(diag.get("newton_step_seconds", 0.0)),
        "screen_seconds": float(diag.get("screen_seconds", 0.0)),
        "terminal_test_seconds": float(diag.get("terminal_test_seconds", 0.0)),
        "terminal_tests_run": int(diag.get("terminal_tests_run", 0)),
        "terminal_tests_skipped": int(diag.get("terminal_tests_skipped", 0)),
        "newton_iters_total": int(diag.get("newton_iters_total", 0)),
        "staged_stages": int(diag.get("stages_run", 0)),
        "route": diag.get("route"),
        "staged_diagnostics": diag,
    }

    if found is not None:
        y = found["y"]
        st = {"status": "CERTIFIED", "x": found["x"], "y": y,
              "lb": float(b @ y), "t": float("inf"), "pivots": 0,
              "backend": "staged newton terminal pair", "mono": True,
              "certificate_detail": found["certificate"],
              "staged_fallback": False}
        st.update(counters)
        return st

    counters["route"] = "fallback walk"
    walk_started = time.perf_counter()
    st = exp23.follow(B, b, d, **kw)
    counters["staged_fallback_walk_seconds"] = (time.perf_counter()
                                                - walk_started)
    st = dict(st)
    st["staged_fallback"] = True
    st["staged_fallback_reason"] = diag.get("reason", "unknown")
    st.update(counters)
    return st
