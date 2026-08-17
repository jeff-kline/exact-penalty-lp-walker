"""oracle_terminal: an UNTRUSTED interior-point shortcut to the terminal face
of the Mangasarian dual path, with the ordinary walker as the fallback.

WHY THIS CAN BE UNTRUSTED
-------------------------
exp23's certificate gates (``certificate_pair`` / ``_certificate_detail``) are
original-data KKT tests, componentwise-scaled on the primal and dual
residuals, on the ORIGINAL raw ``B, b, d``.  They are
frozen here: this module imports and calls them unchanged and never edits any
existing file.  A wrong oracle hint therefore costs one fallback to the
ordinary walk, never a wrong certificate.  Every threshold, tolerance and
ratio in this file is a ROUTER heuristic -- it only chooses which candidate
face to test.  The gates alone decide CERTIFIED.

THE TERMINAL TEST (t-free)
--------------------------
On a face with support ``S``, ``exp23_path_primal_dual.piece_y`` returns
``g, h, ua, uc, dres`` with the contract

    B_S ua = g - b_S,        B_S uc = h,        y_S(t) = t*g + h,

where ``g = (I - Pi_S) b_S`` and ``h`` is the min-norm solution of
``B_S' h = d``.  Define the primal candidate and its slack

    x* := -ua,               sigma := B x* - b     (sigma_S = -g, exactly).

If ``g == 0`` and ``sigma >= 0`` componentwise, then

  * ``y_S(t) = h`` for EVERY ``t`` -- the piece never ends, so this face is
    the terminal face and no ratio test can leave it;
  * ``x*`` is primal feasible with ``B_S x* = b_S`` (actives exactly ``S``);
  * ``y* := h`` on ``S``, zero elsewhere, satisfies ``B' y* = d`` (up to
    ``dres``) and is complementary with ``x*``, so ``d'x* = y*'B x* = b'y*``
    is a strong-duality IDENTITY, not a limit.

That pair is handed to the frozen gates.  Nothing else in this module is
allowed to declare success.

THE t-BAR RATIO (closed form on a candidate face)
-------------------------------------------------
Off-support slack of the path piece is ``r_j(t) = t*sigma_j - (B uc)_j``
(substitute ``sigma_j = -(b_j + B_j ua)`` into exp23's
``r = -(t*(b + B ua) + B uc)``).  The face is valid exactly while ``r >= 0``:

    t_bar = max{ (B uc)_j / sigma_j : j not in S, (B uc)_j > 0, sigma_j > 0 }.

If some ``j`` has ``sigma_j <= 0 < (B uc)_j`` no finite ``t`` works on this
face -- that row must JOIN the face.  This is self-diagnosing and is reported
as ``blocking_rows``.

ORACLE ESTIMATES
----------------
``scipy.optimize.linprog(d, A_ub=-B, b_ub=-b, method='highs-ipm')`` solves the
primal directly; the dual is ``y~ = -res.ineqlin.marginals`` (sign verified
empirically -- see ``oracle_solve.__doc__`` and the report).  ``S~`` is a
relative-threshold sweep on ``y~``.

Import side effects: none.  This module is imported by nothing in the default
stack.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.optimize import linprog

import scipy.sparse as sp
import scipy.sparse.linalg as spla

import exp23_path_primal_dual as exp23
import woodbury_face_solver as wfs
from exp23_path_primal_dual import certificate_pair, piece_y, settle

# ---- router heuristics ONLY.  None of these can make a certificate pass. ----
THRESHOLDS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
TOL_G = 1e-8          # "is g numerically zero" (relative to |b_S|)
TOL_SIGMA = 1e-8      # componentwise primal-slack sign slack (relative)
TOL_DRES = 1e-7       # piece_y dual-equality residual admissibility
SAFETY_FACTOR = 10.0  # t_hat = SAFETY_FACTOR * t_bar
ATTEMPT_SECONDS = 45.0

# ---------------------------------------------------------------------- S1 --
# TIKHONOV-WOODBURY FACE SOLVER, opt-in and OFF by default.  With no
# classifier installed every path below is the incumbent verbatim: the branch
# is one ``is None`` test and ``piece_y`` / ``lstsq`` run unchanged.
#
# ``set_face_classifier(cls)`` installs it; ``staged_newton`` does that once
# per model behind the unit-row-fraction gate (woodbury_face_solver.TIK_MIN_UNIT)
# and clears it in a ``finally``.
#
# WHY THIS CANNOT CERTIFY SOMETHING FALSE (proof, not a measurement).
# ``g = b_W + B_W ua`` is a genuine least-squares residual AT A GENUINE POINT,
# so ``||g_computed|| >= ||g_true||`` unconditionally.  The route can only ever
# OVER-state ``g``, i.e. produce a false stage-1 REJECT -- never a false
# ACCEPT.  Every failure mode is a fallback, and the frozen gates
# (``certificate_pair`` / ``exp23._certificate_detail``) still decide every
# result on the ORIGINAL ``(B, b, d)``.  On top of that proof the route runs
# two fail-closed self-checks (orthogonality of ``g`` against range(B_W), and
# an independent LSMR upper bound on ``||g||``) and declines on either.
_FACE_CLS = None
TIK_STATS = {"tik_piece": 0, "svd_piece": 0, "tik_fallback": 0,
             "tik_seconds": 0.0, "svd_seconds": 0.0, "cand_reuse": 0,
             "worst_ortho": 0.0, "lsmr_checks": 0, "lsmr_vetoes": 0,
             # ladder instrumentation (see ``_tik_piece_y``)
             "tik_stage_0": 0, "tik_stage_1": 0, "tik_stage_2": 0,
             "tik_stage_3": 0, "tik_stage_attempts": 0,
             "worst_a1": 0.0, "worst_solve_residual": 0.0}
TIK_ORTHO_TOL = 1e-9   # ||B_W' g||_inf / (|B_W|_max |b_W|_inf) budget
TIK_LSMR_CHECK = True  # independent one-sided upper bound on ||g||
TIK_LSMR_ITERS = 400
TIK_LSMR_SLACK = 4.0   # LSMR must beat us by this factor to veto
TIK_LSMR_FLOOR = 1e-9  # below this ||g|| is already an ACCEPT -- sound, skip

# ---- S1 TERMINAL-SIDE RETRY LADDER (the fit1d-declines fix) ----------------
# ``wfs.TIK_LADDER`` is the same ladder of ``(eps_rel, deflate, refine)`` SOLVER
# settings the STEP route already runs (``newton_oracle._tik_step``), and for
# the same measured reason: one fixed shift cannot serve both of fit1d's
# regimes (shift-on-top-of-real-spectrum TRUNCATION, which needs a SMALLER
# shift, and SHALLOW DEFLATION, which needs a deeper projector at the DEFAULT
# shift).  Stage 0 IS the previously shipped single configuration, so a model
# whose faces the default already serves runs stage 0 and nothing else, with
# bit-identical arithmetic (measured: ship04s, 9/9 faces at stage 0).
#
# TWO NEW FAIL-CLOSED SELF-CHECKS ride the ladder, both derived from the
# thresholds the terminal test itself decides on rather than tuned:
#
#  * ``TIK_A1_TOL`` -- the RE-ORTHOGONALISATION CORRECTION.  ``g`` is returned
#    as ``b_W + B_W ua`` minus a correction ``B_W H^+(B_W' g)``.  That
#    correction is mathematically ZERO (``B_W' b_W`` lies in ``range(H)``), so
#    its measured size is pure solver error -- and while it is applied to
#    ``g`` it is NOT applied to ``ua``, which the caller hands to the frozen
#    gate as ``x* = -ua``.  A correction larger than 1 % of the stage-1 ``g``
#    threshold (``TOL_G``) can therefore make ``g`` look terminal at a point
#    whose real face residual is not.  Measured on the shipped route: 8.8e-7
#    on two fit1d faces it ACCEPTED, against <= 5.7e-13 on all nine ship04s
#    faces.
#  * ``TIK_PIECE_TOL`` -- the HONEST RESIDUAL of each of the two solves, in
#    exactly the form the step route uses
#    (``newton_oracle.FaceStepKernel._residual``):
#    ``||B_W'(B_W x) + r_null - rhs||_inf / ||rhs||_inf`` on the ORIGINAL
#    sparse rows.  Set at a tenth of the dual-equality threshold the test
#    decides on (``TOL_DRES``).
#
# Safety is unchanged and is still false-reject-only: the shift is removed
# exactly at every stage (deflation before/after, refinement in the original
# sparse rows), so each candidate is ``H^+ r`` for its own rank cut split by an
# exact orthogonal projector -- the freedom the exact-split law grants
# (``opportunity_degenerate.md`` section 3) -- ``g`` is still ``b_W`` minus an
# element of ``range(B_W)`` and so can only be OVER-stated, a failing stage's
# answer is discarded, and running out of stages falls back to the incumbent.
TIK_LADDER = wfs.TIK_LADDER
TIK_A1_TOL = 1e-10     # = TOL_G / 100
TIK_PIECE_TOL = 1e-8   # = TOL_DRES / 10

# ---------------------------------------------------------------------- S2 --
# PIVOTED-CHOLESKY TERMINAL FACE SOLVER.  ``woodbury_face_solver.piv_chol_step``
# was wired into the Newton STEP path but never into this call site, so the
# whole m <= 700 cohort still paid a dense SVD of ``B_W`` per terminal test.
# One pivoted Cholesky of ``H = B_W'B_W`` serves BOTH right-hand sides at once
# (``piv_chol_step`` is linear in ``grad`` and vectorises over its columns), so
# the face costs one ``dpstrf`` instead of one ``svd``: measured 3.3x-9.2x
# faster on the captured faces of sctap1/bandm/brandy/capri/degen2/scorpion/
# grow7.
#
# ``A1`` HOLDS BY CONSTRUCTION here -- ``g := b_W + B_W ua`` with NO
# re-orthogonalisation, because ``U G^{-1} U'`` is the EXACT orthogonal
# projector onto ``range(H)`` and a second application of the same ``H^+``
# provably cannot move ``g`` (its residual lives entirely in the directions the
# rank cut discarded, which ``H^+`` annihilates).
#
# THE PRICE, AND THE GUARDS.  The rank cut is taken at SQUARED precision
# (``dpstrf`` cuts at ``m eps max_pivot`` on ``H``, i.e. at
# ``s_i/s_max ~ sqrt(m eps) ~ 3e-7``), so on a face whose smallest kept
# singular value sits near that floor the cut MOVES relative to the incumbent's
# ``rcond`` cut and ``g``/``dres`` come out over-stated -- a FALSE REJECT, never
# a false accept (``g`` remains ``b_W`` minus an element of ``range(B_W)``, and
# ``dres`` remains a genuine residual of ``min_uc ||H uc - d||``, so both are
# bounded below by the incumbent's values).  Four fail-closed guards, every one
# of them calibrated on 45 captured m<=700 terminal faces, not guessed:
#
#  1. ``m <= PCHOL_MAXM`` -- the step route's own size guard.
#  2. ``clean >= PCHOL_MIN_CLEAN`` -- ``piv_chol_step``'s cut-cleanliness
#     ratio.  Measured: at ``clean >= 1e3`` every captured face agreed with the
#     incumbent to <= 2.2e-11 on ``g``; the faces that disagreed (lotfi 4.6e-6
#     .. 5.5e-6, share1b 3.3e-2, capri 1.1e-7) all had ``clean <= 1.5e2``.
#     Ten times the step route's own screen (``wfs.PCHOL_CLEAN_GAP``), because
#     a step is judged by a residual afterwards while a terminal verdict is
#     not.
#  3. ``ortho <= TIK_ORTHO_TOL`` -- the same orthogonality budget the S1 route
#     uses, measured on the ORIGINAL sparse operator.  Independently catches
#     the share1b face (6.9e-9) that guard 2 also rejects.
#  4. Decision-margin guards.  The route declines whenever its answer lands
#     near either threshold the terminal test decides on -- ``dres`` above a
#     tenth of ``TOL_DRES``, or ``g_relative`` inside a decade of ``TOL_G`` --
#     so a close call is always taken by the incumbent, and the route's
#     bounded over-statement (<= 2.2e-11 measured, against a 1e-8 threshold)
#     can never flip a verdict.
#
# No ``splu``.  No dense factorization larger than the ``m x m`` Gram matrix,
# which is smaller than the ``|W| x m`` the incumbent already densifies.
PCHOL_TERMINAL = True       # master switch for the route (A/B arms flip this)
PCHOL_MAXM = wfs.PCHOL_MAXM         # 700, mirrors newton_oracle's dense limit
PCHOL_MIN_CLEAN = 1e3               # 10x wfs.PCHOL_CLEAN_GAP; see guard 2
PCHOL_DRES_MARGIN = 0.1             # accept only dres <= 0.1 * TOL_DRES
PCHOL_G_BAND = 10.0                 # decline g_relative within a decade of TOL_G
PCHOL_STATS = {"pchol_piece": 0, "pchol_attempts": 0, "pchol_decline_size": 0,
               "pchol_decline_clean": 0, "pchol_decline_ortho": 0,
               "pchol_decline_margin": 0, "pchol_decline_error": 0,
               "pchol_seconds": 0.0, "pchol_worst_ortho": 0.0,
               "pchol_worst_clean": 0.0, "pchol_deficiency": 0,
               "pchol_confirm": 0}


def set_face_classifier(cls):
    """Install (or clear, with ``None``) the S1 face solver for this module."""
    global _FACE_CLS
    _FACE_CLS = cls


def tik_stats():
    out = dict(TIK_STATS)
    out.update(PCHOL_STATS)
    return out


def reset_tik_stats():
    for stats in (TIK_STATS, PCHOL_STATS):
        for key in stats:
            stats[key] = 0 if isinstance(stats[key], int) else 0.0


def _tik_stage(B, b, d, W, eps_rel, deflate, refine):
    """ONE ladder stage.  ``(g, h, ua, uc, dres, face, BW, BWT, bW, checks)``.

    ``checks`` carries the fail-closed measurements the caller judges; the
    arithmetic producing the five returned quantities is unchanged from the
    single-configuration route when ``(eps_rel, deflate, refine)`` are the
    ``wfs`` defaults, which is what ladder stage 0 passes.
    """
    face = wfs.TikhonovFace(_FACE_CLS, W, eps_rel=eps_rel, deflate=deflate)
    BW = _FACE_CLS.Bs[face.idx]
    BWT = sp.csr_matrix(BW.T)
    bW = np.asarray(b, dtype=float)[face.idx]
    uc, rn_c = face.pinv(d, refine=refine)
    h = BW @ uc
    dres = float(np.linalg.norm(BWT @ h - d)) / max(
        1.0, float(np.linalg.norm(d)))
    rhs_a = BWT @ bW
    ua_pos, rn_a = face.pinv(rhs_a, refine=refine)
    ua = -ua_pos
    g = bW + BW @ ua
    corr, _ = face.pinv(BWT @ g, refine=refine)     # re-orthogonalise g
    g_orth = g - BW @ corr
    # ---- FAIL-CLOSED measurements, all on the ORIGINAL sparse rows -------
    bnorm = max(float(np.abs(BW.data).max()) if BW.nnz else 1.0, 1e-300)
    bwscale = max(1.0, float(np.abs(bW).max()) if bW.size else 1.0)
    checks = {
        # (1) orthogonality: g must be orthogonal to range(B_W).  Measured
        #     against the PROBLEM scale (|b_W|), not |g|: g == 0 is the good
        #     case and must not be punished by a vanishing denominator.
        "ortho": float(np.abs(BWT @ g_orth).max()) / (bnorm * bwscale),
        # (2) how far the re-orthogonalisation had to move g -- zero in exact
        #     arithmetic, so this is pure solver error, and it is NOT applied
        #     to the ``ua`` the frozen gate is handed.
        "a1": float(np.abs(g_orth - g).max()) / bwscale,
        # (3)-(4) honest residuals of the two solves, in the step route's own
        #     form: ||B_W'(B_W x) + r_null - rhs||_inf / ||rhs||_inf.
        "res_c": (float(np.abs(BWT @ h + rn_c - d).max())
                  / max(float(np.abs(d).max()), 1e-300)),
        "res_a": (float(np.abs(BWT @ (BW @ ua_pos) + rn_a - rhs_a).max())
                  / max(float(np.abs(rhs_a).max()), 1e-300)),
    }
    return g_orth, h, ua, uc, dres, face, BW, BWT, bW, checks


def _tik_piece_y(B, b, d, W, refine=wfs.TIK_REFINE):
    """``(g, h, ua, uc, dres, face, B_W, B_W')`` or ``None``.

    All five incumbent quantities are pseudoinverse expressions in
    ``H = B_W'B_W``:  ``uc = H^+ d``, ``h = B_W uc``, ``ua = -H^+ B_W' b_W``,
    ``g = b_W + B_W ua``.  ``None`` means "unavailable or not trusted" and the
    caller runs ``exp23.piece_y`` unchanged.

    Runs the ``TIK_LADDER`` of solver settings, stage 0 first (the previously
    shipped single configuration, so a model the default serves is
    bit-identical and pays nothing extra); a later stage is reached ONLY when
    an earlier one misses one of the fail-closed self-checks, and a failing
    stage's answer is discarded.  Running out of stages returns ``None``.
    """
    if _FACE_CLS is None:
        return None
    started = time.perf_counter()
    for stage, (eps_rel, deflate, refine_s) in enumerate(TIK_LADDER):
        TIK_STATS["tik_stage_attempts"] += 1
        try:
            (g, h, ua, uc, dres, face, BW, BWT, bW,
             checks) = _tik_stage(B, b, d, W, eps_rel, deflate, refine_s)
        except (ValueError, MemoryError, ArithmeticError,
                np.linalg.LinAlgError):
            continue
        TIK_STATS["worst_ortho"] = max(TIK_STATS["worst_ortho"],
                                       checks["ortho"])
        TIK_STATS["worst_a1"] = max(TIK_STATS["worst_a1"], checks["a1"])
        TIK_STATS["worst_solve_residual"] = max(
            TIK_STATS["worst_solve_residual"],
            checks["res_a"], checks["res_c"])
        ok = (bool(np.all(np.isfinite(g)))
              and checks["ortho"] <= TIK_ORTHO_TOL
              and checks["a1"] <= TIK_A1_TOL
              and checks["res_a"] <= TIK_PIECE_TOL
              and checks["res_c"] <= TIK_PIECE_TOL)
        # (5) one-sided cross-check: LSMR gives an INDEPENDENT upper bound on
        #     the same minimum ||b_W - B_W x||.  If LSMR beats us by more than
        #     TIK_LSMR_SLACK our g is over-stated (the shift truncated real
        #     spectrum) and the incumbent SVD runs instead.  Reported honestly
        #     as a PARTIAL guard: it was measured NOT to fire on the models
        #     that need it most (israel, brandy), because LSMR is no better
        #     conditioned on those faces -- which is why the model-level
        #     unit-fraction gate, not this check, is what keeps those models on
        #     the incumbent.
        if ok and TIK_LSMR_CHECK:
            bwscale = max(1.0, float(np.abs(bW).max()) if bW.size else 1.0)
            gn = float(np.linalg.norm(g))
            if gn > TIK_LSMR_FLOOR * bwscale * max(1.0,
                                                   np.sqrt(BW.shape[0])):
                rn = spla.lsmr(BW, bW, atol=1e-12, btol=1e-12,
                               maxiter=TIK_LSMR_ITERS)[3]
                TIK_STATS["lsmr_checks"] += 1
                if float(rn) < gn / TIK_LSMR_SLACK:
                    TIK_STATS["lsmr_vetoes"] += 1
                    ok = False
        if ok:
            TIK_STATS["tik_seconds"] += time.perf_counter() - started
            TIK_STATS["tik_piece"] += 1
            TIK_STATS["tik_stage_%d" % min(stage, 3)] += 1
            return g, h, ua, uc, dres, face, BW, BWT
    TIK_STATS["tik_seconds"] += time.perf_counter() - started
    TIK_STATS["tik_fallback"] += 1
    return None


def _pchol_piece_y(B, b, d, W):
    """``(g, h, ua, uc, dres)`` from ONE pivoted Cholesky, or ``None``.

    ``H = B_W'B_W`` is factored once and applied to BOTH right-hand sides
    (``d`` and ``B_W' b_W``) in a single ``piv_chol_step`` call -- the kernel is
    linear in ``grad`` and vectorises over its columns, so the face costs one
    ``dpstrf`` rather than two, and rather than the incumbent's dense SVD of
    ``B_W``.  ``None`` means "unavailable or not trusted"; the caller then runs
    ``exp23.piece_y`` unchanged.  See the guard rationale above.
    """
    n, m = np.shape(B)
    if not PCHOL_TERMINAL or m <= 0 or m > PCHOL_MAXM:
        PCHOL_STATS["pchol_decline_size"] += 1
        return None
    started = time.perf_counter()
    PCHOL_STATS["pchol_attempts"] += 1
    try:
        idx = np.where(np.asarray(W, dtype=bool))[0]
        if idx.size == 0:
            raise ValueError("empty face")
        BW = sp.csr_matrix(np.asarray(B, dtype=float)[idx])
        BWT = sp.csr_matrix(BW.T)
        bW = np.asarray(b, dtype=float)[idx]
        dv = np.asarray(d, dtype=float)
        H = np.asfortranarray((BWT @ BW).toarray())
        rhs = np.empty((m, 2))
        rhs[:, 0] = dv
        rhs[:, 1] = BWT @ bW
        delta, _r_null, rank, clean = wfs.piv_chol_step(H, rhs, m)
        uc = np.ascontiguousarray(delta[:, 0])
        ua = -np.ascontiguousarray(delta[:, 1])
        h = BW @ uc
        # A1 holds by CONSTRUCTION: g is literally b_W + B_W ua, and a second
        # application of the same H^+ provably cannot move it.
        g = bW + BW @ ua
        dres = float(np.linalg.norm(BWT @ h - dv)) / max(
            1.0, float(np.linalg.norm(dv)))
    except (ValueError, MemoryError, ArithmeticError,
            np.linalg.LinAlgError):
        PCHOL_STATS["pchol_seconds"] += time.perf_counter() - started
        PCHOL_STATS["pchol_decline_error"] += 1
        return None
    # ---- FAIL-CLOSED guards, in increasing cost -------------------------
    PCHOL_STATS["pchol_worst_clean"] = (
        clean if PCHOL_STATS["pchol_worst_clean"] == 0.0
        else min(PCHOL_STATS["pchol_worst_clean"], clean))
    PCHOL_STATS["pchol_deficiency"] = max(PCHOL_STATS["pchol_deficiency"],
                                          int(m - rank))
    verdict = None
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(h))
            and np.all(np.isfinite(ua)) and np.all(np.isfinite(uc))
            and np.isfinite(dres)):
        verdict = "pchol_decline_error"
    elif not (float(clean) >= PCHOL_MIN_CLEAN):
        verdict = "pchol_decline_clean"
    if verdict is None:
        bnorm = max(float(np.abs(BW.data).max()) if BW.nnz else 1.0, 1e-300)
        bwscale = max(1.0, float(np.abs(bW).max()) if bW.size else 1.0)
        ortho = float(np.abs(BWT @ g).max()) / (bnorm * bwscale)
        PCHOL_STATS["pchol_worst_ortho"] = max(
            PCHOL_STATS["pchol_worst_ortho"], ortho)
        g_rel = (float(np.abs(g).max()) if g.size else 0.0) / bwscale
        if ortho > TIK_ORTHO_TOL:
            verdict = "pchol_decline_ortho"
        elif dres > PCHOL_DRES_MARGIN * TOL_DRES:
            verdict = "pchol_decline_margin"
        elif (g_rel > TOL_G / PCHOL_G_BAND
              and g_rel < TOL_G * PCHOL_G_BAND):
            verdict = "pchol_decline_margin"
    PCHOL_STATS["pchol_seconds"] += time.perf_counter() - started
    if verdict is not None:
        PCHOL_STATS[verdict] += 1
        return None
    PCHOL_STATS["pchol_piece"] += 1
    return g, h, ua, uc, dres


def _piece_y_routed(B, b, d, W):
    """``(g, h, ua, uc, dres, fac, route)``; ``fac`` is ``None`` on the
    incumbent, ``route`` names which solver produced the answer.

    Route order: S1 Tikhonov-Woodbury (only when a classifier is installed,
    i.e. on a model above the unit-row-fraction gate), then S2 pivoted
    Cholesky (only for ``m <= PCHOL_MAXM``), then the incumbent dense SVD.
    The two gates are disjoint on the shipped panel -- the S1 models are
    fit1d (m=1026) and ship04s (m=1458), both above the S2 size guard -- so
    in practice each face sees exactly one replacement route before the
    incumbent.  ``fac`` is returned only by S1; the S2 route returns ``None``
    so ``x_candidates`` / ``y_candidates`` run their incumbent ``lstsq``
    branch line-for-line.
    """
    out = _tik_piece_y(B, b, d, W)
    if out is not None:
        return (out[0], out[1], out[2], out[3], out[4],
                (out[5], out[6], out[7]), "tik")
    out = _pchol_piece_y(B, b, d, W)
    if out is not None:
        return out[0], out[1], out[2], out[3], out[4], None, "pchol"
    return _dense_piece_y(B, b, d, W)


def _dense_piece_y(B, b, d, W):
    started = time.perf_counter()
    g, h, ua, uc, dres = piece_y(B, b, d, W)
    TIK_STATS["svd_piece"] += 1
    TIK_STATS["svd_seconds"] += time.perf_counter() - started
    return g, h, ua, uc, dres, None, "dense"


# --------------------------------------------------------------- oracle solve

def oracle_solve(B, b, d, methods=("highs-ipm", "highs"), backend="scipy",
                 highspy_kwargs=None):
    """Run HiGHS on the primal and extract an approximate optimal pair.

    SIGN CONVENTION (verified empirically on afiro and grow7, not from
    memory): with ``A_ub = -B`` and ``b_ub = -b``, scipy returns ``res.x``
    as the primal ``x~`` directly, and the dual of ``Bx >= b`` is

        y~ = -res.ineqlin.marginals

    With the ``+`` sign the residual ``|B'y - d|_inf`` was 2.0 and
    ``min y`` was -10 on afiro; with the ``-`` sign the residual was 1.8e-17
    and ``min y`` was 0.  Same module-level convention as
    ``exp23_path_primal_dual._highs_fallback`` (line 430).

    ``methods[0]`` is the oracle actually consumed downstream; the remaining
    methods are recorded for comparison only.

    ``backend`` selects the underlying solve.  Default ``"scipy"`` below is
    UNCHANGED from the original implementation -- byte-identical code path,
    same default argument.  ``backend="highspy_ipm_nocrossover"`` instead
    calls ``_oracle_solve_highspy`` (highspy Python bindings directly,
    ``solver='ipm'``, ``run_crossover='off'``): a genuinely crossover-free
    interior-point oracle, which scipy's ``highs-ipm`` cannot produce (it
    silently forces ``run_crossover=True`` -- see
    ``records/late_obstruction_captures/oracle_terminal_report.md``
    section 6).  ``highspy_kwargs`` forwards tolerance / iteration-limit
    ablation knobs to that backend only; ignored for ``backend="scipy"``.
    """
    if backend == "highspy_ipm_nocrossover":
        return _oracle_solve_highspy(B, b, d, **(highspy_kwargs or {}))
    if backend != "scipy":
        raise ValueError(f"unknown oracle backend: {backend!r}")
    m = B.shape[1]
    records = {}
    primary = None
    for method in methods:
        started = time.perf_counter()
        try:
            res = linprog(d, A_ub=-B, b_ub=-b, bounds=[(None, None)] * m,
                          method=method)
        except Exception as error:            # pragma: no cover - defensive
            records[method] = {"ok": False, "error": type(error).__name__,
                               "seconds": time.perf_counter() - started}
            continue
        wall = time.perf_counter() - started
        if not res.success:
            records[method] = {"ok": False, "status": int(res.status),
                               "seconds": wall}
            continue
        x = np.asarray(res.x, dtype=float)
        y = -np.asarray(res.ineqlin.marginals, dtype=float)
        yscale = max(1.0, float(np.abs(y).max()) if y.size else 1.0)
        record = {
            "ok": True,
            "seconds": wall,
            "objective": float(res.fun),
            "dual_residual": (float(np.max(np.abs(B.T @ y - d)))
                              / max(1.0, float(np.abs(d).max()))),
            "min_y": float(y.min()) if y.size else 0.0,
            "nnz_1e6": int((y > 1e-6 * yscale).sum()),
            "nnz_1e9": int((y > 1e-9 * yscale).sum()),
            "gap": abs(float(d @ x) - float(b @ y))
                   / max(1.0, abs(float(d @ x))),
        }
        records[method] = record
        if primary is None:
            primary = {"x": x, "y": y, "method": method, "seconds": wall,
                       "objective": float(res.fun)}
    return primary, records


def _oracle_solve_highspy(B, b, d, run_crossover="off",
                          ipm_optimality_tolerance=None,
                          ipm_iteration_limit=None, presolve=None,
                          time_limit=None):
    """Genuinely CROSSOVER-FREE interior-point oracle via highspy directly.

    ``scipy.optimize.linprog(method='highs-ipm')`` cannot do this: HiGHS
    rejects ``run_crossover='off'`` through scipy with *"only True or False
    is allowed. Using default: True"*, so the ``oracle_solve`` scipy path
    above is always IPM+crossover, i.e. a vertex.  Here the model is built
    directly against the highspy bindings (``highspy.HighsLp`` /
    ``highspy.Highs``), which DOES honor ``run_crossover='off'`` -- verified
    empirically below, not assumed.

    Model: ``min d'x s.t. Bx >= b``, ``x`` free (``col_lower_ = -inf``,
    ``col_upper_ = +inf``), rows ``row_lower_ = b``, ``row_upper_ = +inf``,
    column-wise CSC of ``B`` as the constraint matrix -- the same ``B, b, d``
    convention as ``oracle_solve``'s scipy path (``A_ub = -B, b_ub = -b`` is
    scipy's own internal transform of the identical constraint ``Bx >= b``).

    SIGN CONVENTION -- verified empirically on afiro, lotfi, grow7, brandy,
    degen2 (``xfree_probe1.py``, NOT copied from the scipy convention above,
    which needs a different sign because it solves scipy's differently-posed
    ``A_ub``/``b_ub`` problem).  With rows built as ``row_lower_=b`` directly
    (no negation of ``B`` or ``b``), HiGHS's own ``solution.row_dual``
    satisfies, WITHOUT negation:

        y = solution.row_dual ;  B'y ~ d ;  y >= 0 (up to ~1e-13) ;
        b'y ~ d'x

    Measured: afiro resid=6.44e-12 min_y=-9.0e-14 b'y=d'x=-464.753143;
    lotfi resid=2.75e-12 min_y=-1.4e-14 b'y=d'x=-25.264706; grow7
    resid=1.38e-08 min_y=0.0 b'y=d'x (matches KNOWN_LB to the reported
    digits); brandy resid=2.13e-09 min_y=-5.0e-13 b'y=d'x=1518.509897
    (matches KNOWN_LB).  The negated sign gives resid~2.0, min_y~-10 --
    wrong, mirroring the scipy check in ``oracle_solve.__doc__``.

    CROSSOVER-REALLY-OFF EVIDENCE.  ``info.crossover_iteration_count`` is
    read back and recorded (0 whenever ``run_crossover='off'`` was honored,
    measured on every model probed) rather than trusted from the option
    string alone.  ``ipm_iteration_count`` is also recorded.

    ``ipm_optimality_tolerance`` GENUINELY BITES here (unlike scipy, where
    it is silently ignored -- oracle_terminal_report.md section 6): on
    afiro, tol in {1e-1 .. 1e-5} all converge to the SAME 6-iteration,
    9.0e-9-residual point, while tol in {1e-6 .. 1e-10} converge to a
    DIFFERENT, tighter 7-iteration, 6.4e-12-residual point -- the achieved
    objective and residual visibly change at the 1e-5/1e-6 boundary.
    ``ipm_iteration_limit`` also bites (verified: forces "Iteration limit
    reached" status with the reported iterate at exactly the requested
    count) and is exposed as a second ablation knob for models where the
    tolerance option's floor is reached before the campaign's loosest
    interesting point.

    Returns ``(primary, records)`` in the same shape as ``oracle_solve`` so
    it plugs into the identical downstream pipeline unmodified.
    """
    import highspy
    from scipy.sparse import csc_matrix

    n, m = B.shape
    method = "highspy_ipm_nocrossover"
    matrix = csc_matrix(np.asarray(B, dtype=float))
    lp = highspy.HighsLp()
    lp.num_col_ = m
    lp.num_row_ = n
    lp.col_cost_ = np.asarray(d, dtype=float)
    lp.col_lower_ = np.full(m, -highspy.kHighsInf)
    lp.col_upper_ = np.full(m, highspy.kHighsInf)
    lp.row_lower_ = np.asarray(b, dtype=float)
    lp.row_upper_ = np.full(n, highspy.kHighsInf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
    lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
    lp.a_matrix_.value_ = matrix.data.astype(float)
    lp.sense_ = highspy.ObjSense.kMinimize

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", run_crossover)
    if ipm_optimality_tolerance is not None:
        h.setOptionValue("ipm_optimality_tolerance",
                         float(ipm_optimality_tolerance))
    if ipm_iteration_limit is not None:
        h.setOptionValue("ipm_iteration_limit", int(ipm_iteration_limit))
    if presolve is not None:
        h.setOptionValue("presolve", presolve)
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    h.passModel(lp)

    started = time.perf_counter()
    h.run()
    wall = time.perf_counter() - started
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    x = np.asarray(sol.col_value, dtype=float)
    y = np.asarray(sol.row_dual, dtype=float)   # NOT negated -- see docstring

    records = {}
    ok = (status == "Optimal") and bool(x.size) and bool(y.size)
    if not ok:
        records[method] = {"ok": False, "status": status, "seconds": wall,
                           "crossover_iteration_count": int(
                               getattr(info, "crossover_iteration_count", -1)),
                           "ipm_iteration_count": int(
                               getattr(info, "ipm_iteration_count", -1))}
        return None, records

    yscale = max(1.0, float(np.abs(y).max()) if y.size else 1.0)
    record = {
        "ok": True,
        "seconds": wall,
        "objective": float(info.objective_function_value),
        "dual_residual": (float(np.max(np.abs(B.T @ y - d)))
                          / max(1.0, float(np.abs(d).max()))),
        "min_y": float(y.min()) if y.size else 0.0,
        "nnz_1e6": int((y > 1e-6 * yscale).sum()),
        "nnz_1e9": int((y > 1e-9 * yscale).sum()),
        "gap": abs(float(d @ x) - float(b @ y))
               / max(1.0, abs(float(d @ x))),
        # crossover-really-off EVIDENCE -- read back, not trusted from the
        # option string alone.
        "run_crossover_option": run_crossover,
        "crossover_iteration_count": int(
            getattr(info, "crossover_iteration_count", -1)),
        "ipm_iteration_count": int(getattr(info, "ipm_iteration_count", -1)),
        "max_complementarity_violation": float(
            getattr(info, "max_complementarity_violation", float("nan"))),
        "ipm_optimality_tolerance": ipm_optimality_tolerance,
        "ipm_iteration_limit": ipm_iteration_limit,
        "model_status": status,
    }
    records[method] = record
    primary = {"x": x, "y": y, "method": method, "seconds": wall,
              "objective": float(info.objective_function_value)}
    return primary, records


# ------------------------------------------------------------ terminal algebra

def _t_bar(B, W, uc, sigma, b_scale, tol_sigma=TOL_SIGMA):
    """Closed-form t-bar for face ``W`` given an off-support slack ``sigma``.

    Returns ``(t_bar, blocking_rows)``.  ``blocking_rows`` counts the
    self-diagnosing rows with ``sigma_j <= 0 < (B uc)_j``: no finite ``t``
    keeps them off the face, so they belong IN it.
    """
    Buc = B @ uc
    off = ~W
    positive = off & (Buc > 0.0)
    feasible = positive & (sigma > tol_sigma * b_scale)
    blocking = int((positive & (sigma <= tol_sigma * b_scale)).sum())
    if feasible.any():
        return float((Buc[feasible] / sigma[feasible]).max()), blocking
    return 0.0, blocking


def y_candidates(B, d, W, h, y_tilde=None, fac=None):
    """Candidate duals on the affine set ``{y : supp(y) <= W, B_W' y = d}``.

    Exactly the dual mirror of ``x_candidates``.  When the oracle face ``W`` is
    LARGER than a vertex support -- which is what a crossover-free interior
    solver returns, since its limit is the analytic center of the optimal dual
    face -- that affine set has positive dimension and exp23's min-norm
    selector ``h`` need not be nonnegative.  Measured on grow7 with a clarabel
    oracle at tol 1e-10: the face is exactly terminal (``g_rel = 1.1e-15``,
    ``sigma_rel = -2.1e-16``) yet ``min h = -42.8``, so the frozen gate
    rejects on ``nonnegative``.  The second candidate projects the oracle dual
    onto the same affine set,

        y = y~_W - B_W^{+T} (B_W' y~_W - d),

    which is nonnegative whenever ``y~`` was, up to the oracle's own dual
    residual.  Any member of this set is an exact optimal dual: it is
    complementary with any ``x`` having ``B_W x = b_W``, so
    ``d'x = y'Bx = b'y`` is an identity.  The min-norm member is additionally
    the Mangasarian path limit; the projected member is not, and the accepted
    selector is recorded so the distinction stays visible.
    """
    yield "min-norm", np.asarray(h, dtype=float)
    if y_tilde is None:
        return
    if fac is not None:
        # S1 reuse.  ``(B_W')^+ r = B_W H^+ r`` is the SAME min-norm
        # least-squares solution ``lstsq(B_W', r, rcond=None)`` computes, from
        # the factorization ``terminal_test`` already built for ``piece_y``:
        # no second decomposition of B_W.  ``fac`` is None unless the S1 route
        # produced this face, so the default path is the lstsq below.
        face, BW, BWT = fac
        residual = BWT @ y_tilde[W] - d
        correction = BW @ face.pinv(residual, refine=wfs.TIK_REFINE)[0]
        TIK_STATS["cand_reuse"] += 1
        yield "oracle-projected", np.asarray(y_tilde[W] - correction,
                                             dtype=float)
        return
    BWt = B[W].T
    residual = BWt @ y_tilde[W] - d
    correction, _r, _rank, _sv = np.linalg.lstsq(BWt, residual, rcond=None)
    yield "oracle-projected", np.asarray(y_tilde[W] - correction, dtype=float)


def x_candidates(B, b, W, ua, x_tilde=None, fac=None):
    """Candidate primal points on the affine set ``{x : B_W x = b_W}``.

    When ``rank(B_W) < m`` -- routine on real sparse LPs -- that set is an
    affine FAMILY, and exp23's min-norm selector ``-ua`` is only one member.
    On grow7 the min-norm member violates other rows by 1.3e5 while the face
    itself is exactly terminal (``|g|_inf = 5e-24``), so a second candidate is
    required: the oracle primal ``x~`` projected orthogonally onto the same
    affine set,

        x = x~ - B_W^+ (B_W x~ - b_W),

    which keeps ``B_W x = b_W`` exactly and stays near a point already known
    to be primal feasible.  Both are only CANDIDATES; the frozen gates decide.
    """
    yield "min-norm", -np.asarray(ua, dtype=float)
    if x_tilde is None:
        return
    if fac is not None:
        # S1 reuse: ``B_W^+ r = H^+ B_W' r`` -- same min-norm least-squares
        # solution as ``lstsq(B_W, r, rcond=None)``, from the existing
        # factorization.  ``fac`` is None on the incumbent path.
        face, BW, BWT = fac
        residual = BW @ x_tilde - b[face.idx]
        correction = face.pinv(BWT @ residual, refine=wfs.TIK_REFINE)[0]
        TIK_STATS["cand_reuse"] += 1
        yield "oracle-projected", np.asarray(x_tilde - correction,
                                             dtype=float)
        return
    BW, bW = B[W], b[W]
    residual = BW @ x_tilde - bW
    correction, _res, _rank, _sv = np.linalg.lstsq(BW, residual, rcond=None)
    yield "oracle-projected", np.asarray(x_tilde - correction, dtype=float)


def terminal_test(B, b, d, W, x_tilde=None, y_tilde=None, tol_g=TOL_G,
                  tol_sigma=TOL_SIGMA, tol_dres=TOL_DRES):
    """t-free terminal test on face ``W``.  Returns (passed, info-dict).

    Stage 1 is the face test proper: ``dres`` admissible and ``g == 0``.  That
    is what makes the piece t-free -- ``y_W(t) = h`` for every ``t``.
    Stage 2 searches the affine family ``{x : B_W x = b_W}`` for a primal
    feasible member (``sigma = Bx - b >= 0``).  ``info`` is populated even on
    failure, including ``t_bar``/``t_hat`` and ``blocking_rows``.
    """
    W = np.asarray(W, dtype=bool)
    info = {"support": int(W.sum())}
    if not W.any():
        info["reason"] = "empty support"
        return False, info
    g, h, ua, uc, dres, fac, route = _piece_y_routed(B, b, d, W)
    b_scale = max(1.0, float(np.abs(b).max()))
    g_scale = max(1.0, float(np.abs(b[W]).max()))
    g_err = float(np.abs(g).max()) if g.size else 0.0

    # ---- CONFIRM-ON-PASS (S2 only) -------------------------------------
    # A face the piv-chol route calls TERMINAL is a certificate candidate, and
    # its ``h``/``ua`` are what the frozen gate is about to be handed.  Those
    # two quantities are formed through ``H^+`` and so carry the GRAM matrix's
    # conditioning, the square of the face's -- fine for a schedule decision,
    # measurably not free for a certificate (worst ``lb`` disagreement against
    # the independent cross-check moved 6.3e-11 -> 2.7e-10 when the route's
    # own numbers were used).  So the moment the route says "terminal", the
    # face is re-solved by the INCUMBENT and every number from here on is the
    # incumbent's.  This cannot flip the verdict: ``g`` and ``dres`` can only
    # be OVER-stated by the route, so a face that passed on the route's
    # numbers passes on the incumbent's a fortiori.  Net effect: the S2 route
    # can change WHICH face is tested at WHICH t, and nothing else -- every
    # quantity a certificate is built from is computed exactly as before.
    # Cost is one dense ``piece_y`` per face that reaches stage 1 (typically
    # one or two per model, i.e. what the incumbent paid anyway on that face).
    if (route == "pchol" and dres <= tol_dres
            and g_err <= tol_g * g_scale):
        g, h, ua, uc, dres, fac, route = _dense_piece_y(B, b, d, W)
        g_err = float(np.abs(g).max()) if g.size else 0.0
        PCHOL_STATS["pchol_confirm"] += 1
        info["pchol_confirmed"] = True

    info.update(dres=float(dres), g_inf=g_err, g_relative=g_err / g_scale,
                h_min=float(h.min()) if h.size else 0.0,
                piece_route=route)

    y_star = np.zeros(B.shape[0])
    y_star[W] = h
    info["y_star"] = y_star
    info["x_star"] = -ua

    # walker-relevant t_bar uses exp23's min-norm selector, as the walk does
    sigma_min_norm = B @ (-ua) - b
    info["sigma_min_norm"] = float(sigma_min_norm.min())
    t_bar, blocking = _t_bar(B, W, uc, sigma_min_norm, b_scale, tol_sigma)
    info["t_bar"] = t_bar
    info["blocking_rows"] = blocking
    info["t_hat"] = SAFETY_FACTOR * max(t_bar, 1.0)

    if dres > tol_dres:
        info["reason"] = "dual equality"
        return False, info
    if g_err > tol_g * g_scale:
        info["reason"] = "g nonzero"
        return False, info
    info["face_terminal"] = True          # stage 1 passed: piece is t-free

    # stage 2a: pick a NONNEGATIVE member of the dual affine family
    y_scale_all = max(1.0, float(np.abs(d).max()))
    best_y = None
    for label, y_face in y_candidates(B, d, W, h, y_tilde, fac=fac):
        y_min = float(y_face.min()) if y_face.size else 0.0
        y_res = float(np.max(np.abs(B[W].T @ y_face - d))) / y_scale_all
        info.setdefault("dual_candidates", []).append(
            {"candidate": label, "y_min": y_min, "dual_residual": y_res})
        if best_y is None or y_min > best_y[1]:
            best_y = (label, y_min, y_face)
        if y_min >= -tol_sigma and y_res <= tol_dres:
            best_y = (label, y_min, y_face)
            break
    if best_y is not None:
        y_star = np.zeros(B.shape[0])
        y_star[W] = best_y[2]
        info["y_star"] = y_star
        info["dual_candidate"] = best_y[0]
        info["y_min"] = best_y[1]

    # stage 2b: pick a PRIMAL FEASIBLE member of the primal affine family
    best = None
    for label, x in x_candidates(B, b, W, ua, x_tilde, fac=fac):
        sigma = B @ x - b
        sigma_min = float(sigma.min()) if sigma.size else 0.0
        info.setdefault("candidates", []).append(
            {"candidate": label, "sigma_min": sigma_min,
             "sigma_min_relative": sigma_min / b_scale})
        if best is None or sigma_min > best[1]:
            best = (label, sigma_min, x, sigma)
        if sigma_min >= -tol_sigma * b_scale:
            info["x_star"] = x
            info["candidate"] = label
            info["sigma_min"] = sigma_min
            info["sigma_min_relative"] = sigma_min / b_scale
            cand_t_bar, cand_blocking = _t_bar(B, W, uc, sigma, b_scale,
                                               tol_sigma)
            info["t_bar_candidate"] = cand_t_bar
            info["blocking_rows_candidate"] = cand_blocking
            info["reason"] = "terminal"
            return True, info
    if best is not None:
        info["x_star"] = best[2]
        info["candidate"] = best[0]
        info["sigma_min"] = best[1]
        info["sigma_min_relative"] = best[1] / b_scale
    info["reason"] = "primal violation"
    return False, info


def _gate(B, b, d, info):
    """FROZEN gates, called unchanged on the original ``B, b, d``.

    ``certificate_pair(B, b, d, x, y)`` is the decision.  If our candidate
    ``x`` fails, ``_certificate_detail(B, b, d, y)`` is given the chance to
    build its own ``x`` from the dual support (it does an lstsq plus, on rank
    deficiency, a null-space ``newton4`` violation-penalty repair, exp23:257).
    ``_certificate_detail`` calls ``certificate_pair`` internally before
    returning True, so accepting its ``x`` is not a weaker test.

    S4 SHORT-CIRCUIT.  ``_certificate_detail`` is called ONLY when
    ``certificate_pair`` failed, i.e. only when its ``x`` can still change the
    decision.  When the pair already passed, the old code called it and then
    returned on the very next line without ever consulting ``det_x`` -- the
    result was built and discarded.  Neither frozen gate is modified; one of
    them is simply not called when its answer cannot be used.  The returned
    boolean, ``info["x_star"]`` and ``info["y_star"]`` are unchanged on both
    branches (``_certificate_detail`` is a pure function of its arguments: it
    mutates nothing and draws no global RNG), so the decision is identical.
    The three diagnostic keys stay present and are marked as skipped.
    """
    y = info["y_star"]
    passed, detail = certificate_pair(B, b, d, info["x_star"], y)
    detail = dict(detail)
    detail["gate_x"] = info.get("candidate", "min-norm")
    if passed:
        detail["certificate_detail_skipped"] = True
        detail["certificate_detail_ok"] = None
        detail["certificate_detail_error"] = None
        detail["certificate_detail_reason"] = "skipped (pair passed)"
        return True, detail
    try:
        det_ok, det_x, det_err, det_reason = exp23._certificate_detail(
            B, b, d, y)
    except Exception as error:                # pragma: no cover - defensive
        det_ok, det_x = False, None
        det_err, det_reason = float("inf"), type(error).__name__
    detail["certificate_detail_skipped"] = False
    detail["certificate_detail_ok"] = bool(det_ok)
    detail["certificate_detail_error"] = float(det_err)
    detail["certificate_detail_reason"] = str(det_reason)
    if det_ok and det_x is not None:
        info["x_star"] = np.asarray(det_x, dtype=float)
        detail["gate_x"] = "certificate-detail lstsq/newton4"
        detail["reason"] = "passed"
        return True, detail
    return False, detail


# ---------------------------------------------------------------- the attempt

def terminal_attempt(B, b, d, oracle=None, thresholds=THRESHOLDS,
                     seconds=ATTEMPT_SECONDS, projection_retry=True,
                     settle_retry=True, diagnostics=None,
                     oracle_backend="scipy", oracle_highspy_kwargs=None):
    """Try to jump straight to the terminal face.  Fails closed, always.

    Returns ``(result_or_None, diag)``.  ``result_or_None`` is a certified
    ``(x, y)`` pair dict only when a frozen gate passed.  ``oracle_backend``/
    ``oracle_highspy_kwargs`` are forwarded to ``oracle_solve`` only when
    ``oracle`` is not already supplied; default ``"scipy"`` is unchanged.
    """
    diag = {} if diagnostics is None else diagnostics
    deadline = time.perf_counter() + seconds
    diag.setdefault("attempts", [])
    diag["route"] = None

    if oracle is None:
        oracle, records = oracle_solve(B, b, d, backend=oracle_backend,
                                       highspy_kwargs=oracle_highspy_kwargs)
        diag["oracle_methods"] = records
    if oracle is None:
        diag["reason"] = "oracle failed"
        return None, diag

    y_tilde = oracle["y"]
    x_tilde = oracle["x"]
    yscale = max(1.0, float(np.abs(y_tilde).max()) if y_tilde.size else 1.0)
    diag["oracle_method"] = oracle.get("method")
    diag["oracle_objective"] = oracle.get("objective")
    sigma_tilde = B @ x_tilde - b
    diag["oracle_sigma_min"] = float(sigma_tilde.min())

    seen = set()
    best_t_hat = 0.0

    def _try(W, route, label):
        nonlocal best_t_hat
        key = np.packbits(W).tobytes()
        if key in seen:
            return None
        seen.add(key)
        passed, info = terminal_test(B, b, d, W, x_tilde=x_tilde,
                                     y_tilde=y_tilde)
        best_t_hat = max(best_t_hat, float(info.get("t_hat", 0.0)))
        record = {k: v for k, v in info.items()
                  if k not in ("x_star", "y_star")}
        record["label"] = label
        record["route"] = route
        record["terminal"] = bool(passed)
        # Stage 1 (t-free face) can pass while every candidate x violates.
        # Give the frozen gate its own shot at building x before giving up.
        if not passed and not info.get("face_terminal"):
            diag["attempts"].append(record)
            return None
        gate_ok, detail = _gate(B, b, d, info)
        record["gate"] = detail
        diag["attempts"].append(record)
        if not gate_ok:
            return None
        diag["route"] = route
        diag["accepted_support"] = int(W.sum())
        diag["accepted_mask"] = W
        diag["t_hat"] = float(info["t_hat"])
        diag["t_bar"] = float(info["t_bar"])
        return {"x": info["x_star"], "y": info["y_star"],
                "certificate": detail, "support": W}

    # ---- route 1: direct threshold sweep on the oracle dual -----------------
    for threshold in thresholds:
        if time.perf_counter() > deadline:
            diag["reason"] = "attempt deadline"
            return None, diag
        W = y_tilde > threshold * yscale
        found = _try(W, "oracle threshold", f"thr={threshold:g}")
        if found is not None:
            return found, diag

    # ---- route 2: settle the best face at t_hat -----------------------------
    if settle_retry:
        t_hat = max(best_t_hat, 1.0)
        diag["settle_t_hat"] = float(t_hat)
        for threshold in (1e-6, 1e-8):
            if time.perf_counter() > deadline:
                diag["reason"] = "attempt deadline"
                return None, diag
            W0 = y_tilde > threshold * yscale
            if not W0.any():
                continue
            try:
                W1, _coeff, _ok = settle(B, b, d, W0, t_hat, rounds=20)
            except Exception as error:
                diag["attempts"].append({"route": "settle",
                                         "label": f"thr={threshold:g}",
                                         "reason": type(error).__name__})
                continue
            found = _try(W1, "oracle settle", f"settle thr={threshold:g}")
            if found is not None:
                return found, diag

    # ---- route 3: one tight projection of t_hat*b, then threshold ITS support
    if projection_retry and time.perf_counter() < deadline:
        t_hat = max(best_t_hat, 1.0)
        started = time.perf_counter()
        try:
            from dual_ascent_repair import _solve_projection
            y_proj, _u, status = _solve_projection(B, t_hat * b, d, tol=1e-11)
            diag["projection_status"] = str(status)
            diag["projection_seconds"] = time.perf_counter() - started
            pscale = max(1.0, float(np.abs(y_proj).max()))
            for threshold in (1e-6, 1e-8, 1e-10):
                if time.perf_counter() > deadline:
                    break
                W = y_proj > threshold * pscale
                found = _try(W, "oracle projection", f"proj thr={threshold:g}")
                if found is not None:
                    return found, diag
        except Exception as error:
            diag["projection_status"] = f"EXCEPTION {type(error).__name__}"
            diag["projection_seconds"] = time.perf_counter() - started

    diag.setdefault("reason", "no terminal face certified")
    diag["t_hat"] = float(best_t_hat)
    return None, diag


# ------------------------------------------------------------------- pipeline

def follow_with_oracle(B, b, d, oracle_seconds=ATTEMPT_SECONDS,
                       oracle_thresholds=THRESHOLDS,
                       oracle_projection_retry=True,
                       oracle_settle_retry=True,
                       oracle_diagnostics=None,
                       oracle_backend="scipy", oracle_highspy_kwargs=None,
                       **kw):
    """Oracle terminal shortcut with the UNCHANGED walker as the fallback.

    On success returns a ``follow``-shaped dict with ``pivots=0`` and
    ``backend="oracle terminal pair"``.  On ANY failure -- oracle failure,
    terminal test failure, gate failure, exception, deadline -- returns
    ``exp23_path_primal_dual.follow(B, b, d, **kw)`` verbatim, with the oracle
    counters attached so the wasted time is visible.  Warm-starting the walk
    from the oracle is deliberately out of scope.

    ``oracle_backend`` (default ``"scipy"``, unchanged) / ``oracle_highspy_kwargs``
    select and configure the oracle solve -- see ``oracle_solve`` and
    ``_oracle_solve_highspy``.
    """
    diag = {} if oracle_diagnostics is None else oracle_diagnostics
    ipm_started = time.perf_counter()
    try:
        oracle, records = oracle_solve(B, b, d, backend=oracle_backend,
                                       highspy_kwargs=oracle_highspy_kwargs)
        diag["oracle_methods"] = records
    except Exception as error:                # pragma: no cover - defensive
        oracle, diag["oracle_methods"] = None, {
            "error": type(error).__name__}
    ipm_seconds = time.perf_counter() - ipm_started
    primary = diag.get("oracle_methods", {}).get(
        oracle["method"] if oracle else "", {})
    diag["oracle_ipm_seconds"] = float(primary.get("seconds", ipm_seconds))
    diag["oracle_total_solve_seconds"] = float(ipm_seconds)

    attempt_started = time.perf_counter()
    found = None
    try:
        found, diag = terminal_attempt(
            B, b, d, oracle=oracle, thresholds=oracle_thresholds,
            seconds=oracle_seconds,
            projection_retry=oracle_projection_retry,
            settle_retry=oracle_settle_retry, diagnostics=diag,
            oracle_backend=oracle_backend,
            oracle_highspy_kwargs=oracle_highspy_kwargs)
    except Exception as error:                # fail closed on ANY exception
        diag["reason"] = f"EXCEPTION {type(error).__name__}: {error}"
    attempt_seconds = time.perf_counter() - attempt_started
    diag["oracle_attempt_seconds"] = float(attempt_seconds)
    diag["thresholds_tried"] = len(diag.get("attempts", []))

    counters = {
        "oracle_ipm_seconds": diag["oracle_ipm_seconds"],
        "oracle_total_solve_seconds": diag["oracle_total_solve_seconds"],
        "oracle_attempt_seconds": diag["oracle_attempt_seconds"],
        "oracle_overhead_seconds": (diag["oracle_total_solve_seconds"]
                                    + diag["oracle_attempt_seconds"]),
        "thresholds_tried": diag["thresholds_tried"],
        "route": diag.get("route"),
        "oracle_diagnostics": diag,
    }

    if found is not None:
        x = found["x"]
        y = found["y"]
        st = {
            "status": "CERTIFIED",
            "x": x,
            "y": y,
            "lb": float(b @ y),
            "t": float("inf"),
            "pivots": 0,
            "backend": "oracle terminal pair",
            "mono": True,
            "certificate_detail": found["certificate"],
            "oracle_fallback": False,
        }
        st.update(counters)
        return st

    counters["route"] = "fallback walk"
    walk_started = time.perf_counter()
    st = exp23.follow(B, b, d, **kw)
    counters["oracle_fallback_walk_seconds"] = time.perf_counter() - walk_started
    st = dict(st)
    st["oracle_fallback"] = True
    st["oracle_fallback_reason"] = diag.get("reason", "unknown")
    st.update(counters)
    return st


# ------------------------------------------------------------------ utilities

def support_agreement(y_reference, mask, tol=exp23.TOL_Y):
    """Precision/recall of a candidate mask against a reference dual support."""
    y_reference = np.asarray(y_reference, dtype=float)
    scale = max(1.0, float(np.abs(y_reference).max())
                if y_reference.size else 1.0)
    reference = y_reference > tol * scale
    mask = np.asarray(mask, dtype=bool)
    inter = int((mask & reference).sum())
    return {
        "candidate": int(mask.sum()),
        "reference": int(reference.sum()),
        "intersection": inter,
        "precision": (inter / int(mask.sum())) if mask.any() else 0.0,
        "recall": (inter / int(reference.sum())) if reference.any() else 0.0,
    }
