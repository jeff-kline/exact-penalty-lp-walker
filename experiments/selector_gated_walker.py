"""Selector-gated critical-cone walker for the Mangasarian dual path.

Reordered event machinery (see agent_reports/11): the forward-affine
multiplier LP is the ONLY face-acceptance gate, the exact critical-cone right
derivative (cached equality reduction, endpoint hygiene) is the primary
degenerate-transition oracle, and the historical settle / priced
combinatorial repair is demoted to an optional last resort.  Every
original-data certificate is reused unchanged from exp23; a run can return
CERTIFIED only through those gates.
"""

from __future__ import annotations

import time

import numpy as np

from critical_cone_path import _prepare, critical_derivative
from dual_ascent_repair import _solve_projection, repair_face
from exp23_path_primal_dual import (FWD, TIE, TOL_FEAS, TOL_Y,
                                    _certificate_detail, _emit_trace,
                                    certificate_pair, piece_y,
                                    qp_corrector, settle)
from face_factor_updating import (FactorUpdatingFaceSolver,
                                  FactorUpdatingFallback)
from path_piece_factor import PathPieceConditionError
from path_piece_skeleton import PathPieceSkeletonFactor
from rz_core_face_solver import PathPieceRZFactor
from sparse_face_solver import SparseFaceDecline, SparseFaceSolver
from unit_core_face_solver import UnitCoreFaceSolver, UnitCoreFallback
from primal_face_pricing import (priced_settle_escape,
                                 select_forward_affine_multiplier)
from sparse_b_support import SparseB


# --------------------------------------------------------------------- S5 --
# W-CERT: do not recompute a least-squares solution the walker already holds.
#
# At a crawl face the walker calls ``certificate_pair(B, b, d, -ua, h)`` and,
# when that fails, ``exp23._certificate_detail(B, b, d, y)``.  The FIRST act of
# ``_certificate_detail`` is
#
#     x, _, rank, _ = np.linalg.lstsq(B[A], b[A], rcond=None)
#
# a GELSD SVD of a ~1030x1026 (fit1d) / ~1500x1458 (ship04s) matrix, measured
# at 0.178 s / 0.935 s per call on an idle machine.
#
#   PROPOSITION.  If ``A == W`` then ``x = -ua`` exactly.
#   PROOF.  ``piece_y`` sets ``g = (I - U_r U_r') b_W`` and returns ``ua`` as
#   the min-norm solution of ``B_W ua = g - b_W``.  Hence
#   ``B_W(-ua) - b_W = -g`` with ``g`` orthogonal to ``range(B_W)``, so
#   ``-ua`` is a least-squares solution of ``B_W x ~= b_W``, and it is
#   min-norm because ``ua`` is built as a pseudo-inverse image and lies in
#   ``range(B_W')``.  ``numpy.linalg.lstsq`` returns the min-norm
#   least-squares solution of the same problem when ``A == W``.  QED
#
# ``-ua`` is already in the walker's hand -- it is ``terminal_x``, passed to
# ``certificate_pair`` one statement earlier -- so the validity condition is an
# O(n) index comparison.  THE GUARD IS LOAD-BEARING: measured over the frozen
# captures, ``A == W`` on 122/122 fit1d faces (agreement 1.7e-11 relative) and
# 6/7 ship04s faces (3.1e-13), and the single ``A != W`` face disagrees by
# 1.3e-3 -- nine orders worse.  That face falls through to the unchanged call.
#
# WHAT IS *NOT* SHORT-CIRCUITED, AND WHY (measured, and it corrects the source
# lens).  ``_certificate_detail`` uses one more thing from its lstsq than the
# solution: the RANK.  When ``consist`` is inside tolerance but ``viol`` is
# not, it asks ``rank < m`` and, if so, runs a null-space ``newton4``
# violation-penalty repair that can turn a rejection into a certificate.
# Reproducing that decision needs lstsq's own rank (singular values above
# ``s_max max(shape) eps``), and every exact way to get it costs at least what
# the lstsq costs -- measured on a real fit1d face (1026 x 1026):
#
#     np.linalg.lstsq            0.203 s      <- what we are trying to avoid
#     np.linalg.svd (values)     0.214 s
#     np.linalg.svd (full)       0.345 s
#     np.linalg.qr (mode="r")    0.048 s      <- cheap, but a DIFFERENT cut
#
# So on a face that reaches the rank question there is nothing to win, and
# this wrapper DELEGATES to the untouched ``_certificate_detail``.  A first
# implementation instead recomputed the SVD to get the rank and was 390 ms per
# call against the incumbent's 239 ms -- a 21 s LOSS on fit1d, measured in both
# arm orders.  The source lens's POC avoided that cost by returning "primal
# violation" WITHOUT consulting the rank at all; that is fail-closed but it is
# NOT decision-identical, and it is where its 1.36x on fit1d came from.  It is
# available here as ``W_CERT_SKIP_RANK`` (opt-in, default OFF).
#
# What remains, and is taken by default, is every face that does NOT reach the
# rank question: ``consist`` out of tolerance, or ``viol`` inside it (the
# branch that leads to the duality-gap test and ``certificate_pair``).  There
# the lstsq is genuinely redundant and the guard makes its removal exact.
W_CERT = True
W_CERT_SKIP_RANK = False
W_CERT_STATS = {"fast": 0, "fallback": 0, "early": 0, "repair": 0,
                "rank_delegate": 0, "t_fast": 0.0, "t_fallback": 0.0}


def _reset_wcert_stats():
    for key in W_CERT_STATS:
        W_CERT_STATS[key] = 0.0 if key.startswith("t_") else 0


def _wcert_fallback(B, b, d, y, started):
    W_CERT_STATS["fallback"] += 1
    out = _certificate_detail(B, b, d, y)
    W_CERT_STATS["t_fallback"] += time.perf_counter() - started
    return out


def _certificate_detail_wcert(B, b, d, y, idxW, x_hat):
    """``_certificate_detail`` with the redundant lstsq removed when A == W."""
    started = time.perf_counter()
    if not W_CERT or x_hat is None or idxW is None:
        return _wcert_fallback(B, b, d, y, started)
    n, m = B.shape
    sc_d = max(1.0, float(np.abs(d).max()))
    dres = float(np.max(np.abs(B.T @ y - d)))
    if dres > TOL_FEAS * sc_d or (y.size and float(y.min()) < -TOL_FEAS):
        # the incumbent returns HERE, before the lstsq: nothing to save
        W_CERT_STATS["early"] += 1
        out = _certificate_detail(B, b, d, y)
        W_CERT_STATS["t_fallback"] += time.perf_counter() - started
        return out
    A = np.where(y > TOL_Y * max(1.0, float(y.max()) if y.size else 1.0))[0]
    x = np.asarray(x_hat, dtype=float)
    if (A.size == 0 or x.shape != (m,) or A.size != np.size(idxW)
            or not np.array_equal(A, idxW)):
        return _wcert_fallback(B, b, d, y, started)
    out = _wcert_fast(B, b, d, y, A, x, m)
    if out is None:
        # the call needs lstsq's RANK; hand it to the untouched incumbent
        W_CERT_STATS["rank_delegate"] += 1
        return _wcert_fallback(B, b, d, y, started)
    W_CERT_STATS["fast"] += 1
    W_CERT_STATS["t_fast"] += time.perf_counter() - started
    return out


def _wcert_fast(B, b, d, y, A, x, m):
    """The incumbent's tests, verbatim, on the walker's own ``x = -ua``.

    ``None`` means "this call reaches the rank question" -- see the header.
    """
    consist = float(np.max(np.abs(B[A] @ x - b[A])))
    viol = max(0.0, float(-(B @ x - b).min()))
    sc_b = max(1.0, float(np.abs(b).max()))
    if consist <= TOL_FEAS * sc_b and viol > TOL_FEAS * sc_b:
        if not W_CERT_SKIP_RANK:
            return None
        W_CERT_STATS["repair"] += 1
    if consist > TOL_FEAS * sc_b:
        return False, x, consist / sc_b, "active consistency"
    if viol > TOL_FEAS * sc_b:
        return False, x, viol / sc_b, "primal violation"
    px, dy = float(d @ x), float(b @ y)
    gap = abs(px - dy) / max(1.0, abs(px), abs(dy))
    if gap > TOL_FEAS:
        return False, x, gap, "duality gap"
    pair_ok, pair_detail = certificate_pair(B, b, d, x, y)
    if not pair_ok:
        pair_err = max(pair_detail.get("primal", 0.0),
                       pair_detail.get("dual", 0.0),
                       pair_detail.get("nonnegative", 0.0),
                       pair_detail.get("gap", 0.0))
        return False, x, pair_err, "componentwise " + pair_detail["reason"]
    return True, x, max(consist, viol) / sc_b, "passed"


def follow_selector_gated(B, b, d, t=1.0, maxpiv=4000, tmax=1e8, tol=TOL_FEAS,
                          trace=None, progress=None,
                          max_critical_transitions=100, legacy_repair=True,
                          allow_settle_acceptance=True, resync_attempts=8,
                          endpoint_reject_scale=1e-6,
                          ratio_multiplier="min_norm", stats=None,
                          dual_ascent_repair=True, repair_order="adaptive",
                          ladder_settle_rounds=None, drift_guard=True,
                          capture=None, init_seed=None, init_projection=True,
                          fast_path=True, unit_core=False,
                          factor_update=False, sparse_b=False,
                          rz_core=False, sparse_face=False):
    """Follow the dual path with selector acceptance and cone transitions.

    ``stats`` may be a caller-owned dict; counters accumulate there so an
    interrupted run (SIGALRM) still exposes them.

    ``ratio_multiplier`` selects the multiplier used for off-face breakpoint
    events on an accepted face: ``"min_norm"`` (the unique bounded choice,
    exp23 parity) or ``"selector"`` (the forward-affine LP vertex, certified
    only on [t, t+delta]).  The selector LP remains the acceptance gate in
    both modes.  ``endpoint_reject_scale`` bounds how negative an exact
    breakpoint endpoint may be before the critical transition refuses it;
    the 1e-6 default matches exp23's critical_vertex_escape.

    ``drift_guard`` verifies breakpoint endpoints against one independent
    tight projection when a cheap trigger fires (ill-conditioned walking
    face, or a blown-up event multiplier).  A face can pass every relative
    backward-error gate and still extrapolate off the true path (measured
    9.0e-2 at Lotfi pivot 80 with all internal residuals at 1e-13); the
    walker's own algebra cannot see that, so the only detector is an
    independent re-projection.  On material disagreement the face is
    re-derived from the projection support through the standard acceptance
    ladder; fail-closed otherwise.

    ``repair_order`` sequences the two degenerate-event repairs after direct
    acceptance fails at a breakpoint: ``"ladder_first"`` runs the critical
    epsilon ladder then M5; ``"m5_first"`` runs the one-shot M5 dual-ascent
    repair first and reaches the epsilon ladder only when M5 fails;
    ``"adaptive"`` chooses per event — ladder first iff the critical right
    support contracts, an endpoint probe is not eps-flat, and no fixed-t
    repair already failed at this t (lens C predicate).
    ``ladder_settle_rounds`` bounds the settle depth inside the epsilon
    ladder; ``None`` selects 20 under ``"ladder_first"`` (incumbent) and 5
    under ``"m5_first"`` (the ladder is then a fallback and its deep settles
    are the residual waste).

    ``init_seed="highs_basis"`` (opt-in, default ``None`` = unchanged
    behavior) seeds initialization and resync attempt 0 with the support of
    one bounded auxiliary HiGHS Phase-1 basic solution of ``{B'y = d,
    y >= 0}`` — a basic support is rank-complete by construction, targeting
    the initialization failure class (brandy/grow7).  The seed only feeds
    the standard acceptance machinery; every gate is unchanged and failure
    falls open to the existing initialization path.

    ``init_projection=True`` adds an initialization rung — one tight fixed-t
    Mangasarian projection whose support is settled by exp23's ordinary
    settle (adds allowed) and then fed to the standard acceptance ladder —
    tried only AFTER the corrector-seeded ladder fails at initialization
    (and after everything else fails at a resync attempt, before t
    escalation).  Placement is deliberately fallback-only: models that
    initialize today take a bit-identical path; the rung targets the
    brandy-class failure where the projection support settles in ~2 rounds
    but the walker never seeds from the projection (Addendum 9).

    ``fast_path=True`` (default) runs the selector's OWN endpoint gate on
    the min-norm affine multiplier the face solve already produced, before
    the selector LP is built.  The min-norm pair satisfies both LP equality
    blocks exactly by construction (``piece_y``: ``B_W ua = g - b_W``,
    ``B_W uc = h``), so the only open question is off-face forward
    feasibility -- a pure mat-vec test.  When it passes, the LP is skipped
    and the min-norm multiplier is the walking multiplier, which is exactly
    what the LP path returns under ``ratio_multiplier="min_norm"``; the
    fast path is therefore restricted to that mode and is a pure
    computational shortcut there.  Any failure falls through to the
    unchanged selector-LP path.  ``fast_path=False`` restores the
    LP-on-every-acceptance behavior for A/B runs.

    ``factor_update=True`` (opt-in; default False leaves the block inert
    and the walk byte-identical) routes the face solve through
    ``face_factor_updating.FactorUpdatingFaceSolver``, which keeps one
    economy QR of ``B[W]`` alive across events by row insert/delete and
    refactorizes only under monitoring: an honest piece residual on
    original data after every updated solve, a hard cadence, a
    diagonal-ratio floor before any downdate, and a fresh factorization
    for any support change wider than two coordinates.  It sits BEHIND
    ``unit_core`` and IN FRONT of the persistent skeleton QR, so with both
    flags on it replaces exactly the skeleton fallback (the measured
    68.8-73.0 % wall bucket on fit1d/ship04s) and with ``unit_core=False``
    it is the primary face solve.  Every decline falls through to the
    unchanged path; no acceptance gate moves.

    ``sparse_b=True`` (opt-in; default False leaves every line below on the
    dense arrays and the walk byte-identical) stores ``B``, ``|B|``,
    ``B/column_scale`` and ``|B/column_scale|`` as CSR
    (``sparse_b_support.SparseB``) and routes the event loop's OWN products
    through them: the selector fast-path endpoint battery
    (``min_norm_endpoint_gate``, four boolean row slices of dense ``n x m``
    copies plus six mat-vecs per acceptance) and the breakpoint slopes
    ``B[off] @ ua`` / ``B[off] @ uc``.  Those two sites are the measured
    event-loop glue: on fit1d (2077 x 1026, 0.77 % dense, cProfile over 300
    pivots) 9.5 ms per gate call and 4.8 ms per pivot in the loop body,
    against 22 ms for the face solve.

    NOTHING ELSE MOVES.  Every frozen collaborator -- exp23's
    ``certificate_pair`` / ``_certificate_detail`` / ``settle`` /
    ``qp_corrector`` / ``piece_y``, ``dual_ascent_repair.repair_face``,
    ``_solve_projection``, ``critical_cone_path``, and all three face
    factorizations -- is still called with the ORIGINAL dense ``B``, so no
    certificate is ever computed through a different code path.  The
    selector LP is handed the same LP: ``select_forward_affine_multiplier``
    receives the CSR ``B`` plus the dense-computed ``column_scale``, and
    builds its blocks with ``scipy.sparse.bmat`` instead of ``np.block``,
    which is what HiGHS gets internally either way (verified event by event:
    identical multipliers).

    The stored VALUES are bit-identical to the dense ones; the SUMMATION
    ORDER of the products is not (CSR accumulates a row in column-index
    order and skips structural zeros).  Breakpoint times therefore agree to
    round-off rather than to the last bit, which can in principle reorder a
    marginal degenerate tie -- hence opt-in, and hence the A/B.
    """
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    if B.ndim != 2:
        raise ValueError("B must be a two-dimensional array")
    n, m = B.shape
    if n == 0 or m == 0 or b.shape != (n,) or d.shape != (m,):
        raise ValueError("expected B.shape == (len(b), len(d))")
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(b))
            and np.all(np.isfinite(d))):
        raise ValueError("B, b, and d must contain only finite values")
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and strictly positive")
    if not np.isfinite(tmax) or tmax < t:
        raise ValueError("tmax must be finite and no smaller than t")
    if (maxpiv < 0 or resync_attempts < 1 or max_critical_transitions < 0
            or not np.isfinite(endpoint_reject_scale)
            or endpoint_reject_scale <= 0.0):
        raise ValueError("iteration limits must be nonnegative, resync "
                         "attempts positive, and endpoint_reject_scale a "
                         "positive finite scale")
    if ratio_multiplier not in ("min_norm", "selector"):
        raise ValueError("ratio_multiplier must be 'min_norm' or 'selector'")
    if repair_order not in ("ladder_first", "m5_first", "adaptive"):
        raise ValueError("repair_order must be 'ladder_first', 'm5_first', "
                         "or 'adaptive'")
    if ladder_settle_rounds is not None and ladder_settle_rounds < 1:
        raise ValueError("ladder_settle_rounds must be positive when given")
    if init_seed not in (None, "highs_basis"):
        raise ValueError("init_seed must be None or 'highs_basis'")
    ladder_rounds = (ladder_settle_rounds if ladder_settle_rounds is not None
                     else (5 if repair_order == "m5_first" else 20))

    st = stats if stats is not None else {}
    _reset_wcert_stats()
    st.update(corr=0, jumps=0, mono=True, face_solves=0,
              factor_fallbacks=0, factor_condition_fallbacks=0,
              accept_calls=0, accept_failures=0, accept_fast_path=0,
              accept_fast_path_calls=0, accept_fast_path_max_error=0.0,
              accept_fast_path_strict_offface=0,
              accept_direct=0, accept_after_short_settle=0,
              critical_calls=0, critical_transitions=0,
              critical_rejections=0, accept_via_settle_gate=0,
              legacy_repair_calls=0, legacy_repairs=0, resyncs=0,
              m5_calls=0, m5_successes=0, m5_failures=0, m5_wall_total=0.0,
              cone_raw_b_retries=0, cone_entered_last=0, cone_entered_max=0,
              adaptive_ladder_choices=0, adaptive_m5_choices=0,
              drift_triggers=0, drift_repairs=0, drift_agreements=0,
              drift_repair_failures=0, drift_reprojections_wall=0.0,
              init_seed_calls=0, init_seed_accepted=0, init_seed_wall=0.0,
              init_projection_calls=0, init_projection_accepted=0,
              init_projection_wall=0.0,
              # A reused caller-owned stats dict must not leak routing
              # memory (m5_last_failed_t suppresses M5 at a float-equal t
              # and flips the adaptive predicate) or selector-LP counts
              # across runs.
              m5_last_failed_t=None,
              primal_pricing_calls=0, primal_pricing_affine_lp_calls=0,
              primal_pricing_affine_accepts=0,
              primal_pricing_candidate_evaluations=0,
              primal_pricing_steps=0,
              primal_pricing_objective_descent_steps=0,
              primal_pricing_settled=0, primal_pricing_cycle_failures=0,
              primal_pricing_no_proposal_failures=0,
              primal_pricing_exhausted_failures=0,
              primal_pricing_round_cap_failures=0)
    st.setdefault("critical_rejection_reasons", {})
    st.setdefault("drift_trigger_reasons", {})

    # ``rz_core=True`` (opt-in; default False leaves the walk byte-identical)
    # swaps the persistent skeleton factor for the subclass that solves its
    # rank core with ONE dtzrzf instead of two GELSY calls.  Same class, same
    # gates, same residual tests, same decline semantics -- only the two
    # ``lstsq(lapack_driver="gelsy")`` calls move, and they were re-deriving a
    # rank-revealing factorization the persistent QR already holds.  Measured
    # on the 839 faces five netlib walks actually solve: 2.20x on face-solve
    # wall, accuracy identical to GELSY against piece_y's SVD truth, and zero
    # additional declines.
    factor = None
    factor_active = False
    factor_class = PathPieceRZFactor if rz_core else PathPieceSkeletonFactor
    try:
        factor = factor_class(B, b, d)
        factor_active = True
    except (ValueError, np.linalg.LinAlgError):
        st["factor_fallbacks"] += 1

    # ``unit_core=True`` (opt-in; default False leaves this block inert and
    # the walk byte-identical) installs the structure-exploiting face solver
    # in FRONT of the persistent QR.  It returns the same
    # ``(g, h, ua, uc, dres)`` contract, declines any face it cannot solve to
    # a residual well inside the incumbent gate, and every decline routes to
    # the unchanged path below.  See unit_core_face_solver.py.
    unit_core_solver = None
    if unit_core:
        st.update(unit_core_solves=0, unit_core_fallbacks=0)
        st.setdefault("unit_core_fallback_reasons", {})
        try:
            unit_core_solver = UnitCoreFaceSolver(B, b, d)
            st["unit_core_structure"] = unit_core_solver.structure()
        except (ValueError, np.linalg.LinAlgError):
            unit_core_solver = None
            st["unit_core_fallback_reasons"]["init failed"] = 1

    # ``factor_update=True`` (opt-in; default False leaves this block inert
    # and the walk byte-identical) installs the update-with-monitored-
    # refactorization face solver between unit_core and the persistent
    # skeleton QR.  Same ``(g, h, ua, uc, dres)`` contract, same fail-closed
    # residual gate; every decline routes to the unchanged path below.
    fup_solver = None
    if factor_update:
        st.update(fup_declines=0)
        try:
            fup_solver = FactorUpdatingFaceSolver(B, b, d, stats=st)
        except (ValueError, np.linalg.LinAlgError):
            fup_solver = None
            st["fup_declines"] += 1

    # ``sparse_face=True`` (opt-in; default False leaves the walk
    # byte-identical) puts the SuiteSparseQR minimum-norm face solve in front
    # of the dense persistent factor.  The faces are 0.6-4.6 % dense and the
    # dense rank-revealing QR is O(|W| m^2), so the dense path costs 37 ms per
    # face on sctap1 (m=480) against 2.6 ms here -- 14x, 60/60 accurate.
    # Every decline falls through to the unchanged dense chain below.
    sparse_face_solver = None
    if sparse_face:
        st.update(sparse_face_solves=0, sparse_face_declines=0)
        try:
            sparse_face_solver = SparseFaceSolver(B, b, d, residual_gate=tol)
        except (ValueError, ImportError, np.linalg.LinAlgError):
            sparse_face_solver = None

    def solve_path_piece(mask):
        # Per-face SVD fallback only; the persistent factor is never disabled
        # permanently (unlike exp23) because settle-driven batch jumps are no
        # longer the common case here.
        st["face_solves"] += 1
        if sparse_face_solver is not None:
            try:
                coefficients = sparse_face_solver.solve(mask)
                st["sparse_face_solves"] += 1
                return coefficients
            except (SparseFaceDecline, ValueError, RuntimeError,
                    np.linalg.LinAlgError):
                st["sparse_face_declines"] += 1
        if unit_core_solver is not None:
            try:
                coefficients = unit_core_solver.solve(mask)
                st["unit_core_solves"] += 1
                return coefficients
            except UnitCoreFallback as exc:
                st["unit_core_fallbacks"] += 1
                reasons = st["unit_core_fallback_reasons"]
                reasons[exc.code] = reasons.get(exc.code, 0) + 1
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                st["unit_core_fallbacks"] += 1
                reasons = st["unit_core_fallback_reasons"]
                reasons["exception"] = reasons.get("exception", 0) + 1
        if fup_solver is not None:
            try:
                return fup_solver.solve(mask)
            except FactorUpdatingFallback:
                st["fup_declines"] += 1
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                st["fup_declines"] += 1
        if factor_active:
            try:
                return factor.solve(mask)
            except PathPieceConditionError:
                st["factor_condition_fallbacks"] += 1
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                st["factor_fallbacks"] += 1
        return piece_y(B, b, d, mask)

    # Equality reduction for the critical cone, computed once per model.
    try:
        E_cached, _f, _sel, _rs, reduction_consistency = _prepare(B, d)
    except (ValueError, np.linalg.LinAlgError, ImportError):
        E_cached, reduction_consistency = None, np.inf
    st["reduction_consistency"] = float(reduction_consistency)

    # Conditioning companion (lens 3): the cone QP's objective carries a huge
    # constant 0.5||P_range(E') b||^2 that swamps clarabel's determinability.
    # Projecting b onto null(E) preserves the argmin exactly (the feasible
    # cone lies inside {Ev = 0}) and removes that constant.  Verified on the
    # frozen pivot-20 capture: dominant-coordinate agreement 3.5e-12, same
    # 12 ms solve.  The qnormal-perp refinement was tried and REJECTED: it
    # degraded clarabel to AlmostSolved with 4.9e-2 dominant disagreement.
    null_E_basis = None
    if E_cached is not None and reduction_consistency <= tol:
        try:
            q_full, _r_full = np.linalg.qr(E_cached.toarray().T,
                                           mode="complete")
            null_E_basis = q_full[:, E_cached.shape[0]:]
        except (ValueError, np.linalg.LinAlgError):
            null_E_basis = None

    # Model-constant scalings used by the selector's endpoint gate
    # (primal_face_pricing.select_forward_affine_multiplier lines 75-77 and
    # 126-144).  Hoisted here so the fast path costs mat-vecs only.
    fp_column_scale = np.linalg.norm(B, axis=0)
    fp_column_scale = np.where(fp_column_scale > 0.0, fp_column_scale, 1.0)
    # ``sparse_b=True`` replaces the two dense n x m companions with CSR
    # forms carrying the SAME values (see sparse_b_support); the dense
    # copies are then never built, which is also where the memory goes
    # (fit1d: 34 MB of companions against 0.4 MB of CSR).
    spB = None
    fp_B_scaled = fp_absB = fp_absB_scaled = None
    if sparse_b:
        spB = SparseB(B, fp_column_scale)
        st["sparse_b"] = spB.summary()
    else:
        fp_B_scaled = B / fp_column_scale
        fp_absB = np.abs(B)
        fp_absB_scaled = np.abs(fp_B_scaled)
    fp_active = bool(fast_path) and ratio_multiplier == "min_norm"
    # fast_path="audit": identical walk to fast_path=True, but every
    # fast-path accept ALSO runs the selector LP and records whether the LP
    # would have accepted the same face.  Pure measurement of the
    # fast-path/LP decision gap; never changes what is returned.
    fp_audit = fast_path == "audit"
    if fp_audit:
        st.update(accept_fast_path_audit_calls=0,
                  accept_fast_path_audit_lp_rejects=0)
        st.setdefault("accept_fast_path_audit_reasons", {})

    def min_norm_endpoint_gate(mask, parameter, coefficients):
        """Selector endpoint gate evaluated on the min-norm multiplier.

        Byte-for-byte the residual battery of
        ``primal_face_pricing.select_forward_affine_multiplier`` (same
        probes ``(1e-6, 1e-8, 1e-10)``, same relative scalings, same
        ``tol``, same ``max(...) <= tol`` acceptance), with the LP's
        solution vector replaced by the min-norm pair ``u(t) = t*ua + uc``
        that the face solve already returned.  That pair satisfies the LP's
        equality blocks exactly (``piece_y``), so ``eq0_error`` and
        ``eqa_error`` are round-off; the discriminating terms are the
        off-face inequalities at ``t`` and ``t + delta``, which the LP
        imposes as constraints and this test verifies directly.  The
        battery is duplicated rather than imported because
        ``primal_face_pricing`` is outside this change's edit authority and
        the residual block there is inline in the LP loop; any future edit
        to that block must be mirrored here.

        Returns ``(error, detail)`` on pass, ``(None, detail)`` otherwise.
        """
        g_c, h_c, ua_c, _uc_c, dres_c = coefficients
        y0 = parameter * g_c + h_c
        yscale = 1.0 + np.abs(y0)
        nonnegative0 = max(0.0, float(np.max(-y0 / yscale)))
        if nonnegative0 > tol:
            return None, {"reason": "negative face point",
                          "error": nonnegative0}
        off = ~mask
        rhs0 = y0 - parameter * b[mask]
        rhsa = g_c - b[mask]
        objective_invariance_target = float(h_c @ b[mask])
        # u0 = u(t), ua = du/dt: the LP's two variable blocks, supplied by
        # the min-norm face solve instead of by HiGHS.
        u0 = parameter * ua_c + _uc_c
        u0s = u0 * fp_column_scale
        uas = ua_c * fp_column_scale
        # Under ``sparse_b`` the four boolean row slices of dense n x m
        # copies are replaced by full CSR products indexed afterwards:
        # ``B_W @ v`` is ``(B @ v)[mask]`` and ``B_off @ v`` is
        # ``(B @ v)[off]``, same entries, O(nnz) instead of O(n m).
        if spB is not None:
            eq0_row = (spB.Bs @ u0s)[mask]
            eqa_row = (spB.Bs @ uas)[mask]
            eq0_absrow = (spB.absBs @ np.abs(u0s))[mask]
            eqa_absrow = (spB.absBs @ np.abs(uas))[mask]
        else:
            BW = fp_B_scaled[mask]
            absBW = fp_absB_scaled[mask]
            eq0_row = BW @ u0s
            eqa_row = BW @ uas
            eq0_absrow = absBW @ np.abs(u0s)
            eqa_absrow = absBW @ np.abs(uas)
        eq0_scale = 1.0 + np.abs(rhs0) + eq0_absrow
        eqa_scale = 1.0 + np.abs(rhsa) + eqa_absrow
        eq0_error = float(np.max(np.abs(eq0_row - rhs0) / eq0_scale))
        eqa_error = float(np.max(np.abs(eqa_row - rhsa) / eqa_scale))
        off0_error = 0.0
        Boff_raw = absBoff_raw = b_off = None
        if off.any():
            b_off = b[off]
            if spB is not None:
                q0 = parameter * b_off + (spB.B @ u0)[off]
                scale0 = (1.0 + np.abs(parameter * b_off)
                          + (spB.absB @ np.abs(u0))[off])
            else:
                Boff_raw = B[off]
                absBoff_raw = fp_absB[off]
                q0 = parameter * b_off + Boff_raw @ u0
                scale0 = (1.0 + np.abs(parameter * b_off)
                          + absBoff_raw @ np.abs(u0))
            off0_error = max(0.0, float(np.max(q0 / scale0)))
        objective = float(d @ (-ua_c))
        invariance_error = abs(objective - objective_invariance_target) / max(
            1.0, abs(objective), abs(objective_invariance_target))
        last = {"reason": "no probe"}
        for relative_probe in (1e-6, 1e-8, 1e-10):
            delta = float(relative_probe) * max(1.0, abs(float(parameter)))
            tp = float(parameter) + delta
            yp = tp * g_c + h_c
            yp_scale = 1.0 + np.abs(yp)
            yp_nonnegative = max(0.0, float(np.max(-yp / yp_scale)))
            if yp_nonnegative > tol:
                last = {"reason": "forward y leaves face", "delta": delta,
                        "error": yp_nonnegative}
                continue
            offp_error = 0.0
            if off.any():
                up = u0 + delta * ua_c
                if spB is not None:
                    qp = tp * b_off + (spB.B @ up)[off]
                    scalep = (1.0 + np.abs(tp * b_off)
                              + (spB.absB @ np.abs(up))[off])
                else:
                    qp = tp * b_off + Boff_raw @ up
                    scalep = (1.0 + np.abs(tp * b_off)
                              + absBoff_raw @ np.abs(up))
                offp_error = max(0.0, float(np.max(qp / scalep)))
            error = max(eq0_error, eqa_error, off0_error, offp_error,
                        nonnegative0, yp_nonnegative, invariance_error,
                        float(dres_c))
            detail = {
                "reason": "min-norm accepted" if error <= tol
                          else "endpoint residual",
                "delta": delta, "t_to": tp, "error": error,
                "equality_at_t": eq0_error, "equality_slope": eqa_error,
                "offface_at_t": off0_error, "offface_at_t_to": offp_error,
                "nonnegative_at_t": nonnegative0,
                "nonnegative_at_t_to": yp_nonnegative,
                "objective_invariance": invariance_error,
                "objective": objective,
            }
            if error <= tol:
                return error, detail
            last = detail
        return None, last

    def record_progress(stage, pivot=-1, mask=None, certified_face=False,
                        **fields):
        if progress is None:
            return
        record = dict(stage=str(stage), pivot=int(pivot), t=float(t))
        if mask is not None:
            mask_copy = np.asarray(mask, dtype=bool).copy()
            record["working_set"] = mask_copy
            record["support"] = int(mask_copy.sum())
        record.update(fields)
        progress["updates"] = int(progress.get("updates", 0)) + 1
        progress["current"] = record
        if certified_face:
            # "Accepted" not "certified": settle-gate and legacy faces pass
            # exact fixed-t gates but no original-data LP certificate.
            progress["last_accepted_face"] = record

    def accept_face(mask, parameter, seen=None, detail_sink=None):
        """The single face-acceptance gate: solve + selector LP."""
        if mask is None or not mask.any():
            if detail_sink is not None:
                detail_sink.append({"stage": "accept", "reason": "empty mask"})
            return None
        if seen is not None:
            key = np.packbits(mask).tobytes()
            if key in seen:
                if detail_sink is not None:
                    detail_sink.append({"stage": "accept", "reason": "seen"})
                return None
            seen.add(key)
        st["accept_calls"] += 1
        try:
            coefficients = solve_path_piece(mask)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            st["accept_failures"] += 1
            if detail_sink is not None:
                detail_sink.append({"stage": "accept", "reason": "face solve",
                                    "exception": type(error).__name__})
            return None
        g, h, _ua, _uc, dres = coefficients
        if float(dres) > tol:
            st["accept_failures"] += 1
            if detail_sink is not None:
                detail_sink.append({"stage": "accept",
                                    "reason": "dual equality",
                                    "dres": float(dres)})
            return None
        yW = parameter * g + h
        yscale = 1e-8 * max(1.0, float(np.abs(yW).max()) if yW.size else 1.0)
        if yW.size and float(yW.min()) < -yscale:
            st["accept_failures"] += 1
            if detail_sink is not None:
                detail_sink.append({"stage": "accept",
                                    "reason": "negative face point",
                                    "min_yW": float(yW.min()),
                                    "scale": float(yscale)})
            return None
        if fp_active:
            # Selector fast path: the min-norm multiplier is already an
            # exact solution of the LP's equality blocks, so run the
            # selector's own endpoint battery on it first.  A pass means
            # the LP's feasible set is non-empty at this probe (the
            # min-norm pair is a witness) and, under
            # ratio_multiplier="min_norm", the multiplier returned below is
            # the very one the LP path would have returned -- so the LP is
            # pure overhead here and is skipped.  A failure changes
            # nothing: control falls through to the untouched LP path.
            st["accept_fast_path_calls"] += 1
            fp_error, fp_detail = min_norm_endpoint_gate(
                mask, parameter, coefficients)
            if fp_error is not None:
                st["accept_fast_path"] += 1
                st["accept_fast_path_max_error"] = max(
                    float(st.get("accept_fast_path_max_error", 0.0)),
                    float(fp_error))
                if (float(fp_detail.get("offface_at_t", 0.0)) > 0.0
                        or float(fp_detail.get("offface_at_t_to", 0.0)) > 0.0):
                    # Off-face residual positive but within tol: the
                    # min-norm witness is tol-feasible, not exactly
                    # feasible, for the LP.  Counted so the equivalence gap
                    # is measurable rather than assumed.
                    st["accept_fast_path_strict_offface"] += 1
                if fp_audit:
                    audit_stats = {}
                    audit_selected, audit_detail = (
                        select_forward_affine_multiplier(
                            B, b, d, mask, parameter, coefficients, tol=tol,
                            stats=audit_stats))
                    st["accept_fast_path_audit_calls"] += 1
                    if audit_selected is None:
                        st["accept_fast_path_audit_lp_rejects"] += 1
                        audit_reasons = st["accept_fast_path_audit_reasons"]
                        key_r = str(audit_detail.get("reason"))
                        audit_reasons[key_r] = audit_reasons.get(key_r, 0) + 1
                if detail_sink is not None:
                    detail_sink.append({"stage": "fast path",
                                        "accepted": True,
                                        "support": int(mask.sum()),
                                        "detail": fp_detail})
                g_fp, h_fp, ua_fp, uc_fp, _dres_fp = coefficients
                return g_fp, h_fp, ua_fp, uc_fp
            if detail_sink is not None:
                detail_sink.append({"stage": "fast path", "accepted": False,
                                    "support": int(mask.sum()),
                                    "detail": fp_detail})
        # Under ``sparse_b`` the selector is handed the CSR ``B`` and the
        # dense-computed column scaling, so its blocks are assembled with
        # ``scipy.sparse.bmat`` instead of ``np.block``.  The LP itself is
        # unchanged: the explicit zeros ``np.block`` writes are exactly the
        # structural zeros ``bmat`` omits, and HiGHS is handed a sparse
        # matrix either way.
        selected, _detail = select_forward_affine_multiplier(
            B if spB is None else spB.B, b, d, mask, parameter, coefficients,
            tol=tol, stats=st,
            column_scale=None if spB is None else fp_column_scale)
        if detail_sink is not None:
            detail_sink.append({"stage": "selector",
                                "accepted": selected is not None,
                                "support": int(mask.sum()),
                                "detail": _detail})
        if selected is None:
            st["accept_failures"] += 1
            return None
        if ratio_multiplier == "min_norm":
            # The selector LP proves forward feasibility but its multiplier
            # is an arbitrary basic vertex with uncontrolled norm in
            # null(B_W) directions.  Walk with the unique min-norm
            # multiplier instead (exp23 parity); acceptance stands.
            g_mn, h_mn, ua_mn, uc_mn, _dres = coefficients
            return g_mn, h_mn, ua_mn, uc_mn
        return selected

    def acceptance_ladder(mask, parameter, seen, short_rounds=3,
                          settle_gate=False, detail_sink=None):
        """Trial 1 (direct accept) then trial 2 (short settle + accept).

        With ``settle_gate=True`` (used only inside the critical epsilon
        ladder and resync, mirroring exp23's incumbent acceptance semantics)
        a face whose plain settle passes its exact fixed-t gates is accepted
        with the min-norm multiplier when the forward-affine selector
        declines — forward feasibility can be genuinely infeasible exactly at
        a tie, where requiring the selector deadlocks the walk.
        """
        pc_local = accept_face(mask, parameter, seen=seen,
                               detail_sink=detail_sink)
        if pc_local is not None:
            return mask, pc_local, "direct"
        settle_diag = {} if detail_sink is not None else None
        settled_W, settled_pc, settled_ok = settle(
            B, b, d, mask.copy(), parameter, rounds=short_rounds,
            diagnostics=settle_diag, piece_solver=solve_path_piece)
        if detail_sink is not None:
            detail_sink.append({
                "stage": "settle", "ok": bool(settled_ok),
                "diagnostics": settle_diag,
                "support": (int(settled_W.sum())
                            if settled_W is not None else 0)})
        if settled_W is not None and settled_W.any():
            pc_local = accept_face(settled_W, parameter, seen=seen,
                                   detail_sink=detail_sink)
            if pc_local is not None:
                return settled_W, pc_local, "short settle"
        if (settle_gate and allow_settle_acceptance and settled_ok
                and settled_pc is not None):
            st["accept_via_settle_gate"] += 1
            return settled_W, settled_pc, "settle gate"
        return mask, None, None

    def note_accept(how):
        if how == "direct":
            st["accept_direct"] += 1
        elif how == "short settle":
            st["accept_after_short_settle"] += 1

    def legacy_ladder(mask, parameter):
        """Trial 4: full settle then the priced combinatorial repair."""
        st["legacy_repair_calls"] += 1
        settled_W, _c, _ok = settle(B, b, d, mask.copy(), parameter,
                                    rounds=20, piece_solver=solve_path_piece)
        if settled_W is not None and settled_W.any():
            pc_local = accept_face(settled_W, parameter)
            if pc_local is not None:
                st["legacy_repairs"] += 1
                return settled_W, pc_local
        try:
            priced_W, priced_pc, priced_ok = priced_settle_escape(
                B, b, d, mask.copy(), parameter,
                piece_solver=solve_path_piece, max_candidates=12, rounds=30,
                tol=tol, stats=st)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return mask, None
        if priced_ok and priced_pc is not None:
            # priced_settle_escape terminates through the same selector, so
            # its success already carries the acceptance certificate.
            st["legacy_repairs"] += 1
            return priced_W, priced_pc
        return mask, None

    def dual_ascent_face(parameter):
        """M5: exact dual-ascent face repair at fixed t (lens 4).

        Solves the Mangasarian projection once, then moves the non-unique
        multiplier inside null(B_P) until the tight set spans full rank.
        The returned working set still passes the walker's STANDARD
        settle-gate acceptance (unchanged gates); fail-closed otherwise.
        """
        if not dual_ascent_repair:
            return None, None
        if st.get("m5_last_failed_t") == float(parameter):
            return None, None
        st["m5_calls"] += 1
        started = time.perf_counter()
        try:
            repaired = repair_face(B, b, d, parameter, tol=tol)
        except (ValueError, RuntimeError, ImportError,
                np.linalg.LinAlgError) as error:
            repaired = {"status": "EXCEPTION",
                        "detail": {"exception": type(error).__name__}}
        st["m5_wall_total"] += time.perf_counter() - started
        mask = repaired.get("working_set")
        if repaired.get("status") != "PASS" or mask is None:
            st["m5_failures"] += 1
            st["m5_last_failed_t"] = float(parameter)
            _emit_trace(trace, "dual ascent repair failed",
                        t=float(parameter),
                        status=repaired.get("status"),
                        reason=repaired.get("detail", {}).get("reason"))
            return None, None
        mask = np.asarray(mask, dtype=bool)
        accepted_W, accepted_pc, how = acceptance_ladder(
            mask, parameter, None, short_rounds=20, settle_gate=True)
        if accepted_pc is None:
            st["m5_failures"] += 1
            st["m5_last_failed_t"] = float(parameter)
            _emit_trace(trace, "dual ascent face not accepted",
                        t=float(parameter), support=int(mask.sum()))
            return None, None
        st["m5_successes"] += 1
        _emit_trace(trace, "dual ascent repair", t=float(parameter),
                    support=int(accepted_W.sum()), route=how,
                    wall=repaired.get("wall"),
                    threshold=repaired.get("detail", {}).get(
                        "winning_threshold"))
        return accepted_W, accepted_pc

    def reject_critical(reason, **fields):
        st["critical_rejections"] += 1
        reasons = st["critical_rejection_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        _emit_trace(trace, "critical transition rejected", reason=reason,
                    **fields)
        return None

    def critical_prepare(y_vertex, parameter):
        """Endpoint hygiene + cone solve; returns prep dict or None."""
        st["critical_calls"] += 1
        if E_cached is None or reduction_consistency > tol:
            return reject_critical("equality reduction",
                                   error=float(reduction_consistency))
        if st["critical_transitions"] >= max_critical_transitions:
            return reject_critical("transition cap")
        yscale = max(1.0, float(np.abs(y_vertex).max()) if y_vertex.size
                     else 1.0)
        min_neg = float(y_vertex.min()) if y_vertex.size else 0.0
        if min_neg < -endpoint_reject_scale * yscale:
            return reject_critical("negative endpoint", minimum=min_neg,
                                   t=float(parameter))
        yv = np.maximum(y_vertex, 0.0)
        tol_z = max(1e-8 * yscale, 2.0 * abs(min(min_neg, 0.0)))
        zero = yv <= tol_z
        # Zero-classified coordinates must contribute exactly t*b_i to the
        # normal; a stale sub-threshold value here makes the (zero, qnormal)
        # pair self-inconsistent and the derivative solves the wrong cone.
        yv = yv.copy()
        yv[zero] = 0.0
        qnormal = parameter * b - yv

        def cone_solve(b_used):
            derivative = critical_derivative(E_cached, b_used, qnormal, zero)
            if "Solved" not in derivative["status"]:
                return None, ("critical QP",
                              {"status": derivative["status"]})
            v_local = derivative["v"]
            v_local_scale = max(1.0, float(np.abs(v_local).max())
                                if v_local.size else 1.0)
            equality = (float(np.linalg.norm(E_cached @ v_local, np.inf))
                        / max(1.0, float(np.linalg.norm(v_local, np.inf))))
            tangent = (max(0.0, float(-v_local[zero].min())) / v_local_scale
                       if zero.any() else 0.0)
            orthogonality = abs(float(qnormal @ v_local)) / max(
                1.0, float(np.linalg.norm(qnormal))
                * float(np.linalg.norm(v_local)))
            gate_values.update(equality=equality, tangent=tangent,
                               orthogonality=orthogonality)
            if max(equality, tangent, orthogonality) > tol:
                return None, ("critical residual",
                              {"equality": equality, "tangent": tangent,
                               "orthogonality": orthogonality})
            return v_local, None

        gate_values = {}

        if null_E_basis is not None:
            b_eff = null_E_basis @ (null_E_basis.T @ b)
        else:
            b_eff = b
        v, rejection = cone_solve(b_eff)
        if v is None and null_E_basis is not None:
            st["cone_raw_b_retries"] += 1
            v, rejection = cone_solve(b)
        if v is None:
            return reject_critical(rejection[0], **rejection[1])
        vscale = max(1.0, float(np.abs(v).max()) if v.size else 1.0)
        entered_count = int(np.sum(zero & (v > 1e-8 * vscale)))
        st["cone_entered_last"] = entered_count
        st["cone_entered_max"] = max(st["cone_entered_max"], entered_count)
        right_support = (yv > tol_z) | (v > 1e-8 * vscale)
        if not right_support.any():
            return reject_critical("zero right derivative")
        return {"right_support": right_support, "zero": zero, "yv": yv,
                "qnormal": qnormal, "v": v, "tol_z": tol_z,
                "gate_values": gate_values,
                "y_vertex_raw": np.asarray(y_vertex, dtype=float)}

    def critical_ladder_run(prep, parameter, rounds=None):
        """Epsilon ladder over the prepared right support."""
        rounds = ladder_rounds if rounds is None else rounds
        right_support = prep["right_support"]
        zero = prep["zero"]
        ladder_details = [] if capture is not None else None
        for exponent in (-10, -9, -8, -7, -6):
            tp = parameter + (10.0 ** exponent) * max(1.0, abs(parameter))
            sink = [] if capture is not None else None
            mask2, pc2, how = acceptance_ladder(right_support.copy(), tp,
                                                None,
                                                short_rounds=rounds,
                                                settle_gate=True,
                                                detail_sink=sink)
            if ladder_details is not None:
                ladder_details.append({"exponent": int(exponent),
                                       "tp": float(tp),
                                       "accepted": pc2 is not None,
                                       "route": how, "trials": sink})
            if pc2 is not None:
                st["critical_transitions"] += 1
                _emit_trace(trace, "critical transition",
                            t_from=float(parameter), t_to=float(tp),
                            support=int(mask2.sum()),
                            entered=int(np.sum(zero & mask2)), how=how)
                return mask2, pc2, tp
        if capture is not None:
            capture({"kind": "right_face_not_accepted",
                     "t": float(parameter),
                     "y_vertex_raw": prep["y_vertex_raw"].copy(),
                     "y_vertex_clipped": prep["yv"].copy(),
                     "zero_mask": zero.copy(),
                     "qnormal": prep["qnormal"].copy(),
                     "v": np.asarray(prep["v"], dtype=float).copy(),
                     "right_support": right_support.copy(),
                     "tol_z": float(prep["tol_z"]),
                     "gates": {key: float(value)
                               for key, value in prep["gate_values"].items()},
                     "ladder": ladder_details})
        return reject_critical("right face not accepted")

    def critical_transition(y_vertex, parameter, rounds=None):
        """Trial 3: exact right-derivative transition from the endpoint."""
        prep = critical_prepare(y_vertex, parameter)
        if prep is None:
            return None
        return critical_ladder_run(prep, parameter, rounds=rounds)

    def endpoint_probe(mask, parameter):
        """Single-rung probe: is the candidate face's negativity eps-flat?

        One face solve at t+eps, no settle.  An eps-flat negativity (the
        Lotfi pivot-20 signature: min y ~ -1.6e-2 at every eps) means the
        right support needs fixed-t rank completion, not a ladder; a clean
        or eps-scaled minimum means the ladder can advance t.  Surrogate
        bound, not a certificate: acceptance downstream is unchanged.
        """
        eps = 1e-6 * max(1.0, abs(parameter))
        try:
            g_p, h_p, _ua_p, _uc_p, dres_p = solve_path_piece(mask)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return False
        if float(dres_p) > tol:
            return False
        yWp = (parameter + eps) * g_p + h_p
        if not yWp.size:
            return False
        yscale_p = max(1.0, float(np.abs(yWp).max()))
        return float(yWp.min()) >= -max(1e-8 * yscale_p, eps)

    # Opt-in HiGHS Phase-1 basis seed (bounded auxiliary LP, charter-
    # admissible).  The feasible set C = {y >= 0 : B'y = d} does not move
    # with t, so the basic solution is computed once and cached; a simplex
    # basis has a rank-complete support by construction, which is exactly
    # the structure the degenerate-initialization class lacks.
    phase1_state = {"computed": False, "mask": None}

    def phase1_basis_mask():
        if phase1_state["computed"]:
            return phase1_state["mask"]
        phase1_state["computed"] = True
        from scipy.optimize import linprog as _linprog
        for name, objective in (("zeros", np.zeros(n)), ("ones", np.ones(n))):
            try:
                result = _linprog(objective, A_eq=B.T, b_eq=d,
                                  bounds=[(0.0, None)] * n,
                                  method="highs-ds")
            except Exception as error:
                # Harness alarms must escape; solver errors fail open.
                if type(error).__name__ == "Alarm":
                    raise
                continue
            if not getattr(result, "success", False):
                continue
            y_seed = np.asarray(result.x, dtype=float)
            scale = max(1.0, float(np.abs(y_seed).max()) if y_seed.size
                        else 0.0)
            mask = y_seed > 1e-9 * scale
            if mask.any():
                phase1_state["mask"] = mask
                st["init_seed_objective"] = name
                st["init_seed_support"] = int(mask.sum())
                break
        return phase1_state["mask"]

    def seeded_acceptance(parameter):
        """Feed the Phase-1 basis support through the standard machinery."""
        if init_seed != "highs_basis":
            return None, None, None
        started = time.perf_counter()
        mask = phase1_basis_mask()
        if mask is None:
            st["init_seed_wall"] += time.perf_counter() - started
            return None, None, None
        st["init_seed_calls"] += 1
        W_s, pc_s, how_s = acceptance_ladder(
            mask.copy(), parameter, set(), short_rounds=20, settle_gate=True)
        st["init_seed_wall"] += time.perf_counter() - started
        if pc_s is None:
            _emit_trace(trace, "phase1 seed not accepted", t=float(parameter),
                        seed_support=int(mask.sum()))
            return None, None, None
        st["init_seed_accepted"] += 1
        _emit_trace(trace, "phase1 seed accepted", t=float(parameter),
                    seed_support=int(mask.sum()),
                    support=int(W_s.sum()), route=how_s)
        return W_s, pc_s, "phase1 seed (" + str(how_s) + ")"

    def init_projection_rung(parameter):
        """Fallback rung: tight projection support + ordinary settle.

        The brandy-class initialization failure (Addendum 9): the projection
        at t solves cleanly and exp23's ordinary settle certifies a face
        from its support in ~2 rounds, but the walker only ever seeds from
        the corrector.  This rung supplies that face.  Acceptance runs the
        standard ladder with the settle gate (exp23's incumbent
        initialization semantics); all gates unchanged, fail-open.
        """
        if not init_projection:
            return None, None, None
        st["init_projection_calls"] += 1
        started = time.perf_counter()
        try:
            y_proj, _u_proj, proj_status = _solve_projection(
                B, parameter * b, d, tol=1e-11)
        except Exception as error:  # clarabel raises plain Exceptions
            if type(error).__name__ == "Alarm":
                raise
            y_proj, proj_status = None, "EXCEPTION"
        if y_proj is None or "Solved" not in proj_status:
            st["init_projection_wall"] += time.perf_counter() - started
            _emit_trace(trace, "init projection rung failed",
                        t=float(parameter), reason=str(proj_status))
            return None, None, None
        proj_scale = max(1.0, float(np.abs(y_proj).max()))
        mask = y_proj > 1e-9 * proj_scale
        W_r = pc_r = how_r = None
        if mask.any():
            settled_W, _spc, _sok = settle(
                B, b, d, mask.copy(), parameter, rounds=20,
                piece_solver=solve_path_piece)
            seed = (settled_W if _sok and settled_W is not None
                    and settled_W.any() else mask)
            W_r, pc_r, how_r = acceptance_ladder(
                seed.copy(), parameter, set(), short_rounds=20,
                settle_gate=True)
        st["init_projection_wall"] += time.perf_counter() - started
        if pc_r is None:
            _emit_trace(trace, "init projection rung failed",
                        t=float(parameter), reason="not accepted",
                        seed_support=int(mask.sum()))
            return None, None, None
        st["init_projection_accepted"] += 1
        _emit_trace(trace, "init projection rung accepted",
                    t=float(parameter), support=int(W_r.sum()),
                    route=how_r)
        return W_r, pc_r, "init projection (" + str(how_r) + ")"

    prev_lb = -np.inf

    def out(k, status, x, lb, y=None, backend="selector-gated dual path"):
        st.update(pivots=k, status=status, x=x, y=y, lb=lb, t=t,
                  backend=backend)
        st["selector_lp_calls"] = st.get("primal_pricing_affine_lp_calls", 0)
        st["selector_accepts"] = st.get("primal_pricing_affine_accepts", 0)
        st["w_cert"] = dict(W_CERT_STATS)
        if factor is not None:
            st.update(factor.diagnostics())
        if fup_solver is not None:
            st.update(fup_solver.diagnostics())
        record_progress("return", pivot=k, status=status, backend=backend)
        return st

    # Initialization: opt-in Phase-1 seed first, else corrector, then the
    # acceptance ladder.  Seed failure falls open to the unchanged path.
    pc = how = None
    W2 = None
    if init_seed is not None:
        W2, pc, how = seeded_acceptance(t)
        if pc is not None:
            W = W2
    if pc is None:
        corrector_detail = {}
        W = qp_corrector(B, b, d, t, diagnostics=corrector_detail)
        st["corr"] += 1
        W2, pc, how = acceptance_ladder(np.asarray(W, dtype=bool), t, set())
        if pc is None:
            W_r, pc_r, how_r = init_projection_rung(t)
            if pc_r is not None:
                W2, pc, how = W_r, pc_r, how_r
        if pc is None and legacy_repair:
            W2, pc = legacy_ladder(np.asarray(W, dtype=bool), t)
            how = "legacy" if pc is not None else None
    ok = pc is not None
    if ok:
        W = W2
        note_accept(how)
    _emit_trace(trace, "initialization", t=float(t),
                support=int(np.asarray(W2).sum()), settle_ok=bool(ok),
                accept_route=how)
    record_progress("initialization", mask=W2, certified_face=bool(ok),
                    settle_ok=bool(ok), accept_route=how)

    for k in range(maxpiv):
        if not ok:
            st["resyncs"] += 1
            record_progress("resync required", pivot=k, mask=W,
                            settle_ok=False)
            for att in range(resync_attempts):
                if att == 0:
                    # M5 first: one exact projection + dual ascent replaces
                    # the multi-second tangent projection corrector when it
                    # succeeds; fail-closed to the corrector otherwise.
                    W_m5, pc_m5 = dual_ascent_face(t)
                    if pc_m5 is not None:
                        W = W_m5
                        pc = pc_m5
                        ok = True
                        _emit_trace(trace, "resync", pivot=k, attempt=att,
                                    t=float(t), settle_ok=True,
                                    accept_route="dual ascent")
                        record_progress("resync", pivot=k, mask=W,
                                        certified_face=True, attempt=0,
                                        route="dual ascent")
                        break
                    if init_seed is not None:
                        W_s, pc_s, how_s = seeded_acceptance(t)
                        if pc_s is not None:
                            W = W_s
                            pc = pc_s
                            ok = True
                            _emit_trace(trace, "resync", pivot=k,
                                        attempt=att, t=float(t),
                                        settle_ok=True, accept_route=how_s)
                            record_progress("resync", pivot=k, mask=W,
                                            certified_face=True, attempt=0,
                                            route=how_s)
                            break
                corrector_detail = {}
                Wc = qp_corrector(B, b, d, t, attempt=att,
                                  active_set_repair=(att == 0),
                                  diagnostics=corrector_detail,
                                  retain_projection_working_set=(att == 0))
                st["corr"] += 1
                projected_y = corrector_detail.get("projection_y")
                if projected_y is not None:
                    okp, xp, _err, reason = _certificate_detail(
                        B, b, d, projected_y)
                    _emit_trace(trace, "projected LP certificate", pivot=k,
                                t=float(t), passed=bool(okp), reason=reason)
                    if okp:
                        return out(k, "CERTIFIED", xp,
                                   float(b @ projected_y),
                                   np.asarray(projected_y, dtype=float),
                                   backend="selector-gated + tangent "
                                           "projection")
                Wc = np.asarray(Wc, dtype=bool)
                W2, pc, how = acceptance_ladder(Wc, t, set(),
                                                short_rounds=20,
                                                settle_gate=True)
                if pc is None and legacy_repair:
                    W2, pc = legacy_ladder(Wc, t)
                    how = "legacy" if pc is not None else None
                _emit_trace(trace, "resync", pivot=k, attempt=att,
                            t=float(t), settle_ok=pc is not None,
                            accept_route=how,
                            corrector_backend=corrector_detail.get("backend"))
                if pc is not None:
                    W = W2
                    ok = True
                    note_accept(how)
                    if capture is not None:
                        capture({
                            "kind": "resync_accepted", "pivot": int(k),
                            "t": float(t), "attempt": int(att),
                            "route": how, "W": W2.copy(),
                            "pc": tuple(np.asarray(z, dtype=float).copy()
                                        for z in pc),
                            "projection_y": (
                                np.asarray(projected_y, dtype=float).copy()
                                if projected_y is not None else None),
                            "corrector_backend": corrector_detail.get(
                                "backend")})
                    record_progress("resync", pivot=k, mask=W,
                                    certified_face=True, attempt=int(att))
                    break
                if att == 0:
                    # Last rung before escalation: the projection-support
                    # initialization rung (fires only where the walk would
                    # otherwise escalate t — currently-working resyncs are
                    # bit-identical).
                    W_r, pc_r, how_r = init_projection_rung(t)
                    if pc_r is not None:
                        W = W_r
                        pc = pc_r
                        ok = True
                        _emit_trace(trace, "resync", pivot=k, attempt=att,
                                    t=float(t), settle_ok=True,
                                    accept_route=how_r)
                        record_progress("resync", pivot=k, mask=W,
                                        certified_face=True, attempt=int(att),
                                        route=how_r)
                        break
                t = 2.0 * t + 1.0
                st["jumps"] += 1
                if t > tmax:
                    return out(k, "t cap (resync)", None, prev_lb)
            if not ok:
                return out(k, "resync failed", None, prev_lb)

        g, h, ua, uc = pc
        record_progress("affine piece", pivot=k, mask=W, certified_face=True,
                        settle_ok=True)
        idxW = np.where(W)[0]
        y = np.zeros(n)
        y[idxW] = np.maximum(t * g + h, 0.0)
        lb = float(b @ y)
        if lb < prev_lb - 1e-7 * max(1.0, abs(lb)):
            st["mono"] = False
        prev_lb = lb

        # Terminal-certificate gate, identical to exp23.follow.
        gate = float(g @ g) <= 1e-12 * max(1.0, float(b[idxW] @ b[idxW]))
        if gate:
            terminal_y = np.zeros(n)
            terminal_y[idxW] = h
            terminal_x = -ua
            terminal_ok, terminal_detail = certificate_pair(
                B, b, d, terminal_x, terminal_y)
            _emit_trace(trace, "terminal pair certificate", pivot=k,
                        t=float(t), support=int(W.sum()),
                        passed=bool(terminal_ok), **terminal_detail)
            if terminal_ok:
                return out(k, "CERTIFIED", terminal_x,
                           float(b @ terminal_y), terminal_y,
                           backend="selector-gated stabilized terminal pair")
            # The incumbent recovers the primal with a dense lstsq and, on a
            # rank-deficient face, a dense SVD for the null-space basis --
            # both on a matrix the sparse solver just factorized.  When that
            # factorization is still current, reuse it; otherwise fall through
            # to the untouched dense path.  On sctap1 the SVD alone was 10 %
            # of the run and the lstsq another 7 %.
            sparse_cert = None
            if sparse_face_solver is not None:
                sparse_cert = sparse_face_solver.certificate(
                    y, terminal_x, idxW)
            if sparse_cert is not None:
                okc, x, _err, cert_reason = sparse_cert
            else:
                okc, x, _err, cert_reason = _certificate_detail_wcert(
                    B, b, d, y, idxW, terminal_x)
            _emit_trace(trace, "certificate", pivot=k, t=float(t),
                        support=int(W.sum()), gate=True, passed=bool(okc),
                        reason=cert_reason)
            if okc:
                return out(k, "CERTIFIED", x, lb, y)

        # Breakpoints from the accepted multiplier (min-norm or selector,
        # per ratio_multiplier; leaving events g, h are multiplier-free).
        s = np.empty(n)
        c = np.empty(n)
        s[idxW] = g
        c[idxW] = h
        off = ~W
        if spB is not None:
            # One CSR product over the whole matrix, indexed afterwards,
            # instead of two dense boolean row slices (17 MB each on fit1d)
            # and two dense mat-vecs over them.
            s[off] = -(b[off] + (spB.B @ ua)[off])
            c[off] = -((spB.B @ uc)[off])
        else:
            s[off] = -(b[off] + B[off] @ ua)
            c[off] = -(B[off] @ uc)
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = np.where(s < -1e-13, -c / s, np.inf)
        fmask = np.isfinite(tc) & (tc > t * (1 + FWD) + FWD)
        if not fmask.any():
            if not gate:
                # same proposition, same guard; ``-ua`` is the min-norm
                # least-squares solution on this face whether or not the
                # ||g||-gate fired
                okc, x, _err, cert_reason = _certificate_detail_wcert(
                    B, b, d, y, idxW, -ua)
                _emit_trace(trace, "certificate", pivot=k, t=float(t),
                            support=int(W.sum()), gate=False,
                            passed=bool(okc), reason=cert_reason)
                if okc:
                    return out(k, "CERTIFIED", x, lb, y)
            t = 2.0 * t + 1.0
            st["jumps"] += 1
            if t > tmax:
                return out(k, "no event, cert fails", None, prev_lb)
            ok = False
            continue
        tn = float(tc[fmask].min())
        if tn > tmax:
            return out(k, "t cap (breakpoint)", None, prev_lb)
        tie = fmask & (np.abs(tc - tn) <= TIE * max(1.0, tn))
        y_vertex = np.zeros(n)
        y_vertex[idxW] = tn * g + h
        if capture is not None:
            capture({"kind": "breakpoint", "pivot": int(k), "t": float(tn),
                     "W_before": W.copy(), "g": np.asarray(g).copy(),
                     "h": np.asarray(h).copy(),
                     "ua": np.asarray(ua).copy(),
                     "uc": np.asarray(uc).copy(),
                     "tie_mask": tie.copy(),
                     "y_vertex": y_vertex.copy()})
        # Drift guard (lens B stream): the endpoint tn*g+h can sit far off
        # the true path while every internal residual is ~1e-13 — replaying
        # the pivot-80 spike showed fresh piece_y, the skeleton factor, and
        # the recorded endpoint all agreeing at drift 9.0e-2, so the error
        # lives in the FACE (its true-path events were invisible to the
        # walker), not in multiplier algebra.  Verify against one tight
        # projection when a cheap trigger fires and re-derive the face from
        # the projection support on material disagreement.  Gates unchanged:
        # the re-derived face passes the ordinary acceptance ladder or the
        # original endpoint is kept (fail-closed).
        if drift_guard:
            guard_reason = None
            u_event = tn * ua + uc
            u_scale = max(1.0, float(np.linalg.norm(tn * b[idxW])))
            if float(np.linalg.norm(u_event)) > 1e3 * u_scale:
                guard_reason = "multiplier norm"
            elif (factor_active
                  and np.array_equal(factor.indices, idxW)
                  and factor.last_diagonal_ratio < 1e-5):
                # The skeleton factor deliberately disables the parent's
                # condition dispatch; re-purpose that 1e-5 threshold as the
                # drift trigger (pivot 80's face: diagonal ratio 1.6e-7,
                # multiplier ratio only 0.87 — the norm test alone misses it).
                guard_reason = "ill-conditioned face"
            elif (fup_solver is not None
                  and np.array_equal(fup_solver.indices, idxW)
                  and fup_solver.last_diagonal_ratio < 1e-5):
                # When the updating lane answered this face the skeleton
                # factor never saw it, so the trigger above cannot fire.
                # The updating factor exposes the same diagonal ratio at the
                # same 1e-5 threshold, keeping the guard alive rather than
                # letting an opt-in speed lever silently disarm it.  (Its
                # ratio is min/max over the maintained diagonal, which after
                # row updates is no longer pivot-ordered; that is the
                # conservative direction — it triggers at least as often.)
                guard_reason = "ill-conditioned face"
            if guard_reason is not None:
                st["drift_triggers"] += 1
                reasons = st["drift_trigger_reasons"]
                reasons[guard_reason] = reasons.get(guard_reason, 0) + 1
                guard_started = time.perf_counter()
                try:
                    y_proj, _u_proj, proj_status = _solve_projection(
                        B, tn * b, d, tol=1e-11)
                except Exception as error:  # clarabel raises plain Exceptions
                    # Harness alarms must escape; only solver errors are
                    # absorbed (fail-closed to the un-guarded event path).
                    if type(error).__name__ == "Alarm":
                        raise
                    y_proj, proj_status = None, "EXCEPTION"
                st["drift_reprojections_wall"] += (time.perf_counter()
                                                   - guard_started)
                if y_proj is not None and "Solved" in proj_status:
                    proj_scale = max(1.0, float(np.abs(y_proj).max()))
                    gap = float(np.abs(y_vertex - y_proj).max())
                    if gap <= 1e-6 * proj_scale:
                        st["drift_agreements"] += 1
                    else:
                        # Re-derive from the projection support: drop-only
                        # trim (rank-deficient supports carry a few
                        # sub-noise members whose min-norm value is
                        # negative), then the ordinary acceptance ladder.
                        # Frozen-p80 unit: 2 drops -> selector ACCEPT,
                        # re-derived drift 6.0e-6 from 9.0e-2.
                        gpc = None
                        for guard_thr in (1e-7, 1e-8, 1e-9):
                            gS = y_proj > guard_thr * proj_scale
                            for _ in range(3):
                                if not gS.any():
                                    break
                                try:
                                    g_t, h_t, _u1, _u2, _dr = (
                                        solve_path_piece(gS))
                                except (ValueError, RuntimeError,
                                        np.linalg.LinAlgError):
                                    break
                                yWt = tn * g_t + h_t
                                t_scale = 1e-8 * max(
                                    1.0, float(np.abs(yWt).max()))
                                negs = yWt < -t_scale
                                if not negs.any():
                                    break
                                gdrop = np.zeros_like(gS)
                                gdrop[np.where(gS)[0]] = negs
                                gS = gS & ~gdrop
                            if not gS.any():
                                continue
                            gW, gpc, ghow = acceptance_ladder(
                                gS, tn, None, short_rounds=3)
                            if gpc is not None:
                                break
                        if gpc is not None:
                            st["drift_repairs"] += 1
                            _emit_trace(trace, "drift repair", pivot=k,
                                        t=float(tn), gap=float(gap),
                                        reason=guard_reason, route=ghow,
                                        support=int(gW.sum()))
                            W = gW
                            pc = gpc
                            t = tn
                            ok = True
                            record_progress("event accepted", pivot=k,
                                            mask=W, certified_face=True,
                                            route="drift repair")
                            continue
                        st["drift_repair_failures"] += 1
        W = W.copy()
        W[tie] = ~W[tie]
        t = tn
        record_progress("breakpoint", pivot=k, mask=W,
                        tie_count=int(tie.sum()), pending=True)

        W2, pc, how = acceptance_ladder(W, t, set())
        if pc is not None:
            W = W2
            ok = True
            note_accept(how)
            record_progress("event accepted", pivot=k, mask=W,
                            certified_face=True, route=how)
            continue
        # repair_order sequences the two degenerate-event repairs; the
        # corrector-backed resync below remains last in every order.
        # "adaptive" (lens C): decide per event from cheap observables —
        # ladder first iff the critical right support CONTRACTS (support-
        # contracting events are where the ladder's t-advance dominates,
        # measured 110x on sctap1), the endpoint probe is not eps-flat, and
        # no fixed-t repair already failed at this t.  The contraction test
        # is a surrogate, not a theorem; every downstream acceptance gate is
        # unchanged, so a wrong choice costs order, never validity.
        repaired_event = False
        if repair_order == "adaptive":
            prep = critical_prepare(y_vertex, t)
            ladder_leads = False
            if prep is not None:
                contracts = (int(prep["right_support"].sum())
                             <= int(idxW.size))
                non_recurrent = st.get("m5_last_failed_t") != float(t)
                ladder_leads = (contracts and non_recurrent
                                and endpoint_probe(
                                    prep["right_support"], t))
            if ladder_leads:
                st["adaptive_ladder_choices"] += 1
                crossed = critical_ladder_run(prep, t, rounds=20)
                if crossed is not None:
                    W, pc, t = crossed
                    ok = True
                    repaired_event = True
                    record_progress("event accepted", pivot=k, mask=W,
                                    certified_face=True, route="critical")
                if not repaired_event:
                    W_m5, pc_m5 = dual_ascent_face(t)
                    if pc_m5 is not None:
                        W, pc, ok, repaired_event = W_m5, pc_m5, True, True
                        record_progress("event accepted", pivot=k, mask=W,
                                        certified_face=True,
                                        route="dual ascent")
            else:
                st["adaptive_m5_choices"] += 1
                W_m5, pc_m5 = dual_ascent_face(t)
                if pc_m5 is not None:
                    W, pc, ok, repaired_event = W_m5, pc_m5, True, True
                    record_progress("event accepted", pivot=k, mask=W,
                                    certified_face=True, route="dual ascent")
                if not repaired_event and prep is not None:
                    crossed = critical_ladder_run(prep, t, rounds=5)
                    if crossed is not None:
                        W, pc, t = crossed
                        ok = True
                        repaired_event = True
                        record_progress("event accepted", pivot=k, mask=W,
                                        certified_face=True,
                                        route="critical")
        if not repaired_event and repair_order == "m5_first":
            W_m5, pc_m5 = dual_ascent_face(t)
            if pc_m5 is not None:
                W, pc, ok, repaired_event = W_m5, pc_m5, True, True
                record_progress("event accepted", pivot=k, mask=W,
                                certified_face=True, route="dual ascent")
        if not repaired_event and repair_order != "adaptive":
            crossed = critical_transition(y_vertex, t)
            if crossed is not None:
                W, pc, t = crossed
                ok = True
                repaired_event = True
                record_progress("event accepted", pivot=k, mask=W,
                                certified_face=True, route="critical")
        if not repaired_event and repair_order == "ladder_first":
            W_m5, pc_m5 = dual_ascent_face(t)
            if pc_m5 is not None:
                W, pc, ok, repaired_event = W_m5, pc_m5, True, True
                record_progress("event accepted", pivot=k, mask=W,
                                certified_face=True, route="dual ascent")
        if repaired_event:
            continue
        if legacy_repair:
            W2, pc = legacy_ladder(W, t)
            if pc is not None:
                W = W2
                ok = True
                record_progress("event accepted", pivot=k, mask=W,
                                certified_face=True, route="legacy")
                continue
        ok = False

    return out(maxpiv, "pivot cap", None, prev_lb)
