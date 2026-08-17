"""newton_oracle: an IPM-FREE, external-solver-free phase-one for the
Mangasarian projection face, feeding the FROZEN terminal-certification
machinery of ``oracle_terminal``.

WHY
---
``oracle_terminal`` needs a hint at the terminal projection face.  Until now
that hint came from HiGHS (scipy ``highs-ipm``, or highspy IPM with crossover
off).  This module computes the same hint with nothing but numpy/scipy sparse
linear algebra: a semismooth (finite) Newton maximization of the Mangasarian
dual of the projection itself.  Nothing in this file can make a certificate
pass -- ``oracle_terminal.terminal_test`` and exp23's frozen
``certificate_pair`` / ``_certificate_detail`` gates decide, unchanged, on the
original ``B, b, d``.  Every failure path falls back to ``exp23.follow``.

THE DERIVATION (verified numerically, see ``newton_oracle_report.md``)
---------------------------------------------------------------------
The path point is the projection

    y(t) = argmin { 0.5||y - t b||^2 : B'y = d, y >= 0 },     B is n x m.

Dualize the equality with a multiplier ``x`` in R^m.  With ``v = t b + B x``,

    min_{y>=0} 0.5||y - tb||^2 - x'(B'y - d)
      = d'x - 0.5||(v)_+||^2 + 0.5 t^2 ||b||^2,

so up to the ``x``-free constant the dual is the unconstrained CONCAVE
piecewise-quadratic

    theta_t(x) = d'x - 0.5 || (t b + B x)_+ ||^2,
    grad theta_t(x) = d - B' (t b + B x)_+ ,
    generalized Hessian = -B_S' B_S,   S = { i : (t b + B x)_i > 0 }.

At a maximizer, ``y = (t b + B x)_+`` is exactly ``y(t)`` and ``S`` is its
support.  RELATION TO exp23's PIECE ALGEBRA (checked, not assumed): with
``piece_y``'s ``(g, h, ua, uc)`` contract the multiplier of this dual is
``x(t) = t*ua + uc`` (exp23's own ``u`` is ``-x``: ``qp_corrector`` writes
``y = (t b - B u)_+``), so ``-x(t)/t -> -ua``, which is precisely the primal
LP candidate ``x* = -ua`` that ``oracle_terminal`` certifies.  That is where
this module's primal hint ``x_tilde = -x/t`` comes from.

CONDITIONING / SCALED FORM
--------------------------
The terminal ``t`` is large in practice (lotfi 3.4e5, fit1d 2.5e6, ship04s
2.2e7) and ``x(t) ~ t*ua`` blows up with it.  Substituting ``x = t x'`` gives
the mathematically identical

    theta'_t(x') = (d/t)'x' - 0.5 || (b + B x')_+ ||^2,
    y = t (b + B x')_+ ,    x_LP-candidate = -x' ,

whose maximizer ``x' = ua + uc/t`` is bounded and nearly ``t``-independent, so
warm starts across ``t`` stages are almost exact and no quantity grows with
``t``.  Both formulations are implemented (``scaled=True/False``) and measured;
the scaled one is the default.

THE NEWTON STEP (the singular-Hessian choice, stated honestly)
--------------------------------------------------------------
``H = B_S' B_S`` is PSD and routinely SINGULAR (rank(B_S) < m).  The step is
the pseudoinverse step ``delta = pinv(H) grad``, computed either from ONE thin
SVD of ``B_S`` (small faces) or, for large ones, from TWO sparse LSQR solves,

    z = argmin ||B_S' z - grad||  (min-norm)   -> z = pinv(B_S') grad
    delta = argmin ||B_S delta - z|| (min-norm) -> delta = pinv(B_S) z

which compose to ``pinv(H) grad`` without ever forming ``H`` (whose condition
number is squared).  The residual ``||H delta + r - grad||_inf /
||grad||_inf`` is measured every iteration and reported
(``max_step_residual``); it is NOT assumed to be zero.  MEASURED on the
27-model panel: > 1 on two models (grow7 2.5e4, e226 2.9e1), <= 0.88 on the
other 25.  A meaningless step is survivable here because the exact line search
below needs only an ASCENT DIRECTION and the frozen gates need only the FACE.

A nonzero residual is not always numerical error: ``grad`` can have a genuine
component ``r = grad - B_S' pinv(B_S') grad`` in ``null(B_S)``, along which
``theta`` is exactly LINEAR and increasing (``grad'r = ||r||^2 > 0``) and no
Newton step can move.  Each iteration therefore takes TWO line-searched steps,
first along the Newton direction and then along ``r``; without the second the
iteration stalls on degenerate faces.

LINE SEARCH -- exact, not Armijo (a measured decision)
------------------------------------------------------
``theta`` restricted to a ray is a concave piecewise QUADRATIC with at most
``n`` breakpoints, so its maximizer is available in closed form by sorting the
breakpoints and walking the segments (``_exact_line_search``, O(n log n)).
Armijo backtracking plus a first-breakpoint ratio test was implemented first
and MEASURED TO STALL: on afiro at t=1 it froze at ``theta = 387.7`` against
the true maximum ``theta* = 516.995`` (clarabel reference), because a
degenerate row with ``v_i = 0`` exactly gives ratio-test step ``alpha = 0``
and Armijo then accepts nothing.  The exact search steps straight through
such rows.  Verified against a brute-force scan on 300 random rays with forced
exact-zero degeneracies: 0 mismatches.

CAP-HIT IS NOT SUCCESS, BUT ITS FACE IS STILL TESTED
----------------------------------------------------
Iteration cap ``MAXITER``, plus a stall guard on ``theta`` (monotone by
construction, unlike ``|grad||``).  Neither sets ``converged``.  A stalled
point is nonetheless only a HINT: ``newton_terminal`` still hands its face to
the frozen gates, which can only REJECT it.  That is what carries fit1d and
ship04s, whose accepting stages are both non-converged.  Fail-closed is
preserved by the gates, not by the convergence flag.

Import side effects: none.  Nothing in the default stack imports this module.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import linalg
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp23_path_primal_dual as exp23
import oracle_terminal as ot
import woodbury_face_solver as wfs
from unit_core_face_solver import UnitCoreFaceSolver

# ---- router heuristics ONLY.  None of these can make a certificate pass. ----
GRAD_TOL = 1e-9         # ||grad||_inf <= GRAD_TOL * ||c||_inf  (scale free)
STALL_WINDOW = 25       # iterations of no THETA progress == stop
# PROMOTED (b1_churn_mask, agent_reports/28).  ``FaceStepKernel.step``'s
# support-churn accounting is a DIAGNOSTIC: ``stats["support_changes"]`` is
# written there and read in exactly one place, ``FaceStepKernel.counters()``,
# which turns it into the ``support_change`` summary block.  No routing
# decision, tolerance, cadence or gate reads it -- the ``max_changes`` cadence
# rule of ``_FastQR`` computes its OWN ``changes``.  So only the VALUE has to
# be preserved, and it is exactly: with A = idx and Bp = the previous idx,
# |A \ Bp| + |Bp \ A| = |A XOR Bp| = count_nonzero(S != S_prev), because ``idx``
# is exactly the support of the boolean mask ``S`` (the call site is
# ``kernel.step(S, np.where(S)[0], ...)``) and ``S`` has the same fixed length
# on every call.  One O(n) pass replaces two sort-based ``setdiff1d`` calls.
# Set False to run the pre-promotion prologue.
CHURN_MASK = True
STALL_FACTOR = 1e-12    # relative theta gain over the window that counts
MAXITER = 300           # cap-hit is an oracle FAILURE (fail closed)
LSQR_ATOL = 1e-13
DENSE_LIMIT = 700       # min(|S|, m) below which one dense SVD beats LSQR
NULL_TOL = 1e-12        # "grad has a null(B_S) component" trigger
T_MAX = 1e9             # escalation ceiling
T_STAGE_FACTOR = 2.0    # t_next = T_STAGE_FACTOR * t_bar
ORACLE_SECONDS = 60.0   # total wall cap on newton_terminal

# ---- fast-kernel (kernel="fast") knobs.  Router heuristics ONLY: every fast
# route is checked against the SAME honest residual the SVD route reports and
# declines to the next route on failure, so nothing here can change a gate. ---
# ||H delta + r - grad||_inf / ||grad||_inf a fast route must meet.  Set at
# the MEDIAN residual the incumbent SVD step itself achieves on this panel
# (~1e-6; its measured worst is 0.88 on 25 models and 2.5e4 on grow7), so an
# accepted fast step is never materially worse-conditioned than the step it
# replaces.  Measured A/B (scratchpad/newtfast_tune.py, 3 warm stages):
# 1e-8 declines so often the SVD still runs (lotfi 1.84 s), 1e-6 gives
# lotfi 1.37 s / brandy 0.62 s / e226 0.85 s, 1e-4 gives 1.16 / 0.60 / 0.64.
FAST_STEP_TOL = 1e-6
FAST_QR_CADENCE = 32    # row updates before a fresh pivoted factorization
FAST_QR_MAX_CHANGES = 24    # bigger support moves take a fresh factor
FAST_QR_DIAG_FLOOR = 1e-7   # refuse to downdate below this diagonal ratio
FAST_UC_COND_ITERS = 2      # power iterations in the unit-core cond(G) guard
FAST_UC_REFINE = 2          # Bjorck corrected-semi-normal refinement sweeps
FAST_STRIKES = 12       # consecutive declines after which a route is retired
# Below this column count the incumbent's dense SVD is pure LAPACK on a tiny
# matrix and the fast routes lose on Python/SciPy call overhead alone
# (measured: kb2 m=41 0.35x, blend m=83 0.56x, afiro m=32 0.56x, adlittle
# m=97 0.84x -- all slower than the SVD they replace).
FAST_MIN_COLUMNS = 100
# The maintained DENSE QR is only run on models whose COLUMN COUNT is below
# the incumbent's own dense/sparse crossover.  Above ``DENSE_LIMIT`` the
# incumbent switches to twin sparse LSQR solves, which a dense factorization
# cannot beat: measured on fit1d (1038 x 1026), LSQR costs 16.8 ms per step
# against 116 ms for one pivoted QR of the same face.  The test is on ``m``,
# not on ``min(|S|, m)``: early iterations of a big model have small supports
# and would pass a per-iteration test, and the handful of fast steps they
# take diverts the trajectory of a model the kernel cannot then help --
# measured on fit1d, one accepted fast step out of 531 and the wall went
# 14.4 s -> 22.0 s.  For ``m <= DENSE_LIMIT`` the two tests are identical.
FAST_QR_DENSE_LIMIT = DENSE_LIMIT

# ---- F2: honest-residual check on the INCUMBENT step, with an exact-split
# repair.  Measured (opportunity_ship04s.md section 3, POC 1b): on ship04s the
# incumbent's TWIN-LSQR step returns honest residuals up to 3.7e-02 and steps
# that differ from the correctly-solved ones by up to 1650 % -- the Newton
# crawl on that model has been driven by BADLY SOLVED steps.  Every fast route
# in FaceStepKernel is already held to ``tol``; the incumbent route was the one
# path whose residual was measured, reported and then used regardless.
#
# WHAT THE REPAIR MAY BE, AND WHY IT IS NOT A PIVOTED CHOLESKY.  The failing
# route is the pair of sparse LSQR solves, i.e. an INEXACT ITERATIVE solve --
# a solver failure, not a rank-cut choice.  The repair is therefore the exact
# route the incumbent would ITSELF have taken had the face been small enough:
# a thin dense SVD of ``B_S`` with the identical rank cut
# (``s > s[0] max(shape) eps``).  No new rank-cut convention is introduced.
# That restriction is measured, not stylistic: a first implementation repaired
# with a pivoted Cholesky of ``H`` (also an exact orthogonal split, but at
# LAPACK's squared-precision rank cut).  It lowered the honest residual on
# lotfi from 1.2e-05 to 1.1e-05 and cost 253 -> 572 Newton iterations and
# 2.15 s -> 30.75 s, because MOVING the rank cut moves ``null(B_S)`` and the
# null ray is the convergence engine (the exact-split law,
# opportunity_degenerate.md section 3).  A repair is only safe if it changes
# the ARITHMETIC, not the CUT.
#
# The dense-SVD route's own residual is measured and counted but NOT repaired:
# it is already an exact split at the incumbent's own cut, so a large residual
# there is a conditioning fact about the face, not a fixable solve.
# DEFAULTS, and why.  The CHECK is unconditional and free (the residual is
# already computed); it is counted per route so the failure rate is visible on
# every model.  The REPAIR is OFF by default -- measured on fit1d, the only
# panel model that reaches the twin-LSQR route with a failing residual:
#   kernel=fast          12.05 s / 398 it / max resid 4.3e-03   (repair off)
#                        11.68 s / 316 it / max resid 7.9e-07   (repair on)
#   kernel=fast+tik+pchol 8.65 s / 336 it / max resid 4.8e-04   (repair off)
#                        10.56 s / 347 it / max resid 1.2e-02   (repair on)
# It does what it says -- 50/50 steps repaired, three to four decades of
# residual -- and ``lb`` is BIT-IDENTICAL in all four arms, i.e. the badly
# solved steps cost iterations, not accuracy.  On the shipped lane (the second
# pair) it costs 22 % wall and does not reach budget, because S1 has already
# taken the well-conditioned faces and what is left for LSQR is the hard
# residue.  Opt in with ``exp84_staged_newton --f2-repair``.
F2_REPAIR = False
F2_STEP_TOL = FAST_STEP_TOL   # residual above which the repair is attempted
F2_LSQR_ROUNDS = 2            # cheap refinement rounds tried first
F2_DENSE_FALLBACK = False     # then the (costly) forced dense SVD
F2_REPAIR_MAXM = 4000         # never densify a face wider than this
# Cost bound: the repair may spend at most this multiple of the incumbent
# route's own accumulated seconds, so a model where it fires on every step
# pays at most a bounded multiple rather than an unbounded one.
F2_TIME_SHARE = 1.0


# ---------------------------------------------------- fast Newton-step kernel

class _TallQR:
    """Pivoted economy QR of a TALL matrix, maintained by insert/delete.

    Two orientations are needed because the Newton face ``B_S`` crosses the
    square shape constantly (measured on lotfi: ``|S| >= m`` on only 49 % of
    iterations):

    * ``which="row"`` -- the factor of ``B_S`` itself, used when
      ``|S| >= m``.  ``R`` is then ``m x m``; if it has full rank the Newton
      step is two triangular solves and ``r_null`` is exactly zero.
    * ``which="col"`` -- the factor of ``B_S'`` (an ``m x |S|`` tall matrix),
      used when ``|S| < m``, where ``B_S`` can never have full column rank.
      ``R`` is then ``|S| x |S|``; if it has full rank,
      ``pinv(B_S'B_S) = Q (RR')^{-1} Q'`` and
      ``r_null = grad - Q Q' grad`` -- again two triangular solves plus two
      thin matrix-vector products.

    Row (resp. column) ORDER is irrelevant here: ``B_S'B_S`` and
    ``range(B_S')`` are invariant under permutations of the face rows, unlike
    ``piece_y``'s ``g, h``.  So additions are appended at the end in one
    blocked call instead of being spliced into sorted position.

    The update discipline is `face_factor_updating.FactorUpdatingFaceSolver`'s,
    reproduced here because that class's ``solve`` returns ``piece_y``'s five
    outputs and refuses faces with fewer rows than columns:

      * a fresh pivoted factorization every ``cadence`` accumulated updates;
      * a fresh factor when the support moves by more than ``max_changes``
        coordinates (a jump, not a crawl);
      * downdates refused when the maintained diagonal ratio has fallen below
        ``diag_floor`` (removing information from an ill-conditioned factor is
        the one operation whose backward error is unbounded);
      * a fresh factor on any SciPy update exception or shape violation;
      * and -- the actual trust mechanism -- the caller re-checks every solve
        against ORIGINAL ``B`` and refactors, then declines, if it fails.
    """

    def __init__(self, B, which, cadence=FAST_QR_CADENCE,
                 max_changes=FAST_QR_MAX_CHANGES,
                 diag_floor=FAST_QR_DIAG_FLOOR, stats=None):
        self.B = B
        self.n, self.m = B.shape
        self.which = which
        self.cadence = int(cadence)
        self.max_changes = int(max_changes)
        self.diag_floor = float(diag_floor)
        self.order = []                 # face row indices, factor order
        self.Q = None
        self.R = None
        self.perm = np.arange(self.m)   # column permutation ("row" only)
        self.rank = 0
        self.updates = 0
        self.stats = {} if stats is None else stats

    def _bump(self, key):
        self.stats[key] = self.stats.get(key, 0) + 1

    @staticmethod
    def _rank_of(R):
        diagonal = np.abs(np.diag(R))
        if not diagonal.size:
            return 0
        top = float(diagonal.max())
        if top == 0.0:
            return 0
        return int(np.sum(diagonal > top * max(R.shape) * np.finfo(float).eps))

    def _diag_ratio(self):
        """Smallest/largest maintained diagonal, over the WHOLE diagonal.

        `face_factor_updating` measures this over the leading ``rank`` entries
        only, which is right when the factor is used at its own rank.  Here the
        SciPy downdate operates on the FULL triangle, so a trailing near-zero
        diagonal is exactly the entry whose reorthogonalization divides by
        round-off (observed directly: ``ZeroDivisionError`` inside
        ``scipy.linalg._decomp_update.reorthx``, raised in an
        exception-ignored context and therefore NOT catchable -- the only
        defence is to not attempt the downdate).
        """
        if self.R is None or not self.rank:
            return 1.0
        diagonal = np.abs(np.diag(self.R))
        top = float(diagonal.max())
        return 0.0 if top == 0.0 else float(diagonal.min() / top)

    def refactor(self, idx, reason):
        face = self.B[idx]
        if self.which == "row":
            self.Q, self.R, pivot = linalg.qr(face, mode="economic",
                                              pivoting=True, check_finite=False)
            self.perm = np.asarray(pivot, dtype=int)
        else:
            self.Q, self.R, pivot = linalg.qr(np.ascontiguousarray(face.T),
                                              mode="economic", pivoting=True,
                                              check_finite=False)
            idx = np.asarray(idx)[np.asarray(pivot, dtype=int)]
        self.order = [int(i) for i in idx]
        self.rank = self._rank_of(self.R)
        self.updates = 0
        self._bump(reason)

    def sync(self, idx):
        """Move the factor onto ``idx``; returns False if it must be rebuilt."""
        if self.Q is None:
            self.refactor(idx, "fast_qr_refactor_init")
            return True
        old = set(self.order)
        new = set(int(i) for i in idx)
        additions = sorted(new - old)
        removals = old - new
        changes = len(additions) + len(removals)
        if changes == 0:
            return True
        if changes > self.max_changes:
            self.refactor(idx, "fast_qr_refactor_multichange")
            return True
        if self.updates + changes > self.cadence:
            self.refactor(idx, "fast_qr_refactor_cadence")
            return True
        size = len(self.order) - len(removals) + len(additions)
        tall = size >= self.m if self.which == "row" else size <= self.m
        if not tall:
            self.refactor(idx, "fast_qr_refactor_shape")
            return True
        if removals and self._diag_ratio() < self.diag_floor:
            self.refactor(idx, "fast_qr_refactor_degenerate")
            return True

        order = list(self.order)
        Q, R = self.Q, self.R
        which = "row" if self.which == "row" else "col"
        try:
            for index in sorted(removals, key=order.index, reverse=True):
                position = order.index(index)
                Q, R = linalg.qr_delete(Q, R, position, which=which,
                                        overwrite_qr=True, check_finite=False)
                order.pop(position)
            if additions:
                rows = np.asarray(additions, dtype=int)
                block = self.B[rows]
                block = (block[:, self.perm] if self.which == "row"
                         else np.ascontiguousarray(block.T))
                Q, R = linalg.qr_insert(Q, R, block, len(order), which=which,
                                        overwrite_qru=True,
                                        check_finite=False)
                order.extend(int(i) for i in rows)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError,
                FloatingPointError):
            self.refactor(idx, "fast_qr_refactor_exception")
            return True
        self.Q, self.R = Q, R
        self.order = order
        self.rank = self._rank_of(R)
        self.updates += changes
        self.stats["fast_qr_inserts"] = (self.stats.get("fast_qr_inserts", 0)
                                         + len(additions))
        self.stats["fast_qr_deletes"] = (self.stats.get("fast_qr_deletes", 0)
                                         + len(removals))
        self._bump("fast_qr_updates")
        return False

    def _gram_solver(self):
        """A cheap application of ``G^{-1}``, ``G = R_r R_r'`` (``r x r``).

        Both orientations reduce to inverting the Gram matrix of the leading
        ``r`` rows of the maintained ``R``.  Splitting
        ``R_r = [R11 R12]`` with ``R11`` upper triangular and nonsingular,

            G = R11 R11' + R12 R12' = R11 (I + W W') R11',   W = R11^{-1} R12

        so ``G^{-1} = R11^{-T} (I + WW')^{-1} R11^{-1}`` and, by Woodbury,
        ``(I + WW')^{-1} = I - W (I + W'W)^{-1} W'`` with ``W'W`` only
        ``p x p``, ``p = k - r`` the rank DEFICIENCY.

        That is the point: the complete-orthogonal completion this replaces
        costs one economy QR of an ``m x r`` block, i.e. O(m r^2) with
        ``r`` near ``m``.  This costs O(r^2 p + p^3) with ``p`` typically
        under twenty, and computes the SAME pseudoinverse -- the deficiency,
        not the rank, is what has to be small.
        """
        R = self.R
        r = self.rank
        R11 = R[:r, :r]
        p = R.shape[1] - r if self.which == "row" else R.shape[0] - r
        if p <= 0:
            return lambda y: linalg.solve_triangular(
                R11.T, linalg.solve_triangular(R11, y, lower=False,
                                               check_finite=False),
                lower=True, check_finite=False)
        W = linalg.solve_triangular(R11, R[:r, r:], lower=False,
                                    check_finite=False)
        core = W.T @ W
        core[np.diag_indices_from(core)] += 1.0
        chol = linalg.cho_factor(core, lower=True, check_finite=False)

        def apply(y):
            u = linalg.solve_triangular(R11, y, lower=False,
                                        check_finite=False)
            u = u - W @ linalg.cho_solve(chol, W.T @ u, check_finite=False)
            return linalg.solve_triangular(R11.T, u, lower=True,
                                           check_finite=False)
        return apply

    def step(self, grad):
        """``(delta, r_null, rank_deficient)``, or ``None`` if rank 0.

        FULL RANK in the ``"row"`` orientation (``rank == m``) is the one case
        that reproduces the incumbent's step independently of any cutoff:
        ``R`` is square and nonsingular, ``pinv(B_S'B_S) = R^{-1}R^{-T}`` is
        two triangular solves and ``r_null`` is exactly zero.

        Otherwise the step is built from ``G^{-1}`` above.  With
        ``U = R_r'`` (``m x r``, full column rank) the ``"row"`` face has
        ``H_P = U U'``, so ``pinv(H_P) = U G^{-2} U'`` and the projector onto
        ``range(B_S')`` is ``U G^{-1} U'``; the ``"col"`` face has
        ``H = Q_r G Q_r'`` with ``pinv(H) = Q_r G^{-1} Q_r'`` and projector
        ``Q_r Q_r'``.  Here, and only here, the answer depends on a numerical
        RANK CUTOFF -- the QR diagonal's rather than the SVD's -- so the
        caller's honest-residual gate is what decides whether it is believed.
        """
        R = self.R
        k = R.shape[0]
        r = self.rank
        if not r:
            return None
        if self.which == "row":
            g = grad[self.perm]
            if r == k:
                u = linalg.solve_triangular(R.T, g, lower=True,
                                            check_finite=False)
                step = linalg.solve_triangular(R, u, lower=False,
                                               check_finite=False)
                delta = np.empty(self.m)
                delta[self.perm] = step
                return delta, np.zeros(self.m), False
            Rr = R[:r]
            gram = self._gram_solver()
            z = gram(Rr @ g)
            null = np.empty(self.m)
            null[self.perm] = g - Rr.T @ z
            delta = np.empty(self.m)
            delta[self.perm] = Rr.T @ gram(z)
            return delta, null, True
        Qr = self.Q if r == k else self.Q[:, :r]
        a = Qr.T @ grad
        return Qr @ self._gram_solver()(a), grad - Qr @ a, r != k


class FaceStepKernel:
    """Composed Newton-step solver: unit-core, updated QR, fresh SVD.

    ``step(S, idx, grad, gnorm, iter_lim)`` returns
    ``(delta, r_null, route, residual)`` where ``residual`` is the SAME
    honest quantity ``newton_projection`` already reports,
    ``||B_S'(B_S delta) + r_null - grad||_inf / ||grad||_inf``, measured on the
    ORIGINAL operator.  A fast route whose residual exceeds ``tol`` is
    discarded and the next route runs, so the answer handed to the line search
    is never worse-conditioned than the incumbent's by more than ``tol``.

    Route order (each is counted):

    1. ``unit-core`` -- ``unit_core_face_solver``'s bordered
       ``K = [[D, A'], [A, -I]]``, whose Schur complement is exactly
       ``H = B_S'B_S``, with that module's own row classification, cond(G)
       guard and Bjorck refinement.  Applicable only when ``|S| >= m``: the
       bordered solve computes ``H^{-1} grad`` and asserts ``r_null = 0``,
       which is true only for a full-column-rank face.  **OPT-IN ONLY**
       (``mode="fast+unitcore"``) -- see the SuperLU note below.
    2. ``qr-update`` -- ``_TallQR`` above, factor UPDATED across Newton
       iterations rather than rebuilt.  Run only on models the incumbent
       itself would solve densely (``m <= FAST_QR_DENSE_LIMIT``); above that
       the incumbent's twin sparse LSQR is cheaper than any dense
       factorization of the same face.
    3. ``svd`` -- the unchanged ``_pinv_step`` (dense SVD or twin LSQR).

    WHY THE BORDERED ROUTE IS OPT-IN: SuperLU (``scipy...splu``) SEGFAULTS on
    this workload.  Measured on ship04s: exit 139, faulthandler stack ending
    in ``scipy/sparse/linalg/_dsolve/linsolve.py:438 splu`` (crash reports
    name ``_superlu.so dgstrf``/``dpanel_bmod``, EXC_BAD_ACCESS), reproduced
    four times.  A segfault kills the interpreter, so ``follow_with_newton``'s
    fail-closed ``except Exception`` never runs and the model yields no
    certificate at all -- strictly worse than a slow answer.  The trigger is
    NOT an input property that can be screened: the crashing matrix was
    captured and, replayed 500 times in a fresh process, raises SuperLU's
    ordinary ``RuntimeError: Factor is exactly singular`` every time and never
    crashes.  It is state dependent (SuperLU's singular-abort path runs many
    times on these faces before the crash).  The structural validation in
    ``_unit_core`` is still applied and counted (``splu_structural_veto``)
    because it is cheap and removes one class of bad input, but it is NOT a
    safety argument.  The only guarantee is not calling ``splu``, which is
    what ``mode="fast"`` does.

    Retirement: a route that declines ``FAST_STRIKES`` times in a row on one
    model stops being tried, so a model whose faces it can never serve pays
    its cost a bounded number of times rather than on every iteration.
    EXCEPTION -- the S1 tik route, whose attempt is ~5x cheaper than the
    twin-LSQR call it replaces on exactly the models its gate admits: it is
    governed by a wasted-time share instead (``_tik_admit``), because the
    strike rule turned one hard 12-face stretch on fit1d into permanent
    retirement, forfeiting 263 of 330 calls of which 116+ were serviceable.
    """

    def __init__(self, B, Bs, b, d, mode="fast", tol=FAST_STEP_TOL,
                 cadence=FAST_QR_CADENCE, max_changes=FAST_QR_MAX_CHANGES,
                 diag_floor=FAST_QR_DIAG_FLOOR, unit_core=None,
                 qr_update=True, BsT=None, f2_repair=None, f2_tol=None):
        self.B = np.asarray(B, dtype=float)
        self.Bs = Bs
        # opt-in explicit CSR of B' (see _transpose_csr): bit-identical
        # products, better locality.  None keeps the CSC view.
        self.BsT = Bs.T if BsT is None else BsT
        self.n, self.m = self.B.shape
        self.mode = mode
        self.tol = float(tol)
        self.stats = {"step_unit_core": 0, "step_qr_update": 0, "step_svd": 0,
                      "step_qr_deficient": 0,
                      "step_calls": 0, "fast_qr_inserts": 0,
                      "fast_qr_deletes": 0, "fast_qr_updates": 0,
                      "uc_declines": 0, "qr_declines": 0,
                      "splu_structural_veto": 0,
                      "t_unit_core": 0.0, "t_qr_sync": 0.0, "t_qr_solve": 0.0,
                      "t_residual": 0.0, "t_svd": 0.0,
                      "step_tik": 0, "tik_declines": 0, "t_tik": 0.0,
                      "tik_attempts": 0, "tik_wasted": 0.0,
                      "tik_stage_hist": [0] * len(wfs.TIK_LADDER),
                      "tik_readmits": 0,
                      "step_pchol": 0, "pchol_declines": 0, "t_pchol": 0.0,
                      "pchol_rank_deficient": 0, "pchol_defic_max": 0,
                      "pchol_decline_cut": 0,
                      "step_f2_checked": 0, "step_f2_over_tol": 0,
                      "step_f2_fired": 0, "step_f2_repaired": 0,
                      "step_f2_refine": 0, "step_f2_dense": 0, "t_f2": 0.0,
                      "support_changes": []}
        self.decline_reasons = {}
        parts = set(str(mode).split("+")) if mode else set()
        fast = str(mode).startswith("fast")
        if unit_core is None:
            unit_core = "unitcore" in parts
        self.use_unit_core = bool(unit_core) and fast
        self.use_qr = bool(qr_update) and fast
        # ---- S1: Tikhonov-Woodbury.  MODEL-LEVEL applicability gate on the
        # unit-row fraction (see woodbury_face_solver): below it the shift
        # truncates real spectrum and the route over-states ``g``.  The gate
        # is checked here, so an unsuitable model never even builds the
        # classifier and runs the incumbent verbatim.
        self.use_tik = "tik" in parts and fast
        self.tik_unit_fraction = None
        self._tik_cls = None
        self._tik_blocked = False
        if self.use_tik:
            self.tik_unit_fraction = wfs.unit_row_fraction(self.B)
            if self.tik_unit_fraction < wfs.TIK_MIN_UNIT:
                self.use_tik = False
                self._decline_init = "tik unit fraction"
            else:
                try:
                    self._tik_cls = wfs.FaceClassifier(self.B)
                except (ValueError, MemoryError):
                    self.use_tik = False
        # ---- S2: pivoted Cholesky of H.  Size guard mirrors
        # FAST_QR_DENSE_LIMIT (never form a dense m x m H on the two models
        # the lane cannot serve), plus screen-and-retire WITH HYSTERESIS.
        self.use_pchol = "pchol" in parts and fast
        self._pchol_retired = (not self.use_pchol) or self.m > wfs.PCHOL_MAXM
        self._pchol_screening = True
        self._pchol_attempts = 0
        self._pchol_declines = 0
        if self.use_pchol and self.m > wfs.PCHOL_MAXM:
            self.stats["pchol_retired_size"] = 1
        # ---- F2: incumbent-step residual check + exact-split repair.  The
        # module constants are read HERE, not baked into the signature, so a
        # harness flag that flips ``F2_REPAIR`` takes effect.
        self.f2_repair = bool(F2_REPAIR if f2_repair is None else f2_repair)
        self.f2_tol = float(F2_STEP_TOL if f2_tol is None else f2_tol)
        self.f2_dense = bool(F2_DENSE_FALLBACK)
        self._f2_strikes = 0
        # Can ANY fast route run on this model?  If not, ``step`` is the
        # incumbent verbatim and should not pay even the support-delta
        # bookkeeping.  (That bookkeeping is not what makes the sub-second
        # models look slower: kb2 measured 0.139 s and 0.111 s in two runs
        # with identical routing, i.e. the spread there is the run-to-run
        # wobble the oracle report already documents, not overhead.)
        self._active = ((self.use_unit_core
                         or (self.use_qr and self.m <= FAST_QR_DENSE_LIMIT)
                         or self.use_tik
                         or (self.use_pchol and not self._pchol_retired))
                        and self.m >= FAST_MIN_COLUMNS)
        self._uc = None
        self._uc_strikes = 0
        self._qr_strikes = 0
        self._qr = {}
        self._cadence = int(cadence)
        self._max_changes = int(max_changes)
        self._diag_floor = float(diag_floor)
        self._previous = None
        self._previous_mask = None       # b1_churn_mask; see CHURN_MASK
        self._pad_cache = {}
        if self.use_unit_core:
            try:
                self._uc = UnitCoreFaceSolver(self.B, np.asarray(b, float),
                                              np.asarray(d, float))
            except (ValueError, MemoryError):
                self.use_unit_core = False

    # ------------------------------------------------------------ helpers
    def _decline(self, code):
        self.decline_reasons[code] = self.decline_reasons.get(code, 0) + 1

    def _residual(self, S, delta, r_null, grad, gnorm):
        """``||B_S'(B_S delta) + r_null - grad||_inf / ||grad||_inf``.

        Computed with the ORIGINAL sparse operator and a mask instead of a
        row slice, so it costs two matrix-vector products and no copy.
        """
        started = time.perf_counter()
        p = self.Bs @ delta
        p = np.where(S, p, 0.0)
        out = float(np.abs(self.BsT @ p + r_null - grad).max()) / max(
            gnorm, 1e-300)
        self.stats["t_residual"] += time.perf_counter() - started
        return out

    # ------------------------------------------------- route 1: unit core
    def _unit_core(self, idx, grad, gnorm):
        uc, m = self._uc, self.m
        kinds = uc.kind[idx]
        pos_unit = np.where(kinds == 1)[0]
        pos_core = np.where(kinds == 2)[0]
        ju = uc.unit_col[idx[pos_unit]]
        su = uc.unit_val[idx[pos_unit]]
        A = (uc.A_core[uc.core_pos[idx[pos_core]]] if pos_core.size
             else sp.csr_matrix((0, m)))
        k = A.shape[0]
        D = (np.bincount(ju, weights=su * su, minlength=m) if ju.size
             else np.zeros(m))
        # STRUCTURAL GUARD, before SuperLU sees the matrix.  Column ``j`` of
        # the bordered ``K`` is ``[D_jj; A[:,j]]``; if no unit row of the face
        # lands in column ``j`` AND no core row of the face touches it, that
        # column is IDENTICALLY ZERO, ``K`` is structurally singular, and
        # ``B_S`` has a zero column so this route was never applicable.
        covered = D > 0.0
        if k:
            covered = covered | (A.getnnz(axis=0) > 0)
        if not covered.all():
            self.stats["splu_structural_veto"] += 1
            raise ArithmeticError("structurally singular (%d empty columns)"
                                  % int((~covered).sum()))
        if k:
            K = sp.bmat([[sp.diags(D), A.T],
                         [A, -sp.eye(k, format="csr")]], format="csc")
        else:
            K = sp.diags(D).tocsc()
        K = K.tocsc()
        # Full structural + numerical validation of what SuperLU is handed.
        # This does NOT make ``splu`` safe (see the class docstring): the
        # crash observed on ship04s replays cleanly on the captured matrix, so
        # it is process-state dependent and no input-level check can exclude
        # it.  The check is kept because it removes one whole class of bad
        # inputs cheaply and its veto count is worth reporting.
        columns = np.diff(K.indptr)
        rows = np.bincount(K.indices, minlength=K.shape[0])
        if (not columns.size or columns.min() == 0 or rows.min() == 0
                or not np.all(np.isfinite(K.data))
                or K.indices.size and (int(K.indices.min()) < 0
                                       or int(K.indices.max())
                                       >= K.shape[0])):
            self.stats["splu_structural_veto"] += 1
            raise ArithmeticError("bordered matrix failed validation")
        lu = spla.splu(K)
        pad = self._pad_cache.get(k)
        if pad is None:
            pad = self._pad_cache[k] = np.zeros(k)

        def solve_G(r):
            return lu.solve(np.concatenate((r, pad)))[:m] if k else lu.solve(r)

        def apply_G(u):
            out = D * u
            return out + (A.T @ (A @ u)) if k else out

        cond_G = uc._condition_estimate(apply_G, solve_G, m)
        if not np.isfinite(cond_G) or cond_G > uc.max_cond_G:
            raise ArithmeticError("cond(G) %.2e" % cond_G)
        delta = solve_G(grad)
        for _ in range(FAST_UC_REFINE):
            correction = grad - apply_G(delta)
            if float(np.abs(correction).max()) <= 1e-15 * gnorm:
                break
            delta = delta + solve_G(correction)
        if not np.all(np.isfinite(delta)):
            raise ArithmeticError("nonfinite")
        return delta, np.zeros(m)

    # ------------------------------------------ route 0a (S1): Tikhonov core
    def _tik_step(self, S, grad, gnorm):
        """``delta = H^+ grad``, ``r_null = P_null(B_S) grad`` from
        ``H_e = (D + e) + A'A`` and ONE ``k x k`` Cholesky (``k`` = core rows
        of the face).  No splu, no dense factorization of size ``m``.

        Runs the ``wfs.TIK_LADDER`` of solver settings (see the rationale
        there): stage 0 is the previous single configuration; a later stage
        runs only when the earlier one misses the honest residual on the
        ORIGINAL operator.  Returns ``(delta, r_null, residual)`` of the
        first PASSING stage, else of the best attempt (which the caller then
        declines on the same residual test as before).  The shift is removed
        exactly at every stage, so each candidate is ``H^+ grad`` plus an
        orthogonal residual for its own cut -- never a regularised step --
        and a failing candidate is discarded, so every failure mode is still
        a fallback (false reject only)."""
        best = None
        for stage, (eps_rel, deflate, refine) in enumerate(wfs.TIK_LADDER):
            try:
                face = wfs.TikhonovFace(self._tik_cls, S, eps_rel=eps_rel,
                                        deflate=deflate)
                delta, r_null = face.pinv(grad, refine=refine)
                if not np.all(np.isfinite(delta)):
                    raise ArithmeticError("nonfinite")
                residual = self._residual(S, delta, r_null, grad, gnorm)
            except (ArithmeticError, ValueError, MemoryError,
                    np.linalg.LinAlgError):
                continue
            if residual <= self.tol:
                self.stats["tik_stage_hist"][stage] += 1
                return delta, r_null, residual
            if best is None or residual < best[2]:
                best = (delta, r_null, residual)
        if best is None:
            raise ArithmeticError("all tik ladder stages failed")
        return best

    def _tik_admit(self):
        """S1 admission: bounded wasted-time share, not consecutive strikes.

        Measured on fit1d (scratchpad f1d_capture/f1d_analyze): the
        FAST_STRIKES rule retired the route for good at call 67/330 after one
        hard stretch, forfeiting 263 calls of which 116+ would have passed --
        each paid a ~20 ms twin-LSQR call where a ~2-4 ms tik step was
        available.  Attempts are free-ish on exactly the models the unit-
        fraction gate admits, so the bound is on wasted SECONDS relative to
        productive step seconds, with a small always-attempt grace window.
        The route re-admits itself when incumbent time accumulates."""
        if self.stats["tik_attempts"] < wfs.TIK_MIN_ATTEMPTS:
            return True
        productive = (self.stats["t_svd"]
                      + self.stats["t_tik"] - self.stats["tik_wasted"])
        return self.stats["tik_wasted"] <= wfs.TIK_TIME_SHARE * productive

    # ----------------------------------------- route 0b (S2): pivoted Chol.
    def _pchol_step(self, S, grad):
        """Exact pinv step from one rank-revealing pivoted Cholesky of ``H``.

        The CUT-CLEANLINESS gate is applied only while screening: if the
        smallest kept pivot sits on LAPACK's own floor the rank decision is a
        guess, and the route declines.  Once the screening window closes the
        gate is switched off (or the route retires); see ``_pchol_verdict``.
        """
        BS = self.Bs[S]
        H = np.asfortranarray((BS.T @ BS).toarray())
        delta, r_null, rank, clean = wfs.piv_chol_step(H, grad, self.m)
        if self._pchol_screening and clean < wfs.PCHOL_CLEAN_GAP:
            self._pchol_declines += 1
            self.stats["pchol_decline_cut"] += 1
            raise np.linalg.LinAlgError("rank cut not resolved")
        if rank < self.m:
            self.stats["pchol_rank_deficient"] += 1
            self.stats["pchol_defic_max"] = max(
                self.stats["pchol_defic_max"], int(self.m - rank))
        return delta, r_null

    def _pchol_verdict(self):
        """Close (or extend) the screening window.  HYSTERESIS: retire only on
        a clear majority of declines, keep only on a clear minority, and when
        the measured rate lands inside the band EXTEND the window rather than
        decide on a coin flip.  (The source lens measured boeing2 flapping
        9/20 vs 10/20 declines across two runs of the same panel, i.e. right
        on a single 50 % threshold.)"""
        n = self._pchol_attempts
        if n < wfs.PCHOL_SCREEN:
            return
        rate = self._pchol_declines / float(n)
        if rate >= wfs.PCHOL_RETIRE_HI:
            self._pchol_screening = False
            self._pchol_retired = True
            self.stats["pchol_retired_screen"] = 1
        elif rate <= wfs.PCHOL_RETIRE_LO:
            self._pchol_screening = False
        elif n >= wfs.PCHOL_SCREEN_MAX:
            self._pchol_screening = False
            if rate >= 0.5:
                self._pchol_retired = True
                self.stats["pchol_retired_screen"] = 1
        self.stats["pchol_screen_attempts"] = int(n)
        self.stats["pchol_screen_declines"] = int(self._pchol_declines)

    # --------------------------------------------------- F2: step repair
    def _f2_exact_repair(self, S, grad, gnorm, delta, r_null, residual,
                         iter_lim=0):
        """Exact-split second opinion on an incumbent step that missed ``tol``.

        Returns ``(delta, r_null, residual, repaired)``.  The repaired step is
        adopted ONLY when its honest residual on the ORIGINAL operator is
        strictly smaller, so this can never degrade the direction handed to
        the exact line search.
        """
        started = time.perf_counter()
        try:
            BS = self.Bs[S]
            # (1) REFINE, cheaply.  The honest residual is
            # ``B_S'(B_S delta) - u`` with ``u = grad - r_null``: ``r_null``
            # is exactly the part of ``grad`` the step cannot reach and is not
            # in question, so the whole error lives in ``delta``.  Refining it
            # with the SAME twin-LSQR composition the incumbent uses
            # (``pinv(H) rho = pinv(B_S) pinv(B_S') rho``) improves the
            # arithmetic without moving the rank cut.
            u = grad - r_null
            d2, r2 = delta, r_null
            res2 = residual
            for _ in range(F2_LSQR_ROUNDS):
                rho = u - self.BsT @ np.where(S, self.Bs @ d2, 0.0)
                w = lsqr(BS.T, rho, atol=LSQR_ATOL, btol=LSQR_ATOL,
                         iter_lim=iter_lim)[0]
                d2 = d2 + lsqr(BS, w, atol=LSQR_ATOL, btol=LSQR_ATOL,
                               iter_lim=iter_lim)[0]
                res2 = self._residual(S, d2, r2, grad, gnorm)
                if res2 <= self.f2_tol:
                    break
            self.stats["step_f2_refine"] += 1
            if res2 > self.f2_tol and self.f2_dense:
                # (2) the incumbent's OWN dense route, forced: same rank cut,
                # exact orthogonal split, only the arithmetic changes
                d3, r3, _method = _pinv_step(BS, grad, iter_lim=0,
                                             dense_limit=max(BS.shape) + 1)
                res3 = self._residual(S, d3, r3, grad, gnorm)
                self.stats["step_f2_dense"] += 1
                if res3 < res2:
                    d2, r2, res2 = d3, r3, res3
        except (ArithmeticError, ValueError, MemoryError,
                np.linalg.LinAlgError) as error:
            self._decline("f2 " + type(error).__name__)
            self.stats["t_f2"] += time.perf_counter() - started
            self._f2_strikes += 1
            return delta, r_null, residual, False
        self.stats["t_f2"] += time.perf_counter() - started
        if res2 < residual:
            self._f2_strikes = 0
            self.stats["step_f2_repaired"] += 1
            self.stats["f2_resid_before_max"] = max(
                self.stats.get("f2_resid_before_max", 0.0), float(residual))
            self.stats["f2_resid_after_max"] = max(
                self.stats.get("f2_resid_after_max", 0.0), float(res2))
            return d2, r2, res2, True
        self._f2_strikes += 1
        self._decline("f2 no improvement")
        return delta, r_null, residual, False

    # ------------------------------------------------------------- driver
    def step(self, S, idx, grad, gnorm, iter_lim):
        self.stats["step_calls"] += 1
        if self._active:
            if CHURN_MASK:
                # DIAGNOSTIC ONLY; see CHURN_MASK for why the value is exact.
                # ``S`` is freshly built by the caller on every iteration
                # (``S = v > 0.0``), so holding the reference is safe.  The
                # shape test costs a tuple compare and makes a mask of changed
                # length skip the count instead of raising -- it cannot happen
                # on the shipped call paths, where a new kernel is built for a
                # new row set.
                prev = self._previous_mask
                if prev is not None and prev.shape == S.shape:
                    self.stats["support_changes"].append(
                        int(np.count_nonzero(S != prev)))
                self._previous_mask = S
            else:
                if self._previous is not None:
                    moved = np.setdiff1d(idx, self._previous,
                                         assume_unique=True).size
                    moved += np.setdiff1d(self._previous, idx,
                                          assume_unique=True).size
                    self.stats["support_changes"].append(int(moved))
                self._previous = idx

        if self._active:
            # ---- S2 first where it applies (m <= PCHOL_MAXM): measured
            # 2.13x on the step bucket over the shipped ``fast`` kernel on the
            # 15 models of that cohort.  S1 then takes the unit-heavy large
            # models S2's size guard excludes -- the two tile.
            if self.use_pchol and not self._pchol_retired:
                # DRY-RUN SCREEN.  While screening, the route is evaluated but
                # its answer is NOT used: a route that will retire must not
                # have diverted the trajectory on its way out.  (Measured:
                # with the screen "live", lotfi accepted ONE piv-chol step
                # inside the window before retiring 19/20 and went 253 -> 476
                # Newton iterations, 2.15 s -> 3.32 s.  The same one-accepted-
                # step hazard the fast QR kernel documents on fit1d.)  The
                # window costs 20 factorizations and buys a decision that is
                # free of that hazard.
                screening = self._pchol_screening
                started = time.perf_counter()
                if screening:
                    self._pchol_attempts += 1
                try:
                    delta, r_null = self._pchol_step(S, grad)
                    self.stats["t_pchol"] += time.perf_counter() - started
                    started = None
                    residual = self._residual(S, delta, r_null, grad, gnorm)
                    if residual <= self.tol:
                        self._pchol_verdict()
                        if not screening:
                            self.stats["step_pchol"] += 1
                            return delta, r_null, "piv-chol", residual
                        self.stats["pchol_screen_ok"] = (
                            self.stats.get("pchol_screen_ok", 0) + 1)
                    else:
                        self._decline("pchol residual")
                        if screening:
                            self._pchol_declines += 1
                        self.stats["pchol_declines"] += 1
                        self._pchol_verdict()
                except (ArithmeticError, ValueError, MemoryError,
                        np.linalg.LinAlgError) as error:
                    self._decline("pchol " + type(error).__name__)
                    if started is not None:
                        self.stats["t_pchol"] += (time.perf_counter()
                                                  - started)
                    self.stats["pchol_declines"] += 1
                    self._pchol_verdict()

            if self._tik_cls is not None and self._tik_admit():
                if self._tik_blocked:
                    # coming back after the wasted-time bound had the route
                    # paused -- the recovery FAST_STRIKES could never make
                    # (fit1d died for good at call 67/330 under it)
                    self._tik_blocked = False
                    self.stats["tik_readmits"] += 1
                self.stats["tik_attempts"] += 1
                started = time.perf_counter()
                accepted = False
                declined_on = "tik residual"
                try:
                    delta, r_null, residual = self._tik_step(S, grad, gnorm)
                    accepted = residual <= self.tol
                except (ArithmeticError, ValueError, MemoryError,
                        np.linalg.LinAlgError) as error:
                    declined_on = "tik " + type(error).__name__
                wall = time.perf_counter() - started
                self.stats["t_tik"] += wall
                if accepted:
                    self.stats["step_tik"] += 1
                    return delta, r_null, "tikhonov", residual
                # every path that reaches here declined: its whole wall is
                # wasted time, the quantity the admission policy bounds
                self._decline(declined_on)
                self.stats["tik_wasted"] += wall
                self.stats["tik_declines"] += 1
            elif self._tik_cls is not None:
                self._tik_blocked = True

            if (self.use_unit_core and self._uc is not None
                    and idx.size >= self.m
                    and self._uc_strikes < FAST_STRIKES):
                started = time.perf_counter()
                try:
                    delta, r_null = self._unit_core(idx, grad, gnorm)
                    self.stats["t_unit_core"] += time.perf_counter() - started
                    started = None
                    residual = self._residual(S, delta, r_null, grad, gnorm)
                    if residual <= self.tol:
                        self._uc_strikes = 0
                        self.stats["step_unit_core"] += 1
                        return delta, r_null, "unit-core", residual
                    self._decline("uc residual")
                except (ArithmeticError, RuntimeError, ValueError,
                        MemoryError, np.linalg.LinAlgError) as error:
                    self._decline("uc " + str(error).split("(")[0][:20])
                    if started is not None:
                        self.stats["t_unit_core"] += (time.perf_counter()
                                                      - started)
                self.stats["uc_declines"] += 1
                self._uc_strikes += 1

            if (self.use_qr and self._qr_strikes < FAST_STRIKES
                    and self.m <= FAST_QR_DENSE_LIMIT):
                which = "row" if idx.size >= self.m else "col"
                factor = self._qr.get(which)
                if factor is None:
                    factor = self._qr[which] = _TallQR(
                        self.B, which, cadence=self._cadence,
                        max_changes=self._max_changes,
                        diag_floor=self._diag_floor, stats=self.stats)
                try:
                    started = time.perf_counter()
                    fresh = factor.sync(idx)
                    mark = time.perf_counter()
                    self.stats["t_qr_sync"] += mark - started
                    outcome = factor.step(grad)
                    self.stats["t_qr_solve"] += time.perf_counter() - mark
                    if outcome is not None:
                        delta, r_null, deficient = outcome
                        residual = self._residual(S, delta, r_null, grad,
                                                  gnorm)
                        if residual > self.tol and not fresh:
                            # an UPDATED factor answered outside budget: take a
                            # fresh one and re-solve before believing anything
                            factor.refactor(idx, "fast_qr_refactor_residual")
                            outcome = factor.step(grad)
                            if outcome is not None:
                                delta, r_null, deficient = outcome
                                residual = self._residual(S, delta, r_null,
                                                          grad, gnorm)
                            else:
                                residual = np.inf
                        if residual <= self.tol:
                            self._qr_strikes = 0
                            self.stats["step_qr_update"] += 1
                            self.stats["step_qr_deficient"] += int(deficient)
                            return delta, r_null, "qr-update", residual
                        self._decline("qr residual")
                    else:
                        self._decline("qr zero rank")
                except (ValueError, ZeroDivisionError, MemoryError,
                        np.linalg.LinAlgError) as error:
                    self._decline("qr " + type(error).__name__)
                self.stats["qr_declines"] += 1
                self._qr_strikes += 1

        started = time.perf_counter()
        BS = self.Bs[S]
        delta, r_null, method = _pinv_step(BS, grad, iter_lim)
        residual = float(np.abs(BS.T @ (BS @ delta) + r_null - grad).max()
                         ) / max(gnorm, 1e-300)
        self.stats["step_svd"] += 1
        self.stats["t_svd"] += time.perf_counter() - started
        # F2: the incumbent route is the one path whose honest residual was
        # measured and then used regardless of what it said.  Check it always
        # (counted per route), repair only the INEXACT one.
        self.stats["step_f2_checked"] += 1
        key = "f2_resid_max_" + method
        self.stats[key] = max(self.stats.get(key, 0.0), float(residual))
        self.stats["f2_incumbent_resid_max"] = max(
            self.stats.get("f2_incumbent_resid_max", 0.0), float(residual))
        if residual > self.f2_tol:
            self.stats["step_f2_over_tol"] += 1
        if (self.f2_repair and method == "lsqr" and residual > self.f2_tol
                and self.m <= F2_REPAIR_MAXM
                and self._f2_strikes < FAST_STRIKES
                and self.stats["t_f2"] <= F2_TIME_SHARE
                * self.stats["t_svd"]):
            self.stats["step_f2_fired"] += 1
            delta, r_null, residual, repaired = self._f2_exact_repair(
                S, grad, gnorm, delta, r_null, residual, iter_lim)
            if repaired:
                method = method + "+f2"
        return delta, r_null, method, residual

    # -------------------------------------------------------- diagnostics
    def counters(self):
        changes = np.asarray(self.stats["support_changes"], dtype=float)
        out = {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in self.stats.items() if k != "support_changes"}
        out["step_routes"] = {"piv-chol": self.stats["step_pchol"],
                              "tikhonov": self.stats["step_tik"],
                              "unit-core": self.stats["step_unit_core"],
                              "qr-update": self.stats["step_qr_update"],
                              "svd": self.stats["step_svd"]}
        out["decline_reasons"] = dict(self.decline_reasons)
        out["tik"] = {"enabled": bool(self._tik_cls is not None),
                      "unit_fraction": self.tik_unit_fraction,
                      "threshold": wfs.TIK_MIN_UNIT,
                      "attempts": self.stats["tik_attempts"],
                      "accepted": self.stats["step_tik"],
                      "declines": self.stats["tik_declines"],
                      "stage_hist": list(self.stats["tik_stage_hist"]),
                      "readmits": self.stats["tik_readmits"],
                      "wasted_seconds": round(self.stats["tik_wasted"], 4),
                      "time_share": wfs.TIK_TIME_SHARE}
        out["pchol"] = {"enabled": bool(self.use_pchol),
                        "retired": bool(self._pchol_retired),
                        "screening": bool(self._pchol_screening),
                        "attempts": int(self._pchol_attempts),
                        "declines": int(self._pchol_declines)}
        out["f2"] = {"enabled": bool(self.f2_repair), "tol": self.f2_tol,
                     "checked": self.stats["step_f2_checked"],
                     "over_tol": self.stats["step_f2_over_tol"],
                     "fired": self.stats["step_f2_fired"],
                     "repaired": self.stats["step_f2_repaired"],
                     "incumbent_resid_max": self.stats.get(
                         "f2_incumbent_resid_max"),
                     "resid_max_lsqr": self.stats.get("f2_resid_max_lsqr"),
                     "resid_max_dense_svd": self.stats.get(
                         "f2_resid_max_dense-svd"),
                     "resid_after_max": self.stats.get("f2_resid_after_max"),
                     "seconds": round(self.stats["t_f2"], 4)}
        if changes.size:
            out["support_change"] = {
                "n": int(changes.size),
                "median": float(np.median(changes)),
                "mean": round(float(changes.mean()), 2),
                "p90": float(np.percentile(changes, 90)),
                "max": int(changes.max()),
                "frac_le_2": round(float((changes <= 2).mean()), 3),
                "frac_le_max_changes": round(
                    float((changes <= self._max_changes).mean()), 3)}
        return out


# ------------------------------------------------------------ the Newton loop

def _as_sparse(B):
    if sp.issparse(B):
        return B.tocsr()
    return sp.csr_matrix(np.asarray(B, dtype=float))


def _transpose_csr(Bs):
    """An explicit CSR of ``B'`` for the loop's transpose products.

    ``Bs.T`` is a CSC VIEW that shares ``Bs``'s arrays, so ``Bs.T @ w``
    already costs O(nnz) -- this is a locality change, not an asymptotic
    one.  It is also EXACT: for sorted indices a CSC mat-vec accumulates
    ``out[i]`` over columns in increasing column order, which is the same
    order the CSR row loop uses, so the two products agree bit for bit and
    no trajectory can move.  (Measured on this panel's matrices the saving
    is 10-15 % of a product that is itself ~0.6 % of the Newton wall -- see
    the sparse-B report.  The lane exists because it was asked for and
    because it is free, not because it is material.)
    """
    return Bs.T.tocsr()


def _pinv_step(BS, grad, iter_lim, dense_limit=DENSE_LIMIT):
    """delta = pinv(B_S' B_S) grad, and the unreachable part of ``grad``.

    Returns ``(delta, r_null, method)``.  ``r_null = grad - P_{range(B_S')}
    grad`` lies in ``null(B_S)``.

    Two implementations, chosen by size (both compute the SAME quantity; the
    dense one is exact up to LAPACK, the sparse one up to LSQR's tolerance,
    and the caller measures the residual either way):

    * dense: ONE thin SVD of ``B_S``.  ``delta = V (V'grad / s^2)``,
      ``r_null = grad - V V' grad``, same rank cutoff as
      ``lstsq(rcond=None)`` and as ``exp23.piece_y``.
    * sparse: two LSQR solves, ``z = pinv(B_S') grad`` then
      ``delta = pinv(B_S) z``, which composes to ``pinv(H) grad`` without ever
      forming ``H`` (whose condition number is squared).
    """
    k, m = BS.shape
    if min(k, m) <= dense_limit:
        A = BS.toarray() if sp.issparse(BS) else np.asarray(BS)
        _U, s, Vt = np.linalg.svd(A, full_matrices=False)
        if s.size:
            keep = s > s[0] * max(A.shape) * np.finfo(float).eps
            Vr, sr = Vt[keep], s[keep]
        else:
            Vr, sr = Vt, s
        if sr.size:
            proj = Vr @ grad
            return Vr.T @ (proj / sr ** 2), grad - Vr.T @ proj, "dense-svd"
        return np.zeros(m), grad.copy(), "dense-svd"
    z = lsqr(BS.T, grad, atol=LSQR_ATOL, btol=LSQR_ATOL, iter_lim=iter_lim)[0]
    r_null = grad - BS.T @ z
    delta = lsqr(BS, z, atol=LSQR_ATOL, btol=LSQR_ATOL, iter_lim=iter_lim)[0]
    return delta, r_null, "lsqr"


def _exact_line_search(K, v, p, s_cap=1e30):
    """argmax_{s>=0} phi(s) = const + s*K - 0.5||(v + s p)_+||^2, exactly.

    ``K = c'delta`` and ``p = B delta``.  ``phi`` is concave piecewise
    quadratic; ``phi'(s) = K - sum_{i in A(s)} p_i (v_i + s p_i)`` is
    continuous, piecewise linear and nonincreasing, so the maximizer is the
    first sign change of ``phi'``.  Walk the breakpoints ``s_i = -v_i/p_i`` in
    order, maintaining ``sum p_i v_i`` and ``sum p_i^2`` over the active set.
    Returns ``inf`` if ``phi`` is unbounded on the ray (that means the
    projection problem is infeasible -- the caller treats it as failure).
    """
    active = v > 0.0
    acc1 = float(p[active] @ v[active]) if active.any() else 0.0
    acc2 = float(p[active] @ p[active]) if active.any() else 0.0
    nonzero = p != 0.0
    s_event = np.full(v.shape, np.inf)
    s_event[nonzero] = -v[nonzero] / p[nonzero]
    # a row is an EVENT only if it actually changes membership for s >= 0:
    # p_i > 0 rows enter at s_i; p_i < 0 rows leave, but only if they were
    # active at s = 0 (v_i > 0).
    valid = nonzero & np.isfinite(s_event) & (s_event >= 0.0) & (
        (p > 0.0) | (v > 0.0))
    idx = np.where(valid)[0]
    order = idx[np.argsort(s_event[idx], kind="stable")]
    s_cur = 0.0
    for i in order:
        s_next = float(s_event[i])
        if s_next > s_cur:
            if acc2 > 0.0:
                s_root = (K - acc1) / acc2
                if s_root <= s_cur:
                    return s_cur
                if s_root < s_next:
                    return s_root
            elif K - acc1 <= 0.0:
                return s_cur
            s_cur = s_next
        pi, vi = float(p[i]), float(v[i])
        if pi > 0.0:
            acc1 += pi * vi
            acc2 += pi * pi
        else:
            acc1 -= pi * vi
            acc2 -= pi * pi
    if acc2 > 0.0:
        return max((K - acc1) / acc2, s_cur)
    if K - acc1 <= 0.0:
        return s_cur
    return float("inf") if s_cur < s_cap else s_cur


def newton_projection(B, b, d, t, x0=None, scaled=True, maxiter=MAXITER,
                      grad_tol=GRAD_TOL, Bs=None, deadline=None, kernel=None,
                      BsT=None):
    """Semismooth Newton maximization of ``theta_t``.  Returns a dict.

    ``x`` in the returned dict is always in UNSCALED units (the multiplier of
    ``y = (t b + B x)_+``); ``x_work`` is the working variable of whichever
    formulation ran (``x_work = x/t`` when ``scaled``).  ``y`` is the projection
    point in original units, so ``B' y ~ d`` is the meaningful residual and is
    reported as ``dres``.

    ``converged=False`` (iteration cap, line-search stall, deadline) is a
    FAILURE the caller must treat as "no hint", never as a result.
    """
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    Bs = _as_sparse(B) if Bs is None else Bs
    # ``BsT`` (opt-in) is an explicit CSR of B'; ``Bs.T`` is the CSC view
    # this loop has always used.  Bit-identical products either way.
    BT = Bs.T if BsT is None else BsT
    n, m = B.shape
    t = float(t)

    # unified driver: theta(xv) = c'xv - 0.5||(bb + B xv)_+||^2
    if scaled:
        c, bb = d / t, b
    else:
        c, bb = d, t * b
    cscale = max(float(np.abs(c).max()) if c.size else 1.0, 1e-300)
    dscale = max(1.0, float(np.abs(d).max()) if d.size else 1.0)

    xv = np.zeros(m) if x0 is None else np.asarray(x0, dtype=float).copy()
    iter_lim = 4 * max(n, m)

    stats = {"iterations": 0, "newton_steps": 0, "null_steps": 0,
             "max_step_residual": 0.0, "unbounded_ray": 0, "solver": None}
    step_seconds = 0.0
    line_seconds = 0.0
    reason = "max iterations"
    converged = False
    value = float("nan")
    history = []

    for iteration in range(maxiter):
        stats["iterations"] = iteration + 1
        if deadline is not None and time.perf_counter() > deadline:
            reason = "deadline"
            break
        v = bb + Bs @ xv
        w = np.maximum(v, 0.0)
        S = v > 0.0
        grad = c - BT @ w
        gnorm = float(np.abs(grad).max()) if grad.size else 0.0
        value = float(c @ xv - 0.5 * w @ w)
        if gnorm <= grad_tol * cscale:
            converged, reason = True, "gradient"
            break
        # STALL GUARD, on THETA -- which every step increases by construction,
        # unlike |grad|, which oscillates (a |grad|-based guard was tried and
        # fired after 26 iterations at |grad|_rel = 45, wrecking the face).
        # Measured (newt_tol.py): lotfi at t=1 reaches |grad|_rel = 9.5e-7 in
        # 108 iterations and does not improve in the next 2892 -- degeneracy
        # pins it there.  Grinding costs wall and buys no face.
        history.append(value)
        if len(history) > STALL_WINDOW:
            past = history[-STALL_WINDOW - 1]
            if value - past <= STALL_FACTOR * max(1.0, abs(value)):
                reason = "theta stall"
                break
        if not S.any():
            # no curvature at all: theta is linear, the whole gradient is a ray
            delta, r_null = np.zeros(m), grad.copy()
        elif kernel is None:
            _started = time.perf_counter()
            BS = Bs[S]
            delta, r_null, method = _pinv_step(BS, grad, iter_lim)
            stats["solver"] = method
            # H delta must equal the REACHABLE part of grad, grad - r_null
            resid = float(np.abs(BS.T @ (BS @ delta) + r_null - grad).max())
            stats["max_step_residual"] = max(stats["max_step_residual"],
                                             resid / max(gnorm, 1e-300))
            step_seconds += time.perf_counter() - _started
        else:
            _started = time.perf_counter()
            delta, r_null, method, resid = kernel.step(
                S, np.where(S)[0], grad, gnorm, iter_lim)
            stats["solver"] = method
            stats["max_step_residual"] = max(stats["max_step_residual"], resid)
            step_seconds += time.perf_counter() - _started
        _started = time.perf_counter()
        moved = False
        # TWO exact line searches per iteration: the Newton direction, then
        # the null(B_S) ray the Newton step provably cannot reach.
        for kind, direction in (("newton", delta), ("null", r_null)):
            if not np.any(direction):
                continue
            # phi'(0) = grad'direction; the line search itself needs K = c'dir
            if float(grad @ direction) <= 0.0:
                continue
            p = Bs @ direction
            step = _exact_line_search(float(c @ direction), v, p)
            if not np.isfinite(step):
                stats["unbounded_ray"] += 1
                reason = "unbounded ray (projection infeasible?)"
                moved = False
                break
            if step <= 0.0:
                continue
            xv = xv + step * direction
            stats["newton_steps" if kind == "newton" else "null_steps"] += 1
            moved = True
            if kind == "newton":                # refresh for the null step
                v = bb + Bs @ xv
                w = np.maximum(v, 0.0)
                grad = c - BT @ w
        line_seconds += time.perf_counter() - _started
        if not moved:
            reason = ("unbounded ray (projection infeasible?)"
                      if stats["unbounded_ray"] else "no ascent direction")
            break

    v = bb + Bs @ xv
    w = np.maximum(v, 0.0)
    S = v > 0.0
    grad = c - BT @ w
    gnorm = float(np.abs(grad).max()) if grad.size else 0.0
    y = t * w if scaled else w
    x_true = t * xv if scaled else xv
    dres = float(np.abs(BT @ y - d).max()) / dscale
    return {
        "x": x_true, "x_work": xv, "y": y, "S": S, "t": t, "scaled": bool(scaled),
        "converged": bool(converged), "reason": reason,
        "iterations": int(stats["iterations"]),
        "grad_norm": gnorm, "grad_relative": gnorm / cscale,
        "dres": dres, "theta": float(value), "support": int(S.sum()),
        "newton_steps": int(stats["newton_steps"]),
        "null_steps": int(stats["null_steps"]),
        "unbounded_ray": int(stats["unbounded_ray"]),
        "max_step_residual": float(stats["max_step_residual"]),
        "step_solver": stats["solver"],
        "step_seconds": float(step_seconds),
        "line_search_seconds": float(line_seconds),
    }


# --------------------------------------------------- staged terminal escalation

def repair_rows(B, b, d, W, info, x_tilde=None, y_tilde=None, rounds=3,
                tol=ot.TOL_SIGMA):
    """Add the rows the candidate x violates to the face, and retest.

    ``terminal_test``'s failure NAMES its own repair: if the accepted face's
    primal candidate has ``sigma_j = (Bx)_j - b_j < 0``, row ``j`` belongs in
    the face.  This is the ``blocking_rows`` diagnosis of
    ``oracle_terminal._t_bar`` applied to the primal side.  Pure exp23 algebra
    (one ``piece_y`` per round); the FROZEN gates still decide.

    Returns ``(found_or_None, records)``.
    """
    records = []
    b_scale = max(1.0, float(np.abs(b).max()))
    W = np.asarray(W, dtype=bool).copy()
    for attempt in range(rounds):
        x_star = np.asarray(info.get("x_star"), dtype=float)
        if x_star.size == 0:
            break
        sigma = B @ x_star - b
        add = (~W) & (sigma < -tol * b_scale)
        if not add.any():
            break
        W = W | add
        passed, info = ot.terminal_test(B, b, d, W, x_tilde=x_tilde,
                                        y_tilde=y_tilde)
        record = {"round": attempt, "rows_added": int(add.sum()),
                  "support": int(W.sum()), "terminal": bool(passed),
                  "reason": info.get("reason"),
                  "sigma_min_relative": info.get("sigma_min_relative")}
        if passed or info.get("face_terminal"):
            gate_ok, detail = ot._gate(B, b, d, info)
            record["gate_primal"] = detail.get("primal")
            record["gate_ok"] = bool(gate_ok)
            records.append(record)
            if gate_ok:
                return (info, detail, W), records
            continue
        records.append(record)
    return None, records


def t_bar_of(info):
    """The larger of the two t_bar readings ``terminal_test`` reports."""
    values = [float(info.get("t_bar", 0.0) or 0.0),
              float(info.get("t_bar_candidate", 0.0) or 0.0)]
    return max(values)


def newton_terminal(B, b, d, seconds=ORACLE_SECONDS, scaled=True,
                    t0=1.0, t_max=T_MAX, stage_factor=T_STAGE_FACTOR,
                    max_stages=14, maxiter=MAXITER, thresholds=ot.THRESHOLDS,
                    settle_retry=True, diagnostics=None, kernel="svd",
                    kernel_tol=FAST_STEP_TOL, sparse=False):
    """Staged, warm-started Newton hunt for the terminal face.  Fails closed.

    Stage 1 solves at ``t0``.  Every stage tests its own face with
    ``oracle_terminal.terminal_test`` (unchanged) and, if the face is
    t-free, the FROZEN gates via ``oracle_terminal._gate`` (unchanged).  A
    failed stage yields the closed-form ``t_bar`` of the tested face; the next
    stage re-solves, WARM-STARTED, at ``stage_factor * t_bar`` (or 10x the
    current ``t`` when ``t_bar`` is uninformative).  ``x_work`` is reused
    directly across stages -- in the scaled formulation it is nearly
    ``t``-independent, which is the point of that formulation.

    Returns ``(found_or_None, diag)``.  No external solver is called anywhere
    on this path: no HiGHS, no clarabel, no ``linprog``.
    """
    diag = {} if diagnostics is None else diagnostics
    diag.setdefault("stages", [])
    diag["route"] = None
    diag["backend_family"] = "newton"
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    Bs = _as_sparse(B)
    # ``sparse=True`` (opt-in; default False leaves every product on the CSC
    # view the loop has always used) stores B' explicitly as CSR as well.
    # The Newton loop was ALREADY sparse -- ``_as_sparse`` above predates
    # this lane -- so the only storage left to change is the transpose
    # orientation, and the products are bit-identical (see _transpose_csr).
    BsT = _transpose_csr(Bs) if sparse else None
    diag["sparse"] = bool(sparse)
    diag["sparse_density"] = round(Bs.nnz / float(max(Bs.shape[0]
                                                      * Bs.shape[1], 1)), 6)
    started = time.perf_counter()
    deadline = started + seconds
    # ONE kernel per model: the unit-core row classification and the QR
    # factors are both cross-stage state, and the warm start means consecutive
    # stages continue the same active-set crawl.
    if kernel not in (None, "svd", "fast", "fast+unitcore", "fast+tik",
                      "fast+pchol", "fast+tik+pchol"):
        raise ValueError("kernel must be 'svd', 'fast', 'fast+unitcore', "
                         "'fast+tik', 'fast+pchol' or 'fast+tik+pchol'")
    step_kernel = (None if kernel in (None, "svd")
                   else FaceStepKernel(B, Bs, b, d, mode=kernel,
                                       tol=kernel_tol, BsT=BsT))
    diag["kernel"] = kernel

    total_iters = 0
    newton_seconds = 0.0
    step_seconds = 0.0
    line_seconds = 0.0
    x_work = None
    t = float(t0)
    found = None

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
        n_started = time.perf_counter()
        sol = newton_projection(B, b, d, t, x0=x_work, scaled=scaled,
                                maxiter=maxiter, Bs=Bs, deadline=deadline,
                                kernel=step_kernel, BsT=BsT)
        n_wall = time.perf_counter() - n_started
        newton_seconds += n_wall
        step_seconds += sol["step_seconds"]
        line_seconds += sol["line_search_seconds"]
        total_iters += sol["iterations"]
        record = {"stage": stage, "t": t, "newton_seconds": round(n_wall, 4),
                  "iterations": sol["iterations"],
                  "converged": sol["converged"], "reason": sol["reason"],
                  "support": sol["support"], "dres": sol["dres"],
                  "grad_relative": sol["grad_relative"],
                  "max_step_residual": sol["max_step_residual"],
                  "null_steps": sol["null_steps"],
                  "newton_steps": sol["newton_steps"],
                  "step_solver": sol["step_solver"]}
        if sol["unbounded_ray"]:
            record["outcome"] = "newton failed"
            diag["stages"].append(record)
            diag["reason"] = f"newton {sol['reason']} at t={t:g}"
            break
        # A cap-hit / stalled Newton point is NOT a success, but its face is
        # still just a HINT: testing it costs one piece_y and can only be
        # REJECTED by the frozen gates, never wrongly accepted.  So the face
        # is tested either way and the honest convergence flag is recorded.
        x_work = sol["x_work"]                      # warm start for next stage
        x_tilde = -sol["x"] / t                     # LP primal candidate
        y_tilde = sol["y"]                          # already nonnegative
        W = sol["S"]
        passed, info = ot.terminal_test(B, b, d, W, x_tilde=x_tilde,
                                        y_tilde=y_tilde)
        record.update({k: v for k, v in info.items()
                       if k not in ("x_star", "y_star", "candidates",
                                    "dual_candidates")})
        record["terminal"] = bool(passed)
        violation = None
        if passed or info.get("face_terminal"):
            gate_ok, detail = ot._gate(B, b, d, info)
            record["gate"] = {k: v for k, v in detail.items()}
            if gate_ok:
                record["outcome"] = "certified"
                diag["stages"].append(record)
                found = _finish(info, detail, stage, "newton terminal")
                break
            violation = float(detail.get("primal", 0.0) or 0.0)
            repaired, rep_records = repair_rows(B, b, d, W, info,
                                                x_tilde=x_tilde,
                                                y_tilde=y_tilde)
            if rep_records:
                record["repair"] = rep_records
            if repaired is not None:
                r_info, r_detail, _rW = repaired
                record["outcome"] = "certified (row repair)"
                diag["stages"].append(record)
                found = _finish(r_info, r_detail, stage,
                                "newton terminal + row repair")
                break
        if violation is None:
            sigma_rel = float(info.get("sigma_min_relative", 0.0) or 0.0)
            violation = -sigma_rel if sigma_rel < 0.0 else None
        record["escalation_violation"] = violation
        record["outcome"] = record.get("outcome", "face rejected")
        diag["stages"].append(record)

        # ESCALATION.  The t_bar ratio formula is the principled rule, but it
        # is computed from exp23's MIN-NORM selector, which on rank-deficient
        # faces is wildly primal-infeasible (grow7: sigma_min = -1.3e5), so
        # t_bar comes back ~0.02 and escalates nothing.  Measured instead:
        # the gate's own primal residual decays like C/t (grow7 stages, t 1.4
        # -> 1.4e6: 0.9996, 0.9963, 0.975, 0.760, 0.241, 0.032, 0.0032), so
        # the violation itself predicts the t that would clear TOL_FEAS.  Both
        # rules are used; the larger wins, floored at 4x per stage.
        t_next = max(stage_factor * t_bar_of(info), 4.0 * t)
        if violation and violation > 0.0:
            t_next = max(t_next, t * min(max(violation / 1e-9, 4.0), 1e3))
        if t_next > t_max:
            if t >= t_max:
                diag["reason"] = "t ceiling"
                break
            t_next = t_max
        t = t_next

    diag["newton_seconds"] = float(newton_seconds)
    diag["newton_step_seconds"] = float(step_seconds)
    diag["newton_line_search_seconds"] = float(line_seconds)
    diag["newton_iters_total"] = int(total_iters)
    diag["stages_run"] = len(diag["stages"])
    if step_kernel is not None:
        diag["kernel_counters"] = step_kernel.counters()
    if found is not None:
        diag.setdefault("reason", "certified")
        return found, diag

    # ---- last route: exp23's own settle + threshold sweep on the Newton dual,
    # driven through oracle_terminal.terminal_attempt with the EXTERNAL solver
    # routes disabled (projection_retry=False keeps clarabel out of the path).
    if x_work is not None and time.perf_counter() < deadline:
        oracle = {"x": -np.asarray(x_work, dtype=float),
                  "y": np.asarray(
                      t * np.maximum(b + Bs @ np.asarray(x_work), 0.0)
                      if scaled else
                      np.maximum(t * b + Bs @ np.asarray(x_work), 0.0)),
                  "method": "newton", "seconds": 0.0, "objective": None}
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
        diag["sweep_attempts"] = sweep.get("attempts")
        diag["sweep_reason"] = sweep.get("reason")
        if found is not None:
            diag["route"] = "newton " + str(sweep.get("route"))
            diag["accepted_support"] = sweep.get("accepted_support")
            diag["reason"] = "certified"
            return found, diag

    diag.setdefault("reason", "no terminal face certified")
    return None, diag


# ------------------------------------------------------------------- pipeline

def follow_with_newton(B, b, d, newton_seconds=ORACLE_SECONDS, scaled=True,
                       newton_t0=1.0, newton_maxiter=MAXITER,
                       newton_max_stages=14, newton_settle_retry=True,
                       newton_diagnostics=None, newton_kernel="svd",
                       newton_kernel_tol=FAST_STEP_TOL, newton_sparse=False,
                       **kw):
    """Newton terminal shortcut with the UNCHANGED walker as the fallback.

    On success returns a ``follow``-shaped dict with ``pivots=0`` and
    ``backend="newton oracle terminal pair"``.  On ANY failure the return is
    ``exp23_path_primal_dual.follow(B, b, d, **kw)`` verbatim with the Newton
    counters attached, exactly as ``oracle_terminal.follow_with_oracle`` does.
    """
    diag = {} if newton_diagnostics is None else newton_diagnostics
    started = time.perf_counter()
    found = None
    try:
        found, diag = newton_terminal(
            B, b, d, seconds=newton_seconds, scaled=scaled, t0=newton_t0,
            maxiter=newton_maxiter, max_stages=newton_max_stages,
            settle_retry=newton_settle_retry, diagnostics=diag,
            kernel=newton_kernel, kernel_tol=newton_kernel_tol,
            sparse=newton_sparse)
    except Exception as error:                  # fail closed on ANY exception
        diag["reason"] = f"EXCEPTION {type(error).__name__}: {error}"
    attempt_seconds = time.perf_counter() - started

    counters = {
        "newton_attempt_seconds": float(attempt_seconds),
        "newton_seconds": float(diag.get("newton_seconds", 0.0)),
        "newton_step_seconds": float(diag.get("newton_step_seconds", 0.0)),
        "newton_line_search_seconds": float(
            diag.get("newton_line_search_seconds", 0.0)),
        "newton_iters_total": int(diag.get("newton_iters_total", 0)),
        "newton_stages": int(diag.get("stages_run", 0)),
        "route": diag.get("route"),
        "newton_diagnostics": diag,
    }

    if found is not None:
        y = found["y"]
        st = {
            "status": "CERTIFIED",
            "x": found["x"],
            "y": y,
            "lb": float(b @ y),
            "t": float("inf"),
            "pivots": 0,
            "backend": "newton oracle terminal pair",
            "mono": True,
            "certificate_detail": found["certificate"],
            "newton_fallback": False,
        }
        st.update(counters)
        return st

    counters["route"] = "fallback walk"
    walk_started = time.perf_counter()
    st = exp23.follow(B, b, d, **kw)
    counters["newton_fallback_walk_seconds"] = (time.perf_counter()
                                                - walk_started)
    st = dict(st)
    st["newton_fallback"] = True
    st["newton_fallback_reason"] = diag.get("reason", "unknown")
    st.update(counters)
    return st
