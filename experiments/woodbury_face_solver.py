"""Woodbury face/step kernels: Tikhonov-Woodbury (S1) and pivoted Cholesky (S2).

Both routes compute the SAME quantity the incumbent computes -- an EXACT
pseudoinverse of ``H = B_S' B_S`` for SOME rank cut, split by an EXACT
orthogonal projector -- and both are checked by the caller against the honest
residual on the ORIGINAL sparse operator before their answer is used.  Neither
calls ``splu``; neither forms a dense factorization of a matrix larger than
the one the incumbent already forms.

THE ADMISSION LAW (measured, ``opportunity_degenerate.md`` section 3).  A
replacement Newton step is admissible iff it is an exact pseudoinverse step
for some rank cut, split by an exact orthogonal projector.  The rank cut may
move freely; the exactness of the split may not, because ``theta`` is exactly
linear along a true ``null(B_S)`` ray and those (1e7-1e8 long) traversals are
the convergence engine.  Every inexact-split family measured -- Tikhonov as a
STEP, epsilon-active widening/shrinking, smoothing -- loses the Newton route.

Both routes here obey it:

* ``TikhonovFace`` uses a shift only as a SOLVER for ``H^+``; the shift is
  removed exactly (null deflation before and after refinement, refinement in
  the original sparse rows), so what the caller receives is ``H^+ r`` and the
  orthogonal residual, not a regularised step.
* ``piv_chol_step`` factors ``H = U U'`` with ``U`` of full column rank, so
  ``U G^{-1} U'`` is the exact orthogonal projector onto ``range(H)``.

--------------------------------------------------------------------------
S1 -- TIKHONOV-WOODBURY (``FaceClassifier`` + ``TikhonovFace``)

``H = D + A'A`` where ``D`` collects the UNIT rows of the face (exactly one
nonzero) and ``A`` is its ``k`` CORE rows.  Shift, then remove the shift:

    H_e = H + e I = De + A'A,   De = D + e > 0
    H_e^{-1} = De^{-1} - De^{-1} A' M^{-1} A De^{-1},   M = I_k + A De^{-1} A'

``M`` is ``k x k`` SPD -- the ONLY dense factorization, and its size is the
core-row count of the face, independent of ``m``.  ``e H_e^{-1}`` is the
null-space projector up to ``e / lam_min+`` (squaring under repetition), which
deflates the right-hand side before refinement and the solution after; the
result is the MIN-NORM solution ``H^+ r``.

Applicability is a MODEL property, not a face property: when most columns
carry ``D_j = 0`` the shift truncates real spectrum and ``g`` comes out
over-stated.  ``g = b_W + B_W ua`` is a genuine least-squares residual at a
genuine point, so ``||g_computed|| >= ||g_true||`` unconditionally: the route
can only ever cause a FALSE REJECT, never a false accept.  Every failure mode
is therefore a fallback.  The gate below (unit-row fraction >= 0.70) is
CALIBRATED on 12 models, not derived, and is deliberately conservative.

--------------------------------------------------------------------------
S2 -- PIVOTED CHOLESKY (``piv_chol_step``)

LAPACK ``dpstrf`` gives ``P'HP = L L'`` with ``L`` (m x r) lower trapezoidal
and ``r`` the numerical rank at ``m * eps * max_pivot``.  With ``U = P L``
(full column rank, ``H = U U'`` to 9.9e-17 relative):

    G      = U'U = L11'(I + W'W)L11,   W = L21 L11^{-1},   p = m - r
    r_null = grad - U G^{-1} U' grad        (exact orthogonal projector)
    delta  = U G^{-2} U' grad               ( = pinv(H) grad at rank r)

``(I + W'W)^{-1} = I - W'(I + W W')^{-1} W`` with ``W W'`` only ``p x p`` --
the deficiency-sized Woodbury already banked in ``newton_oracle._TallQR``,
moved from a pivoted QR of ``B_S`` (``O(|S| m^2)``, whose downdates are
unusable) onto a pivoted Cholesky of ``H`` (``O(m^3/3)``, independent of
``|S|``, needing no update at all).

The price is a rank cut at SQUARED precision.  ``clean_ratio`` measures how
far the smallest kept pivot sits above LAPACK's own floor; the caller screens
on it with hysteresis (see ``newton_oracle.FaceStepKernel``).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.linalg import lapack

# ---- router heuristics ONLY.  None of these can make a certificate pass. ----
# S1 gate: unit-row fraction of the MODEL.  Calibrated over 12 netlib models
# (ship04s 0.84 and fit1d 0.99 pass; degen2 0.45 works but is excluded,
# israel 0.48 does not work and is excluded).  Conservative, not sharp.
TIK_MIN_UNIT = 0.70
TIK_EPS_REL = 1e-12         # shift, relative to the face's own scale
TIK_DEFLATE = 2             # null-projector applications (error squares)
TIK_REFINE = 3              # refinement rounds in the ORIGINAL sparse rows

# S1 STEP-ROUTE retry ladder of (eps_rel, deflate, refine) SOLVER settings.
# (The terminal route in oracle_terminal keeps the single default
# configuration; nothing here changes TikhonovFace defaults.)
#
# Why a ladder (measured, scratchpad f1d_ladder.py, 330 captured fit1d step
# calls): the single default config passes the honest residual on 164/330.
# The 166 failures are SOLVER failures of one fixed shift, not rank walls,
# and they come in two mutually incompatible regimes:
#
# * TRUNCATION -- fit1d's unit rows all carry value 1 while its largest core
#   row norm is 9.65e3, so ``eps = 1e-12 * scale ~ 9.3e-5`` sits ON TOP of
#   real face spectrum at 1e-5..1e-3 and the deflation removes genuine range
#   directions.  A smaller shift fixes these (residual 1e-2 -> 1e-8).
# * SHALLOW DEFLATION -- other faces have a comfortable null/range gap but
#   need more null-projector applications; there a SMALLER shift is the wrong
#   move (``M = I + A De^{-1} A'`` loses conditioning as 1/eps) and a deeper
#   deflation at the DEFAULT shift fixes them.
#
# The ladder covers both: 330/330 accepted at 4.2 ms mean per call against
# the incumbent twin-LSQR's 19.8 ms (which itself misses the same 1e-6 bar
# on 9/40 sampled calls).  SAFETY IS UNCHANGED: eps is still removed exactly
# (deflation before/after, refinement in the original sparse rows), every
# stage's answer is judged by the same honest residual on the ORIGINAL
# operator, a failing stage's answer is discarded, and ``g``/steps can only
# be over-stated -- false reject only, never false accept.  The first stage
# is the previous single configuration, so a model whose faces the default
# serves (ship04s: 200/200) is bit-identical and pays nothing extra.
TIK_LADDER = ((TIK_EPS_REL, TIK_DEFLATE, TIK_REFINE),
              (TIK_EPS_REL, 4, TIK_REFINE),
              (1e-15, 2, 8),
              (1e-15, 4, 8))
# S1 step-route admission: bounded WASTED-TIME share instead of retirement on
# FAST_STRIKES consecutive declines.  Measured on fit1d: the consecutive-
# strike rule retired the route at call 67 of 330 after one hard stretch of
# faces (12 straight declines), and 116 of the 263 forfeited calls would have
# passed even at the old single config -- a ~20 ms LSQR call paid where a
# ~2-4 ms accepted tik step was available.  The strike rule was designed for
# routes whose attempt costs what the incumbent costs; the tik attempt is
# ~5x cheaper than the twin-LSQR it replaces on exactly the models the gate
# admits.  Policy: always attempt during a small grace window, then keep
# attempting while cumulative time spent on DECLINED attempts stays below
# TIK_TIME_SHARE of the productive step time (incumbent seconds + accepted-
# tik seconds).  Overhead is bounded by construction:
# <= TIK_MIN_ATTEMPTS ladder attempts + a TIK_TIME_SHARE fraction of the
# step bucket, and the route re-admits itself after a hard stretch instead
# of dying for the rest of the model.
TIK_MIN_ATTEMPTS = 20
TIK_TIME_SHARE = 0.25

# S2 guards.
PCHOL_MAXM = 700            # mirrors newton_oracle.FAST_QR_DENSE_LIMIT
PCHOL_CLEAN_GAP = 1e2       # cut-cleanliness threshold used while screening
# PROMOTED (c5_pchol_screen, agent_reports/25): the screening window was 20.
# The A/B measured c1+c5 at -3.233 s panel over 405 interleaved runs with no
# change to any certificate component, and the composition run that produced
# the 12.219 s benchmark of record carried it.  ROUTING ONLY -- a shorter
# window retires (or keeps) the piv-chol route sooner; every step it produces
# is still judged by the same honest residual, and the hysteresis band
# (PCHOL_RETIRE_HI/LO) and PCHOL_SCREEN_MAX are deliberately unchanged, so a
# decline rate inside the band still EXTENDS the window instead of deciding on
# a coin flip.  ``PCHOL_SCREEN_PRE_PROMOTION`` is what
# ``quarantined_speedups.active(())`` restores for the historical control arm.
PCHOL_SCREEN = 8            # attempts in the screening window
PCHOL_SCREEN_PRE_PROMOTION = 20
# HYSTERESIS on the retirement decision (the source lens measured boeing2
# flapping 9/20 vs 10/20 declines across two runs of the same panel, i.e. the
# threshold sits exactly on the observed rate).  Two-sided band: retire only
# on a clear majority, keep only on a clear minority, and when the rate lands
# inside the band EXTEND the window instead of deciding on a coin flip.
PCHOL_RETIRE_HI = 0.60      # decline fraction at/above which the route retires
PCHOL_RETIRE_LO = 0.40      # decline fraction at/below which it is kept
PCHOL_SCREEN_MAX = 60       # hard cap on the extended screening window

# ---------------------------------------------------------------------------
# PROMOTED ALLOCATION / CONSTANT-CACHE DEFAULTS
#
# Four measured changes that remove MEMORY TRAFFIC only.  None of them changes
# a floating-point operation, an operand order, or the order in which any sum
# is accumulated, and that is not asserted here but MEASURED: see
# ``quarantined_pivchol_alloc.py --differential`` (byte equality of ``delta``,
# ``r_null``, ``rank`` and ``clean`` on 550/550 real captured ``H``, both arms
# computed from copies so buffer alignment applies to both) and
# ``quarantined_speedups.py --self-check`` for the TikhonovFace constant cache.
#
# EXACT-SPLIT LAW UNTOUCHED.  ``rank`` and ``clean`` come out of ``dpstrf``
# unread and unmoved by a1/a2; a3 only replaces ``float`` of an ``np.abs``
# array element by ``abs`` of the same ``float``.  The rank cut and the
# orthogonal split are bit-identical by construction.
#
# Each flag is here so the PRE-PROMOTION body stays reachable: setting it to
# False runs the historical code line for line.  ``quarantined_pivchol_alloc.
# active(names)`` and ``quarantined_speedups.active(names)`` set exactly these
# flags, so every historical A/B still has its true control arm.
TIK_CONST_CACHE = True      # c1: per-model core row-norm cache; lazy AiDe
ALLOC_TRIL_SLICE = True     # a1: np.tril(c)[:, :rank] -> np.tril(c[:, :rank])
ALLOC_U_GATHER = True       # a2: U = zeros; U[P] = L  ->  U = L[inverse perm]
ALLOC_DIAG_SCALAR = True    # a3: np.abs(np.diag(c)[:rank]) -> the two scalars

PROMOTED_FLAGS = ("TIK_CONST_CACHE", "ALLOC_TRIL_SLICE", "ALLOC_U_GATHER",
                  "ALLOC_DIAG_SCALAR")


def unit_row_fraction(B):
    """Fraction of rows of ``B`` with exactly one nonzero."""
    B = np.asarray(B, dtype=float)
    if B.shape[0] == 0:
        return 0.0
    counts = (B != 0.0).sum(axis=1)
    return float((counts == 1).sum()) / float(B.shape[0])


class FaceClassifier:
    """One-time per-model row classification (unit rows vs core rows)."""

    def __init__(self, B):
        self.B = np.asarray(B, dtype=float)
        self.Bs = sp.csr_matrix(self.B)
        self.n, self.m = self.B.shape
        nz = self.B != 0.0
        counts = nz.sum(axis=1)
        self.kind = np.where(counts == 1, 1, np.where(counts == 0, 0, 2))
        self.unit_col = np.full(self.n, -1, dtype=np.intp)
        self.unit_val = np.zeros(self.n)
        urows = np.where(self.kind == 1)[0]
        if urows.size:
            cols = np.argmax(nz[urows], axis=1)
            self.unit_col[urows] = cols
            self.unit_val[urows] = self.B[urows, cols]
        crows = np.where(self.kind == 2)[0]
        self.core_rows = crows
        self.core_pos = np.full(self.n, -1, dtype=np.intp)
        self.core_pos[crows] = np.arange(crows.size)
        self.A_core = (sp.csr_matrix(self.B[crows]) if crows.size
                       else sp.csr_matrix((0, self.m)))
        self.unit_fraction = float(urows.size) / max(1, self.n)
        self._core_rownorms = None

    def core_rownorms(self):
        """Squared row norms of EVERY core row of the model, cached per model.

        A per-MODEL constant: row norms are a property of ``A_core``'s ROWS and
        are invariant under the row gather that builds a face, so a face reads
        a slice instead of recomputing.  BIT-IDENTICAL to the per-face
        recompute: a CSR row gather preserves within-row index order, so the
        per-row products and their summation order are unchanged, and ``max``
        over the same multiset of doubles is the same double.
        """
        if self._core_rownorms is None:
            core = self.A_core
            self._core_rownorms = np.asarray(
                core.multiply(core).sum(axis=1)).ravel()
        return self._core_rownorms


class TikhonovFace:
    """``H^+`` for one face, through ``H_e = De + A'A`` and one k x k Cholesky."""

    def __init__(self, cls, S, eps_rel=TIK_EPS_REL, deflate=TIK_DEFLATE):
        self.cls = cls
        m = cls.m
        idx = np.where(S)[0] if S.dtype == bool else np.asarray(S)
        self.idx = idx
        uS = idx[cls.kind[idx] == 1]
        cS = idx[cls.kind[idx] == 2]
        self.D = (np.bincount(cls.unit_col[uS], weights=cls.unit_val[uS] ** 2,
                              minlength=m) if uS.size else np.zeros(m))
        sel = cls.core_pos[cS]
        A = (cls.A_core[sel] if cS.size
             else sp.csr_matrix((0, m)))
        self.A = A.tocsr()
        self.At = sp.csr_matrix(self.A.T)
        self.k = A.shape[0]
        scale = float(self.D.max()) if self.D.size else 1.0
        if self.k:
            if TIK_CONST_CACHE:
                rownorm = cls.core_rownorms()[sel]       # c1 (a)
            else:
                rownorm = np.asarray(
                    self.A.multiply(self.A).sum(axis=1)).ravel()
            scale = max(scale, float(rownorm.max()) if rownorm.size else 1.0)
        self.scale = max(scale, 1.0)
        self.eps = float(eps_rel) * self.scale
        self.De = self.D + self.eps
        self.iDe = 1.0 / self.De
        self.deflate_rounds = int(deflate)
        if self.k:
            AiDe = self.A.multiply(self.iDe)            # k x m (COO)
            M = np.eye(self.k) + (AiDe @ self.At).toarray()
            self.cho = sla.cho_factor(M, lower=True, check_finite=False)
            if TIK_CONST_CACHE:
                # c1 (b): ``AiDe`` is DEAD on every shipped call path --
                # solve_e / apply_H / null_part / pinv read iDe, A, At and cho
                # and never AiDe.  Defer the COO->CSR conversion to the first
                # reader (``__getattr__`` below), so the attribute still exists
                # for anyone who wants it and costs nothing when unread.  The
                # ``M`` Cholesky above still consumes the same COO object, so
                # no arithmetic is touched.
                self._AiDe_coo = AiDe
            else:
                self.AiDe = sp.csr_matrix(AiDe)
        else:
            self.cho = None

    def __getattr__(self, name):
        # Reached ONLY when normal lookup fails, i.e. under TIK_CONST_CACHE and
        # only if something actually reads ``AiDe``.
        if name == "AiDe":
            coo = self.__dict__.get("_AiDe_coo")
            if coo is None:
                raise AttributeError("AiDe")
            value = sp.csr_matrix(coo)
            self.AiDe = value
            return value
        raise AttributeError(name)

    # -------------------------------------------------------------- kernels
    def solve_e(self, r):
        """``H_e^{-1} r``."""
        v = self.iDe * r
        if self.k:
            w = sla.cho_solve(self.cho, self.A @ v, check_finite=False)
            v = v - self.iDe * (self.At @ w)
        return v

    def apply_H(self, x):
        out = self.D * x
        if self.k:
            out = out + self.At @ (self.A @ x)
        return out

    def null_part(self, r):
        """``P_null(H) r``, to ``(e / lam_min+)^deflate``."""
        z = r
        for _ in range(self.deflate_rounds):
            z = self.eps * self.solve_e(z)
        return z

    def pinv(self, r, refine=TIK_REFINE):
        """``(H^+ r, P_null(H) r)``, refined in the ORIGINAL sparse rows."""
        r = np.asarray(r, dtype=float)
        rn = self.null_part(r)
        rr = r - rn
        x = self.solve_e(rr)
        x = x - self.null_part(x)
        for _ in range(refine):
            rho = rr - self.apply_H(x)
            rho = rho - self.null_part(rho)
            x = x + self.solve_e(rho)
            x = x - self.null_part(x)
        return x, rn


# ------------------------------------------------------------ S2: piv-chol

def _gram_apply(L11, W, chol):
    """``y -> G^{-1} y`` with ``G = L11'(I + W'W) L11``."""

    def apply(y):
        u = sla.solve_triangular(L11, y, lower=True, trans="T",
                                 check_finite=False)
        if W is not None:
            u = u - W.T @ sla.cho_solve(chol, W @ u, check_finite=False)
        return sla.solve_triangular(L11, u, lower=True, check_finite=False)

    return apply


def piv_chol_step(H, grad, m):
    """``(delta, r_null, rank, clean_ratio)`` from one pivoted Cholesky.

    ``H`` must be a Fortran-ordered dense ``m x m`` array of ``B_S'B_S``; it is
    overwritten.  ``clean_ratio = L[r-1,r-1]^2 / (m eps L[0,0]^2)`` says how
    far the smallest KEPT pivot sits above LAPACK's own rank floor: at ratio
    ~1 the rank decision is a guess about where ``range(H)`` ends.

    Raises ``numpy.linalg.LinAlgError`` when ``dpstrf`` cannot be used.

    ``ALLOC_TRIL_SLICE`` / ``ALLOC_U_GATHER`` / ``ALLOC_DIAG_SCALAR`` (a1/a2/a3
    of agent_reports/28) are ON by default and remove memory traffic only; set
    them False to run the pre-promotion body line for line.
    """
    c, piv, rank, info = lapack.dpstrf(H, lower=1)
    rank = int(rank)
    # info > 0 is the NORMAL rank-deficient return of dpstrf, not a failure.
    if info < 0 or rank <= 0:
        raise np.linalg.LinAlgError("dpstrf rank 0 (info %d)" % info)
    if ALLOC_TRIL_SLICE:
        # a1: tril the m x rank SLICE.  ``tril`` keeps (i, j) iff i >= j, so
        # restricting j to [0, rank) commutes with tril exactly; the elements
        # are identical and only the materialised size (m*m -> m*rank) and
        # ``L``'s layout (strided view -> contiguous) change.
        L = np.tril(c[:, :rank])
    else:
        L = np.tril(c)[:, :rank]
    if ALLOC_DIAG_SCALAR:
        # a3: only dl[0] and dl[-1] are ever read, and float(np.abs(x)) ==
        # abs(float(x)) exactly for every finite double (abs clears the sign
        # bit), so ``clean`` is the same Python expression on the same floats.
        first = abs(float(c[0, 0]))
        last = abs(float(c[rank - 1, rank - 1]))
    else:
        dl = np.abs(np.diag(c)[:rank])
        first = float(dl[0])
        last = float(dl[-1])
    clean = ((last ** 2)
             / max(m * np.finfo(float).eps * first ** 2, 1e-300))
    P = np.asarray(piv, dtype=int) - 1
    if ALLOC_U_GATHER:
        # a2: ``P`` is a full permutation of 0..m-1 (dpstrf pivots every row),
        # so ``U[P] = L`` defines EVERY row of U and the zero-fill is dead.
        # With Q the inverse permutation, U = L[Q] exactly -- a pure
        # permutation COPY either way, no arithmetic.
        Q = np.empty(m, dtype=P.dtype)
        Q[P] = np.arange(m, dtype=P.dtype)
        U = np.take(L, Q, axis=0)
    else:
        U = np.zeros((m, rank))
        U[P] = L
    p = m - rank
    if p > 0:
        W = sla.solve_triangular(L[:rank, :rank], L[rank:, :rank].T,
                                 lower=True, trans="T",
                                 check_finite=False).T
        core = W @ W.T
        core[np.diag_indices_from(core)] += 1.0
        chol = sla.cho_factor(core, lower=True, check_finite=False)
    else:
        W, chol = None, None
    gsolve = _gram_apply(L[:rank, :rank], W, chol)
    a = U.T @ grad
    z = gsolve(a)
    r_null = grad - U @ z
    delta = U @ gsolve(z)
    if not (np.all(np.isfinite(delta)) and np.all(np.isfinite(r_null))):
        raise np.linalg.LinAlgError("nonfinite piv-chol step")
    return delta, r_null, rank, clean
