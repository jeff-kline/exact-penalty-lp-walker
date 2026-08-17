"""Structure-exploiting ("unit-core") face solver for the dual path pieces.

OPT-IN ONLY.  ``selector_gated_walker.follow_selector_gated`` builds one of
these when ``unit_core=True``; the default walker path is untouched.

WHAT A FACE SOLVE IS
--------------------
For a working set ``W`` the walker needs the affine piece

    y_W(t) = t * g + h,      g, h in R^{|W|}

that solves ``min 0.5 ||y_W - t b_W||^2  s.t.  B_W' y_W = d`` for every ``t``,
together with the minimum-norm multipliers ``ua, uc in R^m`` satisfying the
contract exp23's ``piece_y`` publishes:

    B_W ua = g - b_W          B_W uc = h          y_W(t) = t g + h

DERIVATION OF THE NORMAL-EQUATION SPLIT (verified numerically, see the report)
-----------------------------------------------------------------------------
Write ``B_W = U S V'`` (thin SVD, rank r).  ``piece_y`` returns

    h  = U_r S_r^{-1} V_r' d                  (min-norm solution of B_W' h = d)
    g  = (I - U_r U_r') b_W                   (residual of b_W on range(B_W))
    ua = -V_r' S_r^{-1} U_r' b_W              (min-norm B_W ua = g - b_W)
    uc = V_r' S_r^{-2} V_r' d                 (min-norm B_W uc = h)

Let ``G = B_W' B_W = V S^2 V'``, so ``G^+ = V_r S_r^{-2} V_r'``.  Then

    G^+ d              = V_r S_r^{-2} V_r' d              = uc
    B_W uc             = U_r S_r V_r' V_r S_r^{-2} V_r' d
                       = U_r S_r^{-1} V_r' d              = h
    -G^+ (B_W' b_W)    = -V_r S_r^{-2} V_r' V_r S_r U_r' b_W
                       = -V_r S_r^{-1} U_r' b_W           = ua
    b_W + B_W ua       = b_W - U_r U_r' b_W               = g

So the ENTIRE piece is two solves against ``G`` plus two multiplications by
``B_W``:

    uc = G^+ d              h = B_W uc
    ua = -G^+ (B_W' b_W)    g = b_W + B_W ua

``g`` and ``h`` are then *defined* as ``b_W + B_W ua`` and ``B_W uc``, so the
two multiplier identities of the contract hold to machine round-off BY
CONSTRUCTION -- unlike the SVD path, where they are a numerical consequence.
The only open questions are (i) that ``G^+`` really is ``G^{-1}`` (i.e. ``B_W``
has full COLUMN rank, otherwise the pseudoinverse and the inverse disagree and
``ua``/``uc`` would not be minimum-norm) and (ii) the accuracy loss of the
normal equations.  Both are guarded below.

STRUCTURE EXPLOITATION
----------------------
The converted netlib operator is [bound rows; constraint core] interleaved.
Rows with exactly one nonzero ("unit rows", value ``s_i`` in column ``j_i``)
dominate: fit1d 2052/2077, ship04s 1642/2214, lotfi 315/556.  Splitting the
rows of ``W`` into unit rows, core rows ``A`` (>= 2 nonzeros) and empty rows,

    G = B_W' B_W = D + A' A,     D_jj = sum_{unit i in W, j_i = j} s_i^2

``D`` is diagonal and free; ``A`` is the only real matrix.  Rather than form
``A' A`` (which squares the sparsity -- and for fit1d, whose core rows are
essentially dense, is a full 1026x1026 matrix), the solver factors the BORDERED
system once per face

    K = [ D    A' ]      K [x; z] = [r; 0]  =>  (D + A'A) x = r
        [ A   -I  ]

whose Schur complement onto the leading block is exactly ``G``.  This is the
same Woodbury / Sherman-Morrison algebra as forming ``M0 = D + A_s'A_s`` and
correcting for the dense rows, but in a form that (a) never squares sparsity,
(b) stays nonsingular when ``D`` has zero entries (fit1d faces routinely have
1-2 columns untouched by any unit row, which makes the ``M0`` of the explicit
Woodbury split singular), and (c) needs no density threshold to tune.

For fit1d this is a 1038x1038 matrix with ~26k nonzeros in place of a dense
1035x1026 SVD; for ship04s a 1686x1686 matrix with ~15k nonzeros in place of a
dense 1583x1458 SVD.

ACCURACY AND FALLBACK
---------------------
Normal equations square the condition number, so the raw ``ua`` from a single
solve carries a relative error of order ``cond(B_W)^2 * eps`` (measured
``cond(B_W) ~ 4e4`` on fit1d, i.e. ~1e-7 -- far too coarse).  The solver
therefore runs Bjorck's corrected-semi-normal-equations refinement: one or two
steps of

    ua <- ua - G^{-1} (B_W' g)         (drives B_W' g -> 0)
    uc <- uc + G^{-1} (d - B_W' h)     (drives B_W' h -> d)

reusing the same factorization.  This restores accuracy comparable to a
backward-stable factorization whenever ``cond(B_W)^2 * eps << 1``.

Every solve is then gated.  The solver DECLINES (raises ``UnitCoreFallback``,
the walker routes to the unchanged SVD/QR path and counts it) when

  * the face has fewer rows than columns, so ``B_W`` cannot have full column
    rank;
  * ``splu`` fails or produces a non-finite solve;
  * the estimated ``cond(G) = lambda_max/lambda_min`` (power / inverse-power
    iteration on the factor) exceeds ``UNIT_CORE_MAX_COND_G``.  This is the
    rank-deficiency detector: a column-rank-deficient ``B_W`` makes ``G``
    singular, ``G^+ != G^{-1}``, and the computed multipliers would not be
    minimum-norm.  lotfi's faces are rank-deficient (measured
    ``sigma_min(B_W) ~ 1e-17``) and decline here on every face;
  * the a-posteriori residuals ``||B_W' g||/||g||`` or
    ``||B_W' h - d||/||d||`` exceed ``UNIT_CORE_ORTHO_TOL`` /
    ``UNIT_CORE_DRES_TOL``, both far tighter than anything the walker's gates
    accept.

Correctness beats speed: a decline is always safe, it just costs the old path.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------------------------------------------- thresholds
# cond(G) = cond(B_W)^2.  1e12 admits cond(B_W) <= 1e6, for which the
# corrected semi-normal equations retain full working accuracy
# (cond(B_W)^2 * eps = 1e-4 << 1).  Measured cond(G): fit1d ~1.8e9,
# ship04s ~2.3e4, lotfi ~1e38 (rank deficient -> declines).
UNIT_CORE_MAX_COND_G = 1e12
# ||B_W' g||_inf / max(1, ||g||_inf), the SAME scaling the skeleton factor
# gates at 1e-7 (path_piece_skeleton.py:144).  The normal-equation route
# plateaus near 2e-9 on fit1d no matter how much refinement is applied --
# that is a scaling floor (||B_W|| ~ 3.3e3 there), not lost accuracy: faces
# with an internal residual in [1e-9,1e-8] were measured to agree with
# piece_y's g,h,ua,uc to 7.5e-12 (fit1d) and 4.3e-11 (brandy).  1e-8 is
# therefore set above that floor but still 10x tighter than the incumbent
# gate and than the walker's TOL_FEAS; no face in any measured model landed
# between 1e-8 and 1e-7, so this is not an "accept everything" threshold.
UNIT_CORE_ORTHO_TOL = 1e-8
# same definition as piece_y's dres; the walker's own gate is TOL_FEAS = 1e-7.
UNIT_CORE_DRES_TOL = 1e-8
# refinement sweeps per right-hand side (each is one extra triangular solve).
UNIT_CORE_REFINE_ROUNDS = 2
# iterations of the (inverse) power method used for the condition estimate.
UNIT_CORE_COND_ITERS = 3


class UnitCoreFallback(Exception):
    """The structured solver declines this face; use the unchanged path.

    ``code`` is a stable short reason (safe to use as a counter key);
    ``str(exc)`` carries the measured value as well.
    """

    def __init__(self, code, detail=""):
        super().__init__("%s %s" % (code, detail) if detail else code)
        self.code = code
        self.detail = detail


class UnitCoreFaceSolver:
    """Per-model row classification + per-face bordered sparse face solve."""

    def __init__(self, B, b, d, max_cond_G=UNIT_CORE_MAX_COND_G,
                 ortho_tol=UNIT_CORE_ORTHO_TOL, dres_tol=UNIT_CORE_DRES_TOL,
                 refine_rounds=UNIT_CORE_REFINE_ROUNDS,
                 cond_iters=UNIT_CORE_COND_ITERS):
        self.B = np.asarray(B, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.n, self.m = self.B.shape
        if self.b.shape != (self.n,) or self.d.shape != (self.m,):
            raise ValueError("expected B.shape == (len(b), len(d))")
        self.max_cond_G = float(max_cond_G)
        self.ortho_tol = float(ortho_tol)
        self.dres_tol = float(dres_tol)
        self.refine_rounds = int(refine_rounds)
        self.cond_iters = int(cond_iters)
        self.d_norm = max(1.0, float(np.linalg.norm(self.d)))

        # ---- one-time row classification ---------------------------------
        nonzero = self.B != 0.0
        counts = nonzero.sum(axis=1)
        self.kind = np.where(counts == 1, 1,
                             np.where(counts == 0, 0, 2)).astype(np.int8)
        unit_rows = np.where(self.kind == 1)[0]
        self.unit_col = np.full(self.n, -1, dtype=np.intp)
        self.unit_val = np.zeros(self.n)
        if unit_rows.size:
            cols = np.argmax(nonzero[unit_rows], axis=1)
            self.unit_col[unit_rows] = cols
            self.unit_val[unit_rows] = self.B[unit_rows, cols]
        core_rows = np.where(self.kind == 2)[0]
        self.core_pos = np.full(self.n, -1, dtype=np.intp)
        self.core_pos[core_rows] = np.arange(core_rows.size)
        self.core_rows = core_rows
        self.A_core = sp.csr_matrix(self.B[core_rows]) if core_rows.size \
            else sp.csr_matrix((0, self.m))
        self.n_unit = int(unit_rows.size)
        self.n_core = int(core_rows.size)
        self.n_empty = int((self.kind == 0).sum())

        # counters (mirrored into the walker's stats dict by the caller)
        self.solves = 0
        self.declines = 0
        self.decline_reasons = {}
        self.last_cond_G = 0.0
        self.last_ortho = 0.0
        self.last_dres = 0.0
        self.refinement_sweeps = 0

    # ------------------------------------------------------------ helpers
    def _decline(self, code, detail=""):
        self.declines += 1
        self.decline_reasons[code] = self.decline_reasons.get(code, 0) + 1
        raise UnitCoreFallback(code, detail)

    def structure(self):
        return {"unit_rows": self.n_unit, "core_rows": self.n_core,
                "empty_rows": self.n_empty,
                "core_nnz": int(self.A_core.nnz)}

    # ------------------------------------------------------------- solve
    def solve(self, W):
        """Return ``(g, h, ua, uc, dres)`` exactly as ``piece_y`` does."""
        W = np.asarray(W, dtype=bool)
        if W.shape != (self.n,) or not W.any():
            raise ValueError("path support must be a nonempty Boolean mask")
        idx = np.where(W)[0]
        rows = idx.size
        m = self.m
        if rows < m:
            # B_W has more columns than rows: column rank deficient for sure.
            self._decline("face shorter than column count")

        kinds = self.kind[idx]
        pos_unit = np.where(kinds == 1)[0]
        pos_core = np.where(kinds == 2)[0]
        ju = self.unit_col[idx[pos_unit]]
        su = self.unit_val[idx[pos_unit]]
        A = self.A_core[self.core_pos[idx[pos_core]]] if pos_core.size \
            else sp.csr_matrix((0, m))
        k = A.shape[0]
        bW = self.b[idx]

        # D_jj = sum of s_i^2 over unit rows of W landing in column j
        D = np.bincount(ju, weights=su * su, minlength=m) if ju.size \
            else np.zeros(m)

        def apply_BW(u):
            """B_W u, laid out in ascending row order like piece_y's output."""
            out = np.zeros(rows)
            if pos_unit.size:
                out[pos_unit] = su * u[ju]
            if k:
                out[pos_core] = A @ u
            return out

        def apply_BWt(v):
            """B_W' v."""
            out = np.bincount(ju, weights=su * v[pos_unit], minlength=m) \
                if ju.size else np.zeros(m)
            if k:
                out = out + (A.T @ v[pos_core])
            return out

        def apply_G(u):
            out = D * u
            if k:
                out = out + (A.T @ (A @ u))
            return out

        # ---- factor the bordered system once ----------------------------
        if k:
            K = sp.bmat([[sp.diags(D), A.T],
                         [A, -sp.eye(k, format="csr")]], format="csc")
        else:
            K = sp.diags(D).tocsc()
        try:
            lu = spla.splu(K)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            self._decline("splu failed", type(exc).__name__)

        pad = np.zeros(k)

        def solve_G(r):
            rhs = np.concatenate((r, pad)) if k else r
            sol = lu.solve(rhs)
            return sol[:m] if k else sol

        # ---- rank / conditioning guard ----------------------------------
        cond_G = self._condition_estimate(apply_G, solve_G, m)
        self.last_cond_G = cond_G
        if not np.isfinite(cond_G) or cond_G > self.max_cond_G:
            self._decline("cond(G) too large", "%.3e" % cond_G)

        # ---- uc, h  (uc = G^{-1} d, h = B_W uc) -------------------------
        uc = solve_G(self.d)
        h = apply_BW(uc)
        for _ in range(self.refine_rounds):
            residual = self.d - apply_BWt(h)
            if float(np.linalg.norm(residual)) <= 1e-15 * self.d_norm:
                break
            uc = uc + solve_G(residual)
            h = apply_BW(uc)
            self.refinement_sweeps += 1

        # ---- ua, g  (ua = -G^{-1} B_W' b_W, g = b_W + B_W ua) -----------
        ua = -solve_G(apply_BWt(bW))
        g = bW + apply_BW(ua)
        g_scale = max(1.0, float(np.linalg.norm(g, np.inf)))
        for _ in range(self.refine_rounds):
            residual = -apply_BWt(g)
            if float(np.linalg.norm(residual, np.inf)) <= 1e-15 * g_scale:
                break
            ua = ua + solve_G(residual)
            g = bW + apply_BW(ua)
            g_scale = max(1.0, float(np.linalg.norm(g, np.inf)))
            self.refinement_sweeps += 1

        # ---- a-posteriori gates -----------------------------------------
        for array in (g, h, ua, uc):
            if not np.all(np.isfinite(array)):
                self._decline("nonfinite coefficients")
        ortho = (float(np.linalg.norm(apply_BWt(g), np.inf))
                 / max(1.0, float(np.linalg.norm(g, np.inf))))
        self.last_ortho = ortho
        if ortho > self.ortho_tol:
            self._decline("orthogonality", "%.3e" % ortho)
        dres = float(np.linalg.norm(apply_BWt(h) - self.d)) / self.d_norm
        self.last_dres = dres
        if dres > self.dres_tol:
            self._decline("dres", "%.3e" % dres)

        self.solves += 1
        return g, h, ua, uc, dres

    # -------------------------------------------------------- conditioning
    def _condition_estimate(self, apply_G, solve_G, m):
        """(lambda_max / lambda_min) of the SPSD matrix G, power iterations.

        ``G`` is symmetric positive semidefinite, so its eigenvalues are its
        singular values.  A few forward power iterations bound
        ``lambda_max`` from below and a few inverse iterations bound
        ``lambda_min`` from above; the resulting ratio therefore
        UNDER-estimates nothing in the dangerous direction -- a
        rank-deficient face drives the inverse iteration's Rayleigh quotient
        to round-off and the ratio explodes, which is what triggers the
        fallback.
        """
        rng = np.random.default_rng(12345)
        v = rng.standard_normal(m)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return np.inf
        v /= norm
        lam_max = 0.0
        for _ in range(self.cond_iters):
            w = apply_G(v)
            norm = float(np.linalg.norm(w))
            if not np.isfinite(norm) or norm == 0.0:
                return np.inf
            lam_max = norm
            v = w / norm
        u = rng.standard_normal(m)
        norm = float(np.linalg.norm(u))
        if norm == 0.0:
            return np.inf
        u /= norm
        lam_min = np.inf
        for _ in range(self.cond_iters):
            w = solve_G(u)
            norm = float(np.linalg.norm(w))
            if not np.isfinite(norm) or norm == 0.0:
                return np.inf
            lam_min = 1.0 / norm
            u = w / norm
        if lam_min <= 0.0:
            return np.inf
        return float(lam_max / lam_min)


__all__ = ["UnitCoreFaceSolver", "UnitCoreFallback", "UNIT_CORE_MAX_COND_G",
           "UNIT_CORE_ORTHO_TOL", "UNIT_CORE_DRES_TOL"]
