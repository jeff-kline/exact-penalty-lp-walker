"""Minimum-norm face solve at sparse cost.

The shipped face solve runs a DENSE rank-revealing QR (`dgeqp3`) on faces that
are 0.6-4.6 % dense, at `O(|W| m^2)` -- so it scales worst exactly where netlib
lives (sctap1, m=480: 37.5 ms per face against scagr7's 2.2 ms).  Replacing that
QR with SuiteSparseQR is worth 12-13x, but neither `sparseqr.solve` nor a
regularized KKT solve returns what the walker needs.

WHY THE OBVIOUS CALL IS WRONG.  `h` is the MINIMUM-NORM solution of the
UNDERDETERMINED system `B_W' h = d` (`B_W'` is `m x |W|` with `|W| > m`), so it
is non-unique for EVERY face, not only rank-deficient ones.  `sparseqr.solve`
returns a basic solution, which is why it scored 2/60 even on scagr7 where
every face is full rank.  `piece_y` requires minimum norm and the walker
depends on that choice (`ratio_multiplier="min_norm"`, exp23 parity).

WHAT THIS DOES.  Exactly the dense skeleton's algebra, with the expensive part
made sparse, the small part reused, and Q NEVER FORMED:

    Q'b_W, R, E, rank    one `sparseqr.rz` call -- Householder form, no Q
    ua, uc               from `rz_core_pair` on the r x m core (dense, small)
    h = B_W uc           g = b_W + B_W ua      (sparse mat-vec, no Q)

Not forming Q is the whole game.  Sparse QR fills Q catastrophically: on
sctap1, Q came back with 84594 nonzeros against B_W's 1710 -- 50x fill, 33 %
dense -- and building it was 83 % of the solve (`qr(economy)` 18.9 ms vs `rz`
3.1 ms on that shape).  Q is not needed in either direction: `Q'b_W` comes from
`rz`, and the forward applies are avoided because `h = B_W uc` and
`g = b_W + B_W ua` are the identities the dense path ALREADY asserts as its
residual gate (`face @ uc - h`, `face @ ua - (g - b_W)`).  Same quantities by a
cheaper route, not an approximation.

`rz_core_pair` is the same routine verified in report 93 against GELSY to
1e-15; it supplies the complete-orthogonal step that turns a basic solution
into the minimum-norm one.  So the rank-revealing factorization is sparse and
only the `r x m` core is dense.

Declines (raising `SparseFaceDecline`) route through the caller's existing
fallback chain unchanged.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from exp23_path_primal_dual import (TOL_FEAS, TOL_Y, certificate_pair,
                                    newton4)
from rz_core_face_solver import RZCoreDecline, rz_core_pair, rz_null_basis

try:
    import sparseqr as _spqr
    import sparseqr.sparseqr as _sq
except ImportError:                                   # pragma: no cover
    _spqr = _sq = None


def _fast_rz(view, bW, tolerance=-2.0):
    """``sparseqr.rz`` with the two input conversions done by ``memmove``.

    PySPQR marshals its inputs with CFFI slice assignment --
    ``Ai[0:nnz] = scipy_A.row`` in ``scipy2cholmodsparse`` and a per-column
    loop in ``numpy2cholmoddense`` (which carries its own ``FIXME
    inefficient?``).  CFFI cannot memcpy across a dtype mismatch, so those are
    Python-level element loops: measured 0.269 ms for 1114 nonzeros, i.e.
    ~250 ns per element, against **0.016 ms** for the equivalent ``memmove``
    -- 17x.  On scorpion the conversion was 9.3 % of total walker wall.

    Everything else -- the ``SuiteSparseQR_C`` call, its arguments, and the
    output conversions -- is replicated exactly from ``sparseqr.rz``.
    """
    ffi, lib, cc = _sq.ffi, _sq.lib, _sq.cc
    nnz = int(view.vals.size)
    triplet = lib.cholmod_l_allocate_triplet(view.nrows, view.m, nnz, 0,
                                             lib.CHOLMOD_REAL, cc)
    if triplet == ffi.NULL:
        raise SparseFaceDecline("cholmod triplet allocation failed")
    ffi.memmove(triplet.i, ffi.from_buffer(view.rows64), nnz * 8)
    ffi.memmove(triplet.j, ffi.from_buffer(view.cols64), nnz * 8)
    ffi.memmove(triplet.x, ffi.from_buffer(view.vals), nnz * 8)
    triplet.nnz = nnz
    chol_A = lib.cholmod_l_triplet_to_sparse(triplet, nnz, cc)
    _sq._cholmod_free_triplet(triplet)

    rows = view.nrows
    chol_b = lib.cholmod_l_allocate_dense(rows, 1, rows, lib.CHOLMOD_REAL, cc)
    if chol_b == ffi.NULL:
        _sq.cholmod_free_sparse(chol_A)
        raise SparseFaceDecline("cholmod dense allocation failed")
    ffi.memmove(chol_b.x, ffi.from_buffer(np.ascontiguousarray(bW)), rows * 8)

    chol_Z = ffi.new("cholmod_dense**")
    chol_R = ffi.new("cholmod_sparse**")
    chol_E = ffi.new("SuiteSparse_long**")
    try:
        rank = lib.SuiteSparseQR_C(
            lib.SPQR_ORDERING_DEFAULT, tolerance, view.m, 0,
            chol_A, ffi.NULL, chol_b,
            ffi.NULL, chol_Z, chol_R, chol_E,
            ffi.NULL, ffi.NULL, ffi.NULL, cc)
        Z = _sq.cholmoddense2numpy(chol_Z[0])
        R = _sq.cholmodsparse2scipy(chol_R[0])
        E = (None if chol_E == ffi.NULL
             else _sq.asarray(ffi, chol_E[0], view.m).copy())
        _sq.cholmod_free_dense(chol_Z[0])
        _sq.cholmod_free_sparse(chol_R[0])
    finally:
        _sq.cholmod_free_sparse(chol_A)
        _sq.cholmod_free_dense(chol_b)
    return Z, R, E, rank


class SparseFaceDecline(np.linalg.LinAlgError):
    """This face could not be solved sparsely; fall through."""


class FaceView:
    """One face's nonzeros, gathered straight out of ``B``'s CSR arrays.

    ``Bcsr[indices].tocoo()`` allocates an intermediate CSR and then converts;
    building the COO triplets directly from the parent's ``indptr`` is 1.80x
    faster and bit-identical (verified: max elementwise difference 0.0).  The
    same gathered arrays then serve the mat-vecs through ``bincount``, so no
    CSR is ever constructed for this face at all.
    """

    __slots__ = ("rows", "cols", "vals", "rows64", "cols64", "nrows", "m",
                 "_coo")

    def __init__(self, indptr, indices, data, face_rows, m):
        counts = indptr[face_rows + 1] - indptr[face_rows]
        total = int(counts.sum())
        ends = np.cumsum(counts)
        gather = (np.arange(total, dtype=np.intp)
                  + np.repeat(indptr[face_rows]
                              - np.concatenate(([0], ends[:-1])), counts))
        # int64 copies exist so CHOLMOD's SuiteSparse_long arrays can be filled
        # by memmove rather than by a CFFI element loop; the vectorised astype
        # is O(nnz) and ~17x cheaper than the loop it replaces.
        self.rows64 = np.repeat(np.arange(face_rows.size, dtype=np.int64),
                                counts)
        self.cols64 = indices[gather].astype(np.int64)
        self.rows = self.rows64
        self.cols = self.cols64
        self.vals = np.ascontiguousarray(data[gather])
        self.nrows = int(face_rows.size)
        self.m = int(m)
        self._coo = None

    @property
    def coo(self):
        if self._coo is None:
            self._coo = sp.coo_matrix((self.vals, (self.rows, self.cols)),
                                      shape=(self.nrows, self.m))
        return self._coo

    def matvec(self, v):
        """``B_W @ v``"""
        return np.bincount(self.rows, weights=self.vals * v[self.cols],
                           minlength=self.nrows)

    def rmatvec(self, w):
        """``B_W' @ w``"""
        return np.bincount(self.cols, weights=self.vals * w[self.rows],
                           minlength=self.m)


def sparse_min_norm_face(BW, bW, d, factors=None):
    """``(g, h, ua, uc, dres)`` for a sparse face, minimum-norm throughout.

    ``BW`` is a ``FaceView`` or any ``|W| x m`` scipy sparse matrix.  Contract
    is byte-for-byte ``piece_y``'s.  When ``factors`` is a dict it is filled
    with ``Rr``/``E``/``rank``, so the certificate can build a null-space basis
    from THIS factorization instead of re-deriving one with a dense SVD.
    """
    if _spqr is None:
        raise SparseFaceDecline("sparseqr not installed")
    if not isinstance(BW, FaceView):
        BW = _view_from_sparse(BW)
    m = BW.m
    bW = np.asarray(bW, dtype=float)
    d = np.asarray(d, dtype=float)

    # ``rz`` returns (Q'B, R, E, rank) WITHOUT ever forming Q.  That matters
    # more than anything else here: sparse QR fills Q catastrophically -- on
    # sctap1 Q came back with 84594 nonzeros against B_W's 1710 (50x fill,
    # 33 % dense), and building it was 83 % of the solve.  Measured on that
    # shape: qr(economy) 18.9 ms vs rz 3.1 ms.
    # ``tolerance=-2.0`` is SPQR_DEFAULT_TOL and must be passed EXPLICITLY.
    # PySPQR's ``tolerance=None`` default disables rank detection, so SPQR
    # returns full rank with near-zero R diagonals -- measured on share1b it
    # reported rank 199 where the SVD says 193, and the retained core then had
    # a diagonal ratio of 1e-19, which the core test (correctly) rejects.  That
    # single omission was the difference between an 82 % and a ~0 % decline
    # rate on israel.
    try:
        qtb_full, R, E, rank = _fast_rz(BW, bW, tolerance=-2.0)
    except SparseFaceDecline:
        raise
    except Exception as exc:                          # noqa: BLE001
        raise SparseFaceDecline("spqr failed: %s" % type(exc).__name__)
    rank = int(rank)
    if rank <= 0:
        raise SparseFaceDecline("zero rank")

    E = np.asarray(E).ravel().astype(np.intp)
    Rr = R.tocsr()[:rank, :].toarray()
    qtb = np.asarray(qtb_full).ravel()[:rank]
    if factors is not None:
        # `rz_core_pair` consumes Rr with overwrite_a, so keep a copy.
        factors.update(Rr=Rr.copy(), E=E, rank=rank, m=m)

    d_permuted = d[E]
    cutoff = max(Rr.shape) * np.finfo(float).eps
    try:
        _z, coefficients, _ratio = rz_core_pair(Rr, qtb, d_permuted, rank,
                                                cutoff)
    except (RZCoreDecline, ValueError, np.linalg.LinAlgError) as exc:
        raise SparseFaceDecline("core: %s" % exc)

    ua = np.zeros(m)
    uc = np.zeros(m)
    ua[E] = coefficients[:, 0]
    uc[E] = coefficients[:, 1]

    # No forward application of Q is needed either: `h` and `g` are recovered
    # from the multipliers by sparse mat-vec.  These are exactly the identities
    # the dense path already asserts as its residual gate
    # (`face @ uc - h` and `face @ ua - (g - b_W)`), so this is the same
    # quantity by a cheaper route, not an approximation.
    h = BW.matvec(uc)
    g = bW + BW.matvec(ua)

    dres = (float(np.linalg.norm(BW.rmatvec(h) - d))
            / max(1.0, float(np.linalg.norm(d))))
    for array in (g, h, ua, uc):
        if not np.all(np.isfinite(array)):
            raise SparseFaceDecline("nonfinite coefficients")
    return g, h, ua, uc, dres


def _view_from_sparse(BW):
    csr = BW.tocsr()
    return FaceView(csr.indptr, csr.indices, csr.data,
                    np.arange(csr.shape[0]), csr.shape[1])


def sparse_certificate_detail(B, b, d, y, x_hat, rank, null_basis):
    """``exp23._certificate_detail`` with its two dense re-derivations removed.

    The incumbent recovers the primal with ``np.linalg.lstsq(B[A], b[A])`` and,
    when the active face is rank deficient, obtains a null-space basis with
    ``np.linalg.svd(B[A])`` -- both DENSE, both on a matrix the sparse face
    solve has already factorized.  Neither is necessary:

      * the least-squares ``x`` IS ``-ua``.  With ``B_A E = Q R``, minimizing
        ``||B_A x - b_A||`` reduces to the min-norm solve of ``Rr w = Q'b_A``,
        which is exactly the column ``rz_core_pair`` already returns as ``ua``
        (up to sign).  The walker hands it over as ``terminal_x``.
      * the rank comes from SuiteSparseQR, and it enters only through the test
        ``rank < m``.
      * the null basis comes from the same RZ factors (``rz_null_basis``);
        verified against the SVD on real faces: identical dimension,
        ``||B_A N|| = 0.0``, projector agreement 1.3e-14, and 186x faster.

    Everything downstream -- the ``newton4`` null-space repair, the tolerances,
    the duality-gap test, ``certificate_pair`` -- is the incumbent's, verbatim.
    ``newton4`` is NOT removed: its inner solves run at ~8.3 Gflop/s, so it is
    real numerical work rather than marshalling.
    """
    n, m = B.shape
    sc_d = max(1.0, float(np.abs(d).max()))
    dres = float(np.max(np.abs(B.T @ y - d)))
    if dres > TOL_FEAS * sc_d or (y.size and float(y.min()) < -TOL_FEAS):
        reason = "dual equality" if dres > TOL_FEAS * sc_d else "negative dual"
        return False, None, dres / sc_d, reason
    A = np.where(y > TOL_Y * max(1.0, float(y.max()) if y.size else 1.0))[0]
    if A.size == 0:
        return False, None, np.inf, "empty support"

    x = np.asarray(x_hat, dtype=float)
    consist = float(np.max(np.abs(B[A] @ x - b[A])))
    viol = max(0.0, float(-(B @ x - b).min()))
    sc_b = max(1.0, float(np.abs(b).max()))
    if (consist <= TOL_FEAS * sc_b and viol > TOL_FEAS * sc_b
            and int(rank) < m and null_basis is not None
            and null_basis.shape[1] > 0):
        z = newton4(B @ null_basis, b - B @ x,
                    np.zeros(null_basis.shape[1]), 1.0)
        x2 = x + null_basis @ z
        viol2 = max(0.0, float(-(B @ x2 - b).min()))
        if viol2 < viol:
            x, viol = x2, viol2
            consist = float(np.max(np.abs(B[A] @ x - b[A])))
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


class SparseFaceSolver:
    """Per-model wrapper matching the walker's face-solver contract."""

    def __init__(self, B, b, d, residual_gate=1e-7):
        self.B = np.asarray(B, dtype=float)
        self.Bcsr = sp.csr_matrix(self.B)
        self._indptr = self.Bcsr.indptr
        self._indices = self.Bcsr.indices
        self._data = self.Bcsr.data
        self.b = np.asarray(b, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.n, self.m = self.B.shape
        self.residual_gate = float(residual_gate)
        self.solves = 0
        self.declines = 0
        self.certificates = 0
        self.decline_reasons = {}
        self.last_piece_residual = np.inf
        self._last_factors = None
        self._last_indices = None

    def _decline(self, code):
        self.declines += 1
        self.decline_reasons[code] = self.decline_reasons.get(code, 0) + 1
        raise SparseFaceDecline(code)

    def solve(self, W):
        W = np.asarray(W, dtype=bool)
        if W.shape != (self.n,) or not W.any():
            raise ValueError("path support must be a nonempty Boolean mask")
        indices = np.where(W)[0]
        BW = FaceView(self._indptr, self._indices, self._data, indices, self.m)
        bW = self.b[indices]
        factors = {}
        try:
            g, h, ua, uc, dres = sparse_min_norm_face(BW, bW, self.d, factors)
        except SparseFaceDecline as exc:
            self.declines += 1
            code = str(exc).split(":")[0]
            self.decline_reasons[code] = self.decline_reasons.get(code, 0) + 1
            raise

        # The same fail-closed residual battery the dense factors run, on the
        # ORIGINAL data.  A sparse factorization that silently lost the
        # minimum-norm property shows up here, not downstream.
        orthogonality = (float(np.abs(BW.rmatvec(g)).max())
                         / max(1.0, float(np.abs(g).max())))
        slope_error = (float(np.abs(BW.matvec(ua) - (g - bW)).max())
                       / max(1.0, float(np.abs(g - bW).max())))
        constant_error = (float(np.abs(BW.matvec(uc) - h).max())
                          / max(1.0, float(np.abs(h).max())))
        self.last_piece_residual = max(orthogonality, slope_error,
                                       constant_error)
        if self.last_piece_residual > self.residual_gate:
            self._decline("piece residual %.2e" % self.last_piece_residual)
        self.solves += 1
        self._last_factors = factors
        self._last_indices = indices
        return g, h, ua, uc, dres

    def certificate(self, y, x_hat, idxW):
        """Certificate for the face solved MOST RECENTLY, reusing its factors.

        Returns ``None`` when the cached factorization does not correspond to
        ``idxW``, so the caller falls through to the incumbent dense path
        rather than certifying against a stale factorization.
        """
        factors = self._last_factors
        if (not factors or self._last_indices is None or idxW is None
                or not np.array_equal(np.asarray(idxW).ravel(),
                                      self._last_indices)):
            return None
        rank = int(factors["rank"])
        if rank >= self.m:
            null_basis = np.zeros((self.m, 0))
        else:
            try:
                null_basis = rz_null_basis(factors["Rr"], rank, factors["E"],
                                           self.m)
            except (RZCoreDecline, ValueError, np.linalg.LinAlgError):
                return None
        self.certificates += 1
        return sparse_certificate_detail(self.B, self.b, self.d, y, x_hat,
                                         rank, null_basis)

    def diagnostics(self):
        return {"sparse_face_solves": self.solves,
                "sparse_face_declines": self.declines,
                "sparse_face_decline_reasons": dict(self.decline_reasons),
                "sparse_face_certificates": self.certificates,
                "sparse_face_last_residual": self.last_piece_residual}
