"""Face solve without re-deriving a factorization the walker already holds.

``PathPieceSkeletonFactor`` maintains a persistent rank-revealing QR of ``B_W``
and then calls ``scipy.linalg.lstsq(..., lapack_driver="gelsy")`` twice on its
own ``R`` factor.  GELSY is ``dgeqp3`` (rank-revealing QR, BLAS-2, expensive)
followed by ``dtzrzf`` and triangular solves -- so the ``dgeqp3`` stage
re-computes, on ``Rr``, exactly the kind of factorization ``Rr`` already is.

This subclass keeps every gate, counter and residual test of its parent and
replaces only that redundancy.  ONE ``dtzrzf`` of ``Rr`` serves BOTH solves:

    Rr = [T 0] Z,   T  rank x rank upper triangular,  Z  m x m orthogonal

  * min-norm  ``Rr u = rhs``      ->  u = Z'[T^-1 rhs ; 0]
  * least-sq  ``min||Rr' z - d||`` ->  z = T^-T (Z d)[:rank]

The second identity is what makes the sharing possible: the parent solves it as
a separate GELSY on ``Rr.T`` (which is LOWER trapezoidal, so no LAPACK routine
exploits its structure), but ``Rr' z ~ d`` is equivalent to ``T' z = (Zd)[:r]``
because ``Z`` is orthogonal and drops out of the norm.

WHAT IS GIVEN UP, AND WHAT REPLACES IT.  The parent used GELSY's returned rank
as an independent cross-check against the persistent QR (``_cod_solve`` raises
when they disagree).  Dropping GELSY drops that oracle, so this class tests the
rank core directly: ``T``'s diagonal ratio against the same cutoff.  A face
whose core is numerically singular DECLINES, and the caller's existing
fallback chain handles it exactly as it handles every other decline.

MEASURED, on the ``Rr`` shapes the walker actually produces (m = 140):
    rank == m   ->  the whole thing is two triangular solves,  30x
    rank <  m   ->  RZ path vs GELSY,  3.7x - 6.3x,  agreement 1e-15

Opt-in.  Nothing imports this by default.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from path_piece_skeleton import PathPieceSkeletonFactor

_TZRZF, _ORMRZ, _TRTRS = linalg.get_lapack_funcs(
    ("tzrzf", "ormrz", "trtrs"), (np.zeros((1, 1)),))


class RZCoreDecline(np.linalg.LinAlgError):
    """The rank core is too ill-conditioned to solve through."""


def rz_core_pair(Rr, qtb, d_permuted, rank, cutoff):
    """Both face solves from ONE RZ factorization of ``Rr``.

    ``Rr`` is ``rank x m`` upper trapezoidal (the leading rows of a pivoted
    QR's R).  The two halves are ordered rather than independent: the
    least-squares solve produces ``z``, and ``z`` is then the second
    right-hand side of the minimum-norm solve, so a single factorization
    serves both.

    Returns ``(z, coefficients, core_ratio)`` with ``coefficients[:, 0]`` the
    minimum-norm solution of ``Rr u = -qtb`` and ``coefficients[:, 1]`` that
    of ``Rr u = z``.
    """
    rows, m = Rr.shape
    if rows != rank:
        raise ValueError("Rr must be exactly the rank core rows")
    if rank == m:
        # Square upper triangular: Z is the identity, no RZ step at all.
        core = np.asfortranarray(Rr)
        tau = None
    else:
        core, tau, info = _TZRZF(np.asfortranarray(Rr), overwrite_a=1)
        if info != 0:
            raise RZCoreDecline("tzrzf failed (info=%d)" % info)

    T = core[:, :rank]
    diagonal = np.abs(np.diag(T))
    if not diagonal.size:
        raise RZCoreDecline("empty rank core")
    core_ratio = float(diagonal.min() / diagonal.max())
    if not np.isfinite(core_ratio) or core_ratio <= cutoff:
        raise RZCoreDecline("rank core diagonal ratio %.3e" % core_ratio)

    # ---- least squares  min || Rr' z - d ||  ->  T' z = (Z d)[:rank] -----
    # ||Rr'z - d|| = ||Z'[T';0]z - d|| = ||[T';0]z - Zd||, so the tail of Zd
    # is the irreducible residual and the head fixes z.
    if tau is None:
        e_head = np.asfortranarray(d_permuted[:rank].reshape(rank, 1))
    else:
        e, info = _ORMRZ(core, tau,
                         np.asfortranarray(d_permuted.reshape(m, 1)),
                         side="L", trans="N")
        if info != 0:
            raise RZCoreDecline("ormrz(N) failed (info=%d)" % info)
        e_head = np.asfortranarray(np.asarray(e)[:rank].reshape(rank, 1))
    z, info = _TRTRS(T, e_head, lower=0, trans=1)
    if info != 0:
        raise RZCoreDecline("trtrs(T') singular (info=%d)" % info)
    z = np.asarray(z).reshape(rank)

    # ---- minimum norm  Rr u = rhs  ->  u = Z' [T^-1 rhs ; 0] ------------
    rhs = np.asfortranarray(np.column_stack((-qtb, z)))
    head, info = _TRTRS(T, rhs, lower=0, trans=0)
    if info != 0:
        raise RZCoreDecline("trtrs(T) singular (info=%d)" % info)
    head = np.asarray(head)
    if tau is None:
        coefficients = head
    else:
        padded = np.zeros((m, 2), order="F")
        padded[:rank] = head
        coefficients, info = _ORMRZ(core, tau, padded, side="L", trans="T")
        if info != 0:
            raise RZCoreDecline("ormrz(T) failed (info=%d)" % info)
    return z, np.asarray(coefficients), core_ratio


def rz_null_basis(Rr, rank, permutation, m):
    """Orthonormal basis of ``null(B_W)`` from the SAME RZ factorization.

    ``_certificate_detail`` obtains this with a dense ``np.linalg.svd(B[A])``
    -- on a matrix the sparse solver has already factorized.  It is not needed:
    with ``Rr = [T 0] Z`` and ``B_W E = Q R`` where ``Q`` has full column rank,

        B_W x = 0  <=>  Rr (E'x) = 0  <=>  (Z E'x)[:rank] = 0

    so the null space is spanned by the LAST ``m - rank`` rows of ``Z``, which
    ``dormrz`` produces by applying ``Z'`` to the trailing identity block --
    the same call the min-norm solve already makes.  On sctap1 the SVD it
    replaces was 0.650 s of a 6.29 s run (10 %).

    Returns an ``m x (m - rank)`` array with orthonormal columns, in the
    ORIGINAL (unpermuted) coordinates.
    """
    rows, cols = Rr.shape
    if rank >= m:
        return np.zeros((m, 0))
    if rows != rank:
        raise ValueError("Rr must be exactly the rank core rows")
    if rank == cols:
        core, tau = np.asfortranarray(Rr), None
    else:
        core, tau, info = _TZRZF(np.asfortranarray(Rr), overwrite_a=1)
        if info != 0:
            raise RZCoreDecline("tzrzf failed in null basis (info=%d)" % info)
    trailing = np.zeros((m, m - rank), order="F")
    trailing[rank:] = np.eye(m - rank)
    if tau is None:
        basis_permuted = trailing
    else:
        basis_permuted, info = _ORMRZ(core, tau, trailing, side="L", trans="T")
        if info != 0:
            raise RZCoreDecline("ormrz failed in null basis (info=%d)" % info)
    basis = np.zeros((m, m - rank))
    basis[np.asarray(permutation).ravel()] = np.asarray(basis_permuted)
    return basis


class PathPieceRZFactor(PathPieceSkeletonFactor):
    """Persistent face QR whose rank-core solves skip GELSY's ``dgeqp3``."""

    def __init__(self, B, b, d, **kwargs):
        super().__init__(B, b, d, **kwargs)
        self.rz_solves = 0
        self.rz_declines = 0
        self.rz_square_solves = 0
        self.last_core_ratio = 1.0

    def solve(self, W):
        W = np.asarray(W, dtype=bool)
        if W.shape != (self.n,) or not W.any():
            raise ValueError("path support must be a nonempty Boolean mask")
        indices = np.where(W)[0]
        self._update(indices)
        self.solves += 1
        self.core_solves += 1

        face_rows = indices.size
        rank = self.rank
        bW = self.b[indices]
        Qr = self.Q[:, :rank]
        Rr = self.R[:rank, :]
        d_permuted = self.d[self.permutation]
        self.last_core_rank = int(rank)
        self.last_skeleton_columns = int(rank)
        self.last_passive_columns = int(self.m - rank)
        self.last_passive_face_directions = int(face_rows - rank)

        if rank:
            diagonal = np.abs(np.diag(self.R))
            self.last_diagonal_ratio = float(diagonal[rank - 1] / diagonal[0])
            leverage = np.sum(Qr * Qr, axis=1)
            self.last_leverage_min = float(np.min(leverage))
            self.last_leverage_max = float(np.max(leverage))

            qtb = Qr.T @ bW
            cutoff = max(Rr.shape) * np.finfo(float).eps
            try:
                z, coefficients, ratio = rz_core_pair(
                    Rr, qtb, d_permuted, rank, cutoff)
            except (RZCoreDecline, ValueError, np.linalg.LinAlgError):
                self.rz_declines += 1
                raise
            self.rz_solves += 1
            if rank == self.m:
                self.rz_square_solves += 1
            self.last_core_ratio = ratio
            self.last_core_rank = int(rank)
            ua_permuted = coefficients[:, 0]
            uc_permuted = coefficients[:, 1]
            h = Qr @ z
            g = bW - Qr @ qtb
            g -= Qr @ (Qr.T @ g)
        else:
            self.last_diagonal_ratio = 1.0
            self.last_leverage_min = 0.0
            self.last_leverage_max = 0.0
            self.last_core_ratio = 1.0
            h = np.zeros(face_rows)
            g = bW.copy()
            ua_permuted = np.zeros(self.m)
            uc_permuted = np.zeros(self.m)

        ua = np.zeros(self.m)
        uc = np.zeros(self.m)
        ua[self.permutation] = ua_permuted
        uc[self.permutation] = uc_permuted
        face = self.B[indices]
        dres = (float(np.linalg.norm(face.T @ h - self.d))
                / max(1.0, float(np.linalg.norm(self.d))))
        orthogonality = (float(np.linalg.norm(face.T @ g, np.inf))
                         / max(1.0, float(np.linalg.norm(g, np.inf))))
        slope_residual = face @ ua - (g - bW)
        constant_residual = face @ uc - h
        slope_error = (float(np.linalg.norm(slope_residual, np.inf))
                       / max(1.0, float(np.linalg.norm(g - bW, np.inf))))
        constant_error = (float(np.linalg.norm(constant_residual, np.inf))
                          / max(1.0, float(np.linalg.norm(h, np.inf))))
        self.last_piece_residual = max(
            orthogonality, slope_error, constant_error)
        if self.last_piece_residual > 1e-7:
            raise np.linalg.LinAlgError(
                f"rz piece residual {self.last_piece_residual:.3e}")
        arrays = (g, h, ua, uc)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise np.linalg.LinAlgError("nonfinite rz coefficients")
        return g, h, ua, uc, dres

    def diagnostics(self):
        result = super().diagnostics()
        result.update({
            "rz_solves": self.rz_solves,
            "rz_declines": self.rz_declines,
            "rz_square_solves": self.rz_square_solves,
            "rz_last_core_ratio": self.last_core_ratio,
        })
        return result
