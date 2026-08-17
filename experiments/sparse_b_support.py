"""CSR storage for the hot-loop products of the selector-gated walker.

WHY
---
The converted Netlib programs ``to_Bxgeb`` produces are 0.27-2.6 % dense
(measured: ship04s 0.27 %, sctap1 0.59 %, fit1d 0.77 %, degen2 0.89 %,
lotfi 1.36 %, brandy 2.64 %) but every consumer stores and SLICES them
densely.  Two walker sites dominate the event loop's unattributed wall:

* ``selector_gated_walker.min_norm_endpoint_gate`` -- four boolean row
  slices of ``n x m`` dense copies (``B[mask]``, ``|B|[mask]``, ``B[off]``,
  ``|B|[off]``) plus six mat-vecs, per face acceptance;
* the breakpoint block -- ``B[off] @ ua`` and ``B[off] @ uc``, per pivot.

Measured on fit1d (2077 x 1026, 0.77 % dense, ``factor_update=True``,
cProfile over 300 pivots): the gate costs 9.5 ms per call and the loop body
4.8 ms per pivot, against 22 ms for the face solve.  Primitive costs on the
same matrix: ``B[off]`` alone 1.46 ms, ``B[off] @ u`` 1.74 ms, versus
``(csr @ u)[off]`` 0.041 ms -- a 42x ratio, because the dense form moves
17 MB and the sparse form moves 0.2 MB.

WHAT THIS IS NOT
----------------
This module is a STORAGE change for INTERNAL products only.  Nothing here
touches a certificate: exp23's ``certificate_pair`` / ``_certificate_detail``
and every frozen collaborator (``settle``, ``qp_corrector``, ``piece_y``,
``repair_face``, ``critical_derivative``, the face factorizations) keep
receiving the ORIGINAL dense ``B``.

EXACTNESS
---------
Every stored value is bit-identical to its dense counterpart: the scaled
form divides the same ``B[i, j]`` by the same ``column_scale[j]``, and
``abs`` is exact.  What is NOT identical is the SUMMATION ORDER of the
products -- a CSR mat-vec accumulates a row in column-index order and skips
structural zeros, while BLAS ``dgemv`` accumulates with its own blocking.
Results therefore agree only to round-off, which is why the walker's sparse
lane is opt-in and its trajectory is A/B'd rather than assumed.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def column_scale_of(B):
    """``np.linalg.norm(B, axis=0)``, zero columns replaced by ``1.0``.

    Computed on the DENSE array so it is bit-identical to what
    ``primal_face_pricing.select_forward_affine_multiplier`` and the walker's
    own hoisted scaling compute today.
    """
    scale = np.linalg.norm(np.asarray(B, dtype=float), axis=0)
    return np.where(scale > 0.0, scale, 1.0)


def scale_columns(matrix, column_scale):
    """``matrix / column_scale`` for a CSR matrix, entry-wise exact.

    ``data[k] / column_scale[indices[k]]`` is the same IEEE division the
    dense ``B / column_scale`` performs on that entry, so the stored values
    match bit for bit.
    """
    out = matrix.tocsr(copy=True)
    if out.nnz:
        out.data = out.data / np.asarray(column_scale, dtype=float)[out.indices]
    return out


class SparseB:
    """The four CSR forms of ``B`` the walker's hot loop needs.

    ``B`` (raw), ``absB`` (``|B|``), ``Bs`` (``B / column_scale``) and
    ``absBs`` (``|B / column_scale|``) -- exactly the four dense arrays
    ``follow_selector_gated`` hoists today, at ~1 % of the memory.
    """

    __slots__ = ("B", "absB", "Bs", "absBs", "shape", "nnz", "density",
                 "column_scale")

    def __init__(self, B, column_scale=None):
        dense = np.asarray(B, dtype=float)
        if column_scale is None:
            column_scale = column_scale_of(dense)
        self.column_scale = np.asarray(column_scale, dtype=float)
        csr = sp.csr_matrix(dense)
        csr.sort_indices()
        self.B = csr
        self.absB = abs(csr)
        self.Bs = scale_columns(csr, self.column_scale)
        self.absBs = abs(self.Bs)
        self.shape = tuple(int(v) for v in dense.shape)
        self.nnz = int(csr.nnz)
        self.density = (self.nnz / float(self.shape[0] * self.shape[1])
                        if self.shape[0] and self.shape[1] else 0.0)

    def summary(self):
        return {"rows": self.shape[0], "cols": self.shape[1],
                "nnz": self.nnz, "density": round(self.density, 6)}
