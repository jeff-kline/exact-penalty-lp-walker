"""Rank-core numerical skeleton for exact affine Mangasarian path pieces.

The geometric support is never thinned.  The persistent QR maintained by
``PathPieceFactor`` represents the entire support, while sensitive
minimum-norm solves are performed by complete orthogonal decomposition on its
rank core.  This replaces a full-face SVD fallback without changing the QP.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from path_piece_factor import PathPieceFactor


class PathPieceSkeletonFactor(PathPieceFactor):
    """Persistent face QR with minimum-norm recovery on the numerical core."""

    def __init__(self, B, b, d, **kwargs):
        # The parent condition threshold dispatches to a full-face SVD.  This
        # subclass instead checks a complete-orthogonal solve on the rank core.
        kwargs["min_diagonal_ratio"] = 0.0
        super().__init__(B, b, d, **kwargs)
        self.core_solves = 0
        self.core_rank_failures = 0
        self.core_refinements = 0
        self.last_core_rank = 0
        self.last_skeleton_columns = 0
        self.last_passive_columns = 0
        self.last_passive_face_directions = 0
        self.last_leverage_min = 0.0
        self.last_leverage_max = 0.0

    @staticmethod
    def _relative_residual(matrix, solution, rhs):
        residual = matrix @ solution - rhs
        denominator = 1.0 + np.linalg.norm(rhs, axis=0)
        values = np.linalg.norm(residual, axis=0) / denominator
        return float(np.max(np.atleast_1d(values)))

    @staticmethod
    def _cod_solve(matrix, rhs, expected_rank):
        """Minimum-norm GELSY solve with one checked refinement."""
        rhs = np.asarray(rhs, dtype=float)
        vector = rhs.ndim == 1
        rhs2 = rhs[:, None] if vector else rhs
        cutoff = max(matrix.shape) * np.finfo(float).eps
        solution, _residuals, rank, _singular = linalg.lstsq(
            matrix, rhs2, cond=cutoff, lapack_driver="gelsy",
            check_finite=False)
        if int(rank) != int(expected_rank):
            raise np.linalg.LinAlgError(
                "rank-core COD disagrees with persistent QR "
                f"({rank} != {expected_rank})")
        initial = PathPieceSkeletonFactor._relative_residual(
            matrix, solution, rhs2)
        refined = False
        if initial > 1e-12:
            correction_rhs = rhs2 - matrix @ solution
            correction, _residuals, correction_rank, _singular = linalg.lstsq(
                matrix, correction_rhs, cond=cutoff,
                lapack_driver="gelsy", check_finite=False)
            if int(correction_rank) != int(expected_rank):
                raise np.linalg.LinAlgError(
                    "rank-core refinement changed numerical rank")
            candidate = solution + correction
            if (PathPieceSkeletonFactor._relative_residual(
                    matrix, candidate, rhs2) < initial):
                solution = candidate
                refined = True
        return (solution[:, 0] if vector else solution), int(rank), refined

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
            self.last_diagonal_ratio = float(
                diagonal[rank - 1] / diagonal[0])
            leverage = np.sum(Qr * Qr, axis=1)
            self.last_leverage_min = float(np.min(leverage))
            self.last_leverage_max = float(np.max(leverage))
            try:
                z, zrank, refined_z = self._cod_solve(
                    Rr.T, d_permuted, rank)
                h = Qr @ z
                qtb = Qr.T @ bW
                coefficients, crank, refined_coefficients = self._cod_solve(
                    Rr, np.column_stack((-qtb, z)), rank)
            except np.linalg.LinAlgError:
                self.core_rank_failures += 1
                raise
            self.last_core_rank = min(zrank, crank)
            self.core_refinements += int(refined_z)
            self.core_refinements += int(refined_coefficients)
            ua_permuted = coefficients[:, 0]
            uc_permuted = coefficients[:, 1]
            g = bW - Qr @ qtb
            g -= Qr @ (Qr.T @ g)
        else:
            self.last_diagonal_ratio = 1.0
            self.last_leverage_min = 0.0
            self.last_leverage_max = 0.0
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
                f"skeleton piece residual {self.last_piece_residual:.3e}")
        arrays = (g, h, ua, uc)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise np.linalg.LinAlgError("nonfinite skeleton coefficients")
        return g, h, ua, uc, dres

    def diagnostics(self):
        result = super().diagnostics()
        result.update({
            "path_skeleton_core_solves": self.core_solves,
            "path_skeleton_core_rank_failures": self.core_rank_failures,
            "path_skeleton_core_refinements": self.core_refinements,
            "path_skeleton_last_core_rank": self.last_core_rank,
            "path_skeleton_last_columns": self.last_skeleton_columns,
            "path_skeleton_last_passive_columns": self.last_passive_columns,
            "path_skeleton_last_passive_face_directions":
            self.last_passive_face_directions,
            "path_skeleton_last_leverage_min": self.last_leverage_min,
            "path_skeleton_last_leverage_max": self.last_leverage_max,
        })
        return result

