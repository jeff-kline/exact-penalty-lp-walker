"""Persistent rank-aware QR factor for affine pieces of the dual path."""

from __future__ import annotations

from bisect import bisect_left

import numpy as np
from scipy import linalg


class PathPieceConditionError(np.linalg.LinAlgError):
    """The updated face is too ill-conditioned for QR multiplier recovery."""


class PathPieceFactor:
    """Update ``B[W]`` by rows and recover the minimum-norm affine piece.

    A pivoted QR is built initially.  Small support changes retain its column
    permutation and use QR row insertions/deletions.  Rank changes, large batch
    changes, or residual growth trigger a fresh pivoted QR.
    """

    def __init__(self, B, b, d, max_batch_updates=8, refresh_updates=32,
                 min_diagonal_ratio=1e-5):
        self.B = np.asarray(B, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.n, self.m = self.B.shape
        self.max_batch_updates = int(max_batch_updates)
        self.refresh_updates = int(refresh_updates)
        self.min_diagonal_ratio = float(min_diagonal_ratio)
        self.indices = np.zeros(0, dtype=int)
        self.permutation = np.arange(self.m)
        self.Q = None
        self.R = None
        self.rank = 0
        self.updates_since_rebuild = 0
        self.rebuilds = 0
        self.row_updates = 0
        self.rank_change_rebuilds = 0
        self.residual_rebuilds = 0
        self.condition_rejections = 0
        self.solves = 0
        self.last_piece_residual = np.inf
        self.last_diagonal_ratio = 1.0

    @staticmethod
    def _rank_from_r(R, shape):
        diagonal = np.abs(np.diag(R))
        if not diagonal.size:
            return 0
        cutoff = diagonal[0] * max(shape) * np.finfo(float).eps
        return int(np.sum(diagonal > cutoff))

    def _rebuild(self, indices):
        indices = np.asarray(indices, dtype=int)
        face = self.B[indices]
        self.Q, self.R, pivot = linalg.qr(
            face, mode="economic", pivoting=True, check_finite=False)
        self.permutation = np.asarray(pivot, dtype=int)
        self.indices = indices.copy()
        self.rank = self._rank_from_r(self.R, face.shape)
        self.updates_since_rebuild = 0
        self.rebuilds += 1

    def _factor_error(self):
        if self.Q is None:
            return np.inf
        target = self.B[self.indices][:, self.permutation]
        residual = self.Q @ self.R - target
        factor_error = (float(np.linalg.norm(residual, np.inf))
                        / max(1.0, float(np.linalg.norm(target, np.inf))))
        identity = np.eye(self.Q.shape[1])
        orthogonality = (float(np.linalg.norm(self.Q.T @ self.Q - identity,
                                              np.inf))
                         if identity.size else 0.0)
        return max(factor_error, orthogonality)

    def _update(self, new_indices):
        new_indices = np.asarray(new_indices, dtype=int)
        old_set = set(int(i) for i in self.indices)
        new_set = set(int(i) for i in new_indices)
        additions = sorted(new_set - old_set)
        removals = sorted(old_set - new_set, reverse=True)
        changes = len(additions) + len(removals)
        if (self.Q is None or changes > self.max_batch_updates
                or self.updates_since_rebuild + changes > self.refresh_updates):
            self._rebuild(new_indices)
            return
        if changes == 0:
            return

        current = list(int(i) for i in self.indices)
        old_rank = self.rank
        try:
            for index in additions:
                position = bisect_left(current, index)
                row = self.B[index, self.permutation]
                self.Q, self.R = linalg.qr_insert(
                    self.Q, self.R, row, position, which="row", rcond=1e-12,
                    overwrite_qru=False, check_finite=False)
                current.insert(position, index)
            for index in removals:
                position = current.index(index)
                self.Q, self.R = linalg.qr_delete(
                    self.Q, self.R, position, which="row",
                    overwrite_qr=False, check_finite=False)
                current.pop(position)
        except (ValueError, np.linalg.LinAlgError):
            self._rebuild(new_indices)
            return

        self.indices = np.asarray(current, dtype=int)
        self.row_updates += changes
        self.updates_since_rebuild += changes
        new_rank = self._rank_from_r(self.R, (len(current), self.m))
        if new_rank != old_rank:
            self.rank_change_rebuilds += 1
            self._rebuild(new_indices)
            return
        self.rank = new_rank
        if self._factor_error() > 1e-9:
            self.residual_rebuilds += 1
            self._rebuild(new_indices)

    def solve(self, W):
        W = np.asarray(W, dtype=bool)
        if W.shape != (self.n,) or not W.any():
            raise ValueError("path support must be a nonempty Boolean mask")
        indices = np.where(W)[0]
        self._update(indices)
        self.solves += 1

        face_rows = indices.size
        rank = self.rank
        bW = self.b[indices]
        Qr = self.Q[:, :rank]
        Rr = self.R[:rank, :]
        d_permuted = self.d[self.permutation]

        if rank:
            diagonal = np.abs(np.diag(self.R))
            self.last_diagonal_ratio = float(
                diagonal[rank - 1] / diagonal[0])
            # A rank-revealing QR can correctly identify the numerical rank
            # while still leaving multiplier recovery highly sensitive.  In
            # that regime tiny subspace errors are amplified by Rr^+, changing
            # off-face breakpoint coordinates even though g and h look sound.
            # Reject the face before using those multipliers; the caller keeps
            # this updated factor but evaluates this one face with the checked
            # SVD implementation.
            # The threshold is intentionally conservative.  The coefficient
            # shadow on boeing2 showed a 2.7e-6 forward mismatch at a diagonal
            # ratio of 1.1e-8 even though all backward residuals passed.  A
            # 1e-5 boundary leaves roughly ten decimal digits of triangular
            # amplification headroom and routes ambiguous faces to SVD.
            if self.last_diagonal_ratio < self.min_diagonal_ratio:
                self.condition_rejections += 1
                raise PathPieceConditionError(
                    "path face is too ill-conditioned for QR multiplier "
                    f"recovery (diagonal ratio "
                    f"{self.last_diagonal_ratio:.3e})")
            # Rr' = Z*T is a small full-column-rank QR. It supplies stable
            # minimum-norm solves with Rr without forming normal equations.
            Z, triangular = linalg.qr(
                Rr.T, mode="economic", pivoting=False, check_finite=False)
            z = linalg.solve_triangular(
                triangular, Z.T @ d_permuted, lower=False,
                check_finite=False)
            h = Qr @ z
            qtb = Qr.T @ bW
            g = bW - Qr @ qtb
            g -= Qr @ (Qr.T @ g)

            def minimum_norm(rhs):
                coefficient = linalg.solve_triangular(
                    triangular.T, rhs, lower=True, check_finite=False)
                return Z @ coefficient

            ua_permuted = minimum_norm(-qtb)
            uc_permuted = minimum_norm(z)
        else:
            self.last_diagonal_ratio = 1.0
            h = np.zeros(face_rows)
            g = bW.copy()
            ua_permuted = np.zeros(self.m)
            uc_permuted = np.zeros(self.m)

        ua = np.zeros(self.m)
        uc = np.zeros(self.m)
        ua[self.permutation] = ua_permuted
        uc[self.permutation] = uc_permuted
        dres = (float(np.linalg.norm(self.B[indices].T @ h - self.d))
                / max(1.0, float(np.linalg.norm(self.d))))
        face = self.B[indices]
        orthogonality = (float(np.linalg.norm(face.T @ g, np.inf))
                         / max(1.0, float(np.linalg.norm(g, np.inf))))
        slope_residual = face @ ua - (g - bW)
        constant_residual = face @ uc - h
        slope_error = (float(np.linalg.norm(slope_residual, np.inf))
                       / max(1.0, float(np.linalg.norm(g - bW, np.inf))))
        constant_error = (float(np.linalg.norm(constant_residual, np.inf))
                          / max(1.0, float(np.linalg.norm(h, np.inf))))
        # ``dres`` is deliberately not part of the factor-rejection test: an
        # inconsistent trial support is a normal input to ``settle``, which
        # uses dres to reject/repair that support.  The factor must reproduce
        # its minimum-residual affine piece rather than pre-empt that logic.
        self.last_piece_residual = max(
            orthogonality, slope_error, constant_error)
        if self.last_piece_residual > 1e-7:
            raise np.linalg.LinAlgError(
                f"path-piece residual {self.last_piece_residual:.3e}")
        arrays = (g, h, ua, uc)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise np.linalg.LinAlgError("nonfinite path coefficients")
        return g, h, ua, uc, dres

    def diagnostics(self):
        return {
            "path_factor_solves": self.solves,
            "path_factor_rebuilds": self.rebuilds,
            "path_factor_row_updates": self.row_updates,
            "path_factor_rank_change_rebuilds": self.rank_change_rebuilds,
            "path_factor_residual_rebuilds": self.residual_rebuilds,
            "path_factor_condition_rejections": self.condition_rejections,
            "path_factor_last_diagonal_ratio": self.last_diagonal_ratio,
            "path_factor_last_residual": self.last_piece_residual,
        }
