"""Sparse-basis, rank-one-updated face solves for bound-constrained projections."""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu


def _chol_rank_one(lower, vector, sign):
    """Return the lower Cholesky factor of ``L L' + sign*x*x'``.

    The downdate is rejected if roundoff or the requested change would destroy
    positive definiteness.  Callers retain the explicit Gram matrix and can
    rebuild its factor as a checked recovery.
    """
    factor = np.asarray(lower, dtype=float).copy()
    x = np.asarray(vector, dtype=float).copy()
    for k in range(x.size):
        diagonal2 = factor[k, k] * factor[k, k] + sign * x[k] * x[k]
        floor = 64.0 * np.finfo(float).eps * max(
            1.0, factor[k, k] * factor[k, k])
        if diagonal2 <= floor:
            raise np.linalg.LinAlgError("nonpositive Cholesky downdate")
        diagonal = float(np.sqrt(diagonal2))
        cosine = diagonal / factor[k, k]
        sine = x[k] / factor[k, k]
        factor[k, k] = diagonal
        if k + 1 < x.size:
            factor[k + 1:, k] = (
                factor[k + 1:, k] + sign * sine * x[k + 1:]) / cosine
            x[k + 1:] = (cosine * x[k + 1:]
                         - sine * factor[k + 1:, k])
    return factor


class IncrementalFaceFactor:
    """Maintain one sparse equality basis and an updated reduced Gram factor.

    If ``J`` is the sparse equality basis and ``N`` the remaining free
    variables, ``T=E_J^{-1}E_N`` and ``M=I+T T'``.  A nonbasic bound pivot is a
    rank-one update of ``M``. Basis-column departures use product-form eta
    updates; only a periodic stability refresh or an explicit tangent reset
    refactors the sparse equality basis.
    """

    def __init__(self, E, f, free, pivot_tol=1e-11, max_eta_updates=24,
                 basis=None, equilibrate_rows=False):
        self.original_E = csc_matrix(E)
        self.original_f = np.asarray(f, dtype=float)
        self.equilibrate_rows = bool(equilibrate_rows)
        if equilibrate_rows:
            norms = np.sqrt(np.asarray(
                self.original_E.multiply(self.original_E).sum(axis=1)
            ).ravel())
            self.row_scale = 1.0 / np.maximum(
                norms, np.sqrt(np.finfo(float).tiny))
            self.row_scale = np.clip(self.row_scale, 1e-12, 1e12)
        else:
            self.row_scale = np.ones(self.original_E.shape[0])
        self.E = diags(self.row_scale, format="csc") @ self.original_E
        self.f = self.row_scale * self.original_f
        self.r, self.n = self.E.shape
        self.pivot_tol = float(pivot_tol)
        self.max_eta_updates = int(max_eta_updates)
        self.basis = []
        self.basis_lu = None
        self.etas = []
        self.coordinates = np.zeros((self.r, self.n))
        self.nonbasic_free = np.zeros(self.n, dtype=bool)
        self.h = np.zeros(self.r)
        self.gram = np.eye(self.r)
        self.cholesky = np.eye(self.r)
        self.gram_usable = True
        self.basis_refactorizations = 0
        self.gram_rebuilds = 0
        self.rank_updates = 0
        self.rank_downdates = 0
        self.basis_updates = 0
        self.face_solves = 0
        self.refinements = 0
        self._updates_since_check = 0
        self.reset(free, basis=basis)

    def _select_basis(self, free):
        if self.r == 0:
            return []
        candidates = np.where(free)[0]
        if candidates.size < self.r:
            raise np.linalg.LinAlgError("free set smaller than equality rank")
        # Equilibration is a linear-solve preconditioner, not a combinatorial
        # crash rule.  Select the equality basis from original data so merely
        # enabling row scaling cannot change the initial basis labels.
        dense = self.original_E[:, candidates].toarray()
        _, triangular, pivot = linalg.qr(
            dense, mode="economic", pivoting=True, check_finite=True)
        diagonal = np.abs(np.diag(triangular))
        cutoff = ((diagonal[0] if diagonal.size else 0.0)
                  * max(dense.shape) * np.finfo(float).eps)
        if int(np.sum(diagonal > cutoff)) != self.r:
            raise np.linalg.LinAlgError("free set does not span equality rows")
        return [int(candidates[j]) for j in pivot[:self.r]]

    def _factor_basis(self):
        if self.r == 0:
            self.basis_lu = None
            self.etas = []
            return
        matrix = csc_matrix(self.E[:, self.basis])
        self.basis_lu = splu(matrix, permc_spec="COLAMD",
                             diag_pivot_thresh=0.1)
        self.etas = []
        self.basis_refactorizations += 1

    @staticmethod
    def _eta_inverse(rhs, position, column):
        """Solve ``E_eta x=rhs`` where E_eta replaces column p by column."""
        result = np.asarray(rhs, dtype=float).copy()
        pivot = float(column[position])
        if abs(pivot) <= np.finfo(float).tiny:
            raise np.linalg.LinAlgError("singular eta update")
        if result.ndim == 1:
            xp = result[position] / pivot
            result -= column * xp
            result[position] = xp
        else:
            xp = result[position, :] / pivot
            result -= column[:, None] * xp[None, :]
            result[position, :] = xp
        return result

    @staticmethod
    def _eta_transpose_inverse(rhs, position, column):
        """Solve ``E_eta' x=rhs`` for a product-form basis update."""
        result = np.asarray(rhs, dtype=float).copy()
        pivot = float(column[position])
        if abs(pivot) <= np.finfo(float).tiny:
            raise np.linalg.LinAlgError("singular eta transpose update")
        if result.ndim == 1:
            numerator = (result[position] - float(column @ result)
                         + pivot * result[position])
            result[position] = numerator / pivot
        else:
            numerator = (result[position, :] - column @ result
                         + pivot * result[position, :])
            result[position, :] = numerator / pivot
        return result

    def _solve_basis(self, rhs, transpose=False):
        if self.r == 0:
            return np.zeros_like(rhs, dtype=float)
        if not transpose:
            result = self.basis_lu.solve(rhs)
            for position, column in self.etas:
                result = self._eta_inverse(result, position, column)
            return result
        result = np.asarray(rhs, dtype=float)
        for position, column in reversed(self.etas):
            result = self._eta_transpose_inverse(result, position, column)
        return self.basis_lu.solve(result, trans="T")

    def _rebuild_gram(self):
        if self.r == 0:
            self.gram = np.zeros((0, 0))
            self.cholesky = np.zeros((0, 0))
            self.gram_usable = True
            return
        indices = np.where(self.nonbasic_free)[0]
        tableau = self.coordinates[:, indices]
        self.gram = np.eye(self.r) + tableau @ tableau.T
        self.gram = 0.5 * (self.gram + self.gram.T)
        try:
            self.cholesky = np.linalg.cholesky(self.gram)
            self.gram_usable = True
        except np.linalg.LinAlgError:
            # A very ill-conditioned sparse basis can make I+T*T' lose its
            # unit spectral floor in floating point.  The caller will use the
            # checked sparse augmented-KKT solve until a later pivot restores
            # a usable reduced factor.
            self.cholesky = None
            self.gram_usable = False
        self.gram_rebuilds += 1
        self._updates_since_check = 0

    def rebuild(self, free, refactor_basis=False):
        free = np.asarray(free, dtype=bool)
        if free.shape != (self.n,):
            raise ValueError("free mask has wrong shape")
        if len(self.basis) != self.r or any(not free[i] for i in self.basis):
            raise np.linalg.LinAlgError("invalid equality basis")
        if refactor_basis:
            self._factor_basis()
        self.coordinates.fill(0.0)
        self.nonbasic_free = free.copy()
        if self.basis:
            self.nonbasic_free[np.asarray(self.basis)] = False
        if self.r:
            self.h = self._solve_basis(self.f)
            indices = np.where(self.nonbasic_free)[0]
            if indices.size:
                self.coordinates[:, indices] = self._solve_basis(
                    self.E[:, indices].toarray())
        self._rebuild_gram()

    def reset(self, free, basis=None):
        free = np.asarray(free, dtype=bool)
        self.basis = (self._select_basis(free) if basis is None
                      else [int(i) for i in basis])
        self.rebuild(free, refactor_basis=True)

    def set_rhs(self, f):
        """Change only the equality right-hand side, retaining every factor.

        This is the cheap operation needed by an equality-RHS homotopy.  The
        sparse equality basis, its eta updates, tableau coordinates, reduced
        Gram matrix, and Cholesky factor are independent of ``f``.  Only the
        basic feasible offset ``h=E_J^{-1}f`` changes.
        """
        f = np.asarray(f, dtype=float)
        if f.shape != (self.r,):
            raise ValueError("equality right-hand side has wrong shape")
        self.original_f = f.copy()
        self.f = self.row_scale * self.original_f
        self.h = self._solve_basis(self.f) if self.r else np.zeros(0)

    def _coordinate(self, index):
        index = int(index)
        if self.nonbasic_free[index]:
            return self.coordinates[:, index].copy()
        if self.r == 0:
            return np.zeros(0)
        return self._solve_basis(self.E[:, index].toarray().ravel())

    def _update_gram(self, coordinate, sign):
        if self.r == 0:
            return
        if not self.gram_usable:
            self._rebuild_gram()
            return
        self.gram += sign * np.outer(coordinate, coordinate)
        self.gram = 0.5 * (self.gram + self.gram.T)
        try:
            self.cholesky = _chol_rank_one(self.cholesky, coordinate, sign)
        except np.linalg.LinAlgError:
            self._rebuild_gram()
            return
        self._updates_since_check += 1
        if self._updates_since_check >= 64:
            residual = self.cholesky @ self.cholesky.T - self.gram
            error = (float(np.linalg.norm(residual, np.inf))
                     / max(1.0, float(np.linalg.norm(self.gram, np.inf))))
            if error > 1e-10:
                self._rebuild_gram()
            else:
                self._updates_since_check = 0

    def release_bound(self, index, free):
        index = int(index)
        if free[index]:
            return
        if index in self.basis:
            raise np.linalg.LinAlgError("an equality basis index cannot be active")
        coordinate = self._coordinate(index)
        free[index] = True
        self.nonbasic_free[index] = True
        self.coordinates[:, index] = coordinate
        self._update_gram(coordinate, +1.0)
        self.rank_updates += 1

    def _replacement(self, position, free, leaving, forbidden_bases=None):
        basis_set = set(self.basis)
        forbidden_bases = set() if forbidden_bases is None else forbidden_bases
        best = None
        best_score = -np.inf
        # Prefer already-free columns so the basis exchange changes only the
        # intended bound.  Among them choose the strongest tableau pivot.
        for require_free in (True, False):
            for candidate in range(self.n):
                if (candidate == leaving or candidate in basis_set
                        or bool(free[candidate]) != require_free):
                    continue
                trial_basis = list(self.basis)
                trial_basis[position] = candidate
                if tuple(trial_basis) in forbidden_bases:
                    continue
                coordinate = self._coordinate(candidate)
                scale = max(1.0, float(np.linalg.norm(coordinate, np.inf)))
                score = abs(float(coordinate[position])) / scale
                if score > self.pivot_tol and score > best_score:
                    best = (candidate, coordinate)
                    best_score = score
            if best is not None:
                return best
        return None

    def activate_bound(self, index, free, forbidden_bases=None):
        """Activate one bound; return ``(accepted, released_replacement)``."""
        index = int(index)
        if not free[index]:
            return True, None
        if index not in self.basis:
            coordinate = self.coordinates[:, index].copy()
            free[index] = False
            self.nonbasic_free[index] = False
            self.coordinates[:, index] = 0.0
            self._update_gram(coordinate, -1.0)
            self.rank_downdates += 1
            return True, None

        position = self.basis.index(index)
        replacement = self._replacement(position, free, index,
                                        forbidden_bases=forbidden_bases)
        if replacement is None:
            return False, None
        candidate, _ = replacement
        was_active = not bool(free[candidate])
        coordinate = self._coordinate(candidate)
        # Transform the current tableau from the old basis to the new one by
        # one product-form (eta) update.  This is the incremental sparse-basis
        # step that avoids refactoring A at every basic departure.
        self.h = self._eta_inverse(self.h, position, coordinate)
        nonbasic = np.where(self.nonbasic_free)[0]
        if nonbasic.size:
            self.coordinates[:, nonbasic] = self._eta_inverse(
                self.coordinates[:, nonbasic], position, coordinate)
        free[index] = False
        free[candidate] = True
        self.nonbasic_free[index] = False
        self.nonbasic_free[candidate] = False
        self.coordinates[:, index] = 0.0
        self.coordinates[:, candidate] = 0.0
        self.basis[position] = int(candidate)
        self.etas.append((position, coordinate))
        self.basis_updates += 1
        if len(self.etas) >= self.max_eta_updates:
            self.rebuild(free, refactor_basis=True)
        else:
            self._rebuild_gram()
        return True, int(candidate) if was_active else None

    def solve(self, c, refinement_steps=2):
        c = np.asarray(c, dtype=float)
        if c.shape != (self.n,):
            raise ValueError("face target has wrong shape")
        self.face_solves += 1
        free = self.nonbasic_free.copy()
        if self.basis:
            free[np.asarray(self.basis)] = True
        y = np.zeros(self.n)
        if self.r == 0:
            y[free] = c[free]
            return y, np.zeros(0), 0.0
        if not self.gram_usable:
            raise np.linalg.LinAlgError("reduced Gram factor is unusable")

        nonbasic = np.where(self.nonbasic_free)[0]
        tableau = self.coordinates[:, nonbasic]
        rhs = c[np.asarray(self.basis)] - self.h
        if nonbasic.size:
            rhs = rhs + tableau @ c[nonbasic]
        mu = linalg.cho_solve((self.cholesky, True), rhs,
                              check_finite=False)
        rhs_scale = max(1.0, float(np.linalg.norm(rhs, np.inf)))
        for _ in range(refinement_steps):
            residual = rhs - self.gram @ mu
            if (float(np.linalg.norm(residual, np.inf))
                    <= 32.0 * np.finfo(float).eps * rhs_scale):
                break
            mu += linalg.cho_solve((self.cholesky, True), residual,
                                   check_finite=False)
            self.refinements += 1

        y[np.asarray(self.basis)] = c[np.asarray(self.basis)] - mu
        if nonbasic.size:
            y[nonbasic] = c[nonbasic] - tableau.T @ mu
        scaled_multiplier = self._solve_basis(mu, transpose=True)
        # Ebar=R E implies Ebar' lambdabar=E' lambda with
        # lambda=R*lambdabar.  Return the original-coordinate multiplier so
        # every caller and certificate continues to see the unscaled KKT
        # system.
        multiplier = self.row_scale * scaled_multiplier

        equality_residual = np.asarray(
            self.original_E @ y - self.original_f).ravel()
        stationarity = y[free] - c[free]
        stationarity += np.asarray(
            self.original_E[:, free].T @ multiplier).ravel()
        equality_scale = (1.0 + np.abs(self.original_f)
                          + np.asarray(
                              np.abs(self.original_E) @ np.abs(y)).ravel())
        stationarity_scale = (
            1.0 + np.abs(y[free]) + np.abs(c[free])
            + np.asarray(np.abs(self.original_E[:, free]).T
                         @ np.abs(multiplier)).ravel())
        equality_error = (float(np.max(np.abs(equality_residual)
                                       / equality_scale))
                          if equality_residual.size else 0.0)
        stationarity_error = (float(np.max(np.abs(stationarity)
                                           / stationarity_scale))
                              if stationarity.size else 0.0)
        return y, multiplier, max(equality_error, stationarity_error)

    def diagnostics(self):
        positive_scale = self.row_scale[self.row_scale > 0.0]
        scale_span = (float(positive_scale.max() / positive_scale.min())
                      if positive_scale.size else 1.0)
        return {
            "basis_refactorizations": self.basis_refactorizations,
            "gram_rebuilds": self.gram_rebuilds,
            "rank_updates": self.rank_updates,
            "rank_downdates": self.rank_downdates,
            "basis_updates": self.basis_updates,
            "eta_updates": len(self.etas),
            "face_solves": self.face_solves,
            "factor_refinements": self.refinements,
            "gram_usable": self.gram_usable,
            "row_equilibrated": bool(np.any(self.row_scale != 1.0)),
            "row_equilibration_span": scale_span,
        }
