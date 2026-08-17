#include "revised_basis_solver.hpp"

#include <SuiteSparseQR_C.h>
#include <vecLib/cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

extern "C" {
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda,
             int *info);
void dpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *a, const int *lda, double *b, const int *ldb,
             int *info);
}

namespace twalker::revised {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double inf_norm(const std::vector<double> &values) {
  double result = 0.0;
  for (double value : values) result = std::max(result, std::abs(value));
  return result;
}

double norm2(const std::vector<double> &values) {
  long double sum = 0.0L;
  for (double value : values)
    sum += static_cast<long double>(value) * value;
  return std::sqrt(static_cast<double>(sum));
}

bool cholesky(std::vector<double> &matrix, int n) {
  if (n == 0) return false;
  const char lower = 'L';
  int info = 0;
  dpotrf_(&lower, &n, matrix.data(), &n, &info);
  if (info != 0) return false;
  for (int column = 0; column < n; ++column)
    for (int row = 0; row < column; ++row)
      matrix[row + static_cast<std::size_t>(n) * column] = 0.0;
  return true;
}

bool solve_cholesky(const std::vector<double> &factor, int n,
                    std::vector<double> &right, int right_count = 1) {
  if (n == 0 || factor.size() != static_cast<std::size_t>(n) * n
      || right.size() != static_cast<std::size_t>(n) * right_count)
    return false;
  const char lower = 'L';
  int info = 0;
  dpotrs_(&lower, &n, &right_count, factor.data(), &n, right.data(), &n,
          &info);
  return info == 0;
}

bool cholesky_rank_one(std::vector<double> &factor, int n,
                       std::vector<double> x, int sign) {
  for (int k = 0; k < n; ++k) {
    const auto diagonal_index = k + static_cast<std::size_t>(n) * k;
    const double diagonal = factor[diagonal_index];
    if (!(diagonal > 0.0) || !std::isfinite(diagonal)) return false;
    const long double radicand =
        static_cast<long double>(diagonal) * diagonal
        + sign * static_cast<long double>(x[k]) * x[k];
    if (!(radicand > 0.0L)) return false;
    const double updated_diagonal = std::sqrt(static_cast<double>(radicand));
    const double c = updated_diagonal / diagonal;
    const double s = x[k] / diagonal;
    factor[diagonal_index] = updated_diagonal;
    for (int row = k + 1; row < n; ++row) {
      const auto index = row + static_cast<std::size_t>(n) * k;
      const double updated = (factor[index] + sign * s * x[row]) / c;
      x[row] = c * x[row] - s * updated;
      factor[index] = updated;
    }
  }
  return true;
}

double diagonal_ratio(const std::vector<double> &factor, int n) {
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = 0.0;
  for (int i = 0; i < n; ++i) {
    const double value = std::abs(factor[i + static_cast<std::size_t>(n) * i]);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  return maximum > 0.0 ? minimum / maximum : 0.0;
}

}  // namespace

RevisedBasisSolver::RevisedBasisSolver(const Fixture &fixture,
                                       std::vector<double> target_shift)
    : fixture_(fixture), target_shift_(std::move(target_shift)) {
  cholmod_l_start(&common_);
  common_.nmethods = 1;
  common_.method[0].ordering = CHOLMOD_AMD;
  common_.postorder = 1;
}

RevisedBasisSolver::~RevisedBasisSolver() { cholmod_l_finish(&common_); }

double RevisedBasisSolver::sparse_row_dot(std::uint32_t left,
                                          std::uint32_t right) const {
  auto lp = fixture_.indptr[left], lend = fixture_.indptr[left + 1];
  auto rp = fixture_.indptr[right], rend = fixture_.indptr[right + 1];
  long double result = 0.0L;
  while (lp < lend && rp < rend) {
    if (fixture_.indices[lp] < fixture_.indices[rp]) {
      ++lp;
    } else if (fixture_.indices[rp] < fixture_.indices[lp]) {
      ++rp;
    } else {
      result += static_cast<long double>(fixture_.values[lp])
                * fixture_.values[rp];
      ++lp;
      ++rp;
    }
  }
  return static_cast<double>(result);
}

double RevisedBasisSolver::sparse_row_norm2(std::uint32_t row) const {
  long double result = 0.0L;
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
    result += static_cast<long double>(fixture_.values[p]) * fixture_.values[p];
  return static_cast<double>(result);
}

bool RevisedBasisSolver::row_coordinates(
    std::uint32_t row, std::vector<double> &coordinates,
    double &relative_residual) const {
  const int rank = static_cast<int>(basis_rows_.size());
  coordinates.resize(rank);
  for (int basis = 0; basis < rank; ++basis)
    coordinates[basis] = sparse_row_dot(basis_rows_[basis], row);
  const auto cross = coordinates;
  if (!solve_cholesky(basis_factor_, rank, coordinates)) return false;
  (void)cross;
  // Form the residual through the immutable sparse rows.  The cheaper
  // quadratic identity ||a||^2-a'C'(CC')^-1Ca loses enough digits to mistake
  // a dependent entering row for a rank increase.
  std::vector<long double> residual(fixture_.m, 0.0L);
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
    residual[fixture_.indices[p]] = fixture_.values[p];
  for (int basis = 0; basis < rank; ++basis) {
    const auto basis_row = basis_rows_[basis];
    const long double coefficient = coordinates[basis];
    for (auto p = fixture_.indptr[basis_row];
         p < fixture_.indptr[basis_row + 1]; ++p)
      residual[fixture_.indices[p]] -= coefficient * fixture_.values[p];
  }
  long double residual2 = 0.0L;
  for (long double value : residual) residual2 += value * value;
  const double row_norm2 = sparse_row_norm2(row);
  relative_residual = std::sqrt(
      static_cast<double>(residual2) / std::max(1.0, row_norm2));
  return std::isfinite(relative_residual);
}

bool RevisedBasisSolver::rebuild(const std::vector<std::uint32_t> &rows) {
  const auto start = Clock::now();
  valid_ = false;
  rows_ = rows;
  basis_rows_.clear();
  basis_factor_.clear();
  coordinate_factor_.clear();
  coordinates_.clear();

  // Refactorization is the only rank-revealing operation.  Sparse pivoted QR
  // of A' identifies independent active rows; ordinary same-rank transitions
  // below do not repeat this work.
  const int m = static_cast<int>(fixture_.m);
  const int active = static_cast<int>(rows.size());
  if (m == 0 || active == 0) return false;
  std::size_t face_nnz = 0;
  for (auto row : rows)
    face_nnz += fixture_.indptr[row + 1] - fixture_.indptr[row];
  auto *triplet = cholmod_l_allocate_triplet(
      m, active, face_nnz, 0, CHOLMOD_REAL, &common_);
  if (!triplet) return false;
  auto *ti = static_cast<std::int64_t *>(triplet->i);
  auto *tj = static_cast<std::int64_t *>(triplet->j);
  auto *tx = static_cast<double *>(triplet->x);
  std::size_t cursor = 0;
  for (int local = 0; local < active; ++local) {
    const auto row = rows[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      ti[cursor] = fixture_.indices[p];
      tj[cursor] = local;
      tx[cursor] = fixture_.values[p];
      ++cursor;
    }
  }
  triplet->nnz = cursor;
  auto *transpose = cholmod_l_triplet_to_sparse(triplet, cursor, &common_);
  cholmod_l_free_triplet(&triplet, &common_);
  if (!transpose) return false;
  cholmod_sparse *R = nullptr;
  std::int64_t *permutation = nullptr;
  const auto rank64 = SuiteSparseQR_C(
      SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, m, 0, transpose, nullptr,
      nullptr, nullptr, nullptr, &R, &permutation, nullptr, nullptr, nullptr,
      &common_);
  cholmod_l_free_sparse(&transpose, &common_);
  if (R) cholmod_l_free_sparse(&R, &common_);
  if (rank64 <= 0 || rank64 > std::min(m, active)) {
    if (permutation)
      cholmod_l_free(active, sizeof(std::int64_t), permutation, &common_);
    return false;
  }
  const int rank = static_cast<int>(rank64);
  if (rank == 0) return false;
  basis_rows_.reserve(rank);
  for (int basis = 0; basis < rank; ++basis) {
    const int local = permutation ? static_cast<int>(permutation[basis])
                                  : basis;
    if (local < 0 || local >= active) return false;
    basis_rows_.push_back(rows[local]);
  }
  if (permutation)
    cholmod_l_free(active, sizeof(std::int64_t), permutation, &common_);
  std::vector<double> basis_gram(static_cast<std::size_t>(rank) * rank, 0.0);
  for (int column = 0; column < rank; ++column)
    for (int row = column; row < rank; ++row)
      basis_gram[row + static_cast<std::size_t>(rank) * column] =
          sparse_row_dot(basis_rows_[row], basis_rows_[column]);
  if (!cholesky(basis_gram, rank)) return false;
  basis_factor_ = std::move(basis_gram);
  std::vector<double> coordinate_gram(static_cast<std::size_t>(rank) * rank,
                                      0.0);
  for (auto row : rows) {
    std::vector<double> coordinate;
    double residual = 0.0;
    if (!row_coordinates(row, coordinate, residual)) return false;
    stats_.worst_row_reconstruction = std::max(
        stats_.worst_row_reconstruction, residual);
    // This diagnostic is evaluated as ||a||^2-a'C'(CC')^-1Ca and loses
    // digits by cancellation on represented rows.  It is not an answer gate;
    // original-operator residuals and frozen face oracles are the answer gates.
    if (residual > 2e-5) return false;
    coordinates_.emplace(row, coordinate);
    for (int column = 0; column < rank; ++column)
      for (int i = column; i < rank; ++i)
        coordinate_gram[i + static_cast<std::size_t>(rank) * column] +=
            coordinate[i] * coordinate[column];
  }
  if (!cholesky(coordinate_gram, rank)) return false;
  coordinate_factor_ = std::move(coordinate_gram);
  valid_ = true;
  ++stats_.rebuilds;
  stats_.rebuild_ms += milliseconds_since(start);
  return true;
}

bool RevisedBasisSolver::transition(
    const std::vector<std::uint32_t> &rows) {
  const auto start = Clock::now();
  if (!valid_) return rebuild(rows);
  if (rows == rows_) {
    ++stats_.unchanged_reuses;
    return true;
  }

  std::vector<std::uint32_t> additions, removals;
  std::set_difference(rows.begin(), rows.end(), rows_.begin(), rows_.end(),
                      std::back_inserter(additions));
  std::set_difference(rows_.begin(), rows_.end(), rows.begin(), rows.end(),
                      std::back_inserter(removals));
  const int rank = static_cast<int>(basis_rows_.size());
  std::vector<std::pair<std::uint32_t, std::vector<double>>> new_coordinates;
  for (auto row : additions) {
    std::vector<double> coordinate;
    double residual = 0.0;
    if (!row_coordinates(row, coordinate, residual)) return rebuild(rows);
    stats_.worst_row_reconstruction = std::max(
        stats_.worst_row_reconstruction, residual);
    if (residual > 2e-10) {
      ++stats_.rank_changes;
      return rebuild(rows);
    }
    new_coordinates.emplace_back(row, std::move(coordinate));
  }

  // Add before delete, then replace a leaving basis row by pivoting a
  // dependent active row into its position.  This changes coordinates but
  // preserves the represented row space and avoids a rank-revealing QR.
  for (const auto &entry : new_coordinates)
    coordinates_[entry.first] = entry.second;
  bool exchanged = false;
  for (auto leaving : removals) {
    const auto basis_position =
        std::find(basis_rows_.begin(), basis_rows_.end(), leaving);
    if (basis_position == basis_rows_.end()) continue;
    ++stats_.basis_removals;
    const int pivot_position =
        static_cast<int>(basis_position - basis_rows_.begin());
    std::uint32_t entering = 0;
    double pivot_magnitude = 0.0;
    bool found = false;
    for (auto candidate : rows) {
      if (std::find(basis_rows_.begin(), basis_rows_.end(), candidate)
          != basis_rows_.end())
        continue;
      const auto coordinate = coordinates_.find(candidate);
      if (coordinate == coordinates_.end()) continue;
      const double magnitude = std::abs(coordinate->second[pivot_position]);
      if (magnitude > pivot_magnitude) {
        pivot_magnitude = magnitude;
        entering = candidate;
        found = true;
      }
    }
    if (!found || pivot_magnitude <= 1e-10) {
      ++stats_.rank_changes;
      return rebuild(rows);
    }
    const auto pivot_row = coordinates_.at(entering);
    const double pivot = pivot_row[pivot_position];
    for (auto &[row, coordinate] : coordinates_) {
      (void)row;
      const double new_pivot = coordinate[pivot_position] / pivot;
      for (int column = 0; column < rank; ++column)
        if (column != pivot_position)
          coordinate[column] -= new_pivot * pivot_row[column];
      coordinate[pivot_position] = new_pivot;
    }
    auto &unit = coordinates_.at(entering);
    std::fill(unit.begin(), unit.end(), 0.0);
    unit[pivot_position] = 1.0;
    basis_rows_[pivot_position] = entering;
    exchanged = true;
    ++stats_.basis_exchanges;
  }

  if (exchanged) {
    for (auto row : removals) coordinates_.erase(row);
    rows_ = rows;
    if (!refactor_from_coordinates()) {
      ++stats_.update_failures;
      return rebuild(rows);
    }
    ++stats_.local_transitions;
    stats_.row_additions += additions.size();
    stats_.row_removals += removals.size();
    stats_.transition_ms += milliseconds_since(start);
    return true;
  }

  auto trial = coordinate_factor_;
  for (const auto &[row, coordinate] : new_coordinates) {
    (void)row;
    if (!cholesky_rank_one(trial, rank, coordinate, +1)) {
      ++stats_.update_failures;
      return rebuild(rows);
    }
  }
  for (auto row : removals) {
    const auto found = coordinates_.find(row);
    if (found == coordinates_.end()
        || !cholesky_rank_one(trial, rank, found->second, -1)) {
      ++stats_.update_failures;
      return rebuild(rows);
    }
  }
  coordinate_factor_ = std::move(trial);
  for (auto row : removals) coordinates_.erase(row);
  rows_ = rows;
  ++stats_.local_transitions;
  stats_.row_additions += additions.size();
  stats_.row_removals += removals.size();
  stats_.transition_ms += milliseconds_since(start);
  return true;
}

bool RevisedBasisSolver::refactor_from_coordinates() {
  const int rank = static_cast<int>(basis_rows_.size());
  if (rank == 0) return false;
  const int m = static_cast<int>(fixture_.m);
  std::vector<double> basis_dense(static_cast<std::size_t>(rank) * m, 0.0);
  for (int basis = 0; basis < rank; ++basis) {
    const auto row = basis_rows_[basis];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      basis_dense[basis + static_cast<std::size_t>(rank)
                            * fixture_.indices[p]] = fixture_.values[p];
  }
  std::vector<double> basis_gram(static_cast<std::size_t>(rank) * rank, 0.0);
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, rank, m, 1.0,
              basis_dense.data(), rank, 0.0, basis_gram.data(), rank);
  if (!cholesky(basis_gram, rank)) return false;

  const int active = static_cast<int>(rows_.size());
  std::vector<double> coordinate_dense(
      static_cast<std::size_t>(active) * rank, 0.0);
  for (int local = 0; local < active; ++local) {
    const auto found = coordinates_.find(rows_[local]);
    if (found == coordinates_.end()) return false;
    for (int column = 0; column < rank; ++column)
      coordinate_dense[local + static_cast<std::size_t>(active) * column] =
          found->second[column];
  }
  std::vector<double> coordinate_gram(static_cast<std::size_t>(rank) * rank,
                                      0.0);
  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, rank, active, 1.0,
              coordinate_dense.data(), active, 0.0,
              coordinate_gram.data(), rank);
  if (!cholesky(coordinate_gram, rank)) return false;
  basis_factor_ = std::move(basis_gram);
  coordinate_factor_ = std::move(coordinate_gram);
  return true;
}

bool RevisedBasisSolver::form_solution(RevisedFaceSolution &solution) {
  const auto solve_start = Clock::now();
  const int rank = static_cast<int>(basis_rows_.size());
  if (!valid_ || rank == 0) return false;
  // Both cores are normal-equation factors.  Residuals cannot certify forward
  // accuracy on a poorly conditioned basis (the stored Brandy/Israel/Boeing2
  // controls demonstrate that directly), so fail closed before answering.
  constexpr double factor_diagonal_gate = 1e-3;
  if (diagonal_ratio(basis_factor_, rank) < factor_diagonal_gate
      || diagonal_ratio(coordinate_factor_, rank) < factor_diagonal_gate) {
    ++stats_.condition_declines;
    return false;
  }

  std::vector<double> adjusted_d = fixture_.d;
  if (!target_shift_.empty()) {
    for (auto row : rows_)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        adjusted_d[fixture_.indices[p]] -=
            fixture_.values[p] * target_shift_[row];
  }

  std::vector<double> ltb(rank, 0.0), cd(rank, 0.0);
  for (auto row : rows_) {
    const auto &coordinate = coordinates_.at(row);
    for (int j = 0; j < rank; ++j)
      ltb[j] += coordinate[j] * fixture_.b[row];
  }
  for (int basis = 0; basis < rank; ++basis) {
    const auto row = basis_rows_[basis];
    long double value = 0.0L;
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      value += static_cast<long double>(fixture_.values[p])
               * adjusted_d[fixture_.indices[p]];
    cd[basis] = static_cast<double>(value);
  }

  // ua = -C' (C C')^-1 (L' L)^-1 L'b
  std::vector<double> g_head = ltb;
  if (!solve_cholesky(coordinate_factor_, rank, g_head))
    return false;
  std::vector<double> ua_head = g_head;
  if (!solve_cholesky(basis_factor_, rank, ua_head)) return false;
  for (double &value : ua_head) value = -value;
  for (double &value : g_head) value = -value;

  // h-s = L (L'L)^-1 (C C')^-1 C(d-A's)
  std::vector<double> h_head = cd;
  if (!solve_cholesky(basis_factor_, rank, h_head)
      || !solve_cholesky(coordinate_factor_, rank, h_head))
    return false;
  // uc = C' (C C')^-1 (L'L)^-1 (C C')^-1 C(d-A's)
  std::vector<double> uc_head = h_head;
  if (!solve_cholesky(basis_factor_, rank, uc_head)) return false;

  solution = RevisedFaceSolution{};
  solution.rows = rows_;
  solution.rank = rank;
  solution.basis_diagonal_ratio = diagonal_ratio(basis_factor_, rank);
  solution.coordinate_diagonal_ratio =
      diagonal_ratio(coordinate_factor_, rank);
  solution.ua.assign(fixture_.m, 0.0);
  solution.uc.assign(fixture_.m, 0.0);
  for (int basis = 0; basis < rank; ++basis) {
    const auto row = basis_rows_[basis];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      solution.ua[fixture_.indices[p]] += fixture_.values[p] * ua_head[basis];
      solution.uc[fixture_.indices[p]] += fixture_.values[p] * uc_head[basis];
    }
  }
  solution.g.resize(rows_.size());
  solution.h.resize(rows_.size());
  for (std::size_t local = 0; local < rows_.size(); ++local) {
    const auto row = rows_[local];
    const auto &coordinate = coordinates_.at(row);
    double projected_b = 0.0, projected_d = 0.0;
    for (int j = 0; j < rank; ++j) {
      projected_b += coordinate[j] * g_head[j];
      projected_d += coordinate[j] * h_head[j];
    }
    solution.g[local] = fixture_.b[row] + projected_b;
    solution.h[local] = (target_shift_.empty() ? 0.0 : target_shift_[row])
                        + projected_d;
  }
  stats_.solve_ms += milliseconds_since(solve_start);

  const auto products_start = Clock::now();
  solution.bua.assign(fixture_.n, 0.0);
  solution.buc.assign(fixture_.n, 0.0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      solution.bua[row] += fixture_.values[p] * solution.ua[column];
      solution.buc[row] += fixture_.values[p] * solution.uc[column];
    }
  stats_.products_ms += milliseconds_since(products_start);

  std::vector<double> transpose_g(fixture_.m, 0.0),
                      dual_residual(fixture_.m, 0.0);
  for (std::size_t local = 0; local < rows_.size(); ++local) {
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      transpose_g[column] += fixture_.values[p] * solution.g[local];
      dual_residual[column] += fixture_.values[p] * solution.h[local];
    }
  }
  for (std::size_t column = 0; column < fixture_.m; ++column)
    dual_residual[column] -= fixture_.d[column];
  solution.dres = norm2(dual_residual) / std::max(1.0, norm2(fixture_.d));
  solution.piece_residual =
      inf_norm(transpose_g) / std::max(1.0, inf_norm(solution.g));
  const double residual = std::max(solution.dres, solution.piece_residual);
  if (!std::isfinite(residual) || residual > 1e-10) {
    ++stats_.residual_declines;
    return false;
  }
  return true;
}

bool RevisedBasisSolver::solve(const std::vector<std::uint32_t> &rows,
                               RevisedFaceSolution &solution) {
  ++stats_.calls;
  if (!std::is_sorted(rows.begin(), rows.end()) || !transition(rows)) {
    ++stats_.declines;
    valid_ = false;
    return false;
  }
  if (!form_solution(solution)) {
    ++stats_.declines;
    return false;
  }
  return true;
}

double relative_inf_error(const std::vector<double> &actual,
                          const std::vector<double> &expected) {
  if (actual.size() != expected.size())
    return std::numeric_limits<double>::infinity();
  double difference = 0.0, scale = 1.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    difference = std::max(difference, std::abs(actual[i] - expected[i]));
    scale = std::max(scale, std::abs(expected[i]));
  }
  return difference / scale;
}

}  // namespace twalker::revised
