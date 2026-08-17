#include "maintained_rowspace_solver.hpp"

#include <vecLib/cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>

extern "C" {
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda,
             int *info);
void dpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *a, const int *lda, double *b, const int *ldb,
             int *info);
void dgesdd_(const char *jobz, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt,
             const int *ldvt, double *work, const int *lwork, int *iwork,
             int *info);
void dormrz_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const int *l, const double *a, const int *lda,
             const double *tau, double *c, const int *ldc, double *work,
             const int *lwork, int *info);
}

namespace twalker::revised {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double inf_norm(const std::vector<double> &values) {
  double result = 0.0;
  for (double value : values) result = std::max(result, std::abs(value));
  return result;
}

double diagonal_ratio(const std::vector<double> &factor, int rank) {
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = 0.0;
  for (int j = 0; j < rank; ++j) {
    const double value =
        std::abs(factor[j + static_cast<std::size_t>(rank) * j]);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  return maximum > 0.0 ? minimum / maximum : 0.0;
}

bool cholesky_rank_one(std::vector<double> &factor, int rank,
                       std::vector<double> vector, int sign) {
  if (rank <= 0 || factor.size() != static_cast<std::size_t>(rank) * rank
      || vector.size() != static_cast<std::size_t>(rank)
      || (sign != 1 && sign != -1))
    return false;
  for (int column = 0; column < rank; ++column) {
    const auto diagonal_index =
        column + static_cast<std::size_t>(rank) * column;
    const double diagonal = factor[diagonal_index];
    const double square = diagonal * diagonal
                          + sign * vector[column] * vector[column];
    if (!(square > 0.0) || !std::isfinite(square)) return false;
    const double replacement = std::sqrt(square);
    const double c = replacement / diagonal;
    const double s = vector[column] / diagonal;
    factor[diagonal_index] = replacement;
    for (int row = column + 1; row < rank; ++row) {
      auto &entry = factor[row + static_cast<std::size_t>(rank) * column];
      const double old_entry = entry;
      entry = (old_entry + sign * s * vector[row]) / c;
      vector[row] = c * vector[row] - s * entry;
    }
  }
  return std::all_of(factor.begin(), factor.end(),
                     [](double value) { return std::isfinite(value); });
}

bool cholesky_append(const std::vector<double> &factor, int rank,
                     const std::vector<double> &cross, double diagonal,
                     std::vector<double> &enlarged) {
  if (cross.size() != static_cast<std::size_t>(rank)
      || factor.size() != static_cast<std::size_t>(rank) * rank
      || !(diagonal > 0.0))
    return false;
  std::vector<double> solved = cross;
  for (int row = 0; row < rank; ++row) {
    double value = solved[row];
    for (int column = 0; column < row; ++column)
      value -= factor[row + static_cast<std::size_t>(rank) * column]
               * solved[column];
    const double pivot =
        factor[row + static_cast<std::size_t>(rank) * row];
    if (!(std::abs(pivot) > 0.0)) return false;
    solved[row] = value / pivot;
  }
  double schur = diagonal;
  for (double value : solved) schur -= value * value;
  if (!(schur > 0.0) || !std::isfinite(schur)) return false;
  enlarged.assign(static_cast<std::size_t>(rank + 1) * (rank + 1), 0.0);
  for (int column = 0; column < rank; ++column)
    for (int row = column; row < rank; ++row)
      enlarged[row + static_cast<std::size_t>(rank + 1) * column] =
          factor[row + static_cast<std::size_t>(rank) * column];
  for (int column = 0; column < rank; ++column)
    enlarged[rank + static_cast<std::size_t>(rank + 1) * column] =
        solved[column];
  enlarged[rank + static_cast<std::size_t>(rank + 1) * rank] =
      std::sqrt(schur);
  return true;
}

}  // namespace

void MaintainedRowspaceSolver::invalidate() {
  valid_ = false;
  rows_.clear();
  rowspace_.clear();
  coordinates_.clear();
  factor_.clear();
  cross_b_.clear();
  rank_ = 0;
  updates_since_refactor_ = 0;
  cached_solution_valid_ = false;
  cached_solution_ = RevisedSlopeSolution{};
}

bool MaintainedRowspaceSolver::row_coordinates(
    std::uint32_t row, std::vector<double> &coordinate,
    std::vector<double> &residual, double &relative_residual) const {
  if (!valid_ || row >= fixture_.n || rank_ <= 0
      || rowspace_.size() != static_cast<std::size_t>(rank_) * fixture_.m)
    return false;
  coordinate.assign(rank_, 0.0);
  residual.assign(fixture_.m, 0.0);
  double row_square = 0.0;
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
    const auto column = fixture_.indices[p];
    const double value = fixture_.values[p];
    residual[column] = value;
    row_square += value * value;
    for (int component = 0; component < rank_; ++component)
      coordinate[component] +=
          rowspace_[component + static_cast<std::size_t>(rank_) * column]
          * value;
  }
  long double residual_square = 0.0L;
  for (std::size_t column = 0; column < fixture_.m; ++column) {
    long double projected = 0.0L;
    for (int component = 0; component < rank_; ++component)
      projected += static_cast<long double>(coordinate[component])
          * rowspace_[component
                      + static_cast<std::size_t>(rank_) * column];
    residual[column] -= static_cast<double>(projected);
    residual_square += static_cast<long double>(residual[column])
                       * residual[column];
  }
  relative_residual = std::sqrt(static_cast<double>(
      residual_square / std::max(1.0, row_square)));
  return std::isfinite(relative_residual);
}

bool MaintainedRowspaceSolver::refactor() {
  const int active = static_cast<int>(rows_.size());
  if (active < rank_ || rank_ <= 0
      || coordinates_.size() != static_cast<std::size_t>(active) * rank_)
    return false;
  factor_.assign(static_cast<std::size_t>(rank_) * rank_, 0.0);
  // Row-major C (active-by-rank) is the same memory as column-major C'.
  // Therefore this forms C'*C directly into a column-major lower triangle.
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, rank_, active, 1.0,
              coordinates_.data(), rank_, 0.0, factor_.data(), rank_);
  const char lower = 'L';
  int info = 0;
  dpotrf_(&lower, &rank_, factor_.data(), &rank_, &info);
  if (info != 0 || diagonal_ratio(factor_, rank_) < 1e-12) return false;
  cross_b_.assign(rank_, 0.0);
  for (int local = 0; local < active; ++local)
    for (int component = 0; component < rank_; ++component)
      cross_b_[component] +=
          coordinates_[static_cast<std::size_t>(local) * rank_ + component]
          * fixture_.b[rows_[local]];
  ++stats_.refactors;
  updates_since_refactor_ = 0;
  return true;
}

bool MaintainedRowspaceSolver::seed(
    const std::vector<std::uint32_t> &rows, const FaceSolution &direct) {
  const auto started = Clock::now();
  invalidate();
  if (rows.empty() || direct.rows != rows || direct.rank <= 0
      || direct.rank > static_cast<std::int64_t>(fixture_.m))
    return false;
  rows_ = rows;
  rank_ = static_cast<int>(direct.rank);
  if (direct.svd_row_space.size()
      == static_cast<std::size_t>(rank_) * fixture_.m) {
    rowspace_ = direct.svd_row_space;
  } else if (direct.factored_seed_rank == rank_
             && direct.factored_rz_core.size()
                    == static_cast<std::size_t>(rank_) * fixture_.m
             && direct.factored_rz_tau.size()
                    == static_cast<std::size_t>(rank_)
             && direct.factored_permutation.size() == fixture_.m) {
    const int columns = static_cast<int>(fixture_.m);
    std::vector<double> orthonormal_columns(
        fixture_.m * static_cast<std::size_t>(rank_), 0.0);
    for (int component = 0; component < rank_; ++component)
      orthonormal_columns[
          component + static_cast<std::size_t>(columns) * component] = 1.0;
    if (rank_ < columns) {
      const char side = 'L', transpose = 'T';
      const int tail = columns - rank_;
      int info = 0, lwork = -1;
      double query = 0.0;
      dormrz_(&side, &transpose, &columns, &rank_, &rank_, &tail,
              direct.factored_rz_core.data(), &rank_,
              direct.factored_rz_tau.data(), orthonormal_columns.data(),
              &columns, &query, &lwork, &info);
      if (info != 0 || !std::isfinite(query)) return false;
      lwork = std::max(1, static_cast<int>(std::ceil(query)));
      std::vector<double> work(lwork);
      dormrz_(&side, &transpose, &columns, &rank_, &rank_, &tail,
              direct.factored_rz_core.data(), &rank_,
              direct.factored_rz_tau.data(), orthonormal_columns.data(),
              &columns, work.data(), &lwork, &info);
      if (info != 0) return false;
    }
    rowspace_.assign(static_cast<std::size_t>(rank_) * fixture_.m, 0.0);
    for (std::size_t position = 0; position < fixture_.m; ++position) {
      const auto original = direct.factored_permutation[position];
      if (original < 0
          || original >= static_cast<std::int64_t>(fixture_.m))
        return false;
      for (int component = 0; component < rank_; ++component)
        rowspace_[component
                  + static_cast<std::size_t>(rank_) * original] =
            orthonormal_columns[
                position + fixture_.m * static_cast<std::size_t>(component)];
    }
  } else {
    return false;
  }
  const int active = static_cast<int>(rows_.size());
  coordinates_.assign(static_cast<std::size_t>(active) * rank_, 0.0);
  cross_b_.assign(rank_, 0.0);
  for (int local = 0; local < active; ++local) {
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const double value = fixture_.values[p];
      for (int component = 0; component < rank_; ++component)
        coordinates_[static_cast<std::size_t>(local) * rank_ + component] +=
            value
            * rowspace_[component
                        + static_cast<std::size_t>(rank_) * column];
    }
    for (int component = 0; component < rank_; ++component)
      cross_b_[component] +=
          coordinates_[static_cast<std::size_t>(local) * rank_ + component]
          * fixture_.b[row];
  }

  double orthogonality = 0.0;
  for (int left = 0; left < rank_; ++left)
    for (int right = 0; right < rank_; ++right) {
      double product = 0.0;
      for (std::size_t column = 0; column < fixture_.m; ++column)
        product += rowspace_[left + static_cast<std::size_t>(rank_) * column]
            * rowspace_[right
                        + static_cast<std::size_t>(rank_) * column];
      orthogonality = std::max(
          orthogonality,
          std::abs(product - (left == right ? 1.0 : 0.0)));
    }
  double representation = 0.0;
  for (int local = 0; local < active; ++local) {
    std::vector<double> dense(fixture_.m, 0.0);
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      dense[fixture_.indices[p]] = fixture_.values[p];
    double row_scale = 1.0;
    for (double value : dense)
      row_scale = std::max(row_scale, std::abs(value));
    for (std::size_t column = 0; column < fixture_.m; ++column) {
      double reconstructed = 0.0;
      for (int component = 0; component < rank_; ++component)
        reconstructed +=
            coordinates_[static_cast<std::size_t>(local) * rank_ + component]
            * rowspace_[component
                        + static_cast<std::size_t>(rank_) * column];
      representation = std::max(
          representation,
          std::abs(reconstructed - dense[column]) / row_scale);
    }
  }
  stats_.worst_orthogonality =
      std::max(stats_.worst_orthogonality, orthogonality);
  stats_.worst_rowspace_residual =
      std::max(stats_.worst_rowspace_residual, representation);
  if (orthogonality > 2e-10 || representation > 2e-10 || !refactor()) {
    invalidate();
    return false;
  }
  valid_ = true;
  ++stats_.seeds;
  stats_.seed_ms += milliseconds_since(started);
  return true;
}

bool MaintainedRowspaceSolver::seed_from_rows(
    const std::vector<std::uint32_t> &rows, std::int64_t expected_rank) {
  const auto started = Clock::now();
  if (rows.empty() || rows.size() > static_cast<std::size_t>(
          std::numeric_limits<int>::max())
      || fixture_.m > static_cast<std::size_t>(
          std::numeric_limits<int>::max()))
    return false;
  const int active = static_cast<int>(rows.size());
  const int columns = static_cast<int>(fixture_.m);
  const int thin = std::min(active, columns);
  if (thin <= 0) return false;
  std::vector<double> matrix(static_cast<std::size_t>(active) * columns, 0.0);
  for (int local = 0; local < active; ++local) {
    const auto row = rows[local];
    if (row >= fixture_.n) return false;
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      matrix[local + static_cast<std::size_t>(active) * fixture_.indices[p]] =
          fixture_.values[p];
  }
  std::vector<double> singular(thin);
  std::vector<double> left(static_cast<std::size_t>(active) * thin);
  std::vector<double> transpose(static_cast<std::size_t>(thin) * columns);
  std::vector<int> iwork(8 * thin);
  const char job = 'S';
  const int lda = active, ldu = active, ldvt = thin;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &active, &columns, matrix.data(), &lda, singular.data(),
          left.data(), &ldu, transpose.data(), &ldvt, &query, &lwork,
          iwork.data(), &info);
  if (info != 0 || !std::isfinite(query)) return false;
  lwork = std::max(1, static_cast<int>(std::ceil(query)));
  std::vector<double> work(lwork);
  dgesdd_(&job, &active, &columns, matrix.data(), &lda, singular.data(),
          left.data(), &ldu, transpose.data(), &ldvt, work.data(), &lwork,
          iwork.data(), &info);
  if (info != 0) return false;
  int rank = 0;
  const double cutoff = singular.front() * std::max(active, columns)
                        * std::numeric_limits<double>::epsilon();
  while (rank < thin && singular[rank] > cutoff) ++rank;
  if (rank <= 0 || (expected_rank >= 0 && expected_rank != rank)) return false;
  FaceSolution artifact;
  artifact.rows = rows;
  artifact.rank = rank;
  artifact.svd_row_space.resize(static_cast<std::size_t>(rank) * columns);
  for (int column = 0; column < columns; ++column)
    for (int component = 0; component < rank; ++component)
      artifact.svd_row_space[
          component + static_cast<std::size_t>(rank) * column] =
          transpose[component + static_cast<std::size_t>(thin) * column];
  const double previous_seed_ms = stats_.seed_ms;
  const bool seeded = seed(rows, artifact);
  stats_.seed_ms = previous_seed_ms + milliseconds_since(started);
  return seeded;
}

bool MaintainedRowspaceSolver::add_row(std::uint32_t row) {
  std::vector<double> coordinate, residual;
  double relative_residual = 0.0;
  if (!row_coordinates(row, coordinate, residual, relative_residual))
    return false;
  const auto insertion = std::lower_bound(rows_.begin(), rows_.end(), row);
  const auto local = static_cast<std::size_t>(insertion - rows_.begin());
  if (relative_residual <= 1e-10) {
    auto trial_factor = factor_;
    if (!cholesky_rank_one(trial_factor, rank_, coordinate, +1)
        || diagonal_ratio(trial_factor, rank_) < 1e-12)
      return false;
    factor_ = std::move(trial_factor);
    for (int component = 0; component < rank_; ++component)
      cross_b_[component] += coordinate[component] * fixture_.b[row];
    coordinates_.insert(coordinates_.begin() + local * rank_,
                        coordinate.begin(), coordinate.end());
    rows_.insert(insertion, row);
    ++stats_.additions;
    ++updates_since_refactor_;
    return true;
  }
  if (relative_residual < 1e-7) {
    ++stats_.rank_change_declines;
    return false;
  }

  double residual_square = 0.0;
  for (double value : residual) residual_square += value * value;
  const double rho = std::sqrt(residual_square);
  if (!(rho > 0.0) || !std::isfinite(rho)) return false;
  auto updated_factor = factor_;
  if (!cholesky_rank_one(updated_factor, rank_, coordinate, +1)) return false;
  std::vector<double> cross(rank_);
  for (int component = 0; component < rank_; ++component)
    cross[component] = coordinate[component] * rho;
  std::vector<double> enlarged_factor;
  if (!cholesky_append(updated_factor, rank_, cross, residual_square,
                       enlarged_factor))
    return false;

  const int old_rank = rank_;
  const int new_rank = rank_ + 1;
  std::vector<double> enlarged_rowspace(
      static_cast<std::size_t>(new_rank) * fixture_.m, 0.0);
  for (std::size_t column = 0; column < fixture_.m; ++column) {
    for (int component = 0; component < old_rank; ++component)
      enlarged_rowspace[component
                        + static_cast<std::size_t>(new_rank) * column] =
          rowspace_[component
                    + static_cast<std::size_t>(old_rank) * column];
    enlarged_rowspace[old_rank
                      + static_cast<std::size_t>(new_rank) * column] =
        residual[column] / rho;
  }
  std::vector<double> enlarged_coordinates;
  enlarged_coordinates.reserve((rows_.size() + 1) * new_rank);
  for (std::size_t target = 0; target < rows_.size() + 1; ++target) {
    if (target == local) {
      enlarged_coordinates.insert(enlarged_coordinates.end(),
                                  coordinate.begin(), coordinate.end());
      enlarged_coordinates.push_back(rho);
    } else {
      const auto source = target < local ? target : target - 1;
      enlarged_coordinates.insert(
          enlarged_coordinates.end(),
          coordinates_.begin() + source * old_rank,
          coordinates_.begin() + (source + 1) * old_rank);
      enlarged_coordinates.push_back(0.0);
    }
  }
  cross_b_.push_back(rho * fixture_.b[row]);
  for (int component = 0; component < old_rank; ++component)
    cross_b_[component] += coordinate[component] * fixture_.b[row];
  rank_ = new_rank;
  rowspace_ = std::move(enlarged_rowspace);
  coordinates_ = std::move(enlarged_coordinates);
  factor_ = std::move(enlarged_factor);
  rows_.insert(insertion, row);
  ++stats_.additions;
  ++stats_.rank_increases;
  ++updates_since_refactor_;
  return true;
}

bool MaintainedRowspaceSolver::remove_row(std::uint32_t row) {
  const auto found = std::lower_bound(rows_.begin(), rows_.end(), row);
  if (found == rows_.end() || *found != row || rows_.size() <= 1)
    return false;
  const auto local = static_cast<std::size_t>(found - rows_.begin());
  std::vector<double> coordinate(
      coordinates_.begin() + local * rank_,
      coordinates_.begin() + (local + 1) * rank_);
  auto trial_factor = factor_;
  bool ordinary = cholesky_rank_one(trial_factor, rank_, coordinate, -1)
                  && diagonal_ratio(trial_factor, rank_) >= 1e-12;
  if (!ordinary) {
    // Decide numerical rank from a freshly formed C'*C, not from a long
    // sequence of updated factors.
    if (refactor()) {
      if (std::getenv("TWALKER_MAINTAINED_RANK_DROP")
          && remove_rank_row(row, local, coordinate))
        return true;
      trial_factor = factor_;
      ordinary = cholesky_rank_one(trial_factor, rank_, coordinate, -1)
                 && diagonal_ratio(trial_factor, rank_) >= 1e-12;
    }
  }
  if (!ordinary) {
    ++stats_.rank_change_declines;
    return false;
  }
  factor_ = std::move(trial_factor);
  for (int component = 0; component < rank_; ++component)
    cross_b_[component] -= coordinate[component] * fixture_.b[row];
  coordinates_.erase(coordinates_.begin() + local * rank_,
                     coordinates_.begin() + (local + 1) * rank_);
  rows_.erase(found);
  ++stats_.removals;
  ++updates_since_refactor_;
  return true;
}

bool MaintainedRowspaceSolver::remove_rank_row(
    std::uint32_t row, std::size_t local,
    const std::vector<double> &coordinate) {
  if (rank_ <= 1 || coordinate.size() != static_cast<std::size_t>(rank_)
      || local >= rows_.size())
    return false;
  // If c is the essential row, z=G^{-1}c spans the null space of
  // G-c*c'.  Its leverage c'*z is one.  This test separates a true rank loss
  // from a merely unstable downdate.
  std::vector<double> null_direction = coordinate;
  const char lower = 'L';
  const int rhs_count = 1;
  int info = 0;
  dpotrs_(&lower, &rank_, &rhs_count, factor_.data(), &rank_,
          null_direction.data(), &rank_, &info);
  if (info != 0) return false;
  double leverage = 0.0, norm2 = 0.0;
  for (int component = 0; component < rank_; ++component) {
    leverage += coordinate[component] * null_direction[component];
    norm2 += null_direction[component] * null_direction[component];
  }
  if (std::abs(leverage - 1.0) > 2e-8 || !(norm2 > 0.0)
      || !std::isfinite(norm2))
    return false;
  const double inverse_norm = 1.0 / std::sqrt(norm2);
  for (double &value : null_direction) value *= inverse_norm;

  // Symmetric Householder H maps z to the final coordinate.  Then C*H has a
  // zero final column after the essential row is removed, while H*Q remains
  // an orthonormal row-space basis.  Drop that coordinate and refactor r-1.
  std::vector<double> reflector = null_direction;
  reflector.back() -= 1.0;
  double reflector_square = 0.0;
  for (double value : reflector) reflector_square += value * value;
  const double beta = reflector_square > 1e-28
                          ? 2.0 / reflector_square
                          : 0.0;
  const int old_rank = rank_;
  const int new_rank = rank_ - 1;
  std::vector<double> reduced_coordinates;
  reduced_coordinates.reserve((rows_.size() - 1) * new_rank);
  double discarded_max = 0.0, coordinate_scale = 1.0;
  for (std::size_t source = 0; source < rows_.size(); ++source) {
    if (source == local) continue;
    const double *old = coordinates_.data() + source * old_rank;
    double projection = 0.0;
    for (int component = 0; component < old_rank; ++component) {
      projection += old[component] * reflector[component];
      coordinate_scale = std::max(coordinate_scale, std::abs(old[component]));
    }
    for (int component = 0; component < new_rank; ++component)
      reduced_coordinates.push_back(
          old[component] - beta * projection * reflector[component]);
    discarded_max = std::max(
        discarded_max,
        std::abs(old[new_rank] - beta * projection * reflector[new_rank]));
  }
  if (discarded_max > 2e-8 * coordinate_scale) return false;

  std::vector<double> reduced_rowspace(
      static_cast<std::size_t>(new_rank) * fixture_.m, 0.0);
  for (std::size_t column = 0; column < fixture_.m; ++column) {
    double projection = 0.0;
    for (int component = 0; component < old_rank; ++component)
      projection += reflector[component]
          * rowspace_[component
                      + static_cast<std::size_t>(old_rank) * column];
    for (int component = 0; component < new_rank; ++component)
      reduced_rowspace[
          component + static_cast<std::size_t>(new_rank) * column] =
          rowspace_[component
                    + static_cast<std::size_t>(old_rank) * column]
          - beta * reflector[component] * projection;
  }
  rows_.erase(rows_.begin() + static_cast<std::ptrdiff_t>(local));
  coordinates_ = std::move(reduced_coordinates);
  rowspace_ = std::move(reduced_rowspace);
  rank_ = new_rank;
  factor_.clear();
  cross_b_.clear();
  if (!refactor()) return false;
  ++stats_.removals;
  ++stats_.rank_decreases;
  return true;
}

bool MaintainedRowspaceSolver::transition(
    const std::vector<std::uint32_t> &rows) {
  const auto started = Clock::now();
  if (!valid_ || !std::is_sorted(rows.begin(), rows.end())) return false;
  if (rows == rows_) return true;
  cached_solution_valid_ = false;
  std::vector<std::uint32_t> additions, removals;
  std::set_difference(rows.begin(), rows.end(), rows_.begin(), rows_.end(),
                      std::back_inserter(additions));
  std::set_difference(rows_.begin(), rows_.end(), rows.begin(), rows.end(),
                      std::back_inserter(removals));
  if (additions.size() + removals.size() > 8) return false;
  const auto exchanges = std::min(additions.size(), removals.size());
  for (std::size_t i = 0; i < exchanges; ++i)
    if (!add_row(additions[i]) || !remove_row(removals[i])) return false;
  for (std::size_t i = exchanges; i < additions.size(); ++i)
    if (!add_row(additions[i])) return false;
  for (std::size_t i = exchanges; i < removals.size(); ++i)
    if (!remove_row(removals[i])) return false;
  if (rows_ != rows) return false;
  if (updates_since_refactor_ >= 64 && !refactor()) return false;
  ++stats_.local_transitions;
  stats_.transition_ms += milliseconds_since(started);
  return true;
}

bool MaintainedRowspaceSolver::form_solution(
    RevisedSlopeSolution &solution) {
  const auto started = Clock::now();
  if (!valid_ || rank_ <= 0) return false;
  // Re-form the small right-hand side from the maintained coordinates.  The
  // incremental cross_b_ is useful bookkeeping, but must not be allowed to
  // accumulate enough rounding error to choose a different path event.
  std::vector<double> exact_cross(rank_, 0.0);
  std::vector<double> active_b(rows_.size());
  for (std::size_t local = 0; local < rows_.size(); ++local)
    active_b[local] = fixture_.b[rows_[local]];
  cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(rows_.size()),
              rank_, 1.0, coordinates_.data(), rank_, active_b.data(), 1,
              0.0, exact_cross.data(), 1);
  std::vector<double> alpha = exact_cross;
  for (double &value : alpha) value = -value;
  const char lower = 'L';
  const int rhs_count = 1;
  int info = 0;
  dpotrs_(&lower, &rank_, &rhs_count, factor_.data(), &rank_, alpha.data(),
          &rank_, &info);
  if (info != 0) return false;

  // Cholesky updates preserve the factor cheaply but their rounding error is
  // amplified on deficient faces.  Refine against the exact maintained
  // operator C'*C; this is O(|W|r) and retains the cheap branch.
  std::vector<double> active_product(rows_.size(), 0.0);
  std::vector<double> correction(rank_, 0.0);
  for (int iteration = 0; iteration < 3; ++iteration) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(rows_.size()), rank_, 1.0,
                coordinates_.data(), rank_, alpha.data(), 1, 0.0,
                active_product.data(), 1);
    correction = exact_cross;
    for (double &value : correction) value = -value;
    cblas_dgemv(CblasRowMajor, CblasTrans,
                static_cast<int>(rows_.size()), rank_, -1.0,
                coordinates_.data(), rank_, active_product.data(), 1, 1.0,
                correction.data(), 1);
    const double correction_rhs = inf_norm(correction);
    if (correction_rhs <= 4e-15 * std::max(1.0, inf_norm(exact_cross))) break;
    dpotrs_(&lower, &rank_, &rhs_count, factor_.data(), &rank_,
            correction.data(), &rank_, &info);
    if (info != 0) return false;
    for (int component = 0; component < rank_; ++component)
      alpha[component] += correction[component];
  }

  solution = RevisedSlopeSolution{};
  solution.rows = rows_;
  solution.rank = rank_;
  solution.ua.assign(fixture_.m, 0.0);
  cblas_dgemv(CblasColMajor, CblasTrans, rank_,
              static_cast<int>(fixture_.m), 1.0, rowspace_.data(), rank_,
              alpha.data(), 1, 0.0, solution.ua.data(), 1);
  solution.g.resize(rows_.size());
  for (std::size_t local = 0; local < rows_.size(); ++local)
    solution.g[local] = fixture_.b[rows_[local]];
  cblas_dgemv(CblasRowMajor, CblasNoTrans,
              static_cast<int>(rows_.size()), rank_, 1.0,
              coordinates_.data(), rank_, alpha.data(), 1, 1.0,
              solution.g.data(), 1);
  solution.bua.assign(fixture_.n, 0.0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      solution.bua[row] +=
          fixture_.values[p] * solution.ua[fixture_.indices[p]];

  std::vector<double> transpose(fixture_.m, 0.0);
  for (std::size_t local = 0; local < rows_.size(); ++local)
    for (auto p = fixture_.indptr[rows_[local]];
         p < fixture_.indptr[rows_[local] + 1]; ++p)
      transpose[fixture_.indices[p]] += fixture_.values[p] * solution.g[local];
  solution.slope_residual =
      inf_norm(transpose) / std::max(1.0, inf_norm(solution.g));
  stats_.worst_slope_residual =
      std::max(stats_.worst_slope_residual, solution.slope_residual);
  stats_.solve_ms += milliseconds_since(started);
  if (!std::isfinite(solution.slope_residual)
      || solution.slope_residual > 2e-10)
    return false;
  ++stats_.solves;
  return true;
}

bool MaintainedRowspaceSolver::solve(
    const std::vector<std::uint32_t> &rows,
    RevisedSlopeSolution &solution) {
  ++stats_.calls;
  if (!valid_) return false;
  if (rows == rows_ && cached_solution_valid_) {
    solution = cached_solution_;
    return true;
  }
  if (!transition(rows)) {
    ++stats_.numerical_declines;
    invalidate();
    return false;
  }
  if (!form_solution(solution)) {
    // A stale updated factor is recoverable without re-revealing rank.
    if (!refactor() || !form_solution(solution)) {
      ++stats_.numerical_declines;
      invalidate();
      return false;
    }
  }
  cached_solution_ = solution;
  cached_solution_valid_ = true;
  return true;
}

}  // namespace twalker::revised
