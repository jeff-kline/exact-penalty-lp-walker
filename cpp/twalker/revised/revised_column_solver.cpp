#include "revised_column_solver.hpp"

#include <SuiteSparseQR_C.h>
#include <vecLib/cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

extern "C" {
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda,
             int *info);
void dpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *a, const int *lda, double *b, const int *ldb,
             int *info);
void dgeqrf_(const int *m, const int *n, double *a, const int *lda,
             double *tau, double *work, const int *lwork, int *info);
void dgels_(const char *trans, const int *m, const int *n, const int *nrhs,
            double *a, const int *lda, double *b, const int *ldb,
            double *work, const int *lwork, int *info);
void dgelss_(const int *m, const int *n, const int *nrhs, double *a,
             const int *lda, double *b, const int *ldb, double *singular,
            const double *rcond, int *rank, double *work, const int *lwork,
            int *info);
void dtrtrs_(const char *uplo, const char *trans, const char *diag,
             const int *n, const int *nrhs, const double *a, const int *lda,
             double *b, const int *ldb, int *info);
void dormrz_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const int *l, const double *a, const int *lda,
             const double *tau, double *c, const int *ldc, double *work,
             const int *lwork, int *info);
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
  const char lower = 'L';
  int info = 0;
  dpotrf_(&lower, &n, matrix.data(), &n, &info);
  if (info != 0) return false;
  for (int column = 0; column < n; ++column)
    for (int row = 0; row < column; ++row)
      matrix[row + static_cast<std::size_t>(n) * column] = 0.0;
  return true;
}

// Return L=R' where A=Q*R.  Thus A'*A=L*L', but unlike a Cholesky
// factorization of an explicitly formed Gram matrix this retains the
// backward stability and rank information of QR.
bool qr_square_root(std::vector<double> matrix, int rows, int columns,
                    std::vector<double> &lower) {
  if (rows < columns || columns <= 0
      || matrix.size() != static_cast<std::size_t>(rows) * columns)
    return false;
  std::vector<double> tau(columns);
  int info = 0, query = -1;
  double requested = 0.0;
  dgeqrf_(&rows, &columns, matrix.data(), &rows, tau.data(), &requested,
          &query, &info);
  if (info != 0 || !std::isfinite(requested)) return false;
  const int work_size = std::max(1, static_cast<int>(requested));
  std::vector<double> work(work_size);
  dgeqrf_(&rows, &columns, matrix.data(), &rows, tau.data(), work.data(),
          &work_size, &info);
  if (info != 0) return false;
  lower.assign(static_cast<std::size_t>(columns) * columns, 0.0);
  for (int rrow = 0; rrow < columns; ++rrow) {
    const double sign = matrix[rrow + static_cast<std::size_t>(rows) * rrow]
                                < 0.0
                            ? -1.0
                            : 1.0;
    for (int rcolumn = rrow; rcolumn < columns; ++rcolumn)
      lower[rcolumn + static_cast<std::size_t>(columns) * rrow] =
          sign * matrix[rrow + static_cast<std::size_t>(rows) * rcolumn];
  }
  for (int diagonal = 0; diagonal < columns; ++diagonal)
    if (!(lower[diagonal
                + static_cast<std::size_t>(columns) * diagonal] > 0.0))
      return false;
  return true;
}

bool qr_least_squares(std::vector<double> design, int rows, int columns,
                      std::vector<double> right, int right_count,
                      std::vector<double> &solution) {
  if (rows < columns || columns <= 0 || right_count <= 0
      || design.size() != static_cast<std::size_t>(rows) * columns
      || right.size() != static_cast<std::size_t>(rows) * right_count)
    return false;
  const char no_transpose = 'N';
  int info = 0, query = -1;
  double requested = 0.0;
  dgels_(&no_transpose, &rows, &columns, &right_count, design.data(), &rows,
         right.data(), &rows, &requested, &query, &info);
  if (info != 0 || !std::isfinite(requested)) return false;
  const int work_size = std::max(1, static_cast<int>(requested));
  std::vector<double> work(work_size);
  dgels_(&no_transpose, &rows, &columns, &right_count, design.data(), &rows,
         right.data(), &rows, work.data(), &work_size, &info);
  if (info != 0) return false;
  solution.assign(static_cast<std::size_t>(columns) * right_count, 0.0);
  for (int rhs = 0; rhs < right_count; ++rhs)
    for (int row = 0; row < columns; ++row)
      solution[row + static_cast<std::size_t>(columns) * rhs] =
          right[row + static_cast<std::size_t>(rows) * rhs];
  return true;
}

bool solve_cholesky(const std::vector<double> &factor, int n,
                    std::vector<double> &right, int right_count = 1) {
  if (n <= 0 || right.size() != static_cast<std::size_t>(n) * right_count)
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
    const long double radicand =
        static_cast<long double>(diagonal) * diagonal
        + sign * static_cast<long double>(x[k]) * x[k];
    if (!(diagonal > 0.0) || !(radicand > 0.0L)) return false;
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

// Delete one row/column from A=L*L'.  In upper-triangular form R=L',
// deleting a column leaves one subdiagonal which is chased out with Givens
// rotations.  This is the same O(n^2) operation used by revised-simplex
// basis maintenance; it avoids refactoring the retained principal minor.
bool cholesky_delete(const std::vector<double> &factor, int n, int deleted,
                     std::vector<double> &result) {
  if (n <= 1 || deleted < 0 || deleted >= n) return false;
  const int reduced = n - 1;
  std::vector<double> work(static_cast<std::size_t>(n) * reduced, 0.0);
  for (int column = 0; column < reduced; ++column) {
    const int source_column = column < deleted ? column : column + 1;
    for (int row = 0; row <= source_column; ++row)
      work[row + static_cast<std::size_t>(n) * column] =
          factor[source_column + static_cast<std::size_t>(n) * row];
  }
  for (int column = deleted; column < reduced; ++column) {
    const auto top_index = column + static_cast<std::size_t>(n) * column;
    const auto bottom_index = column + 1
                              + static_cast<std::size_t>(n) * column;
    const double a = work[top_index], b = work[bottom_index];
    const double radius = std::hypot(a, b);
    if (!(radius > 0.0) || !std::isfinite(radius)) return false;
    const double c = a / radius, s = b / radius;
    for (int trailing = column; trailing < reduced; ++trailing) {
      const auto x_index = column + static_cast<std::size_t>(n) * trailing;
      const auto y_index = column + 1
                           + static_cast<std::size_t>(n) * trailing;
      const double x = work[x_index], y = work[y_index];
      work[x_index] = c * x + s * y;
      work[y_index] = -s * x + c * y;
    }
  }
  result.assign(static_cast<std::size_t>(reduced) * reduced, 0.0);
  for (int column = 0; column < reduced; ++column)
    for (int row = column; row < reduced; ++row)
      result[row + static_cast<std::size_t>(reduced) * column] =
          work[column + static_cast<std::size_t>(n) * row];
  return true;
}

bool cholesky_append(const std::vector<double> &factor, int n,
                     const std::vector<double> &cross, double diagonal,
                     std::vector<double> &result) {
  if (cross.size() != static_cast<std::size_t>(n)) return false;
  std::vector<double> tail = cross;
  // L tail = cross.
  for (int row = 0; row < n; ++row) {
    long double value = tail[row];
    for (int column = 0; column < row; ++column)
      value -= static_cast<long double>(factor[
                   row + static_cast<std::size_t>(n) * column])
               * tail[column];
    const double pivot = factor[row + static_cast<std::size_t>(n) * row];
    if (!(pivot > 0.0)) return false;
    tail[row] = static_cast<double>(value / pivot);
  }
  long double schur = diagonal;
  for (double value : tail)
    schur -= static_cast<long double>(value) * value;
  if (!(schur > 0.0L)) return false;
  result.assign(static_cast<std::size_t>(n + 1) * (n + 1), 0.0);
  for (int column = 0; column < n; ++column)
    for (int row = column; row < n; ++row)
      result[row + static_cast<std::size_t>(n + 1) * column] =
          factor[row + static_cast<std::size_t>(n) * column];
  for (int row = 0; row < n; ++row)
    result[n + static_cast<std::size_t>(n + 1) * row] = tail[row];
  result[n + static_cast<std::size_t>(n + 1) * n] =
      std::sqrt(static_cast<double>(schur));
  return true;
}

// QR update for [C 0; c' rho] given L=R' from C=Q*R.  Rotating the appended
// row directly is stable even when the new Schur complement is tiny; forming
// rho^2-cross'G^-1cross is not.
bool qr_append_row_and_column(const std::vector<double> &factor, int n,
                              const std::vector<double> &coordinate,
                              double rho, std::vector<double> &result) {
  if (factor.size() != static_cast<std::size_t>(n) * n
      || coordinate.size() != static_cast<std::size_t>(n)
      || !(rho > 0.0))
    return false;
  std::vector<double> work(n + 1);
  std::copy(coordinate.begin(), coordinate.end(), work.begin());
  work[n] = rho;
  result.assign(static_cast<std::size_t>(n + 1) * (n + 1), 0.0);
  for (int k = 0; k < n; ++k) {
    const double diagonal =
        factor[k + static_cast<std::size_t>(n) * k];
    const double radius = std::hypot(diagonal, work[k]);
    if (!(radius > 0.0) || !std::isfinite(radius)) return false;
    const double c = diagonal / radius, s = work[k] / radius;
    for (int column = k; column < n; ++column) {
      const double x =
          factor[column + static_cast<std::size_t>(n) * k];
      const double y = work[column];
      result[column + static_cast<std::size_t>(n + 1) * k] = c * x + s * y;
      work[column] = -s * x + c * y;
    }
    work[n] *= c;
  }
  if (!(std::abs(work[n]) > 0.0) || !std::isfinite(work[n])) return false;
  result[n + static_cast<std::size_t>(n + 1) * n] = std::abs(work[n]);
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

RevisedColumnSolver::RevisedColumnSolver(
    const Fixture &fixture, std::vector<double> target_shift,
    bool centered_slope_mode, bool force_shared_recurrence)
    : fixture_(fixture), target_shift_(std::move(target_shift)),
      centered_slope_mode_(centered_slope_mode),
      forced_shared_recurrence_(force_shared_recurrence) {
  // Walker represents "no nudge" as an n-vector of exact zeros.  Normalize
  // that representation here so the coefficient recurrence is not
  // needlessly disabled on the default path.
  if (std::all_of(target_shift_.begin(), target_shift_.end(),
                  [](double value) { return value == 0.0; }))
    target_shift_.clear();
  cholmod_l_start(&common_);
  common_.final_ll = 0;
  common_.print = 0;
  persistent_rank_updates_ =
      std::getenv("TWALKER_REVISED_PERSIST_RANK") != nullptr
      || std::getenv("TWALKER_REVISED_RECURRENCE") != nullptr;
  coefficient_recurrence_ = centered_slope_mode_
      || std::getenv("TWALKER_REVISED_RECURRENCE") != nullptr;
  shared_direct_seed_ = centered_slope_mode_
      || std::getenv("TWALKER_REVISED_SHARE_DIRECT_SEED") != nullptr;
  factored_direct_seed_ = !centered_slope_mode_
      && std::getenv("TWALKER_DISABLE_REVISED_FACTORED_SEED") == nullptr;
  reduced_equilibration_ =
      std::getenv("TWALKER_REVISED_EQUILIBRATE") != nullptr;
  shadow_square_root_ =
      std::getenv("TWALKER_REVISED_SHADOW_SQRT") != nullptr;
  shadow_solve_only_ =
      std::getenv("TWALKER_REVISED_SHADOW_SOLVE") != nullptr;
  rz_orthonormal_ =
      std::getenv("TWALKER_REVISED_RZ_ORTHONORMAL") != nullptr;
  svd_orthonormal_ =
      std::getenv("TWALKER_DISABLE_REVISED_SVD_ROWSPACE") == nullptr;
  if (const char *raw = std::getenv("TWALKER_REVISED_MAX_SEEDS"))
    max_direct_seeds_ = static_cast<std::uint32_t>(std::stoul(raw));
  if (centered_slope_mode_) max_direct_seeds_ = 128;
  if (forced_shared_recurrence_) {
    persistent_rank_updates_ = true;
    coefficient_recurrence_ = true;
    shared_direct_seed_ = true;
    factored_direct_seed_ = false;
    max_direct_seeds_ = 128;
  }
  if (shared_direct_seed_ && factored_direct_seed_)
    throw std::runtime_error(
        "dense and factored direct seeds are mutually exclusive");
}

RevisedColumnSolver::~RevisedColumnSolver() {
  if (reduced_update_column_)
    cholmod_l_free_sparse(&reduced_update_column_, &common_);
  if (reduced_factor_) cholmod_l_free_factor(&reduced_factor_, &common_);
  cholmod_l_finish(&common_);
}

bool RevisedColumnSolver::can_reseed_epoch() const {
  const auto rank = basis_columns_.size();
  return factored_direct_seed_
         && !std::getenv("TWALKER_DISABLE_REVISED_MULTI_EPOCH")
         && direct_seed_count_ < max_direct_seeds_
         && successful_faces_ > 0 && maximum_seed_deficiency_ <= 4
         && rank <= fixture_.m
         && fixture_.m - rank <= 4;
}

bool RevisedColumnSolver::can_request_direct_seed() const {
  if (centered_slope_mode_ || forced_shared_recurrence_)
    return direct_seed_count_ < max_direct_seeds_;
  // Preserve the original bounded initial/rebase behavior.  Additional
  // epochs require demonstrated useful returns and small rank deficiency.
  return direct_seed_count_ < 2 || can_reseed_epoch();
}

bool RevisedColumnSolver::form_slope_solution(
    RevisedSlopeSolution &solution) {
  const int active = static_cast<int>(rows_.size());
  const int m = static_cast<int>(fixture_.m);
  if (active <= 0
      || pseudoinverse_.size() != static_cast<std::size_t>(m) * active)
    return false;

  std::vector<double> active_b(active);
  for (int local = 0; local < active; ++local)
    active_b[local] = fixture_.b[rows_[local]];

  solution = RevisedSlopeSolution{};
  solution.rows = rows_;
  solution.rank = recurrence_rank_;
  if (recurrence_coefficients_valid_
      && recurrence_ua_.size() == static_cast<std::size_t>(m)) {
    // Applying a materialized pseudoinverse to b can lose many digits on a
    // marginal face.  The stable direct slope is the epoch anchor; Greville
    // row updates modify that solved vector directly between audits.
    solution.ua = recurrence_ua_;
  } else {
    solution.ua.assign(m, 0.0);
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, active, -1.0,
                pseudoinverse_.data(), m, active_b.data(), 1, 0.0,
                solution.ua.data(), 1);
  }
  solution.g = active_b;
  for (int local = 0; local < active; ++local) {
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      solution.g[local] +=
          fixture_.values[p] * solution.ua[fixture_.indices[p]];
  }

  solution.bua.assign(fixture_.n, 0.0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      solution.bua[row] +=
          fixture_.values[p] * solution.ua[fixture_.indices[p]];

  std::vector<double> transpose_g(m, 0.0);
  for (int local = 0; local < active; ++local) {
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      transpose_g[fixture_.indices[p]] +=
          fixture_.values[p] * solution.g[local];
  }
  const double scale = std::max(1.0, inf_norm(active_b));
  solution.slope_residual = inf_norm(transpose_g) / scale;
  if (centered_slope_mode_
      && std::getenv("TWALKER_CENTERED_BASIS_TRACE"))
    std::cerr << "centered slope rows=" << active
              << " rank=" << recurrence_rank_
              << " updates=" << recurrence_updates_
              << " residual=" << solution.slope_residual << '\n';
  return std::isfinite(solution.slope_residual)
         && solution.slope_residual <= 2e-10;
}

bool RevisedColumnSolver::solve_slope(
    const std::vector<std::uint32_t> &rows,
    RevisedSlopeSolution &solution) {
  ++stats_.calls;
  if (!centered_slope_mode_ || retired_ || !std::is_sorted(rows.begin(), rows.end())) {
    ++stats_.declines;
    return false;
  }
  if (!valid_) {
    ++stats_.declines;
    return false;
  }
  const bool transitioned = transition_recurrence(rows);
  const bool formed = transitioned && form_slope_solution(solution);
  if (!transitioned || !formed) {
    if (std::getenv("TWALKER_CENTERED_BASIS_TRACE"))
      std::cerr << "centered basis decline transitioned=" << transitioned
                << " formed=" << formed << '\n';
    // A failed Greville update is not a path failure.  Invalidate only the
    // numerical basis so the caller can cold-reveal this same centered face
    // and continue from the already accepted endpoint.
    valid_ = false;
    direct_seeded_ = false;
    pseudoinverse_.clear();
    recurrence_coefficients_valid_ = false;
    ++stats_.declines;
    return false;
  }
  ++successful_faces_;
  return true;
}

bool RevisedColumnSolver::seed_slope_from_direct(
    const std::vector<std::uint32_t> &rows,
    FaceSolution &direct_solution) {
  if (!centered_slope_mode_) return false;
  const bool seeded = seed_from_direct(rows, direct_solution);
  if (std::getenv("TWALKER_CENTERED_BASIS_TRACE"))
    {
      double seed_error = 0.0;
      if (seeded) {
        std::vector<double> active_b(rows.size()), reconstructed(fixture_.m);
        for (std::size_t local = 0; local < rows.size(); ++local)
          active_b[local] = fixture_.b[rows[local]];
        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    static_cast<int>(fixture_.m),
                    static_cast<int>(rows.size()), -1.0,
                    pseudoinverse_.data(), static_cast<int>(fixture_.m),
                    active_b.data(), 1, 0.0, reconstructed.data(), 1);
        for (std::size_t column = 0; column < fixture_.m; ++column)
          seed_error = std::max(
              seed_error,
              std::abs(reconstructed[column] - direct_solution.ua[column]));
      }
      std::cerr << "centered basis seed rows=" << rows.size()
                << " rank=" << direct_solution.rank
                << " seeded=" << seeded
                << " rowspace=" << recurrence_row_space_.size()
                << " ua_error=" << seed_error << '\n';
    }
  return seeded;
}

bool RevisedColumnSolver::needs_direct_seed() const {
  return ((coefficient_recurrence_ && shared_direct_seed_)
          || factored_direct_seed_)
         && !valid_ && !retired_ && can_request_direct_seed();
}

bool RevisedColumnSolver::row_coordinates(
    std::uint32_t row, std::vector<double> &coordinates,
    std::vector<double> &residual, double &relative_residual) const {
  const int rank = static_cast<int>(basis_columns_.size());
  if (reduced_sparse_factored_) {
    coordinates.assign(rank, 0.0);
    long double scale2 = 0.0L;
    std::vector<double> row_space_coordinates(rank, 0.0);
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const double value = fixture_.values[p];
      scale2 += static_cast<long double>(value) * value;
      if (const auto position = basis_position_[column]; position >= 0)
        coordinates[position] = value;
      for (int basis = 0; basis < rank; ++basis)
        row_space_coordinates[basis] +=
            rank_test_transform_[basis
                                 + static_cast<std::size_t>(rank) * column]
            * value;
    }
    long double projection2 = 0.0L;
    for (double value : row_space_coordinates)
      projection2 += static_cast<long double>(value) * value;
    const long double residual2 = std::max(0.0L, scale2 - projection2);
    residual.clear();
    relative_residual = std::sqrt(static_cast<double>(
        residual2 / std::max(1.0L, scale2)));
    // Norm subtraction cannot resolve directions near roundoff.  Treat the
    // ambiguous band as same-rank here; the original-operator 1e-10 face
    // residual remains the fail-closed authority before any return.
    if (relative_residual < 1e-6) relative_residual = 0.0;
    return std::isfinite(relative_residual);
  }
  if (factored_direct_seed_) {
    coordinates.assign(rank, 0.0);
    long double scale2 = 0.0L;
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const double value = fixture_.values[p];
      scale2 += static_cast<long double>(value) * value;
      if (orthonormal_factored_) {
        for (int basis = 0; basis < rank; ++basis)
          coordinates[basis] +=
              transform_[basis + static_cast<std::size_t>(rank) * column]
              * value;
      } else if (const auto position = basis_position_[column]; position >= 0) {
        coordinates[position] = value;
      }
    }
    residual.assign(fixture_.m, 0.0);
    cblas_dgemv(CblasColMajor, CblasTrans, rank, fixture_.m, -1.0,
                transform_.data(), rank, coordinates.data(), 1, 0.0,
                residual.data(), 1);
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      residual[fixture_.indices[p]] += fixture_.values[p];
    long double residual2 = 0.0L;
    for (double value : residual)
      residual2 += static_cast<long double>(value) * value;
    relative_residual = std::sqrt(static_cast<double>(
        residual2 / std::max(1.0L, scale2)));
    return std::isfinite(relative_residual);
  }
  coordinates.assign(rank, 0.0);
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
    const auto column = fixture_.indices[p];
    for (int basis = 0; basis < rank; ++basis)
      coordinates[basis] +=
          transform_[basis + static_cast<std::size_t>(rank) * column]
          * fixture_.values[p];
  }
  if (!solve_cholesky(transform_factor_, rank, coordinates)) return false;
  std::vector<long double> work(fixture_.m, 0.0L);
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
    work[fixture_.indices[p]] = fixture_.values[p];
  for (std::size_t column = 0; column < fixture_.m; ++column) {
    long double represented = 0.0L;
    for (int basis_column = 0; basis_column < rank; ++basis_column)
      represented += static_cast<long double>(coordinates[basis_column])
                     * transform_[basis_column
                                  + static_cast<std::size_t>(rank) * column];
    work[column] -= represented;
  }
  long double residual2 = 0.0L, scale2 = 0.0L;
  residual.resize(fixture_.m);
  for (std::size_t column = 0; column < fixture_.m; ++column) {
    residual[column] = static_cast<double>(work[column]);
    residual2 += work[column] * work[column];
  }
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
    scale2 += static_cast<long double>(fixture_.values[p]) * fixture_.values[p];
  relative_residual =
      std::sqrt(static_cast<double>(residual2 / std::max(1.0L, scale2)));
  return std::isfinite(relative_residual);
}

bool RevisedColumnSolver::rebuild(
    const std::vector<std::uint32_t> &rows) {
  const auto start = Clock::now();
  valid_ = false;
  rows_ = rows;
  basis_columns_.clear();
  basis_position_.assign(fixture_.m, -1);
  transform_.clear();
  coordinates_.clear();
  active_factor_.clear();
  transform_factor_.clear();
  active_cross_b_.clear();
  fixed_target_head_.clear();
  pseudoinverse_.clear();
  recurrence_ua_.clear();
  recurrence_g_.clear();
  recurrence_gamma_.clear();
  recurrence_uc_.clear();
  recurrence_coefficients_valid_ = false;
  direct_seeded_ = false;
  if (rows.empty()) return false;

  std::size_t face_nnz = 0;
  for (auto row : rows)
    face_nnz += fixture_.indptr[row + 1] - fixture_.indptr[row];
  auto *triplet = cholmod_l_allocate_triplet(
      rows.size(), fixture_.m, face_nnz, 0, CHOLMOD_REAL, &common_);
  if (!triplet) return false;
  auto *ti = static_cast<std::int64_t *>(triplet->i);
  auto *tj = static_cast<std::int64_t *>(triplet->j);
  auto *tx = static_cast<double *>(triplet->x);
  std::size_t cursor = 0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      ti[cursor] = local;
      tj[cursor] = fixture_.indices[p];
      tx[cursor] = fixture_.values[p];
      ++cursor;
    }
  }
  triplet->nnz = cursor;
  auto *A = cholmod_l_triplet_to_sparse(triplet, cursor, &common_);
  cholmod_l_free_triplet(&triplet, &common_);
  if (!A) return false;
  cholmod_sparse *R = nullptr;
  std::int64_t *permutation = nullptr;
  const auto rank64 = SuiteSparseQR_C(
      SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, fixture_.m, 0, A, nullptr,
      nullptr, nullptr, nullptr, &R, &permutation, nullptr, nullptr, nullptr,
      &common_);
  cholmod_l_free_sparse(&A, &common_);
  if (R) cholmod_l_free_sparse(&R, &common_);
  if (rank64 <= 0 || rank64 > fixture_.m) {
    if (permutation)
      cholmod_l_free(fixture_.m, sizeof(std::int64_t), permutation, &common_);
    return false;
  }
  const int rank = static_cast<int>(rank64);
  basis_columns_.resize(rank);
  for (int position = 0; position < rank; ++position) {
    const auto column = permutation ? permutation[position] : position;
    if (column < 0 || column >= fixture_.m) return false;
    basis_columns_[position] = static_cast<std::uint32_t>(column);
    basis_position_[column] = position;
  }
  if (permutation)
    cholmod_l_free(fixture_.m, sizeof(std::int64_t), permutation, &common_);

  const int active = static_cast<int>(rows.size());
  std::vector<double> C(static_cast<std::size_t>(active) * rank, 0.0);
  std::vector<double> dense_A(static_cast<std::size_t>(active) * fixture_.m,
                              0.0);
  for (int local = 0; local < active; ++local) {
    const auto row = rows[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      dense_A[local + static_cast<std::size_t>(active) * column] =
          fixture_.values[p];
      const auto position = basis_position_[column];
      if (position >= 0)
        C[local + static_cast<std::size_t>(active) * position] =
            fixture_.values[p];
    }
  }
  if (persistent_rank_updates_) {
    if (!qr_square_root(C, active, rank, active_factor_)) return false;
  } else {
    std::vector<double> active_gram(static_cast<std::size_t>(rank) * rank,
                                    0.0);
    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, rank, active, 1.0,
                C.data(), active, 0.0, active_gram.data(), rank);
    active_factor_ = std::move(active_gram);
    if (!cholesky(active_factor_, rank)) return false;
  }

  if (persistent_rank_updates_) {
    if (!qr_least_squares(C, active, rank, dense_A, fixture_.m, transform_))
      return false;
  } else {
    transform_.assign(static_cast<std::size_t>(rank) * fixture_.m, 0.0);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, fixture_.m,
                active, 1.0, C.data(), active, dense_A.data(), active, 0.0,
                transform_.data(), rank);
    if (!solve_cholesky(active_factor_, rank, transform_, fixture_.m))
      return false;
  }
  if (persistent_rank_updates_) {
    std::vector<double> transform_transpose(
        static_cast<std::size_t>(fixture_.m) * rank);
    for (int basis = 0; basis < rank; ++basis)
      for (std::size_t column = 0; column < fixture_.m; ++column)
        transform_transpose[column
                            + static_cast<std::size_t>(fixture_.m) * basis] =
            transform_[basis + static_cast<std::size_t>(rank) * column];
    if (!qr_square_root(std::move(transform_transpose), fixture_.m, rank,
                        transform_factor_))
      return false;
  } else {
    std::vector<double> transform_gram(
        static_cast<std::size_t>(rank) * rank, 0.0);
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, rank, fixture_.m,
                1.0, transform_.data(), rank, 0.0,
                transform_gram.data(), rank);
    transform_factor_ = std::move(transform_gram);
    if (!cholesky(transform_factor_, rank)) return false;
  }

  coordinates_.assign(static_cast<std::size_t>(active) * rank, 0.0);
  for (int local = 0; local < active; ++local)
    for (int position = 0; position < rank; ++position)
      coordinates_[static_cast<std::size_t>(local) * rank + position] =
          C[local + static_cast<std::size_t>(active) * position];

  active_cross_b_.assign(rank, 0.0);
  for (int local = 0; local < active; ++local)
    for (int position = 0; position < rank; ++position)
      active_cross_b_[position] +=
          C[local + static_cast<std::size_t>(active) * position]
          * fixture_.b[rows[local]];
  recurrence_rank_ = rank;
  recurrence_updates_ = 0;
  if (coefficient_recurrence_ && !build_pseudoinverse()) return false;
  valid_ = true;
  ++stats_.rebuilds;
  stats_.rebuild_ms += milliseconds_since(start);
  return true;
}

bool RevisedColumnSolver::add_row(std::uint32_t row) {
  last_update_failure_.clear();
  const int rank = static_cast<int>(basis_columns_.size());
  std::vector<double> coordinate, residual;
  double relative_residual = 0.0;
  const auto rank_test_start = Clock::now();
  if (!row_coordinates(row, coordinate, residual, relative_residual)) {
    last_update_failure_ = "add_coordinates";
    return false;
  }
  if (reduced_sparse_factored_)
    stats_.rank_test_ms += milliseconds_since(rank_test_start);
  stats_.worst_entering_representation = std::max(
      stats_.worst_entering_representation, relative_residual);

  const auto insertion = std::lower_bound(rows_.begin(), rows_.end(), row);
  const auto local = static_cast<std::size_t>(insertion - rows_.begin());
  if (reduced_sparse_factored_) {
    if (relative_residual > 1e-10) {
      ++stats_.rank_changes;
      if (!can_reseed_epoch()) {
        ++stats_.retirements;
        retired_ = true;
      }
      last_update_failure_ = "add_reduced_rank_change";
      return false;
    }
    auto shadow = active_factor_;
    if ((shadow_square_root_ || shadow_solve_only_)
        && !cholesky_rank_one(shadow, rank, coordinate, +1)) {
      last_update_failure_ = "add_shadow_sqrt";
      return false;
    }
    if (!update_reduced_factor(coordinate, true)) {
      last_update_failure_ = "add_reduced_updown";
      return false;
    }
    if (shadow_square_root_ || shadow_solve_only_)
      active_factor_ = std::move(shadow);
    for (int j = 0; j < rank; ++j)
      active_cross_b_[j] += coordinate[j] * fixture_.b[row];
    rows_.insert(insertion, row);
    return true;
  }
  if (relative_residual <= 1e-10) {
    auto factor = active_factor_;
    if (!cholesky_rank_one(factor, rank, coordinate, +1)
        && (!refresh_active_factor()
            || !(factor = active_factor_,
                 cholesky_rank_one(factor, rank, coordinate, +1)))) {
      last_update_failure_ = "add_same_rank_cholesky";
      return false;
    }
    active_factor_ = std::move(factor);
    for (int j = 0; j < rank; ++j)
      active_cross_b_[j] += coordinate[j] * fixture_.b[row];
    coordinates_.insert(coordinates_.begin() + local * rank,
                        coordinate.begin(), coordinate.end());
    rows_.insert(insertion, row);
    return true;
  }
  if (!persistent_rank_updates_) {
    ++stats_.rank_changes;
    if (factored_direct_seed_ && can_reseed_epoch()) {
      valid_ = false;
      direct_seeded_ = false;
    } else {
      ++stats_.retirements;
      retired_ = true;
    }
    last_update_failure_ = "add_rank_change_retired";
    return false;
  }

  const double rho = norm2(residual);
  if (!(rho > 0.0) || !std::isfinite(rho)) {
    last_update_failure_ = "add_residual_norm";
    return false;
  }
  for (double &value : residual) value /= rho;

  std::vector<double> enlarged_active;
  if (!qr_append_row_and_column(active_factor_, rank, coordinate, rho,
                                enlarged_active)) {
    if (!refresh_active_factor()) {
      last_update_failure_ = "add_active_refresh";
      return false;
    }
    if (!qr_append_row_and_column(active_factor_, rank, coordinate, rho,
                                  enlarged_active)) {
      last_update_failure_ = "add_active_append";
      return false;
    }
  }

  std::vector<double> transform_cross(rank, 0.0);
  for (std::size_t column = 0; column < fixture_.m; ++column)
    for (int j = 0; j < rank; ++j)
      transform_cross[j] +=
          transform_[j + static_cast<std::size_t>(rank) * column]
          * residual[column];
  std::vector<double> enlarged_transform_factor;
  if (!cholesky_append(transform_factor_, rank, transform_cross, 1.0,
                       enlarged_transform_factor))
    { last_update_failure_ = "add_transform_append"; return false; }

  std::vector<double> enlarged_transform(
      static_cast<std::size_t>(rank + 1) * fixture_.m);
  for (std::size_t column = 0; column < fixture_.m; ++column) {
    for (int j = 0; j < rank; ++j)
      enlarged_transform[j + static_cast<std::size_t>(rank + 1) * column] =
          transform_[j + static_cast<std::size_t>(rank) * column];
    enlarged_transform[rank + static_cast<std::size_t>(rank + 1) * column] =
        residual[column];
  }
  std::vector<double> enlarged_coordinates;
  enlarged_coordinates.reserve((rows_.size() + 1) * (rank + 1));
  for (std::size_t old = 0; old <= rows_.size(); ++old) {
    if (old == local) {
      enlarged_coordinates.insert(enlarged_coordinates.end(),
                                  coordinate.begin(), coordinate.end());
      enlarged_coordinates.push_back(rho);
    } else {
      const auto source = old < local ? old : old - 1;
      enlarged_coordinates.insert(
          enlarged_coordinates.end(),
          coordinates_.begin() + source * rank,
          coordinates_.begin() + (source + 1) * rank);
      enlarged_coordinates.push_back(0.0);
    }
  }
  const auto old_cross = active_cross_b_;
  active_cross_b_.resize(rank + 1);
  for (int j = 0; j < rank; ++j)
    active_cross_b_[j] = old_cross[j] + coordinate[j] * fixture_.b[row];
  active_cross_b_[rank] = rho * fixture_.b[row];
  basis_columns_.push_back(fixture_.m + stats_.rank_increases);
  active_factor_ = std::move(enlarged_active);
  transform_factor_ = std::move(enlarged_transform_factor);
  transform_ = std::move(enlarged_transform);
  coordinates_ = std::move(enlarged_coordinates);
  rows_.insert(insertion, row);
  ++stats_.rank_changes;
  ++stats_.rank_increases;
  return true;
}

bool RevisedColumnSolver::remove_row(std::uint32_t row) {
  last_update_failure_.clear();
  const int rank = static_cast<int>(basis_columns_.size());
  const auto found = std::lower_bound(rows_.begin(), rows_.end(), row);
  if (found == rows_.end() || *found != row) {
    last_update_failure_ = "remove_missing";
    return false;
  }
  const auto local = static_cast<std::size_t>(found - rows_.begin());
  if (reduced_sparse_factored_) {
    std::vector<double> coordinate(rank, 0.0);
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto position = basis_position_[fixture_.indices[p]];
      if (position >= 0) coordinate[position] = fixture_.values[p];
    }
    auto shadow = active_factor_;
    if ((shadow_square_root_ || shadow_solve_only_)
        && !cholesky_rank_one(shadow, rank, coordinate, -1)) {
      last_update_failure_ = "remove_shadow_sqrt";
      return false;
    }
    if (!update_reduced_factor(coordinate, false)) {
      last_update_failure_ = "remove_reduced_updown";
      return false;
    }
    if (shadow_square_root_ || shadow_solve_only_)
      active_factor_ = std::move(shadow);
    for (int j = 0; j < rank; ++j)
      active_cross_b_[j] -= coordinate[j] * fixture_.b[row];
    rows_.erase(found);
    return true;
  }
  std::vector<double> coordinate(
      coordinates_.begin() + local * rank,
      coordinates_.begin() + (local + 1) * rank);
  auto trial = active_factor_;
  if (cholesky_rank_one(trial, rank, coordinate, -1)
      && diagonal_ratio(trial, rank) >= 1e-10) {
    active_factor_ = std::move(trial);
    for (int j = 0; j < rank; ++j)
      active_cross_b_[j] -= coordinate[j] * fixture_.b[row];
    coordinates_.erase(coordinates_.begin() + local * rank,
                       coordinates_.begin() + (local + 1) * rank);
    rows_.erase(found);
    return true;
  }
  if (!persistent_rank_updates_) {
    ++stats_.rank_changes;
    if (factored_direct_seed_ && can_reseed_epoch()) {
      valid_ = false;
      direct_seeded_ = false;
    } else {
      ++stats_.retirements;
      retired_ = true;
    }
    last_update_failure_ = "remove_rank_change_retired";
    return false;
  }
  // Long update chains can make the maintained factor slightly less accurate
  // than the coordinates it represents.  Refresh this one Gram core and retry
  // before declaring a genuine rank loss.
  if (!refresh_active_factor()) {
    last_update_failure_ = "remove_active_refresh";
    return false;
  }
  trial = active_factor_;
  if (cholesky_rank_one(trial, rank, coordinate, -1)
      && diagonal_ratio(trial, rank) >= 1e-10) {
    active_factor_ = std::move(trial);
    for (int j = 0; j < rank; ++j)
      active_cross_b_[j] -= coordinate[j] * fixture_.b[row];
    coordinates_.erase(coordinates_.begin() + local * rank,
                       coordinates_.begin() + (local + 1) * rank);
    rows_.erase(found);
    return true;
  }
  if (rank <= 1) { last_update_failure_ = "remove_rank_one"; return false; }

  std::vector<double> null_vector = coordinate;
  if (!solve_cholesky(active_factor_, rank, null_vector)) {
    last_update_failure_ = "remove_null_solve";
    return false;
  }
  int deleted = 0;
  for (int j = 1; j < rank; ++j)
    if (std::abs(null_vector[j]) > std::abs(null_vector[deleted])) deleted = j;
  const double pivot = null_vector[deleted];
  if (!(std::abs(pivot) > 0.0) || !std::isfinite(pivot)) {
    last_update_failure_ = "remove_null_pivot";
    return false;
  }
  std::vector<double> weights;
  weights.reserve(rank - 1);
  for (int j = 0; j < rank; ++j)
    if (j != deleted) weights.push_back(-null_vector[j] / pivot);

  std::vector<double> reduced_active;
  if (!cholesky_delete(active_factor_, rank, deleted, reduced_active))
    { last_update_failure_ = "remove_active_delete"; return false; }
  std::vector<double> reduced_coordinate;
  reduced_coordinate.reserve(rank - 1);
  for (int j = 0; j < rank; ++j)
    if (j != deleted) reduced_coordinate.push_back(coordinate[j]);
  if (!cholesky_rank_one(reduced_active, rank - 1, reduced_coordinate, -1)) {
    // Near a true rank loss, deleting from the old factor and then
    // downdating can lose the tiny Schur complement to roundoff.  Re-form the
    // smaller core from the maintained tableau coordinates; this is still a
    // BLAS-only refresh and does not perform rank-revealing QR.
    std::vector<double> reduced_tableau;
    reduced_tableau.reserve((rows_.size() - 1) * (rank - 1));
    for (std::size_t active = 0; active < rows_.size(); ++active) {
      if (active == local) continue;
      for (int j = 0; j < rank; ++j)
        if (j != deleted)
          reduced_tableau.push_back(coordinates_[active * rank + j]);
    }
    std::vector<double> reduced_col_major(reduced_tableau.size());
    const int reduced_rows = static_cast<int>(rows_.size() - 1);
    for (int active = 0; active < reduced_rows; ++active)
      for (int j = 0; j < rank - 1; ++j)
        reduced_col_major[active
                          + static_cast<std::size_t>(reduced_rows) * j] =
            reduced_tableau[static_cast<std::size_t>(active) * (rank - 1)
                             + j];
    if (!qr_square_root(std::move(reduced_col_major), reduced_rows, rank - 1,
                        reduced_active)) {
      last_update_failure_ = "remove_active_downdate";
      return false;
    }
  }

  std::vector<double> h_cross;
  h_cross.reserve(rank - 1);
  double h_diagonal = 0.0;
  for (int j = 0; j < rank; ++j) {
    double product = 0.0;
    for (std::size_t column = 0; column < fixture_.m; ++column)
      product += transform_[j + static_cast<std::size_t>(rank) * column]
                 * transform_[deleted
                              + static_cast<std::size_t>(rank) * column];
    if (j == deleted)
      h_diagonal = product;
    else
      h_cross.push_back(product);
  }
  std::vector<double> reduced_transform_factor;
  if (!cholesky_delete(transform_factor_, rank, deleted,
                       reduced_transform_factor))
    { last_update_failure_ = "remove_transform_delete"; return false; }
  std::vector<double> a(rank - 1), update(rank - 1), downdate(rank - 1);
  const double inverse_root_two = 1.0 / std::sqrt(2.0);
  for (int j = 0; j < rank - 1; ++j) {
    a[j] = h_cross[j] + 0.5 * h_diagonal * weights[j];
    update[j] = (a[j] + weights[j]) * inverse_root_two;
    downdate[j] = (a[j] - weights[j]) * inverse_root_two;
  }
  if (!cholesky_rank_one(reduced_transform_factor, rank - 1, update, +1)
      || !cholesky_rank_one(reduced_transform_factor, rank - 1, downdate,
                            -1))
    { last_update_failure_ = "remove_transform_updates"; return false; }

  std::vector<double> reduced_transform(
      static_cast<std::size_t>(rank - 1) * fixture_.m);
  int target_row = 0;
  for (int source_row = 0; source_row < rank; ++source_row) {
    if (source_row == deleted) continue;
    for (std::size_t column = 0; column < fixture_.m; ++column)
      reduced_transform[target_row
                        + static_cast<std::size_t>(rank - 1) * column] =
          transform_[source_row + static_cast<std::size_t>(rank) * column]
          + weights[target_row]
                * transform_[deleted
                             + static_cast<std::size_t>(rank) * column];
    ++target_row;
  }
  std::vector<double> reduced_coordinates;
  reduced_coordinates.reserve((rows_.size() - 1) * (rank - 1));
  for (std::size_t active = 0; active < rows_.size(); ++active) {
    if (active == local) continue;
    for (int j = 0; j < rank; ++j)
      if (j != deleted)
        reduced_coordinates.push_back(
            coordinates_[active * rank + j]);
  }
  std::vector<double> reduced_cross;
  reduced_cross.reserve(rank - 1);
  for (int j = 0; j < rank; ++j)
    if (j != deleted)
      reduced_cross.push_back(active_cross_b_[j]
                              - coordinate[j] * fixture_.b[row]);
  basis_columns_.erase(basis_columns_.begin() + deleted);
  active_factor_ = std::move(reduced_active);
  transform_factor_ = std::move(reduced_transform_factor);
  transform_ = std::move(reduced_transform);
  coordinates_ = std::move(reduced_coordinates);
  active_cross_b_ = std::move(reduced_cross);
  rows_.erase(found);
  ++stats_.rank_changes;
  ++stats_.rank_decreases;
  return true;
}

bool RevisedColumnSolver::refresh_active_factor() {
  const auto start = Clock::now();
  const int rank = static_cast<int>(basis_columns_.size());
  const int active = static_cast<int>(rows_.size());
  if (rank <= 0 || active < rank
      || coordinates_.size() != static_cast<std::size_t>(active) * rank)
    return false;
  std::vector<double> coordinate_col_major(coordinates_.size());
  for (int local = 0; local < active; ++local)
    for (int j = 0; j < rank; ++j)
      coordinate_col_major[local + static_cast<std::size_t>(active) * j] =
          coordinates_[static_cast<std::size_t>(local) * rank + j];
  std::vector<double> refreshed;
  if (!qr_square_root(std::move(coordinate_col_major), active, rank,
                      refreshed))
    return false;
  active_factor_ = std::move(refreshed);
  active_cross_b_.assign(rank, 0.0);
  std::vector<double> active_b(active);
  for (int local = 0; local < active; ++local)
    active_b[local] = fixture_.b[rows_[local]];
  cblas_dgemv(CblasRowMajor, CblasTrans, active, rank, 1.0,
              coordinates_.data(), rank, active_b.data(), 1, 0.0,
              active_cross_b_.data(), 1);
  ++stats_.active_refreshes;
  stats_.active_refresh_ms += milliseconds_since(start);
  return true;
}

bool RevisedColumnSolver::build_reduced_factor() {
  if (reduced_update_column_)
    cholmod_l_free_sparse(&reduced_update_column_, &common_);
  if (reduced_factor_)
    cholmod_l_free_factor(&reduced_factor_, &common_);
  reduced_rcond_ = 0.0;
  const int rank = static_cast<int>(basis_columns_.size());
  if (rank <= 0 || rows_.empty()) return false;
  reduced_scale_.assign(rank, 1.0);
  std::size_t nnz = 0;
  for (auto row : rows_)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      nnz += basis_position_[fixture_.indices[p]] >= 0;
  auto *columns = cholmod_l_allocate_sparse(
      rank, rows_.size(), nnz, 1, 1, 0, CHOLMOD_REAL, &common_);
  if (!columns) return false;
  auto *cp = static_cast<std::int64_t *>(columns->p);
  auto *ci = static_cast<std::int64_t *>(columns->i);
  auto *cx = static_cast<double *>(columns->x);
  std::vector<long double> column_norm2(rank, 0.0L);
  std::size_t cursor = 0;
  for (std::size_t local = 0; local < rows_.size(); ++local) {
    cp[local] = static_cast<std::int64_t>(cursor);
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto position = basis_position_[fixture_.indices[p]];
      if (position < 0) continue;
      ci[cursor] = position;
      cx[cursor] = fixture_.values[p];
      column_norm2[position] +=
          static_cast<long double>(fixture_.values[p]) * fixture_.values[p];
      ++cursor;
    }
  }
  cp[rows_.size()] = static_cast<std::int64_t>(cursor);
  if (reduced_equilibration_) {
    std::vector<double> norms;
    norms.reserve(rank);
    for (long double norm2 : column_norm2)
      if (norm2 > 0.0L) norms.push_back(std::sqrt(static_cast<double>(norm2)));
    if (norms.size() != static_cast<std::size_t>(rank)) {
      cholmod_l_free_sparse(&columns, &common_);
      return false;
    }
    const auto middle = norms.begin() + norms.size() / 2;
    std::nth_element(norms.begin(), middle, norms.end());
    const double reference = *middle;
    int minimum_exponent = 8, maximum_exponent = -8;
    for (int position = 0; position < rank; ++position) {
      const double norm = std::sqrt(static_cast<double>(column_norm2[position]));
      const int exponent = std::clamp(
          static_cast<int>(std::nearbyint(std::log2(reference / norm))),
          -8, 8);
      reduced_scale_[position] = std::ldexp(1.0, exponent);
      minimum_exponent = std::min(minimum_exponent, exponent);
      maximum_exponent = std::max(maximum_exponent, exponent);
    }
    for (std::size_t entry = 0; entry < cursor; ++entry)
      cx[entry] *= reduced_scale_[ci[entry]];
    if (std::getenv("TWALKER_REVISED_TRACE_FAILURE"))
      std::cerr << "revised equilibrium rank=" << rank
                << " scale_exponents=[" << minimum_exponent << ','
                << maximum_exponent << "] reference=" << reference << '\n';
  }
  auto *gram = cholmod_l_aat(columns, nullptr, 0, 1, &common_);
  cholmod_l_free_sparse(&columns, &common_);
  if (!gram) return false;
  gram->stype = 1;
  reduced_factor_ = cholmod_l_analyze(gram, &common_);
  const bool ok = reduced_factor_
                  && cholmod_l_factorize(gram, reduced_factor_, &common_)
                  && reduced_factor_->minor == rank;
  cholmod_l_free_sparse(&gram, &common_);
  if (!ok) {
    if (reduced_factor_)
      cholmod_l_free_factor(&reduced_factor_, &common_);
    return false;
  }
  reduced_inverse_perm_.assign(rank, 0);
  if (reduced_factor_->Perm) {
    const auto *permutation =
        static_cast<const std::int64_t *>(reduced_factor_->Perm);
    for (int position = 0; position < rank; ++position)
      reduced_inverse_perm_[permutation[position]] = position;
  } else {
    for (int position = 0; position < rank; ++position)
      reduced_inverse_perm_[position] = position;
  }
  reduced_rcond_ = cholmod_l_rcond(reduced_factor_, &common_);
  reduced_update_column_ = cholmod_l_allocate_sparse(
      rank, 1, rank, 1, 1, 0, CHOLMOD_REAL, &common_);
  if (!reduced_update_column_) return false;
  reduced_update_entries_.clear();
  reduced_update_entries_.reserve(rank);
  return std::isfinite(reduced_rcond_) && reduced_rcond_ >= 1e-10;
}

bool RevisedColumnSolver::update_reduced_factor(
    const std::vector<double> &coordinate, bool add) {
  const auto update_start = Clock::now();
  const int rank = static_cast<int>(basis_columns_.size());
  if (!reduced_factor_ || coordinate.size() != static_cast<std::size_t>(rank))
    return false;
  if (!reduced_update_column_) return false;
  reduced_update_entries_.clear();
  for (int position = 0; position < rank; ++position)
    if (coordinate[position] != 0.0)
      reduced_update_entries_.emplace_back(reduced_inverse_perm_[position],
                                           reduced_scale_[position]
                                               * coordinate[position]);
  std::sort(reduced_update_entries_.begin(), reduced_update_entries_.end());
  auto *cp = static_cast<std::int64_t *>(reduced_update_column_->p);
  auto *ci = static_cast<std::int64_t *>(reduced_update_column_->i);
  auto *cx = static_cast<double *>(reduced_update_column_->x);
  cp[0] = 0;
  cp[1] = static_cast<std::int64_t>(reduced_update_entries_.size());
  for (std::size_t i = 0; i < reduced_update_entries_.size(); ++i) {
    ci[i] = reduced_update_entries_[i].first;
    cx[i] = reduced_update_entries_[i].second;
  }
  const bool ok = cholmod_l_updown(add ? 1 : 0, reduced_update_column_,
                                   reduced_factor_, &common_);
  stats_.factor_update_ms += milliseconds_since(update_start);
  return ok && reduced_factor_->minor == rank;
}

bool RevisedColumnSolver::build_pseudoinverse() {
  const int active = static_cast<int>(rows_.size());
  const int m = static_cast<int>(fixture_.m);
  if (active <= 0 || m <= 0) return false;
  if (!std::getenv("TWALKER_REVISED_SVD_INIT")) {
    const int rank = static_cast<int>(basis_columns_.size());
    if (rank <= 0
        || coordinates_.size() != static_cast<std::size_t>(active) * rank)
      return false;
    // Row-major active-by-rank coordinates are column-major rank-by-active C'.
    std::vector<double> core = coordinates_;
    if (!solve_cholesky(active_factor_, rank, core, active)
        || !solve_cholesky(transform_factor_, rank, core, active))
      return false;
    pseudoinverse_.assign(static_cast<std::size_t>(m) * active, 0.0);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, active, rank, 1.0,
                transform_.data(), rank, core.data(), rank, 0.0,
                pseudoinverse_.data(), m);
    return true;
  }
  std::vector<double> dense_A(static_cast<std::size_t>(active) * m, 0.0);
  for (int local = 0; local < active; ++local) {
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      dense_A[local + static_cast<std::size_t>(active)
                          * fixture_.indices[p]] = fixture_.values[p];
  }
  const int leading_b = std::max(active, m);
  std::vector<double> right(static_cast<std::size_t>(leading_b) * active,
                            0.0);
  for (int i = 0; i < active; ++i)
    right[i + static_cast<std::size_t>(leading_b) * i] = 1.0;
  std::vector<double> singular(std::min(active, m));
  // Match the scale of SPQR's default numerical-rank decision rather than
  // LAPACK's machine-epsilon cutoff, which over-resolves tiny face modes.
  const double rcond = 1e-12;
  int numerical_rank = 0, info = 0, query = -1;
  double requested = 0.0;
  dgelss_(&active, &m, &active, dense_A.data(), &active, right.data(),
          &leading_b, singular.data(), &rcond, &numerical_rank, &requested,
          &query, &info);
  if (info != 0 || !std::isfinite(requested)) return false;
  const int work_size = std::max(1, static_cast<int>(requested));
  std::vector<double> work(work_size);
  dgelss_(&active, &m, &active, dense_A.data(), &active, right.data(),
          &leading_b, singular.data(), &rcond, &numerical_rank, work.data(),
          &work_size, &info);
  if (info != 0 || numerical_rank <= 0) return false;
  if (std::getenv("TWALKER_REVISED_TRACE_FAILURE"))
    std::cerr << "recurrence SVD active=" << active << " m=" << m
              << " spqr_rank=" << basis_columns_.size()
              << " svd_rank=" << numerical_rank
              << " sigma_max=" << singular.front()
              << " sigma_min=" << singular.back() << '\n';
  recurrence_rank_ = numerical_rank;
  pseudoinverse_.assign(static_cast<std::size_t>(m) * active, 0.0);
  for (int rhs = 0; rhs < active; ++rhs)
    std::copy(right.begin() + static_cast<std::size_t>(leading_b) * rhs,
              right.begin() + static_cast<std::size_t>(leading_b) * rhs + m,
              pseudoinverse_.begin() + static_cast<std::size_t>(m) * rhs);
  return true;
}

bool RevisedColumnSolver::add_row_recurrence(std::uint32_t row) {
  const int active = static_cast<int>(rows_.size());
  const int m = static_cast<int>(fixture_.m);
  std::vector<double> dense_row(m, 0.0);
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
    dense_row[fixture_.indices[p]] = fixture_.values[p];

  std::vector<double> d(active, 0.0);
  cblas_dgemv(CblasColMajor, CblasTrans, m, active, 1.0,
              pseudoinverse_.data(), m, dense_row.data(), 1, 0.0, d.data(),
              1);
  std::vector<long double> residual(dense_row.begin(), dense_row.end());
  const bool stable_row_projection = centered_slope_mode_
      && recurrence_rank_ > 0
      && recurrence_row_space_.size()
             == static_cast<std::size_t>(recurrence_rank_) * m;
  if (stable_row_projection) {
    std::vector<long double> coordinate(recurrence_rank_, 0.0L);
    for (int component = 0; component < recurrence_rank_; ++component)
      for (int column = 0; column < m; ++column)
        coordinate[component] +=
            static_cast<long double>(
                recurrence_row_space_[component
                                      + recurrence_rank_ * column])
            * dense_row[column];
    for (int component = 0; component < recurrence_rank_; ++component)
      for (int column = 0; column < m; ++column)
        residual[column] -= coordinate[component]
            * recurrence_row_space_[component
                                    + recurrence_rank_ * column];
  } else {
    for (int local = 0; local < active; ++local) {
      const long double coefficient = d[local];
      const auto old_row = rows_[local];
      for (auto p = fixture_.indptr[old_row];
           p < fixture_.indptr[old_row + 1]; ++p)
        residual[fixture_.indices[p]] -= coefficient * fixture_.values[p];
    }
  }
  long double residual_square = 0.0L, row_square = 0.0L;
  for (int column = 0; column < m; ++column)
    residual_square += residual[column] * residual[column];
  for (double value : dense_row)
    row_square += static_cast<long double>(value) * value;
  const double relative_residual = std::sqrt(static_cast<double>(
      residual_square / std::max(1.0L, row_square)));
  stats_.worst_entering_representation = std::max(
      stats_.worst_entering_representation, relative_residual);
  if (!std::isfinite(relative_residual)) return false;

  std::vector<double> greville_column(m, 0.0);
  bool rank_increase = relative_residual > 1e-10;
  if (centered_slope_mode_ && stable_row_projection && !rank_increase
      && !std::getenv("TWALKER_CENTERED_RECURRENCE_AUDIT"))
    return false;
  if (rank_increase) {
    // A tiny independent direction would make the rank-one recurrence
    // numerically unsafe; hand that transition to the guarded incumbent.
    if (relative_residual < 1e-6) return false;
    const double square = static_cast<double>(residual_square);
    if (!(square > 0.0)) return false;
    for (int column = 0; column < m; ++column)
      greville_column[column] = static_cast<double>(residual[column]) / square;
  } else {
    double denominator = 1.0;
    for (double value : d) denominator += value * value;
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, active, 1.0,
                pseudoinverse_.data(), m, d.data(), 1, 0.0,
                greville_column.data(), 1);
    for (double &value : greville_column) value /= denominator;
  }

  const auto insertion = std::lower_bound(rows_.begin(), rows_.end(), row);
  const int local = static_cast<int>(insertion - rows_.begin());
  std::vector<double> updated_ua, updated_g, updated_gamma;
  const bool update_coefficients = recurrence_coefficients_valid_
                                   && target_shift_.empty()
                                   && recurrence_ua_.size()
                                          == static_cast<std::size_t>(m)
                                   && recurrence_g_.size()
                                          == static_cast<std::size_t>(active)
                                   && recurrence_gamma_.size()
                                          == static_cast<std::size_t>(active);
  if (update_coefficients) {
    double entering_residual = fixture_.b[row];
    for (int column = 0; column < m; ++column)
      entering_residual += dense_row[column] * recurrence_ua_[column];
    updated_ua = recurrence_ua_;
    for (int column = 0; column < m; ++column)
      updated_ua[column] -= greville_column[column] * entering_residual;
    updated_g.resize(active + 1);
    for (int target = 0; target < active + 1; ++target) {
      if (target == local) {
        double row_action = 0.0;
        for (int column = 0; column < m; ++column)
          row_action += dense_row[column] * greville_column[column];
        updated_g[target] = entering_residual * (1.0 - row_action);
      } else {
        const int old = target < local ? target : target - 1;
        double row_action = 0.0;
        const auto old_row = rows_[old];
        for (auto p = fixture_.indptr[old_row];
             p < fixture_.indptr[old_row + 1]; ++p)
          row_action += fixture_.values[p]
                        * greville_column[fixture_.indices[p]];
        updated_g[target] = recurrence_g_[old]
                            - row_action * entering_residual;
      }
    }
    double target_coefficient = 0.0;
    for (int column = 0; column < m; ++column)
      target_coefficient += greville_column[column] * fixture_.d[column];
    updated_gamma.resize(active + 1);
    for (int target = 0; target < active + 1; ++target) {
      if (target == local) {
        updated_gamma[target] = target_coefficient;
      } else {
        const int old = target < local ? target : target - 1;
        updated_gamma[target] = recurrence_gamma_[old]
                                - d[old] * target_coefficient;
      }
    }
  }
  std::vector<double> enlarged(static_cast<std::size_t>(m) * (active + 1));
  for (int target = 0; target < active + 1; ++target) {
    const double *source = nullptr;
    if (target == local)
      source = greville_column.data();
    else {
      const int old = target < local ? target : target - 1;
      source = pseudoinverse_.data() + static_cast<std::size_t>(m) * old;
    }
    auto destination = enlarged.begin() + static_cast<std::size_t>(m) * target;
    std::copy(source, source + m, destination);
    if (target != local) {
      const int old = target < local ? target : target - 1;
      for (int column = 0; column < m; ++column)
        destination[column] -= greville_column[column] * d[old];
    }
  }
  pseudoinverse_ = std::move(enlarged);
  rows_.insert(insertion, row);
  if (centered_slope_mode_ && stable_row_projection && rank_increase) {
    const auto old_rank = recurrence_rank_;
    const auto new_rank = old_rank + 1;
    std::vector<double> enlarged_space(
        static_cast<std::size_t>(new_rank) * m, 0.0);
    for (int column = 0; column < m; ++column) {
      for (int component = 0; component < old_rank; ++component)
        enlarged_space[component + new_rank * column] =
            recurrence_row_space_[component + old_rank * column];
      enlarged_space[old_rank + new_rank * column] =
          static_cast<double>(residual[column])
          / std::sqrt(static_cast<double>(residual_square));
    }
    recurrence_row_space_ = std::move(enlarged_space);
  }
  if (update_coefficients) {
    recurrence_ua_ = std::move(updated_ua);
    recurrence_g_ = std::move(updated_g);
    recurrence_gamma_ = std::move(updated_gamma);
    recurrence_uc_.clear();
  } else {
    recurrence_g_.clear();
    recurrence_coefficients_valid_ = false;
  }
  ++recurrence_updates_;
  if (rank_increase) {
    ++recurrence_rank_;
    ++stats_.rank_changes;
    ++stats_.rank_increases;
  }
  return true;
}

bool RevisedColumnSolver::remove_row_recurrence(std::uint32_t row) {
  const int active = static_cast<int>(rows_.size());
  const int m = static_cast<int>(fixture_.m);
  const auto found = std::lower_bound(rows_.begin(), rows_.end(), row);
  if (found == rows_.end() || *found != row || active <= 1) return false;
  // A stable QR/SVD row-space downdate is the next maintenance primitive.
  // Until it is present, reseed this uncommon transition before mutating the
  // live centered basis; never continue with a stale row-space projector.
  if (centered_slope_mode_ && !recurrence_row_space_.empty()
      && !std::getenv("TWALKER_CENTERED_RECURRENCE_AUDIT"))
    return false;
  const int removed = static_cast<int>(found - rows_.begin());
  std::vector<double> column(
      pseudoinverse_.begin() + static_cast<std::size_t>(m) * removed,
      pseudoinverse_.begin() + static_cast<std::size_t>(m) * (removed + 1));
  double leverage = 0.0;
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
    leverage += fixture_.values[p] * column[fixture_.indices[p]];
  const double complement = 1.0 - leverage;

  std::vector<double> reduced(static_cast<std::size_t>(m) * (active - 1));
  for (int target = 0; target < active - 1; ++target) {
    const int source = target < removed ? target : target + 1;
    std::copy(pseudoinverse_.begin() + static_cast<std::size_t>(m) * source,
              pseudoinverse_.begin()
                  + static_cast<std::size_t>(m) * (source + 1),
              reduced.begin() + static_cast<std::size_t>(m) * target);
  }
  std::vector<double> correction(active - 1, 0.0);
  bool rank_decrease = std::abs(complement) <= 1e-8;
  if (!rank_decrease) {
    if (!(complement > 1e-5)) return false;
    for (int target = 0; target < active - 1; ++target) {
      double value = 0.0;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        value += reduced[fixture_.indices[p]
                         + static_cast<std::size_t>(m) * target]
                 * fixture_.values[p];
      correction[target] = value / complement;
    }
    cblas_dger(CblasColMajor, m, active - 1, 1.0, column.data(), 1,
               correction.data(), 1, reduced.data(), m);
  } else {
    double square = 0.0;
    for (double value : column) square += value * value;
    if (!(square > 0.0)) return false;
    cblas_dgemv(CblasColMajor, CblasTrans, m, active - 1, 1.0,
                reduced.data(), m, column.data(), 1, 0.0,
                correction.data(), 1);
    for (double &value : correction) value /= square;
    cblas_dger(CblasColMajor, m, active - 1, -1.0, column.data(), 1,
               correction.data(), 1, reduced.data(), m);
  }
  std::vector<double> updated_ua, updated_g, updated_gamma;
  const bool update_coefficients = recurrence_coefficients_valid_
                                   && target_shift_.empty()
                                   && recurrence_ua_.size()
                                          == static_cast<std::size_t>(m)
                                   && recurrence_g_.size()
                                          == static_cast<std::size_t>(active)
                                   && recurrence_gamma_.size()
                                          == static_cast<std::size_t>(active);
  if (update_coefficients) {
    const double sign = rank_decrease ? -1.0 : 1.0;
    double correction_b = 0.0;
    for (int target = 0; target < active - 1; ++target) {
      const int source = target < removed ? target : target + 1;
      correction_b += correction[target] * fixture_.b[rows_[source]];
    }
    updated_ua = recurrence_ua_;
    const double ua_scale = fixture_.b[row] - sign * correction_b;
    for (int j = 0; j < m; ++j)
      updated_ua[j] += column[j] * ua_scale;
    updated_g.resize(active - 1);
    for (int target = 0; target < active - 1; ++target) {
      const int source = target < removed ? target : target + 1;
      double row_action = 0.0;
      const auto surviving_row = rows_[source];
      for (auto p = fixture_.indptr[surviving_row];
           p < fixture_.indptr[surviving_row + 1]; ++p)
        row_action += fixture_.values[p] * column[fixture_.indices[p]];
      updated_g[target] = recurrence_g_[source]
                          + row_action * ua_scale;
    }
    double target_coefficient = 0.0;
    for (int j = 0; j < m; ++j)
      target_coefficient += column[j] * fixture_.d[j];
    updated_gamma.resize(active - 1);
    for (int target = 0; target < active - 1; ++target) {
      const int source = target < removed ? target : target + 1;
      updated_gamma[target] = recurrence_gamma_[source]
                              + sign * correction[target]
                                    * target_coefficient;
    }
  }
  pseudoinverse_ = std::move(reduced);
  rows_.erase(found);
  if (update_coefficients) {
    recurrence_ua_ = std::move(updated_ua);
    recurrence_g_ = std::move(updated_g);
    recurrence_gamma_ = std::move(updated_gamma);
    recurrence_uc_.clear();
  } else {
    recurrence_g_.clear();
    recurrence_coefficients_valid_ = false;
  }
  ++recurrence_updates_;
  if (rank_decrease) {
    --recurrence_rank_;
    // The Greville pseudoinverse downdate is complete, but the optional
    // auxiliary row-space projector would require its own orthogonal
    // downdate.  Discard only that accelerator; subsequent rank tests can be
    // formed exactly from A*A+.
    recurrence_row_space_.clear();
    ++stats_.rank_changes;
    ++stats_.rank_decreases;
  }
  return true;
}

bool RevisedColumnSolver::transition_recurrence(
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
  if (additions.size() + removals.size() > 8) {
    if (shared_direct_seed_ && can_request_direct_seed()) {
      valid_ = false;
      direct_seeded_ = false;
      pseudoinverse_.clear();
      recurrence_coefficients_valid_ = false;
      return false;
    }
    if (recurrence_rebases_ == 0) {
      ++recurrence_rebases_;
      return rebuild(rows);
    }
    return false;
  }
  const auto exchanges = std::min(additions.size(), removals.size());
  for (std::size_t i = 0; i < exchanges; ++i)
    if (!add_row_recurrence(additions[i])
        || !remove_row_recurrence(removals[i]))
      return false;
  for (std::size_t i = exchanges; i < additions.size(); ++i)
    if (!add_row_recurrence(additions[i])) return false;
  for (std::size_t i = exchanges; i < removals.size(); ++i)
    if (!remove_row_recurrence(removals[i])) return false;
  if (rows_ != rows) return false;
  ++stats_.local_transitions;
  stats_.row_additions += additions.size();
  stats_.row_removals += removals.size();
  stats_.transition_ms += milliseconds_since(start);
  return true;
}

bool RevisedColumnSolver::form_recurrence_solution(
    RevisedFaceSolution &solution) {
  const auto solve_start = Clock::now();
  const int active = static_cast<int>(rows_.size());
  const int m = static_cast<int>(fixture_.m);
  if (active <= 0
      || pseudoinverse_.size() != static_cast<std::size_t>(m) * active)
    return false;
  if (recurrence_updates_ == 0 && !direct_seeded_
      && (diagonal_ratio(active_factor_, basis_columns_.size()) < 1e-5
          || diagonal_ratio(transform_factor_, basis_columns_.size()) < 1e-5))
    return false;

  std::vector<double> active_b(active), adjusted_d = fixture_.d;
  for (int local = 0; local < active; ++local) {
    const auto row = rows_[local];
    active_b[local] = fixture_.b[row];
    if (!target_shift_.empty())
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        adjusted_d[fixture_.indices[p]] -=
            fixture_.values[p] * target_shift_[row];
  }
  solution = RevisedFaceSolution{};
  solution.rows = rows_;
  solution.rank = recurrence_rank_;
  solution.basis_diagonal_ratio = 1.0;
  solution.coordinate_diagonal_ratio = 1.0;
  std::vector<double> gamma;
  if (recurrence_coefficients_valid_ && target_shift_.empty()
      && recurrence_gamma_.size() == static_cast<std::size_t>(active)) {
    solution.ua = recurrence_ua_;
    gamma = recurrence_gamma_;
    if (recurrence_uc_.size() != static_cast<std::size_t>(m)) {
      recurrence_uc_.assign(m, 0.0);
      cblas_dgemv(CblasColMajor, CblasNoTrans, m, active, 1.0,
                  pseudoinverse_.data(), m, gamma.data(), 1, 0.0,
                  recurrence_uc_.data(), 1);
    }
    solution.uc = recurrence_uc_;
  } else {
    solution.ua.assign(m, 0.0);
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, active, -1.0,
                pseudoinverse_.data(), m, active_b.data(), 1, 0.0,
                solution.ua.data(), 1);
    gamma.assign(active, 0.0);
    cblas_dgemv(CblasColMajor, CblasTrans, m, active, 1.0,
                pseudoinverse_.data(), m, adjusted_d.data(), 1, 0.0,
                gamma.data(), 1);
    solution.uc.assign(m, 0.0);
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, active, 1.0,
                pseudoinverse_.data(), m, gamma.data(), 1, 0.0,
                solution.uc.data(), 1);
    if (target_shift_.empty()) {
      recurrence_ua_ = solution.ua;
      recurrence_gamma_ = gamma;
      recurrence_uc_ = solution.uc;
      recurrence_coefficients_valid_ = true;
    }
  }
  if (recurrence_coefficients_valid_ && target_shift_.empty()
      && recurrence_g_.size() == static_cast<std::size_t>(active)) {
    solution.g = recurrence_g_;
  } else {
    solution.g = active_b;
    for (int local = 0; local < active; ++local) {
      const auto row = rows_[local];
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        solution.g[local] +=
            fixture_.values[p] * solution.ua[fixture_.indices[p]];
    }
  }
  solution.h.resize(active);
  for (int local = 0; local < active; ++local)
    solution.h[local] =
        (target_shift_.empty() ? 0.0 : target_shift_[rows_[local]])
        + gamma[local];
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

  std::vector<double> transpose_g(m, 0.0), dual_residual(m, 0.0);
  for (int local = 0; local < active; ++local) {
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      transpose_g[column] += fixture_.values[p] * solution.g[local];
      dual_residual[column] += fixture_.values[p] * solution.h[local];
    }
  }
  for (int column = 0; column < m; ++column)
    dual_residual[column] -= fixture_.d[column];
  solution.dres = norm2(dual_residual) / std::max(1.0, norm2(fixture_.d));
  solution.piece_residual =
      inf_norm(transpose_g) / std::max(1.0, inf_norm(solution.g));
  const double residual = std::max(solution.dres, solution.piece_residual);
  if (std::getenv("TWALKER_REVISED_TRACE_FAILURE"))
    std::cerr << "recurrence face rank=" << recurrence_rank_
              << " dres=" << solution.dres
              << " piece=" << solution.piece_residual << '\n';
  if (!std::isfinite(residual) || residual > 2e-11) return false;
  return true;
}

bool RevisedColumnSolver::transition(
    const std::vector<std::uint32_t> &rows) {
  if (coefficient_recurrence_) return transition_recurrence(rows);
  const auto start = Clock::now();
  if (!valid_) return factored_direct_seed_ ? false : rebuild(rows);
  if (rows == rows_) {
    ++stats_.unchanged_reuses;
    return true;
  }
  std::vector<std::uint32_t> additions, removals;
  std::set_difference(rows.begin(), rows.end(), rows_.begin(), rows_.end(),
                      std::back_inserter(additions));
  std::set_difference(rows_.begin(), rows_.end(), rows.begin(), rows.end(),
                      std::back_inserter(removals));
  // This lane is for ordinary tableau pivots, not wholesale support repairs.
  // A large jump is cheaper and safer in the incumbent direct solver, and
  // retiring prevents the same degenerate repair from being replayed here.
  std::size_t maximum_local_change = 8;
  if (const char *raw = std::getenv("TWALKER_REVISED_MAX_CHANGE"))
    maximum_local_change = static_cast<std::size_t>(std::stoul(raw));
  if (factored_direct_seed_
      && additions.size() + removals.size() > maximum_local_change) {
    if (can_request_direct_seed()) {
      valid_ = false;
      direct_seeded_ = false;
      return false;
    }
    retired_ = true;
    ++stats_.retirements;
    return false;
  }
  if (persistent_rank_updates_
      && additions.size() + removals.size() > maximum_local_change) {
    retired_ = true;
    ++stats_.retirements;
    return false;
  }
  // Pair entering and leaving rows, and enter first within each pair.  Doing
  // all entries first can inflate the intermediate rank during a large settle;
  // doing all exits first manufactures rank drop/rise pairs.  Interleaving is
  // the revised-simplex exchange order.
  const auto exchanges = std::min(additions.size(), removals.size());
  auto report_failure = [&](const char *operation, std::uint32_t row) {
    static int reports = 0;
    if (std::getenv("TWALKER_REVISED_TRACE_FAILURE") && reports++ < 20)
      std::cerr << "revised transition " << operation << " row=" << row
                << " rank=" << basis_columns_.size()
                << " additions=" << additions.size()
                << " removals=" << removals.size()
                << " failure=" << last_update_failure_ << '\n';
  };
  auto recover_or_rebuild = [&]() {
    if (!factored_direct_seed_) return retired_ ? false : rebuild(rows);
    if (retired_) return false;
    if (can_request_direct_seed()) {
      valid_ = false;
      direct_seeded_ = false;
      return false;
    }
    retired_ = true;
    ++stats_.retirements;
    return false;
  };
  for (std::size_t i = 0; i < exchanges; ++i) {
    if (!add_row(additions[i])) {
      report_failure("add", additions[i]);
      ++stats_.update_failures;
      return recover_or_rebuild();
    }
    if (!remove_row(removals[i])) {
      report_failure("remove", removals[i]);
      ++stats_.update_failures;
      return recover_or_rebuild();
    }
  }
  for (std::size_t i = exchanges; i < additions.size(); ++i)
    if (!add_row(additions[i])) {
      report_failure("add", additions[i]);
      ++stats_.update_failures;
      return recover_or_rebuild();
    }
  for (std::size_t i = exchanges; i < removals.size(); ++i)
    if (!remove_row(removals[i])) {
      report_failure("remove", removals[i]);
      ++stats_.update_failures;
      return recover_or_rebuild();
    }
  if (rows_ != rows) return recover_or_rebuild();
  if (reduced_sparse_factored_) {
    const auto condition_start = Clock::now();
    reduced_rcond_ = cholmod_l_rcond(reduced_factor_, &common_);
    stats_.condition_ms += milliseconds_since(condition_start);
    const bool condition_failed = shadow_square_root_
        ? diagonal_ratio(active_factor_, basis_columns_.size()) < 1e-5
        : (!std::isfinite(reduced_rcond_) || reduced_rcond_ < 1e-10);
    if (condition_failed) {
      if (std::getenv("TWALKER_REVISED_TRACE_FAILURE"))
        std::cerr << "revised transition condition decline shadow_ratio="
                  << diagonal_ratio(active_factor_, basis_columns_.size())
                  << " gram_rcond=" << reduced_rcond_ << '\n';
      ++stats_.condition_declines;
      if (can_reseed_epoch()) {
        valid_ = false;
        direct_seeded_ = false;
      } else {
        retired_ = true;
        ++stats_.retirements;
      }
      return false;
    }
  }
  ++stats_.local_transitions;
  stats_.row_additions += additions.size();
  stats_.row_removals += removals.size();
  stats_.transition_ms += milliseconds_since(start);
  return true;
}

bool RevisedColumnSolver::form_solution(RevisedFaceSolution &solution) {
  if (coefficient_recurrence_) return form_recurrence_solution(solution);
  const auto solve_start = Clock::now();
  const int rank = static_cast<int>(basis_columns_.size());
  const double factor_diagonal_gate =
      (persistent_rank_updates_ || factored_direct_seed_) ? 1e-5 : 1e-3;
  const double active_ratio = reduced_sparse_factored_
                                  ? (shadow_square_root_
                                         ? diagonal_ratio(active_factor_, rank)
                                         : std::sqrt(std::max(
                                               0.0, reduced_rcond_)))
                                  : diagonal_ratio(active_factor_, rank);
  const double transform_ratio = diagonal_ratio(transform_factor_, rank);
  const bool audit_below_gate = svd_orthonormal_
      && !std::getenv("TWALKER_DISABLE_REVISED_SVD_AUDIT_BELOW_GATE");
  if ((active_ratio < factor_diagonal_gate && !audit_below_gate)
      || transform_ratio < factor_diagonal_gate) {
    if (std::getenv("TWALKER_REVISED_TRACE_FAILURE"))
      std::cerr << "revised condition decline active=" << active_ratio
                << " transform=" << transform_ratio
                << " gate=" << factor_diagonal_gate << '\n';
    ++stats_.condition_declines;
    if (persistent_rank_updates_) {
      retired_ = true;
      ++stats_.retirements;
    }
    return false;
  }

  std::vector<double> adjusted_d = fixture_.d;
  if (!target_shift_.empty())
    for (auto row : rows_)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        adjusted_d[fixture_.indices[p]] -=
            fixture_.values[p] * target_shift_[row];

  auto solve_reduced = [&](std::vector<double> &right, int right_count = 1) {
    if (shadow_square_root_ || shadow_solve_only_)
      return solve_cholesky(active_factor_, rank, right, right_count);
    auto *rhs = cholmod_l_allocate_dense(rank, right_count, rank, CHOLMOD_REAL,
                                         &common_);
    if (!rhs) return false;
    auto *rhs_values = static_cast<double *>(rhs->x);
    for (int column = 0; column < right_count; ++column)
      for (int row = 0; row < rank; ++row)
        rhs_values[row + static_cast<std::size_t>(rank) * column] =
            reduced_scale_[row]
            * right[row + static_cast<std::size_t>(rank) * column];
    auto *answer = cholmod_l_solve(CHOLMOD_A, reduced_factor_, rhs, &common_);
    cholmod_l_free_dense(&rhs, &common_);
    if (!answer) return false;
    const auto *values = static_cast<const double *>(answer->x);
    for (int column = 0; column < right_count; ++column)
      for (int row = 0; row < rank; ++row)
        right[row + static_cast<std::size_t>(rank) * column] =
            reduced_scale_[row]
            * values[row + static_cast<std::size_t>(rank) * column];
    cholmod_l_free_dense(&answer, &common_);
    return true;
  };

  auto refine_active = [&](const std::vector<double> &right,
                           std::vector<double> &answer) {
    if (reduced_sparse_factored_) {
      // Equilibration can make the scaled factor look well conditioned while
      // the mapped-back solution still carries the original system's normal-
      // equation error.  Do one correction in original coordinates before
      // granting the wider, corrected-solution residual gate.
      if (active_ratio >= 1e-3 && !reduced_equilibration_) return true;
      const int correction_count =
          std::getenv("TWALKER_REVISED_SECOND_CORRECTION") ? 2 : 1;
      for (int iteration = 0; iteration < correction_count; ++iteration) {
        std::vector<long double> residual(right.begin(), right.end());
        for (auto row : rows_) {
          long double product = 0.0L;
          for (auto p = fixture_.indptr[row];
               p < fixture_.indptr[row + 1]; ++p) {
            const auto position = basis_position_[fixture_.indices[p]];
            if (position >= 0)
              product += static_cast<long double>(fixture_.values[p])
                         * answer[position];
          }
          for (auto p = fixture_.indptr[row];
               p < fixture_.indptr[row + 1]; ++p) {
            const auto position = basis_position_[fixture_.indices[p]];
            if (position >= 0)
              residual[position] -=
                  static_cast<long double>(fixture_.values[p]) * product;
          }
        }
        std::vector<double> correction(rank);
        for (int j = 0; j < rank; ++j)
          correction[j] = static_cast<double>(residual[j]);
        if (!solve_reduced(correction)) return false;
        for (int j = 0; j < rank; ++j) answer[j] += correction[j];
        ++stats_.refinements;
      }
      return true;
    }
    const bool marginal = !reduced_sparse_factored_
                          && diagonal_ratio(active_factor_, rank) < 1e-3;
    const int iterations = marginal ? (persistent_rank_updates_ ? 3 : 1) : 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const int active = static_cast<int>(rows_.size());
      std::vector<double> correction(rank);
      if (marginal && persistent_rank_updates_) {
        std::vector<long double> residual(right.begin(), right.end());
        for (int local = 0; local < active; ++local) {
          long double product = 0.0L;
          for (int j = 0; j < rank; ++j)
            product += static_cast<long double>(
                           coordinates_[static_cast<std::size_t>(local)
                                        * rank + j])
                       * answer[j];
          for (int j = 0; j < rank; ++j)
            residual[j] -= static_cast<long double>(
                               coordinates_[static_cast<std::size_t>(local)
                                            * rank + j])
                           * product;
        }
        for (int j = 0; j < rank; ++j)
          correction[j] = static_cast<double>(residual[j]);
      } else {
        std::vector<double> product(active, 0.0);
        correction = right;
        cblas_dgemv(CblasRowMajor, CblasNoTrans, active, rank, 1.0,
                    coordinates_.data(), rank, answer.data(), 1, 0.0,
                    product.data(), 1);
        cblas_dgemv(CblasRowMajor, CblasTrans, active, rank, -1.0,
                    coordinates_.data(), rank, product.data(), 1, 1.0,
                    correction.data(), 1);
      }
      if (!solve_cholesky(active_factor_, rank, correction)) return false;
      for (int j = 0; j < rank; ++j) answer[j] += correction[j];
      ++stats_.refinements;
    }
    return true;
  };

  const auto coefficient_start = Clock::now();
  std::vector<double> target_head;
  if (target_shift_.empty()
      && fixed_target_head_.size() == static_cast<std::size_t>(rank)) {
    target_head = fixed_target_head_;
  } else {
    std::vector<double> td(rank, 0.0);
    cblas_dgemv(CblasColMajor, CblasNoTrans, rank, fixture_.m, 1.0,
                transform_.data(), rank, adjusted_d.data(), 1, 0.0,
                td.data(), 1);
    target_head = std::move(td);
    if (!orthonormal_factored_ && !reduced_sparse_factored_
        && !solve_cholesky(transform_factor_, rank, target_head))
      return false;
    if (target_shift_.empty()) fixed_target_head_ = target_head;
  }
  std::vector<double> active_heads(static_cast<std::size_t>(rank) * 2);
  std::copy(active_cross_b_.begin(), active_cross_b_.end(),
            active_heads.begin());
  std::copy(target_head.begin(), target_head.end(),
            active_heads.begin() + rank);
  if (reduced_sparse_factored_) {
    if (!solve_reduced(active_heads, 2)) return false;
  } else if (!solve_cholesky(active_factor_, rank, active_heads, 2)) {
    return false;
  }
  std::vector<double> alpha(active_heads.begin(),
                            active_heads.begin() + rank);
  std::vector<double> h_head(active_heads.begin() + rank,
                             active_heads.end());
  const auto refinements_before = stats_.refinements;
  if (!refine_active(active_cross_b_, alpha)
      || !refine_active(target_head, h_head))
    return false;
  const bool marginally_refined = stats_.refinements > refinements_before;
  std::copy(alpha.begin(), alpha.end(), active_heads.begin());
  std::copy(h_head.begin(), h_head.end(), active_heads.begin() + rank);
  if (!orthonormal_factored_ && !reduced_sparse_factored_
      && !solve_cholesky(transform_factor_, rank, active_heads, 2))
    return false;
  std::vector<double> ua_head(active_heads.begin(),
                              active_heads.begin() + rank);
  std::vector<double> uc_head(active_heads.begin() + rank,
                              active_heads.end());
  stats_.coefficient_ms += milliseconds_since(coefficient_start);

  const auto projection_start = Clock::now();
  solution = RevisedFaceSolution{};
  solution.rows = rows_;
  solution.rank = rank;
  solution.basis_diagonal_ratio = diagonal_ratio(active_factor_, rank);
  solution.coordinate_diagonal_ratio = diagonal_ratio(transform_factor_, rank);
  std::vector<double> projected_heads(static_cast<std::size_t>(rank) * 2);
  for (int j = 0; j < rank; ++j) {
    projected_heads[j] = -ua_head[j];
    projected_heads[rank + j] = uc_head[j];
  }
  std::vector<double> projected_coefficients(fixture_.m * 2, 0.0);
  if (reduced_sparse_factored_) {
    // With the independent columns first, T=[I,D] and
    // M'=T'(TT')^-1.  Woodbury applies M' with work proportional to the
    // rank deficiency k=m-r, rather than multiplying the dense m-by-r M'.
    // The full-rank case becomes a permutation/copy.
    const int deficiency = static_cast<int>(dependent_columns_.size());
    std::vector<double> minimum_norm_heads = projected_heads;
    if (deficiency > 0) {
      if (dependency_transform_.size()
              != static_cast<std::size_t>(rank) * deficiency
          || dependency_factor_.size()
                 != static_cast<std::size_t>(deficiency) * deficiency)
        return false;
      std::vector<double> correction(static_cast<std::size_t>(deficiency) * 2,
                                     0.0);
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, deficiency, 2,
                  rank, 1.0, dependency_transform_.data(), rank,
                  projected_heads.data(), rank, 0.0, correction.data(),
                  deficiency);
      if (!solve_cholesky(dependency_factor_, deficiency, correction, 2))
        return false;
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, 2,
                  deficiency, -1.0, dependency_transform_.data(), rank,
                  correction.data(), deficiency, 1.0,
                  minimum_norm_heads.data(), rank);
    }
    for (int rhs = 0; rhs < 2; ++rhs) {
      for (int basis = 0; basis < rank; ++basis)
        projected_coefficients[basis_columns_[basis]
                               + static_cast<std::size_t>(fixture_.m) * rhs] =
            minimum_norm_heads[basis + static_cast<std::size_t>(rank) * rhs];
    }
    if (deficiency > 0) {
      std::vector<double> dependent_values(
          static_cast<std::size_t>(deficiency) * 2, 0.0);
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, deficiency, 2,
                  rank, 1.0, dependency_transform_.data(), rank,
                  minimum_norm_heads.data(), rank, 0.0,
                  dependent_values.data(), deficiency);
      for (int rhs = 0; rhs < 2; ++rhs)
        for (int dependent = 0; dependent < deficiency; ++dependent)
          projected_coefficients[
              dependent_columns_[dependent]
              + static_cast<std::size_t>(fixture_.m) * rhs] =
              dependent_values[dependent
                               + static_cast<std::size_t>(deficiency) * rhs];
    }
  } else {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, fixture_.m, 2,
                rank, 1.0, transform_.data(), rank, projected_heads.data(),
                rank, 0.0, projected_coefficients.data(), fixture_.m);
  }
  solution.ua.assign(projected_coefficients.begin(),
                     projected_coefficients.begin() + fixture_.m);
  solution.uc.assign(projected_coefficients.begin() + fixture_.m,
                     projected_coefficients.end());
  solution.g.resize(rows_.size());
  solution.h.resize(rows_.size());
  if (reduced_sparse_factored_) {
    for (std::size_t local = 0; local < rows_.size(); ++local) {
      const auto row = rows_[local];
      double ca = 0.0, ch = 0.0;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto position = basis_position_[fixture_.indices[p]];
        if (position < 0) continue;
        ca += fixture_.values[p] * alpha[position];
        ch += fixture_.values[p] * h_head[position];
      }
      solution.g[local] = fixture_.b[row] - ca;
      solution.h[local] =
          (target_shift_.empty() ? 0.0 : target_shift_[row]) + ch;
    }
  } else {
    std::vector<double> face_heads(static_cast<std::size_t>(rank) * 2);
    for (int j = 0; j < rank; ++j) {
      face_heads[2 * j] = alpha[j];
      face_heads[2 * j + 1] = h_head[j];
    }
    std::vector<double> face_values(rows_.size() * 2, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_.size(), 2,
                rank, 1.0, coordinates_.data(), rank, face_heads.data(), 2,
                0.0, face_values.data(), 2);
    for (std::size_t local = 0; local < rows_.size(); ++local) {
      const auto row = rows_[local];
      solution.g[local] = fixture_.b[row] - face_values[2 * local];
      solution.h[local] =
          (target_shift_.empty() ? 0.0 : target_shift_[row])
          + face_values[2 * local + 1];
    }
  }
  stats_.projection_ms += milliseconds_since(projection_start);
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

  const auto residual_start = Clock::now();
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
  double reduced_residual_gate = marginally_refined ? 1e-10 : 1e-11;
  if (const char *raw = std::getenv("TWALKER_REVISED_RESIDUAL_GATE"))
    reduced_residual_gate = std::stod(raw);
  const double residual_gate =
      reduced_sparse_factored_ ? reduced_residual_gate : 1e-10;
  if (!std::isfinite(residual) || residual > residual_gate) {
    if (std::getenv("TWALKER_REVISED_TRACE_FAILURE"))
      std::cerr << "revised residual decline dres=" << solution.dres
                << " piece=" << solution.piece_residual << '\n';
    ++stats_.residual_declines;
    if (persistent_rank_updates_) {
      retired_ = true;
      ++stats_.retirements;
    }
    return false;
  }

  if ((std::getenv("TWALKER_REVISED_LOCAL_BOUND")
       || std::getenv("TWALKER_REVISED_BOUND_AUDIT"))
      && reduced_sparse_factored_ && !reduced_equilibration_
      && std::isfinite(reduced_rcond_) && reduced_rcond_ > 0.0
      && std::isfinite(projection_inf_norm_)
      && projection_inf_norm_ > 0.0) {
    std::vector<long double> normal_residual(
        static_cast<std::size_t>(rank) * 2, 0.0L);
    std::vector<long double> gram_row_sum(rank, 0.0L);
    for (int j = 0; j < rank; ++j) {
      normal_residual[j] = active_cross_b_[j];
      normal_residual[rank + j] = target_head[j];
    }
    for (auto row : rows_) {
      long double row_l1 = 0.0L;
      long double products[2] = {0.0L, 0.0L};
      for (auto p = fixture_.indptr[row];
           p < fixture_.indptr[row + 1]; ++p) {
        const auto position = basis_position_[fixture_.indices[p]];
        if (position < 0) continue;
        const long double value = fixture_.values[p];
        row_l1 += std::abs(value);
        products[0] += value * static_cast<long double>(alpha[position]);
        products[1] += value * static_cast<long double>(h_head[position]);
      }
      for (auto p = fixture_.indptr[row];
           p < fixture_.indptr[row + 1]; ++p) {
        const auto position = basis_position_[fixture_.indices[p]];
        if (position < 0) continue;
        const long double value = fixture_.values[p];
        gram_row_sum[position] += std::abs(value) * row_l1;
        normal_residual[position] -= value * products[0];
        normal_residual[rank + position] -= value * products[1];
      }
    }
    long double gram_inf = 0.0L;
    for (auto value : gram_row_sum) gram_inf = std::max(gram_inf, value);
    if (gram_inf > 0.0L && std::isfinite(gram_inf)) {
      const std::vector<double> *heads[2] = {&alpha, &h_head};
      const std::vector<double> *coefficients[2] = {&solution.ua,
                                                    &solution.uc};
      double relative_bounds[2] = {
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity()};
      bool finite = true;
      for (int right = 0; right < 2; ++right) {
        long double residual_inf = 0.0L;
        for (int j = 0; j < rank; ++j)
          residual_inf = std::max(
              residual_inf,
              std::abs(normal_residual[right * rank + j]));
        const long double head_error =
            10.0L * residual_inf
            / (static_cast<long double>(reduced_rcond_) * gram_inf);
        const long double projection_roundoff =
            64.0L * std::numeric_limits<double>::epsilon()
            * static_cast<long double>(projection_inf_norm_)
            * std::max(1.0L,
                       static_cast<long double>(inf_norm(*heads[right])));
        const long double coefficient_error =
            static_cast<long double>(projection_inf_norm_) * head_error
            + projection_roundoff;
        relative_bounds[right] = static_cast<double>(
            coefficient_error
            / std::max(1.0L, static_cast<long double>(
                                  inf_norm(*coefficients[right]))));
        finite = finite && std::isfinite(relative_bounds[right])
                 && relative_bounds[right] >= 0.0;
      }
      if (finite) {
        solution.forward_bound_valid = true;
        solution.ua_relative_error_bound = relative_bounds[0];
        solution.uc_relative_error_bound = relative_bounds[1];
        solution.reduced_residual_a.resize(rank);
        solution.reduced_residual_c.resize(rank);
        for (int j = 0; j < rank; ++j) {
          solution.reduced_residual_a[j] =
              static_cast<double>(normal_residual[j]);
          solution.reduced_residual_c[j] =
              static_cast<double>(normal_residual[rank + j]);
        }
        solution.reduced_head_a = alpha;
        solution.reduced_head_c = h_head;
        solution.reduced_gram_inf = static_cast<double>(gram_inf);
        solution.reduced_rcond = reduced_rcond_;
        solution.projection_inf_norm = projection_inf_norm_;
      }
    }
  }
  stats_.residual_ms += milliseconds_since(residual_start);
  return true;
}

bool RevisedColumnSolver::solve(const std::vector<std::uint32_t> &rows,
                                RevisedFaceSolution &solution) {
  ++stats_.calls;
  if (retired_) {
    ++stats_.declines;
    return false;
  }
  // In the shared-start experiment, an uninitialized recurrence waits for
  // the incumbent direct solve.  Walker then hands back the accepted COD
  // artifact; no second SPQR factorization is performed.
  if (((coefficient_recurrence_ && shared_direct_seed_)
       || factored_direct_seed_) && !valid_) {
    ++stats_.declines;
    return false;
  }
  if (!std::is_sorted(rows.begin(), rows.end()) || !transition(rows)) {
    ++stats_.declines;
    if (coefficient_recurrence_) {
      if (forced_shared_recurrence_ && can_request_direct_seed()) {
        // A failed local Greville exchange is a request for a fresh artifact
        // from the direct solve already required for this face, not a reason
        // to abandon the maintained deficient-face lane permanently.
        valid_ = false;
        direct_seeded_ = false;
        pseudoinverse_.clear();
        recurrence_g_.clear();
        recurrence_coefficients_valid_ = false;
      } else if (!(shared_direct_seed_ && !valid_
                   && can_request_direct_seed())) {
        retired_ = true;
        ++stats_.retirements;
      }
    } else {
      if (factored_direct_seed_) {
        if (!(valid_ == false && can_request_direct_seed())) {
          retired_ = true;
          ++stats_.retirements;
        }
      } else {
        valid_ = false;
      }
    }
    return false;
  }
  const auto residual_declines_before = stats_.residual_declines;
  if (!form_solution(solution)) {
    if (std::getenv("TWALKER_REVISED_TRACE_FAILURE"))
      std::cerr << "revised form decline residual_delta="
                << (stats_.residual_declines - residual_declines_before)
                << " condition_declines=" << stats_.condition_declines
                << " rcond=" << reduced_rcond_ << '\n';
    ++stats_.declines;
    if (coefficient_recurrence_) {
      if (forced_shared_recurrence_ && can_request_direct_seed()) {
        // The residual gate is the cheap epoch-health test.  Rebase from the
        // incumbent oracle on this same face; never propagate a marginal
        // pseudoinverse update into subsequent path decisions.
        valid_ = false;
        direct_seeded_ = false;
        pseudoinverse_.clear();
        recurrence_g_.clear();
        recurrence_coefficients_valid_ = false;
      } else {
        retired_ = true;
        ++stats_.retirements;
      }
    }
    if (factored_direct_seed_) {
      const bool residual_reseed =
          !std::getenv("TWALKER_DISABLE_REVISED_MULTI_EPOCH")
          && stats_.residual_declines > residual_declines_before
          && can_reseed_epoch();
      if (residual_reseed) {
        // The declined face is recomputed by the incumbent direct lane.  Ask
        // that already-required solve to export a new factored epoch instead
        // of permanently abandoning cheap row updates.
        valid_ = false;
        direct_seeded_ = false;
      } else {
        retired_ = true;
        ++stats_.retirements;
      }
    }
    return false;
  }
  ++successful_faces_;
  return true;
}

bool RevisedColumnSolver::seed_from_direct(
    const std::vector<std::uint32_t> &rows, FaceSolution &direct_solution) {
  if ((!coefficient_recurrence_ || !shared_direct_seed_)
      && !factored_direct_seed_)
    return false;
  if (valid_ || retired_ || !can_request_direct_seed())
    return false;
  const auto active = rows.size();
  const auto m = fixture_.m;
  if (factored_direct_seed_) {
    const auto seed_start = Clock::now();
    const int rank = static_cast<int>(direct_solution.factored_seed_rank);
    if (!std::getenv("TWALKER_DISABLE_REVISED_COST_AWARE_SEED")
        && direct_seed_count_ == 0
        && (!direct_solution.used_dense_fallback
            || rank <= 0
            || m - static_cast<std::size_t>(rank) > 4)) {
      // Stay dormant until an actually expensive, near-full-rank direct face
      // supplies a seed.  Do not construct revised state speculatively.
      return false;
    }
    if (rows.empty() || direct_solution.rows != rows || rank <= 0
        || rank > static_cast<int>(m)
        || direct_solution.factored_qr_core.size()
               != static_cast<std::size_t>(rank) * m
        || direct_solution.factored_rz_core.size()
               != static_cast<std::size_t>(rank) * m
        || direct_solution.factored_rz_tau.size()
               != static_cast<std::size_t>(rank)
        || direct_solution.factored_permutation.size() != m)
      return false;
    maximum_seed_deficiency_ = std::max(
        maximum_seed_deficiency_,
        static_cast<std::uint32_t>(m - static_cast<std::size_t>(rank)));
    rows_ = rows;
    basis_columns_.resize(rank);
    basis_position_.assign(m, -1);
    for (int position = 0; position < rank; ++position) {
      const auto column = direct_solution.factored_permutation[position];
      if (column < 0 || column >= static_cast<std::int64_t>(m)
          || basis_position_[column] >= 0)
        return false;
      basis_columns_[position] = static_cast<std::uint32_t>(column);
      basis_position_[column] = position;
    }

    if (rz_orthonormal_ || svd_orthonormal_) {
      if (svd_orthonormal_) {
        if (direct_solution.svd_row_space.size()
            != static_cast<std::size_t>(rank) * m)
          return false;
        transform_ = direct_solution.svd_row_space;
      } else {
        std::vector<double> orthonormal_columns(
            m * static_cast<std::size_t>(rank), 0.0);
        for (int basis = 0; basis < rank; ++basis)
          orthonormal_columns[basis + m * static_cast<std::size_t>(basis)] =
              1.0;
        if (rank < static_cast<int>(m)) {
          const char side = 'L', transpose = 'T';
          const int mm = static_cast<int>(m), reflector_tail = mm - rank;
          int info = 0, lwork = -1;
          double query = 0.0;
          dormrz_(&side, &transpose, &mm, &rank, &rank, &reflector_tail,
                  direct_solution.factored_rz_core.data(), &rank,
                  direct_solution.factored_rz_tau.data(),
                  orthonormal_columns.data(), &mm, &query, &lwork, &info);
          if (info != 0) return false;
          lwork = std::max(1, static_cast<int>(std::ceil(query)));
          std::vector<double> work(lwork);
          dormrz_(&side, &transpose, &mm, &rank, &rank, &reflector_tail,
                  direct_solution.factored_rz_core.data(), &rank,
                  direct_solution.factored_rz_tau.data(),
                  orthonormal_columns.data(), &mm, work.data(), &lwork,
                  &info);
          if (info != 0) return false;
        }
        transform_.assign(static_cast<std::size_t>(rank) * m, 0.0);
        for (std::size_t position = 0; position < m; ++position) {
          const auto original = direct_solution.factored_permutation[position];
          if (original < 0 || original >= static_cast<std::int64_t>(m))
            return false;
          for (int basis = 0; basis < rank; ++basis)
            transform_[basis + static_cast<std::size_t>(rank) * original] =
                orthonormal_columns[position
                                    + m * static_cast<std::size_t>(basis)];
        }
      }
      projection_inf_norm_ = 0.0;
      for (std::size_t column = 0; column < m; ++column) {
        long double row_sum = 0.0L;
        for (int basis = 0; basis < rank; ++basis)
          row_sum += std::abs(static_cast<long double>(
              transform_[basis + static_cast<std::size_t>(rank) * column]));
        projection_inf_norm_ = std::max(
            projection_inf_norm_, static_cast<double>(row_sum));
      }
      if (!std::isfinite(projection_inf_norm_)
          || projection_inf_norm_ <= 0.0)
        return false;
      coordinates_.assign(active * static_cast<std::size_t>(rank), 0.0);
      for (std::size_t local = 0; local < active; ++local) {
        const auto row = rows[local];
        for (auto p = fixture_.indptr[row];
             p < fixture_.indptr[row + 1]; ++p) {
          const auto column = fixture_.indices[p];
          const double value = fixture_.values[p];
          for (int basis = 0; basis < rank; ++basis)
            coordinates_[local * rank + basis] +=
                value * transform_[basis
                                   + static_cast<std::size_t>(rank) * column];
        }
      }
      if (svd_orthonormal_) {
        std::vector<double> coordinate_columns(
            active * static_cast<std::size_t>(rank));
        for (std::size_t local = 0; local < active; ++local)
          for (int basis = 0; basis < rank; ++basis)
            coordinate_columns[local + active * static_cast<std::size_t>(basis)] =
                coordinates_[local * rank + basis];
        if (!qr_square_root(std::move(coordinate_columns),
                            static_cast<int>(active), rank, active_factor_))
          return false;
      } else {
        active_factor_.assign(static_cast<std::size_t>(rank) * rank, 0.0);
        for (int column = 0; column < rank; ++column)
          for (int row = column; row < rank; ++row) {
            long double value = 0.0L;
            for (std::size_t local = 0; local < active; ++local)
              value += static_cast<long double>(
                           coordinates_[local * rank + row])
                       * coordinates_[local * rank + column];
            active_factor_[row + static_cast<std::size_t>(rank) * column] =
                static_cast<double>(value);
          }
        if (!cholesky(active_factor_, rank)) return false;
      }
      transform_factor_.assign(static_cast<std::size_t>(rank) * rank, 0.0);
      for (int diagonal = 0; diagonal < rank; ++diagonal)
        transform_factor_[diagonal
                          + static_cast<std::size_t>(rank) * diagonal] = 1.0;
      rank_test_transform_ = transform_;
      representation_transform_ = transform_;
      dependent_columns_.clear();
      dependency_transform_.clear();
      dependency_factor_.clear();
      fixed_target_head_.assign(rank, 0.0);
      cblas_dgemv(CblasColMajor, CblasNoTrans, rank, fixture_.m, 1.0,
                  transform_.data(), rank, fixture_.d.data(), 1, 0.0,
                  fixed_target_head_.data(), 1);
      active_cross_b_.assign(rank, 0.0);
      for (std::size_t local = 0; local < active; ++local)
        for (int basis = 0; basis < rank; ++basis)
          active_cross_b_[basis] += coordinates_[local * rank + basis]
                                    * fixture_.b[rows[local]];
      reduced_sparse_factored_ = false;
      orthonormal_factored_ = true;
      valid_ = true;
      RevisedFaceSolution seed_check;
      if (form_solution(seed_check)) {
        direct_solution.factored_qr_core.clear();
        direct_solution.factored_rz_core.clear();
        direct_solution.factored_rz_tau.clear();
        direct_solution.svd_row_space.clear();
        direct_solution.factored_permutation.clear();
        direct_seeded_ = true;
        ++direct_seed_count_;
        ++stats_.direct_seeds;
        stats_.direct_seed_ms += milliseconds_since(seed_start);
        return true;
      }
      // The SVD row space is authoritative, but it may still be a poor
      // normal-equation coordinate system.  Reject it locally and continue
      // with the independent-column representation from the same direct
      // face; no second rank-revealing factorization is needed.
      valid_ = false;
      orthonormal_factored_ = false;
      fixed_target_head_.clear();
    }

    const auto &qr_core = direct_solution.factored_qr_core;
    active_factor_.assign(static_cast<std::size_t>(rank) * rank, 0.0);
    for (int column = 0; column < rank; ++column) {
      const double diagonal =
          qr_core[column + static_cast<std::size_t>(rank) * column];
      if (diagonal == 0.0 || !std::isfinite(diagonal)) return false;
      const double sign = diagonal < 0.0 ? -1.0 : 1.0;
      for (int row = column; row < rank; ++row)
        active_factor_[row + static_cast<std::size_t>(rank) * column] =
            sign * qr_core[column + static_cast<std::size_t>(rank) * row];
    }

    std::vector<double> permuted_transform = qr_core;
    {
      const char upper = 'U', no_transpose = 'N', nonunit = 'N';
      const int right_count = static_cast<int>(m);
      int info = 0;
      dtrtrs_(&upper, &no_transpose, &nonunit, &rank, &right_count,
              qr_core.data(), &rank, permuted_transform.data(), &rank, &info);
      if (info != 0) return false;
    }
    transform_.assign(static_cast<std::size_t>(rank) * m, 0.0);
    for (std::size_t position = 0; position < m; ++position) {
      const auto original = direct_solution.factored_permutation[position];
      if (original < 0 || original >= static_cast<std::int64_t>(m))
        return false;
      for (int basis = 0; basis < rank; ++basis)
        transform_[basis + static_cast<std::size_t>(rank) * original] =
            permuted_transform[basis
                               + static_cast<std::size_t>(rank) * position];
    }
    std::vector<double> transform_transpose(m * static_cast<std::size_t>(rank));
    for (int basis = 0; basis < rank; ++basis)
      for (std::size_t column = 0; column < m; ++column)
        transform_transpose[column + m * static_cast<std::size_t>(basis)] =
            transform_[basis + static_cast<std::size_t>(rank) * column];
    if (!qr_square_root(std::move(transform_transpose), static_cast<int>(m),
                        rank, transform_factor_))
      return false;

    // Keep C as actual sparse columns.  Precompute M=(TT')^-1 T once, so
    // ua=-M'alpha and uc=M'gamma need no H solve on an ordinary pivot.
    representation_transform_ = transform_;
    dependent_columns_.clear();
    for (std::size_t column = 0; column < m; ++column)
      if (basis_position_[column] < 0)
        dependent_columns_.push_back(static_cast<std::uint32_t>(column));
    const int deficiency = static_cast<int>(dependent_columns_.size());
    dependency_transform_.assign(
        static_cast<std::size_t>(rank) * deficiency, 0.0);
    for (int dependent = 0; dependent < deficiency; ++dependent) {
      const auto column = dependent_columns_[dependent];
      std::copy_n(representation_transform_.begin()
                      + static_cast<std::size_t>(rank) * column,
                  rank,
                  dependency_transform_.begin()
                      + static_cast<std::size_t>(rank) * dependent);
    }
    dependency_factor_.assign(
        static_cast<std::size_t>(deficiency) * deficiency, 0.0);
    if (deficiency > 0) {
      for (int diagonal = 0; diagonal < deficiency; ++diagonal)
        dependency_factor_[diagonal
                           + static_cast<std::size_t>(deficiency) * diagonal] =
            1.0;
      cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, deficiency, rank,
                  1.0, dependency_transform_.data(), rank, 1.0,
                  dependency_factor_.data(), deficiency);
      if (!cholesky(dependency_factor_, deficiency)) return false;
    }
    rank_test_transform_ = transform_;
    {
      const char lower = 'L', no_transpose = 'N', nonunit = 'N';
      const int right_count = static_cast<int>(m);
      int info = 0;
      dtrtrs_(&lower, &no_transpose, &nonunit, &rank, &right_count,
              transform_factor_.data(), &rank, rank_test_transform_.data(),
              &rank, &info);
      if (info != 0) return false;
    }
    if (!solve_cholesky(transform_factor_, rank, transform_,
                        static_cast<int>(m)))
      return false;
    projection_inf_norm_ = 0.0;
    for (std::size_t column = 0; column < m; ++column) {
      long double row_sum = 0.0L;
      for (int basis = 0; basis < rank; ++basis)
        row_sum += std::abs(static_cast<long double>(
            transform_[basis + static_cast<std::size_t>(rank) * column]));
      projection_inf_norm_ = std::max(
          projection_inf_norm_, static_cast<double>(row_sum));
    }
    if (!std::isfinite(projection_inf_norm_)
        || projection_inf_norm_ <= 0.0)
      return false;
    fixed_target_head_.assign(rank, 0.0);
    cblas_dgemv(CblasColMajor, CblasNoTrans, rank, fixture_.m, 1.0,
                transform_.data(), rank, fixture_.d.data(), 1, 0.0,
                fixed_target_head_.data(), 1);

    coordinates_.clear();
    active_cross_b_.assign(rank, 0.0);
    for (std::size_t local = 0; local < active; ++local) {
      const auto row = rows[local];
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto position = basis_position_[fixture_.indices[p]];
        if (position < 0) continue;
        const double value = fixture_.values[p];
        active_cross_b_[position] += value * fixture_.b[row];
      }
    }
    reduced_sparse_factored_ = true;
    orthonormal_factored_ = false;
    if (!build_reduced_factor()) return false;
    direct_solution.factored_qr_core.clear();
    direct_solution.factored_rz_core.clear();
    direct_solution.factored_rz_tau.clear();
    direct_solution.svd_row_space.clear();
    direct_solution.factored_permutation.clear();
    direct_seeded_ = true;
    ++direct_seed_count_;
    ++stats_.direct_seeds;
    stats_.direct_seed_ms += milliseconds_since(seed_start);
    valid_ = true;
    if (std::getenv("TWALKER_REVISED_TRACE_FAILURE")) {
      RevisedFaceSolution seed_solution;
      const bool formed = form_solution(seed_solution);
      auto relative_error = [](const std::vector<double> &left,
                               const std::vector<double> &right) {
        if (left.size() != right.size())
          return std::numeric_limits<double>::infinity();
        double difference = 0.0, scale = 1.0;
        for (std::size_t i = 0; i < left.size(); ++i) {
          difference = std::max(difference, std::abs(left[i] - right[i]));
          scale = std::max(scale, std::abs(right[i]));
        }
        return difference / scale;
      };
      const double error = formed
          ? std::max(relative_error(seed_solution.ua, direct_solution.ua),
                     relative_error(seed_solution.uc, direct_solution.uc))
          : std::numeric_limits<double>::infinity();
      std::cerr << "factored direct seed active=" << active
                << " rank=" << rank << " formed=" << formed
                << " error=" << error << '\n';
    }
    return true;
  }
  if (rows.empty() || direct_solution.rows != rows
      || direct_solution.recurrence_seed_rank <= 0
      || direct_solution.recurrence_pseudoinverse.size() != m * active
      || direct_solution.ua.size() != m || direct_solution.uc.size() != m)
    return false;
  rows_ = rows;
  pseudoinverse_ = std::move(direct_solution.recurrence_pseudoinverse);
  recurrence_rank_ = direct_solution.recurrence_seed_rank;
  if (direct_solution.svd_row_space.size()
      == static_cast<std::size_t>(recurrence_rank_) * m)
    recurrence_row_space_ = direct_solution.svd_row_space;
  else if (direct_solution.factored_rz_core.size()
               == static_cast<std::size_t>(recurrence_rank_) * m
           && direct_solution.factored_rz_tau.size()
                  == static_cast<std::size_t>(recurrence_rank_)
           && direct_solution.factored_permutation.size() == m) {
    const int rank = static_cast<int>(recurrence_rank_);
    const int mm = static_cast<int>(m);
    std::vector<double> orthonormal_columns(
        m * static_cast<std::size_t>(rank), 0.0);
    for (int component = 0; component < rank; ++component)
      orthonormal_columns[component + m * static_cast<std::size_t>(component)]
          = 1.0;
    if (rank < mm) {
      const char side = 'L', transpose = 'T';
      const int reflector_tail = mm - rank;
      int info = 0, lwork = -1;
      double query = 0.0;
      dormrz_(&side, &transpose, &mm, &rank, &rank, &reflector_tail,
              direct_solution.factored_rz_core.data(), &rank,
              direct_solution.factored_rz_tau.data(),
              orthonormal_columns.data(), &mm, &query, &lwork, &info);
      if (info != 0 || !std::isfinite(query)) return false;
      lwork = std::max(1, static_cast<int>(std::ceil(query)));
      std::vector<double> work(lwork);
      dormrz_(&side, &transpose, &mm, &rank, &rank, &reflector_tail,
              direct_solution.factored_rz_core.data(), &rank,
              direct_solution.factored_rz_tau.data(),
              orthonormal_columns.data(), &mm, work.data(), &lwork, &info);
      if (info != 0) return false;
    }
    recurrence_row_space_.assign(
        static_cast<std::size_t>(rank) * m, 0.0);
    for (std::size_t position = 0; position < m; ++position) {
      const auto original = direct_solution.factored_permutation[position];
      if (original < 0 || original >= static_cast<std::int64_t>(m))
        return false;
      for (int component = 0; component < rank; ++component)
        recurrence_row_space_[component
                              + static_cast<std::size_t>(rank) * original] =
            orthonormal_columns[position
                                + m * static_cast<std::size_t>(component)];
    }
  } else
    recurrence_row_space_.clear();
  recurrence_updates_ = 0;
  // A later wholesale repair should return to the incumbent, not pay for the
  // old duplicate cold rebuild that this shared seed is meant to remove.
  recurrence_rebases_ = 1;
  recurrence_ua_ = direct_solution.ua;
  recurrence_g_ = direct_solution.g;
  recurrence_uc_ = direct_solution.uc;
  if (target_shift_.empty()
      && direct_solution.h.size() == active) {
    // As with g, preserve the orthogonal oracle's accurate minimum-norm
    // constant and update it algebraically.  Re-forming A+^T*d can carry the
    // forward error of a very ill-conditioned pseudoinverse despite a tiny
    // equality residual.
    recurrence_gamma_ = direct_solution.h;
  } else {
    recurrence_gamma_.assign(active, 0.0);
    cblas_dgemv(CblasColMajor, CblasTrans, static_cast<int>(m),
                static_cast<int>(active), 1.0, pseudoinverse_.data(),
                static_cast<int>(m), fixture_.d.data(), 1, 0.0,
                recurrence_gamma_.data(), 1);
  }
  recurrence_coefficients_valid_ = target_shift_.empty()
      && recurrence_g_.size() == active;
  direct_seeded_ = true;
  ++direct_seed_count_;
  valid_ = true;
  return true;
}

}  // namespace twalker::revised
