#include <SuiteSparseQR.hpp>
#include <SuiteSparseQR_C.h>
#include <cholmod.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Kernel {
  std::int64_t n = 0, m = 0;
  cholmod_common common{};
  cholmod_sparse *matrix = nullptr;
  cholmod_sparse *transpose = nullptr;
  SuiteSparseQR_C_factorization *factor = nullptr;
  SuiteSparseQR_C_factorization *transpose_factor = nullptr;
  std::vector<double> base, transpose_base;
  std::uint64_t calls = 0, accepts = 0, failures = 0;
  double numeric_ms = 0.0, solve_ms = 0.0;

  ~Kernel() {
    if (factor) SuiteSparseQR_C_free(&factor, &common);
    if (transpose_factor)
      SuiteSparseQR_C_free(&transpose_factor, &common);
    if (matrix) cholmod_l_free_sparse(&matrix, &common);
    if (transpose) cholmod_l_free_sparse(&transpose, &common);
    cholmod_l_finish(&common);
  }
};

double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double inf_norm(const double *x, std::int64_t n) {
  double result = 0.0;
  for (std::int64_t i = 0; i < n; ++i)
    result = std::max(result, std::abs(x[i]));
  return result;
}

}  // namespace

extern "C" {

void *twq_spqr_create(std::int64_t n, std::int64_t m, std::int64_t nnz,
                      const std::int64_t *indptr,
                      const std::int64_t *indices,
                      const double *values) {
  if (n <= 0 || m <= 0 || nnz <= 0 || !indptr || !indices || !values)
    return nullptr;
  auto *kernel = new (std::nothrow) Kernel;
  if (!kernel) return nullptr;
  kernel->n = n;
  kernel->m = m;
  if (!cholmod_l_start(&kernel->common)) {
    delete kernel;
    return nullptr;
  }
  kernel->common.print = 0;
  auto *triplet = cholmod_l_allocate_triplet(
      n, m, nnz, 0, CHOLMOD_REAL, &kernel->common);
  if (!triplet) {
    delete kernel;
    return nullptr;
  }
  auto *ti = static_cast<std::int64_t *>(triplet->i);
  auto *tj = static_cast<std::int64_t *>(triplet->j);
  auto *tx = static_cast<double *>(triplet->x);
  std::int64_t cursor = 0;
  for (std::int64_t row = 0; row < n; ++row) {
    if (indptr[row] < 0 || indptr[row + 1] < indptr[row]
        || indptr[row + 1] > nnz) {
      cholmod_l_free_triplet(&triplet, &kernel->common);
      delete kernel;
      return nullptr;
    }
    for (std::int64_t p = indptr[row]; p < indptr[row + 1]; ++p) {
      if (indices[p] < 0 || indices[p] >= m || !std::isfinite(values[p])) {
        cholmod_l_free_triplet(&triplet, &kernel->common);
        delete kernel;
        return nullptr;
      }
      ti[cursor] = row;
      tj[cursor] = indices[p];
      tx[cursor] = values[p];
      ++cursor;
    }
  }
  triplet->nnz = cursor;
  kernel->matrix = cholmod_l_triplet_to_sparse(
      triplet, cursor, &kernel->common);
  cholmod_l_free_triplet(&triplet, &kernel->common);
  if (!kernel->matrix) {
    delete kernel;
    return nullptr;
  }
  const auto *starts = static_cast<const std::int64_t *>(kernel->matrix->p);
  const auto stored = starts[kernel->matrix->ncol];
  const auto *x = static_cast<const double *>(kernel->matrix->x);
  kernel->base.assign(x, x + stored);
  kernel->transpose = cholmod_l_transpose(kernel->matrix, 1, &kernel->common);
  if (!kernel->transpose) {
    delete kernel;
    return nullptr;
  }
  const auto *transpose_starts = static_cast<const std::int64_t *>(
      kernel->transpose->p);
  const auto transpose_stored = transpose_starts[kernel->transpose->ncol];
  const auto *transpose_x = static_cast<const double *>(kernel->transpose->x);
  kernel->transpose_base.assign(transpose_x,
                                transpose_x + transpose_stored);
  kernel->factor = SuiteSparseQR_C_symbolic(
      SPQR_ORDERING_AMD, 1, kernel->matrix, &kernel->common);
  kernel->transpose_factor = SuiteSparseQR_C_symbolic(
      SPQR_ORDERING_AMD, 1, kernel->transpose, &kernel->common);
  if (!kernel->factor || !kernel->transpose_factor) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

int twq_spqr_step(void *opaque, const std::uint8_t *support,
                  const double *gradient, double *delta,
                  double *null_residual, double *residual_out,
                  std::int64_t *rank_out) {
  auto *kernel = static_cast<Kernel *>(opaque);
  if (!kernel || !support || !gradient || !delta || !null_residual)
    return 0;
  ++kernel->calls;
  const auto n = kernel->n, m = kernel->m;
  auto *rows = static_cast<const std::int64_t *>(kernel->matrix->i);
  const auto *starts = static_cast<const std::int64_t *>(kernel->matrix->p);
  auto *x = static_cast<double *>(kernel->matrix->x);
  const auto stored = starts[kernel->matrix->ncol];
  for (std::int64_t p = 0; p < stored; ++p)
    x[p] = support[rows[p]] ? kernel->base[p] : 0.0;
  const auto *transpose_starts = static_cast<const std::int64_t *>(
      kernel->transpose->p);
  auto *transpose_x = static_cast<double *>(kernel->transpose->x);
  for (std::int64_t column = 0; column < n; ++column)
    for (std::int64_t p = transpose_starts[column];
         p < transpose_starts[column + 1]; ++p)
      transpose_x[p] = support[column] ? kernel->transpose_base[p] : 0.0;

  auto mark = Clock::now();
  if (!SuiteSparseQR_C_numeric(SPQR_DEFAULT_TOL, kernel->matrix,
                               kernel->factor, &kernel->common)
      || !SuiteSparseQR_C_numeric(SPQR_DEFAULT_TOL, kernel->transpose,
                                  kernel->transpose_factor,
                                  &kernel->common)) {
    ++kernel->failures;
    return 0;
  }
  kernel->numeric_ms += milliseconds(mark);
  auto *internal = static_cast<
      SuiteSparseQR_factorization<double, std::int64_t> *>(
          kernel->factor->factors);
  auto *transpose_internal = static_cast<
      SuiteSparseQR_factorization<double, std::int64_t> *>(
          kernel->transpose_factor->factors);
  const std::int64_t rank = internal ? internal->rank : 0;
  const std::int64_t transpose_rank = transpose_internal
                                          ? transpose_internal->rank : 0;
  if (rank_out) *rank_out = rank;
  if (rank <= 0 || transpose_rank != rank) {
    ++kernel->failures;
    return 0;
  }

  mark = Clock::now();
  auto *rhs = cholmod_l_allocate_dense(m, 1, m, CHOLMOD_REAL,
                                       &kernel->common);
  if (!rhs) {
    ++kernel->failures;
    return 0;
  }
  std::memcpy(rhs->x, gradient, static_cast<std::size_t>(m) * sizeof(double));
  auto *coordinates = SuiteSparseQR_C_qmult(
      SPQR_QTX, kernel->transpose_factor, rhs, &kernel->common);
  cholmod_l_free_dense(&rhs, &kernel->common);
  if (!coordinates || coordinates->nrow < static_cast<std::size_t>(m)) {
    if (coordinates) cholmod_l_free_dense(&coordinates, &kernel->common);
    ++kernel->failures;
    return 0;
  }
  auto *coordinate_values = static_cast<double *>(coordinates->x);
  for (std::int64_t row = rank; row < m; ++row)
    coordinate_values[row] = 0.0;
  auto *reachable = SuiteSparseQR_C_qmult(
      SPQR_QX, kernel->transpose_factor, coordinates, &kernel->common);
  cholmod_l_free_dense(&coordinates, &kernel->common);
  if (!reachable || reachable->nrow < static_cast<std::size_t>(m)) {
    if (reachable) cholmod_l_free_dense(&reachable, &kernel->common);
    ++kernel->failures;
    return 0;
  }
  auto *first = SuiteSparseQR_C_solve(
      SPQR_RTX_EQUALS_ETB, kernel->factor, reachable, &kernel->common);
  auto *answer = first ? SuiteSparseQR_C_solve(
      SPQR_RETX_EQUALS_B, kernel->factor, first, &kernel->common) : nullptr;
  if (first) cholmod_l_free_dense(&first, &kernel->common);
  if (!answer || answer->nrow < static_cast<std::size_t>(m)) {
    if (answer) cholmod_l_free_dense(&answer, &kernel->common);
    ++kernel->failures;
    return 0;
  }
  const auto *solution = static_cast<const double *>(answer->x);
  std::copy(solution, solution + m, delta);
  cholmod_l_free_dense(&answer, &kernel->common);
  const auto *reachable_values = static_cast<const double *>(reachable->x);
  for (std::int64_t column = 0; column < m; ++column)
    null_residual[column] = gradient[column] - reachable_values[column];
  cholmod_l_free_dense(&reachable, &kernel->common);

  std::vector<double> product(n, 0.0), normal(m, 0.0);
  for (std::int64_t column = 0; column < m; ++column)
    for (std::int64_t p = starts[column]; p < starts[column + 1]; ++p)
      if (support[rows[p]]) product[rows[p]] += kernel->base[p] * delta[column];
  for (std::int64_t column = 0; column < m; ++column)
    for (std::int64_t p = starts[column]; p < starts[column + 1]; ++p)
      if (support[rows[p]])
        normal[column] += kernel->base[p] * product[rows[p]];
  std::vector<double> null_image(n, 0.0);
  for (std::int64_t column = 0; column < m; ++column)
    for (std::int64_t p = starts[column]; p < starts[column + 1]; ++p)
      if (support[rows[p]])
        null_image[rows[p]] += kernel->base[p] * null_residual[column];
  const double gradient_scale = std::max(1.0, inf_norm(gradient, m));
  double matrix_scale = 0.0;
  for (double value : kernel->base) matrix_scale = std::max(matrix_scale,
                                                             std::abs(value));
  const double null_scale = std::max(1.0, matrix_scale * gradient_scale);
  double split_error = 0.0;
  for (std::int64_t column = 0; column < m; ++column)
    split_error = std::max(split_error, std::abs(
        normal[column] + null_residual[column] - gradient[column]));
  split_error /= gradient_scale;
  const double null_error = inf_norm(null_image.data(), n) / null_scale;
  const bool finite = std::all_of(delta, delta + m,
                                  [](double v) { return std::isfinite(v); })
                      && std::all_of(null_residual, null_residual + m,
                                     [](double v) { return std::isfinite(v); })
                      && std::isfinite(null_error);
  if (residual_out) *residual_out = std::max(null_error, split_error);
  kernel->solve_ms += milliseconds(mark);
  if (!finite) {
    ++kernel->failures;
    return 0;
  }
  ++kernel->accepts;
  return 1;
}

void twq_spqr_stats(void *opaque, std::uint64_t *calls,
                    std::uint64_t *accepts, std::uint64_t *failures,
                    double *numeric_ms, double *solve_ms) {
  auto *kernel = static_cast<Kernel *>(opaque);
  if (!kernel) return;
  if (calls) *calls = kernel->calls;
  if (accepts) *accepts = kernel->accepts;
  if (failures) *failures = kernel->failures;
  if (numeric_ms) *numeric_ms = kernel->numeric_ms;
  if (solve_ms) *solve_ms = kernel->solve_ms;
}

void twq_spqr_destroy(void *opaque) {
  delete static_cast<Kernel *>(opaque);
}

}  // extern "C"
