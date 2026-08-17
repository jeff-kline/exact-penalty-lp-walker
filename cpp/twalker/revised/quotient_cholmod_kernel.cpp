#include <cholmod.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Kernel {
  std::int64_t n = 0, m = 0;
  double epsilon = 0.0;
  cholmod_common common{};
  cholmod_sparse *matrix = nullptr;
  cholmod_factor *factor = nullptr;
  std::vector<double> base;
  std::vector<std::vector<std::pair<std::int64_t, double>>> row_entries;
  std::vector<std::uint8_t> support;
  std::vector<std::int64_t> inverse_permutation;
  std::uint64_t calls = 0, rebuilds = 0, updates = 0, downdates = 0;
  std::uint64_t failures = 0;
  double factor_ms = 0.0, solve_ms = 0.0;

  ~Kernel() {
    if (factor) cholmod_l_free_factor(&factor, &common);
    if (matrix) cholmod_l_free_sparse(&matrix, &common);
    cholmod_l_finish(&common);
  }
};

double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double inf_norm(const std::vector<double> &x) {
  double result = 0.0;
  for (double value : x) result = std::max(result, std::abs(value));
  return result;
}

bool rebuild(Kernel &kernel, const std::uint8_t *target) {
  if (kernel.factor)
    cholmod_l_free_factor(&kernel.factor, &kernel.common);
  kernel.factor = nullptr;
  std::size_t active_nnz = 0, active_rows = 0;
  for (std::int64_t row = 0; row < kernel.n; ++row) {
    if (!target[row] || kernel.row_entries[row].empty()) continue;
    active_nnz += kernel.row_entries[row].size();
    ++active_rows;
  }
  auto *columns = cholmod_l_allocate_sparse(
      kernel.m, active_rows + kernel.m, active_nnz + kernel.m,
      1, 1, 0, CHOLMOD_REAL, &kernel.common);
  if (!columns) return false;
  auto *cp = static_cast<std::int64_t *>(columns->p);
  auto *ci = static_cast<std::int64_t *>(columns->i);
  auto *cx = static_cast<double *>(columns->x);
  std::size_t cursor = 0, local = 0;
  for (std::int64_t row = 0; row < kernel.n; ++row) {
    if (!target[row] || kernel.row_entries[row].empty()) continue;
    cp[local++] = static_cast<std::int64_t>(cursor);
    for (const auto &[column, value] : kernel.row_entries[row]) {
      ci[cursor] = column;
      cx[cursor++] = value;
    }
  }
  const double root_epsilon = std::sqrt(kernel.epsilon);
  for (std::int64_t column = 0; column < kernel.m; ++column) {
    cp[local++] = static_cast<std::int64_t>(cursor);
    ci[cursor] = column;
    cx[cursor++] = root_epsilon;
  }
  cp[local] = static_cast<std::int64_t>(cursor);
  auto *gram = cholmod_l_aat(columns, nullptr, 0, 1, &kernel.common);
  cholmod_l_free_sparse(&columns, &kernel.common);
  if (!gram) return false;
  gram->stype = 1;
  kernel.factor = cholmod_l_analyze(gram, &kernel.common);
  const bool ok = kernel.factor
      && cholmod_l_factorize(gram, kernel.factor, &kernel.common)
      && kernel.factor->minor == kernel.m;
  cholmod_l_free_sparse(&gram, &kernel.common);
  if (!ok) return false;
  kernel.inverse_permutation.assign(kernel.m, 0);
  if (kernel.factor->Perm) {
    const auto *perm = static_cast<const std::int64_t *>(kernel.factor->Perm);
    for (std::int64_t position = 0; position < kernel.m; ++position)
      kernel.inverse_permutation[perm[position]] = position;
  } else {
    for (std::int64_t position = 0; position < kernel.m; ++position)
      kernel.inverse_permutation[position] = position;
  }
  kernel.support.assign(target, target + kernel.n);
  ++kernel.rebuilds;
  return true;
}

bool update_row(Kernel &kernel, std::int64_t target_row, bool add) {
  std::vector<std::pair<std::int64_t, double>> entries;
  entries.reserve(kernel.row_entries[target_row].size());
  for (const auto &[column, value] : kernel.row_entries[target_row])
    entries.emplace_back(kernel.inverse_permutation[column], value);
  if (entries.empty()) {
    kernel.support[target_row] = add;
    return true;
  }
  std::sort(entries.begin(), entries.end());
  auto *column = cholmod_l_allocate_sparse(
      kernel.m, 1, entries.size(), 1, 1, 0, CHOLMOD_REAL, &kernel.common);
  if (!column) return false;
  auto *cp = static_cast<std::int64_t *>(column->p);
  auto *ci = static_cast<std::int64_t *>(column->i);
  auto *cx = static_cast<double *>(column->x);
  cp[0] = 0;
  cp[1] = static_cast<std::int64_t>(entries.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    ci[i] = entries[i].first;
    cx[i] = entries[i].second;
  }
  const bool ok = cholmod_l_updown(add ? 1 : 0, column, kernel.factor,
                                   &kernel.common);
  cholmod_l_free_sparse(&column, &kernel.common);
  if (!ok || kernel.factor->minor != kernel.m) return false;
  kernel.support[target_row] = add;
  if (add) ++kernel.updates; else ++kernel.downdates;
  return true;
}

bool transition(Kernel &kernel, const std::uint8_t *target) {
  if (!kernel.factor) return rebuild(kernel, target);
  for (std::int64_t row = 0; row < kernel.n; ++row)
    if (!kernel.support[row] && target[row]
        && !update_row(kernel, row, true))
      return rebuild(kernel, target);
  for (std::int64_t row = 0; row < kernel.n; ++row)
    if (kernel.support[row] && !target[row]
        && !update_row(kernel, row, false))
      return rebuild(kernel, target);
  return true;
}

bool solve_e(Kernel &kernel, const std::vector<double> &rhs,
             std::vector<double> &answer) {
  auto *dense = cholmod_l_allocate_dense(kernel.m, 1, kernel.m, CHOLMOD_REAL,
                                         &kernel.common);
  if (!dense) return false;
  auto *x = static_cast<double *>(dense->x);
  std::copy(rhs.begin(), rhs.end(), x);
  auto *solution = cholmod_l_solve(CHOLMOD_A, kernel.factor, dense,
                                   &kernel.common);
  cholmod_l_free_dense(&dense, &kernel.common);
  if (!solution) return false;
  const auto *values = static_cast<const double *>(solution->x);
  answer.assign(values, values + kernel.m);
  cholmod_l_free_dense(&solution, &kernel.common);
  return std::all_of(answer.begin(), answer.end(),
                     [](double value) { return std::isfinite(value); });
}

void apply_matrix(Kernel &kernel, const std::vector<double> &x,
                  std::vector<double> &product) {
  product.assign(kernel.n, 0.0);
  const auto *starts = static_cast<const std::int64_t *>(kernel.matrix->p);
  const auto *rows = static_cast<const std::int64_t *>(kernel.matrix->i);
  for (std::int64_t column = 0; column < kernel.m; ++column)
    for (std::int64_t p = starts[column]; p < starts[column + 1]; ++p)
      if (kernel.support[rows[p]])
        product[rows[p]] += kernel.base[p] * x[column];
}

void apply_gram(Kernel &kernel, const std::vector<double> &x,
                std::vector<double> &normal) {
  std::vector<double> product;
  apply_matrix(kernel, x, product);
  normal.assign(kernel.m, 0.0);
  const auto *starts = static_cast<const std::int64_t *>(kernel.matrix->p);
  const auto *rows = static_cast<const std::int64_t *>(kernel.matrix->i);
  for (std::int64_t column = 0; column < kernel.m; ++column)
    for (std::int64_t p = starts[column]; p < starts[column + 1]; ++p)
      if (kernel.support[rows[p]])
        normal[column] += kernel.base[p] * product[rows[p]];
}

bool null_part(Kernel &kernel, const std::vector<double> &rhs,
               std::vector<double> &answer) {
  answer = rhs;
  for (int round = 0; round < 4; ++round) {
    std::vector<double> next;
    if (!solve_e(kernel, answer, next)) return false;
    for (double &value : next) value *= kernel.epsilon;
    answer.swap(next);
  }
  return true;
}

}  // namespace

extern "C" {

void *twq_chol_create(std::int64_t n, std::int64_t m, std::int64_t nnz,
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
  kernel->common.final_ll = 0;
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
  std::vector<long double> row_square(n, 0.0L);
  for (std::int64_t row = 0; row < n; ++row) {
    for (std::int64_t p = indptr[row]; p < indptr[row + 1]; ++p) {
      ti[cursor] = row;
      tj[cursor] = indices[p];
      tx[cursor] = values[p];
      row_square[row] += static_cast<long double>(values[p]) * values[p];
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
  kernel->row_entries.resize(n);
  const auto *rows = static_cast<const std::int64_t *>(kernel->matrix->i);
  for (std::int64_t column = 0; column < m; ++column)
    for (std::int64_t p = starts[column]; p < starts[column + 1]; ++p)
      kernel->row_entries[rows[p]].emplace_back(column, kernel->base[p]);
  long double scale = 1.0L;
  for (auto value : row_square) scale = std::max(scale, value);
  kernel->epsilon = 1e-8 * static_cast<double>(scale);
  kernel->support.assign(n, 0);
  return kernel;
}

int twq_chol_step(void *opaque, const std::uint8_t *support,
                  const double *gradient_raw, double *delta_raw,
                  double *null_raw, double *residual_out) {
  auto *kernel = static_cast<Kernel *>(opaque);
  if (!kernel || !support || !gradient_raw || !delta_raw || !null_raw)
    return 0;
  ++kernel->calls;
  const auto factor_mark = Clock::now();
  if (!transition(*kernel, support)) {
    ++kernel->failures;
    return 0;
  }
  kernel->factor_ms += milliseconds(factor_mark);
  const auto solve_mark = Clock::now();
  std::vector<double> gradient(gradient_raw, gradient_raw + kernel->m);
  std::vector<double> r_null, reachable(kernel->m), delta, removed;
  if (!null_part(*kernel, gradient, r_null)) {
    ++kernel->failures;
    return 0;
  }
  for (std::int64_t i = 0; i < kernel->m; ++i)
    reachable[i] = gradient[i] - r_null[i];
  if (!solve_e(*kernel, reachable, delta)
      || !null_part(*kernel, delta, removed)) {
    ++kernel->failures;
    return 0;
  }
  for (std::int64_t i = 0; i < kernel->m; ++i) delta[i] -= removed[i];
  for (int refinement = 0; refinement < 3; ++refinement) {
    std::vector<double> normal, residual(kernel->m), projected, correction;
    apply_gram(*kernel, delta, normal);
    for (std::int64_t i = 0; i < kernel->m; ++i)
      residual[i] = reachable[i] - normal[i];
    if (!null_part(*kernel, residual, projected)) {
      ++kernel->failures;
      return 0;
    }
    for (std::int64_t i = 0; i < kernel->m; ++i)
      residual[i] -= projected[i];
    if (!solve_e(*kernel, residual, correction)
        || !null_part(*kernel, correction, projected)) {
      ++kernel->failures;
      return 0;
    }
    for (std::int64_t i = 0; i < kernel->m; ++i)
      delta[i] += correction[i] - projected[i];
  }

  std::vector<double> normal, null_image;
  apply_gram(*kernel, delta, normal);
  apply_matrix(*kernel, r_null, null_image);
  double split_error = 0.0;
  for (std::int64_t i = 0; i < kernel->m; ++i)
    split_error = std::max(split_error,
        std::abs(normal[i] + r_null[i] - gradient[i]));
  const double gradient_scale = std::max(1.0, inf_norm(gradient));
  split_error /= gradient_scale;
  const double null_error = inf_norm(null_image) / gradient_scale;
  const double residual = std::max(split_error, null_error);
  std::copy(delta.begin(), delta.end(), delta_raw);
  std::copy(r_null.begin(), r_null.end(), null_raw);
  if (residual_out) *residual_out = residual;
  kernel->solve_ms += milliseconds(solve_mark);
  return std::isfinite(residual) ? 1 : 0;
}

void twq_chol_stats(void *opaque, std::uint64_t *rebuilds,
                    std::uint64_t *updates, std::uint64_t *downdates,
                    std::uint64_t *failures, double *factor_ms,
                    double *solve_ms) {
  auto *kernel = static_cast<Kernel *>(opaque);
  if (!kernel) return;
  if (rebuilds) *rebuilds = kernel->rebuilds;
  if (updates) *updates = kernel->updates;
  if (downdates) *downdates = kernel->downdates;
  if (failures) *failures = kernel->failures;
  if (factor_ms) *factor_ms = kernel->factor_ms;
  if (solve_ms) *solve_ms = kernel->solve_ms;
}

void twq_chol_destroy(void *opaque) {
  delete static_cast<Kernel *>(opaque);
}

}  // extern "C"
