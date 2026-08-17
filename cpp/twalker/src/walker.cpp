#include "walker.hpp"
#include "dense_nnls.hpp"
#include "highs_c_api_minimal.hpp"
#include "kernel_projection.hpp"
#include "triangular_projection.hpp"
#ifdef TWALKER_ENABLE_REVISED_COLUMN
#include "augmented_kkt_basis.hpp"
#include "maintained_deficient_qr_solver.hpp"
#include "maintained_rowspace_solver.hpp"
#include "maintained_svd_face_solver.hpp"
#include "revised_column_solver.hpp"
#endif

#include <osqp.h>
#include <vecLib/cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {
void dposv_(const char *uplo, const int *n, const int *nrhs, double *a,
            const int *lda, double *b, const int *ldb, int *info);
void dgesdd_(const char *jobz, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt,
             const int *ldvt, double *work, const int *lwork, int *iwork,
             int *info);
void dgesv_(const int *n, const int *nrhs, double *a, const int *lda,
            int *ipiv, double *b, const int *ldb, int *info);
void dgetrs_(const char *trans, const int *n, const int *nrhs,
             const double *a, const int *lda, const int *ipiv, double *b,
             const int *ldb, int *info);
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda,
             int *info);
void dpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *a, const int *lda, double *b, const int *ldb,
             int *info);
}

namespace twalker {
namespace {

constexpr double kTolerance = 1e-7;
constexpr double kDualSupport = 1e-9;
constexpr double kForward = 1e-12;
constexpr double kTie = 1e-9;

bool anchored_basis_repair_enabled() {
  return std::getenv("TWALKER_DISABLE_ANCHORED_BASIS_REPAIR") == nullptr;
}

double inf_norm(const std::vector<double> &values) {
  double result = 0.0;
  for (double value : values) result = std::max(result, std::abs(value));
  return result;
}

double dot(const std::vector<double> &left, const std::vector<double> &right) {
  double result = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) result += left[i] * right[i];
  return result;
}

double stable_norm2(const std::vector<double> &values) {
  long double scale = 0.0L, sum = 1.0L;
  for (double value : values) {
    const long double magnitude = std::abs(static_cast<long double>(value));
    if (magnitude == 0.0L) continue;
    if (scale < magnitude) {
      const long double ratio = scale / magnitude;
      sum = 1.0L + sum * ratio * ratio;
      scale = magnitude;
    } else {
      const long double ratio = magnitude / scale;
      sum += ratio * ratio;
    }
  }
  return scale == 0.0L ? 0.0 : static_cast<double>(scale * std::sqrt(sum));
}

std::vector<std::uint32_t> support_rows(
    const std::vector<std::uint8_t> &support) {
  std::vector<std::uint32_t> rows;
  for (std::size_t i = 0; i < support.size(); ++i)
    if (support[i]) rows.push_back(static_cast<std::uint32_t>(i));
  return rows;
}

// Select a numerically independent square row basis with pivoted, twice-
// reorthogonalized modified Gram-Schmidt.  This is a cold construction used
// only by the quarantined endpoint crossover.  The admitted implementation
// will retain this basis and update it by row exchange across later pivots.
bool select_row_basis(const Fixture &fixture,
                      const std::vector<std::uint8_t> &eligible,
                      std::vector<std::uint32_t> &basis) {
  const int m = static_cast<int>(fixture.m);
  std::vector<std::uint32_t> rows = support_rows(eligible);
  if (m <= 0 || static_cast<int>(rows.size()) < m) return false;
  const int candidates = static_cast<int>(rows.size());
  std::vector<double> residual(static_cast<std::size_t>(candidates) * m, 0.0);
  std::vector<double> norm2(candidates, 0.0);
  double initial_max = 0.0;
  for (int local = 0; local < candidates; ++local) {
    const auto row = rows[local];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      const double value = fixture.values[p];
      residual[local + static_cast<std::size_t>(candidates)
                           * fixture.indices[p]] = value;
      norm2[local] += value * value;
    }
    initial_max = std::max(initial_max, norm2[local]);
  }
  if (!(initial_max > 0.0) || !std::isfinite(initial_max)) return false;
  std::vector<std::uint8_t> selected(candidates, 0);
  basis.clear();
  basis.reserve(m);
  for (int step = 0; step < m; ++step) {
    int pivot = -1;
    double pivot_norm2 = -1.0;
    for (int local = 0; local < candidates; ++local) {
      if (!selected[local] && norm2[local] > pivot_norm2) {
        pivot = local;
        pivot_norm2 = norm2[local];
      }
    }
    if (pivot < 0 || !(pivot_norm2 > initial_max * 1e-30)
        || !std::isfinite(pivot_norm2))
      return false;
    selected[pivot] = 1;
    basis.push_back(rows[pivot]);
    const double inverse_norm = 1.0 / std::sqrt(pivot_norm2);
    std::vector<double> q(m);
    for (int column = 0; column < m; ++column)
      q[column] = residual[pivot + static_cast<std::size_t>(candidates)
                                      * column] * inverse_norm;
    for (int pass = 0; pass < 2; ++pass) {
      for (int local = 0; local < candidates; ++local) {
        if (selected[local]) continue;
        double coefficient = 0.0;
        for (int column = 0; column < m; ++column)
          coefficient += q[column]
              * residual[local + static_cast<std::size_t>(candidates)
                                     * column];
        for (int column = 0; column < m; ++column)
          residual[local + static_cast<std::size_t>(candidates) * column]
              -= coefficient * q[column];
      }
    }
    for (int local = 0; local < candidates; ++local) {
      if (selected[local]) continue;
      double value = 0.0;
      for (int column = 0; column < m; ++column) {
        const double entry =
            residual[local + static_cast<std::size_t>(candidates) * column];
        value += entry * entry;
      }
      norm2[local] = value;
    }
  }
  return true;
}

bool solve_basis_system(const Fixture &fixture,
                        const std::vector<std::uint32_t> &basis,
                        const std::vector<double> &rhs,
                        std::vector<double> &answer) {
  const int m = static_cast<int>(fixture.m);
  if (static_cast<int>(basis.size()) != m
      || static_cast<int>(rhs.size()) != m)
    return false;
  std::vector<double> matrix(static_cast<std::size_t>(m) * m, 0.0);
  for (int equation = 0; equation < m; ++equation) {
    const auto row = basis[equation];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
      matrix[equation + static_cast<std::size_t>(m) * fixture.indices[p]] =
          fixture.values[p];
  }
  const auto original = matrix;
  answer = rhs;
  std::vector<int> pivots(m);
  const int nrhs = 1, lda = m, ldb = m;
  int info = 0;
  dgesv_(&m, &nrhs, matrix.data(), &lda, pivots.data(), answer.data(), &ldb,
         &info);
  if (info != 0) return false;
  // Two inexpensive refinement steps are important here: a crossover basis
  // is selected for independence, but may still be poorly scaled.
  const char trans = 'N';
  for (int refinement = 0; refinement < 2; ++refinement) {
    std::vector<double> correction(m);
    double residual_inf = 0.0, scale_inf = 1.0;
    for (int equation = 0; equation < m; ++equation) {
      double product = 0.0, magnitude = 0.0;
      for (int column = 0; column < m; ++column) {
        const double value = original[equation
                                      + static_cast<std::size_t>(m) * column];
        product += value * answer[column];
        magnitude += std::abs(value) * std::abs(answer[column]);
      }
      correction[equation] = rhs[equation] - product;
      residual_inf = std::max(residual_inf, std::abs(correction[equation]));
      scale_inf = std::max(scale_inf,
                           std::abs(rhs[equation]) + magnitude);
    }
    if (residual_inf <= 1e-13 * scale_inf) break;
    dgetrs_(&trans, &m, &nrhs, matrix.data(), &lda, pivots.data(),
            correction.data(), &ldb, &info);
    if (info != 0) return false;
    for (int column = 0; column < m; ++column)
      answer[column] += correction[column];
  }
  return std::all_of(answer.begin(), answer.end(),
                     [](double value) { return std::isfinite(value); });
}

bool correct_endpoint_equalities(const Fixture &fixture,
                                 std::vector<double> &y,
                                 const std::vector<std::uint8_t> &eligible,
                                 double &relative_correction) {
  const int n = static_cast<int>(fixture.n);
  const int m = static_cast<int>(fixture.m);
  std::vector<int> positive;
  for (int row = 0; row < n; ++row)
    if (y[row] > 0.0
        || (eligible.size() == fixture.n && eligible[row]))
      positive.push_back(row);
  const int pcount = static_cast<int>(positive.size());
  if (pcount == 0) return false;
  std::vector<double> residual(fixture.d.begin(), fixture.d.end());
  for (int local = 0; local < pcount; ++local) {
    const int row = positive[local];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
      residual[fixture.indices[p]] -= fixture.values[p] * y[row];
  }
  std::vector<double> matrix(static_cast<std::size_t>(m) * pcount, 0.0);
  for (int local = 0; local < pcount; ++local) {
    const int row = positive[local];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
      matrix[fixture.indices[p] + static_cast<std::size_t>(m) * local] =
          fixture.values[p];
  }
  const int thin = std::min(m, pcount);
  std::vector<double> singular(thin);
  std::vector<double> left(static_cast<std::size_t>(m) * m);
  std::vector<double> right(static_cast<std::size_t>(pcount) * pcount);
  std::vector<int> iwork(8 * std::max(1, thin));
  const char job = 'A';
  const int lda = m, ldu = m, ldvt = pcount;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &m, &pcount, matrix.data(), &lda, singular.data(),
          left.data(), &ldu, right.data(), &ldvt, &query, &lwork,
          iwork.data(), &info);
  if (info != 0 || !std::isfinite(query) || query < 1.0) return false;
  lwork = static_cast<int>(std::ceil(query));
  std::vector<double> work(lwork);
  dgesdd_(&job, &m, &pcount, matrix.data(), &lda, singular.data(),
          left.data(), &ldu, right.data(), &ldvt, work.data(), &lwork,
          iwork.data(), &info);
  if (info != 0 || singular.empty() || !(singular.front() > 0.0)) return false;
  // This is iterative refinement of an already accepted endpoint, not a rank
  // decision.  Retain near-null directions and let the explicit correction
  // magnitude/nonnegativity gates decide whether they are safe.
  const double cutoff = singular.front() * 1e-16;
  std::vector<double> correction(pcount, 0.0);
  for (int component = 0; component < thin; ++component) {
    if (!(singular[component] > cutoff)) continue;
    double coefficient = 0.0;
    for (int column = 0; column < m; ++column)
      coefficient += left[column + static_cast<std::size_t>(m) * component]
                     * residual[column];
    coefficient /= singular[component];
    for (int local = 0; local < pcount; ++local)
      correction[local] +=
          right[component + static_cast<std::size_t>(pcount) * local]
          * coefficient;
  }
  double correction_inf = 0.0, y_inf = 1.0;
  for (int local = 0; local < pcount; ++local) {
    correction_inf = std::max(correction_inf, std::abs(correction[local]));
    y_inf = std::max(y_inf, std::abs(y[positive[local]]));
    y[positive[local]] += correction[local];
    if (y[positive[local]] < -1e-12 * y_inf) return false;
    if (y[positive[local]] < 0.0) y[positive[local]] = 0.0;
  }
  relative_correction = correction_inf / y_inf;
  return std::isfinite(relative_correction) && relative_correction <= 1e-7;
}

void products(const Fixture &fixture, const std::vector<double> &vector,
              std::vector<double> &product, std::vector<double> *absolute) {
  product.assign(fixture.n, 0.0);
  if (absolute) absolute->assign(fixture.n, 0.0);
  for (std::size_t row = 0; row < fixture.n; ++row) {
    double value = 0.0, magnitude = 0.0;
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      value += fixture.values[p] * vector[fixture.indices[p]];
      if (absolute)
        magnitude += std::abs(fixture.values[p])
                     * std::abs(vector[fixture.indices[p]]);
    }
    product[row] = value;
    if (absolute) (*absolute)[row] = magnitude;
  }
}

void two_absolute_products(const Fixture &fixture,
                           const std::vector<double> &left,
                           const std::vector<double> &right,
                           std::vector<double> &abs_left,
                           std::vector<double> &abs_right) {
  abs_left.assign(fixture.n, 0.0);
  abs_right.assign(fixture.n, 0.0);
  for (std::size_t row = 0; row < fixture.n; ++row) {
    double left_sum = 0.0, right_sum = 0.0;
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      const double magnitude = std::abs(fixture.values[p]);
      const auto column = fixture.indices[p];
      left_sum += magnitude * std::abs(left[column]);
      right_sum += magnitude * std::abs(right[column]);
    }
    abs_left[row] = left_sum;
    abs_right[row] = right_sum;
  }
}

void affine_absolute_product(const Fixture &fixture,
                             const std::vector<double> &base,
                             const std::vector<double> &direction,
                             double delta, std::vector<double> &absolute) {
  absolute.assign(fixture.n, 0.0);
  for (std::size_t row = 0; row < fixture.n; ++row) {
    double sum = 0.0;
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      const auto column = fixture.indices[p];
      sum += std::abs(fixture.values[p])
             * std::abs(base[column] + delta * direction[column]);
    }
    absolute[row] = sum;
  }
}

bool svd_newton_step(const Fixture &fixture,
                     const std::vector<std::uint32_t> &rows,
                     const std::vector<double> &gradient,
                     std::vector<double> &delta,
                     std::vector<double> &null_residual) {
  const int active = static_cast<int>(rows.size());
  const int columns = static_cast<int>(fixture.m);
  delta.assign(columns, 0.0);
  null_residual = gradient;
  if (active == 0) return true;
  const int thin = std::min(active, columns);
  std::vector<double> matrix(static_cast<std::size_t>(active) * columns, 0.0);
  for (int local = 0; local < active; ++local) {
    const auto row = rows[local];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
      matrix[local + static_cast<std::size_t>(active) * fixture.indices[p]] =
          fixture.values[p];
  }
  std::vector<double> singular(thin);
  std::vector<double> left(static_cast<std::size_t>(active) * thin);
  std::vector<double> right(static_cast<std::size_t>(thin) * columns);
  std::vector<int> iwork(8 * std::max(1, thin));
  const char job = 'S';
  const int lda = active, ldu = active, ldvt = thin;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &active, &columns, matrix.data(), &lda, singular.data(),
          left.data(), &ldu, right.data(), &ldvt, &query, &lwork,
          iwork.data(), &info);
  if (info != 0 || !std::isfinite(query) || query < 1.0) return false;
  lwork = static_cast<int>(std::ceil(query));
  std::vector<double> work(lwork);
  dgesdd_(&job, &active, &columns, matrix.data(), &lda, singular.data(),
          left.data(), &ldu, right.data(), &ldvt, work.data(), &lwork,
          iwork.data(), &info);
  if (info != 0 || singular.empty() || !(singular.front() > 0.0))
    return false;
  const double cutoff = singular.front()
      * std::max(active, columns) * std::numeric_limits<double>::epsilon();
  for (int component = 0; component < thin; ++component) {
    if (!(singular[component] > cutoff)) continue;
    long double projected = 0.0L;
    for (int column = 0; column < columns; ++column)
      projected += static_cast<long double>(
                       right[component
                             + static_cast<std::size_t>(thin) * column])
                   * gradient[column];
    const double coefficient = static_cast<double>(projected);
    const double inverse_square = 1.0
        / (singular[component] * singular[component]);
    for (int column = 0; column < columns; ++column) {
      const double basis =
          right[component + static_cast<std::size_t>(thin) * column];
      delta[column] += basis * coefficient * inverse_square;
      null_residual[column] -= basis * coefficient;
    }
  }
  return std::all_of(
      delta.begin(), delta.end(), [](double value) { return std::isfinite(value); })
      && std::all_of(null_residual.begin(), null_residual.end(),
                     [](double value) { return std::isfinite(value); });
}

bool tikhonov_newton_step(const Fixture &fixture,
                          const std::vector<std::uint32_t> &rows,
                          const std::vector<double> &gradient,
                          std::vector<double> &delta,
                          std::vector<double> &null_residual) {
  const int columns = static_cast<int>(fixture.m);
  if (columns < 100 || rows.empty()) return false;
  int model_unit_rows = 0;
  for (std::size_t row = 0; row < fixture.n; ++row)
    model_unit_rows += fixture.indptr[row + 1] - fixture.indptr[row] == 1;
  if (10 * model_unit_rows < 7 * static_cast<int>(fixture.n)) return false;

  std::vector<double> diagonal(columns, 0.0);
  std::vector<std::uint32_t> core_rows;
  double scale = 1.0;
  for (auto row : rows) {
    const auto count = fixture.indptr[row + 1] - fixture.indptr[row];
    if (count == 1) {
      const auto p = fixture.indptr[row];
      const double entry = fixture.values[p];
      diagonal[fixture.indices[p]] += entry * entry;
      scale = std::max(scale, diagonal[fixture.indices[p]]);
    } else if (count > 1) {
      core_rows.push_back(row);
      long double square = 0.0L;
      for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
        square += static_cast<long double>(fixture.values[p])
                  * fixture.values[p];
      scale = std::max(scale, static_cast<double>(square));
    }
  }
  const int core = static_cast<int>(core_rows.size());
  if (core == 0 || core > 1200) return false;
  const double epsilon = 1e-12 * scale;
  if (!(epsilon > 0.0) || !std::isfinite(epsilon)) return false;
  std::vector<double> inverse(columns);
  for (int column = 0; column < columns; ++column)
    inverse[column] = 1.0 / (diagonal[column] + epsilon);

  // Weighted A, column-major core-by-m.  Its lower symmetric product is
  // A diag(1/(D+eps)) A' and the identity completes Woodbury's system.
  std::vector<double> weighted(static_cast<std::size_t>(core) * columns, 0.0);
  for (int local = 0; local < core; ++local) {
    const auto row = core_rows[local];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
      weighted[local + static_cast<std::size_t>(core) * fixture.indices[p]] =
          fixture.values[p] * std::sqrt(inverse[fixture.indices[p]]);
  }
  std::vector<double> factor(static_cast<std::size_t>(core) * core, 0.0);
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, core, columns, 1.0,
              weighted.data(), core, 0.0, factor.data(), core);
  for (int i = 0; i < core; ++i)
    factor[i + static_cast<std::size_t>(core) * i] += 1.0;
  const char lower = 'L';
  int info = 0;
  dpotrf_(&lower, &core, factor.data(), &core, &info);
  if (info != 0) return false;

  auto solve_e = [&](const std::vector<double> &rhs,
                     std::vector<double> &answer) {
    answer.resize(columns);
    for (int column = 0; column < columns; ++column)
      answer[column] = inverse[column] * rhs[column];
    std::vector<double> small(core, 0.0);
    for (int local = 0; local < core; ++local) {
      const auto row = core_rows[local];
      for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
        small[local] += fixture.values[p] * answer[fixture.indices[p]];
    }
    const int one = 1;
    int solve_info = 0;
    dpotrs_(&lower, &core, &one, factor.data(), &core, small.data(), &core,
            &solve_info);
    if (solve_info != 0) return false;
    for (int local = 0; local < core; ++local) {
      const auto row = core_rows[local];
      for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
        const auto column = fixture.indices[p];
        answer[column] -= inverse[column] * fixture.values[p] * small[local];
      }
    }
    return std::all_of(answer.begin(), answer.end(),
                       [](double value) { return std::isfinite(value); });
  };
  auto null_part = [&](const std::vector<double> &rhs,
                       std::vector<double> &answer) {
    answer = rhs;
    for (int round = 0; round < 2; ++round) {
      std::vector<double> next;
      if (!solve_e(answer, next)) return false;
      for (double &value : next) value *= epsilon;
      answer.swap(next);
    }
    return true;
  };
  auto apply_h = [&](const std::vector<double> &x,
                     std::vector<double> &answer) {
    answer.resize(columns);
    for (int column = 0; column < columns; ++column)
      answer[column] = diagonal[column] * x[column];
    for (auto row : core_rows) {
      long double product = 0.0L;
      for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
        product += static_cast<long double>(fixture.values[p])
                   * x[fixture.indices[p]];
      for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
        answer[fixture.indices[p]] +=
            fixture.values[p] * static_cast<double>(product);
    }
  };

  if (!null_part(gradient, null_residual)) return false;
  std::vector<double> reachable(columns);
  for (int column = 0; column < columns; ++column)
    reachable[column] = gradient[column] - null_residual[column];
  if (!solve_e(reachable, delta)) return false;
  std::vector<double> remove;
  if (!null_part(delta, remove)) return false;
  for (int column = 0; column < columns; ++column)
    delta[column] -= remove[column];
  for (int refinement = 0; refinement < 3; ++refinement) {
    std::vector<double> applied, residual(columns), projected, correction;
    apply_h(delta, applied);
    for (int column = 0; column < columns; ++column)
      residual[column] = reachable[column] - applied[column];
    if (!null_part(residual, projected)) return false;
    for (int column = 0; column < columns; ++column)
      residual[column] -= projected[column];
    if (!solve_e(residual, correction)) return false;
    if (!null_part(correction, projected)) return false;
    for (int column = 0; column < columns; ++column)
      delta[column] += correction[column] - projected[column];
  }
  return std::all_of(
      delta.begin(), delta.end(), [](double value) { return std::isfinite(value); })
      && std::all_of(null_residual.begin(), null_residual.end(),
                     [](double value) { return std::isfinite(value); });
}

double exact_projection_line_search(const std::vector<double> &v,
                                    const std::vector<double> &p,
                                    double linear) {
  struct Event {
    double step;
    std::size_t row;
  };
  long double active_linear = 0.0L, active_square = 0.0L;
  std::vector<Event> events;
  events.reserve(v.size());
  for (std::size_t row = 0; row < v.size(); ++row) {
    if (v[row] > 0.0) {
      active_linear += static_cast<long double>(p[row]) * v[row];
      active_square += static_cast<long double>(p[row]) * p[row];
    }
    if (p[row] == 0.0) continue;
    const double step = -v[row] / p[row];
    if (std::isfinite(step) && step >= 0.0
        && (p[row] > 0.0 || v[row] > 0.0))
      events.push_back({step, row});
  }
  std::stable_sort(events.begin(), events.end(),
                   [](const Event &left, const Event &right) {
                     return left.step < right.step;
                   });
  double current = 0.0;
  for (const auto &event : events) {
    if (event.step > current) {
      if (active_square > 0.0L) {
        const double root = static_cast<double>(
            (static_cast<long double>(linear) - active_linear)
            / active_square);
        if (root <= current) return current;
        if (root < event.step) return root;
      } else if (static_cast<long double>(linear) - active_linear <= 0.0L) {
        return current;
      }
      current = event.step;
    }
    const auto row = event.row;
    if (p[row] > 0.0) {
      active_linear += static_cast<long double>(p[row]) * v[row];
      active_square += static_cast<long double>(p[row]) * p[row];
    } else {
      active_linear -= static_cast<long double>(p[row]) * v[row];
      active_square -= static_cast<long double>(p[row]) * p[row];
      if (active_square < 0.0L
          && std::abs(active_square) <= 1e-18L
                 * std::max(1.0L, std::abs(active_linear)))
        active_square = 0.0L;
    }
  }
  if (active_square > 0.0L) {
    const double root = static_cast<double>(
        (static_cast<long double>(linear) - active_linear) / active_square);
    return std::max(root, current);
  }
  if (static_cast<long double>(linear) - active_linear <= 0.0L)
    return current;
  return std::numeric_limits<double>::infinity();
}

std::vector<double> build_target_shift(const Fixture &fixture,
                                       double epsilon) {
  std::vector<double> shift(fixture.n, 0.0);
  if (epsilon == 0.0) return shift;
  if (!std::isfinite(epsilon) || epsilon < 0.0)
    throw std::invalid_argument("target nudge must be finite and nonnegative");
  double b_scale = 0.0, maximum_row_norm = 0.0;
  std::vector<double> row_norm(fixture.n, 0.0);
  for (std::size_t row = 0; row < fixture.n; ++row) {
    b_scale = std::max(b_scale, std::abs(fixture.b[row]));
    double square = 0.0;
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
      square += fixture.values[p] * fixture.values[p];
    row_norm[row] = std::sqrt(square);
    maximum_row_norm = std::max(maximum_row_norm, row_norm[row]);
  }
  b_scale = std::max(b_scale, 1e-12);
  maximum_row_norm = std::max(maximum_row_norm, 1e-300);
  for (std::size_t row = 0; row < fixture.n; ++row)
    if (row_norm[row] > 0.0)
      shift[row] = epsilon * b_scale * row_norm[row] / maximum_row_norm;
  return shift;
}

}  // namespace

Walker::Walker(const Fixture &fixture, bool enable_face_cache,
               double gram_min_rcond, double target_nudge_epsilon,
               bool allow_revised_column, bool enable_native_seed)
    : fixture_(fixture),
      enable_native_seed_(enable_native_seed),
      target_shift_(build_target_shift(fixture, target_nudge_epsilon)),
      bound_core_solver_(fixture, target_shift_),
      gram_solver_(fixture, gram_min_rcond, target_shift_),
      solver_(fixture, enable_face_cache, target_shift_, -1,
              std::getenv("TWALKER_DIRECT_CANDIDATE") != nullptr),
      extension_enabled_(gram_min_rcond < 5e-4),
      trace_path_(std::getenv("TWALKER_TRACE_PATH") != nullptr) {
  rank_lift_live_requested_ =
      std::getenv("TWALKER_RANK_LIFT_LIVE") != nullptr;
  rank_lift_audit_enabled_ = rank_lift_live_requested_
                             || std::getenv("TWALKER_RANK_LIFT_AUDIT");
  if (const char *raw = std::getenv("TWALKER_RANK_LIFT_AUDIT_MAX_FACES")) {
    const long requested = std::strtol(raw, nullptr, 10);
    if (requested > 0 && requested <= 100000)
      rank_lift_audit_max_faces_ = static_cast<int>(requested);
  }
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if (allow_revised_column
      && !std::getenv("TWALKER_DISABLE_REVISED_COLUMN"))
    revised_column_solver_ =
        new revised::RevisedColumnSolver(fixture_, target_shift_);
  // This state is deliberately independent of the affine revised lane.  It
  // remains available on deeply deficient faces and never stores h=y-t*g.
  centered_slope_solver_ =
      new revised::RevisedColumnSolver(fixture_, target_shift_, true);
  augmented_kkt_basis_ = new revised::AugmentedKktBasis(fixture_);
  maintained_rowspace_audit_enabled_ =
      std::getenv("TWALKER_MAINTAINED_ROWSPACE_AUDIT") != nullptr;
  deficient_face_audit_enabled_ =
      std::getenv("TWALKER_DEFICIENT_FACE_AUDIT") != nullptr;
  deficient_face_live_enabled_ =
      std::getenv("TWALKER_DEFICIENT_FACE_LIVE") != nullptr;
  maintained_svd_audit_enabled_ =
      std::getenv("TWALKER_MAINTAINED_SVD_AUDIT") != nullptr;
  maintained_svd_live_enabled_ =
      std::getenv("TWALKER_MAINTAINED_SVD_LIVE") != nullptr;
  const bool quotient_basis_live =
      std::getenv("TWALKER_QUOTIENT_BASIS_LIVE") != nullptr;
  maintained_rowspace_trace_ =
      std::getenv("TWALKER_MAINTAINED_ROWSPACE_TRACE") != nullptr;
  if (maintained_rowspace_audit_enabled_)
    maintained_rowspace_solver_ =
        new revised::MaintainedRowspaceSolver(fixture_);
  if (maintained_rowspace_audit_enabled_ || quotient_basis_live
      || deficient_face_audit_enabled_ || deficient_face_live_enabled_)
    maintained_deficient_qr_solver_ =
        new revised::MaintainedDeficientQrSolver(fixture_, target_shift_);
  if (deficient_face_audit_enabled_ || deficient_face_live_enabled_)
    deficient_recurrence_solver_ =
        new revised::RevisedColumnSolver(fixture_, target_shift_, false, true);
  if (maintained_svd_audit_enabled_ || maintained_svd_live_enabled_)
    maintained_svd_face_solver_ =
        new revised::MaintainedSvdFaceSolver(fixture_, target_shift_);
  if (maintained_rowspace_audit_enabled_ || quotient_basis_live
      || deficient_face_audit_enabled_ || deficient_face_live_enabled_)
    maintained_rowspace_seed_solver_ =
        new FaceSolver(fixture_, false, target_shift_);
  revised_column_primary_ = revised_column_solver_
                            && std::getenv("TWALKER_REVISED_PRIMARY");
#endif
#ifndef TWALKER_ENABLE_REVISED_COLUMN
  (void)allow_revised_column;
#endif
}

Walker::~Walker() {
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  delete static_cast<revised::RevisedColumnSolver *>(revised_column_solver_);
  delete static_cast<revised::RevisedColumnSolver *>(centered_slope_solver_);
  delete static_cast<revised::AugmentedKktBasis *>(augmented_kkt_basis_);
  delete static_cast<revised::MaintainedRowspaceSolver *>(
      maintained_rowspace_solver_);
  delete static_cast<revised::MaintainedDeficientQrSolver *>(
      maintained_deficient_qr_solver_);
  delete static_cast<revised::RevisedColumnSolver *>(
      deficient_recurrence_solver_);
  delete static_cast<revised::MaintainedSvdFaceSolver *>(
      maintained_svd_face_solver_);
  delete maintained_rowspace_seed_solver_;
#endif
  if (selector_) Highs_destroy(selector_);
  if (horizon_selector_) Highs_destroy(horizon_selector_);
  if (primal_selector_) Highs_destroy(primal_selector_);
  if (terminal_support_selector_) Highs_destroy(terminal_support_selector_);
  if (rank_lift_highs_) Highs_destroy(rank_lift_highs_);
  delete rank_lift_solver_;
  delete rank_lift_live_solver_;
  if (osqp_solver_) osqp_cleanup(static_cast<OSQPSolver *>(osqp_solver_));
}

bool Walker::native_newton_seed(std::vector<std::uint8_t> &support) {
  return native_newton_seed(
      support, std::vector<double>(fixture_.m, 0.0), "newton");
}

bool Walker::native_newton_seed(
    std::vector<std::uint8_t> &support,
    const std::vector<double> &initial_multiplier,
    const std::string &route) {
  const auto started = std::chrono::steady_clock::now();
  native_seed_converged_ = false;
  native_seed_iterations_ = 0;
  native_seed_support_ = 0;
  native_seed_dres_ = std::numeric_limits<double>::infinity();
  native_seed_route_ = route;
  const int columns = static_cast<int>(fixture_.m);
  std::vector<double> multiplier = initial_multiplier;
  if (multiplier.size() != fixture_.m)
    multiplier.assign(columns, 0.0);
  std::vector<double> value(fixture_.n), positive(fixture_.n), gradient(columns);
  std::vector<double> history;
  history.reserve(301);
  bool penalty_used = false;
  int total_iterations = 0;
  const double dscale = std::max(1.0, inf_norm(fixture_.d));

  auto state = [&]() {
    products(fixture_, multiplier, value, nullptr);
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      value[row] += target_shift_[row];
      positive[row] = std::max(0.0, value[row]);
    }
    gradient = fixture_.d;
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      if (positive[row] == 0.0) continue;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        gradient[fixture_.indices[p]] -=
            fixture_.values[p] * positive[row];
    }
  };

  auto step_residual = [&](const std::vector<std::uint32_t> &rows,
                           const std::vector<double> &delta,
                           const std::vector<double> &null_residual) {
    std::vector<double> product;
    products(fixture_, delta, product, nullptr);
    std::vector<double> residual = null_residual;
    for (auto row : rows) {
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        residual[fixture_.indices[p]] +=
            fixture_.values[p] * product[row];
    }
    for (int column = 0; column < columns; ++column)
      residual[column] -= gradient[column];
    return inf_norm(residual) / std::max(inf_norm(gradient), 1e-300);
  };

  history.clear();
  state();
  for (int iteration = 0; iteration < 300; ++iteration) {
    native_seed_iterations_ = ++total_iterations;
    const double gnorm = inf_norm(gradient);
    if (gnorm <= 1e-9 * dscale) {
      native_seed_converged_ = true;
      break;
    }
    long double theta = dot(fixture_.d, multiplier);
    for (double entry : positive)
      theta -= 0.5L * static_cast<long double>(entry) * entry;
    history.push_back(static_cast<double>(theta));
    if (history.size() > 25) {
      const double past = history[history.size() - 26];
      if (history.back() - past
          <= 1e-12 * std::max(1.0, std::abs(history.back())))
        break;
    }

    support.assign(fixture_.n, 0);
    for (std::size_t row = 0; row < fixture_.n; ++row)
      support[row] = value[row] > 0.0;
    const auto rows = support_rows(support);
    std::vector<double> delta, null_residual(columns, 0.0);
    bool fast = false;
    if (!rows.empty()) {
      FaceSolution candidate;
      if (bound_core_solver_.solve(rows, candidate)
          || gram_solver_.solve(rows, candidate)) {
        delta.resize(columns);
        for (int column = 0; column < columns; ++column)
          delta[column] = candidate.uc[column] - multiplier[column];
        fast = step_residual(rows, delta, null_residual) <= 1e-6;
      }
    }
    if (!fast) {
      fast = tikhonov_newton_step(fixture_, rows, gradient, delta,
                                  null_residual)
             && step_residual(rows, delta, null_residual) <= 1e-6;
    }
    if (!fast && !svd_newton_step(fixture_, rows, gradient, delta,
                                  null_residual))
      break;

    bool moved = false;
    for (int kind = 0; kind < 2; ++kind) {
      const auto &direction = kind == 0 ? delta : null_residual;
      if (direction.empty()) continue;
      const double ascent = dot(gradient, direction);
      if (!(ascent > 0.0) || !std::isfinite(ascent)) continue;
      std::vector<double> ray;
      products(fixture_, direction, ray, nullptr);
      const double alpha = exact_projection_line_search(
          value, ray, dot(fixture_.d, direction));
      if (!(alpha > 0.0) || !std::isfinite(alpha)) continue;
      for (int column = 0; column < columns; ++column)
        multiplier[column] += alpha * direction[column];
      moved = true;
      if (kind == 0) state();
    }
    if (!moved) break;
    state();
  }

  state();
  native_seed_dres_ = inf_norm(gradient) / dscale;
  native_seed_converged_ = native_seed_dres_ <= 1e-9;
  if (!native_seed_converged_) {
    // Rare fail-closed Phase-I crash.  This is the same maintained-QR
    // Lawson--Hanson penalty projection already validated in cpp/nnls, now
    // in-process.  It supplies only a support hint; the exact fixed-t settle
    // below remains authoritative.  Lotfi is the current panel trigger.
    const int n = static_cast<int>(fixture_.n);
    const int augmented_rows = n + columns;
    double matrix_scale = 1.0;
    for (double entry : fixture_.values)
      matrix_scale = std::max(matrix_scale, std::abs(entry));
    const double weights[] = {1e6 / matrix_scale, 1e5 / matrix_scale,
                              1e4 / matrix_scale, 1e3 / matrix_scale};
    for (double weight : weights) {
      std::vector<double> matrix(
          static_cast<std::size_t>(augmented_rows) * n, 0.0);
      std::vector<double> rhs(augmented_rows, 0.0);
      for (int row = 0; row < n; ++row) {
        matrix[row + static_cast<std::size_t>(augmented_rows) * row] = 1.0;
        rhs[row] = target_shift_[row];
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
          matrix[n + fixture_.indices[p]
                 + static_cast<std::size_t>(augmented_rows) * row] =
              weight * fixture_.values[p];
      }
      for (int column = 0; column < columns; ++column)
        rhs[n + column] = weight * fixture_.d[column];
      auto candidate = dense_nnls(matrix, augmented_rows, n, rhs, 30 * n);
      if (!candidate.converged) continue;
      double ymax = 0.0;
      for (double entry : candidate.x) ymax = std::max(ymax, entry);
      const double threshold = 1e-9 * std::max(1.0, ymax);
      support.assign(fixture_.n, 0);
      std::vector<double> transpose(columns, 0.0);
      for (int row = 0; row < n; ++row) {
        support[row] = candidate.x[row] > threshold;
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
          transpose[fixture_.indices[p]] +=
              fixture_.values[p] * candidate.x[row];
      }
      for (int column = 0; column < columns; ++column)
        transpose[column] -= fixture_.d[column];
      native_seed_dres_ = inf_norm(transpose) / dscale;
      native_seed_converged_ = true;
      native_seed_route_ = route + "+nnls-fallback";
      penalty_used = true;
      break;
    }
  }
  if (!penalty_used) {
    support.assign(fixture_.n, 0);
    for (std::size_t row = 0; row < fixture_.n; ++row)
      support[row] = value[row] > 0.0;
  }
  if (!native_seed_converged_) native_seed_route_ = route + "-failed";
  native_seed_support_ = static_cast<int>(
      std::count(support.begin(), support.end(), std::uint8_t{1}));
  native_seed_ms_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  return native_seed_converged_;
}

bool Walker::kernel_projection_seed(std::vector<std::uint8_t> &support) {
  const auto started = std::chrono::steady_clock::now();
  native_seed_converged_ = false;
  native_seed_iterations_ = 0;
  native_seed_support_ = 0;
  native_seed_dres_ = std::numeric_limits<double>::infinity();
  native_seed_route_ = "kernel-qr";
  // The current kernel projector represents the unshifted projection.  Fail
  // closed if an experimental target nudge is present; Newton remains the
  // general shifted lane.
  for (double value : target_shift_) {
    if (value != 0.0) {
      native_seed_route_ = "kernel-qr-shift-unsupported";
      native_seed_ms_ = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started)
                            .count();
      return false;
    }
  }
  revised::KernelProjector projector(fixture_);
  auto candidate = projector.solve(0.0, -1.0, true);
  native_seed_iterations_ = static_cast<int>(candidate.stats.events);
  native_seed_ms_ = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started)
                        .count();
  native_seed_dres_ = candidate.equality_error;
  if (!candidate.certified || candidate.y.size() != fixture_.n) {
    native_seed_route_ = "kernel-qr-failed:" + candidate.status;
    return false;
  }
  double scale = 1.0;
  for (double value : candidate.y) scale = std::max(scale, std::abs(value));
  const double threshold = 1e-9 * scale;
  support.assign(fixture_.n, 0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    support[row] = candidate.y[row] > threshold;
  native_seed_support_ = static_cast<int>(
      std::count(support.begin(), support.end(), std::uint8_t{1}));
  native_seed_converged_ = true;
  return true;
}

bool Walker::triangular_projection_seed(std::vector<std::uint8_t> &support) {
  const auto started = std::chrono::steady_clock::now();
  native_seed_converged_ = false;
  native_seed_iterations_ = 0;
  native_seed_support_ = 0;
  native_seed_dres_ = std::numeric_limits<double>::infinity();
  native_seed_route_ = "triangular-qr";
  for (double value : target_shift_) {
    if (value != 0.0) {
      native_seed_route_ = "triangular-qr-shift-unsupported";
      native_seed_ms_ = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started)
                            .count();
      return false;
    }
  }
  revised::TriangularProjector projector(fixture_);
  auto candidate = projector.solve(0.0, -1.0, 10000);
  native_seed_iterations_ = static_cast<int>(candidate.stats.events);
  native_seed_ms_ = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started)
                        .count();
  native_seed_dres_ = candidate.equality_error;
  if (!candidate.certified || candidate.y.size() != fixture_.n) {
    native_seed_route_ = "triangular-qr-failed:" + candidate.status;
    return false;
  }
  double scale = 1.0;
  for (double value : candidate.y) scale = std::max(scale, std::abs(value));
  const double threshold = 1e-9 * scale;
  support.assign(fixture_.n, 0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    support[row] = candidate.y[row] > threshold;
  native_seed_support_ = static_cast<int>(
      std::count(support.begin(), support.end(), std::uint8_t{1}));
  native_seed_converged_ = true;
  return true;
}

bool Walker::highs_projection_seed(std::vector<std::uint8_t> &support) {
  const auto started = std::chrono::steady_clock::now();
  native_seed_converged_ = false;
  native_seed_iterations_ = 0;
  native_seed_support_ = 0;
  native_seed_dres_ = std::numeric_limits<double>::infinity();
  native_seed_route_ = "highs-qp";

  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  void *highs = Highs_create();
  if (!highs) {
    native_seed_route_ = "highs-qp-create-failed";
    native_seed_ms_ = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - started)
                          .count();
    return false;
  }

  Highs_setBoolOptionValue(
      highs, "output_flag",
      std::getenv("TWALKER_HIGHS_SEED_TRACE") != nullptr);
  Highs_setIntOptionValue(highs, "threads", 1);
  Highs_setStringOptionValue(highs, "presolve", "on");
  Highs_setStringOptionValue(highs, "solver", "choose");

  // Solve the actual fixed-t=0 projection, not merely dual feasibility:
  //
  //   min 0.5*||y||^2 - shift'y  s.t. B'y=d, y>=0.
  //
  // The fixture CSR arrays are already a columnwise representation of B'
  // because each original row is one dual variable.  The fixed-t settle and
  // original-operator gate below remain authoritative for the returned
  // support, so the HiGHS status alone never admits the seed.
  std::vector<int> starts(n + 1), indices(fixture_.nnz);
  for (int row = 0; row <= n; ++row)
    starts[row] = static_cast<int>(fixture_.indptr[row]);
  for (std::size_t p = 0; p < fixture_.nnz; ++p)
    indices[p] = static_cast<int>(fixture_.indices[p]);
  std::vector<double> cost(n), lower(n, 0.0), upper(n, 1e30);
  for (int row = 0; row < n; ++row) cost[row] = -target_shift_[row];
  std::vector<int> q_start(n + 1), q_index(n);
  std::vector<double> q_value(n, 1.0);
  for (int row = 0; row < n; ++row) {
    q_start[row] = row;
    q_index[row] = row;
  }
  q_start[n] = n;
  const int pass_status = Highs_passModel(
      highs, n, m, static_cast<int>(fixture_.nnz), n, 1, 1, 1, 0.0,
      cost.data(), lower.data(), upper.data(), fixture_.d.data(),
      fixture_.d.data(), starts.data(), indices.data(), fixture_.values.data(),
      q_start.data(), q_index.data(), q_value.data(), nullptr);
  const int run_status = pass_status == 0 ? Highs_run(highs) : -1;
  if (Highs_getIntInfoValue(highs, "qp_iteration_count",
                            &native_seed_iterations_) < 0)
    Highs_getIntInfoValue(highs, "simplex_iteration_count",
                          &native_seed_iterations_);

  std::vector<double> y(n, 0.0);
  const int model_status = Highs_getModelStatus(highs);
  bool candidate =
      pass_status == 0
      && Highs_getSolution(highs, y.data(), nullptr, nullptr, nullptr) >= 0
      && std::all_of(y.begin(), y.end(), [](double value) {
           return std::isfinite(value);
         });
  std::vector<int> basic_variables(m, -1);
  const bool has_basis =
      candidate
      && Highs_getBasicVariables(highs, basic_variables.data()) >= 0;

  if (candidate) {
    std::vector<long double> transpose(m, 0.0L);
    std::vector<long double> transpose_abs(m, 0.0L);
    double y_max = 0.0;
    double nonnegative_error = 0.0;
    for (int row = 0; row < n; ++row) {
      y_max = std::max(y_max, std::abs(y[row]));
      nonnegative_error = std::max(nonnegative_error, std::max(0.0, -y[row]));
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto column = fixture_.indices[p];
        const long double term =
            static_cast<long double>(fixture_.values[p]) * y[row];
        transpose[column] += term;
        transpose_abs[column] += std::abs(term);
      }
    }
    double equality_error = 0.0;
    for (int column = 0; column < m; ++column) {
      const long double scale =
          1.0L + std::abs(static_cast<long double>(fixture_.d[column]))
          + transpose_abs[column];
      equality_error = std::max(
          equality_error,
          static_cast<double>(
              std::abs(transpose[column] - fixture_.d[column]) / scale));
    }
    native_seed_dres_ =
        std::max(equality_error, nonnegative_error / (1.0 + y_max));
    candidate = std::isfinite(native_seed_dres_);

    const double threshold = kDualSupport * std::max(1.0, y_max);
    support.assign(fixture_.n, 0);
    for (int row = 0; row < n; ++row)
      support[row] = y[row] > threshold;
    // Preserve zero-valued structural basic variables.  They are the
    // simplex statuses needed to avoid throwing away rank at the seed.
    if (has_basis)
      for (int variable : basic_variables)
        if (variable >= 0 && variable < n) support[variable] = 1;
    native_seed_support_ = static_cast<int>(
        std::count(support.begin(), support.end(), std::uint8_t{1}));
    native_seed_converged_ =
        run_status >= 0 && model_status == 7
        && native_seed_dres_ <= kTolerance;
    if (!native_seed_converged_)
      native_seed_route_ = "highs-qp-candidate";
  }

  if (!candidate) native_seed_route_ = "highs-qp-failed";
  Highs_destroy(highs);
  native_seed_ms_ = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started)
                        .count();
  // A finite HiGHS point is allowed to nominate a working face even when its
  // own QP status or residual misses our gate.  The exact face solve and
  // fixed-t settle in run() are the authority; an unrepairable nomination
  // still fails closed before the walk begins.
  return candidate;
}

FaceSolution Walker::solve(const std::vector<std::uint8_t> &support) {
  ++face_solves_;
  const auto rows = support_rows(support);
  FaceSolution solution;
  if (rank_lift_live_solver_) {
    solution = rank_lift_live_solver_->solve(rows);
    dense_fallbacks_ += solution.used_dense_fallback;
    record_face_rank(solution);
    if (maintained_rowspace_audit_enabled_)
      audit_maintained_rowspace(support, solution);
    return solution;
  }
  if (std::getenv("TWALKER_BOUND_CORE_LIVE")
      && !std::getenv("TWALKER_BOUND_CORE_WIDE_SHADOW")) {
    if (bound_core_solver_.solve(rows, solution)) {
      ++bound_core_solves_;
      record_face_rank(solution);
      if (maintained_rowspace_audit_enabled_)
        audit_maintained_rowspace(support, solution);
      return solution;
    }
    ++bound_core_declines_;
  }
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  auto try_revised = [&]() {
    if (!revised_column_solver_) return false;
    revised::RevisedFaceSolution revised_solution;
    if (!static_cast<revised::RevisedColumnSolver *>(revised_column_solver_)
             ->solve(rows, revised_solution)) {
      ++revised_column_declines_;
      return false;
    }
    solution.g = std::move(revised_solution.g);
    solution.h = std::move(revised_solution.h);
    solution.ua = std::move(revised_solution.ua);
    solution.uc = std::move(revised_solution.uc);
    solution.bua = std::move(revised_solution.bua);
    solution.buc = std::move(revised_solution.buc);
    solution.rows = std::move(revised_solution.rows);
    solution.rank = revised_solution.rank;
    solution.dres = revised_solution.dres;
    solution.piece_residual = revised_solution.piece_residual;
    solution.core_diagonal_ratio = std::min(
        revised_solution.basis_diagonal_ratio,
        revised_solution.coordinate_diagonal_ratio);
    const bool unguarded = !std::getenv("TWALKER_REVISED_GUARDED");
    const bool local_bound =
        std::getenv("TWALKER_REVISED_LOCAL_BOUND")
        && revised_solution.forward_bound_valid;
    solution.used_extended_gram = !unguarded;
    solution.ua_relative_error_bound =
        unguarded ? 0.0
                  : (local_bound ? revised_solution.ua_relative_error_bound
                                 : 1e-10);
    solution.uc_relative_error_bound =
        unguarded ? 0.0
                  : (local_bound ? revised_solution.uc_relative_error_bound
                                 : 1e-10);
    if (local_bound) {
      solution.affine_bound_valid = true;
      solution.reduced_residual_a =
          std::move(revised_solution.reduced_residual_a);
      solution.reduced_residual_c =
          std::move(revised_solution.reduced_residual_c);
      solution.reduced_head_a = std::move(revised_solution.reduced_head_a);
      solution.reduced_head_c = std::move(revised_solution.reduced_head_c);
      solution.reduced_gram_inf = revised_solution.reduced_gram_inf;
      solution.reduced_rcond = revised_solution.reduced_rcond;
      solution.projection_inf_norm =
          revised_solution.projection_inf_norm;
    }
    if (std::getenv("TWALKER_REVISED_BOUND_AUDIT")) {
      const auto oracle = solver_.solve(rows);
      auto absolute_error = [](const std::vector<double> &left,
                               const std::vector<double> &right) {
        if (left.size() != right.size())
          return std::numeric_limits<double>::infinity();
        double error = 0.0;
        for (std::size_t i = 0; i < left.size(); ++i)
          error = std::max(error, std::abs(left[i] - right[i]));
        return error;
      };
      const double ua_width = revised_solution.ua_relative_error_bound
                              * std::max(1.0, inf_norm(solution.ua));
      const double uc_width = revised_solution.uc_relative_error_bound
                              * std::max(1.0, inf_norm(solution.uc));
      const double ua_error = absolute_error(solution.ua, oracle.ua);
      const double uc_error = absolute_error(solution.uc, oracle.uc);
      solution.audit_oracle_g = oracle.g;
      solution.audit_oracle_h = oracle.h;
      solution.audit_oracle_bua = oracle.bua;
      solution.audit_oracle_buc = oracle.buc;
      const double width = std::max(ua_width, uc_width);
      const bool auditable_bound = revised_solution.forward_bound_valid
                                   && ua_width > 0.0 && uc_width > 0.0;
      const double ratio = auditable_bound
          ? std::max(ua_error / ua_width, uc_error / uc_width)
          : 0.0;
      ++revised_bound_audits_;
      revised_bound_max_width_ = std::max(revised_bound_max_width_, width);
      revised_bound_max_actual_ = std::max(
          revised_bound_max_actual_, std::max(ua_error, uc_error));
      revised_bound_worst_ratio_ =
          std::max(revised_bound_worst_ratio_, ratio);
      if (!auditable_bound || !std::isfinite(ratio) || ratio > 1.0)
        ++revised_bound_violations_;
    }
    ++revised_column_solves_;
    audit_bound_core(rows, solution);
    record_face_rank(solution);
    if (maintained_rowspace_audit_enabled_)
      audit_maintained_rowspace(support, solution);
    return true;
  };
  if (revised_column_primary_ && try_revised()) return solution;
#endif
  if (!gram_retired_) {
    if (gram_solver_.solve(rows, solution)) {
      ++gram_fast_solves_;
      audit_bound_core(rows, solution);
      extension_used_ = extension_used_ || solution.used_extended_gram;
      record_face_rank(solution);
      if (maintained_rowspace_audit_enabled_)
        audit_maintained_rowspace(support, solution);
      return solution;
    }
    ++gram_declines_;
    gram_retired_ = true;
    gram_stability_rearm_pending_ = false;
  }
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if (!revised_column_primary_ && try_revised()) return solution;
  if (revised_column_solver_) {
    const auto *revised_solver =
        static_cast<revised::RevisedColumnSolver *>(revised_column_solver_);
    solver_.set_recurrence_seed_needed(
        revised_solver->needs_recurrence_seed());
    solver_.set_factored_seed_needed(revised_solver->needs_factored_seed());
  }
#endif

#ifdef TWALKER_ENABLE_REVISED_COLUMN
  revised::RevisedFaceSolution svd_candidate;
  bool svd_candidate_ready = false;
  auto *maintained_svd = static_cast<revised::MaintainedSvdFaceSolver *>(
      maintained_svd_face_solver_);
  if ((maintained_svd_audit_enabled_ || maintained_svd_live_enabled_)
      && maintained_svd)
    svd_candidate_ready = maintained_svd->solve(rows, svd_candidate);
  auto import_svd_candidate = [&]() {
    solution.g = std::move(svd_candidate.g);
    solution.h = std::move(svd_candidate.h);
    solution.ua = std::move(svd_candidate.ua);
    solution.uc = std::move(svd_candidate.uc);
    solution.bua = std::move(svd_candidate.bua);
    solution.buc = std::move(svd_candidate.buc);
    solution.rows = std::move(svd_candidate.rows);
    solution.rank = svd_candidate.rank;
    solution.dres = svd_candidate.dres;
    solution.piece_residual = svd_candidate.piece_residual;
  };
  if (maintained_svd_live_enabled_ && svd_candidate_ready) {
    import_svd_candidate();
    record_face_rank(solution);
    return solution;
  }

  revised::RevisedFaceSolution deficient_candidate;
  bool deficient_candidate_ready = false;
  auto *deficient_face_solver =
      static_cast<revised::MaintainedDeficientQrSolver *>(
          maintained_deficient_qr_solver_);
  auto *deficient_recurrence =
      static_cast<revised::RevisedColumnSolver *>(
          deficient_recurrence_solver_);
  bool deficient_candidate_from_recurrence = false;
  if ((deficient_face_audit_enabled_ || deficient_face_live_enabled_)
      && deficient_face_solver)
    deficient_candidate_ready =
        deficient_face_solver->solve_face(rows, deficient_candidate);
  if (!deficient_candidate_ready
      && (deficient_face_audit_enabled_ || deficient_face_live_enabled_)
      && deficient_recurrence) {
    deficient_candidate_ready =
        deficient_recurrence->solve(rows, deficient_candidate);
    deficient_candidate_from_recurrence = deficient_candidate_ready;
  }
  auto import_deficient_candidate = [&]() {
    solution.g = std::move(deficient_candidate.g);
    solution.h = std::move(deficient_candidate.h);
    solution.ua = std::move(deficient_candidate.ua);
    solution.uc = std::move(deficient_candidate.uc);
    solution.bua = std::move(deficient_candidate.bua);
    solution.buc = std::move(deficient_candidate.buc);
    solution.rows = std::move(deficient_candidate.rows);
    solution.rank = deficient_candidate.rank;
    solution.dres = deficient_candidate.dres;
    solution.piece_residual = deficient_candidate.piece_residual;
    solution.core_diagonal_ratio = std::min(
        deficient_candidate.basis_diagonal_ratio,
        deficient_candidate.coordinate_diagonal_ratio);
    // A retained factor is a candidate, not an oracle.  Route every settle
    // and ratio-test decision through the existing interval machinery so a
    // small coefficient uncertainty can never silently reorder a degenerate
    // event.  The recurrence lane receives the wider envelope observed on
    // Lotfi; an ambiguous decision immediately refactors this same face.
    solution.used_extended_gram = true;
    solution.ua_relative_error_bound =
        deficient_candidate_from_recurrence ? 1e-9 : 1e-10;
    solution.uc_relative_error_bound =
        deficient_candidate_from_recurrence ? 1e-9 : 1e-10;
  };
  if (deficient_face_live_enabled_ && deficient_candidate_ready) {
    import_deficient_candidate();
    record_face_rank(solution);
    return solution;
  }
  if (deficient_face_audit_enabled_ || deficient_face_live_enabled_
      || maintained_svd_audit_enabled_ || maintained_svd_live_enabled_)
    solver_.set_factored_seed_needed(true);
#endif
  solution = solver_.solve(rows);
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if ((maintained_svd_audit_enabled_ || maintained_svd_live_enabled_)
      && maintained_svd) {
    if (!svd_candidate_ready && solution.used_dense_fallback) {
      ++maintained_rowspace_audit_.seed_attempts;
      if (maintained_svd->seed(rows, solution)) {
        ++maintained_rowspace_audit_.seed_successes;
        svd_candidate_ready = maintained_svd->solve(rows, svd_candidate);
      }
    }
    if (svd_candidate_ready) {
      ++maintained_rowspace_audit_.checks;
      const double g_error = relative_inf_error(svd_candidate.g, solution.g);
      const double h_error = relative_inf_error(svd_candidate.h, solution.h);
      const double ua_error = relative_inf_error(svd_candidate.ua, solution.ua);
      const double uc_error = relative_inf_error(svd_candidate.uc, solution.uc);
      const double bua_error = relative_inf_error(svd_candidate.bua,
                                                  solution.bua);
      const double buc_error = relative_inf_error(svd_candidate.buc,
                                                  solution.buc);
      maintained_rowspace_audit_.max_g_error = std::max(
          maintained_rowspace_audit_.max_g_error, std::max(g_error, h_error));
      maintained_rowspace_audit_.max_ua_error = std::max(
          maintained_rowspace_audit_.max_ua_error,
          std::max(ua_error, uc_error));
      maintained_rowspace_audit_.max_bua_error = std::max(
          maintained_rowspace_audit_.max_bua_error,
          std::max(bua_error, buc_error));
      const double error = std::max(
          {g_error, h_error, ua_error, uc_error, bua_error, buc_error});
      const bool match = std::isfinite(error) && error <= 2e-10;
      maintained_rowspace_audit_.matches += match;
      maintained_rowspace_audit_.false_admissions += !match;
      if (maintained_rowspace_trace_ && (!match
          || maintained_rowspace_audit_.checks <= 4))
        std::cerr << "MSVD check=" << maintained_rowspace_audit_.checks
                  << " rows=" << rows.size() << " rank=" << solution.rank
                  << " match=" << match << " error=" << error << '\n';
      if (maintained_svd_live_enabled_ && match) import_svd_candidate();
    }
  }
  if ((deficient_face_audit_enabled_ || deficient_face_live_enabled_)
      && deficient_face_solver) {
    if (!deficient_candidate_ready) {
      bool seeded = false;
      if (deficient_qr_seed_cooldown_ == 0) {
        ++maintained_rowspace_audit_.seed_attempts;
        seeded = deficient_face_solver->seed(rows, solution);
        if (!seeded) deficient_qr_seed_cooldown_ = 16;
      } else {
        --deficient_qr_seed_cooldown_;
      }
      if (seeded) {
        deficient_qr_seed_cooldown_ = 0;
        ++maintained_rowspace_audit_.seed_successes;
        deficient_candidate_ready =
            deficient_face_solver->solve_face(rows, deficient_candidate);
      }
      if (!deficient_candidate_ready && deficient_recurrence
          && solution.used_dense_fallback) {
        seeded = deficient_recurrence->seed_from_direct(rows, solution);
        if (seeded) {
          ++maintained_rowspace_audit_.seed_successes;
          deficient_candidate_ready =
              deficient_recurrence->solve(rows, deficient_candidate);
          deficient_candidate_from_recurrence = deficient_candidate_ready;
        }
      }
    }
    if (deficient_candidate_ready) {
      ++maintained_rowspace_audit_.checks;
      const double g_error = relative_inf_error(deficient_candidate.g,
                                                solution.g);
      const double h_error = relative_inf_error(deficient_candidate.h,
                                                solution.h);
      const double ua_error = relative_inf_error(deficient_candidate.ua,
                                                 solution.ua);
      const double uc_error = relative_inf_error(deficient_candidate.uc,
                                                 solution.uc);
      const double bua_error = relative_inf_error(deficient_candidate.bua,
                                                  solution.bua);
      const double buc_error = relative_inf_error(deficient_candidate.buc,
                                                  solution.buc);
      maintained_rowspace_audit_.max_g_error = std::max(
          maintained_rowspace_audit_.max_g_error, std::max(g_error, h_error));
      maintained_rowspace_audit_.max_ua_error = std::max(
          maintained_rowspace_audit_.max_ua_error,
          std::max(ua_error, uc_error));
      maintained_rowspace_audit_.max_bua_error = std::max(
          maintained_rowspace_audit_.max_bua_error,
          std::max(bua_error, buc_error));
      const double error = std::max(
          {g_error, h_error, ua_error, uc_error, bua_error, buc_error});
      const bool match = std::isfinite(error) && error <= 2e-10;
      maintained_rowspace_audit_.matches += match;
      maintained_rowspace_audit_.false_admissions += !match;
      if (maintained_rowspace_trace_ && (!match
          || maintained_rowspace_audit_.checks <= 4))
        std::cerr << "DFQ check=" << maintained_rowspace_audit_.checks
                  << " rows=" << rows.size() << " rank=" << solution.rank
                  << " recurrence=" << deficient_candidate_from_recurrence
                  << " match=" << match << " error=" << error << '\n';
      if (deficient_face_live_enabled_ && match) {
        import_deficient_candidate();
      }
    }
  }
#endif
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if (revised_column_solver_) {
    const bool seeded =
        static_cast<revised::RevisedColumnSolver *>(revised_column_solver_)
            ->seed_from_direct(rows, solution);
    if (seeded) {
      solver_.set_recurrence_seed_needed(false);
      solver_.set_factored_seed_needed(false);
    }
  }
#endif
  dense_fallbacks_ += solution.used_dense_fallback;
  audit_bound_core(rows, solution);
  record_face_rank(solution);
  if (!std::getenv("TWALKER_DISABLE_GRAM_REARM") && gram_retired_
      && (!gram_rearm_used_ || gram_stability_rearm_pending_)
      && solution.rank == static_cast<std::int64_t>(fixture_.m)
      && !solution.used_dense_fallback) {
    gram_retired_ = false;
    gram_rearm_used_ = true;
    gram_stability_rearm_pending_ = false;
    if (std::getenv("TWALKER_GRAM_REARM_TRACE"))
      std::cerr << "GRAM_REARM rows=" << rows.size()
                << " rank=" << solution.rank
                << " core_ratio=" << solution.core_diagonal_ratio << '\n';
  }
  if (maintained_rowspace_audit_enabled_)
    audit_maintained_rowspace(support, solution);
  return solution;
}

void Walker::audit_bound_core(const std::vector<std::uint32_t> &rows,
                              const FaceSolution &oracle) {
  if (!std::getenv("TWALKER_BOUND_CORE_AUDIT")) return;
  FaceSolution candidate;
  if (!bound_core_solver_.solve(rows, candidate)) return;
  const double error = std::max(
      {relative_inf_error(candidate.g, oracle.g),
       relative_inf_error(candidate.h, oracle.h),
       relative_inf_error(candidate.ua, oracle.ua),
       relative_inf_error(candidate.uc, oracle.uc),
       relative_inf_error(candidate.bua, oracle.bua),
       relative_inf_error(candidate.buc, oracle.buc)});
  ++bound_core_audits_;
  bound_core_audit_max_error_ = std::max(bound_core_audit_max_error_, error);
  if (!std::isfinite(error) || error > 1e-10) {
    ++bound_core_audit_violations_;
    if (std::getenv("TWALKER_BOUND_CORE_TRACE"))
      std::cerr << "bound-core audit face=" << face_solves_
                << " error=" << error << " rows=" << rows.size() << '\n';
  }
}

FaceSolution Walker::solve_direct(
    const std::vector<std::uint8_t> &support, bool force_refactor) {
  ++face_solves_;
  auto solution = rank_lift_live_solver_
                      ? rank_lift_live_solver_->solve(support_rows(support))
                      : (force_refactor
                             ? solver_.solve_uncached(support_rows(support))
                             : solver_.solve(support_rows(support)));
  dense_fallbacks_ += solution.used_dense_fallback;
  record_face_rank(solution);
  if (maintained_rowspace_audit_enabled_)
    audit_maintained_rowspace(support, solution);
  return solution;
}

FaceSolution Walker::solve_centered_slope(
    const std::vector<std::uint8_t> &support) {
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if (std::getenv("TWALKER_QUOTIENT_BASIS_LIVE")
      && maintained_deficient_qr_solver_) {
    const auto rows = support_rows(support);
    auto *quotient = static_cast<revised::MaintainedDeficientQrSolver *>(
        maintained_deficient_qr_solver_);
    revised::RevisedSlopeSolution slope;
    auto return_slope = [&]() {
      FaceSolution result;
      result.g = std::move(slope.g);
      result.ua = std::move(slope.ua);
      result.bua = std::move(slope.bua);
      result.rows = std::move(slope.rows);
      result.rank = slope.rank;
      result.piece_residual = slope.slope_residual;
      record_face_rank(result);
      return result;
    };
    if (quotient->solve(rows, slope)) {
      ++face_solves_;
      if (std::getenv("TWALKER_QUOTIENT_BASIS_TRACE"))
        std::cerr << "quotient update rows=" << rows.size()
                  << " rank=" << slope.rank << '\n';
      return return_slope();
    }

    // Fail closed to one rank-revealing seed.  This is native face linear
    // algebra, not a convex-solver side call.  The direct candidate remains
    // authoritative unless the quotient reconstruction matches it.
    solver_.set_recurrence_seed_needed(false);
    solver_.set_factored_seed_needed(true);
    auto direct = solve_direct(support);
    solver_.set_factored_seed_needed(false);
    bool seeded = quotient->seed(rows, direct);
    // A production face-cache hit may intentionally omit cold factor
    // artifacts.  Reveal them with the dedicated native face solver without
    // changing the authoritative cached solution.
    if (!seeded && maintained_rowspace_seed_solver_) {
      maintained_rowspace_seed_solver_->set_recurrence_seed_needed(false);
      maintained_rowspace_seed_solver_->set_factored_seed_needed(true);
      try {
        const auto artifact = maintained_rowspace_seed_solver_->solve(rows);
        seeded = quotient->seed(rows, artifact);
      } catch (const FaceDecline &) {
        seeded = false;
      }
    }
    if (seeded && quotient->solve(rows, slope)) {
      if (std::getenv("TWALKER_QUOTIENT_BASIS_TRACE"))
        std::cerr << "quotient seed rows=" << rows.size()
                  << " rank=" << slope.rank << '\n';
      return return_slope();
    }
    if (std::getenv("TWALKER_QUOTIENT_BASIS_TRACE"))
      std::cerr << "quotient decline rows=" << rows.size()
                << " rank=" << direct.rank << '\n';
    return direct;
  }
  if (centered_slope_solver_) {
    const auto rows = support_rows(support);
    auto *maintained = static_cast<revised::RevisedColumnSolver *>(
        centered_slope_solver_);
    revised::RevisedSlopeSolution slope;
    if (maintained->solve_slope(rows, slope)) {
      ++face_solves_;
      FaceSolution result;
      result.g = std::move(slope.g);
      result.ua = std::move(slope.ua);
      result.bua = std::move(slope.bua);
      result.rows = std::move(slope.rows);
      result.rank = slope.rank;
      result.piece_residual = slope.slope_residual;
      record_face_rank(result);
      return result;
    }

    // The accepted centered endpoint remains authoritative.  Re-reveal only
    // the numerical basis for this support, seed it from the same SPQR work,
    // and immediately resume the maintained representation.
    solver_.set_recurrence_seed_needed(true);
    solver_.set_factored_seed_needed(true);
    auto direct = solve_direct(support);
    solver_.set_recurrence_seed_needed(false);
    solver_.set_factored_seed_needed(false);
    if (std::getenv("TWALKER_CENTERED_BASIS_TRACE"))
      std::cerr << "centered direct artifacts pseudo="
                << direct.recurrence_pseudoinverse.size()
                << " rowspace=" << direct.svd_row_space.size()
                << " dense=" << direct.used_dense_fallback << '\n';
    if (maintained->seed_slope_from_direct(rows, direct)
        && maintained->solve_slope(rows, slope)) {
      FaceSolution result;
      result.g = std::move(slope.g);
      result.ua = std::move(slope.ua);
      result.bua = std::move(slope.bua);
      result.rows = std::move(slope.rows);
      result.rank = slope.rank;
      result.piece_residual = slope.slope_residual;
      return result;
    }
    return direct;
  }
#endif
  return solve_direct(support);
}

bool Walker::audit_maintained_rowspace(
    const std::vector<std::uint8_t> &support,
    const FaceSolution &oracle) {
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if (!maintained_rowspace_solver_) return false;
  const auto rows = support_rows(support);
  revised::RevisedSlopeSolution candidate;
  const bool pseudoinverse_audit =
      std::getenv("TWALKER_MAINTAINED_PINV_AUDIT") != nullptr;
  const bool deficient_qr_audit =
      std::getenv("TWALKER_MAINTAINED_QR_AUDIT") != nullptr;
  auto *maintained = static_cast<revised::MaintainedRowspaceSolver *>(
      maintained_rowspace_solver_);
  auto *pseudoinverse = static_cast<revised::RevisedColumnSolver *>(
      centered_slope_solver_);
  auto *deficient_qr =
      static_cast<revised::MaintainedDeficientQrSolver *>(
          maintained_deficient_qr_solver_);
  bool solved = deficient_qr_audit
                    ? deficient_qr->solve(rows, candidate)
                    : (pseudoinverse_audit
                           ? pseudoinverse->solve_slope(rows, candidate)
                           : maintained->solve(rows, candidate));
  if (!solved) {
    ++maintained_rowspace_audit_.seed_attempts;
    FaceSolution mutable_oracle = oracle;
    bool seeded = deficient_qr_audit
                      ? deficient_qr->seed(rows, oracle)
                      : (pseudoinverse_audit
                             ? pseudoinverse->seed_slope_from_direct(
                                   rows, mutable_oracle)
                             : maintained->seed(rows, oracle));
    if (!seeded && maintained_rowspace_seed_solver_) {
      maintained_rowspace_seed_solver_->set_recurrence_seed_needed(
          !deficient_qr_audit);
      maintained_rowspace_seed_solver_->set_factored_seed_needed(true);
      const auto reveal_start = std::chrono::steady_clock::now();
      try {
        auto artifact = maintained_rowspace_seed_solver_->solve(rows);
        seeded = deficient_qr_audit
                     ? deficient_qr->seed(rows, artifact)
                     : (pseudoinverse_audit
                            ? pseudoinverse->seed_slope_from_direct(
                                  rows, artifact)
                            : maintained->seed(rows, artifact));
      } catch (const FaceDecline &) {
        seeded = false;
      }
      maintained_rowspace_audit_.cold_reveal_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - reveal_start).count();
      ++maintained_rowspace_audit_.cold_reveals;
    }
    if (!seeded && !pseudoinverse_audit && !deficient_qr_audit
        && !maintained->seed_from_rows(rows, oracle.rank)) {
      if (maintained_rowspace_trace_)
        std::cerr << "MRP seed-decline rows=" << rows.size()
                  << " rank=" << oracle.rank
                  << " rowspace=" << oracle.svd_row_space.size() << '\n';
      return false;
    }
    if (!seeded) return false;
    ++maintained_rowspace_audit_.seed_successes;
    solved = deficient_qr_audit
                 ? deficient_qr->solve(rows, candidate)
                 : (pseudoinverse_audit
                        ? pseudoinverse->solve_slope(rows, candidate)
                        : maintained->solve(rows, candidate));
  }
  if (!solved) return false;

  ++maintained_rowspace_audit_.checks;
  const double g_error = relative_inf_error(candidate.g, oracle.g);
  const double ua_error = relative_inf_error(candidate.ua, oracle.ua);
  const double bua_error = relative_inf_error(candidate.bua, oracle.bua);
  maintained_rowspace_audit_.max_g_error =
      std::max(maintained_rowspace_audit_.max_g_error, g_error);
  maintained_rowspace_audit_.max_ua_error =
      std::max(maintained_rowspace_audit_.max_ua_error, ua_error);
  maintained_rowspace_audit_.max_bua_error =
      std::max(maintained_rowspace_audit_.max_bua_error, bua_error);
  const bool match = std::max({g_error, ua_error, bua_error}) <= 2e-10;
  maintained_rowspace_audit_.matches += match;
  maintained_rowspace_audit_.false_admissions += !match;
  if (maintained_rowspace_trace_ && (!match
      || maintained_rowspace_audit_.checks <= 4))
    std::cerr << "MRP check=" << maintained_rowspace_audit_.checks
              << " rows=" << rows.size() << " rank=" << candidate.rank
              << " match=" << match << " g=" << g_error
              << " ua=" << ua_error << " bua=" << bua_error << '\n';
  return match;
#else
  (void)support;
  (void)oracle;
  return false;
#endif
}

bool Walker::audit_augmented_kkt_state(
    const std::vector<std::uint8_t> &support, double t,
    const FaceSolution &solution) {
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if (!augmented_kkt_basis_ || support.size() != fixture_.n
      || solution.bua.size() != fixture_.n
      || solution.buc.size() != fixture_.n)
    return false;
  const double forward = kForward * std::max(1.0, std::abs(t)) + kForward;
  if (!(t > augmented_kkt_last_t_ + forward)) return false;

  std::vector<double> direction(fixture_.n, 0.0);
  std::vector<double> endpoint_value(fixture_.n, 0.0);
  for (std::size_t local = 0; local < solution.rows.size(); ++local) {
    direction[solution.rows[local]] = solution.g[local];
    endpoint_value[solution.rows[local]] = std::fma(
        t, solution.g[local], solution.h[local]);
  }
  std::vector<double> slack(fixture_.n);
  std::vector<double> endpoint_offset(fixture_.n);
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    endpoint_offset[row] = std::fma(t, fixture_.b[row], target_shift_[row]);
    slack[row] = std::fma(
        t, fixture_.b[row] + solution.bua[row],
        target_shift_[row] + solution.buc[row]);
  }
  std::vector<double> endpoint_multiplier(fixture_.m);
  for (std::size_t column = 0; column < fixture_.m; ++column)
    endpoint_multiplier[column] =
        std::fma(t, solution.ua[column], solution.uc[column]);
  std::vector<std::uint8_t> endpoint_constraints(fixture_.n, 0);
  std::vector<std::uint8_t> slope_constraints(fixture_.n, 0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    if (!support[row]) {
      endpoint_constraints[row] = 1;
      // Only a nonnegative endpoint slack requires a right-derivative sign.
      // Strictly negative rows are handled by the ordinary ratio test.
      slope_constraints[row] = slack[row] >= 0.0;
    }

  revised::AugmentedKktSolution endpoint_selected, selected;
  auto *basis = static_cast<revised::AugmentedKktBasis *>(
      augmented_kkt_basis_);
  const std::vector<double> no_warm;
  const auto &endpoint_warm = augmented_kkt_audit_consecutive_ == 0
                                  ? endpoint_multiplier : no_warm;
  const auto &slope_warm = augmented_kkt_audit_consecutive_ == 0
                               ? solution.ua : no_warm;
  if (!basis->select_affine(
          support, endpoint_value, endpoint_offset, endpoint_constraints,
          endpoint_warm, 0, endpoint_selected)) {
    augmented_kkt_audit_consecutive_ = 0;
    if (std::getenv("TWALKER_AUGMENTED_KKT_TRACE"))
      std::cerr << "AUGMENTED_KKT_AUDIT endpoint_declined t="
                << std::setprecision(17) << t
                << " reason=" << basis->last_failure()
                << " blocking=" << basis->blocking_row()
                << " rank=" << endpoint_selected.rank
                << " source_rank=" << solution.rank
                << " active=" << endpoint_selected.active_residual
                << " inactive=" << endpoint_selected.inactive_violation
                << " transpose=" << endpoint_selected.transpose_residual
                << '\n';
    return false;
  }
  if (!basis->select_affine(
          support, direction, fixture_.b, slope_constraints, slope_warm, 1,
          selected)) {
    augmented_kkt_audit_consecutive_ = 0;
    if (std::getenv("TWALKER_AUGMENTED_KKT_TRACE"))
      std::cerr << "AUGMENTED_KKT_AUDIT slope_declined t="
                << std::setprecision(17) << t
                << " reason=" << basis->last_failure()
                << " blocking=" << basis->blocking_row()
                << " rank=" << selected.rank
                << " source_rank=" << solution.rank
                << " active=" << selected.active_residual
                << " inactive=" << selected.inactive_violation
                << " transpose=" << selected.transpose_residual << '\n';
    return false;
  }
  // A repeated augmented fingerprint is harmless after strict t progress;
  // at fixed t it is the exact cycling certificate the state is designed to
  // expose.  This audit never counts a fixed-t exchange as a path update.
  if (!(t > augmented_kkt_last_t_ + forward)
      && selected.fingerprint == augmented_kkt_last_fingerprint_) {
    augmented_kkt_audit_consecutive_ = 0;
    return false;
  }
  augmented_kkt_last_t_ = t;
  augmented_kkt_last_fingerprint_ =
      selected.fingerprint ^ (endpoint_selected.fingerprint << 1);
  ++augmented_kkt_audit_consecutive_;
  if (std::getenv("TWALKER_AUGMENTED_KKT_TRACE"))
    std::cerr << "AUGMENTED_KKT_AUDIT accepted t="
              << std::setprecision(17) << t
              << " consecutive=" << augmented_kkt_audit_consecutive_
              << " rank=" << selected.rank
              << " nullity=" << selected.nullity
              << " selector=" << selected.selector_active_rows.size()
              << " endpoint_selector="
              << endpoint_selected.selector_active_rows.size()
              << " sweeps=" << selected.projection_sweeps
              << " active=" << selected.active_residual
              << " inactive=" << selected.inactive_violation
              << " transpose=" << selected.transpose_residual
              << " fingerprint=" << selected.fingerprint << '\n';
  return true;
#else
  (void)support;
  (void)t;
  (void)solution;
  return false;
#endif
}

void Walker::record_face_rank(const FaceSolution &solution) {
  const auto deficit = static_cast<int>(fixture_.m)
                       - static_cast<int>(solution.rank);
  if (deficit <= 0) return;
  ++rank_deficient_solves_;
  rank_deficit_sum_ += deficit;
  max_rank_deficit_ = std::max(max_rank_deficit_, deficit);
}

void Walker::audit_objective_preserving_rank_lift(
    double t, const FaceSolution &solution) {
  if (!rank_lift_audit_enabled_
      || rank_lift_audit_.audits >= rank_lift_audit_max_faces_)
    return;

  if (!rank_lift_solver_)
    rank_lift_solver_ = new FaceSolver(fixture_, false, target_shift_);
  if (rank_lift_audit_.global_rank < 0) {
    std::vector<std::uint8_t> all_rows(fixture_.n, 1);
    try {
      const auto factor_begin = std::chrono::steady_clock::now();
      const auto global =
          rank_lift_solver_->solve(support_rows(all_rows));
      rank_lift_audit_.global_factor_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - factor_begin)
              .count();
      rank_lift_audit_.global_rank = static_cast<int>(global.rank);
    } catch (const FaceDecline &) {
      // Retain the algebraic column count as a conservative upper bound if
      // the shadow-only global factorization declines.
      rank_lift_audit_.global_rank = static_cast<int>(fixture_.m);
    }
  }
  if (solution.rank >= rank_lift_audit_.global_rank) {
    ++rank_lift_audit_.structurally_maximal_faces;
    return;
  }

  ++rank_lift_audit_.audits;
  rank_lift_audit_.source_dense_fallbacks += solution.used_dense_fallback;
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  std::vector<double> y(n, 0.0);
  for (std::size_t local = 0; local < solution.rows.size(); ++local)
    y[solution.rows[local]] =
        std::max(0.0, std::fma(t, solution.g[local], solution.h[local]));
  const double original_objective = dot(fixture_.b, y);
  const double initial_gate = kDualSupport * std::max(1.0, inf_norm(y));
  std::vector<std::uint8_t> lifted_support(fixture_.n, 0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    lifted_support[row] = y[row] > initial_gate;
  int activated = 0;
  std::int64_t lifted_rank = solution.rank;
  bool lifted_used_dense_fallback = false;
  if (rank_lift_row_weight_.empty()) {
    rank_lift_row_weight_.assign(n, 0.0);
    double maximum_row_norm = 0.0;
    for (int row = 0; row < n; ++row) {
      double square = 0.0;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        square += fixture_.values[p] * fixture_.values[p];
      rank_lift_row_weight_[row] = std::sqrt(square);
      maximum_row_norm =
          std::max(maximum_row_norm, rank_lift_row_weight_[row]);
    }
    maximum_row_norm = std::max(maximum_row_norm, 1e-300);
    for (double &weight : rank_lift_row_weight_)
      weight /= maximum_row_norm;
  }

  // Convexity means successive feasible tangent steps stay in the same
  // objective-level face.  A few rounds let a first step expose directions
  // that were blocked by a zero coordinate without turning this census into
  // an unbounded auxiliary solve.
  for (int pass = 0; pass < 3; ++pass) {
    const double y_scale = std::max(1.0, inf_norm(y));
    const double positive_gate = kDualSupport * y_scale;

    std::vector<double> cost(n, 0.0), lower(n, 0.0), upper(n, 1.0);
    for (int row = 0; row < n; ++row) {
      // Only genuinely positive coordinates may decrease.  This is the
      // tangent cone of y >= 0, not merely the walker's numerical mask.
      if (y[row] > positive_gate) lower[row] = -1.0;
      // Reward only coordinates outside the current numerical face; making a
      // zero row already in W positive cannot increase rank(B_W).
      if (!lifted_support[row])
        cost[row] = -rank_lift_row_weight_[row];
    }
    const auto lp_begin = std::chrono::steady_clock::now();
    int status = 0;
    if (!rank_lift_highs_) {
      std::vector<double> col_scale(m, 0.0);
      for (std::size_t p = 0; p < fixture_.nnz; ++p)
        col_scale[fixture_.indices[p]] =
            std::max(col_scale[fixture_.indices[p]],
                     std::abs(fixture_.values[p]));
      for (double &scale : col_scale) scale = std::max(scale, 1e-300);
      double b_scale = 0.0;
      int b_nnz = 0;
      for (double value : fixture_.b) {
        b_scale = std::max(b_scale, std::abs(value));
        if (value != 0.0) ++b_nnz;
      }
      b_scale = std::max(b_scale, 1e-300);

      // Invariant row-wise matrix for [B^T; b^T] p = 0.  HiGHS copies it;
      // later faces update only bounds and costs and retain the simplex basis.
      std::vector<int> starts(m + 2, 0);
      for (std::size_t p = 0; p < fixture_.nnz; ++p)
        ++starts[static_cast<std::size_t>(fixture_.indices[p]) + 1];
      for (int column = 0; column < m; ++column)
        starts[column + 1] += starts[column];
      starts[m + 1] = starts[m] + b_nnz;
      std::vector<int> next(starts.begin(), starts.begin() + m + 1);
      std::vector<int> indices(fixture_.nnz + b_nnz);
      std::vector<double> values(fixture_.nnz + b_nnz);
      for (int row = 0; row < n; ++row) {
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1];
             ++p) {
          const int constraint = static_cast<int>(fixture_.indices[p]);
          const int slot = next[constraint]++;
          indices[slot] = row;
          values[slot] = fixture_.values[p] / col_scale[constraint];
        }
      }
      int b_slot = starts[m];
      for (int row = 0; row < n; ++row) {
        if (fixture_.b[row] == 0.0) continue;
        indices[b_slot] = row;
        values[b_slot] = fixture_.b[row] / b_scale;
        ++b_slot;
      }
      std::vector<double> row_lower(m + 1, 0.0), row_upper(m + 1, 0.0);
      rank_lift_highs_ = Highs_create();
      if (!rank_lift_highs_) break;
      Highs_setBoolOptionValue(rank_lift_highs_, "output_flag", 0);
      Highs_setIntOptionValue(rank_lift_highs_, "threads", 1);
      Highs_setStringOptionValue(rank_lift_highs_, "presolve", "off");
      Highs_setStringOptionValue(rank_lift_highs_, "solver", "simplex");
      status = Highs_passLp(
          rank_lift_highs_, n, m + 1, static_cast<int>(values.size()), 2, 1,
          0.0, cost.data(), lower.data(), upper.data(), row_lower.data(),
          row_upper.data(), starts.data(), indices.data(), values.data());
    } else {
      status = Highs_changeColsCostByRange(rank_lift_highs_, 0, n - 1,
                                           cost.data());
      if (status == 0)
        status = Highs_changeColsBoundsByRange(
            rank_lift_highs_, 0, n - 1, lower.data(), upper.data());
    }
    if (status == 0) status = Highs_run(rank_lift_highs_);
    rank_lift_audit_.lp_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - lp_begin)
            .count();
    ++rank_lift_audit_.lp_solves;
    std::vector<double> direction(n, 0.0);
    const bool solved = status == 0
                        && Highs_getModelStatus(rank_lift_highs_) == 7
                        && Highs_getSolution(rank_lift_highs_,
                                             direction.data(), nullptr,
                                             nullptr, nullptr) == 0;
    if (!solved) break;

    std::vector<double> dual_direction(m, 0.0), dual_scale(m, 0.0);
    double objective_direction = 0.0, objective_scale = 0.0;
    double rewarded_mass = 0.0;
    for (int row = 0; row < n; ++row) {
      objective_direction += fixture_.b[row] * direction[row];
      objective_scale += std::abs(fixture_.b[row] * direction[row]);
      if (!lifted_support[row])
        rewarded_mass += rank_lift_row_weight_[row] * direction[row];
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto column = fixture_.indices[p];
        dual_direction[column] += fixture_.values[p] * direction[row];
        dual_scale[column] +=
            std::abs(fixture_.values[p] * direction[row]);
      }
    }
    double dual_residual = 0.0;
    for (int column = 0; column < m; ++column)
      dual_residual = std::max(
          dual_residual,
          std::abs(dual_direction[column]) / (1.0 + dual_scale[column]));
    const double objective_residual =
        std::abs(objective_direction) / (1.0 + objective_scale);
    rank_lift_audit_.max_dual_direction_residual = std::max(
        rank_lift_audit_.max_dual_direction_residual, dual_residual);
    rank_lift_audit_.max_objective_direction_residual = std::max(
        rank_lift_audit_.max_objective_direction_residual,
        objective_residual);
    if (dual_residual > kTolerance || objective_residual > kTolerance
        || rewarded_mass <= 1e-10)
      break;

    double alpha_bound = std::numeric_limits<double>::infinity();
    for (int row = 0; row < n; ++row)
      if (direction[row] < -1e-12)
        alpha_bound = std::min(alpha_bound,
                               y[row] / -direction[row]);
    const double alpha = std::isfinite(alpha_bound)
                             ? 0.5 * std::max(0.0, alpha_bound)
                             : y_scale;
    if (!(alpha > 4.0 * positive_gate)) {
      ++rank_lift_audit_.weak_steps;
      break;
    }

    int pass_activated = 0;
    for (int row = 0; row < n; ++row) {
      y[row] = std::max(0.0, y[row] + alpha * direction[row]);
      if (!lifted_support[row] && y[row] > positive_gate) {
        lifted_support[row] = 1;
        ++pass_activated;
      }
    }
    if (pass_activated == 0) {
      ++rank_lift_audit_.weak_steps;
      break;
    }
    activated += pass_activated;
    ++rank_lift_audit_.direction_successes;
    try {
      const auto factor_begin = std::chrono::steady_clock::now();
      const auto lifted =
          rank_lift_solver_->solve(support_rows(lifted_support));
      rank_lift_audit_.factor_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - factor_begin)
              .count();
      rank_lift_audit_.candidate_dense_fallbacks +=
          lifted.used_dense_fallback;
      lifted_used_dense_fallback = lifted_used_dense_fallback
                                   || lifted.used_dense_fallback;
      lifted_rank = std::max(lifted_rank, lifted.rank);
      if (lifted_rank >= rank_lift_audit_.global_rank) break;
    } catch (const FaceDecline &) {
      break;
    }
  }

  rank_lift_audit_.activated_rows += activated;
  const double lifted_objective = dot(fixture_.b, y);
  rank_lift_audit_.max_objective_drift = std::max(
      rank_lift_audit_.max_objective_drift,
      std::abs(lifted_objective - original_objective)
          / (1.0 + std::abs(lifted_objective)
             + std::abs(original_objective)));
  const int gain = static_cast<int>(lifted_rank - solution.rank);
  if (gain > 0) {
    ++rank_lift_audit_.rank_gains;
    rank_lift_audit_.total_rank_gain += gain;
    rank_lift_audit_.max_rank_gain =
        std::max(rank_lift_audit_.max_rank_gain, gain);
    if (lifted_rank >= rank_lift_audit_.global_rank)
      ++rank_lift_audit_.full_rank;
    if (rank_lift_live_requested_ && rank_lift_candidate_y_.empty()
        && solution.used_dense_fallback
        && lifted_rank >= rank_lift_audit_.global_rank
        && !lifted_used_dense_fallback) {
      rank_lift_candidate_support_ = lifted_support;
      rank_lift_candidate_y_ = y;
    }
  }
}

void Walker::coefficient_product_bounds(
    const FaceSolution &solution, std::vector<double> &ua_error,
    std::vector<double> &uc_error) const {
  ua_error.assign(fixture_.n, 0.0);
  uc_error.assign(fixture_.n, 0.0);
  if (!solution.used_extended_gram) return;
  if (solution.affine_bound_valid
      && solution.product_projection_inf_norm.size() == fixture_.n
      && solution.reduced_residual_a.size()
             == solution.reduced_head_a.size()
      && solution.reduced_residual_c.size()
             == solution.reduced_head_c.size()
      && solution.reduced_rcond > 0.0 && solution.reduced_gram_inf > 0.0) {
    auto head_bound = [&](const std::vector<double> &residual,
                          const std::vector<double> &head) {
      long double residual_inf = 0.0L, head_inf = 0.0L;
      for (double value : residual)
        residual_inf = std::max(
            residual_inf, std::abs(static_cast<long double>(value)));
      for (double value : head)
        head_inf = std::max(
            head_inf, std::abs(static_cast<long double>(value)));
      return 10.0L * residual_inf
                 / (static_cast<long double>(solution.reduced_rcond)
                    * solution.reduced_gram_inf)
             + 64.0L * std::numeric_limits<double>::epsilon()
                   * std::max(1.0L, head_inf);
    };
    const long double ua_head = head_bound(
        solution.reduced_residual_a, solution.reduced_head_a);
    const long double uc_head = head_bound(
        solution.reduced_residual_c, solution.reduced_head_c);
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      long double ua_magnitude = 0.0L, uc_magnitude = 0.0L;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const long double magnitude = std::abs(
            static_cast<long double>(fixture_.values[p]));
        const auto column = fixture_.indices[p];
        ua_magnitude += magnitude
                        * std::abs(static_cast<long double>(solution.ua[column]));
        uc_magnitude += magnitude
                        * std::abs(static_cast<long double>(solution.uc[column]));
      }
      const long double mapping = solution.product_projection_inf_norm[row];
      ua_error[row] = static_cast<double>(
          mapping * ua_head
          + 32.0L * std::numeric_limits<double>::epsilon() * ua_magnitude);
      uc_error[row] = static_cast<double>(
          mapping * uc_head
          + 32.0L * std::numeric_limits<double>::epsilon() * uc_magnitude);
    }
    return;
  }
  const double ua_coefficient = solution.ua_relative_error_bound
                                * std::max(1.0, inf_norm(solution.ua));
  const double uc_coefficient = solution.uc_relative_error_bound
                                * std::max(1.0, inf_norm(solution.uc));
  constexpr double kRoundoffSafety =
      32.0 * std::numeric_limits<double>::epsilon();
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    double row_l1 = 0.0, ua_magnitude = 0.0, uc_magnitude = 0.0;
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const double magnitude = std::abs(fixture_.values[p]);
      const auto column = fixture_.indices[p];
      row_l1 += magnitude;
      ua_magnitude += magnitude * std::abs(solution.ua[column]);
      uc_magnitude += magnitude * std::abs(solution.uc[column]);
    }
    ua_error[row] = row_l1 * ua_coefficient
                    + kRoundoffSafety * ua_magnitude;
    uc_error[row] = row_l1 * uc_coefficient
                    + kRoundoffSafety * uc_magnitude;
  }
}

bool Walker::affine_product_bounds(const FaceSolution &solution, double t,
                                   std::vector<double> &error) const {
  if (!solution.affine_bound_valid) return false;
  const auto rank = solution.reduced_head_a.size();
  if (rank == 0 || solution.reduced_head_c.size() != rank
      || solution.reduced_residual_a.size() != rank
      || solution.reduced_residual_c.size() != rank
      || !std::isfinite(solution.reduced_gram_inf)
      || solution.reduced_gram_inf <= 0.0
      || !std::isfinite(solution.reduced_rcond)
      || solution.reduced_rcond <= 0.0
      || !std::isfinite(solution.projection_inf_norm)
      || solution.projection_inf_norm <= 0.0)
    return false;

  long double residual_inf = 0.0L, head_inf = 0.0L;
  long double head_roundoff_scale = 0.0L;
  for (std::size_t j = 0; j < rank; ++j) {
    residual_inf = std::max(
        residual_inf,
        std::abs(-static_cast<long double>(t)
                     * solution.reduced_residual_a[j]
                 + solution.reduced_residual_c[j]));
    head_inf = std::max(
        head_inf,
        std::abs(-static_cast<long double>(t)
                     * solution.reduced_head_a[j]
                 + solution.reduced_head_c[j]));
    head_roundoff_scale = std::max(
        head_roundoff_scale,
        std::abs(static_cast<long double>(t)
                 * solution.reduced_head_a[j])
            + std::abs(static_cast<long double>(
                solution.reduced_head_c[j])));
  }
  const long double head_error =
      10.0L * residual_inf
      / (static_cast<long double>(solution.reduced_rcond)
         * solution.reduced_gram_inf);
  constexpr long double kRoundoffSafety =
      64.0L * std::numeric_limits<double>::epsilon();
  const long double head_absolute =
      head_error + kRoundoffSafety * std::max(1.0L, head_roundoff_scale);
  const long double coefficient_absolute =
      static_cast<long double>(solution.projection_inf_norm) * head_error
      + kRoundoffSafety
            * static_cast<long double>(solution.projection_inf_norm)
            * std::max(1.0L, head_roundoff_scale);
  const long double representation_error =
      std::max(head_absolute, coefficient_absolute);
  if (!std::isfinite(representation_error)) return false;

  error.assign(fixture_.n, 0.0);
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    long double row_l1 = 0.0L, product_magnitude = 0.0L;
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const long double magnitude = std::abs(
          static_cast<long double>(fixture_.values[p]));
      const auto column = fixture_.indices[p];
      row_l1 += magnitude;
      product_magnitude += magnitude
          * (std::abs(static_cast<long double>(t) * solution.ua[column])
             + std::abs(static_cast<long double>(solution.uc[column])));
    }
    const long double propagated =
        solution.product_projection_inf_norm.size() == fixture_.n
            ? static_cast<long double>(
                  solution.product_projection_inf_norm[row]) * head_absolute
            : row_l1 * representation_error;
    error[row] = static_cast<double>(
        propagated + 32.0L * std::numeric_limits<double>::epsilon()
                         * product_magnitude);
    if (!std::isfinite(error[row])) return false;
  }
  return true;
}

bool Walker::settle_decision_stable(
    const std::vector<std::uint8_t> &support, double t,
    const FaceSolution &solution, const std::vector<double> &y,
    const std::vector<double> &q, double y_scale, double q_scale,
    bool one_at_a_time) const {
  if (!solution.used_extended_gram) return true;
  // This repair mode compares severities across rows.  Certifying every ratio
  // ordering costs more than the exceptional direct solve it would save.
  if (one_at_a_time) return false;

  std::vector<double> affine_error(fixture_.n);
  if (!affine_product_bounds(solution, t, affine_error)) {
    std::vector<double> ua_error, uc_error;
    coefficient_product_bounds(solution, ua_error, uc_error);
    for (std::size_t row = 0; row < fixture_.n; ++row)
      affine_error[row] = std::abs(t) * ua_error[row] + uc_error[row];
  }
  double y_scale_error = 0.0, q_scale_error = 0.0;
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    q_scale_error = std::max(q_scale_error, affine_error[row]);
  }
  for (auto row : solution.rows)
    y_scale_error = std::max(y_scale_error, affine_error[row]);
  y_scale_error *= 1e-8;
  q_scale_error *= 1e-8;

  std::vector<std::uint8_t> will_drop(fixture_.n, 0);
  for (std::size_t local = 0; local < solution.rows.size(); ++local) {
    const auto row = solution.rows[local];
    if (std::abs(y[local] + y_scale)
        <= affine_error[row] + y_scale_error) {
      if (std::getenv("TWALKER_REVISED_BOUND_TRACE"))
        std::cerr << "settle interval active row=" << row
                  << " margin=" << std::abs(y[local] + y_scale)
                  << " error=" << affine_error[row] + y_scale_error
                  << " affine=" << solution.affine_bound_valid << '\n';
      return false;
    }
    will_drop[row] = y[local] < -y_scale;
  }
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    if (support[row] && !will_drop[row]) continue;
    if (std::abs(q[row] - q_scale)
        <= affine_error[row] + q_scale_error) {
      if (std::getenv("TWALKER_REVISED_BOUND_TRACE"))
        std::cerr << "settle interval inactive row=" << row
                  << " margin=" << std::abs(q[row] - q_scale)
                  << " error=" << affine_error[row] + q_scale_error
                  << " affine=" << solution.affine_bound_valid << '\n';
      return false;
    }
  }
  return true;
}

bool Walker::event_decision_stable(
    double t, double tmax, const FaceSolution &solution,
    const std::vector<double> &slope,
    const std::vector<double> &constant,
    const std::vector<double> &candidate, double &next,
    bool *retained_tie_certificate) {
  if (retained_tie_certificate) *retained_tie_certificate = false;
  if (!solution.used_extended_gram) return true;
  ++event_interval_.checks;
  auto decline = [&](int &counter) {
    ++counter;
    return false;
  };
  auto certify = [&]() {
    ++event_interval_.certified;
    return true;
  };
  std::vector<double> slope_error, constant_error;
  coefficient_product_bounds(solution, slope_error, constant_error);
  const double lower_event = t * (1.0 + kForward) + kForward;
  constexpr double kSlopeGate = -1e-13;
  auto root_interval = [&](std::size_t row, double &low, double &high) {
    const double slope_low = slope[row] - slope_error[row];
    const double slope_high = slope[row] + slope_error[row];
    if (!(slope_high < kSlopeGate)) return false;
    const double constant_low = constant[row] - constant_error[row];
    const double constant_high = constant[row] + constant_error[row];
    const double roots[4] = {
        -constant_low / slope_low, -constant_low / slope_high,
        -constant_high / slope_low, -constant_high / slope_high};
    low = roots[0];
    high = roots[0];
    for (double root : roots) {
      if (!std::isfinite(root)) return false;
      low = std::min(low, root);
      high = std::max(high, root);
    }
    // Cover the final double divisions as well as the propagated coefficient
    // interval.  This is tiny except at exactly the boundary where direct is
    // the intended route.
    const double roundoff = 16.0 * std::numeric_limits<double>::epsilon()
                            * std::max({1.0, std::abs(low), std::abs(high)});
    low -= roundoff;
    high += roundoff;
    return true;
  };

  std::vector<std::size_t> central_group;
  if (std::isfinite(next)) {
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      if (!std::isfinite(candidate[row])) continue;
      if (std::abs(candidate[row] - next)
          <= kTie * std::max(1.0, next))
        central_group.push_back(row);
    }
  }
  if (std::isfinite(next) && central_group.empty())
    return decline(event_interval_.no_central_group);
  const bool retained_tie_enabled =
      std::getenv("TWALKER_RETAINED_TIE_CERT") != nullptr;
  if (central_group.size() > 1 && !retained_tie_enabled)
    return decline(event_interval_.tie_disabled);

  std::vector<std::uint8_t> in_group(fixture_.n, 0);
  double event_low = std::numeric_limits<double>::infinity();
  double event_high = event_low;
  if (!central_group.empty()) {
    double group_low = std::numeric_limits<double>::infinity();
    double group_high = -std::numeric_limits<double>::infinity();
    double minimum_root_high = std::numeric_limits<double>::infinity();
    for (auto row : central_group) {
      double low = 0.0, high = 0.0;
      if (!root_interval(row, low, high) || !(low > lower_event))
        return decline(event_interval_.winner_interval);
      in_group[row] = 1;
      group_low = std::min(group_low, low);
      group_high = std::max(group_high, high);
      minimum_root_high = std::min(minimum_root_high, high);
    }
    event_low = group_low;
    event_high = minimum_root_high;
    if (central_group.size() > 1) {
      const double group_tie_margin =
          kTie * std::max(1.0, std::abs(group_low));
      if (!(group_high - group_low <= group_tie_margin))
        return decline(event_interval_.tie_width);
      // This is one geometric event, not one walker pivot per row.  The
      // retained factor applies the simultaneous status changes in ascending
      // row order; no intermediate fixed-t representation is counted as
      // progress.
      if (retained_tie_certificate) *retained_tie_certificate = true;
    }
  }

  // Do not demand that the sign of every nearly-zero slope be known.  That
  // caused a cold SPQR reconstruction whenever any irrelevant row straddled
  // the slope threshold.  It is enough to prove that such a row remains
  // strictly feasible through the selected forward horizon.  Confidently
  // decreasing rows still receive the full root-interval comparison.
  const double horizon = !std::isfinite(next)
                             ? tmax
                             : (next > tmax ? tmax : event_high);
  if (!std::isfinite(horizon)) return decline(event_interval_.horizon);
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    if (in_group[row]) continue;
    const double slope_low = slope[row] - slope_error[row];
    const double slope_high = slope[row] + slope_error[row];
    if (slope_low >= kSlopeGate) continue;

    if (slope_high < kSlopeGate) {
      double low = 0.0, high = 0.0;
      if (!root_interval(row, low, high))
        return decline(event_interval_.competitor_interval);
      if (high <= lower_event) continue;
      if (!(low > lower_event))
        return decline(event_interval_.competitor_interval);
      if (!std::isfinite(next)) {
        if (!(low > tmax))
          return decline(event_interval_.competitor_margin);
        continue;
      }
      if (next > tmax) {
        if (!(low > tmax))
          return decline(event_interval_.competitor_margin);
        continue;
      }
      const double tie_margin = kTie * std::max(
          {1.0, std::abs(event_low), std::abs(event_high)});
      if (!(low - event_high > tie_margin))
        return decline(event_interval_.competitor_margin);
      continue;
    }

    // A degenerate simplex basis may legitimately retain a basic variable at
    // zero.  Do not confuse that load-bearing basis row with a leaving event
    // when both its current value and its computed derivative are at
    // roundoff scale.  The row stays in the square factor; this only assigns
    // its bound status lexicographically.  Every later face and final answer
    // still passes the unchanged accept and original-data certificate gates.
    if (!std::getenv("TWALKER_DISABLE_STICKY_BASIC_ZERO")
        && std::binary_search(solution.rows.begin(), solution.rows.end(),
                              static_cast<std::uint32_t>(row))) {
      const double current = std::fma(t, slope[row], constant[row]);
      const double current_scale = 1.0 + std::abs(t * slope[row])
                                   + std::abs(constant[row]);
      const double derivative_scale =
          1.0 + std::abs(fixture_.b[row])
          + std::abs(solution.bua[row]);
      if (std::abs(current) <= 1e-12 * current_scale
          && std::abs(slope[row])
                 <= 64.0 * std::numeric_limits<double>::epsilon()
                        * derivative_scale) {
        ++event_interval_.sticky_basic_zero_certified;
        continue;
      }
    }

    // The slope interval crosses the eligibility threshold.  Since the row
    // is feasible at the accepted endpoint and every admissible decreasing
    // realization is affine, positivity at the forward horizon proves it
    // cannot pre-empt the selected event.  Directed coefficient intervals
    // and an explicit final-operation allowance make this fail closed.
    const long double value_low =
        static_cast<long double>(constant[row] - constant_error[row])
        + static_cast<long double>(horizon)
              * static_cast<long double>(slope_low);
    const long double roundoff =
        32.0L * std::numeric_limits<double>::epsilon()
        * (1.0L
           + std::abs(static_cast<long double>(constant[row]))
           + std::abs(static_cast<long double>(horizon)
                      * static_cast<long double>(slope[row])));
    if (!(value_low > roundoff))
      return decline(event_interval_.uncertain_slope);
  }

  if (!std::isfinite(next)) return certify();
  if (next > tmax)
    return event_low > tmax ? certify()
                            : decline(event_interval_.final_cap);
  return event_high <= tmax ? certify()
                            : decline(event_interval_.final_cap);
}

void Walker::audit_qr_settle_decision(
    const std::vector<std::uint8_t> &support, double t,
    const FaceSolution &solution, bool one_at_a_time,
    const std::vector<std::uint8_t> &authoritative_next) {
  if (!solution.qr_update_audit_candidate
      || solution.qr_update_audit_g.size() != solution.rows.size()
      || solution.qr_update_audit_h.size() != solution.rows.size()
      || solution.qr_update_audit_bua.size() != fixture_.n
      || solution.qr_update_audit_buc.size() != fixture_.n)
    return;
  ++qr_settle_decision_checks_;
  std::vector<double> y(solution.rows.size()), q(fixture_.n);
  double y_max = 0.0, q_max = 0.0;
  for (std::size_t local = 0; local < solution.rows.size(); ++local) {
    y[local] = t * solution.qr_update_audit_g[local]
               + solution.qr_update_audit_h[local];
    y_max = std::max(y_max, std::abs(y[local]));
  }
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    q[row] = t * (fixture_.b[row] + solution.qr_update_audit_bua[row])
             + target_shift_[row] + solution.qr_update_audit_buc[row];
    q_max = std::max(q_max, std::abs(q[row]));
  }
  const double y_scale = 1e-8 * std::max(1.0, y_max);
  const double q_scale = 1e-8 * std::max(1.0, q_max);
  auto active_drop_severity = [&](double value) {
    const double support_severity = -value / y_scale;
    const double endpoint_severity =
        std::max(0.0, -value / (1.0 + std::abs(value))) / kTolerance;
    return std::max(support_severity, endpoint_severity);
  };
  auto candidate_next = support;
  if (one_at_a_time) {
    std::size_t drop_row = fixture_.n, add_row = fixture_.n;
    double drop_severity = 0.0, add_severity = 0.0;
    for (std::size_t local = 0; local < solution.rows.size(); ++local) {
      const double severity = active_drop_severity(y[local]);
      if (severity > 1.0 && severity > drop_severity) {
        drop_severity = severity;
        drop_row = solution.rows[local];
      }
    }
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      if (support[row]) continue;
      const double severity = q[row] / q_scale;
      if (severity > 1.0 && severity > add_severity) {
        add_severity = severity;
        add_row = row;
      }
    }
    if (drop_severity >= add_severity && drop_row < fixture_.n)
      candidate_next[drop_row] = 0;
    else if (add_row < fixture_.n)
      candidate_next[add_row] = 1;
  } else {
    std::vector<std::size_t> drop_rows, add_rows;
    for (std::size_t local = 0; local < solution.rows.size(); ++local)
      if (active_drop_severity(y[local]) > 1.0)
        drop_rows.push_back(solution.rows[local]);
    if (drop_rows.empty()) {
      for (std::size_t row = 0; row < fixture_.n; ++row)
        if (!support[row] && q[row] > q_scale) add_rows.push_back(row);
    } else {
      std::vector<std::uint8_t> will_drop(fixture_.n, 0);
      for (auto row : drop_rows) will_drop[row] = 1;
      for (std::size_t row = 0; row < fixture_.n; ++row)
        if ((!support[row] || will_drop[row]) && q[row] > q_scale)
          add_rows.push_back(row);
    }
    for (auto row : drop_rows) candidate_next[row] = 0;
    for (auto row : add_rows) candidate_next[row] = 1;
  }
  if (candidate_next == authoritative_next)
    ++qr_settle_decision_matches_;
}

void Walker::audit_qr_event_decision(
    double t, double tmax, const FaceSolution &solution, double next,
    const std::vector<std::size_t> &authoritative_ties) {
  if (!solution.qr_update_audit_candidate
      || solution.qr_update_audit_g.size() != solution.rows.size()
      || solution.qr_update_audit_h.size() != solution.rows.size()
      || solution.qr_update_audit_bua.size() != fixture_.n
      || solution.qr_update_audit_buc.size() != fixture_.n)
    return;
  ++qr_event_decision_checks_;
  std::vector<double> slope(fixture_.n), constant(fixture_.n);
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    slope[row] = -(fixture_.b[row] + solution.qr_update_audit_bua[row]);
    constant[row] = -(target_shift_[row]
                      + solution.qr_update_audit_buc[row]);
  }
  for (std::size_t local = 0; local < solution.rows.size(); ++local) {
    slope[solution.rows[local]] = solution.qr_update_audit_g[local];
    constant[solution.rows[local]] = solution.qr_update_audit_h[local];
  }
  double candidate_next = std::numeric_limits<double>::infinity();
  std::vector<double> roots(fixture_.n,
                            std::numeric_limits<double>::infinity());
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    if (slope[row] >= -1e-13) continue;
    const double value = -constant[row] / slope[row];
    if (std::isfinite(value)
        && value > t * (1.0 + kForward) + kForward) {
      roots[row] = value;
      candidate_next = std::min(candidate_next, value);
    }
  }
  std::vector<std::size_t> candidate_ties;
  if (std::isfinite(candidate_next))
    for (std::size_t row = 0; row < fixture_.n; ++row)
      if (std::isfinite(roots[row])
          && std::abs(roots[row] - candidate_next)
                 <= kTie * std::max(1.0, candidate_next))
        candidate_ties.push_back(row);
  const bool authoritative_finite = std::isfinite(next);
  const bool candidate_finite = std::isfinite(candidate_next);
  bool match = authoritative_finite == candidate_finite;
  if (match && (!authoritative_finite
                || candidate_ties == authoritative_ties))
    ++qr_event_tie_set_matches_;
  if (match && authoritative_finite) {
    const double scale = std::max({1.0, std::abs(next),
                                   std::abs(candidate_next)});
    match = std::abs(candidate_next - next) <= kTie * scale
            && (candidate_next > tmax) == (next > tmax)
            && candidate_ties == authoritative_ties;
  }
  if (match) ++qr_event_decision_matches_;
}

bool Walker::selector_feasible(const std::vector<std::uint8_t> &support,
                               double t,
                               const std::vector<double> &face_y,
                               std::vector<double> *selected_u) {
  const int n = static_cast<int>(fixture_.n);
  std::vector<double> active_value(n, 0.0), offset(n);
  std::vector<std::uint8_t> constraints(n, 1);
  std::size_t local = 0;
  for (int row = 0; row < n; ++row) {
    offset[row] = std::fma(t, fixture_.b[row], target_shift_[row]);
    if (support[row]) active_value[row] = face_y[local++];
  }
  if (local != face_y.size()) return false;
  return selector_affine_feasible(support, active_value, offset, constraints,
                                  selected_u);
}

bool Walker::selector_affine_feasible(
    const std::vector<std::uint8_t> &support,
    const std::vector<double> &active_value,
    const std::vector<double> &offset,
    const std::vector<std::uint8_t> &selector_constraints,
    std::vector<double> *selected_u,
    std::vector<double> *selected_product,
    std::uint64_t *basis_fingerprint,
    double equality_relative_band) {
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  if (support.size() != fixture_.n || active_value.size() != fixture_.n
      || offset.size() != fixture_.n
      || selector_constraints.size() != fixture_.n)
    return false;
  const bool basis_trace =
      std::getenv("TWALKER_SIMPLEX_BASIS_TRACE") != nullptr;
  auto decline = [&](const char *reason, int detail = 0) {
    if (basis_trace)
      std::cerr << "SIMPLEX_BASIS declined=" << reason
                << " detail=" << detail << '\n';
    return false;
  };
  std::vector<double> lower(n, -1e30), upper(n, 1e30);
  std::vector<int> starts(n + 1), indices(fixture_.nnz);
  for (int row = 0; row < n; ++row) {
    starts[row] = static_cast<int>(fixture_.indptr[row]);
    if (support[row]) {
      const double rhs = active_value[row] - offset[row];
      const double band = equality_relative_band
          * (1.0 + std::abs(active_value[row]) + std::abs(offset[row]));
      lower[row] = rhs - band;
      upper[row] = rhs + band;
    } else if (selector_constraints[row]) {
      upper[row] = -offset[row];
    }
  }
  starts[n] = static_cast<int>(fixture_.nnz);

  int status = 0;
  if (!selector_) {
    selector_ = Highs_create();
    if (!selector_) return false;
    Highs_setBoolOptionValue(selector_, "output_flag", 0);
    Highs_setIntOptionValue(selector_, "threads", 1);
    Highs_setStringOptionValue(selector_, "presolve", "off");
    Highs_setStringOptionValue(selector_, "solver", "simplex");
    for (std::size_t p = 0; p < fixture_.nnz; ++p)
      indices[p] = static_cast<int>(fixture_.indices[p]);
    std::vector<double> cost(m, 0.0), col_lower(m, -1e30),
        col_upper(m, 1e30);
    status = Highs_passLp(
        selector_, m, n, static_cast<int>(fixture_.nnz), 2, 1, 0.0,
        cost.data(), col_lower.data(), col_upper.data(), lower.data(),
        upper.data(), starts.data(), indices.data(), fixture_.values.data());
  } else {
    status = Highs_changeRowsBoundsByRange(selector_, 0, n - 1,
                                           lower.data(), upper.data());
    // Bound changes preserve the matrix.  Reassert the last admitted basis
    // explicitly so HiGHS performs a warm feasibility repair rather than a
    // cold crash.  This is the fixed-t simplex artifact the walker retains.
    if (status == 0
        && selector_col_status_.size() == fixture_.m
        && selector_row_status_.size() == fixture_.n)
      status = Highs_setBasis(selector_, selector_col_status_.data(),
                              selector_row_status_.data());
  }
  const auto begin = std::chrono::steady_clock::now();
  // HighsStatus::kWarning is still a usable candidate-generator result.  It
  // is the original-data audit below, not the API status, that admits it.
  if (status >= 0) status = Highs_run(selector_);
  selector_ms_ += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
  ++selector_calls_;
  if (status < 0) return decline("api", status);
  const int model_status = Highs_getModelStatus(selector_);
  if (model_status != 7) return decline("model_status", model_status);
  std::vector<double> u(m), product(n, 0.0), product_abs(n, 0.0);
  const int solution_status =
      Highs_getSolution(selector_, u.data(), nullptr, nullptr, nullptr);
  if (solution_status < 0) return decline("solution", solution_status);

  // HiGHS is a candidate generator.  Acceptance remains an original-data,
  // componentwise certificate and never trusts solver status alone.
  for (int row = 0; row < n; ++row) {
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const double term = fixture_.values[p] * u[fixture_.indices[p]];
      product[row] += term;
      product_abs[row] += std::abs(term);
    }
    const double scale = 1.0 + product_abs[row] + std::abs(offset[row])
                         + (support[row] ? std::abs(active_value[row]) : 0.0);
    if (support[row]) {
      const double rhs = active_value[row] - offset[row];
      if (std::abs(product[row] - rhs) / scale > kTolerance)
        return decline("active_audit", row);
    } else if (selector_constraints[row]
               && std::max(0.0, product[row] - upper[row]) / scale
                      > kTolerance) {
      return decline("inactive_audit", row);
    }
  }

  std::vector<int> col_status(m), row_status(n), basic_variables(n);
  const int get_basis_status =
      Highs_getBasis(selector_, col_status.data(), row_status.data());
  const int get_basic_variables_status =
      Highs_getBasicVariables(selector_, basic_variables.data());
  if (get_basis_status < 0 || get_basic_variables_status < 0)
    return decline("basis_extract",
                   std::min(get_basis_status, get_basic_variables_status));
  std::uint64_t fingerprint = 1469598103934665603ULL;
  auto append_fingerprint = [&](std::uint64_t value) {
    fingerprint ^= value + 0x9e3779b97f4a7c15ULL
                   + (fingerprint << 6) + (fingerprint >> 2);
  };
  for (int value : col_status)
    append_fingerprint(static_cast<std::uint64_t>(value + 8));
  for (int value : row_status)
    append_fingerprint(static_cast<std::uint64_t>(value + 16));
  for (int value : basic_variables)
    append_fingerprint(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(value)
        - std::numeric_limits<std::int32_t>::min()));
  selector_previous_basis_fingerprint_ = selector_basis_fingerprint_;
  selector_basis_fingerprint_ = fingerprint;
  selector_basis_repeats_ +=
      selector_basis_snapshots_ > 0
      && selector_previous_basis_fingerprint_ == fingerprint;
  ++selector_basis_snapshots_;
  selector_col_status_ = std::move(col_status);
  selector_row_status_ = std::move(row_status);
  selector_basic_variables_ = std::move(basic_variables);
  if (std::getenv("TWALKER_SIMPLEX_BASIS_TRACE")) {
    int iterations = -1;
    Highs_getIntInfoValue(selector_, "simplex_iteration_count", &iterations);
    std::cerr << "SIMPLEX_BASIS fingerprint=" << fingerprint
              << " repeat="
              << (selector_previous_basis_fingerprint_ == fingerprint)
              << " iterations=" << iterations << '\n';
  }
  if (selected_u) *selected_u = std::move(u);
  if (selected_product) *selected_product = std::move(product);
  if (basis_fingerprint) *basis_fingerprint = fingerprint;
  return true;
}

bool Walker::selector_maximize_forward_interval(
    const std::vector<std::uint8_t> &support,
    const std::vector<double> &endpoint_y,
    const std::vector<double> &endpoint_slack,
    const std::vector<double> &direction, double minimum_forward,
    double delta_cap, std::vector<double> &selected_ua,
    std::vector<double> &selected_bua, double &selected_delta,
    std::uint64_t *basis_fingerprint) {
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  const bool trace =
      std::getenv("TWALKER_SIMPLEX_BASIS_TRACE") != nullptr;
  auto decline = [&](const char *reason, int detail = 0) {
    if (trace)
      std::cerr << "SIMPLEX_HORIZON declined=" << reason
                << " detail=" << detail << '\n';
    return false;
  };
  if (support.size() != fixture_.n || endpoint_y.size() != fixture_.n
      || endpoint_slack.size() != fixture_.n
      || direction.size() != fixture_.n
      || !(minimum_forward >= 0.0) || !std::isfinite(minimum_forward)
      || !(delta_cap > minimum_forward) || !std::isfinite(delta_cap))
    return decline("input");

  // Active coordinates impose their own immutable geometric horizon.  The
  // simplex selector may postpone multiplier-boundary events, but never walk
  // past a coordinate of the unique projection direction reaching zero.
  double upper_delta = delta_cap;
  for (int row = 0; row < n; ++row) {
    if (!support[row] || direction[row] >= -1e-13) continue;
    const double distance = -endpoint_y[row] / direction[row];
    if (std::isfinite(distance) && distance > minimum_forward)
      upper_delta = std::min(upper_delta, distance);
  }
  if (!(upper_delta > minimum_forward)) return decline("no_interval");

  std::vector<double> row_lower(n, -1e30), row_upper(n, 1e30);
  std::vector<double> delta_coefficient(n);
  for (int row = 0; row < n; ++row) {
    if (support[row]) {
      row_lower[row] = row_upper[row] = 0.0;
      delta_coefficient[row] = fixture_.b[row] - direction[row];
    } else {
      // q_i(t0) + B_i*w + Delta*b_i <= 0, with w=Delta*ua.
      row_upper[row] = -std::min(0.0, endpoint_slack[row]);
      delta_coefficient[row] = fixture_.b[row];
    }
  }

  int status = 0;
  if (!horizon_selector_) {
    horizon_selector_ = Highs_create();
    if (!horizon_selector_) return decline("create");
    Highs_setBoolOptionValue(horizon_selector_, "output_flag", 0);
    Highs_setIntOptionValue(horizon_selector_, "threads", 1);
    Highs_setStringOptionValue(horizon_selector_, "presolve", "off");
    Highs_setStringOptionValue(horizon_selector_, "solver", "simplex");

    std::vector<int> starts(n + 1), indices(fixture_.nnz + fixture_.n);
    std::vector<double> values(fixture_.nnz + fixture_.n);
    std::size_t cursor = 0;
    for (int row = 0; row < n; ++row) {
      starts[row] = static_cast<int>(cursor);
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1];
           ++p) {
        indices[cursor] = static_cast<int>(fixture_.indices[p]);
        values[cursor++] = fixture_.values[p];
      }
      indices[cursor] = m;
      values[cursor++] = delta_coefficient[row];
    }
    starts[n] = static_cast<int>(cursor);
    std::vector<double> cost(m + 1, 0.0), col_lower(m + 1, -1e30),
        col_upper(m + 1, 1e30);
    cost[m] = -1.0;
    col_lower[m] = 0.0;
    col_upper[m] = upper_delta;
    status = Highs_passLp(
        horizon_selector_, m + 1, n, static_cast<int>(cursor), 2, 1, 0.0,
        cost.data(), col_lower.data(), col_upper.data(), row_lower.data(),
        row_upper.data(), starts.data(), indices.data(), values.data());
  } else {
    status = Highs_changeRowsBoundsByRange(
        horizon_selector_, 0, n - 1, row_lower.data(), row_upper.data());
    double delta_lower = 0.0;
    if (status >= 0)
      status = Highs_changeColsBoundsByRange(
          horizon_selector_, m, m, &delta_lower, &upper_delta);
    for (int row = 0; row < n && status >= 0; ++row)
      status = Highs_changeCoeff(horizon_selector_, row, m,
                                 delta_coefficient[row]);
    if (status >= 0
        && horizon_col_status_.size() == fixture_.m + 1
        && horizon_row_status_.size() == fixture_.n)
      status = Highs_setBasis(horizon_selector_, horizon_col_status_.data(),
                              horizon_row_status_.data());
  }

  const auto begin = std::chrono::steady_clock::now();
  if (status >= 0) status = Highs_run(horizon_selector_);
  selector_ms_ += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
  ++selector_calls_;
  if (status < 0) return decline("api", status);
  const int model_status = Highs_getModelStatus(horizon_selector_);
  if (model_status != 7) return decline("model_status", model_status);

  std::vector<double> solution(m + 1);
  const int solution_status = Highs_getSolution(
      horizon_selector_, solution.data(), nullptr, nullptr, nullptr);
  if (solution_status < 0) return decline("solution", solution_status);
  const double delta = solution[m];
  if (!(delta > minimum_forward) || !std::isfinite(delta))
    return decline("delta");

  selected_ua.resize(m);
  for (int column = 0; column < m; ++column)
    selected_ua[column] = solution[column] / delta;
  products(fixture_, selected_ua, selected_bua, nullptr);

  double active_error = 0.0, forward_violation = 0.0;
  int active_row = -1, forward_row = -1;
  for (int row = 0; row < n; ++row) {
    if (support[row]) {
      const double rhs = direction[row] - fixture_.b[row];
      const double scale = 1.0 + std::abs(rhs)
                           + std::abs(selected_bua[row]);
      const double error = std::abs(selected_bua[row] - rhs) / scale;
      if (error > active_error) {
        active_error = error;
        active_row = row;
      }
    } else {
      const double at_horizon = endpoint_slack[row]
          + delta * (fixture_.b[row] + selected_bua[row]);
      const double scale = 1.0 + std::abs(endpoint_slack[row])
                           + std::abs(delta * fixture_.b[row])
                           + std::abs(delta * selected_bua[row]);
      const double violation = std::max(0.0, at_horizon) / scale;
      if (violation > forward_violation) {
        forward_violation = violation;
        forward_row = row;
      }
    }
  }
  if (active_error > kTolerance)
    return decline("active_audit", active_row);
  if (forward_violation > kTolerance)
    return decline("forward_audit", forward_row);

  std::vector<int> col_status(m + 1), row_status(n), basic_variables(n);
  const int basis_status = Highs_getBasis(
      horizon_selector_, col_status.data(), row_status.data());
  const int variables_status = Highs_getBasicVariables(
      horizon_selector_, basic_variables.data());
  if (basis_status < 0 || variables_status < 0)
    return decline("basis_extract", std::min(basis_status, variables_status));
  std::uint64_t fingerprint = 1469598103934665603ULL;
  auto append = [&](std::uint64_t value) {
    fingerprint ^= value + 0x9e3779b97f4a7c15ULL
                   + (fingerprint << 6) + (fingerprint >> 2);
  };
  for (int value : col_status)
    append(static_cast<std::uint64_t>(value + 8));
  for (int value : row_status)
    append(static_cast<std::uint64_t>(value + 16));
  for (int value : basic_variables)
    append(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(value)
        - std::numeric_limits<std::int32_t>::min()));
  horizon_previous_basis_fingerprint_ = horizon_basis_fingerprint_;
  horizon_basis_fingerprint_ = fingerprint;
  horizon_basis_repeats_ +=
      horizon_basis_snapshots_ > 0
      && horizon_previous_basis_fingerprint_ == fingerprint;
  ++horizon_basis_snapshots_;
  horizon_col_status_ = std::move(col_status);
  horizon_row_status_ = std::move(row_status);
  horizon_basic_variables_ = std::move(basic_variables);
  selected_delta = delta;
  if (basis_fingerprint) *basis_fingerprint = fingerprint;
  if (trace) {
    int iterations = -1;
    Highs_getIntInfoValue(horizon_selector_, "simplex_iteration_count",
                          &iterations);
    std::cerr << "SIMPLEX_HORIZON delta=" << std::setprecision(17) << delta
              << " cap=" << upper_delta
              << " active=" << active_error
              << " forward=" << forward_violation
              << " fingerprint=" << fingerprint
              << " repeat="
              << (horizon_previous_basis_fingerprint_ == fingerprint)
              << " iterations=" << iterations << '\n';
  }
  return true;
}

bool Walker::accept(const std::vector<std::uint8_t> &support, double t,
                    const FaceSolution &solution) {
  last_reject_.clear();
  if (solution.dres > kTolerance) {
    last_reject_ = "dual residual";
    return false;
  }
  const auto &rows = solution.rows;
  std::vector<double> y0(rows.size());
  double y_max = 0.0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    y0[local] = std::fma(t, solution.g[local], solution.h[local]);
    y_max = std::max(y_max, std::abs(y0[local]));
  }
  const double y_gate = -1e-8 * std::max(1.0, y_max);
  if (std::any_of(y0.begin(), y0.end(),
                  [&](double value) { return value < y_gate; })) {
    last_reject_ = "negative face point";
    return false;
  }

  std::vector<double> u0(fixture_.m), product0(fixture_.n), abs0, abs_a;
  for (std::size_t j = 0; j < fixture_.m; ++j)
    u0[j] = std::fma(t, solution.ua[j], solution.uc[j]);
  // Only the componentwise error scales require fresh absolute products.
  // The signed products are exact linear combinations of the cached face
  // artifacts and need no further sparse matrix passes.
  two_absolute_products(fixture_, u0, solution.ua, abs0, abs_a);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    product0[row] = std::fma(t, solution.bua[row], solution.buc[row]);
  double eq0 = 0.0, eqa = 0.0, off0 = 0.0, nonnegative0 = 0.0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    // Do not recover this affine difference as y(t)-t*b-s: on difficult
    // faces both terms can be large while their difference is modest.  The
    // expanded form is algebraically identical and avoids the cancellation
    // that previously manufactured endpoint errors just above the gate.
    const double rhs0 = std::fma(
        t, solution.g[local] - fixture_.b[row],
        solution.h[local] - target_shift_[row]);
    const double rhsa = solution.g[local] - fixture_.b[row];
    eq0 = std::max(eq0, std::abs(product0[row] - rhs0)
                            / (1.0 + std::abs(rhs0) + abs0[row]));
    eqa = std::max(eqa, std::abs(solution.bua[row] - rhsa)
                            / (1.0 + std::abs(rhsa) + abs_a[row]));
    nonnegative0 = std::max(nonnegative0,
                            std::max(0.0, -y0[local] / (1.0 + std::abs(y0[local]))));
  }
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    if (support[row]) continue;
    const double q = t * fixture_.b[row] + target_shift_[row]
                     + product0[row];
    off0 = std::max(off0, std::max(0.0, q / (1.0 + std::abs(t * fixture_.b[row])
                                                   + std::abs(target_shift_[row])
                                                   + abs0[row])));
  }
  double objective = 0.0, invariant = 0.0;
  for (std::size_t j = 0; j < fixture_.m; ++j)
    objective -= fixture_.d[j] * solution.ua[j];
  double slope_constant_cross = 0.0;
  bool shifted = false;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    invariant += solution.h[local] * fixture_.b[rows[local]];
    slope_constant_cross += solution.h[local] * solution.g[local];
    shifted = shifted || target_shift_[rows[local]] != 0.0;
  }
  if (shifted) invariant -= slope_constant_cross;
  const double invariant_error = std::abs(objective - invariant)
      / std::max({1.0, std::abs(objective), std::abs(invariant)});

  const double uncorrectable = std::max(
      {eqa, nonnegative0, invariant_error, solution.dres});
  if (uncorrectable > kTolerance) {
    std::ostringstream text;
    text << "uncorrectable endpoint gate=" << uncorrectable << " eq0=" << eq0
         << " eqa=" << eqa << " nonnegative=" << nonnegative0
         << " invariant=" << invariant_error << " dres=" << solution.dres;
    last_reject_ = text.str();
    return false;
  }
  // Endpoint equality and off-face residuals are both properties of the
  // combined affine point.  Admit a narrow endpoint envelope only when the
  // underlying affine slope identity is an order of magnitude tighter than
  // the certificate tolerance; otherwise require the original sparse LP
  // selector to certify that same face point.  Slope, nonnegativity,
  // invariant, and dual residual failures remain uncorrectable here.
  const bool bounded_affine_endpoint =
      off0 <= kTolerance && eq0 <= 2.0 * kTolerance
      && eqa <= 0.1 * kTolerance;
  const bool corrected_at_t = bounded_affine_endpoint
                              || selector_feasible(support, t, y0);
  if (bounded_affine_endpoint && eq0 > kTolerance
      && std::getenv("TWALKER_REPAIR_TRACE"))
    std::cerr << "ACCEPT bounded_affine_endpoint t=" << std::setprecision(17)
              << t << " eq=" << eq0 << " slope=" << eqa
              << " off=" << off0 << '\n';
  if (!corrected_at_t) {
    std::ostringstream text;
    text << "endpoint infeasible at t; equality=" << eq0
         << " off-face=" << off0;
    last_reject_ = text.str();
    return false;
  }

  for (double relative : {1e-6, 1e-8, 1e-10}) {
    const double delta = relative * std::max(1.0, std::abs(t));
    const double tp = t + delta;
    double yp_nonnegative = 0.0, offp = 0.0;
    std::vector<double> product_p(fixture_.n), abs_p;
    affine_absolute_product(fixture_, u0, solution.ua, delta, abs_p);
    for (std::size_t row = 0; row < fixture_.n; ++row)
      product_p[row] = product0[row] + delta * solution.bua[row];
    for (std::size_t local = 0; local < rows.size(); ++local) {
      const double yp = tp * solution.g[local] + solution.h[local];
      yp_nonnegative = std::max(
          yp_nonnegative, std::max(0.0, -yp / (1.0 + std::abs(yp))));
    }
    if (yp_nonnegative > kTolerance) continue;
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      if (support[row]) continue;
      const double q = tp * fixture_.b[row] + target_shift_[row]
                       + product_p[row];
      offp = std::max(offp, std::max(0.0, q / (1.0 + std::abs(tp * fixture_.b[row])
                                                     + std::abs(target_shift_[row])
                                                     + abs_p[row])));
    }
    const double base_error = std::max({uncorrectable, yp_nonnegative});
    if (base_error > kTolerance) continue;
    std::vector<double> yp(rows.size());
    for (std::size_t local = 0; local < rows.size(); ++local)
      yp[local] = tp * solution.g[local] + solution.h[local];
    if (offp <= kTolerance || selector_feasible(support, tp, yp))
      return true;
  }
  last_reject_ = "no forward-feasible probe";
  return false;
}

bool Walker::settle(std::vector<std::uint8_t> &support, double t,
                    FaceSolution &solution, int rounds, bool one_at_a_time) {
  // Most pivots are already settled after the event toggle.  Delay exact
  // support-history copies until a settle round actually changes the face.
  std::vector<std::vector<std::uint8_t>> seen;
  for (int round = 0; round < rounds; ++round) {
    ++settle_rounds_;
    if (!std::any_of(support.begin(), support.end(), [](auto v) { return v; })) {
      last_reject_ = "settle empty support";
      return false;
    }
    if (std::find(seen.begin(), seen.end(), support) != seen.end()) {
      last_reject_ = "settle support cycle";
      return false;
    }
    try {
      solution = solve(support);
    } catch (const FaceDecline &error) {
      last_reject_ = "settle face solve declined: " + std::string(error.what());
      return false;
    }
    double y_max = 0.0, q_max = 0.0;
    std::vector<double> y, q;
    auto evaluate = [&]() {
      y_max = q_max = 0.0;
      y.resize(solution.rows.size());
      q.resize(fixture_.n);
      for (std::size_t local = 0; local < solution.rows.size(); ++local) {
        y[local] = t * solution.g[local] + solution.h[local];
        y_max = std::max(y_max, std::abs(y[local]));
      }
      for (std::size_t row = 0; row < fixture_.n; ++row) {
        q[row] = t * (fixture_.b[row] + solution.bua[row])
                 + target_shift_[row] + solution.buc[row];
        q_max = std::max(q_max, std::abs(q[row]));
      }
    };
    evaluate();
    double y_scale = 1e-8 * std::max(1.0, y_max);
    double q_scale = 1e-8 * std::max(1.0, q_max);
    // The scale-relative support threshold may be looser than the unchanged
    // componentwise endpoint certificate when the active multipliers span a
    // wide dynamic range.  In that case settle used to report "unchanged"
    // only for accept() to reject the same face as nonnegative-infeasible.
    // Treat either violation as a required drop.  This does not tighten the
    // certificate or perturb an accepted path: it only makes settle repair a
    // face that the existing endpoint gate cannot accept.
    auto active_drop_severity = [&](double value) {
      const double support_severity = -value / y_scale;
      const double endpoint_severity =
          std::max(0.0, -value / (1.0 + std::abs(value))) / kTolerance;
      return std::max(support_severity, endpoint_severity);
    };
    const bool initially_extended = solution.used_extended_gram;
    if (std::getenv("TWALKER_REVISED_AFFINE_AUDIT")
        && solution.affine_bound_valid
        && solution.audit_oracle_g.size() == solution.g.size()
        && solution.audit_oracle_h.size() == solution.h.size()
        && solution.audit_oracle_bua.size() == fixture_.n
        && solution.audit_oracle_buc.size() == fixture_.n) {
      std::vector<double> affine_bound;
      if (affine_product_bounds(solution, t, affine_bound)) {
        double worst_ratio = 0.0, worst_error = 0.0, worst_width = 0.0;
        for (std::size_t local = 0; local < solution.g.size(); ++local) {
          const double actual = std::abs(
              t * (solution.g[local] - solution.audit_oracle_g[local])
              + solution.h[local] - solution.audit_oracle_h[local]);
          worst_error = std::max(worst_error, actual);
          worst_width = std::max(worst_width,
                                 affine_bound[solution.rows[local]]);
          if (affine_bound[solution.rows[local]] > 0.0)
            worst_ratio = std::max(
                worst_ratio,
                actual / affine_bound[solution.rows[local]]);
        }
        for (std::size_t row = 0; row < fixture_.n; ++row) {
          const double actual = std::abs(
              t * (solution.bua[row] - solution.audit_oracle_bua[row])
              + solution.buc[row] - solution.audit_oracle_buc[row]);
          worst_error = std::max(worst_error, actual);
          worst_width = std::max(worst_width, affine_bound[row]);
          if (affine_bound[row] > 0.0)
            worst_ratio = std::max(worst_ratio, actual / affine_bound[row]);
        }
        std::cerr << "affine audit t=" << std::setprecision(17) << t
                  << " actual=" << worst_error
                  << " width=" << worst_width
                  << " ratio=" << worst_ratio << '\n';
      }
    }
    const bool stable = settle_decision_stable(
        support, t, solution, y, q, y_scale, q_scale, one_at_a_time);
    if (!stable) {
      try {
        solution = solve_direct(support);
      } catch (const FaceDecline &error) {
        last_reject_ = "settle stability refactor declined: "
                       + std::string(error.what());
        return false;
      }
      ++stability_refactors_;
      ++settle_stability_refactors_;
      evaluate();
      y_scale = 1e-8 * std::max(1.0, y_max);
      q_scale = 1e-8 * std::max(1.0, q_max);
    }
    const auto &rows = solution.rows;
    bool changed = false;
    std::size_t changed_row = fixture_.n;
    std::uint8_t changed_value = 0;
    std::vector<std::size_t> drop_rows, add_rows;
    if (one_at_a_time) {
      std::size_t drop_row = fixture_.n, add_row = fixture_.n;
      double drop_severity = 0.0, add_severity = 0.0;
      for (std::size_t local = 0; local < rows.size(); ++local) {
        const double severity = active_drop_severity(y[local]);
        if (severity > 1.0 && severity > drop_severity) {
          drop_severity = severity;
          drop_row = rows[local];
        }
      }
      for (std::size_t row = 0; row < fixture_.n; ++row) {
        if (support[row]) continue;
        const double severity = q[row] / q_scale;
        if (severity > 1.0 && severity > add_severity) {
          add_severity = severity;
          add_row = row;
        }
      }
      if (drop_severity >= add_severity && drop_row < fixture_.n) {
        changed_row = drop_row;
        changed_value = 0;
        changed = true;
      } else if (add_row < fixture_.n) {
        changed_row = add_row;
        changed_value = 1;
        changed = true;
      }
    } else {
      for (std::size_t local = 0; local < rows.size(); ++local)
        if (active_drop_severity(y[local]) > 1.0)
          drop_rows.push_back(rows[local]);
      if (drop_rows.empty()) {
        for (std::size_t row = 0; row < fixture_.n; ++row)
          if (!support[row] && q[row] > q_scale) add_rows.push_back(row);
      } else {
        // Preserve the original drop-then-add batch semantics exactly: a row
        // dropped above is an off-face candidate in the subsequent add scan.
        std::vector<std::uint8_t> will_drop(fixture_.n, 0);
        for (auto row : drop_rows) will_drop[row] = 1;
        for (std::size_t row = 0; row < fixture_.n; ++row)
          if ((!support[row] || will_drop[row]) && q[row] > q_scale)
            add_rows.push_back(row);
      }
      changed = !drop_rows.empty() || !add_rows.empty();
    }
    if (solution.qr_update_audit_candidate) {
      auto authoritative_next = support;
      if (changed) {
        if (one_at_a_time) {
          authoritative_next[changed_row] = changed_value;
        } else {
          for (auto row : drop_rows) authoritative_next[row] = 0;
          for (auto row : add_rows) authoritative_next[row] = 1;
        }
      }
      audit_qr_settle_decision(support, t, solution, one_at_a_time,
                               authoritative_next);
    }
    if (trace_path_)
      std::cerr << "SETTLE t=" << std::setprecision(17) << t
                << " round=" << round << " rows=" << rows.size()
                << " initially_extended=" << initially_extended
                << " stable=" << stable << " changed=" << changed
                << " drops=" << drop_rows.size()
                << " adds=" << add_rows.size() << '\n';
    if (changed) {
      seen.push_back(support);
      if (one_at_a_time) {
        support[changed_row] = changed_value;
      } else {
        for (auto row : drop_rows) support[row] = 0;
        for (auto row : add_rows) support[row] = 1;
      }
    }
    if (!changed) {
      bool accepted = accept(support, t, solution);
      if (!accepted
          && ((extension_enabled_ && solution.used_maintained_gram)
              || solution.used_bound_core)) {
        try {
          solution = solve_direct(support);
        } catch (const FaceDecline &error) {
          last_reject_ = "endpoint stability refactor declined: "
                         + std::string(error.what());
          return false;
        }
        ++stability_refactors_;
        ++settle_stability_refactors_;
        accepted = accept(support, t, solution);
      }
      if (trace_path_)
        std::cerr << "ACCEPT t=" << std::setprecision(17) << t
                  << " extended=" << solution.used_extended_gram
                  << " accepted=" << accepted
                  << " reason=" << last_reject_ << '\n';
      if (accepted)
        audit_objective_preserving_rank_lift(t, solution);
      if (accepted)
        apply_objective_preserving_rank_lift(support, t, solution);
      return accepted;
    }
  }
  last_reject_ = "settle round cap";
  return false;
}

CertificateErrors Walker::certificate_pair(const std::vector<double> &x,
                                           const std::vector<double> &y) const {
  CertificateErrors errors;
  std::vector<double> bx, abs_bx;
  products(fixture_, x, bx, &abs_bx);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    errors.primal = std::max(
        errors.primal, std::max(0.0, fixture_.b[row] - bx[row])
                           / (1.0 + std::abs(fixture_.b[row]) + abs_bx[row]));
  std::vector<double> dual(fixture_.m, 0.0), dual_scale(fixture_.m, 0.0);
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    errors.nonnegative = std::max(errors.nonnegative, -y[row]);
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      dual[column] += fixture_.values[p] * y[row];
      dual_scale[column] += std::abs(fixture_.values[p]) * std::abs(y[row]);
    }
  }
  errors.nonnegative = std::max(0.0, errors.nonnegative)
                       / (1.0 + inf_norm(y));
  for (std::size_t column = 0; column < fixture_.m; ++column)
    errors.dual = std::max(
        errors.dual, std::abs(dual[column] - fixture_.d[column])
                         / (1.0 + std::abs(fixture_.d[column])
                            + dual_scale[column]));
  const double primal_objective = dot(fixture_.d, x);
  const double dual_objective = dot(fixture_.b, y);
  errors.gap = std::abs(primal_objective - dual_objective)
               / (1.0 + std::abs(primal_objective)
                  + std::abs(dual_objective));
  return errors;
}

bool Walker::certificate_passes(const CertificateErrors &errors) const {
  return std::max({errors.primal, errors.dual, errors.nonnegative, errors.gap})
         <= kTolerance;
}

bool Walker::primal_face_candidate(const std::vector<double> &y,
                                   std::vector<double> &x) {
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  const double threshold = kDualSupport * std::max(1.0, inf_norm(y));
  std::vector<double> lower = fixture_.b, upper(n, 1e30);
  for (int row = 0; row < n; ++row)
    if (y[row] > threshold) upper[row] = lower[row];

  int status = 0;
  if (!primal_selector_) {
    primal_selector_ = Highs_create();
    if (!primal_selector_) return false;
    Highs_setBoolOptionValue(primal_selector_, "output_flag", 0);
    Highs_setIntOptionValue(primal_selector_, "threads", 1);
    Highs_setStringOptionValue(primal_selector_, "presolve", "off");
    Highs_setStringOptionValue(primal_selector_, "solver", "simplex");
    std::vector<int> starts(n + 1), indices(fixture_.nnz);
    for (int row = 0; row <= n; ++row)
      starts[row] = static_cast<int>(fixture_.indptr[row]);
    for (std::size_t p = 0; p < fixture_.nnz; ++p)
      indices[p] = static_cast<int>(fixture_.indices[p]);
    std::vector<double> cost(m, 0.0), col_lower(m, -1e30),
        col_upper(m, 1e30);
    status = Highs_passLp(
        primal_selector_, m, n, static_cast<int>(fixture_.nnz), 2, 1, 0.0,
        cost.data(), col_lower.data(), col_upper.data(), lower.data(),
        upper.data(), starts.data(), indices.data(), fixture_.values.data());
  } else {
    status = Highs_changeRowsBoundsByRange(primal_selector_, 0, n - 1,
                                           lower.data(), upper.data());
  }
  if (status == 0) status = Highs_run(primal_selector_);
  if (status != 0 || Highs_getModelStatus(primal_selector_) != 7) return false;
  x.resize(m);
  return Highs_getSolution(primal_selector_, x.data(), nullptr, nullptr,
                           nullptr) == 0;
}

bool Walker::recover_certificate(const std::vector<double> &y,
                                 std::vector<double> &x,
                                 CertificateErrors &errors) {
  const auto recovery_begin = std::chrono::steady_clock::now();
  ++recovery_calls_;
  const bool candidate = primal_face_candidate(y, x);
  if (candidate) errors = certificate_pair(x, y);
  const bool passes = candidate && certificate_passes(errors);
  recovery_ms_ += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - recovery_begin)
                      .count();
  return passes;
}

bool Walker::terminal_support_repair(
    const std::vector<std::uint8_t> &support, std::vector<double> &y,
    std::vector<double> &x, CertificateErrors &errors) {
  const auto repair_begin = std::chrono::steady_clock::now();
  ++terminal_support_repairs_;
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  bool candidate = false;
  int status = 0;

  std::vector<double> lower(n, 0.0), upper(n, 0.0);
  for (int row = 0; row < n; ++row)
    if (support[row]) upper[row] = 1e30;

  if (!terminal_support_selector_) {
    terminal_support_selector_ = Highs_create();
    if (terminal_support_selector_) {
      Highs_setBoolOptionValue(terminal_support_selector_, "output_flag", 0);
      Highs_setIntOptionValue(terminal_support_selector_, "threads", 1);
      Highs_setStringOptionValue(terminal_support_selector_, "presolve",
                                 "off");
      Highs_setStringOptionValue(terminal_support_selector_, "solver",
                                 "simplex");
      std::vector<int> starts(n + 1), indices(fixture_.nnz);
      for (int row = 0; row <= n; ++row)
        starts[row] = static_cast<int>(fixture_.indptr[row]);
      for (std::size_t p = 0; p < fixture_.nnz; ++p)
        indices[p] = static_cast<int>(fixture_.indices[p]);
      std::vector<double> cost(n), equality_lower = fixture_.d,
          equality_upper = fixture_.d;
      for (int row = 0; row < n; ++row) cost[row] = -fixture_.b[row];
      // B is stored by original row.  In the dual LP each such row is one
      // column of B', so the fixture's CSR arrays are already a columnwise
      // representation of the equality matrix B'.
      status = Highs_passLp(
          terminal_support_selector_, n, m,
          static_cast<int>(fixture_.nnz), 1, 1, 0.0, cost.data(),
          lower.data(), upper.data(), equality_lower.data(),
          equality_upper.data(), starts.data(), indices.data(),
          fixture_.values.data());
    } else {
      status = -1;
    }
  } else {
    status = Highs_changeColsBoundsByRange(terminal_support_selector_, 0,
                                           n - 1, lower.data(), upper.data());
    if (status == 0
        && terminal_support_col_status_.size() == fixture_.n
        && terminal_support_row_status_.size() == fixture_.m)
      status = Highs_setBasis(terminal_support_selector_,
                              terminal_support_col_status_.data(),
                              terminal_support_row_status_.data());
  }

  if (status == 0) status = Highs_run(terminal_support_selector_);
  int iterations = 0;
  if (terminal_support_selector_)
    Highs_getIntInfoValue(terminal_support_selector_,
                          "simplex_iteration_count", &iterations);
  terminal_support_repair_iterations_ += iterations;
  if (status == 0 && Highs_getModelStatus(terminal_support_selector_) == 7) {
    y.resize(n);
    candidate = Highs_getSolution(terminal_support_selector_, y.data(),
                                  nullptr, nullptr, nullptr) == 0;
    if (candidate) {
      terminal_support_col_status_.resize(n);
      terminal_support_row_status_.resize(m);
      Highs_getBasis(terminal_support_selector_,
                     terminal_support_col_status_.data(),
                     terminal_support_row_status_.data());
      candidate = primal_face_candidate(y, x);
      if (candidate) errors = certificate_pair(x, y);
      candidate = candidate && certificate_passes(errors);
    }
  }
  if (candidate) ++terminal_support_repair_successes_;
  terminal_support_repair_ms_ +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - repair_begin)
          .count();
  return candidate;
}

std::vector<std::uint8_t> Walker::corrector_support(double t) const {
  const int m = static_cast<int>(fixture_.m);
  std::vector<double> u(m, 0.0), next(m), residual(fixture_.n);
  for (int iteration = 0; iteration < 300; ++iteration) {
    std::vector<std::uint32_t> negative;
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      double value = -t * fixture_.b[row] - target_shift_[row];
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        value += fixture_.values[p] * u[fixture_.indices[p]];
      residual[row] = value;
      if (value < 0.0) negative.push_back(static_cast<std::uint32_t>(row));
    }
    if (negative.empty()) break;
    std::vector<double> gram(static_cast<std::size_t>(m) * m, 0.0);
    std::vector<double> gradient = fixture_.d;
    for (int diagonal = 0; diagonal < m; ++diagonal)
      gram[diagonal + static_cast<std::size_t>(m) * diagonal] = 1e-6;
    for (auto row : negative) {
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto left = static_cast<int>(fixture_.indices[p]);
        const double left_value = fixture_.values[p];
        gradient[left] += left_value * residual[row];
        for (auto q = p; q < fixture_.indptr[row + 1]; ++q) {
          int first = left;
          int second = static_cast<int>(fixture_.indices[q]);
          if (first > second) std::swap(first, second);
          gram[first + static_cast<std::size_t>(m) * second] +=
              left_value * fixture_.values[q];
        }
      }
    }
    const char upper = 'U';
    const int one = 1;
    int info = 0;
    dposv_(&upper, &m, &one, gram.data(), &m, gradient.data(), &m, &info);
    if (info != 0) break;
    double change = 0.0;
    for (int column = 0; column < m; ++column) {
      next[column] = u[column] - gradient[column];
      change = std::max(change, std::abs(next[column] - u[column]));
    }
    u.swap(next);
    if (change <= 1e-14 * std::max(1.0, inf_norm(u))) break;
  }
  std::vector<double> y(fixture_.n);
  double scale = 1.0;
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    double value = t * fixture_.b[row] + target_shift_[row];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      value -= fixture_.values[p] * u[fixture_.indices[p]];
    y[row] = std::max(0.0, value);
    scale = std::max(scale, y[row]);
  }
  std::vector<std::uint8_t> support(fixture_.n, 0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    support[row] = y[row] > 1e-9 * scale;
  return support;
}

std::vector<double> Walker::qp_projection(double t) {
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  const int constraints = m + n;

  // A = [B^T; I] in CSC.  B is stored row-wise, so each original row is
  // already one complete column of A and no transpose or sort is needed.
  std::vector<OSQPInt> a_starts(n + 1), a_indices(fixture_.nnz + n);
  std::vector<OSQPFloat> a_values(fixture_.nnz + n);
  std::size_t cursor = 0;
  for (int column = 0; column < n; ++column) {
    a_starts[column] = static_cast<OSQPInt>(cursor);
    for (auto p = fixture_.indptr[column]; p < fixture_.indptr[column + 1];
         ++p) {
      a_indices[cursor] = static_cast<OSQPInt>(fixture_.indices[p]);
      a_values[cursor] = fixture_.values[p];
      ++cursor;
    }
    a_indices[cursor] = static_cast<OSQPInt>(m + column);
    a_values[cursor] = 1.0;
    ++cursor;
  }
  a_starts[n] = static_cast<OSQPInt>(cursor);

  const double variable_scale = t >= 100.0 ? t : 1.0;
  std::vector<OSQPInt> p_starts(n + 1), p_indices(n);
  std::vector<OSQPFloat> p_values(n, 1.0), cost(n);
  for (int column = 0; column < n; ++column) {
    p_starts[column] = column;
    p_indices[column] = column;
    // At large t, y = t z makes the QP min 1/2 ||z-b||^2,
    // B^T z=d/t, z>=0.  Use the original scale near the origin, where the
    // unscaled equality right-hand side gives OSQP a stronger stopping test.
    cost[column] = -(t * fixture_.b[column] + target_shift_[column])
                   / variable_scale;
  }
  p_starts[n] = n;

  std::vector<OSQPFloat> lower(constraints), upper(constraints);
  for (int row = 0; row < m; ++row)
    lower[row] = upper[row] = fixture_.d[row] / variable_scale;
  for (int row = 0; row < n; ++row) {
    lower[m + row] = 0.0;
    upper[m + row] = OSQP_INFTY;
  }

  OSQPCscMatrix P{n, n, p_starts.data(), p_indices.data(), p_values.data(), n,
                  -1, 0};
  OSQPCscMatrix A{constraints, n, a_starts.data(), a_indices.data(),
                  a_values.data(), static_cast<OSQPInt>(a_values.size()), -1,
                  0};

  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.polishing = 1;
  settings.max_iter = 2000;
  if (const char *raw = std::getenv("TWALKER_QP_MAX_ITER")) {
    const long requested = std::strtol(raw, nullptr, 10);
    if (requested > 0 && requested <= 1000000)
      settings.max_iter = static_cast<OSQPInt>(requested);
  }
  int max_batches = 1;
  if (const char *raw = std::getenv("TWALKER_QP_MAX_BATCHES")) {
    const long requested = std::strtol(raw, nullptr, 10);
    if (requested > 0 && requested <= 1000)
      max_batches = static_cast<int>(requested);
  }
  settings.eps_abs = 1e-5;
  settings.eps_rel = 1e-5;
  settings.rho = 1.0;
  if (const char *raw = std::getenv("TWALKER_QP_RHO")) {
    const double requested = std::strtod(raw, nullptr);
    if (std::isfinite(requested) && requested > 0.0 && requested <= 1e6)
      settings.rho = requested;
  }
  settings.adaptive_rho_interval = 50;
  // Use an iteration budget, not a power-state-dependent wall clock.
  settings.time_limit = 1e6;

  OSQPSolver *solver = static_cast<OSQPSolver *>(osqp_solver_);
  int setup_status = 0;
  const auto qp_begin = std::chrono::steady_clock::now();
  if (!solver) {
    setup_status = osqp_setup(&solver, &P, cost.data(), &A, lower.data(),
                              upper.data(), constraints, n, &settings);
    osqp_solver_ = solver;
  } else {
    setup_status = osqp_update_data_vec(solver, cost.data(), lower.data(),
                                        upper.data());
  }
  if (setup_status != 0 || solver == nullptr) {
    if (std::getenv("TWALKER_QP_TRACE"))
      std::cerr << "OSQP setup status: " << setup_status << '\n';
    if (solver) osqp_cleanup(solver);
    osqp_solver_ = nullptr;
    return {};
  }
  int solve_status = 0;
  bool support_usable = false;
  for (int batch = 0; batch < max_batches; ++batch) {
    solve_status = osqp_solve(solver);
    qp_iterations_ += solver->info->iter;
    support_usable =
        solver->info->status_val == OSQP_SOLVED
        || solver->info->status_val == OSQP_SOLVED_INACCURATE
        || (solver->info->status_val == OSQP_MAX_ITER_REACHED
            && solver->info->prim_res <= 1e-3);
    if (std::getenv("TWALKER_QP_TRACE"))
      std::cerr << "OSQP solve call=" << solve_status
                << " batch=" << (batch + 1)
                << " status=" << solver->info->status_val
                << " iter=" << solver->info->iter
                << " primal=" << solver->info->prim_res
                << " dual=" << solver->info->dual_res << '\n';
    if (solve_status != 0 || support_usable
        || solver->info->status_val != OSQP_MAX_ITER_REACHED)
      break;
  }
  qp_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - qp_begin)
                .count();
  ++qp_calls_;
  const bool solved = solve_status == 0 && solver->solution != nullptr
                      && solver->solution->x != nullptr && support_usable;
  std::vector<double> y(n, 0.0);
  if (solved)
    std::copy_n(solver->solution->x, n, y.begin());
  if (!solved) return {};
  return y;
}

bool Walker::rank_complete_projection(
    double t, const FaceSolution &endpoint,
    std::vector<std::uint8_t> &support, FaceSolution &solution) {
  const auto started = std::chrono::steady_clock::now();
  ++rank_complete_calls_;
  completed_tight_support_.clear();
  completed_endpoint_u_.clear();
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  std::vector<double> y_star(n), u_star(m);
  for (int column = 0; column < m; ++column)
    u_star[column] = std::fma(t, endpoint.ua[column], endpoint.uc[column]);
  for (std::size_t local = 0; local < endpoint.rows.size(); ++local)
    y_star[endpoint.rows[local]] = std::max(
        0.0, std::fma(t, endpoint.g[local], endpoint.h[local]));

  const double y_scale = std::max(1.0, inf_norm(y_star));
  const double eps = std::numeric_limits<double>::epsilon();
  bool repaired = false;
  for (double threshold : {1e-7, 1e-8, 1e-9, 1e-6, 1e-5, 1e-4}) {
    std::vector<std::uint8_t> positive(n, 0), pinned(n, 0);
    int positive_count = 0;
    for (int row = 0; row < n; ++row) {
      positive[row] = y_star[row] > threshold * y_scale;
      positive_count += positive[row];
    }
    if (!positive_count) continue;

    // Full right singular vectors of B_P give an orthonormal basis M for
    // null(B_P).  Moving u inside u+M w leaves the projected y* unchanged.
    const int face_rows = positive_count;
    const int thin = std::min(face_rows, m);
    std::vector<double> BP(static_cast<std::size_t>(face_rows) * m, 0.0);
    int local = 0;
    for (int row = 0; row < n; ++row) {
      if (!positive[row]) continue;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        BP[local + static_cast<std::size_t>(face_rows)
                       * fixture_.indices[p]] = fixture_.values[p];
      ++local;
    }
    std::vector<double> singular(thin);
    std::vector<double> U(static_cast<std::size_t>(face_rows) * face_rows);
    std::vector<double> VT(static_cast<std::size_t>(m) * m);
    std::vector<int> iwork(8 * std::max(1, thin));
    const char job = 'A';
    const int lda = face_rows, ldu = face_rows, ldvt = m;
    int info = 0, lwork = -1;
    double query = 0.0;
    dgesdd_(&job, &face_rows, &m, BP.data(), &lda, singular.data(), U.data(),
            &ldu, VT.data(), &ldvt, &query, &lwork, iwork.data(), &info);
    if (info != 0 || !std::isfinite(query) || query < 1.0) continue;
    lwork = static_cast<int>(std::ceil(query));
    std::vector<double> work(lwork);
    dgesdd_(&job, &face_rows, &m, BP.data(), &lda, singular.data(), U.data(),
            &ldu, VT.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
    if (info != 0) continue;
    int rank = 0;
    const double cutoff = thin
        ? singular.front() * std::max(face_rows, m) * eps : 0.0;
    while (rank < thin && singular[rank] > cutoff) ++rank;
    int nullity = m - rank;
    std::vector<double> M(static_cast<std::size_t>(m) * nullity);
    for (int component = 0; component < nullity; ++component)
      for (int column = 0; column < m; ++column)
        M[column + static_cast<std::size_t>(m) * component] =
            VT[(rank + component) + static_cast<std::size_t>(m) * column];

    std::vector<double> u = u_star;
    std::vector<double> slack(n);
    for (int row = 0; row < n; ++row) {
      slack[row] = t * fixture_.b[row] + target_shift_[row];
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        slack[row] += fixture_.values[p] * u[fixture_.indices[p]];
    }
    std::vector<std::uint8_t> free(n, 1);
    for (int row = 0; row < n; ++row) free[row] = !positive[row];
    int steps = 0;
    const int initial_nullity = nullity;
    while (nullity > 0 && steps < initial_nullity) {
      std::vector<double> BM(static_cast<std::size_t>(n) * nullity, 0.0);
      int chosen = -1;
      double chosen_slack = -std::numeric_limits<double>::infinity();
      for (int row = 0; row < n; ++row) {
        double row_norm2 = 0.0;
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
          const int column = fixture_.indices[p];
          const double value = fixture_.values[p];
          for (int component = 0; component < nullity; ++component)
            BM[row + static_cast<std::size_t>(n) * component] +=
                value * M[column + static_cast<std::size_t>(m) * component];
        }
        for (int component = 0; component < nullity; ++component) {
          const double value =
              BM[row + static_cast<std::size_t>(n) * component];
          row_norm2 += value * value;
        }
        if (free[row] && row_norm2 > 0.0 && std::isfinite(slack[row])
            && slack[row] > chosen_slack) {
          chosen = row;
          chosen_slack = slack[row];
        }
      }
      if (chosen < 0) break;
      std::vector<double> a(nullity);
      double a_norm2 = 0.0;
      for (int component = 0; component < nullity; ++component) {
        a[component] = BM[chosen + static_cast<std::size_t>(n) * component];
        a_norm2 += a[component] * a[component];
      }
      const double a_norm = std::sqrt(a_norm2);
      if (!(a_norm > 0.0)) break;
      std::vector<double> direction(m, 0.0), rate(n, 0.0);
      for (int component = 0; component < nullity; ++component) {
        const double w = a[component] / a_norm;
        for (int column = 0; column < m; ++column)
          direction[column] +=
              M[column + static_cast<std::size_t>(m) * component] * w;
      }
      double rate_scale = 1.0;
      for (int row = 0; row < n; ++row) {
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
          rate[row] += fixture_.values[p] * direction[fixture_.indices[p]];
        rate_scale = std::max(rate_scale, std::abs(rate[row]));
      }
      int hit = -1;
      double alpha = std::numeric_limits<double>::infinity();
      for (int row = 0; row < n; ++row) {
        if (!free[row] || rate[row] <= 1e-12 * rate_scale) continue;
        const double candidate = slack[row] < 0.0
                                     ? -slack[row] / rate[row] : 0.0;
        if (candidate < alpha) {
          alpha = candidate;
          hit = row;
        }
      }
      if (hit < 0 || !std::isfinite(alpha) || alpha < 0.0) break;
      for (int column = 0; column < m; ++column)
        u[column] += alpha * direction[column];
      for (int row = 0; row < n; ++row) slack[row] += alpha * rate[row];
      slack[hit] = 0.0;
      pinned[hit] = 1;
      free[hit] = 0;
      ++steps;

      // Householder-deflate the newly pinned row from M without another SVD.
      std::vector<double> house(nullity);
      double pinned_norm2 = 0.0;
      for (int component = 0; component < nullity; ++component) {
        house[component] =
            BM[hit + static_cast<std::size_t>(n) * component];
        pinned_norm2 += house[component] * house[component];
      }
      const double pinned_norm = std::sqrt(pinned_norm2);
      house[0] += std::copysign(pinned_norm, house[0]);
      double house_norm2 = 0.0;
      for (double value : house) house_norm2 += value * value;
      if (nullity == 1 || !(house_norm2 > 0.0)) {
        M.clear();
        nullity = 0;
        break;
      }
      std::vector<double> mh(m, 0.0);
      for (int component = 0; component < nullity; ++component)
        for (int column = 0; column < m; ++column)
          mh[column] +=
              M[column + static_cast<std::size_t>(m) * component]
              * house[component];
      const double beta = 2.0 / house_norm2;
      std::vector<double> reduced(
          static_cast<std::size_t>(m) * (nullity - 1));
      for (int component = 1; component < nullity; ++component)
        for (int column = 0; column < m; ++column)
          reduced[column + static_cast<std::size_t>(m) * (component - 1)] =
              M[column + static_cast<std::size_t>(m) * component]
              - beta * house[component] * mh[column];
      M.swap(reduced);
      --nullity;
    }

    std::vector<std::uint8_t> trial(n);
    for (int row = 0; row < n; ++row) trial[row] = positive[row] || pinned[row];
    double max_off = -std::numeric_limits<double>::infinity();
    bool finite_slack = true;
    for (int row = 0; row < n; ++row) {
      if (trial[row]) continue;
      finite_slack = finite_slack && std::isfinite(slack[row]);
      if (std::isfinite(slack[row])) max_off = std::max(max_off, slack[row]);
    }
    const double slack_scale = std::max(
        1.0, inf_norm(fixture_.b) * std::max(1.0, std::abs(t)));
    if (std::getenv("TWALKER_RANK_COMPLETE_TRACE"))
      std::cerr << "RANK_COMPLETE threshold=" << threshold
                << " positive=" << positive_count << " rank=" << rank
                << " nullity=" << initial_nullity << " steps=" << steps
                << " working="
                << std::count(trial.begin(), trial.end(), std::uint8_t{1})
                << " max_off=" << max_off << '\n';
    if (!finite_slack || max_off > 1e-6 * slack_scale) continue;
    // Preserve a structurally completed endpoint for the anchored crossover.
    // The ordinary FaceSolver below deliberately chooses a min-norm
    // representation and may reject even though this multiplier and tight set
    // describe the correct endpoint.  Full completion is required: a partial
    // null-space ascent is not a square basis certificate.
    if (nullity == 0 && steps == initial_nullity) {
      completed_tight_support_ = trial;
      completed_endpoint_u_ = u;
    }
    try {
      auto trial_solution = solve(trial);
      bool accepted = accept(trial, t, trial_solution);
      if (!accepted)
        accepted = settle(trial, t, trial_solution, 20, true);
      if (std::getenv("TWALKER_RANK_COMPLETE_TRACE"))
        std::cerr << "RANK_COMPLETE gate t=" << t
                  << " accepted=" << accepted
                  << " reason=" << last_reject_ << '\n';
      if (accepted) {
        support = std::move(trial);
        solution = std::move(trial_solution);
        ++rank_complete_repairs_;
        repaired = true;
        break;
      }
    } catch (const FaceDecline &) {
      continue;
    }
  }
  rank_complete_ms_ += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count();
  return repaired;
}

bool Walker::critical_right_transition(
    double t, const FaceSolution &endpoint,
    std::vector<std::uint8_t> &support, FaceSolution &solution,
    double &advanced_t) {
  const auto started = std::chrono::steady_clock::now();
  ++critical_right_calls_;
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);

  std::vector<double> y(n, 0.0);
  for (std::size_t local = 0; local < endpoint.rows.size(); ++local)
    y[endpoint.rows[local]] =
        std::fma(t, endpoint.g[local], endpoint.h[local]);
  std::vector<double> endpoint_slack(n);
  for (int row = 0; row < n; ++row)
    endpoint_slack[row] = std::fma(
        t, fixture_.b[row] + endpoint.bua[row],
        target_shift_[row] + endpoint.buc[row]);
  // On active rows the cached projection product is the same y value, but it
  // avoids cancellation in t*g+h on long paths.
  std::vector<std::uint8_t> endpoint_active(n, 0);
  for (auto row : endpoint.rows) endpoint_active[row] = 1;
  for (int row = 0; row < n; ++row) {
    if (endpoint_active[row]) {
      // y=t*g+h is the equality-feasible endpoint coordinate; retain it.
      // The product form is used only for inactive projection slacks.
    } else {
      // The left segment is feasible through its minimum-ratio endpoint.
      // A positive value here is cancellation in the global affine
      // coefficients, not permission to step past an earlier event.
      endpoint_slack[row] = std::min(0.0, endpoint_slack[row]);
    }
  }
  const auto endpoint_y = y;

  double zero_relative = 64.0 * std::numeric_limits<double>::epsilon();
  if (const char *raw = std::getenv("TWALKER_CRITICAL_ZERO_TOL")) {
    const double requested = std::strtod(raw, nullptr);
    if (std::isfinite(requested) && requested > 0.0 && requested < 1.0)
      zero_relative = requested;
  }
  std::vector<std::uint8_t> zero(n);
  std::vector<double> zero_gate(n,
      64.0 * std::numeric_limits<double>::epsilon());
  for (std::size_t local = 0; local < endpoint.rows.size(); ++local) {
    const auto row = endpoint.rows[local];
    zero_gate[row] = std::max(
        64.0 * std::numeric_limits<double>::epsilon(),
        zero_relative * (1.0 + std::abs(y[row])));
  }
  for (int row = 0; row < n; ++row) {
    zero[row] = y[row] <= zero_gate[row];
    if (zero[row]) {
      y[row] = 0.0;
    }
    if (endpoint_active[row]) endpoint_slack[row] = y[row];
  }

  // Optional fixed-t simplex basis repair.  The endpoint y is immutable:
  // simplex is asked only to choose a feasible KKT multiplier and a
  // basic/nonbasic status representation for that already accepted point.
  // The retained basis is then warm-started for the right-slope multiplier.
  const bool simplex_basis_live =
      std::getenv("TWALKER_SIMPLEX_BASIS_LIVE") != nullptr;
  bool simplex_endpoint_selected = false;
  std::vector<double> simplex_endpoint_u;
  std::uint64_t simplex_endpoint_fingerprint = 0;
  if (simplex_basis_live) {
    std::vector<double> endpoint_offset(n);
    std::vector<std::uint8_t> endpoint_constraints(n, 1);
    std::vector<std::uint8_t> endpoint_positive(n, 0);
    for (int row = 0; row < n; ++row)
      endpoint_offset[row] =
          std::fma(t, fixture_.b[row], target_shift_[row]);
    // A zero coordinate is a bound-status row even when the face support
    // mask still labels it active.  Forcing equality there is precisely the
    // support-only degeneracy that can make a valid endpoint look
    // infeasible.  Positive coordinates alone require complementary
    // equality; zero coordinates retain the one-sided KKT inequality.
    for (int row = 0; row < n; ++row)
      endpoint_positive[row] = endpoint_active[row] && !zero[row];
    if (std::getenv("TWALKER_SIMPLEX_BASIS_TRACE")) {
      std::vector<double> native_u(m), native_product;
      for (int column = 0; column < m; ++column)
        native_u[column] =
            std::fma(t, endpoint.ua[column], endpoint.uc[column]);
      products(fixture_, native_u, native_product, nullptr);
      double active_error = 0.0, inactive_violation = 0.0;
      int active_row = -1, inactive_row = -1;
      for (int row = 0; row < n; ++row) {
        const double rhs = endpoint_y[row] - endpoint_offset[row];
        const double scale = 1.0 + std::abs(native_product[row])
                             + std::abs(endpoint_offset[row])
                             + std::abs(endpoint_y[row]);
        if (endpoint_positive[row]) {
          const double error =
              std::abs(native_product[row] - rhs) / scale;
          if (error > active_error) {
            active_error = error;
            active_row = row;
          }
        } else {
          const double violation =
              std::max(0.0, native_product[row] + endpoint_offset[row])
              / scale;
          if (violation > inactive_violation) {
            inactive_violation = violation;
            inactive_row = row;
          }
        }
      }
      std::cerr << "SIMPLEX_ENDPOINT_NATIVE active=" << active_error
                << " active_row=" << active_row
                << " inactive=" << inactive_violation
                << " inactive_row=" << inactive_row << '\n';
    }
    simplex_endpoint_selected = selector_affine_feasible(
        endpoint_positive, endpoint_y, endpoint_offset,
        endpoint_constraints,
        &simplex_endpoint_u, nullptr, &simplex_endpoint_fingerprint,
        kTolerance);
    if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
      std::cerr << "SIMPLEX_ENDPOINT selected="
                << simplex_endpoint_selected
                << " fingerprint=" << simplex_endpoint_fingerprint
                << " unchanged_t=" << std::setprecision(17) << t << '\n';
  }

  // Complementarity is componentwise.  A zero coordinate with strict
  // projection slack cannot enter the right critical cone, even when one
  // globally scaled normal-orthogonality residual looks tiny.
  std::vector<std::uint8_t> strict_zero(n, 0);
  int strict_count = 0, zero_count = 0;
  for (int row = 0; row < n; ++row) {
    if (!zero[row]) continue;
    const double slack = endpoint_slack[row];
    const double scale = 1.0 + std::abs(t * (fixture_.b[row]
                                             + endpoint.bua[row]))
        + std::abs(target_shift_[row] + endpoint.buc[row]);
    strict_zero[row] = slack < -1e-10 * scale;
    strict_count += strict_zero[row];
    zero_count += !strict_zero[row];
  }

  // The endpoint projection normal is target-y.  The right derivative is
  // P_K(b), where K is the critical cone: B'v=0, normal'v=0, v_Z>=0.
  std::vector<double> normal(n);
  double normal_norm2 = 0.0;
  for (int row = 0; row < n; ++row) {
    normal[row] = t * fixture_.b[row] + target_shift_[row] - y[row];
    normal_norm2 += normal[row] * normal[row];
  }
  const double normal_norm = std::sqrt(normal_norm2);
  if (!(normal_norm > 0.0) || !std::isfinite(normal_norm)) {
    critical_right_ms_ += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
    return false;
  }

  // Eliminate the equalities with an orthonormal null basis before the QP.
  // This is both rank revealing and much better conditioned than asking ADMM
  // to discover B'v=normal'v=0 alongside the cone inequalities.
  // Strict inactive coordinates are fixed to zero.  Eliminate them before
  // factorization instead of appending hundreds of identity equality rows;
  // this is the identical null space on a substantially smaller matrix.
  std::vector<int> free_rows;
  free_rows.reserve(n - strict_count);
  for (int row = 0; row < n; ++row)
    if (!strict_zero[row]) free_rows.push_back(row);
  const int free_count = static_cast<int>(free_rows.size());
  const int equality_rows = m + 1;
  const int thin = std::min(equality_rows, free_count);
  std::vector<double> equality_matrix(
      static_cast<std::size_t>(equality_rows) * free_count, 0.0);
  for (int local = 0; local < free_count; ++local) {
    const int row = free_rows[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1];
         ++p)
      equality_matrix[fixture_.indices[p]
                      + static_cast<std::size_t>(equality_rows) * local] =
          fixture_.values[p];
    equality_matrix[m + static_cast<std::size_t>(equality_rows) * local] =
        normal[row] / normal_norm;
  }
  std::vector<double> singular(thin);
  std::vector<double> left(
      static_cast<std::size_t>(equality_rows) * equality_rows);
  std::vector<double> right(
      static_cast<std::size_t>(free_count) * free_count);
  std::vector<int> iwork(8 * std::max(1, thin));
  const char job = 'A';
  const int lda = equality_rows, ldu = equality_rows, ldvt = free_count;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &equality_rows, &free_count, equality_matrix.data(), &lda,
          singular.data(), left.data(), &ldu, right.data(), &ldvt, &query,
          &lwork, iwork.data(), &info);
  if (info != 0 || !std::isfinite(query) || query < 1.0) {
    critical_right_ms_ += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
    return false;
  }
  lwork = static_cast<int>(std::ceil(query));
  std::vector<double> work(lwork);
  dgesdd_(&job, &equality_rows, &free_count, equality_matrix.data(), &lda,
          singular.data(), left.data(), &ldu, right.data(), &ldvt,
          work.data(), &lwork, iwork.data(), &info);
  if (info != 0) {
    critical_right_ms_ += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
    return false;
  }
  int equality_rank = 0;
  const double cutoff = thin
      ? singular.front() * std::max(equality_rows, free_count)
            * std::numeric_limits<double>::epsilon()
      : 0.0;
  while (equality_rank < thin && singular[equality_rank] > cutoff)
    ++equality_rank;
  const int nullity = free_count - equality_rank;
  if (nullity <= 0) {
    critical_right_ms_ += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
    return false;
  }
  std::vector<double> null_basis(
      static_cast<std::size_t>(n) * nullity, 0.0);
  for (int component = 0; component < nullity; ++component)
    for (int local = 0; local < free_count; ++local) {
      const int row = free_rows[local];
      null_basis[row + static_cast<std::size_t>(n) * component] =
          right[(equality_rank + component)
                + static_cast<std::size_t>(free_count) * local];
    }

  // In null coordinates, project z0=N'b onto {z : N_Z z >= 0}.  Its dual is
  // the standard NNLS problem min_{lambda>=0} ||N_Z' lambda + z0||.  The
  // maintained-QR Lawson-Hanson solver gives a finite active-set route and
  // avoids the inaccurate ADMM cone boundary observed in the cold probe.
  std::vector<double> z0(nullity, 0.0);
  for (int component = 0; component < nullity; ++component) {
    for (int row = 0; row < n; ++row)
      z0[component] +=
          null_basis[row + static_cast<std::size_t>(n) * component]
          * fixture_.b[row];
  }
  std::vector<double> cone_matrix(
      static_cast<std::size_t>(nullity) * zero_count);
  std::vector<double> nnls_rhs(nullity);
  for (int component = 0; component < nullity; ++component)
    nnls_rhs[component] = -z0[component];
  int zero_column = 0;
  for (int row = 0; row < n; ++row) {
    if (!zero[row] || strict_zero[row]) continue;
    for (int component = 0; component < nullity; ++component)
      cone_matrix[component
                  + static_cast<std::size_t>(nullity) * zero_column] =
          null_basis[row + static_cast<std::size_t>(n) * component];
    ++zero_column;
  }
  const auto nnls = dense_nnls(cone_matrix, nullity, zero_count, nnls_rhs,
                               100 * std::max(1, zero_count));
  if (nnls.x.size() != static_cast<std::size_t>(zero_count)) {
    critical_right_ms_ += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
    return false;
  }
  std::vector<double> z = z0;
  for (int column = 0; column < zero_count; ++column)
    for (int component = 0; component < nullity; ++component)
      z[component] +=
          cone_matrix[component
                      + static_cast<std::size_t>(nullity) * column]
          * nnls.x[column];
  std::vector<double> v(n, 0.0);
  for (int component = 0; component < nullity; ++component)
    for (int row = 0; row < n; ++row)
      v[row] += null_basis[row + static_cast<std::size_t>(n) * component]
                * z[component];
  // Preserve the critical-cone active-set information.  A zero-coordinate
  // inequality with positive NNLS multiplier is fixed at zero on the right;
  // a zero multiplier is a free degenerate status.  This is the native
  // basic/nonbasic distinction that a support mask alone cannot represent.
  std::vector<double> critical_bound_multiplier(
      n, std::numeric_limits<double>::infinity());
  zero_column = 0;
  double critical_multiplier_scale = 1.0;
  for (int row = 0; row < n; ++row) {
    if (!zero[row] || strict_zero[row]) continue;
    critical_bound_multiplier[row] = nnls.x[zero_column++];
    critical_multiplier_scale = std::max(
        critical_multiplier_scale,
        std::abs(critical_bound_multiplier[row]));
  }
  const double critical_multiplier_tolerance =
      64.0 * std::numeric_limits<double>::epsilon()
      * critical_multiplier_scale;

  double equality_error = 0.0;
  std::vector<double> equality(m, 0.0), equality_abs(m, 0.0);
  for (int row = 0; row < n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const int column = fixture_.indices[p];
      equality[column] += fixture_.values[p] * v[row];
      equality_abs[column] += std::abs(fixture_.values[p] * v[row]);
    }
  for (int column = 0; column < m; ++column)
    equality_error = std::max(
        equality_error,
        std::abs(equality[column]) / (1.0 + equality_abs[column]));
  double orthogonality = 0.0, v_norm2 = 0.0, tangent = 0.0;
  for (int row = 0; row < n; ++row) {
    orthogonality += normal[row] * v[row];
    v_norm2 += v[row] * v[row];
    if (strict_zero[row])
      tangent = std::max(tangent, std::abs(v[row]));
    else if (zero[row])
      tangent = std::max(tangent, -v[row]);
  }
  const double v_norm = std::sqrt(v_norm2);
  orthogonality = std::abs(orthogonality)
                  / (1.0 + normal_norm * v_norm);
  tangent = std::max(0.0, tangent) / std::max(1.0, inf_norm(v));
  const bool solved = nnls.converged
                      && std::max({equality_error, orthogonality, tangent})
                             <= 1e-7;
  if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
    std::cerr << "CRITICAL_RIGHT t=" << std::setprecision(17) << t
              << " nnls=" << nnls.converged
              << " inner=" << nnls.inner_iterations
              << " outer=" << nnls.outer_iterations
              << " strict=" << strict_count
              << " kkt=" << nnls.kkt_error
              << " equality=" << equality_error
              << " orthogonality=" << orthogonality
              << " tangent=" << tangent
              << " cumulative_ms="
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - started).count()
              << '\n';

  bool repaired = false;
  if (solved) {
    const double v_scale = std::max(1.0, inf_norm(v));
    // Quarantined POB-1 discriminator: keep the independently certified
    // critical derivative v authoritative and proceed directly to the
    // endpoint-anchored basis construction below.  The ordinary derivative
    // settle recomputes a whole-face slope and can thereby lose the very
    // right-looking signal that the critical projection supplied.  This
    // environment is deliberately non-default until the original-data gates
    // below pass on the frozen Capri obstruction and controls.
    const bool projected_objective_basis_live =
        std::getenv("TWALKER_PROJECTED_OBJECTIVE_BASIS_LIVE") != nullptr;
    if (anchored_basis_repair_enabled() && !projected_objective_basis_live) {
      // Settle only the right derivative.  Endpoint-positive coordinates are
      // superbasics.  Zero/tight coordinates exchange status until the face
      // slope points into the tangent cone on both sides.
      std::vector<std::uint8_t> derivative_support(n, 0);
      for (int row = 0; row < n; ++row)
        derivative_support[row] = !zero[row]
            || (!strict_zero[row]
                && critical_bound_multiplier[row]
                       <= critical_multiplier_tolerance);
      std::vector<std::vector<std::uint8_t>> derivative_history;
      const bool ordered_basis_live =
          std::getenv("TWALKER_ORDERED_BASIS_LIVE") != nullptr;
      if (ordered_basis_live
          && (!ordered_basis_t_valid_
              || std::abs(t - ordered_basis_t_)
                     > kForward * std::max(1.0, std::abs(t)))) {
        ordered_basis_t_valid_ = true;
        ordered_basis_t_ = t;
        ordered_basis_fingerprints_.clear();
      }
      for (int exchange = 0; exchange < 64; ++exchange) {
        if (std::find(derivative_history.begin(), derivative_history.end(),
                      derivative_support) != derivative_history.end())
          break;
        derivative_history.push_back(derivative_support);
        if (ordered_basis_live) {
          // OBNR-1: select one deterministic numerically independent basis,
          // then canonicalize its representation by original row id.  Strict
          // critical inequalities have already been removed from
          // derivative_support; they remain present in the outer event scan.
          std::vector<std::uint32_t> ordered_basis;
          if (!select_row_basis(fixture_, derivative_support,
                                ordered_basis)) {
            if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
              std::cerr << "ORDERED_BASIS no_square_basis exchange="
                        << exchange << " support="
                        << std::count(derivative_support.begin(),
                                      derivative_support.end(),
                                      std::uint8_t{1})
                        << " m=" << m << '\n';
            break;
          }
          std::sort(ordered_basis.begin(), ordered_basis.end());
          std::uint64_t fingerprint = 1469598103934665603ULL;
          auto append = [&](std::uint64_t value) {
            fingerprint ^= value;
            fingerprint *= 1099511628211ULL;
          };
          for (auto row : ordered_basis)
            append(static_cast<std::uint64_t>(row) + 1ULL);
          for (int row = 0; row < n; ++row) {
            const std::uint64_t status = strict_zero[row]
                ? 1ULL : (derivative_support[row] ? 2ULL : 3ULL);
            append((static_cast<std::uint64_t>(row) + 1ULL) * 4ULL
                   + status);
          }
          if (std::find(ordered_basis_fingerprints_.begin(),
                        ordered_basis_fingerprints_.end(), fingerprint)
              != ordered_basis_fingerprints_.end()) {
            if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
              std::cerr << "ORDERED_BASIS revisit fingerprint="
                        << fingerprint << " exchange=" << exchange << '\n';
            break;
          }
          ordered_basis_fingerprints_.push_back(fingerprint);
          if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
            std::cerr << "ORDERED_BASIS exchange=" << exchange
                      << " fingerprint=" << fingerprint
                      << " basis=" << ordered_basis.size()
                      << " first=" << ordered_basis.front()
                      << " last=" << ordered_basis.back()
                      << " locally_pruned=" << strict_count << '\n';
        }
        FaceSolution slope_face;
        try {
          slope_face = std::getenv("TWALKER_CENTERED_BASIS_LIVE")
              ? solve_centered_slope(derivative_support)
              : solve(derivative_support);
        } catch (const FaceDecline &) {
          break;
        }
        std::vector<double> face_g(n, 0.0);
        for (std::size_t local = 0; local < slope_face.rows.size(); ++local)
          face_g[slope_face.rows[local]] = slope_face.g[local];
        int change = -1;
        // Deterministic Bland-style status correction.  Drop a zero active
        // coordinate with negative derivative before admitting an inactive
        // tight coordinate with positive projection-slack derivative.
        for (int row = 0; row < n; ++row) {
          if (derivative_support[row] && zero[row]
              && face_g[row]
                     < -1e-12 * (1.0 + std::abs(face_g[row]))) {
            change = row;
            derivative_support[row] = 0;
            break;
          }
        }
        if (change < 0) {
          for (int row = 0; row < n; ++row) {
            if (!derivative_support[row] && zero[row] && !strict_zero[row]
                && fixture_.b[row] + slope_face.bua[row]
                       > 1e-12 * (1.0 + std::abs(fixture_.b[row])
                                  + std::abs(slope_face.bua[row]))) {
              change = row;
              derivative_support[row] = 1;
              break;
            }
          }
        }
        if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
          std::cerr << "DERIVATIVE_SETTLE exchange=" << exchange
                    << " support="
                    << std::count(derivative_support.begin(),
                                  derivative_support.end(), std::uint8_t{1})
                    << " change=" << change
                    << " critical_v=" << (change >= 0 ? v[change] : 0.0)
                    << " rank=" << slope_face.rank << '\n';
        if (change >= 0) continue;

        bool simplex_horizon_selected = false;

#ifdef TWALKER_ENABLE_REVISED_COLUMN
        // Once the unique face direction has passed the critical status
        // correction, select a persistent multiplier only inside null(B_F).
        // The direction solver remains authoritative; the selector is not
        // allowed to alter g or the free face.
        if (std::getenv("TWALKER_AUGMENTED_KKT_LIVE")
            && augmented_kkt_basis_) {
          std::vector<std::uint8_t> selector_constraints(n, 0);
          for (int row = 0; row < n; ++row)
            selector_constraints[row] = zero[row] && !strict_zero[row];
          revised::AugmentedKktSolution selected;
          auto *basis = static_cast<revised::AugmentedKktBasis *>(
              augmented_kkt_basis_);
          if (basis->select(derivative_support, face_g,
                            selector_constraints, selected)) {
            slope_face.ua = std::move(selected.ua);
            slope_face.bua = std::move(selected.bua);
            slope_face.piece_residual = std::max(
                {slope_face.piece_residual, selected.active_residual,
                 selected.inactive_violation,
                 selected.transpose_residual});
            if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
              std::cerr << "AUGMENTED_KKT accepted rank=" << selected.rank
                        << " nullity=" << selected.nullity
                        << " selector="
                        << selected.selector_active_rows.size()
                        << " sweeps=" << selected.projection_sweeps
                        << " active=" << selected.active_residual
                        << " inactive=" << selected.inactive_violation
                        << " transpose=" << selected.transpose_residual
                        << " fingerprint=" << selected.fingerprint << '\n';
            if (std::getenv("TWALKER_AUGMENTED_KKT_AUDIT")) {
              augmented_kkt_audit_remaining_ = 10;
              augmented_kkt_audit_consecutive_ = 0;
              augmented_kkt_last_t_ = t;
              augmented_kkt_last_fingerprint_ = selected.fingerprint;
            }
          } else if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE")) {
            std::cerr << "AUGMENTED_KKT declined reason="
                      << basis->last_failure()
                      << " blocking=" << basis->blocking_row()
                      << " active=" << selected.active_residual
                      << " inactive=" << selected.inactive_violation
                      << " transpose=" << selected.transpose_residual
                      << " sweeps=" << selected.projection_sweeps << '\n';
          }
        }
#endif

        auto corrected_y = y;
        double endpoint_correction = 0.0;
        // Refine only in the chosen right-face coordinates.  Letting every
        // tight inactive row participate can manufacture a tiny positive
        // coordinate which immediately leaves again before the minimum
        // forward step, recreating a same-t support cycle.
        const auto correction_eligible = derivative_support;
        if (!correct_endpoint_equalities(
                fixture_, corrected_y, correction_eligible,
                endpoint_correction))
          break;
        auto anchored_slack = endpoint_slack;
        for (int row = 0; row < n; ++row)
          if (derivative_support[row])
            anchored_slack[row] = corrected_y[row];

        // Do not convert this segment to global coefficients h=y0-t0*g.
        // On a long path that conversion magnifies a harmless B'*g roundoff
        // by t0 and was the remaining Capri failure.  Instead, walk the one
        // exceptional segment in anchored coordinates and hand the ordinary
        // walker a freshly solved face at its next, strictly later event.
        double anchor_residual2 = 0.0, anchor_scale2 = 0.0;
        std::vector<double> anchor_equality(fixture_.m, 0.0);
        for (int row = 0; row < n; ++row) {
          if (!derivative_support[row]) continue;
          for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1];
               ++p)
            anchor_equality[fixture_.indices[p]] +=
                fixture_.values[p] * corrected_y[row];
        }
        for (int column = 0; column < m; ++column) {
          const double residual =
              anchor_equality[column] - fixture_.d[column];
          anchor_residual2 += residual * residual;
          anchor_scale2 += fixture_.d[column] * fixture_.d[column];
        }
        const double anchor_residual = std::sqrt(anchor_residual2)
            / std::max(1.0, std::sqrt(anchor_scale2));
        double slope_residual = 0.0;
        std::vector<double> slope_equality(fixture_.m, 0.0);
        for (std::size_t local = 0; local < slope_face.rows.size(); ++local) {
          const auto row = slope_face.rows[local];
          for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1];
               ++p)
            slope_equality[fixture_.indices[p]] +=
                fixture_.values[p] * slope_face.g[local];
        }
        for (double residual : slope_equality)
          slope_residual = std::max(slope_residual, std::abs(residual));

        const double minimum_forward =
            kForward * std::max(1.0, std::abs(t)) + kForward;
        const auto native_slope_ua = slope_face.ua;
        const auto native_slope_bua = slope_face.bua;
        double simplex_selected_horizon = 0.0;
        if (simplex_basis_live) {
          std::vector<double> selected_ua, selected_bua;
          double selected_horizon = 0.0;
          std::uint64_t horizon_fingerprint = 0;
          simplex_horizon_selected = selector_maximize_forward_interval(
              derivative_support, corrected_y, anchored_slack, face_g,
              minimum_forward, 1e12 * std::max(1.0, std::abs(t)),
              selected_ua, selected_bua, selected_horizon,
              &horizon_fingerprint);
          if (simplex_horizon_selected) {
            slope_face.ua = std::move(selected_ua);
            slope_face.bua = std::move(selected_bua);
            simplex_selected_horizon = selected_horizon;
          }
          if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
            std::cerr << "SIMPLEX_HORIZON_SELECT selected="
                      << simplex_horizon_selected
                      << " delta=" << std::setprecision(17)
                      << selected_horizon
                      << " fingerprint=" << horizon_fingerprint << '\n';
        }
        double event_delta = std::numeric_limits<double>::infinity();
        std::vector<double> event_distance(
            n, std::numeric_limits<double>::infinity());
        auto find_centered_event = [&]() {
          event_delta = std::numeric_limits<double>::infinity();
          std::fill(event_distance.begin(), event_distance.end(),
                    std::numeric_limits<double>::infinity());
          for (int row = 0; row < n; ++row) {
            const double value = derivative_support[row]
                ? corrected_y[row] : anchored_slack[row];
            const double row_direction = derivative_support[row]
                ? face_g[row] : fixture_.b[row] + slope_face.bua[row];
            const bool approaches_boundary = derivative_support[row]
                ? row_direction < -1e-13 : row_direction > 1e-13;
            if (!approaches_boundary) continue;
            const double distance = -value / row_direction;
            if (std::isfinite(distance) && distance > minimum_forward) {
              event_distance[row] = distance;
              event_delta = std::min(event_delta, distance);
            }
          }
        };
        find_centered_event();
        // The LP's Delta is a certificate, not a suggestion.  If recovered
        // ua creates an earlier event, numerical scaling has invalidated the
        // candidate.  Restore the native direction artifacts exactly.
        if (simplex_horizon_selected
            && (!std::isfinite(event_delta)
                || event_delta
                       < simplex_selected_horizon
                             - 1e-8
                                   * std::max(1.0,
                                              simplex_selected_horizon))) {
          if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
            std::cerr << "SIMPLEX_HORIZON_REJECT certified="
                      << std::setprecision(17) << simplex_selected_horizon
                      << " recovered_event=" << event_delta << '\n';
          slope_face.ua = native_slope_ua;
          slope_face.bua = native_slope_bua;
          simplex_horizon_selected = false;
          find_centered_event();
        }

#ifdef TWALKER_ENABLE_REVISED_COLUMN
        // LHR-1: the local face residual is not the relevant error budget.
        // Only after the event scan reveals Delta can we estimate endpoint
        // drift.  Clean/short segments pay one multiplication and no polish.
        if (!simplex_horizon_selected
            && std::getenv("TWALKER_QUOTIENT_BASIS_LIVE")
            && maintained_deficient_qr_solver_
            && std::isfinite(event_delta)) {
          const double horizon_scale =
              std::max(1.0, inf_norm(fixture_.d));
          const double horizon_budget = 1e-8 * horizon_scale;
          double predicted_drift = event_delta * slope_residual;
          const double raw_residual = slope_residual;
          const double raw_drift = predicted_drift;
          double refined_residual = std::numeric_limits<double>::quiet_NaN();
          double refined_drift = std::numeric_limits<double>::quiet_NaN();
          bool refined = false, cold_polish = false;
          auto refresh_slope = [&]() {
            std::fill(face_g.begin(), face_g.end(), 0.0);
            for (std::size_t local = 0;
                 local < slope_face.rows.size(); ++local)
              face_g[slope_face.rows[local]] = slope_face.g[local];
            std::fill(slope_equality.begin(), slope_equality.end(), 0.0);
            for (std::size_t local = 0;
                 local < slope_face.rows.size(); ++local) {
              const auto row = slope_face.rows[local];
              for (auto p = fixture_.indptr[row];
                   p < fixture_.indptr[row + 1]; ++p)
                slope_equality[fixture_.indices[p]] +=
                    fixture_.values[p] * slope_face.g[local];
            }
            slope_residual = 0.0;
            for (double residual : slope_equality)
              slope_residual = std::max(slope_residual,
                                        std::abs(residual));
            find_centered_event();
            predicted_drift = event_delta * slope_residual;
          };
          if (predicted_drift > horizon_budget) {
            revised::RevisedSlopeSolution polished;
            const auto quotient_rows = support_rows(derivative_support);
            auto *quotient =
                static_cast<revised::MaintainedDeficientQrSolver *>(
                    maintained_deficient_qr_solver_);
            if (quotient->refine(quotient_rows, polished, 2)) {
              slope_face.g = std::move(polished.g);
              slope_face.ua = std::move(polished.ua);
              slope_face.bua = std::move(polished.bua);
              slope_face.rows = std::move(polished.rows);
              slope_face.rank = polished.rank;
              slope_face.piece_residual = polished.slope_residual;
              refined = true;
              refresh_slope();
              refined_residual = slope_residual;
              refined_drift = predicted_drift;
            }
            if (!refined || !std::isfinite(predicted_drift)
                || predicted_drift > horizon_budget) {
              // Rare final polish: one native rank reveal, never a convex
              // solver side call.  This is also the fail-closed path when the
              // retained factor has become too stale for refinement.
              slope_face = solve_direct(derivative_support);
              cold_polish = true;
              // Refinement failure is evidence that the retained factor no
              // longer represents this row space accurately enough.  Do not
              // let the anchored chain reuse it on the following segment.
              quotient->invalidate();
              refresh_slope();
            }
            if (std::getenv("TWALKER_QUOTIENT_BASIS_TRACE"))
              std::cerr << "quotient horizon refined=" << refined
                        << " cold=" << cold_polish
                        << " raw_residual=" << raw_residual
                        << " raw_drift=" << raw_drift
                        << " refined_residual=" << refined_residual
                        << " refined_drift=" << refined_drift
                        << " residual=" << slope_residual
                        << " delta=" << event_delta
                        << " drift=" << predicted_drift
                        << " budget=" << horizon_budget << '\n';
            if (!std::isfinite(predicted_drift)
                || predicted_drift > horizon_budget)
              break;
          }
        }
#endif
        double centered_violation = 0.0;
        int centered_violation_row = -1;
        double centered_violation_value = 0.0;
        double centered_violation_direction = 0.0;
        if (std::isfinite(event_delta)) {
          const double probe_delta = std::min(
              0.5 * event_delta,
              std::max(minimum_forward, 1e-8 * std::max(1.0, std::abs(t))));
          for (int row = 0; row < n; ++row) {
            const double value = derivative_support[row]
                ? corrected_y[row] : anchored_slack[row];
            const double direction = derivative_support[row]
                ? face_g[row] : fixture_.b[row] + slope_face.bua[row];
            const double probe = value + probe_delta * direction;
            const double violation = derivative_support[row]
                ? std::max(0.0, -probe) : std::max(0.0, probe);
            const double relative_violation =
                violation / (1.0 + std::abs(value)
                             + probe_delta * std::abs(direction));
            if (relative_violation > centered_violation) {
              centered_violation = relative_violation;
              centered_violation_row = row;
              centered_violation_value = value;
              centered_violation_direction = direction;
            }
          }
        }
        if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
          std::cerr << "ANCHORED_SEGMENT anchor=" << anchor_residual
                    << " slope=" << slope_residual
                    << " cone=" << centered_violation
                    << " cone_row=" << centered_violation_row
                    << " cone_active="
                    << (centered_violation_row >= 0
                            ? static_cast<int>(
                                  derivative_support[centered_violation_row])
                            : -1)
                    << " cone_value=" << centered_violation_value
                    << " cone_direction=" << centered_violation_direction
                    << " delta=" << event_delta << '\n';

        if (anchor_residual <= kTolerance
            && slope_residual <= kTolerance
            && centered_violation <= kTolerance
            && std::isfinite(event_delta)) {
          const double next_t = t + event_delta;
          std::vector<std::size_t> centered_ties;
          for (int row = 0; row < n; ++row)
            if (std::isfinite(event_distance[row])
                && std::abs(event_distance[row] - event_delta)
                       <= kTie * std::max(1.0, std::abs(next_t)))
              centered_ties.push_back(row);
          auto try_centered_event = [&](bool all_ties, double trial_t,
                                        bool serial) {
            auto trial = derivative_support;
            if (all_ties)
              for (auto row : centered_ties) trial[row] = !trial[row];
            else if (!centered_ties.empty())
              trial[centered_ties.front()] = !trial[centered_ties.front()];
            FaceSolution trial_solution;
            if (!settle(trial, trial_t, trial_solution, 50, serial))
              return false;
            support = std::move(trial);
            solution = std::move(trial_solution);
            advanced_t = trial_t;
            repaired = true;
            ++critical_right_repairs_;
            return true;
          };
          if (!centered_ties.empty()) {
            if (!try_centered_event(true, next_t, false)
                && centered_ties.size() > 1)
              try_centered_event(false, next_t, true);
            if (!repaired) {
              for (int exponent : {-10, -9, -8, -7, -6}) {
                const double epsilon = std::pow(10.0, exponent)
                    * std::max(1.0, std::abs(next_t));
                if (!try_centered_event(true, next_t + epsilon, true)
                    && centered_ties.size() > 1)
                  try_centered_event(false, next_t + epsilon, true);
                if (repaired) break;
              }
            }
          }

          // If the ordinary reconstructor still cycles, keep the anchored
          // state across a bounded number of exceptional segments.  This is
          // the native basis/superbasic escape: fixed-t exchanges settle the
          // derivative, while only a strictly later boundary counts as path
          // progress.  No auxiliary simplex solve is involved.
          if (!repaired && !centered_ties.empty()
              && !std::getenv("TWALKER_TABLEAU_CRASH")) {
            std::vector<double> live_y(n, 0.0), live_slack(n);
            for (int row = 0; row < n; ++row) {
              if (derivative_support[row])
                live_y[row] = corrected_y[row]
                    + event_delta * face_g[row];
              live_slack[row] = anchored_slack[row]
                  + event_delta
                        * (fixture_.b[row] + slope_face.bua[row]);
            }
            auto live_support = derivative_support;
            for (auto row : centered_ties)
              live_support[row] = !live_support[row];
            for (int row = 0; row < n; ++row) {
              if (!live_support[row]) live_y[row] = 0.0;
              else live_slack[row] = live_y[row];
            }
            double live_t = next_t;

            for (int segment = 0; segment < 32 && !repaired; ++segment) {
              FaceSolution live_slope;
              std::vector<double> live_g(n, 0.0);
              bool derivative_settled = false;
              std::vector<std::vector<std::uint8_t>> live_history;
              for (int exchange = 0; exchange < 64; ++exchange) {
                if (std::find(live_history.begin(), live_history.end(),
                              live_support) != live_history.end())
                  break;
                live_history.push_back(live_support);
                try {
                  live_slope = std::getenv("TWALKER_CENTERED_BASIS_LIVE")
                      ? solve_centered_slope(live_support)
                      : solve(live_support);
                } catch (const FaceDecline &) {
                  break;
                }
                std::fill(live_g.begin(), live_g.end(), 0.0);
                for (std::size_t local = 0;
                     local < live_slope.rows.size(); ++local)
                  live_g[live_slope.rows[local]] = live_slope.g[local];
                double y_inf = 1.0, slack_inf = 1.0;
                for (int row = 0; row < n; ++row) {
                  y_inf = std::max(y_inf, std::abs(live_y[row]));
                  slack_inf = std::max(slack_inf,
                                       std::abs(live_slack[row]));
                }
                const double y_tight = 1e-10 * y_inf;
                const double slack_tight = 1e-10 * slack_inf;
                int change = -1;
                for (int row = 0; row < n; ++row) {
                  if (live_support[row] && live_y[row] <= y_tight
                      && live_g[row]
                             < -1e-12 * (1.0 + std::abs(live_g[row]))) {
                    live_support[row] = 0;
                    live_y[row] = 0.0;
                    change = row;
                    break;
                  }
                }
                if (change < 0) {
                  for (int row = 0; row < n; ++row) {
                    const double q_slope =
                        fixture_.b[row] + live_slope.bua[row];
                    if (!live_support[row]
                        && live_slack[row] >= -slack_tight
                        && q_slope
                               > 1e-12 * (1.0 + std::abs(fixture_.b[row])
                                          + std::abs(live_slope.bua[row]))) {
                      live_support[row] = 1;
                      live_y[row] = 0.0;
                      live_slack[row] = 0.0;
                      change = row;
                      break;
                    }
                  }
                }
                if (change < 0) {
                  derivative_settled = true;
                  break;
                }
              }
              if (!derivative_settled) break;

              double live_correction = 0.0;
              if (!correct_endpoint_equalities(
                      fixture_, live_y, live_support, live_correction))
                break;
              for (int row = 0; row < n; ++row)
                if (live_support[row]) live_slack[row] = live_y[row];

              // Reoptimize only the multiplier direction, using the prior
              // horizon basis as the simplex warm start.  This maximizes the
              // next certified interval instead of accepting arbitrary
              // multiplier-boundary events.
              const double live_forward =
                  kForward * std::max(1.0, std::abs(live_t)) + kForward;
              if (simplex_basis_live && simplex_horizon_selected) {
                std::vector<double> selected_ua, selected_bua;
                double selected_horizon = 0.0;
                std::uint64_t live_fingerprint = 0;
                if (selector_maximize_forward_interval(
                        live_support, live_y, live_slack, live_g,
                        live_forward,
                        1e12 * std::max(1.0, std::abs(live_t)),
                        selected_ua, selected_bua, selected_horizon,
                        &live_fingerprint)) {
                  live_slope.ua = std::move(selected_ua);
                  live_slope.bua = std::move(selected_bua);
                  if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
                    std::cerr << "SIMPLEX_LIVE_HORIZON segment=" << segment
                              << " delta=" << std::setprecision(17)
                              << selected_horizon
                              << " fingerprint=" << live_fingerprint
                              << " unchanged_t=" << live_t << '\n';
                }
              }

              double live_delta = std::numeric_limits<double>::infinity();
              std::vector<double> live_distance(
                  n, std::numeric_limits<double>::infinity());
              for (int row = 0; row < n; ++row) {
                const double value = live_support[row]
                    ? live_y[row] : live_slack[row];
                const double direction = live_support[row]
                    ? live_g[row]
                    : fixture_.b[row] + live_slope.bua[row];
                const bool approaches = live_support[row]
                    ? direction < -1e-13 : direction > 1e-13;
                if (!approaches) continue;
                const double distance = -value / direction;
                if (std::isfinite(distance) && distance > live_forward) {
                  live_distance[row] = distance;
                  live_delta = std::min(live_delta, distance);
                }
              }
              if (!std::isfinite(live_delta)) break;
              const double live_next_t = live_t + live_delta;
              std::vector<std::size_t> live_ties;
              for (int row = 0; row < n; ++row)
                if (std::isfinite(live_distance[row])
                    && std::abs(live_distance[row] - live_delta)
                           <= kTie * std::max(1.0, std::abs(live_next_t)))
                  live_ties.push_back(row);
              if (live_ties.empty()) break;

              std::vector<double> next_y(n, 0.0), next_slack(n);
              for (int row = 0; row < n; ++row) {
                if (live_support[row])
                  next_y[row] = live_y[row] + live_delta * live_g[row];
                next_slack[row] = live_slack[row]
                    + live_delta
                          * (fixture_.b[row] + live_slope.bua[row]);
              }
              auto next_support = live_support;
              for (auto row : live_ties)
                next_support[row] = !next_support[row];

              auto attempt = [&](double trial_t) {
                auto trial = next_support;
                FaceSolution trial_solution;
                if (!settle(trial, trial_t, trial_solution, 50, true))
                  return false;
                support = std::move(trial);
                solution = std::move(trial_solution);
                advanced_t = trial_t;
                repaired = true;
                ++critical_right_repairs_;
                return true;
              };
              // Reconstructing the same substantially deficient endpoint is
              // exactly the operation that sent Capri back into settle
              // cycles.  Stay in the cheap centered chain until the rank
              // deficit is small enough for the ordinary representation, or
              // until the bounded chain's final opportunity.
              const bool reentry_candidate =
                  m - live_slope.rank <= 5 || segment == 31;
              if (reentry_candidate) {
                attempt(live_next_t);
                for (int exponent : {-10, -9, -8, -7, -6}) {
                  if (repaired) break;
                  attempt(live_next_t + std::pow(10.0, exponent)
                                        * std::max(1.0,
                                                   std::abs(live_next_t)));
                }
              }
              if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
                std::cerr << "ANCHORED_CHAIN segment=" << segment
                          << " t=" << std::setprecision(17) << live_next_t
                          << " ties=" << live_ties.size()
                          << " rank=" << live_slope.rank
                          << " correction=" << live_correction
                          << " reentry=" << reentry_candidate
                          << " accepted=" << repaired
                          << " cumulative_ms="
                          << std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started)
                                 .count()
                          << '\n';
              if (repaired) break;

              live_support = std::move(next_support);
              live_y = std::move(next_y);
              live_slack = std::move(next_slack);
              live_t = live_next_t;
              for (int row = 0; row < n; ++row) {
                if (!live_support[row]) live_y[row] = 0.0;
                else live_slack[row] = live_y[row];
              }
            }
          }
          if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
            std::cerr << "ANCHORED_SEGMENT event="
                      << std::setprecision(17) << next_t
                      << " ties=" << centered_ties.size()
                      << " accepted=" << repaired
                      << " reason=" << last_reject_ << '\n';
          if (repaired) break;
        }
        FaceSolution anchored = slope_face;
        anchored.h.resize(anchored.rows.size());
        for (std::size_t local = 0; local < anchored.rows.size(); ++local) {
          const auto row = anchored.rows[local];
          anchored.h[local] =
              std::fma(-t, anchored.g[local], corrected_y[row]);
        }
        std::vector<double> endpoint_u(m);
        if (simplex_endpoint_selected) {
          endpoint_u = simplex_endpoint_u;
        } else {
          for (int column = 0; column < m; ++column)
            endpoint_u[column] =
                std::fma(t, endpoint.ua[column], endpoint.uc[column]);
        }
        anchored.uc.resize(m);
        for (int column = 0; column < m; ++column)
          anchored.uc[column] =
              std::fma(-t, anchored.ua[column], endpoint_u[column]);
        anchored.buc.resize(n);
        for (int row = 0; row < n; ++row)
          anchored.buc[row] = std::fma(
              -t, fixture_.b[row] + anchored.bua[row],
              anchored_slack[row] - target_shift_[row]);

        std::vector<double> transpose_h(m, 0.0), transpose_g(m, 0.0);
        double constant_error = 0.0, constant_scale = 1.0;
        for (std::size_t local = 0; local < anchored.rows.size(); ++local) {
          const auto row = anchored.rows[local];
          const double h = anchored.h[local];
          constant_error = std::max(
              constant_error,
              std::abs(anchored.buc[row]
                       - (h - target_shift_[row])));
          constant_scale = std::max(constant_scale, std::abs(h));
          for (auto p = fixture_.indptr[row];
               p < fixture_.indptr[row + 1]; ++p) {
            const auto column = fixture_.indices[p];
            transpose_h[column] += fixture_.values[p] * h;
            transpose_g[column] +=
                fixture_.values[p] * anchored.g[local];
          }
        }
        double dres2 = 0.0, dscale2 = 0.0, g_error = 0.0;
        for (int column = 0; column < m; ++column) {
          transpose_h[column] -= fixture_.d[column];
          dres2 += transpose_h[column] * transpose_h[column];
          dscale2 += fixture_.d[column] * fixture_.d[column];
          g_error = std::max(g_error, std::abs(transpose_g[column]));
        }
        anchored.dres = std::sqrt(dres2)
            / std::max(1.0, std::sqrt(dscale2));
        anchored.piece_residual = std::max(
            g_error / std::max(1.0, inf_norm(anchored.g)),
            constant_error / constant_scale);
        if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
          std::cerr << "DERIVATIVE_SETTLE stable dres=" << anchored.dres
                    << " piece=" << anchored.piece_residual
                    << " endpoint_correction=" << endpoint_correction
                    << '\n';
        if (std::max(anchored.dres, anchored.piece_residual) > kTolerance)
          break;
        for (int exponent : {-10, -9, -8, -7, -6}) {
          const double tp = t + std::pow(10.0, exponent)
                                  * std::max(1.0, std::abs(t));
          if (!accept(derivative_support, tp, anchored)) continue;
          support = std::move(derivative_support);
          solution = std::move(anchored);
          advanced_t = tp;
          repaired = true;
          ++critical_right_repairs_;
          break;
        }
        break;
      }
    }
    if (repaired) {
      critical_right_ms_ += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
      return true;
    }
    // Native anchored basis+superbasic crossover.  The positive endpoint rows
    // and the zero rows with positive right derivative are exactly the right
    // active set.  Select a square basis from that set; all remaining rows are
    // superbasics.  Crucially, retain the accepted endpoint multiplier and
    // form h=y(t0)-t0*v instead of asking a pseudoinverse to reconstruct the
    // same endpoint on a different face.
    if (anchored_basis_repair_enabled()) {
      std::vector<std::uint8_t> right_tight(n, 0);
      for (int row = 0; row < n; ++row)
        right_tight[row] = !zero[row] || v[row] > 1e-10 * v_scale;
      std::vector<double> endpoint_u(m);
      if (simplex_endpoint_selected) {
        endpoint_u = simplex_endpoint_u;
      } else {
        for (int column = 0; column < m; ++column)
          endpoint_u[column] =
              std::fma(t, endpoint.ua[column], endpoint.uc[column]);
      }
      std::vector<std::uint8_t> endpoint_support(n, 0);
      std::vector<double> endpoint_face_y;
      endpoint_face_y.reserve(endpoint.rows.size());
      for (auto row : endpoint.rows) {
        endpoint_support[row] = 1;
        endpoint_face_y.push_back(endpoint_y[row]);
      }
      std::vector<double> selected_endpoint_u;
      const bool endpoint_selected = simplex_endpoint_selected
          || selector_feasible(endpoint_support, t, endpoint_face_y,
                               &selected_endpoint_u);
      if (simplex_endpoint_selected)
        selected_endpoint_u = simplex_endpoint_u;
      if (endpoint_selected) endpoint_u = std::move(selected_endpoint_u);
      std::vector<std::uint8_t> basis_eligible = right_tight;
      if (std::getenv("TWALKER_TABLEAU_CRASH")
          && completed_tight_support_.size() == fixture_.n)
        for (int row = 0; row < n; ++row)
          basis_eligible[row] =
              basis_eligible[row] || completed_tight_support_[row];
      for (int row = 0; row < n; ++row) {
        const double represented = endpoint_slack[row];
        const double scale = 1.0 + std::abs(t * (fixture_.b[row]
                                                 + endpoint.bua[row]))
            + std::abs(target_shift_[row] + endpoint.buc[row]);
        if (std::abs(represented - y[row]) <= 1e-10 * scale)
          basis_eligible[row] = 1;
      }
      std::vector<std::uint32_t> basis;
      const bool dense_basis_selected = select_row_basis(
          fixture_, basis_eligible, basis);
      bool anchored_ok = dense_basis_selected;
      std::vector<double> ua;
      if (anchored_ok) {
        std::vector<double> rhs(m);
        for (int equation = 0; equation < m; ++equation)
          rhs[equation] = v[basis[equation]] - fixture_.b[basis[equation]];
        anchored_ok = solve_basis_system(fixture_, basis, rhs, ua);
      }
      bool used_sparse_tight_solve = false;
      if (!anchored_ok) {
        try {
          // A deficient right face has no square basis.  Its rank-revealing
          // slope solve is still valid; only the constant reconstruction was
          // observed to jump.  Reuse ua and anchor uc at the accepted endpoint.
          const auto slope_face = solve_direct(right_tight);
          ua = slope_face.ua;
          anchored_ok = ua.size() == fixture_.m;
          used_sparse_tight_solve = anchored_ok;
          if (anchored_ok) basis.clear();
        } catch (const FaceDecline &) {
          anchored_ok = false;
        }
      }
      std::vector<double> bua, buc;
      if (anchored_ok) {
        products(fixture_, ua, bua, nullptr);
        double square_basis_error = 0.0;
        for (int row = 0; row < n; ++row) {
          if (!right_tight[row]) continue;
          const double rhs = v[row] - fixture_.b[row];
          square_basis_error = std::max(
              square_basis_error,
              std::abs(bua[row] - rhs)
                  / (1.0 + std::abs(bua[row]) + std::abs(rhs)));
        }
        if (square_basis_error > 1e-10) {
          try {
            const auto slope_face = solve_direct(right_tight);
            ua = slope_face.ua;
            bua = slope_face.bua;
            anchored_ok = ua.size() == fixture_.m;
            used_sparse_tight_solve = anchored_ok;
            if (anchored_ok) basis.clear();
          } catch (const FaceDecline &) {
            anchored_ok = false;
          }
        }
      }

      double endpoint_error = 0.0;
      int endpoint_error_row = -1;
      double endpoint_error_represented = 0.0;
      if (anchored_ok) {
        for (int row = 0; row < n; ++row) {
          const double represented = endpoint_slack[row];
          const double scale = 1.0 + std::abs(t * (fixture_.b[row]
                                                   + endpoint.bua[row]))
              + std::abs(target_shift_[row] + endpoint.buc[row])
              + std::abs(y[row]);
          const bool equality = right_tight[row]
              || std::find(basis.begin(), basis.end(), row) != basis.end();
          const double error = equality
              ? std::abs(represented - y[row]) / scale
              : std::max(0.0, represented) / scale;
          if (error > endpoint_error) {
            endpoint_error = error;
            endpoint_error_row = row;
            endpoint_error_represented = represented;
          }
        }
        anchored_ok = endpoint_error <= 1e-10;
      }

      if (anchored_ok) {
        std::vector<double> uc(m);
        for (int column = 0; column < m; ++column)
          uc[column] = endpoint_u[column] - t * ua[column];
        buc.resize(n);
        for (int row = 0; row < n; ++row)
          buc[row] = std::fma(
              -t, fixture_.b[row] + bua[row],
              endpoint_slack[row] - target_shift_[row]);
        std::vector<std::uint8_t> basis_mask(n, 0);
        if (used_sparse_tight_solve)
          basis_mask = right_tight;
        else
          for (auto row : basis) basis_mask[row] = 1;

        for (double relative : {1e-10, 1e-9, 1e-8, 1e-7, 1e-6}) {
          std::vector<std::uint8_t> right(n, 0);
          for (int row = 0; row < n; ++row)
            right[row] = !zero[row] || v[row] > relative * v_scale
                         || basis_mask[row];

          FaceSolution anchored;
          anchored.rows = support_rows(right);
          anchored.g.reserve(anchored.rows.size());
          anchored.h.reserve(anchored.rows.size());
          anchored.ua = ua;
          anchored.uc = uc;
          anchored.bua = bua;
          anchored.buc = buc;
          anchored.rank = m;
          std::vector<double> transpose_g(m, 0.0), transpose_h(m, 0.0);
          double slope_error = 0.0, constant_error = 0.0;
          double slope_scale = 1.0, constant_scale = 1.0;
          for (auto row : anchored.rows) {
            const double g = v[row];
            const double h = y[row] - t * v[row];
            anchored.g.push_back(g);
            anchored.h.push_back(h);
            const double target_slope = g - fixture_.b[row];
            slope_error = std::max(slope_error,
                                   std::abs(bua[row] - target_slope));
            constant_error = std::max(
                constant_error,
                std::abs(buc[row] - (h - target_shift_[row])));
            slope_scale = std::max(slope_scale, std::abs(target_slope));
            constant_scale = std::max(constant_scale, std::abs(h));
            for (auto p = fixture_.indptr[row];
                 p < fixture_.indptr[row + 1]; ++p) {
              const int column = fixture_.indices[p];
              transpose_g[column] += fixture_.values[p] * g;
              transpose_h[column] += fixture_.values[p] * h;
            }
          }
          double dres2 = 0.0, dscale2 = 0.0;
          double transpose_g_inf = 0.0;
          for (int column = 0; column < m; ++column) {
            transpose_h[column] -= fixture_.d[column];
            dres2 += transpose_h[column] * transpose_h[column];
            dscale2 += fixture_.d[column] * fixture_.d[column];
            transpose_g_inf = std::max(transpose_g_inf,
                                       std::abs(transpose_g[column]));
          }
          anchored.dres = std::sqrt(dres2)
              / std::max(1.0, std::sqrt(dscale2));
          anchored.piece_residual = std::max(
              {transpose_g_inf / std::max(1.0, v_scale),
               slope_error / slope_scale, constant_error / constant_scale});

          double off_slope = 0.0;
          for (int row = 0; row < n; ++row) {
            // A coordinate that is already strictly inactive at t0 may have
            // positive slope: it remains inactive over a finite right-hand
            // interval until its negative slack reaches zero.  Only a row
            // tight at the endpoint must point into the inactive half-space.
            if (right[row] || !zero[row] || strict_zero[row]) continue;
            const double slope = fixture_.b[row] + bua[row];
            off_slope = std::max(off_slope,
                                 std::max(0.0, slope) /
                                     (1.0 + std::abs(fixture_.b[row])
                                      + std::abs(bua[row])));
          }
          if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
            std::cerr << "ANCHORED_BASIS basis=" << basis.size()
                      << " sparse_tight=" << used_sparse_tight_solve
                      << " endpoint_selected=" << endpoint_selected
                      << " support=" << anchored.rows.size()
                      << " endpoint=" << endpoint_error
                      << " dres=" << anchored.dres
                      << " piece=" << anchored.piece_residual
                      << " off_slope=" << off_slope << '\n';
          if (std::max({anchored.dres, anchored.piece_residual,
                        off_slope}) > 1e-10)
            continue;

          for (int exponent : {-10, -9, -8, -7, -6}) {
            const double tp = t + std::pow(10.0, exponent)
                                    * std::max(1.0, std::abs(t));
            if (!accept(right, tp, anchored)) continue;
            support = std::move(right);
            solution = std::move(anchored);
            advanced_t = tp;
            repaired = true;
            ++critical_right_repairs_;
            if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
              std::cerr << "ANCHORED_BASIS accepted_t="
                        << std::setprecision(17) << tp << '\n';
            break;
          }
          if (repaired) break;
        }
      }
      if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE") && !anchored_ok)
        std::cerr << "ANCHORED_BASIS construction_failed endpoint="
                  << endpoint_error
                  << " dense_selected=" << dense_basis_selected
                  << " endpoint_selected=" << endpoint_selected
                  << " selected=" << basis.size()
                  << " eligible="
                  << std::count(basis_eligible.begin(), basis_eligible.end(),
                                std::uint8_t{1})
                  << " error_row=" << endpoint_error_row
                  << " represented=" << endpoint_error_represented
                  << " y=" << (endpoint_error_row >= 0
                                    ? y[endpoint_error_row] : 0.0)
                  << " v=" << (endpoint_error_row >= 0
                                    ? v[endpoint_error_row] : 0.0) << '\n';
    }
    if (repaired) {
      critical_right_ms_ += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
      return true;
    }
    for (double relative : {1e-6, 1e-7, 1e-8, 1e-9}) {
      std::vector<std::uint8_t> right(n);
      for (int row = 0; row < n; ++row)
        right[row] = !zero[row] || v[row] > relative * v_scale;
      // Keep the accepted endpoint's zero-valued rows as representation
      // artifacts.  The critical derivative decides which new coordinates
      // enter, while retaining the live left basis prevents an ill-conditioned
      // right support from jumping to a different solution of B'y=d.
      for (auto row : endpoint.rows) right[row] = 1;
      if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
        std::cerr << "CRITICAL_RIGHT support_tol=" << relative
                  << " positive="
                  << std::count(right.begin(), right.end(), std::uint8_t{1})
                  << '\n';
      for (int exponent : {-10, -9, -8, -7, -6}) {
        const double tp = t + std::pow(10.0, exponent)
                                * std::max(1.0, std::abs(t));
        auto trial = right;
        FaceSolution trial_solution;
        bool accepted = false;
        bool face_solved = false;
        try {
          trial_solution = solve(trial);
          face_solved = true;
          accepted = accept(trial, tp, trial_solution);
          if (!accepted && std::getenv("TWALKER_CRITICAL_RIGHT_TRACE")) {
            double minimum = std::numeric_limits<double>::infinity();
            std::size_t minimum_row = fixture_.n;
            for (std::size_t local = 0;
                 local < trial_solution.rows.size(); ++local) {
              const double value = std::fma(
                  tp, trial_solution.g[local], trial_solution.h[local]);
              if (value < minimum) {
                minimum = value;
                minimum_row = trial_solution.rows[local];
              }
            }
            std::cerr << "CRITICAL_RIGHT min_y=" << minimum
                      << " row=" << minimum_row
                      << " endpoint_y="
                      << (minimum_row < y.size() ? y[minimum_row] : 0.0)
                      << " derivative="
                      << (minimum_row < v.size() ? v[minimum_row] : 0.0)
                      << " zero="
                      << (minimum_row < zero.size()
                              ? static_cast<int>(zero[minimum_row]) : -1)
                      << '\n';
          }
        } catch (const FaceDecline &) {
          accepted = false;
        }
        if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
          std::cerr << "CRITICAL_RIGHT direct epsilon=" << exponent
                    << " accepted=" << accepted
                    << " reason=" << last_reject_ << '\n';
        if (!accepted && face_solved && exponent == -8) {
          auto completed = trial;
          FaceSolution completed_solution;
          if (rank_complete_projection(tp, trial_solution, completed,
                                       completed_solution)) {
            trial = std::move(completed);
            trial_solution = std::move(completed_solution);
            accepted = true;
          }
        }
        if (!accepted) {
          trial = right;
          accepted = settle(trial, tp, trial_solution, 20);
        }
        if (!accepted) {
          trial = right;
          accepted = settle(trial, tp, trial_solution, 50, true);
        }
        if (std::getenv("TWALKER_CRITICAL_RIGHT_TRACE"))
          std::cerr << "CRITICAL_RIGHT epsilon=" << exponent
                    << " accepted=" << accepted
                    << " reason=" << last_reject_ << '\n';
        if (accepted) {
          support = std::move(trial);
          solution = std::move(trial_solution);
          advanced_t = tp;
          repaired = true;
          ++critical_right_repairs_;
          break;
        }
      }
      if (repaired) break;
    }
  }
  critical_right_ms_ += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started)
                            .count();
  return repaired;
}

bool Walker::apply_objective_preserving_rank_lift(
    std::vector<std::uint8_t> &support, double t, FaceSolution &solution) {
  if (!rank_lift_live_requested_ || rank_lift_candidate_y_.empty()
      || rank_lift_candidate_support_.size() != fixture_.n)
    return false;
  std::vector<double> projection_value(fixture_.n, 0.0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    projection_value[row] =
        t * (fixture_.b[row] + solution.bua[row])
        + target_shift_[row] + solution.buc[row];
  auto lifted_shift = target_shift_;
  for (std::size_t row = 0; row < fixture_.n; ++row)
    if (rank_lift_candidate_support_[row])
      lifted_shift[row] +=
          rank_lift_candidate_y_[row] - projection_value[row];
  double shift_delta_inf = 0.0, center_inf = 1.0;
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    shift_delta_inf = std::max(
        shift_delta_inf, std::abs(lifted_shift[row] - target_shift_[row]));
    center_inf = std::max(
        center_inf, std::abs(t * fixture_.b[row] + target_shift_[row]));
  }

  auto *trial_solver =
      new FaceSolver(fixture_, true, lifted_shift, -1,
                     std::getenv("TWALKER_DIRECT_CANDIDATE") != nullptr);
  bool admitted = false;
  FaceSolution trial_solution;
  try {
    trial_solution = trial_solver->solve(
        support_rows(rank_lift_candidate_support_));
    std::vector<double> trial_y(fixture_.n, 0.0);
    for (std::size_t local = 0; local < trial_solution.rows.size(); ++local)
      trial_y[trial_solution.rows[local]] =
          std::fma(t, trial_solution.g[local], trial_solution.h[local]);
    const double candidate_error =
        relative_inf_error(trial_y, rank_lift_candidate_y_);
    const auto original_shift = target_shift_;
    target_shift_ = lifted_shift;
    admitted = candidate_error <= kTolerance
               && accept(rank_lift_candidate_support_, t, trial_solution);
    if (!admitted) target_shift_ = original_shift;
  } catch (const FaceDecline &) {
    admitted = false;
  }
  if (!admitted) {
    delete trial_solver;
    rank_lift_candidate_y_.clear();
    rank_lift_candidate_support_.clear();
    last_reject_.clear();
    return false;
  }

  rank_lift_live_solver_ = trial_solver;
  ++face_solves_;
  dense_fallbacks_ += trial_solution.used_dense_fallback;
  record_face_rank(trial_solution);
  support = rank_lift_candidate_support_;
  solution = std::move(trial_solution);
  gram_retired_ = true;
  rank_lift_audit_enabled_ = false;
  rank_lift_live_requested_ = false;
  rank_lift_audit_.live_applied = 1;
  rank_lift_audit_.live_shift_inf = shift_delta_inf;
  rank_lift_audit_.live_shift_relative = shift_delta_inf / center_inf;
  return true;
}

WalkResult Walker::run(int max_pivots, double tmax) {
  WalkResult result;
  std::vector<std::uint8_t> support = fixture_.post_seed_support;
  double t = fixture_.t0;
  enum class SeedMode { kFixture, kNewton, kKernel, kTriangular, kHighs };
  SeedMode seed_mode = SeedMode::kNewton;
  if (const char *raw = std::getenv("TWALKER_SEED")) {
    const std::string requested(raw);
    if (requested == "newton")
      seed_mode = SeedMode::kNewton;
    else if (requested == "kernel")
      seed_mode = SeedMode::kKernel;
    else if (requested == "triangular")
      seed_mode = SeedMode::kTriangular;
    else if (requested == "highs")
      seed_mode = SeedMode::kHighs;
    else if (requested == "fixture")
      seed_mode = SeedMode::kFixture;
    else
      throw std::invalid_argument(
          "TWALKER_SEED must be newton, kernel, triangular, highs, or fixture");
  } else if (std::getenv("TWALKER_DISABLE_NATIVE_SEED")) {
    // Backward-compatible alias for the old post-seed control mode.
    seed_mode = SeedMode::kFixture;
  }
  if (!enable_native_seed_) seed_mode = SeedMode::kFixture;

  if (seed_mode != SeedMode::kFixture) {
    t = 0.0;
    const bool use_highs = seed_mode == SeedMode::kHighs;
    const bool use_kernel = seed_mode == SeedMode::kKernel;
    const bool use_triangular = seed_mode == SeedMode::kTriangular;
    const bool seeded = use_highs ? highs_projection_seed(support)
                        : use_kernel ? kernel_projection_seed(support)
                        : use_triangular
                            ? triangular_projection_seed(support)
                                     : native_newton_seed(support);
    result.native_seed = !use_highs;
    result.highs_seed = use_highs;
    result.seed_converged = native_seed_converged_;
    result.seed_route = native_seed_route_;
    result.seed_iterations = native_seed_iterations_;
    result.seed_support = native_seed_support_;
    result.seed_ms = native_seed_ms_;
    result.seed_dres = native_seed_dres_;
    result.seed_mask = support;
    if (!seeded) {
      result.status = use_highs ? "HiGHS seed failed"
                      : use_kernel ? "kernel seed failed"
                      : use_triangular ? "triangular seed failed"
                                   : "native seed failed";
      result.t = t;
      return result;
    }
  }
  FaceSolution solution;
  bool lexicographic_mode = false;
  int right_status_repair_chain = 0;
  int last_direct_event_retry_pivot = -1;
  bool initialized = false;
  try {
    solution = solve(support);
    initialized = accept(support, t, solution);
  } catch (const FaceDecline &) {
    initialized = false;
  }
  if (!initialized && !settle(support, t, solution)) {
    result.status = "initial face rejected";
    result.t = t;
    result.face_solves = face_solves_;
    result.dense_fallbacks = dense_fallbacks_;
    result.rank_deficient_solves = rank_deficient_solves_;
    result.rank_deficit_sum = rank_deficit_sum_;
    result.max_rank_deficit = max_rank_deficit_;
    result.settle_rounds = settle_rounds_;
    return result;
  }
  if (result.highs_seed) {
    // At this point the ordinary fixed-t face solve and acceptance gates have
    // certified the face, including any rough HiGHS support nomination.
    result.seed_converged = true;
    if (result.seed_route == "highs-qp-candidate")
      result.seed_route += "+fixed-t-repair";
    result.seed_mask = support;
    result.seed_support = static_cast<int>(
        std::count(support.begin(), support.end(), std::uint8_t{1}));
    result.seed_dres = solution.dres;
  }
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if (std::getenv("TWALKER_AUGMENTED_KKT_AUDIT_ALL")) {
    augmented_kkt_audit_remaining_ = 10;
    augmented_kkt_audit_consecutive_ = 0;
    augmented_kkt_last_t_ = -1.0;
    augmented_kkt_last_fingerprint_ = 0;
  }
#endif
  for (int pivot = 0; pivot < max_pivots; ++pivot) {
    const auto &rows = solution.rows;
#ifdef TWALKER_ENABLE_REVISED_COLUMN
    if (augmented_kkt_audit_remaining_ > 0
        && audit_augmented_kkt_state(support, t, solution))
      --augmented_kkt_audit_remaining_;
#endif
    if (trace_path_) {
      std::uint64_t fingerprint = 1469598103934665603ULL;
      for (auto row : rows) {
        fingerprint ^= static_cast<std::uint64_t>(row) + 1ULL;
        fingerprint *= 1099511628211ULL;
      }
      std::cerr << "PATH pivot=" << pivot << " t=" << std::setprecision(17)
                << t << " rows=" << rows.size() << " hash=" << fingerprint
                << " gram=" << solution.used_maintained_gram
                << " extended=" << solution.used_extended_gram << '\n';
    }
    std::vector<double> active_slope(rows.size()), active_b(rows.size());
    for (std::size_t local = 0; local < rows.size(); ++local) {
      active_slope[local] = solution.g[local];
      active_b[local] = fixture_.b[rows[local]];
    }
    const double slope_norm = stable_norm2(active_slope);
    const double b_norm = stable_norm2(active_b);
    const double terminal_threshold =
        1e-12 * std::max(1.0, b_norm);
    const double terminal_candidate_threshold =
        1e-6 * std::max(1.0, b_norm);
    if (slope_norm <= terminal_candidate_threshold) {
      // A stationary active face is sufficient to *propose* the exact
      // primal/dual pair, but not to justify an expensive feasibility
      // recovery: inactive rows may still create a forward event.  The cheap
      // original-data certificate is authoritative when it passes.  On
      // failure, continue directly to the global ratio scan.
      ++terminal_gates_;
      result.y.assign(fixture_.n, 0.0);
      for (std::size_t local = 0; local < rows.size(); ++local)
        result.y[rows[local]] = solution.h[local];
      result.x.resize(fixture_.m);
      for (std::size_t j = 0; j < fixture_.m; ++j)
        result.x[j] = -solution.ua[j];
      result.certificate = certificate_pair(result.x, result.y);
      if (certificate_passes(result.certificate)) {
        if (solution.used_bound_core) {
          // The fast structured lane may locate a certifiable point just
          // inside the global tolerance.  Termination is rare, so spend one
          // direct solve here to polish accuracy and to determine whether an
          // additional exact path event remains.
          try {
            solution = solve_direct(support);
          } catch (const FaceDecline &error) {
            result.status = "terminal bound-core polish declined: "
                            + std::string(error.what());
            result.pivots = pivot;
            result.t = t;
            break;
          }
          ++stability_refactors_;
          ++terminal_stability_refactors_;
          --pivot;
          continue;
        }
        result.status = "CERTIFIED";
        result.pivots = pivot;
        result.t = t;
        break;
      }
    }
    std::vector<double> slope(fixture_.n), constant(fixture_.n);
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      slope[row] = -(fixture_.b[row] + solution.bua[row]);
      constant[row] = -(target_shift_[row] + solution.buc[row]);
    }
    for (std::size_t local = 0; local < rows.size(); ++local) {
      slope[rows[local]] = solution.g[local];
      constant[rows[local]] = solution.h[local];
    }

    // A breakpoint may admit the endpoint while still carrying the left
    // status of a zero coordinate.  Such a state is feasible at t but is not
    // a valid right affine segment.  An absolute-time ratio test silently
    // drops its root because the root is t (up to roundoff), and can then
    // report a huge event selected by a 1e-13 slope.  Resolve the degenerate
    // status at fixed t with the exact critical-cone transition before any
    // positive-t event is counted.
    const double forward_delta =
        kForward * std::max(1.0, std::abs(t)) + kForward;
    int right_blocking_row = -1;
    double right_blocking_value = 0.0;
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      if (!(slope[row] < -1e-13)) continue;
      const double value = std::fma(t, slope[row], constant[row]);
      const double scale = 1.0 + std::abs(t * slope[row])
                           + std::abs(constant[row]);
      // The coordinate crosses before the smallest event that the outer
      // ratio test is allowed to count.  Bland order makes the status
      // correction deterministic when several rows are simultaneously
      // blocking.
      if (value + forward_delta * slope[row] <= 1e-12 * scale) {
        right_blocking_row = static_cast<int>(row);
        right_blocking_value = value;
        break;
      }
    }
    double next = std::numeric_limits<double>::infinity();
    std::vector<double> candidate(fixture_.n,
                                  std::numeric_limits<double>::infinity());
    for (std::size_t row = 0; row < fixture_.n; ++row) {
      if (slope[row] < -1e-13) {
        const double value = -constant[row] / slope[row];
        if (std::isfinite(value) && value > t * (1.0 + kForward) + kForward) {
          candidate[row] = value;
          next = std::min(next, value);
        }
      }
    }
    bool retained_tie_certificate = false;
    if (!event_decision_stable(t, tmax, solution, slope, constant,
                               candidate, next,
                               &retained_tie_certificate)) {
      try {
        solution = solve_direct(support);
      } catch (const FaceDecline &error) {
        result.status = "event stability refactor declined: "
                        + std::string(error.what());
        result.pivots = pivot;
        result.t = t;
        break;
      }
      ++stability_refactors_;
      ++event_stability_refactors_;
      --pivot;
      continue;
    }
    if (solution.used_bound_core
        && (!std::isfinite(next) || next > tmax)) {
      // A fast structured solve is allowed to carry the ordinary path, but it
      // never decides that the path is over.  Polish the same face through the
      // rank-revealing correctness lane and repeat the ratio scan.  This is a
      // single late repair, not a per-pivot oracle call.
      try {
        solution = solve_direct(support);
      } catch (const FaceDecline &error) {
        result.status = "terminal bound-core polish declined: "
                        + std::string(error.what());
        result.pivots = pivot;
        result.t = t;
        break;
      }
      ++stability_refactors_;
      ++terminal_stability_refactors_;
      --pivot;
      continue;
    }
    if (slope_norm <= terminal_threshold
        && (!std::isfinite(next) || next > tmax)) {
      // Global terminal gate: the active affine piece is stationary and the
      // audited full-row ratio scan has no credible event inside the allowed
      // forward horizon.  The original-data certificate remains decisive;
      // a tiny inactive numerical slope may otherwise manufacture a formal
      // event far beyond tmax and hide an already optimal endpoint.
      ++terminal_gates_;
      result.y.assign(fixture_.n, 0.0);
      for (std::size_t local = 0; local < rows.size(); ++local)
        result.y[rows[local]] = solution.h[local];
      result.x.resize(fixture_.m);
      for (std::size_t j = 0; j < fixture_.m; ++j)
        result.x[j] = -solution.ua[j];
      result.certificate = certificate_pair(result.x, result.y);
      if (certificate_passes(result.certificate)
          || recover_certificate(result.y, result.x, result.certificate)
          || terminal_support_repair(support, result.y, result.x,
                                     result.certificate)) {
        result.status = "CERTIFIED";
        result.pivots = pivot;
        result.t = t;
        break;
      }
    }
    // Do not pay for a critical-cone solve at ordinary breakpoints.  It is an
    // escape hatch only when the positive-time ratio test has no credible
    // event, while a boundary row proves that the stored face is invalid on
    // the right.
    if ((!std::isfinite(next) || next > tmax)
        && right_blocking_row >= 0 && anchored_basis_repair_enabled()) {
      if (++right_status_repair_chain > 64) {
        result.status = "right-status exchange cap";
        result.pivots = pivot;
        result.t = t;
        break;
      }
      auto trial_support = support;
      FaceSolution trial_solution;
      double advanced_t = t;
      if (trace_path_)
        std::cerr << "RIGHT_STATUS t=" << std::setprecision(17) << t
                  << " row=" << right_blocking_row
                  << " active=" << static_cast<int>(support[right_blocking_row])
                  << " value=" << right_blocking_value
                  << " slope=" << slope[right_blocking_row] << '\n';
      if (critical_right_transition(t, solution, trial_support,
                                    trial_solution, advanced_t)) {
        support = std::move(trial_support);
        solution = std::move(trial_solution);
        t = advanced_t;
        lexicographic_mode = true;
        // Fixed-t exchanges are resolution, not path pivots.  The critical
        // transition is allowed to carry the centered state to its next true
        // event, so re-enter the outer loop without consuming this pivot.
        --pivot;
        continue;
      }
    }
    right_status_repair_chain = 0;
    if (!std::isfinite(next)) {
      // Vanishing motion on the active face is not a terminal proof: an
      // inactive row may still enter later.  Reach this gate only after the
      // complete, uncertainty-audited ratio scan proves there is no forward
      // event anywhere in the projection KKT system.
      ++terminal_gates_;
      result.y.assign(fixture_.n, 0.0);
      for (std::size_t local = 0; local < rows.size(); ++local)
        result.y[rows[local]] = std::max(0.0,
            t * solution.g[local] + solution.h[local]);
      result.x.resize(fixture_.m);
      for (std::size_t j = 0; j < fixture_.m; ++j) result.x[j] = -solution.ua[j];
      result.certificate = certificate_pair(result.x, result.y);
      if (certificate_passes(result.certificate)
          || recover_certificate(result.y, result.x, result.certificate)
          || terminal_support_repair(support, result.y, result.x,
                                     result.certificate))
        result.status = "CERTIFIED";
      else
        result.status = "no event, certificate failed";
      result.pivots = pivot;
      result.t = t;
      break;
    }
    if (next > tmax) {
      if (trace_path_) {
        std::size_t limiting_row = fixture_.n;
        std::size_t most_negative_active = fixture_.n;
        double most_negative_active_slope = 0.0;
        double b_dot_g = 0.0;
        std::vector<double> transpose_g(fixture_.m, 0.0);
        double transpose_g_abs = 0.0;
        for (std::size_t local = 0; local < rows.size(); ++local) {
          const auto row = rows[local];
          b_dot_g += fixture_.b[row] * solution.g[local];
          if (solution.g[local] < most_negative_active_slope) {
            most_negative_active_slope = solution.g[local];
            most_negative_active = row;
          }
          for (auto p = fixture_.indptr[row];
               p < fixture_.indptr[row + 1]; ++p) {
            const double term = fixture_.values[p] * solution.g[local];
            transpose_g[fixture_.indices[p]] += term;
            transpose_g_abs += std::abs(term);
          }
        }
        double transpose_g_inf = 0.0;
        for (double value : transpose_g)
          transpose_g_inf = std::max(transpose_g_inf, std::abs(value));
        for (std::size_t row = 0; row < fixture_.n; ++row)
          if (candidate[row] == next) {
            limiting_row = row;
            break;
          }
        std::cerr << "T_CAP pivot=" << pivot
                  << " t=" << std::setprecision(17) << t
                  << " next=" << next
                  << " limiting_row=" << limiting_row
                  << " slope_norm=" << slope_norm
                  << " terminal_threshold=" << terminal_threshold
                  << " b_dot_g_minus_norm2="
                  << b_dot_g - slope_norm * slope_norm
                  << " transpose_g_inf=" << transpose_g_inf
                  << " transpose_g_scale=" << transpose_g_abs
                  << " most_negative_active=" << most_negative_active
                  << " most_negative_active_slope="
                  << most_negative_active_slope
                  << " slope="
                  << (limiting_row < fixture_.n ? slope[limiting_row] : 0.0)
                  << " constant="
                  << (limiting_row < fixture_.n ? constant[limiting_row]
                                                : 0.0)
                  << '\n';
      }
      result.status = "t cap";
      result.pivots = pivot;
      result.t = t;
      break;
    }
    const auto before_event = support;
    const double before_event_t = t;
    int ties = 0;
    std::vector<std::size_t> tie_rows;
    for (std::size_t row = 0; row < fixture_.n; ++row)
      if (std::isfinite(candidate[row])
          && std::abs(candidate[row] - next)
                 <= kTie * std::max(1.0, next)) {
        tie_rows.push_back(row);
        ++ties;
      }
    if (solution.qr_update_audit_candidate)
      audit_qr_event_decision(t, tmax, solution, next, tie_rows);
    if (trace_path_) {
      std::cerr << "EVENT pivot=" << pivot << " next="
                << std::setprecision(17) << next << " ties=" << ties
                << " lex=" << lexicographic_mode << " rows=";
      for (auto row : tie_rows) std::cerr << row << ',';
      std::cerr << '\n';
    }
    if (std::getenv("TWALKER_RETAINED_TIE_CERT")
        && solution.used_maintained_gram && ties > 1
        && !solution.used_extended_gram) {
      // Well-conditioned Gram faces normally do not pay for a posteriori
      // coefficient intervals.  A discrete tie is rare enough to justify a
      // two-RHS refinement through the factor already in memory.  Re-enter
      // the event calculation with hard bounds; no face is reconstructed.
      FaceSolution guarded;
      ++face_solves_;
      if (gram_solver_.solve(solution.rows, guarded, true)) {
        ++gram_fast_solves_;
        ++retained_tie_polishes_;
        solution = std::move(guarded);
        --pivot;
        continue;
      }
      ++gram_declines_;
      ++retained_tie_declines_;
      gram_retired_ = true;
      gram_stability_rearm_pending_ = false;
    }
    if (retained_tie_certificate) ++retained_tie_certificates_;
    if (extension_enabled_ && extension_used_ && ties > 1
        && solution.used_maintained_gram
        && !retained_tie_certificate) {
      // A true multi-row event is exactly where tiny coefficient differences
      // alter the active face.  Recompute this event directly, then keep the
      // degenerate remainder on the direct lane.  Nondegenerate programs do
      // not pay for this isolation policy.
      gram_retired_ = true;
      if (!std::getenv("TWALKER_DISABLE_GRAM_REARM"))
        gram_stability_rearm_pending_ = true;
      try {
        solution = solve_direct(support);
      } catch (const FaceDecline &error) {
        result.status = "tie stability refactor declined: "
                        + std::string(error.what());
        result.pivots = pivot;
        result.t = t;
        break;
      }
      ++stability_refactors_;
      ++event_stability_refactors_;
      --pivot;
      continue;
    }
    if (extension_enabled_ && extension_used_ && ties > 1
        && !retained_tie_certificate
        && !std::getenv("TWALKER_RETAIN_AFTER_TIE_AUDIT")) {
      gram_retired_ = true;
      if (!std::getenv("TWALKER_DISABLE_GRAM_REARM"))
        gram_stability_rearm_pending_ = true;
    }
    if (ties > 1) result.tied_events += ties;
    // Lexicographic perturbation of a degenerate event.  Toggling the full
    // tie set produced long support cycles on sctap1/brandy.  Select the
    // lowest row deterministically; every chosen right face still passes
    // settle + acceptance, and alternate tied rows are tried on rejection.
    support = before_event;
    if (retained_tie_certificate) {
      for (auto row : tie_rows) support[row] = !support[row];
    } else if (!lexicographic_mode && tie_rows.size() > 1) {
      for (auto row : tie_rows) support[row] = !support[row];
    } else {
      support[tie_rows.front()] = !support[tie_rows.front()];
    }
    const auto event_support = support;
    t = next;
    // Keep the accepted left affine segment alive until a right face is
    // admitted.  A failed settle mutates its output through several rejected
    // supports; overwriting `solution` here discarded the only exact anchored
    // endpoint and forced later repairs to reconstruct a deficient face.
    FaceSolution event_solution;
    if (settle(support, t, event_solution)) {
      solution = std::move(event_solution);
    } else {
      if (solution.used_bound_core
          && last_direct_event_retry_pivot != pivot) {
        // A long sequence of inexpensive structured solves can accumulate a
        // few ulps in breakpoint times.  If the proposed right face then
        // fails its original-data gate, retry the *originating* ratio test
        // once from a direct factorization before entering any epsilon or
        // combinatorial repair.  This both corrects the step length and keeps
        // the accepted left face as the anchor.
        support = before_event;
        t = before_event_t;
        try {
          solution = solve_direct(support, true);
        } catch (const FaceDecline &error) {
          result.status = "event-origin polish declined: "
                          + std::string(error.what());
          result.pivots = pivot;
          result.t = t;
          break;
        }
        last_direct_event_retry_pivot = pivot;
        ++stability_refactors_;
        ++event_stability_refactors_;
        if (trace_path_ || std::getenv("TWALKER_BOUND_CORE_TRACE"))
          std::cerr << "EVENT_ORIGIN_REFACTOR pivot=" << pivot
                    << " t=" << std::setprecision(17) << t
                    << " rejected_t=" << next
                    << " reason=" << last_reject_ << '\n';
        --pivot;
        continue;
      }
      const bool repair_trace = std::getenv("TWALKER_REPAIR_TRACE");
      if (repair_trace)
        std::cerr << "REPAIR initial pivot=" << pivot
                  << " faces=" << face_solves_
                  << " reason=" << last_reject_ << '\n';
      bool repaired = false;
      const bool serial_first =
          !std::getenv("TWALKER_DISABLE_SERIAL_FIRST_AFTER_ROUND_CAP")
          && last_reject_ == "settle round cap";
      if (serial_first) {
        constexpr int exponent = -10;
        const double epsilon = std::pow(10.0, exponent)
                               * std::max(1.0, std::abs(next));
        auto trial = event_support;
        FaceSolution trial_solution;
        const int faces_before = face_solves_;
        const bool settled =
            settle(trial, next + epsilon, trial_solution, 50, true);
        if (repair_trace)
          std::cerr << "REPAIR serial_first pivot=" << pivot
                    << " exponent=" << exponent
                    << " settled=" << settled
                    << " faces=" << (face_solves_ - faces_before)
                    << " reason=" << last_reject_ << '\n';
        if (settled) {
          support = std::move(trial);
          solution = std::move(trial_solution);
          t = next + epsilon;
          ++result.epsilon_repairs;
          lexicographic_mode = true;
          repaired = true;
        }
      }
      for (int exponent : {-10, -9, -8, -7, -6}) {
        if (repaired) break;
        const double epsilon = std::pow(10.0, exponent)
                               * std::max(1.0, std::abs(next));
        auto trial = event_support;
        FaceSolution trial_solution;
        const int faces_before = face_solves_;
        const bool settled = settle(trial, next + epsilon, trial_solution, 20);
        if (repair_trace)
          std::cerr << "REPAIR batch pivot=" << pivot
                    << " exponent=" << exponent
                    << " settled=" << settled
                    << " faces=" << (face_solves_ - faces_before)
                    << " reason=" << last_reject_ << '\n';
        if (settled) {
          support = std::move(trial);
          solution = std::move(trial_solution);
          t = next + epsilon;
          ++result.epsilon_repairs;
          lexicographic_mode = true;
          repaired = true;
          break;
        }
      }
      // A failed five-scale batch settle is already strong evidence that the
      // local combinatorial repair is cycling.  The projection changes only
      // the support guess and is followed by the same exact face gate, so try
      // it before paying hundreds of one-row refactors.
      // The native projection is only competitive in the medium rectangular
      // regime measured here.  On the larger-column sctap1 and the small
      // models it failed to repair and merely added exact face solves.
      const bool single_qp_repair =
          std::getenv("TWALKER_QP_ALLOW_MULTIPLE_REPAIRS") == nullptr;
      if (!repaired && fixture_.n > 400 && fixture_.m < 300
          && (!single_qp_repair || result.qp_repairs == 0)) {
        const double projected_t = next + 1e-6 * std::max(1.0, std::abs(next));
        const auto projected = qp_projection(projected_t);
        double projection_scale = 1.0;
        for (double value : projected)
          projection_scale = std::max(projection_scale, std::abs(value));
        for (double relative : {1e-5}) {
          std::vector<std::uint8_t> trial(projected.size());
          for (std::size_t row = 0; row < projected.size(); ++row)
            trial[row] = projected[row] > relative * projection_scale;
          FaceSolution trial_solution;
          if (!trial.empty()
              && settle(trial, projected_t, trial_solution, 100)) {
            support = std::move(trial);
            solution = std::move(trial_solution);
            t = projected_t;
            ++result.qp_repairs;
            lexicographic_mode = true;
            repaired = true;
            break;
          }
        }
      }
      if (!repaired) {
        for (int exponent : {-10, -9, -8, -7, -6}) {
          const double epsilon = std::pow(10.0, exponent)
                                 * std::max(1.0, std::abs(next));
          auto trial = event_support;
          FaceSolution trial_solution;
          const int faces_before = face_solves_;
          const bool settled =
              settle(trial, next + epsilon, trial_solution, 50, true);
          if (repair_trace)
            std::cerr << "REPAIR serial pivot=" << pivot
                      << " exponent=" << exponent
                      << " settled=" << settled
                      << " faces=" << (face_solves_ - faces_before)
                      << " reason=" << last_reject_ << '\n';
          if (settled) {
            support = std::move(trial);
            solution = std::move(trial_solution);
            t = next + epsilon;
            ++result.epsilon_repairs;
            lexicographic_mode = true;
            repaired = true;
            break;
          }
        }
      }
      if (!repaired) {
        for (auto selected : tie_rows) {
          for (int exponent : {-10, -9, -8, -7, -6}) {
            const double epsilon = std::pow(10.0, exponent)
                                   * std::max(1.0, std::abs(next));
            auto trial = before_event;
            trial[selected] = !trial[selected];
            FaceSolution trial_solution;
            const int faces_before = face_solves_;
            const bool settled =
                settle(trial, next + epsilon, trial_solution, 20);
            if (repair_trace)
              std::cerr << "REPAIR subset pivot=" << pivot
                        << " row=" << selected
                        << " exponent=" << exponent
                        << " settled=" << settled
                        << " faces=" << (face_solves_ - faces_before)
                        << " reason=" << last_reject_ << '\n';
            if (settled) {
              support = std::move(trial);
              solution = std::move(trial_solution);
              t = next + epsilon;
              ++result.tie_subset_repairs;
              lexicographic_mode = true;
              repaired = true;
              break;
            }
          }
          if (repaired) break;
        }
      }
      // Last-resort degenerate-basis repair.  It does not displace a face
      // that any cheaper combinatorial/epsilon route can accept: it is tried
      // only after those routes are exhausted.  The accepted left endpoint
      // supplies the projection and multiplier, so no auxiliary QP is needed.
      if (!repaired && !std::getenv("TWALKER_DISABLE_RANK_COMPLETE_REPAIR")) {
        auto trial = event_support;
        FaceSolution trial_solution;
        double repaired_t = next;
        try {
          const auto &endpoint = solution;
          const bool critical_available =
              std::getenv("TWALKER_CRITICAL_RIGHT_REPAIR")
              || anchored_basis_repair_enabled();
          repaired = rank_complete_projection(next, endpoint, trial,
                                              trial_solution);
          // Quarantined negative experiment: the cold critical-cone QP was
          // accurate in its equalities but failed the cone/right-face gate on
          // Capri and cost 26 seconds.  Keep it opt-in for reproducibility.
          if (!repaired && critical_available) {
            repaired = critical_right_transition(
                next, endpoint, trial, trial_solution, repaired_t);
          }
        } catch (const FaceDecline &) {
          repaired = false;
        }
        if (repaired) {
          support = std::move(trial);
          solution = std::move(trial_solution);
          t = repaired_t;
          lexicographic_mode = true;
        }
        if (repair_trace)
          std::cerr << "REPAIR rank_complete pivot=" << pivot
                    << " repaired=" << repaired
                    << " faces=" << face_solves_ << '\n';
      }
      if (!repaired) {
        double corrected_t = next + 1e-8 * std::max(1.0, std::abs(next));
        for (int attempt = 0; attempt < 8 && corrected_t <= tmax; ++attempt) {
          auto trial = corrector_support(corrected_t);
          FaceSolution trial_solution;
          if (settle(trial, corrected_t, trial_solution, 100, true)) {
            support = std::move(trial);
            solution = std::move(trial_solution);
            t = corrected_t;
            ++result.corrector_repairs;
            lexicographic_mode = true;
            repaired = true;
            break;
          }
          corrected_t = 2.0 * corrected_t + 1.0;
        }
      }
      if (!repaired) {
          result.status = "breakpoint face rejected: " + last_reject_;
          result.pivots = pivot + 1;
          result.t = t;
          break;
      }
    }
    result.pivots = pivot + 1;
  }
  if (result.status.empty()) {
    result.status = "pivot cap";
    result.pivots = max_pivots;
    result.t = t;
  }
  result.face_solves = face_solves_;
  result.dense_fallbacks = dense_fallbacks_;
  result.rank_deficient_solves = rank_deficient_solves_;
  result.rank_deficit_sum = rank_deficit_sum_;
  result.max_rank_deficit = max_rank_deficit_;
  result.settle_rounds = settle_rounds_;
  result.selector_calls = selector_calls_;
  result.selector_ms = selector_ms_;
  result.terminal_gates = terminal_gates_;
  result.recovery_calls = recovery_calls_;
  result.recovery_ms = recovery_ms_;
  result.terminal_support_repairs = terminal_support_repairs_;
  result.terminal_support_repair_successes =
      terminal_support_repair_successes_;
  result.terminal_support_repair_iterations =
      terminal_support_repair_iterations_;
  result.terminal_support_repair_ms = terminal_support_repair_ms_;
  result.qp_calls = qp_calls_;
  result.qp_iterations = qp_iterations_;
  result.qp_ms = qp_ms_;
  result.rank_complete_calls = rank_complete_calls_;
  result.rank_complete_repairs = rank_complete_repairs_;
  result.rank_complete_ms = rank_complete_ms_;
  result.critical_right_calls = critical_right_calls_;
  result.critical_right_repairs = critical_right_repairs_;
  result.critical_right_ms = critical_right_ms_;
  result.rank_lift_audit = rank_lift_audit_;
  result.maintained_rowspace_audit = maintained_rowspace_audit_;
  result.gram_fast_solves = gram_fast_solves_;
  result.gram_declines = gram_declines_;
  result.bound_core_solves = bound_core_solves_;
  result.bound_core_declines = bound_core_declines_;
  result.bound_core_maximum_border = bound_core_solver_.stats().maximum_border;
  result.bound_core_refinements = bound_core_solver_.stats().refinements;
  result.bound_core_total_ms = bound_core_solver_.stats().total_ms;
  result.bound_core_audits = bound_core_audits_;
  result.bound_core_audit_violations = bound_core_audit_violations_;
  result.bound_core_audit_max_error = bound_core_audit_max_error_;
  result.revised_column_solves = revised_column_solves_;
  result.revised_column_declines = revised_column_declines_;
  result.revised_bound_audits = revised_bound_audits_;
  result.revised_bound_violations = revised_bound_violations_;
  result.revised_bound_worst_ratio = revised_bound_worst_ratio_;
  result.revised_bound_max_width = revised_bound_max_width_;
  result.revised_bound_max_actual = revised_bound_max_actual_;
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  if (revised_column_solver_) {
    const auto &revised_stats =
        static_cast<revised::RevisedColumnSolver *>(revised_column_solver_)
            ->stats();
    result.revised_column_rebuilds = revised_stats.rebuilds;
    result.revised_column_rank_changes = revised_stats.rank_changes;
    result.revised_column_retirements = revised_stats.retirements;
    result.revised_column_rebuild_ms = revised_stats.rebuild_ms;
    result.revised_column_seed_ms = revised_stats.direct_seed_ms;
    result.revised_column_transition_ms = revised_stats.transition_ms;
    result.revised_column_solve_ms = revised_stats.solve_ms;
    result.revised_column_products_ms = revised_stats.products_ms;
    result.revised_column_coefficient_ms = revised_stats.coefficient_ms;
    result.revised_column_projection_ms = revised_stats.projection_ms;
    result.revised_column_residual_ms = revised_stats.residual_ms;
    result.revised_column_direct_seeds = revised_stats.direct_seeds;
  }
  if (maintained_rowspace_solver_ || maintained_deficient_qr_solver_
      || maintained_svd_face_solver_) {
    if (maintained_svd_audit_enabled_ || maintained_svd_live_enabled_) {
      const auto &stats =
          static_cast<revised::MaintainedSvdFaceSolver *>(
              maintained_svd_face_solver_)->stats();
      result.maintained_rowspace_audit.local_transitions = stats.transitions;
      result.maintained_rowspace_audit.additions = stats.additions;
      result.maintained_rowspace_audit.removals = stats.removals;
      result.maintained_rowspace_audit.rank_change_declines =
          stats.rank_change_declines;
      result.maintained_rowspace_audit.numerical_declines =
          stats.numerical_declines;
      result.maintained_rowspace_audit.seed_ms = stats.seed_ms;
      result.maintained_rowspace_audit.transition_ms = stats.transition_ms;
      result.maintained_rowspace_audit.solve_ms = stats.solve_ms;
      result.maintained_rowspace_audit.worst_rowspace_residual =
          stats.worst_representation_residual;
      result.maintained_rowspace_audit.worst_slope_residual =
          stats.worst_piece_residual;
    } else if (deficient_face_audit_enabled_ || deficient_face_live_enabled_
        || std::getenv("TWALKER_MAINTAINED_QR_AUDIT")) {
      const auto &stats =
          static_cast<revised::MaintainedDeficientQrSolver *>(
              maintained_deficient_qr_solver_)->stats();
      result.maintained_rowspace_audit.local_transitions = stats.transitions;
      result.maintained_rowspace_audit.additions = stats.additions;
      result.maintained_rowspace_audit.removals = stats.removals;
      result.maintained_rowspace_audit.rank_change_declines =
          stats.rank_change_declines;
      result.maintained_rowspace_audit.numerical_declines =
          stats.numerical_declines;
      result.maintained_rowspace_audit.seed_ms = stats.seed_ms;
      result.maintained_rowspace_audit.transition_ms = stats.transition_ms;
      result.maintained_rowspace_audit.solve_ms = stats.solve_ms;
      result.maintained_rowspace_audit.worst_rowspace_residual =
          stats.worst_representation_residual;
      result.maintained_rowspace_audit.worst_slope_residual =
          stats.worst_slope_residual;
    } else if (std::getenv("TWALKER_MAINTAINED_PINV_AUDIT")) {
      const auto &stats =
          static_cast<revised::RevisedColumnSolver *>(centered_slope_solver_)
              ->stats();
      result.maintained_rowspace_audit.local_transitions =
          stats.local_transitions;
      result.maintained_rowspace_audit.additions = stats.row_additions;
      result.maintained_rowspace_audit.removals = stats.row_removals;
      result.maintained_rowspace_audit.rank_increases = stats.rank_increases;
      result.maintained_rowspace_audit.rank_decreases = stats.rank_decreases;
      result.maintained_rowspace_audit.numerical_declines = stats.declines;
      result.maintained_rowspace_audit.seed_ms = stats.direct_seed_ms;
      result.maintained_rowspace_audit.transition_ms = stats.transition_ms;
      result.maintained_rowspace_audit.solve_ms =
          stats.solve_ms + stats.products_ms + stats.coefficient_ms
          + stats.projection_ms + stats.residual_ms;
    } else {
      const auto &stats =
          static_cast<revised::MaintainedRowspaceSolver *>(
              maintained_rowspace_solver_)->stats();
      result.maintained_rowspace_audit.local_transitions =
          stats.local_transitions;
      result.maintained_rowspace_audit.additions = stats.additions;
      result.maintained_rowspace_audit.removals = stats.removals;
      result.maintained_rowspace_audit.rank_increases = stats.rank_increases;
      result.maintained_rowspace_audit.rank_decreases = stats.rank_decreases;
      result.maintained_rowspace_audit.rank_change_declines =
          stats.rank_change_declines;
      result.maintained_rowspace_audit.numerical_declines =
          stats.numerical_declines;
      result.maintained_rowspace_audit.refactors = stats.refactors;
      result.maintained_rowspace_audit.seed_ms = stats.seed_ms;
      result.maintained_rowspace_audit.transition_ms = stats.transition_ms;
      result.maintained_rowspace_audit.solve_ms = stats.solve_ms;
      result.maintained_rowspace_audit.worst_rowspace_residual =
          stats.worst_rowspace_residual;
      result.maintained_rowspace_audit.worst_orthogonality =
          stats.worst_orthogonality;
      result.maintained_rowspace_audit.worst_slope_residual =
          stats.worst_slope_residual;
    }
  }
#endif
  result.stability_refactors = stability_refactors_;
  result.settle_stability_refactors = settle_stability_refactors_;
  result.terminal_stability_refactors = terminal_stability_refactors_;
  result.event_stability_refactors = event_stability_refactors_;
  result.event_interval = event_interval_;
  result.qr_settle_decision_checks = qr_settle_decision_checks_;
  result.qr_settle_decision_matches = qr_settle_decision_matches_;
  result.qr_event_decision_checks = qr_event_decision_checks_;
  result.qr_event_decision_matches = qr_event_decision_matches_;
  result.qr_event_tie_set_matches = qr_event_tie_set_matches_;
  result.retained_tie_polishes = retained_tie_polishes_;
  result.retained_tie_certificates = retained_tie_certificates_;
  result.retained_tie_declines = retained_tie_declines_;
  result.gram_final_rcond = gram_solver_.rcond();
  result.gram_stats = gram_solver_.stats();
  result.direct_stats = rank_lift_live_solver_
                            ? rank_lift_live_solver_->stats()
                            : solver_.stats();
  return result;
}

}  // namespace twalker
