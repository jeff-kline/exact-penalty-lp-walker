#include "bound_core_face_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

extern "C" {
void dgetrf_(const int *m, const int *n, double *a, const int *lda,
             int *ipiv, int *info);
void dgetrs_(const char *trans, const int *n, const int *nrhs,
             const double *a, const int *lda, const int *ipiv, double *b,
             const int *ldb, int *info);
void dgecon_(const char *norm, const int *n, const double *a, const int *lda,
             const double *anorm, double *rcond, double *work, int *iwork,
             int *info);
void dgesdd_(const char *jobz, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt,
             const int *ldvt, double *work, const int *lwork, int *iwork,
             int *info);
}

namespace twalker {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double stable_norm2(const std::vector<double> &values) {
  double scale = 0.0, sum = 1.0;
  for (double value : values) {
    const double magnitude = std::abs(value);
    if (magnitude == 0.0) continue;
    if (scale < magnitude) {
      const double ratio = scale / magnitude;
      sum = 1.0 + sum * ratio * ratio;
      scale = magnitude;
    } else {
      const double ratio = magnitude / scale;
      sum += ratio * ratio;
    }
  }
  return scale == 0.0 ? 0.0 : scale * std::sqrt(sum);
}

}  // namespace

BoundCoreFaceSolver::BoundCoreFaceSolver(
    const Fixture &fixture, std::vector<double> target_shift)
    : fixture_(fixture), target_shift_(std::move(target_shift)),
      unit_column_(fixture.n, -1), unit_value_(fixture.n, 0.0) {
  if (!target_shift_.empty() && target_shift_.size() != fixture_.n)
    return;
  int unit_rows = 0;
  for (std::uint32_t row = 0; row < fixture_.n; ++row) {
    const auto count = fixture_.indptr[row + 1] - fixture_.indptr[row];
    if (count == 1) {
      const auto p = fixture_.indptr[row];
      unit_column_[row] = static_cast<std::int32_t>(fixture_.indices[p]);
      unit_value_[row] = fixture_.values[p];
      if (std::isfinite(unit_value_[row]) && unit_value_[row] != 0.0)
        ++unit_rows;
      else
        unit_column_[row] = -1;
    } else {
      ++general_row_count_;
    }
  }
  // Eligibility is decided by the ACTIVE face below.  A model may contain
  // many general rows while every face used by the Newton seed contains only
  // a small core (Ship04s is the important example).  Rejecting such a model
  // globally forced a large SVD at every seed iteration.  The local core,
  // zero-diagonal, border-size, rank and original-operator residual gates
  // remain authoritative, so broadening this census cannot admit a bad solve.
  structurally_eligible_ = fixture_.m > 0
                           && unit_rows >= static_cast<int>(fixture_.m);
}

bool BoundCoreFaceSolver::solve(const std::vector<std::uint32_t> &rows,
                                FaceSolution &solution) {
  const auto total_start = Clock::now();
  ++stats_.calls;
  auto decline_structure = [&]() {
    ++stats_.structural_declines;
    stats_.total_ms += elapsed_ms(total_start);
    return false;
  };
  auto decline_numerical = [&]() {
    ++stats_.numerical_declines;
    stats_.total_ms += elapsed_ms(total_start);
    return false;
  };
  if (!structurally_eligible_) return decline_structure();

  const int m = static_cast<int>(fixture_.m);
  std::vector<double> diagonal(m, 0.0), atb(m, 0.0), ats(m, 0.0);
  std::vector<std::uint32_t> core_rows;
  core_rows.reserve(general_row_count_);
  std::vector<std::uint8_t> seen(fixture_.n, 0);
  for (auto row : rows) {
    if (row >= fixture_.n || seen[row]) return decline_structure();
    seen[row] = 1;
    const auto unit_column = unit_column_[row];
    if (unit_column >= 0) {
      const double value = unit_value_[row];
      diagonal[unit_column] += value * value;
      atb[unit_column] += value * fixture_.b[row];
      if (!target_shift_.empty())
        ats[unit_column] += value * target_shift_[row];
    } else {
      core_rows.push_back(row);
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto column = fixture_.indices[p];
        const double value = fixture_.values[p];
        atb[column] += value * fixture_.b[row];
        if (!target_shift_.empty())
          ats[column] += value * target_shift_[row];
      }
    }
  }

  std::vector<int> positive, zero, zero_position(m, -1);
  positive.reserve(m);
  zero.reserve(8);
  for (int column = 0; column < m; ++column) {
    if (diagonal[column] > 0.0) {
      positive.push_back(column);
    } else {
      zero_position[column] = static_cast<int>(zero.size());
      zero.push_back(column);
    }
  }
  const int core = static_cast<int>(core_rows.size());
  const int border = core + static_cast<int>(zero.size());
  stats_.maximum_border = std::max(stats_.maximum_border, border);
  const bool wide_rank_revealing =
      std::getenv("TWALKER_BOUND_CORE_WIDE_SHADOW") != nullptr
      || std::getenv("TWALKER_BOUND_CORE_WIDE_LIVE") != nullptr;
  const std::size_t maximum_zero = wide_rank_revealing ? 32 : 8;
  if (core > 48 || zero.size() > maximum_zero || border <= 0 || border > 56)
    return decline_structure();

  // Dense C, column-major: active general rows by original variables.
  std::vector<double> c(static_cast<std::size_t>(core) * m, 0.0);
  for (int local = 0; local < core; ++local) {
    const auto row = core_rows[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      c[local + static_cast<std::size_t>(core) * fixture_.indices[p]] =
          fixture_.values[p];
  }

  const auto factor_start = Clock::now();
  // [I + C_P D_P^-1 C_P'   -C_Z] [w  ] = [C_P D_P^-1 q_P]
  // [C_Z'                         0  ] [x_Z]   [q_Z]
  std::vector<double> bordered(static_cast<std::size_t>(border) * border,
                               0.0);
  for (int i = 0; i < core; ++i)
    bordered[i + static_cast<std::size_t>(border) * i] = 1.0;
  for (int column : positive) {
    const double inverse = 1.0 / diagonal[column];
    const double *column_values = c.data() + static_cast<std::size_t>(core)
                                                * column;
    for (int j = 0; j < core; ++j)
      for (int i = 0; i < core; ++i)
        bordered[i + static_cast<std::size_t>(border) * j] +=
            inverse * column_values[i] * column_values[j];
  }
  for (int local_zero = 0; local_zero < static_cast<int>(zero.size());
       ++local_zero) {
    const double *column_values =
        c.data() + static_cast<std::size_t>(core) * zero[local_zero];
    const int zindex = core + local_zero;
    for (int i = 0; i < core; ++i) {
      bordered[i + static_cast<std::size_t>(border) * zindex] =
          -column_values[i];
      bordered[zindex + static_cast<std::size_t>(border) * i] =
          column_values[i];
    }
  }
  const std::vector<double> bordered_original = bordered;
  std::vector<double> row_scale(border, 1.0), column_scale(border, 1.0);
  std::vector<double> bordered_for_bounds = bordered_original;
  double bordered_inf_norm = 0.0;
  const bool use_rank_revealing_svd = wide_rank_revealing;
  std::vector<int> pivots;
  std::vector<double> singular, left_vectors, right_transpose;
  int info = 0;
  double rcond = 0.0;
  if (use_rank_revealing_svd) {
    auto compute_svd = [&](const std::vector<double> &matrix,
                           std::vector<double> &values,
                           std::vector<double> &left,
                           std::vector<double> &right) {
      values.resize(border);
      left.resize(static_cast<std::size_t>(border) * border);
      right.resize(static_cast<std::size_t>(border) * border);
      std::vector<int> iwork(8 * border);
      const char all_vectors = 'A';
      int lwork = -1, svd_info = 0;
      double query = 0.0;
      auto svd_matrix = matrix;
      dgesdd_(&all_vectors, &border, &border, svd_matrix.data(), &border,
              values.data(), left.data(), &border, right.data(), &border,
              &query, &lwork, iwork.data(), &svd_info);
      if (svd_info != 0 || !std::isfinite(query) || query < 1.0)
        return false;
      lwork = static_cast<int>(std::ceil(query));
      std::vector<double> work(lwork);
      svd_matrix = matrix;
      dgesdd_(&all_vectors, &border, &border, svd_matrix.data(), &border,
              values.data(), left.data(), &border, right.data(), &border,
              work.data(), &lwork, iwork.data(), &svd_info);
      return svd_info == 0 && !values.empty() && values.front() > 0.0;
    };
    std::vector<double> unscaled_singular, unscaled_left, unscaled_right;
    if (!compute_svd(bordered_original, unscaled_singular, unscaled_left,
                     unscaled_right))
      return decline_numerical();
    const double rank_cutoff =
        unscaled_singular.front()
        * std::max<std::size_t>({rows.size(), fixture_.m,
                                 static_cast<std::size_t>(border)})
        * std::numeric_limits<double>::epsilon();
    if (!(unscaled_singular.back() > rank_cutoff))
      return decline_numerical();

    // The retained core can be perfectly nonsingular yet badly scaled.  A
    // few max-norm row/column passes are cheap at k<=56 and make both the SVD
    // solve and its posterior error bound meaningful.  The unscaled SVD above
    // remains the rank gate, so equilibration cannot manufacture rank.
    bordered_for_bounds = bordered_original;
    for (int iteration = 0; iteration < 3; ++iteration) {
      for (int row = 0; row < border; ++row) {
        double maximum = 0.0;
        for (int column = 0; column < border; ++column)
          maximum = std::max(
              maximum,
              std::abs(bordered_for_bounds[
                  row + static_cast<std::size_t>(border) * column]));
        if (!(maximum > 0.0) || !std::isfinite(maximum))
          return decline_numerical();
        const double factor = 1.0 / maximum;
        row_scale[row] *= factor;
        for (int column = 0; column < border; ++column)
          bordered_for_bounds[
              row + static_cast<std::size_t>(border) * column] *= factor;
      }
      for (int column = 0; column < border; ++column) {
        double maximum = 0.0;
        for (int row = 0; row < border; ++row)
          maximum = std::max(
              maximum,
              std::abs(bordered_for_bounds[
                  row + static_cast<std::size_t>(border) * column]));
        if (!(maximum > 0.0) || !std::isfinite(maximum))
          return decline_numerical();
        const double factor = 1.0 / maximum;
        column_scale[column] *= factor;
        for (int row = 0; row < border; ++row)
          bordered_for_bounds[
              row + static_cast<std::size_t>(border) * column] *= factor;
      }
    }
    if (!compute_svd(bordered_for_bounds, singular, left_vectors,
                     right_transpose))
      return decline_numerical();
    rcond = singular.back() / singular.front();
  } else {
    double one_norm = 0.0;
    for (int column = 0; column < border; ++column) {
      double sum = 0.0;
      for (int row = 0; row < border; ++row)
        sum += std::abs(
            bordered[row + static_cast<std::size_t>(border) * column]);
      one_norm = std::max(one_norm, sum);
    }
    pivots.resize(border);
    dgetrf_(&border, &border, bordered.data(), &border, pivots.data(), &info);
    if (info != 0) return decline_numerical();
    const char one_norm_code = '1';
    std::vector<double> condition_work(4 * border);
    std::vector<int> condition_iwork(border);
    dgecon_(&one_norm_code, &border, bordered.data(), &border, &one_norm,
            &rcond, condition_work.data(), condition_iwork.data(), &info);
  }
  for (int row = 0; row < border; ++row) {
    double row_sum = 0.0;
    for (int column = 0; column < border; ++column)
      row_sum += std::abs(
          bordered_for_bounds[row
                              + static_cast<std::size_t>(border) * column]);
    bordered_inf_norm = std::max(bordered_inf_norm, row_sum);
  }
  stats_.factor_ms += elapsed_ms(factor_start);
  stats_.minimum_rcond = std::min(stats_.minimum_rcond, rcond);
  if (std::getenv("TWALKER_BOUND_CORE_TRACE") && rcond < 1e-8)
    std::cerr << "bound-core rows=" << rows.size() << " border=" << border
              << " zeros=" << zero.size() << " rcond=" << rcond << '\n';
  // SPQR's numerical-rank decision remains authoritative at the boundary.
  // The small system must not manufacture an inverse on a face that is
  // numerically deficient under that policy.
  if (info != 0 || !std::isfinite(rcond) || rcond < 1e-20)
    return decline_numerical();

  auto reduced_rhs = [&](const std::vector<double> &q) {
    std::vector<double> small(static_cast<std::size_t>(border), 0.0);
    for (int column : positive) {
      const double scaled = q[column] / diagonal[column];
      const double *column_values =
          c.data() + static_cast<std::size_t>(core) * column;
      for (int i = 0; i < core; ++i)
        small[i] += column_values[i] * scaled;
    }
    for (int local_zero = 0; local_zero < static_cast<int>(zero.size());
         ++local_zero)
      small[core + local_zero] = q[zero[local_zero]];
    return small;
  };
  auto solve_h = [&](const std::vector<double> &q, std::vector<double> &x,
                     std::vector<double> *reduced_solution) {
    std::vector<double> small = reduced_rhs(q);
    if (use_rank_revealing_svd) {
      for (int row = 0; row < border; ++row) small[row] *= row_scale[row];
      const std::vector<double> scaled_rhs = small;
      std::vector<double> transformed(border, 0.0), answer(border, 0.0);
      auto apply_inverse = [&](const std::vector<double> &rhs,
                               std::vector<double> &result) {
        for (int component = 0; component < border; ++component) {
          long double product = 0.0L;
          for (int row = 0; row < border; ++row)
            product += static_cast<long double>(
                           left_vectors[
                               row + static_cast<std::size_t>(border)
                                         * component])
                       * rhs[row];
          transformed[component] =
              static_cast<double>(product) / singular[component];
        }
        for (int row = 0; row < border; ++row) {
          long double product = 0.0L;
          for (int component = 0; component < border; ++component)
            product += static_cast<long double>(
                           right_transpose[
                               component + static_cast<std::size_t>(border)
                                               * row])
                       * transformed[component];
          result[row] = static_cast<double>(product);
        }
      };
      apply_inverse(scaled_rhs, answer);
      for (int iteration = 0; iteration < 2; ++iteration) {
        std::vector<double> residual(border, 0.0);
        long double residual_inf = 0.0L, scale_inf = 1.0L;
        for (int row = 0; row < border; ++row) {
          long double product = 0.0L, magnitude = 0.0L;
          for (int column = 0; column < border; ++column) {
            const long double value = bordered_for_bounds[
                row + static_cast<std::size_t>(border) * column];
            product += value * answer[column];
            magnitude += std::abs(value)
                         * std::abs(static_cast<long double>(answer[column]));
          }
          const long double value =
              static_cast<long double>(scaled_rhs[row]) - product;
          residual[row] = static_cast<double>(value);
          residual_inf = std::max(residual_inf, std::abs(value));
          scale_inf = std::max(
              scale_inf,
              magnitude + std::abs(static_cast<long double>(scaled_rhs[row])));
        }
        if (residual_inf
            <= 64.0L * std::numeric_limits<double>::epsilon() * scale_inf)
          break;
        std::vector<double> correction(border, 0.0);
        apply_inverse(residual, correction);
        for (int component = 0; component < border; ++component)
          answer[component] += correction[component];
        ++stats_.refinements;
      }
      for (int component = 0; component < border; ++component)
        answer[component] *= column_scale[component];
      small.swap(answer);
    } else {
      const char no_transpose = 'N';
      const int one = 1;
      int solve_info = 0;
      dgetrs_(&no_transpose, &border, &one, bordered.data(), &border,
              pivots.data(), small.data(), &border, &solve_info);
      if (solve_info != 0) return false;
    }
    x.assign(m, 0.0);
    for (int column : positive) {
      long double correction = 0.0L;
      const double *column_values =
          c.data() + static_cast<std::size_t>(core) * column;
      for (int i = 0; i < core; ++i)
        correction += static_cast<long double>(column_values[i]) * small[i];
      x[column] = (q[column] - static_cast<double>(correction))
                  / diagonal[column];
    }
    for (int local_zero = 0; local_zero < static_cast<int>(zero.size());
         ++local_zero)
      x[zero[local_zero]] = small[core + local_zero];
    if (reduced_solution) *reduced_solution = small;
    return true;
  };

  std::vector<double> qa(m), qc(m), ua, uc, reduced_a, reduced_c;
  for (int column = 0; column < m; ++column) {
    qa[column] = -atb[column];
    qc[column] = fixture_.d[column] - ats[column];
  }
  if (!solve_h(qa, ua, &reduced_a) || !solve_h(qc, uc, &reduced_c))
    return decline_numerical();

  // Correct the semi-normal solution in original coordinates.  This is only
  // a small bordered solve; residual formation dominates and is bounded to
  // two sweeps.  Long double prevents the correction RHS from losing the
  // low bits that it is intended to restore.
  auto refine = [&](const std::vector<double> &q, std::vector<double> &x,
                    std::vector<double> &reduced_solution) {
    for (int iteration = 0; iteration < 2; ++iteration) {
      std::vector<long double> residual(q.begin(), q.end());
      for (int column = 0; column < m; ++column)
        residual[column] -=
            static_cast<long double>(diagonal[column]) * x[column];
      for (int local = 0; local < core; ++local) {
        long double product = 0.0L;
        for (auto p = fixture_.indptr[core_rows[local]];
             p < fixture_.indptr[core_rows[local] + 1]; ++p)
          product += static_cast<long double>(fixture_.values[p])
                     * x[fixture_.indices[p]];
        for (auto p = fixture_.indptr[core_rows[local]];
             p < fixture_.indptr[core_rows[local] + 1]; ++p)
          residual[fixture_.indices[p]] -=
              static_cast<long double>(fixture_.values[p]) * product;
      }
      double residual_inf = 0.0, q_inf = 1.0;
      std::vector<double> correction_rhs(m);
      for (int column = 0; column < m; ++column) {
        correction_rhs[column] = static_cast<double>(residual[column]);
        residual_inf = std::max(residual_inf,
                                std::abs(correction_rhs[column]));
        q_inf = std::max(q_inf, std::abs(q[column]));
      }
      if (residual_inf <= 5e-13 * q_inf) return true;
      std::vector<double> correction, reduced_correction;
      if (!solve_h(correction_rhs, correction, &reduced_correction))
        return false;
      for (int column = 0; column < m; ++column) x[column] += correction[column];
      for (int component = 0; component < border; ++component)
        reduced_solution[component] += reduced_correction[component];
      ++stats_.refinements;
    }
    return true;
  };
  if (!refine(qa, ua, reduced_a) || !refine(qc, uc, reduced_c))
    return decline_numerical();

  const auto products_start = Clock::now();
  solution = FaceSolution{};
  solution.rows = rows;
  solution.rank = fixture_.m;
  solution.core_diagonal_ratio = 1.0;
  // Guard all discrete decisions.  For the rank-revealing bordered route,
  // retain the two reduced solutions and their residuals so the walker can
  // bound the combined affine quantity -t*ua+uc without destroying its
  // cancellation at large t.
  solution.used_extended_gram = true;
  solution.used_bound_core = true;
  solution.ua_relative_error_bound = 1e-10;
  solution.uc_relative_error_bound = 1e-10;
  if (use_rank_revealing_svd) {
    std::vector<double> scaled_a(border), scaled_c(border);
    for (int component = 0; component < border; ++component) {
      scaled_a[component] = reduced_a[component] / column_scale[component];
      scaled_c[component] = reduced_c[component] / column_scale[component];
    }
    auto reduced_residual = [&](const std::vector<double> &q,
                                const std::vector<double> &head) {
      auto residual = reduced_rhs(q);
      for (int row = 0; row < border; ++row)
        residual[row] *= row_scale[row];
      for (int column = 0; column < border; ++column)
        for (int row = 0; row < border; ++row)
          residual[row] -=
              bordered_for_bounds[
                  row + static_cast<std::size_t>(border) * column]
              * head[column];
      return residual;
    };
    solution.affine_bound_valid = true;
    solution.reduced_residual_a = reduced_residual(qa, scaled_a);
    solution.reduced_residual_c = reduced_residual(qc, scaled_c);
    solution.reduced_head_a = scaled_a;
    solution.reduced_head_c = scaled_c;
    solution.reduced_gram_inf = bordered_inf_norm;
    // dgesdd supplies a two-norm condition estimate.  Dividing by sqrt(k)
    // safely converts the inverse estimate for the infinity-norm residual
    // propagation used by Walker.
    solution.reduced_rcond =
        rcond / std::sqrt(static_cast<double>(std::max(1, border)));
    double projection_inf_norm = 0.0;
    for (int column : positive) {
      double row_sum = 0.0;
      for (int local = 0; local < core; ++local)
        row_sum += std::abs(
            c[local + static_cast<std::size_t>(core) * column]
            * column_scale[local]);
      projection_inf_norm = std::max(
          projection_inf_norm, row_sum / diagonal[column]);
    }
    for (int local_zero = 0; local_zero < static_cast<int>(zero.size());
         ++local_zero)
      projection_inf_norm = std::max(
          projection_inf_norm, std::abs(column_scale[core + local_zero]));
    solution.projection_inf_norm = projection_inf_norm;
    solution.product_projection_inf_norm.assign(fixture_.n, 0.0);
    for (std::uint32_t row = 0; row < fixture_.n; ++row) {
      std::vector<long double> product_map(border, 0.0L);
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const int column = static_cast<int>(fixture_.indices[p]);
        const long double value = fixture_.values[p];
        if (diagonal[column] > 0.0) {
          const double *column_values =
              c.data() + static_cast<std::size_t>(core) * column;
          for (int local = 0; local < core; ++local)
            product_map[local] -= value * column_values[local]
                                  / diagonal[column];
        } else {
          const int local_zero = zero_position[column];
          if (local_zero >= 0) product_map[core + local_zero] += value;
        }
      }
      long double row_norm = 0.0L;
      for (int component = 0; component < border; ++component)
        row_norm += std::abs(product_map[component]
                             * column_scale[component]);
      solution.product_projection_inf_norm[row] =
          static_cast<double>(row_norm);
    }

    auto coefficient_bound = [&](const std::vector<double> &residual,
                                 const std::vector<double> &head,
                                 const std::vector<double> &coefficient) {
      double residual_inf = 0.0, head_inf = 0.0, coefficient_inf = 0.0;
      for (double value : residual)
        residual_inf = std::max(residual_inf, std::abs(value));
      for (double value : head)
        head_inf = std::max(head_inf, std::abs(value));
      for (double value : coefficient)
        coefficient_inf = std::max(coefficient_inf, std::abs(value));
      const double head_error =
          10.0 * residual_inf
          / (solution.reduced_rcond * solution.reduced_gram_inf);
      const double absolute =
          projection_inf_norm * head_error
          + 64.0 * std::numeric_limits<double>::epsilon()
                * projection_inf_norm * std::max(1.0, head_inf);
      return absolute / std::max(1.0, coefficient_inf);
    };
    solution.ua_relative_error_bound = coefficient_bound(
        solution.reduced_residual_a, scaled_a, ua);
    solution.uc_relative_error_bound = coefficient_bound(
        solution.reduced_residual_c, scaled_c, uc);
    if (!std::isfinite(solution.ua_relative_error_bound)
        || !std::isfinite(solution.uc_relative_error_bound))
      return decline_numerical();
    if (std::getenv("TWALKER_BOUND_CORE_TRACE")) {
      auto vector_inf = [](const std::vector<double> &values) {
        double result = 0.0;
        for (double value : values) result = std::max(result, std::abs(value));
        return result;
      };
      std::cerr << "bound-core bounds residual_a="
                << vector_inf(solution.reduced_residual_a)
                << " residual_c=" << vector_inf(solution.reduced_residual_c)
                << " head_a=" << vector_inf(scaled_a)
                << " head_c=" << vector_inf(scaled_c)
                << " Kinf=" << bordered_inf_norm
                << " adjusted_rcond=" << solution.reduced_rcond
                << " projection=" << projection_inf_norm
                << " ua_rel=" << solution.ua_relative_error_bound
                << " uc_rel=" << solution.uc_relative_error_bound << '\n';
    }
  }
  if (std::getenv("TWALKER_BOUND_CORE_UNGUARDED")) {
    solution.used_extended_gram = false;
    solution.affine_bound_valid = false;
  }
  solution.ua = std::move(ua);
  solution.uc = std::move(uc);
  solution.bua.assign(fixture_.n, 0.0);
  solution.buc.assign(fixture_.n, 0.0);
  for (std::uint32_t row = 0; row < fixture_.n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const double value = fixture_.values[p];
      solution.bua[row] += value * solution.ua[column];
      solution.buc[row] += value * solution.uc[column];
    }
  stats_.products_ms += elapsed_ms(products_start);

  solution.g.resize(rows.size());
  solution.h.resize(rows.size());
  std::vector<double> transpose_g(m, 0.0), dual(m, 0.0);
  double g_scale = 1.0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    solution.g[local] = fixture_.b[row] + solution.bua[row];
    solution.h[local] = (target_shift_.empty() ? 0.0 : target_shift_[row])
                        + solution.buc[row];
    g_scale = std::max(g_scale, std::abs(solution.g[local]));
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const double value = fixture_.values[p];
      transpose_g[column] += value * solution.g[local];
      dual[column] += value * solution.h[local];
    }
  }
  for (int column = 0; column < m; ++column) dual[column] -= fixture_.d[column];
  solution.dres = stable_norm2(dual) / std::max(1.0, stable_norm2(fixture_.d));
  solution.piece_residual = stable_norm2(transpose_g)
                            / std::max(1.0, g_scale * std::sqrt(m));
  stats_.worst_dres = std::max(stats_.worst_dres, solution.dres);
  stats_.worst_piece_residual =
      std::max(stats_.worst_piece_residual, solution.piece_residual);
  if (!std::isfinite(solution.dres) || !std::isfinite(solution.piece_residual)
      // The frozen Fit1d oracle itself carries about 2e-9 normalized dual
      // residual.  This fast lane is still two orders tighter than the
      // walker's global 1e-7 certificate, while exact oracle/decision audits
      // remain responsible for promotion.
      || solution.dres > 1e-8 || solution.piece_residual > 1e-8)
    return decline_numerical();
  ++stats_.accepted;
  stats_.total_ms += elapsed_ms(total_start);
  return true;
}

}  // namespace twalker
