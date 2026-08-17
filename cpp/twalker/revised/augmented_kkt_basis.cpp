#include "augmented_kkt_basis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>

extern "C" {
void dgesdd_(const char *jobz, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt,
             const int *ldvt, double *work, const int *lwork, int *iwork,
             int *info);
}

namespace twalker::revised {
namespace {

double inf_norm(const std::vector<double> &values) {
  double result = 0.0;
  for (double value : values) result = std::max(result, std::abs(value));
  return result;
}

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

}  // namespace

bool AugmentedKktBasis::ensure_factor(
    const std::vector<std::uint8_t> &support) {
  if (support == factor_support_ && factor_active_ > 0) {
    ++stats_.factor_reuses;
    return true;
  }
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  factor_rows_.clear();
  for (int row = 0; row < n; ++row)
    if (support[row]) factor_rows_.push_back(row);
  factor_active_ = static_cast<int>(factor_rows_.size());
  if (factor_active_ <= 0) {
    last_failure_ = "empty free face";
    return false;
  }

  const auto factor_begin = std::chrono::steady_clock::now();
  std::vector<double> a(static_cast<std::size_t>(factor_active_) * m, 0.0);
  for (int local = 0; local < factor_active_; ++local) {
    const auto row = factor_rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      a[local + static_cast<std::size_t>(factor_active_)
                    * fixture_.indices[p]] = fixture_.values[p];
  }
  const int thin = std::min(factor_active_, m);
  factor_singular_.assign(thin, 0.0);
  factor_u_.assign(static_cast<std::size_t>(factor_active_) * factor_active_,
                   0.0);
  factor_vt_.assign(static_cast<std::size_t>(m) * m, 0.0);
  std::vector<int> iwork(8 * std::max(1, thin));
  const char job = 'A';
  const int lda = factor_active_, ldu = factor_active_, ldvt = m;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &factor_active_, &m, a.data(), &lda,
          factor_singular_.data(), factor_u_.data(), &ldu, factor_vt_.data(),
          &ldvt, &query, &lwork, iwork.data(), &info);
  if (info != 0 || !std::isfinite(query) || query < 1.0) {
    last_failure_ = "SVD workspace query failed";
    return false;
  }
  lwork = static_cast<int>(std::ceil(query));
  std::vector<double> work(lwork);
  dgesdd_(&job, &factor_active_, &m, a.data(), &lda,
          factor_singular_.data(), factor_u_.data(), &ldu, factor_vt_.data(),
          &ldvt, work.data(), &lwork, iwork.data(), &info);
  if (info != 0) {
    last_failure_ = "SVD factorization failed";
    return false;
  }
  factor_rank_ = 0;
  const double cutoff = thin && factor_singular_.front() > 0.0
      ? factor_singular_.front() * std::max(factor_active_, m)
            * 16.0 * std::numeric_limits<double>::epsilon()
      : 0.0;
  while (factor_rank_ < thin && factor_singular_[factor_rank_] > cutoff)
    ++factor_rank_;
  factor_nullity_ = m - factor_rank_;
  factor_null_basis_.assign(
      static_cast<std::size_t>(m) * factor_nullity_, 0.0);
  for (int component = 0; component < factor_nullity_; ++component)
    for (int column = 0; column < m; ++column)
      factor_null_basis_[column + static_cast<std::size_t>(m) * component] =
          factor_vt_[(factor_rank_ + component)
                     + static_cast<std::size_t>(m) * column];
  factor_support_ = support;
  ++stats_.factorizations;
  stats_.factor_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - factor_begin)
                          .count();
  return true;
}

bool AugmentedKktBasis::select(
    const std::vector<std::uint8_t> &support,
    const std::vector<double> &direction,
    const std::vector<std::uint8_t> &selector_constraints,
    AugmentedKktSolution &solution) {
  return select_affine(support, direction, fixture_.b,
                       selector_constraints, {}, 1, solution);
}

bool AugmentedKktBasis::select_affine(
    const std::vector<std::uint8_t> &support,
    const std::vector<double> &active_value,
    const std::vector<double> &offset,
    const std::vector<std::uint8_t> &selector_constraints,
    const std::vector<double> &warm_multiplier,
    int lane, AugmentedKktSolution &solution) {
  ++stats_.calls;
  last_failure_.clear();
  blocking_row_ = -1;
  solution = {};
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  if (support.size() != fixture_.n || active_value.size() != fixture_.n
      || offset.size() != fixture_.n || lane < 0 || lane > 1
      || selector_constraints.size() != fixture_.n
      || m <= 0) {
    last_failure_ = "dimension mismatch";
    return false;
  }
  if (!ensure_factor(support)) return false;
  solution.rows = factor_rows_;
  const int active = factor_active_;
  const int rank = factor_rank_;
  const int nullity = factor_nullity_;
  const auto &singular = factor_singular_;
  const auto &u = factor_u_;
  const auto &vt = factor_vt_;
  const auto &null_basis = factor_null_basis_;
  std::vector<double> rhs(active);
  for (int local = 0; local < active; ++local) {
    const auto row = solution.rows[local];
    rhs[local] = active_value[row] - offset[row];
  }
  solution.rank = rank;
  solution.nullity = nullity;

  // Minimum-norm particular solution of B_F ua = g_F-b_F.
  std::vector<double> ua0(m, 0.0);
  for (int component = 0; component < rank; ++component) {
    double coefficient = 0.0;
    for (int local = 0; local < active; ++local)
      coefficient += u[local + static_cast<std::size_t>(active) * component]
                     * rhs[local];
    coefficient /= singular[component];
    for (int column = 0; column < m; ++column)
      ua0[column] += vt[component + static_cast<std::size_t>(m) * column]
                     * coefficient;
  }
  const auto projection_begin = std::chrono::steady_clock::now();
  std::vector<double> lambda(nullity, 0.0);
  const auto &warm = warm_multiplier.size() == fixture_.m
                         ? warm_multiplier : last_multiplier_[lane];
  // The first state in a lane is an already accepted walker multiplier.
  // Preserve it exactly when it passes the original sparse equations and
  // inequalities.  Reconstructing that multiplier through a fresh
  // pseudoinverse is the failure mode this persistent state must eliminate.
  auto admit_warm = [&](const std::vector<double> &candidate) {
    if (candidate.size() != fixture_.m) return false;
    std::vector<double> candidate_product(n, 0.0), candidate_abs(n, 0.0);
    for (int row = 0; row < n; ++row)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        candidate_product[row] +=
            fixture_.values[p] * candidate[fixture_.indices[p]];
        candidate_abs[row] +=
            std::abs(fixture_.values[p] * candidate[fixture_.indices[p]]);
      }
    double active_residual = 0.0, inactive_violation = 0.0;
    std::vector<double> transpose(m, 0.0), transpose_abs(m, 0.0);
    for (int row = 0; row < n; ++row) {
      if (support[row]) {
        const double residual =
            offset[row] + candidate_product[row] - active_value[row];
        active_residual = std::max(
            active_residual,
            std::abs(residual)
                / (1.0 + std::abs(offset[row])
                   + candidate_abs[row]
                   + std::abs(active_value[row])));
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
          const int column = fixture_.indices[p];
          transpose[column] += fixture_.values[p] * active_value[row];
          transpose_abs[column] +=
              std::abs(fixture_.values[p] * active_value[row]);
        }
      } else if (selector_constraints[row]) {
        const double slack = offset[row] + candidate_product[row];
        inactive_violation = std::max(
            inactive_violation,
            slack / (1.0 + std::abs(offset[row])
                     + candidate_abs[row]));
      }
    }
    double transpose_residual = 0.0;
    for (int column = 0; column < m; ++column)
      transpose_residual = std::max(
          transpose_residual,
          std::abs(transpose[column]
                   - (lane == 0 ? fixture_.d[column] : 0.0))
              / (1.0 + transpose_abs[column]
                 + (lane == 0 ? std::abs(fixture_.d[column]) : 0.0)));
    inactive_violation = std::max(0.0, inactive_violation);
    if (active_residual > 1e-7 || inactive_violation > 1e-7
        || transpose_residual > 2e-9)
      return false;
    solution.ua = candidate;
    solution.bua = std::move(candidate_product);
    solution.g.clear();
    solution.g.reserve(active);
    for (auto row : solution.rows) solution.g.push_back(active_value[row]);
    solution.active_residual = active_residual;
    solution.inactive_violation = inactive_violation;
    solution.transpose_residual = transpose_residual;
    for (int row = 0; row < n; ++row) {
      if (support[row] || !selector_constraints[row]) continue;
      const double slack = offset[row] + solution.bua[row];
      if (std::abs(slack)
          <= 1e-9 * (1.0 + std::abs(offset[row])
                     + candidate_abs[row]))
        solution.selector_active_rows.push_back(row);
    }
    std::uint64_t fingerprint = 1469598103934665603ULL;
    for (auto row : solution.rows)
      fingerprint = mix(fingerprint, 2ULL * row + 1);
    for (auto row : solution.selector_active_rows)
      fingerprint = mix(fingerprint, 2ULL * row + 2);
    solution.fingerprint = fingerprint;
    stats_.worst_active_residual =
        std::max(stats_.worst_active_residual, active_residual);
    stats_.worst_inactive_violation =
        std::max(stats_.worst_inactive_violation, inactive_violation);
    stats_.worst_transpose_residual =
        std::max(stats_.worst_transpose_residual, transpose_residual);
    last_multiplier_[lane] = candidate;
    ++stats_.accepted;
    return true;
  };
  if (admit_warm(warm_multiplier)) return true;
  if (warm.size() == fixture_.m && nullity > 0) {
    for (int component = 0; component < nullity; ++component)
      for (int column = 0; column < m; ++column)
        lambda[component] +=
            null_basis[column + static_cast<std::size_t>(m) * component]
            * (warm[column] - ua0[column]);
    ++stats_.warm_starts;
  }

  std::vector<std::uint32_t> inactive_rows;
  inactive_rows.reserve(n - active);
  for (int row = 0; row < n; ++row)
    if (!support[row] && selector_constraints[row])
      inactive_rows.push_back(row);
  const int inequalities = static_cast<int>(inactive_rows.size());
  std::vector<double> c(static_cast<std::size_t>(inequalities) * nullity,
                        0.0);
  std::vector<double> bound(inequalities), norm2(inequalities, 0.0);
  std::vector<double> row_scale(inequalities, 1.0);
  for (int local = 0; local < inequalities; ++local) {
    const auto row = inactive_rows[local];
    double base = offset[row];
    row_scale[local] += std::abs(offset[row]);
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      base += fixture_.values[p] * ua0[fixture_.indices[p]];
      row_scale[local] +=
          std::abs(fixture_.values[p] * ua0[fixture_.indices[p]]);
    }
    bound[local] = -base;
    for (int component = 0; component < nullity; ++component) {
      double value = 0.0;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        value += fixture_.values[p]
                 * null_basis[fixture_.indices[p]
                              + static_cast<std::size_t>(m) * component];
      c[local + static_cast<std::size_t>(inequalities) * component] = value;
      norm2[local] += value * value;
    }
  }

  // Dykstra corrections are scalar because each projection correction is
  // parallel to one halfspace normal.  For a consistent polyhedron the
  // iterates converge to the Euclidean projection of the warm start.
  std::vector<double> correction(inequalities, 0.0);
  constexpr int kMaximumSweeps = 20000;
  constexpr double kProjectionFeasibility = 1e-10;
  constexpr double kConstantFeasibility = 5e-8;
  bool feasible = inequalities == 0;
  int sweeps = 0;
  for (; !feasible && sweeps < kMaximumSweeps; ++sweeps) {
    double max_violation = 0.0;
    for (int local = 0; local < inequalities; ++local) {
      const double row_norm2 = norm2[local];
      if (!(row_norm2 > 1e-28)) {
        if (bound[local]
            < -kConstantFeasibility * row_scale[local]) {
          ++stats_.infeasible_constant_rows;
          blocking_row_ = static_cast<int>(inactive_rows[local]);
          last_failure_ = "constant inactive inequality is infeasible";
          return false;
        }
        continue;
      }
      double product = 0.0;
      for (int component = 0; component < nullity; ++component) {
        const double coefficient =
            c[local + static_cast<std::size_t>(inequalities) * component];
        lambda[component] += correction[local] * coefficient;
        product += coefficient * lambda[component];
      }
      const double new_correction =
          std::max(0.0, (product - bound[local]) / row_norm2);
      for (int component = 0; component < nullity; ++component)
        lambda[component] -=
            new_correction
            * c[local + static_cast<std::size_t>(inequalities) * component];
      correction[local] = new_correction;
    }
    feasible = true;
    for (int local = 0; local < inequalities; ++local) {
      double product = -bound[local];
      for (int component = 0; component < nullity; ++component)
        product +=
            c[local + static_cast<std::size_t>(inequalities) * component]
            * lambda[component];
      const double scaled = product
          / (row_scale[local]
             + std::sqrt(norm2[local]) * std::max(1.0, inf_norm(lambda)));
      max_violation = std::max(max_violation, product);
      if (scaled > kProjectionFeasibility) feasible = false;
    }
    (void)max_violation;
  }
  solution.projection_sweeps = sweeps;
  stats_.projection_sweeps += sweeps;
  stats_.projection_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now()
                              - projection_begin)
                              .count();
  if (!feasible) {
    ++stats_.projection_failures;
    last_failure_ = "null-space halfspace projection did not converge";
    return false;
  }

  solution.ua = ua0;
  for (int component = 0; component < nullity; ++component)
    for (int column = 0; column < m; ++column)
      solution.ua[column] +=
          null_basis[column + static_cast<std::size_t>(m) * component]
          * lambda[component];
  solution.bua.assign(n, 0.0);
  std::vector<double> bua_abs(n, 0.0);
  for (int row = 0; row < n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      solution.bua[row] += fixture_.values[p] * solution.ua[fixture_.indices[p]];
      bua_abs[row] += std::abs(
          fixture_.values[p] * solution.ua[fixture_.indices[p]]);
    }
  solution.g.reserve(active);
  for (auto row : solution.rows) solution.g.push_back(active_value[row]);

  const auto audit_begin = std::chrono::steady_clock::now();
  std::vector<double> transpose(m, 0.0), transpose_abs(m, 0.0);
  for (int row = 0; row < n; ++row) {
    if (support[row]) {
      const double residual = offset[row] + solution.bua[row]
                              - active_value[row];
      solution.active_residual = std::max(
          solution.active_residual,
          std::abs(residual)
              / (1.0 + std::abs(offset[row])
                 + bua_abs[row]
                 + std::abs(active_value[row])));
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const int column = fixture_.indices[p];
        transpose[column] += fixture_.values[p] * active_value[row];
        transpose_abs[column] +=
            std::abs(fixture_.values[p] * active_value[row]);
      }
    } else if (selector_constraints[row]) {
      const double slack_slope = offset[row] + solution.bua[row];
      solution.inactive_violation = std::max(
          solution.inactive_violation,
          slack_slope
              / (1.0 + std::abs(offset[row])
                 + bua_abs[row]));
    }
  }
  for (int column = 0; column < m; ++column)
    solution.transpose_residual = std::max(
        solution.transpose_residual,
        std::abs(transpose[column]
                 - (lane == 0 ? fixture_.d[column] : 0.0))
            / (1.0 + transpose_abs[column]
               + (lane == 0 ? std::abs(fixture_.d[column]) : 0.0)));
  solution.inactive_violation = std::max(0.0, solution.inactive_violation);

  const double active_gate = 5e-8;
  const double inactive_gate = 5e-8;
  const double transpose_gate = 2e-9;
  if (solution.active_residual > active_gate
      || solution.inactive_violation > inactive_gate
      || solution.transpose_residual > transpose_gate) {
    ++stats_.audit_failures;
    last_failure_ = "original-operator KKT audit failed";
    return false;
  }
  const double selector_tight = 1e-9;
  for (int local = 0; local < inequalities; ++local) {
    const auto row = inactive_rows[local];
    const double slope = offset[row] + solution.bua[row];
    if (std::abs(slope)
        <= selector_tight
               * (1.0 + std::abs(offset[row])
                  + bua_abs[row]))
      solution.selector_active_rows.push_back(row);
  }
  std::uint64_t fingerprint = 1469598103934665603ULL;
  for (auto row : solution.rows) fingerprint = mix(fingerprint, 2ULL * row + 1);
  for (auto row : solution.selector_active_rows)
    fingerprint = mix(fingerprint, 2ULL * row + 2);
  solution.fingerprint = fingerprint;
  stats_.audit_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - audit_begin)
                         .count();
  stats_.worst_active_residual =
      std::max(stats_.worst_active_residual, solution.active_residual);
  stats_.worst_inactive_violation =
      std::max(stats_.worst_inactive_violation, solution.inactive_violation);
  stats_.worst_transpose_residual =
      std::max(stats_.worst_transpose_residual, solution.transpose_residual);
  last_multiplier_[lane] = solution.ua;
  ++stats_.accepted;
  return true;
}

}  // namespace twalker::revised
