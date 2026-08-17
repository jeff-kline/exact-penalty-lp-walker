#include "gram_face_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace twalker {
namespace {

double norm2(const std::vector<double> &values) {
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

double inf_norm(const double *values, std::size_t size) {
  double result = 0.0;
  for (std::size_t i = 0; i < size; ++i)
    result = std::max(result, std::abs(values[i]));
  return result;
}

}  // namespace

GramFaceSolver::GramFaceSolver(const Fixture &fixture, double min_rcond,
                               std::vector<double> target_shift)
    : fixture_(fixture), target_shift_(std::move(target_shift)),
      min_rcond_(min_rcond), support_(fixture.n, 0),
      atb_(fixture.m, 0.0), ats_(fixture.m, 0.0),
      inverse_perm_(fixture.m) {
  if (!target_shift_.empty() && target_shift_.size() != fixture_.n)
    throw FaceDecline("target shift has wrong dimension");
  if (!cholmod_l_start(&common_)) throw FaceDecline("cholmod start failed");
  common_.final_ll = 0;
  common_.print = 0;
}

GramFaceSolver::~GramFaceSolver() {
  clear_factor();
  cholmod_l_finish(&common_);
}

void GramFaceSolver::clear_factor() {
  if (factor_) cholmod_l_free_factor(&factor_, &common_);
  factor_ = nullptr;
  rcond_ = 0.0;
}

bool GramFaceSolver::rebuild(const std::vector<std::uint32_t> &rows) {
  clear_factor();
  std::fill(support_.begin(), support_.end(), 0);
  std::fill(atb_.begin(), atb_.end(), 0.0);
  std::fill(ats_.begin(), ats_.end(), 0.0);
  std::size_t nnz = 0;
  for (auto row : rows) {
    if (row >= fixture_.n || support_[row]) return false;
    support_[row] = 1;
    nnz += fixture_.indptr[row + 1] - fixture_.indptr[row];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      atb_[fixture_.indices[p]] += fixture_.values[p] * fixture_.b[row];
      if (!target_shift_.empty())
        ats_[fixture_.indices[p]] +=
            fixture_.values[p] * target_shift_[row];
    }
  }

  // C is A' (m by |W|), so C C' is the active-face Gram matrix.
  auto *C = cholmod_l_allocate_sparse(fixture_.m, rows.size(), nnz, 1, 1, 0,
                                      CHOLMOD_REAL, &common_);
  if (!C) return false;
  auto *cp = static_cast<std::int64_t *>(C->p);
  auto *ci = static_cast<std::int64_t *>(C->i);
  auto *cx = static_cast<double *>(C->x);
  std::size_t cursor = 0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    cp[local] = static_cast<std::int64_t>(cursor);
    const auto row = rows[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      ci[cursor] = static_cast<std::int64_t>(fixture_.indices[p]);
      cx[cursor] = fixture_.values[p];
      ++cursor;
    }
  }
  cp[rows.size()] = static_cast<std::int64_t>(cursor);
  auto *gram = cholmod_l_aat(C, nullptr, 0, 1, &common_);
  cholmod_l_free_sparse(&C, &common_);
  if (!gram) return false;
  // cholmod_aat returns a symmetric matrix stored with stype=0.  Mark its
  // upper triangle as authoritative; otherwise cholmod_factorize interprets
  // it as an unsymmetric matrix and factors gram*gram' instead of gram.
  gram->stype = 1;
  factor_ = cholmod_l_analyze(gram, &common_);
  const bool ok = factor_ && cholmod_l_factorize(gram, factor_, &common_)
                  && factor_->minor == fixture_.m;
  cholmod_l_free_sparse(&gram, &common_);
  ++stats_.rebuilds;
  if (!ok) {
    clear_factor();
    return false;
  }
  if (factor_->Perm) {
    const auto *permutation = static_cast<const std::int64_t *>(factor_->Perm);
    for (std::size_t position = 0; position < fixture_.m; ++position)
      inverse_perm_[permutation[position]] = static_cast<std::int64_t>(position);
  } else {
    for (std::size_t position = 0; position < fixture_.m; ++position)
      inverse_perm_[position] = static_cast<std::int64_t>(position);
  }
  rcond_ = cholmod_l_rcond(factor_, &common_);
  return std::isfinite(rcond_) && rcond_ >= min_rcond_;
}

bool GramFaceSolver::apply_row(std::uint32_t row, bool add) {
  const auto begin = fixture_.indptr[row];
  const auto end = fixture_.indptr[row + 1];
  const auto nnz = end - begin;
  auto *column = cholmod_l_allocate_sparse(fixture_.m, 1, nnz, 1, 1, 0,
                                           CHOLMOD_REAL, &common_);
  if (!column) return false;
  auto *cp = static_cast<std::int64_t *>(column->p);
  auto *ci = static_cast<std::int64_t *>(column->i);
  auto *cx = static_cast<double *>(column->x);
  cp[0] = 0;
  cp[1] = static_cast<std::int64_t>(nnz);
  std::vector<std::pair<std::int64_t, double>> permuted;
  permuted.reserve(nnz);
  for (auto p = begin; p < end; ++p)
    permuted.emplace_back(inverse_perm_[fixture_.indices[p]],
                          fixture_.values[p]);
  std::sort(permuted.begin(), permuted.end());
  for (std::size_t local = 0; local < nnz; ++local) {
    ci[local] = permuted[local].first;
    cx[local] = permuted[local].second;
  }
  const bool ok = cholmod_l_updown(add ? 1 : 0, column, factor_, &common_);
  cholmod_l_free_sparse(&column, &common_);
  if (!ok || factor_->minor != fixture_.m) return false;
  const double sign = add ? 1.0 : -1.0;
  for (auto p = begin; p < end; ++p)
    atb_[fixture_.indices[p]] += sign * fixture_.values[p] * fixture_.b[row];
  if (!target_shift_.empty())
    for (auto p = begin; p < end; ++p)
      ats_[fixture_.indices[p]] +=
          sign * fixture_.values[p] * target_shift_[row];
  support_[row] = add;
  if (add)
    ++stats_.updates;
  else
    ++stats_.downdates;
  return true;
}

bool GramFaceSolver::transition(const std::vector<std::uint32_t> &rows) {
  std::vector<std::uint8_t> target(fixture_.n, 0);
  for (auto row : rows) {
    if (row >= fixture_.n || target[row]) return false;
    target[row] = 1;
  }
  // Insert before deleting so an exchange never passes through a needlessly
  // rank-deficient intermediate Gram matrix.
  for (std::size_t row = 0; row < fixture_.n; ++row)
    if (!support_[row] && target[row] && !apply_row(row, true)) return false;
  for (std::size_t row = 0; row < fixture_.n; ++row)
    if (support_[row] && !target[row] && !apply_row(row, false)) return false;
  rcond_ = cholmod_l_rcond(factor_, &common_);
  return std::isfinite(rcond_) && rcond_ >= min_rcond_;
}

bool GramFaceSolver::refine_extended(
    const std::vector<std::uint32_t> &rows, double *x,
    double &ua_error_bound, double &uc_error_bound) {
  constexpr int kMaximumCorrections = 5;
  constexpr double kForwardGate = 5e-11;
  constexpr double kTailGate = 5e-11;
  constexpr double kRoundoffTail = 1e-14;
  constexpr double kMaximumContraction = 0.5;
  constexpr long double kConditionSafety = 10.0L;
  const std::size_t m = fixture_.m;

  ++stats_.guarded_attempts;
  std::vector<long double> rhs(2 * m, 0.0L);
  std::vector<long double> gram_row_sum(m, 0.0L);
  for (std::size_t column = 0; column < m; ++column)
    rhs[m + column] = static_cast<long double>(fixture_.d[column])
                      - static_cast<long double>(ats_[column]);
  for (auto row : rows) {
    long double row_l1 = 0.0L;
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      row_l1 += std::abs(static_cast<long double>(fixture_.values[p]));
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const long double value = static_cast<long double>(fixture_.values[p]);
      rhs[column] -= value * static_cast<long double>(fixture_.b[row]);
      gram_row_sum[column] += std::abs(value) * row_l1;
    }
  }
  long double gram_inf = 0.0L;
  for (auto value : gram_row_sum) gram_inf = std::max(gram_inf, value);
  long double rhs_inf[2] = {0.0L, 0.0L};
  for (std::size_t column = 0; column < m; ++column) {
    rhs_inf[0] = std::max(rhs_inf[0], std::abs(rhs[column]));
    rhs_inf[1] = std::max(rhs_inf[1], std::abs(rhs[m + column]));
  }

  std::vector<long double> residual(2 * m);
  double previous_correction[2] = {
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
  double tail_bound[2] = {std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity()};
  double contraction[2] = {0.0, 0.0};
  double forward_bound[2] = {std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity()};

  for (int correction_count = 0;
       correction_count <= kMaximumCorrections; ++correction_count) {
    residual = rhs;
    for (auto row : rows) {
      long double ax0 = 0.0L, ax1 = 0.0L;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto column = fixture_.indices[p];
        const long double value = static_cast<long double>(fixture_.values[p]);
        ax0 += value * static_cast<long double>(x[column]);
        ax1 += value * static_cast<long double>(x[m + column]);
      }
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto column = fixture_.indices[p];
        const long double value = static_cast<long double>(fixture_.values[p]);
        residual[column] -= value * ax0;
        residual[m + column] -= value * ax1;
      }
    }

    for (int right = 0; right < 2; ++right) {
      long double residual_inf = 0.0L;
      for (std::size_t column = 0; column < m; ++column)
        residual_inf = std::max(
            residual_inf, std::abs(residual[right * m + column]));
      const long double x_inf = static_cast<long double>(
          inf_norm(x + right * m, m));
      const long double denominator = std::max(
          1.0L, gram_inf * x_inf + rhs_inf[right]);
      const long double backward = residual_inf / denominator;
      const long double estimated = kConditionSafety * backward
                                    / static_cast<long double>(rcond_);
      if (!std::isfinite(estimated)) {
        ++stats_.guarded_declines;
        return false;
      }
      forward_bound[right] = static_cast<double>(estimated);
    }
    const double maximum_forward = std::max(forward_bound[0], forward_bound[1]);
    const double maximum_tail = std::max(tail_bound[0], tail_bound[1]);
    if (correction_count > 0 && maximum_forward <= kForwardGate
        && maximum_tail <= kTailGate) {
      // Sum the independent residual and unobserved-tail estimates.  The
      // walker treats these as hard relative coefficient intervals, so using
      // max here would be unjustifiably optimistic at a decision boundary.
      ua_error_bound = forward_bound[0] + tail_bound[0];
      uc_error_bound = forward_bound[1] + tail_bound[1];
      ++stats_.guarded_accepts;
      stats_.worst_accepted_forward_bound = std::max(
          stats_.worst_accepted_forward_bound, maximum_forward);
      stats_.worst_accepted_tail_bound = std::max(
          stats_.worst_accepted_tail_bound, maximum_tail);
      stats_.worst_accepted_contraction = std::max(
          stats_.worst_accepted_contraction,
          std::max(contraction[0], contraction[1]));
      return true;
    }
    if (correction_count == kMaximumCorrections) break;

    auto *correction_rhs = cholmod_l_allocate_dense(
        m, 2, m, CHOLMOD_REAL, &common_);
    if (!correction_rhs) break;
    auto *rv = static_cast<double *>(correction_rhs->x);
    bool finite = true;
    for (std::size_t entry = 0; entry < 2 * m; ++entry) {
      rv[entry] = static_cast<double>(residual[entry]);
      finite = finite && std::isfinite(rv[entry]);
    }
    auto *correction = finite
                           ? cholmod_l_solve(CHOLMOD_A, factor_,
                                            correction_rhs, &common_)
                           : nullptr;
    cholmod_l_free_dense(&correction_rhs, &common_);
    if (!correction) break;
    const auto *dx = static_cast<const double *>(correction->x);
    double correction_relative[2] = {0.0, 0.0};
    for (int right = 0; right < 2; ++right) {
      const double scale = std::max(1.0, inf_norm(x + right * m, m));
      correction_relative[right] = inf_norm(dx + right * m, m) / scale;
    }
    for (std::size_t entry = 0; entry < 2 * m; ++entry)
      x[entry] += dx[entry];
    cholmod_l_free_dense(&correction, &common_);
    ++stats_.refinements;
    ++stats_.extended_refinements;

    for (int right = 0; right < 2; ++right) {
      if (correction_relative[right] <= kRoundoffTail) {
        contraction[right] = 0.0;
        tail_bound[right] = correction_relative[right];
      } else if (std::isfinite(previous_correction[right])
                 && previous_correction[right] > 0.0) {
        contraction[right] = correction_relative[right]
                             / previous_correction[right];
        tail_bound[right] = contraction[right] < kMaximumContraction
                                ? correction_relative[right]
                                      * contraction[right]
                                      / (1.0 - contraction[right])
                                : std::numeric_limits<double>::infinity();
      } else {
        contraction[right] = 0.0;
        tail_bound[right] = std::numeric_limits<double>::infinity();
      }
      previous_correction[right] = correction_relative[right];
    }
  }
  ++stats_.guarded_declines;
  return false;
}

bool GramFaceSolver::solve(const std::vector<std::uint32_t> &rows,
                           FaceSolution &solution,
                           bool force_guarded_refinement) {
  bool state_ok = false;
  if (!factor_) {
    state_ok = rebuild(rows);
  } else {
    state_ok = transition(rows);
    if (!state_ok) state_ok = rebuild(rows);
  }
  if (!state_ok) {
    ++stats_.declines;
    return false;
  }

  auto *rhs = cholmod_l_allocate_dense(fixture_.m, 2, fixture_.m,
                                       CHOLMOD_REAL, &common_);
  if (!rhs) {
    ++stats_.declines;
    return false;
  }
  auto *values = static_cast<double *>(rhs->x);
  for (std::size_t column = 0; column < fixture_.m; ++column) {
    values[column] = -atb_[column];
    values[fixture_.m + column] = fixture_.d[column] - ats_[column];
  }
  auto *answer = cholmod_l_solve(CHOLMOD_A, factor_, rhs, &common_);
  cholmod_l_free_dense(&rhs, &common_);
  if (!answer) {
    ++stats_.declines;
    return false;
  }
  auto *x = static_cast<double *>(answer->x);
  bool used_extended = false;
  double ua_error_bound = 0.0, uc_error_bound = 0.0;

  // In the narrow marginal-condition band, correct the Gram solution using
  // residuals formed through the original active operator.  Well-conditioned
  // pivots avoid this work; faces below min_rcond_ never reach it.
  if (force_guarded_refinement || rcond_ < 5e-4) {
    if (!refine_extended(rows, x, ua_error_bound, uc_error_bound)) {
      cholmod_l_free_dense(&answer, &common_);
      ++stats_.declines;
      return false;
    }
    used_extended = true;
  } else if (rcond_ < 1e-3) {
    for (int iteration = 0; iteration < 2; ++iteration) {
      auto *residual = cholmod_l_allocate_dense(fixture_.m, 2, fixture_.m,
                                                CHOLMOD_REAL, &common_);
      if (!residual) {
        cholmod_l_free_dense(&answer, &common_);
        ++stats_.declines;
        return false;
      }
      auto *rv = static_cast<double *>(residual->x);
      for (std::size_t column = 0; column < fixture_.m; ++column) {
        rv[column] = -atb_[column];
        rv[fixture_.m + column] = fixture_.d[column] - ats_[column];
      }
      for (auto row : rows) {
        double ax0 = 0.0, ax1 = 0.0;
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
          const auto column = fixture_.indices[p];
          const double value = fixture_.values[p];
          ax0 += value * x[column];
          ax1 += value * x[fixture_.m + column];
        }
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
          const auto column = fixture_.indices[p];
          const double value = fixture_.values[p];
          rv[column] -= value * ax0;
          rv[fixture_.m + column] -= value * ax1;
        }
      }
      auto *correction = cholmod_l_solve(CHOLMOD_A, factor_, residual,
                                         &common_);
      cholmod_l_free_dense(&residual, &common_);
      if (!correction) {
        cholmod_l_free_dense(&answer, &common_);
        ++stats_.declines;
        return false;
      }
      const auto *dx = static_cast<const double *>(correction->x);
      for (std::size_t entry = 0; entry < 2 * fixture_.m; ++entry)
        x[entry] += dx[entry];
      cholmod_l_free_dense(&correction, &common_);
      ++stats_.refinements;
    }
  }
  solution = FaceSolution{};
  solution.rank = static_cast<std::int64_t>(fixture_.m);
  solution.rows = rows;
  solution.core_diagonal_ratio = std::sqrt(rcond_);
  solution.g.resize(rows.size());
  solution.h.resize(rows.size());
  solution.ua.assign(x, x + fixture_.m);
  solution.uc.assign(x + fixture_.m, x + 2 * fixture_.m);
  solution.used_maintained_gram = true;
  solution.used_extended_gram = used_extended;
  solution.ua_relative_error_bound = ua_error_bound;
  solution.uc_relative_error_bound = uc_error_bound;
  cholmod_l_free_dense(&answer, &common_);

  solution.bua.assign(fixture_.n, 0.0);
  solution.buc.assign(fixture_.n, 0.0);
  for (std::size_t row = 0; row < fixture_.n; ++row) {
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const double value = fixture_.values[p];
      solution.bua[row] += value * solution.ua[column];
      solution.buc[row] += value * solution.uc[column];
    }
  }

  std::vector<double> transpose_g(fixture_.m, 0.0);
  std::vector<double> dual(fixture_.m, 0.0);
  double g_scale = 1.0, h_scale = 1.0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    const double aua = solution.bua[row];
    const double auc = solution.buc[row];
    solution.g[local] = fixture_.b[row] + aua;
    const double shift = target_shift_.empty() ? 0.0 : target_shift_[row];
    solution.h[local] = shift + auc;
    g_scale = std::max(g_scale, std::abs(solution.g[local]));
    h_scale = std::max(h_scale, std::abs(solution.h[local]));
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const double value = fixture_.values[p];
      transpose_g[column] += value * solution.g[local];
      dual[column] += value * solution.h[local];
    }
  }
  for (std::size_t column = 0; column < fixture_.m; ++column)
    dual[column] -= fixture_.d[column];
  solution.dres = norm2(dual) / std::max(1.0, norm2(fixture_.d));
  solution.piece_residual = norm2(transpose_g)
                            / std::max(1.0, g_scale * std::sqrt(fixture_.m));
  if (!std::isfinite(solution.dres)
      || !std::isfinite(solution.piece_residual)
      || solution.dres > 1e-7 || solution.piece_residual > 1e-7) {
    ++stats_.declines;
    return false;
  }
  return true;
}

}  // namespace twalker
