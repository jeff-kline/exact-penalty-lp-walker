#include "face_solver.hpp"

#include <SuiteSparseQR_C.h>
#include <vecLib/cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <numeric>
#include <utility>

extern "C" {
void dtzrzf_(const int *m, const int *n, double *a, const int *lda,
             double *tau, double *work, const int *lwork, int *info);
void dormrz_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const int *l, const double *a, const int *lda,
             const double *tau, double *c, const int *ldc, double *work,
             const int *lwork, int *info);
void dtrtrs_(const char *uplo, const char *trans, const char *diag,
             const int *n, const int *nrhs, const double *a, const int *lda,
             double *b, const int *ldb, int *info);
void dtrrfs_(const char *uplo, const char *trans, const char *diag,
             const int *n, const int *nrhs, const double *a, const int *lda,
             const double *b, const int *ldb, const double *x, const int *ldx,
             double *ferr, double *berr, double *work, int *iwork, int *info);
void dgesdd_(const char *jobz, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt,
             const int *ldvt, double *work, const int *lwork, int *iwork,
             int *info);
}

namespace twalker {
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
  double scale = 0.0;
  double sum = 1.0;
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

void cache_products(const Fixture &fixture, FaceSolution &solution) {
  solution.bua.assign(fixture.n, 0.0);
  solution.buc.assign(fixture.n, 0.0);
  for (std::size_t row = 0; row < fixture.n; ++row) {
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      const auto column = fixture.indices[p];
      const double value = fixture.values[p];
      solution.bua[row] += value * solution.ua[column];
      solution.buc[row] += value * solution.uc[column];
    }
  }
}

void require_lapack(int info, const char *operation) {
  if (info != 0)
    throw FaceDecline(std::string(operation) + " failed, info="
                      + std::to_string(info));
}

int workspace_size(double query) {
  if (!std::isfinite(query) || query < 1.0)
    throw FaceDecline("invalid LAPACK workspace query");
  return static_cast<int>(std::ceil(query));
}

FaceSolution dense_svd_face(const Fixture &fixture,
                            const std::vector<std::uint32_t> &rows,
                            const std::vector<double> &target_shift,
                            bool export_row_space = false,
                            bool export_pseudoinverse = false) {
  const int face_rows = static_cast<int>(rows.size());
  const int m = static_cast<int>(fixture.m);
  const int thin = std::min(face_rows, m);
  std::vector<double> matrix(static_cast<std::size_t>(face_rows) * m, 0.0);
  std::vector<double> bW(face_rows), sW(face_rows, 0.0);
  std::vector<double> adjusted_d = fixture.d;
  for (int local = 0; local < face_rows; ++local) {
    const auto row = rows[local];
    bW[local] = fixture.b[row];
    if (!target_shift.empty()) sW[local] = target_shift[row];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      matrix[local + static_cast<std::size_t>(face_rows)
                         * fixture.indices[p]] = fixture.values[p];
      adjusted_d[fixture.indices[p]] -= fixture.values[p] * sW[local];
    }
  }
  std::vector<double> singular(thin);
  std::vector<double> U(static_cast<std::size_t>(face_rows) * thin);
  std::vector<double> VT(static_cast<std::size_t>(thin) * m);
  std::vector<int> iwork(8 * thin);
  const char job = 'S';
  const int lda = face_rows, ldu = face_rows, ldvt = thin;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &face_rows, &m, matrix.data(), &lda, singular.data(), U.data(),
          &ldu, VT.data(), &ldvt, &query, &lwork, iwork.data(), &info);
  require_lapack(info, "dgesdd workspace query");
  lwork = workspace_size(query);
  std::vector<double> work(lwork);
  dgesdd_(&job, &face_rows, &m, matrix.data(), &lda, singular.data(), U.data(),
          &ldu, VT.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
  require_lapack(info, "dgesdd");

  int rank = 0;
  if (thin) {
    const double cutoff = singular.front() * std::max(face_rows, m)
                          * std::numeric_limits<double>::epsilon();
    while (rank < thin && singular[rank] > cutoff) ++rank;
  }
  if (!rank) throw FaceDecline("dense fallback found zero rank");
  FaceSolution solution;
  solution.rank = rank;
  solution.used_dense_fallback = true;
  solution.rows = rows;
  solution.g = bW;
  solution.h = sW;
  solution.ua.assign(m, 0.0);
  solution.uc.assign(m, 0.0);
  if (export_row_space) {
    solution.svd_row_space.resize(static_cast<std::size_t>(rank) * m);
    for (int column = 0; column < m; ++column)
      for (int component = 0; component < rank; ++component)
        solution.svd_row_space[
            component + static_cast<std::size_t>(rank) * column] =
            VT[component + static_cast<std::size_t>(thin) * column];
    if (std::getenv("TWALKER_MAINTAINED_SVD_AUDIT")
        || std::getenv("TWALKER_MAINTAINED_SVD_LIVE")) {
      solution.svd_left_space.resize(
          static_cast<std::size_t>(face_rows) * rank);
      for (int component = 0; component < rank; ++component)
        std::copy(U.begin() + static_cast<std::size_t>(face_rows) * component,
                  U.begin() + static_cast<std::size_t>(face_rows)
                                * (component + 1),
                  solution.svd_left_space.begin()
                      + static_cast<std::size_t>(face_rows) * component);
      solution.svd_singular_values.assign(singular.begin(),
                                          singular.begin() + rank);
    }
  }
  if (export_pseudoinverse) {
    std::vector<double> scaled_v(static_cast<std::size_t>(m) * rank);
    for (int component = 0; component < rank; ++component)
      for (int column = 0; column < m; ++column)
        scaled_v[column + static_cast<std::size_t>(m) * component] =
            VT[component + static_cast<std::size_t>(thin) * column]
            / singular[component];
    solution.recurrence_pseudoinverse.assign(
        static_cast<std::size_t>(m) * face_rows, 0.0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, face_rows, rank, 1.0, scaled_v.data(), m,
                U.data(), face_rows, 0.0,
                solution.recurrence_pseudoinverse.data(), m);
    solution.recurrence_seed_rank = rank;
  }
  for (int component = 0; component < rank; ++component) {
    double vd = 0.0, c = 0.0;
    for (int column = 0; column < m; ++column)
      vd += VT[component + static_cast<std::size_t>(thin) * column]
            * adjusted_d[column];
    for (int row = 0; row < face_rows; ++row)
      c += U[row + static_cast<std::size_t>(face_rows) * component] * bW[row];
    const double inv_s = 1.0 / singular[component];
    for (int row = 0; row < face_rows; ++row) {
      const double left = U[row + static_cast<std::size_t>(face_rows)
                                      * component];
      solution.h[row] += left * vd * inv_s;
      solution.g[row] -= left * c;
    }
    for (int column = 0; column < m; ++column) {
      const double right = VT[component + static_cast<std::size_t>(thin)
                                              * column];
      solution.ua[column] -= right * c * inv_s;
      solution.uc[column] += right * vd * inv_s * inv_s;
    }
  }
  cache_products(fixture, solution);
  // Match piece_y's explicit second projection, which is significant on the
  // marginal faces that trigger this fallback.
  for (int component = 0; component < rank; ++component) {
    double projection = 0.0;
    for (int row = 0; row < face_rows; ++row)
      projection += U[row + static_cast<std::size_t>(face_rows) * component]
                    * solution.g[row];
    for (int row = 0; row < face_rows; ++row)
      solution.g[row] -= U[row + static_cast<std::size_t>(face_rows)
                                   * component] * projection;
  }

  std::vector<double> dual_residual(m, 0.0), transpose_g(m, 0.0);
  double slope_error = 0.0, constant_error = 0.0;
  double slope_scale = 1.0, constant_scale = 1.0;
  for (int local = 0; local < face_rows; ++local) {
    const auto row = rows[local];
    const double bua = solution.bua[row];
    const double buc = solution.buc[row];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      const auto column = fixture.indices[p];
      const double value = fixture.values[p];
      dual_residual[column] += value * solution.h[local];
      transpose_g[column] += value * solution.g[local];
    }
    const double target_slope = solution.g[local] - bW[local];
    slope_error = std::max(slope_error, std::abs(bua - target_slope));
    constant_error = std::max(
        constant_error, std::abs(buc - (solution.h[local] - sW[local])));
    slope_scale = std::max(slope_scale, std::abs(target_slope));
    constant_scale = std::max(constant_scale, std::abs(solution.h[local]));
  }
  for (int column = 0; column < m; ++column)
    dual_residual[column] -= fixture.d[column];
  solution.dres = norm2(dual_residual) / std::max(1.0, norm2(fixture.d));
  solution.piece_residual = std::max(
      {inf_norm(transpose_g) / std::max(1.0, inf_norm(solution.g)),
       slope_error / slope_scale, constant_error / constant_scale});
  if (!std::isfinite(solution.dres) || !std::isfinite(solution.piece_residual))
    throw FaceDecline("dense fallback produced nonfinite residual");
  return solution;
}

FaceSolution core_svd_face(
    const Fixture &fixture, const std::vector<std::uint32_t> &rows,
    const std::vector<double> &target_shift, std::vector<double> core,
    int core_rows, const std::vector<double> &qtb,
    const std::vector<std::int64_t> &permutation,
    std::vector<double> *spectrum = nullptr) {
  const int face_rows = static_cast<int>(rows.size());
  const int m = static_cast<int>(fixture.m);
  const int thin = std::min(core_rows, m);
  if (core_rows <= 0 || thin <= 0 || static_cast<int>(qtb.size()) < core_rows)
    throw FaceDecline("invalid QR core for SVD");

  std::vector<double> singular(thin);
  std::vector<double> U(static_cast<std::size_t>(core_rows) * thin);
  std::vector<double> VT(static_cast<std::size_t>(thin) * m);
  std::vector<int> iwork(8 * thin);
  const char job = 'S';
  const int lda = core_rows, ldu = core_rows, ldvt = thin;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &core_rows, &m, core.data(), &lda, singular.data(), U.data(),
          &ldu, VT.data(), &ldvt, &query, &lwork, iwork.data(), &info);
  require_lapack(info, "core dgesdd workspace query");
  lwork = workspace_size(query);
  std::vector<double> work(lwork);
  dgesdd_(&job, &core_rows, &m, core.data(), &lda, singular.data(), U.data(),
          &ldu, VT.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
  require_lapack(info, "core dgesdd");

  int rank = 0;
  if (thin) {
    const double cutoff = singular.front() * std::max(face_rows, m)
                          * std::numeric_limits<double>::epsilon();
    while (rank < thin && singular[rank] > cutoff) ++rank;
  }
  if (!rank) throw FaceDecline("core SVD found zero rank");
  if (spectrum) *spectrum = singular;

  std::vector<double> adjusted_d = fixture.d;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    const double shift = target_shift.empty() ? 0.0 : target_shift[row];
    if (shift == 0.0) continue;
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
      adjusted_d[fixture.indices[p]] -= fixture.values[p] * shift;
  }
  std::vector<double> adjusted_permuted(m);
  for (int j = 0; j < m; ++j)
    adjusted_permuted[j] = adjusted_d[permutation[j]];

  FaceSolution solution;
  solution.rank = rank;
  solution.rows = rows;
  solution.used_core_svd = true;
  solution.ua.assign(m, 0.0);
  solution.uc.assign(m, 0.0);
  for (int component = 0; component < rank; ++component) {
    double uq = 0.0, vd = 0.0;
    for (int row = 0; row < core_rows; ++row)
      uq += U[row + static_cast<std::size_t>(core_rows) * component]
            * qtb[row];
    for (int j = 0; j < m; ++j)
      vd += VT[component + static_cast<std::size_t>(thin) * j]
            * adjusted_permuted[j];
    const double inv_s = 1.0 / singular[component];
    for (int j = 0; j < m; ++j) {
      const double right = VT[component + static_cast<std::size_t>(thin) * j];
      const auto column = permutation[j];
      solution.ua[column] -= right * uq * inv_s;
      solution.uc[column] += right * vd * inv_s * inv_s;
    }
  }
  cache_products(fixture, solution);
  solution.g.resize(rows.size());
  solution.h.resize(rows.size());
  std::vector<double> transpose_g(m, 0.0), dual_residual(m, 0.0);
  double slope_error = 0.0, constant_error = 0.0;
  double slope_scale = 1.0, constant_scale = 1.0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    const double shift = target_shift.empty() ? 0.0 : target_shift[row];
    solution.g[local] = fixture.b[row] + solution.bua[row];
    solution.h[local] = shift + solution.buc[row];
    const double target_slope = solution.g[local] - fixture.b[row];
    slope_error = std::max(
        slope_error, std::abs(solution.bua[row] - target_slope));
    constant_error = std::max(
        constant_error,
        std::abs(solution.buc[row] - (solution.h[local] - shift)));
    slope_scale = std::max(slope_scale, std::abs(target_slope));
    constant_scale = std::max(constant_scale, std::abs(solution.h[local]));
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      const auto column = fixture.indices[p];
      const double value = fixture.values[p];
      transpose_g[column] += value * solution.g[local];
      dual_residual[column] += value * solution.h[local];
    }
  }
  for (int column = 0; column < m; ++column)
    dual_residual[column] -= fixture.d[column];
  solution.dres = norm2(dual_residual) / std::max(1.0, norm2(fixture.d));
  solution.piece_residual = std::max(
      {inf_norm(transpose_g) / std::max(1.0, inf_norm(solution.g)),
       slope_error / slope_scale, constant_error / constant_scale});
  if (!std::isfinite(solution.dres) || !std::isfinite(solution.piece_residual))
    throw FaceDecline("core SVD produced nonfinite residual");
  return solution;
}

// Audit-only full-column-rank QR row updater.  The production answer always
// remains the direct FaceSolver result; this state exists only when explicitly
// enabled and measures whether fixed-permutation update/downdate algebra is a
// viable replacement for repeated SPQR on real walker transitions.
class UpdatedQrAudit {
 public:
  UpdatedQrAudit(const Fixture &fixture, const std::vector<double> &target_shift,
                 FaceSolveStats &stats)
      : fixture_(fixture), target_shift_(target_shift), stats_(stats),
        m_(static_cast<int>(fixture.m)) {}

  bool try_live(const std::vector<std::uint32_t> &rows,
                FaceSolution &candidate) {
    ++stats_.qr_update_observations;
    if (!valid_ || rows == rows_) return false;
    std::vector<std::uint32_t> additions, removals;
    std::set_difference(rows.begin(), rows.end(), rows_.begin(), rows_.end(),
                        std::back_inserter(additions));
    std::set_difference(rows_.begin(), rows_.end(), rows.begin(), rows.end(),
                        std::back_inserter(removals));
    const auto changes = additions.size() + removals.size();
    if (changes == 0) return false;
    if (changes > 2 || updates_since_refactor_ + changes > 32) {
      ++stats_.qr_update_large_changes;
      valid_ = false;
      return false;
    }
    ++stats_.qr_update_eligible;
    const auto transition_start = Clock::now();
    const auto update_start = Clock::now();
    bool updated = true;
    for (auto row : additions) updated = updated && update_row(row, +1);
    for (auto row : removals) updated = updated && update_row(row, -1);
    stats_.qr_update_update_ms += milliseconds_since(update_start);
    if (!updated) {
      ++stats_.qr_update_factor_failures;
      ++stats_.qr_update_numerical_declines;
      valid_ = false;
      return false;
    }
    if (!restore_diagonal_ratio()) {
      ++stats_.qr_update_ratio_declines;
      ++stats_.qr_update_numerical_declines;
      valid_ = false;
      return false;
    }
    rows_ = rows;
    updates_since_refactor_ += changes;
    stats_.qr_update_additions += additions.size();
    stats_.qr_update_downdates += removals.size();
    const auto solve_start = Clock::now();
    bool solved = solve_candidate(rows, candidate);
    if (!solved && local_repivot(true)) {
      ++stats_.qr_update_repivot_retries;
      solved = solve_candidate(rows, candidate);
    }
    stats_.qr_update_solve_ms += milliseconds_since(solve_start);
    if (!solved) {
      ++stats_.qr_update_solve_failures;
      ++stats_.qr_update_numerical_declines;
      valid_ = false;
      return false;
    }
    ++stats_.qr_update_attempts;
    candidate_times_us_.push_back(
        1000.0 * std::chrono::duration<double, std::milli>(
                     Clock::now() - transition_start).count());
    stats_.qr_update_median_us = median(candidate_times_us_);
    const double residual = std::max(candidate.piece_residual, candidate.dres);
    stats_.qr_update_max_residual = std::max(stats_.qr_update_max_residual,
                                             residual);
    if (std::isfinite(residual) && residual <= 1e-10) {
      ++stats_.qr_update_admitted;
      ++stats_.qr_update_live_returns;
      return true;
    }
    valid_ = false;
    return false;
  }

  void reseed(const std::vector<std::uint32_t> &rows,
              const FaceSolution &oracle,
              const std::vector<double> *fresh_core,
              const std::vector<std::int64_t> *fresh_permutation) {
    if (!valid_ && fresh_core && fresh_permutation)
      seed(rows, oracle, *fresh_core, *fresh_permutation);
  }

  void observe(const std::vector<std::uint32_t> &rows,
               const FaceSolution &oracle,
               const std::vector<double> *fresh_core,
               const std::vector<std::int64_t> *fresh_permutation,
               double oracle_ms, FaceSolution *audit_candidate) {
    ++stats_.qr_update_observations;
    if (oracle.rank == m_)
      ++stats_.qr_update_full_rank_observations;
    else
      ++stats_.qr_update_rank_deficient_observations;
    if (last_oracle_rank_ >= 0 && oracle.rank != last_oracle_rank_)
      ++stats_.qr_update_rank_changes;
    last_oracle_rank_ = oracle.rank;
    bool keep_state = valid_;
    if (valid_ && rows != rows_) {
      std::vector<std::uint32_t> additions, removals;
      std::set_difference(rows.begin(), rows.end(), rows_.begin(), rows_.end(),
                          std::back_inserter(additions));
      std::set_difference(rows_.begin(), rows_.end(), rows.begin(), rows.end(),
                          std::back_inserter(removals));
      const auto changes = additions.size() + removals.size();
      if (changes == 0) {
        rows_ = rows;
      } else if (changes > 2 || updates_since_refactor_ + changes > 32) {
        ++stats_.qr_update_large_changes;
        keep_state = false;
      } else {
        ++stats_.qr_update_eligible;
        const auto transition_start = Clock::now();
        const auto update_start = Clock::now();
        bool updated = true;
        // Insert first so a full-rank final face does not pass through an
        // avoidable singular intermediate during a row exchange.
        for (auto row : additions)
          updated = updated && update_row(row, +1);
        for (auto row : removals)
          updated = updated && update_row(row, -1);
        stats_.qr_update_update_ms += milliseconds_since(update_start);
        if (updated && restore_diagonal_ratio()) {
          rows_ = rows;
          updates_since_refactor_ += changes;
          stats_.qr_update_additions += additions.size();
          stats_.qr_update_downdates += removals.size();
          FaceSolution candidate;
          const auto solve_start = Clock::now();
          bool solved = solve_candidate(rows, candidate);
          if (!solved && local_repivot(true)) {
            ++stats_.qr_update_repivot_retries;
            solved = solve_candidate(rows, candidate);
          }
          stats_.qr_update_solve_ms += milliseconds_since(solve_start);
          if (solved) {
            ++stats_.qr_update_attempts;
            const double candidate_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - transition_start).count();
            candidate_times_us_.push_back(
                1000.0 * candidate_ms);
            oracle_times_us_.push_back(1000.0 * oracle_ms);
            stats_.qr_update_median_us = median(candidate_times_us_);
            stats_.qr_update_oracle_median_us = median(oracle_times_us_);
            const double error = std::max(
                {relative_error(candidate.ua, oracle.ua),
                 relative_error(candidate.uc, oracle.uc),
                 relative_error(candidate.g, oracle.g),
                 relative_error(candidate.h, oracle.h)});
            stats_.qr_update_max_error = std::max(
                stats_.qr_update_max_error, error);
            const double residual = std::max(candidate.piece_residual,
                                             candidate.dres);
            stats_.qr_update_max_residual = std::max(
                stats_.qr_update_max_residual, residual);
            const bool admitted = std::isfinite(residual)
                                  && residual <= 1e-10;
            const bool accurate = std::isfinite(error) && error <= 1e-10;
            if (admitted) {
              ++stats_.qr_update_admitted;
              stats_.qr_update_max_admitted_error = std::max(
                  stats_.qr_update_max_admitted_error, error);
              if (!accurate) ++stats_.qr_update_false_admits;
            }
            if (accurate) ++stats_.qr_update_accurate;
            keep_state = admitted && accurate;
            if (keep_state) {
              stats_.qr_update_replaceable_oracle_ms += oracle_ms;
              stats_.qr_update_accurate_candidate_ms += candidate_ms;
              if (audit_candidate)
                *audit_candidate = std::move(candidate);
            }
          } else {
            ++stats_.qr_update_solve_failures;
            ++stats_.qr_update_numerical_declines;
            keep_state = false;
          }
        } else {
          if (!updated)
            ++stats_.qr_update_factor_failures;
          else
            ++stats_.qr_update_ratio_declines;
          ++stats_.qr_update_numerical_declines;
          keep_state = false;
        }
      }
    }
    valid_ = keep_state;
    if (!valid_ && fresh_core && fresh_permutation)
      seed(rows, oracle, *fresh_core, *fresh_permutation);
  }

 private:
  static double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
  }

  static double relative_error(const std::vector<double> &actual,
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

  void seed(const std::vector<std::uint32_t> &rows,
            const FaceSolution &oracle, const std::vector<double> &core,
            const std::vector<std::int64_t> &permutation) {
    valid_ = false;
    if (oracle.rank != m_ || oracle.core_diagonal_ratio < 1e-5
        || core.size() != static_cast<std::size_t>(m_) * m_
        || permutation.size() != static_cast<std::size_t>(m_))
      return;
    R_ = core;
    permutation_ = permutation;
    inverse_permutation_.assign(m_, -1);
    for (int position = 0; position < m_; ++position) {
      const auto original = permutation_[position];
      if (original < 0 || original >= m_
          || inverse_permutation_[original] != -1)
        return;
      inverse_permutation_[original] = position;
    }
    // Cholesky-style row updates use a positive diagonal convention.  A sign
    // flip of an R row leaves R'R unchanged.
    for (int row = 0; row < m_; ++row) {
      if (R_[row + static_cast<std::size_t>(m_) * row] < 0.0)
        for (int column = row; column < m_; ++column)
          R_[row + static_cast<std::size_t>(m_) * column] *= -1.0;
    }
    pattern_words_ = (m_ + 63) / 64;
    row_pattern_.assign(static_cast<std::size_t>(m_) * pattern_words_, 0);
    pattern_entries_ = 0;
    for (int row = 0; row < m_; ++row)
      for (int column = row; column < m_; ++column)
        if (R_[row + static_cast<std::size_t>(m_) * column] != 0.0) {
          row_pattern_[static_cast<std::size_t>(row) * pattern_words_
                       + (column >> 6)] |=
              std::uint64_t{1} << (column & 63);
          ++pattern_entries_;
        }
    const auto triangular_capacity =
        static_cast<std::size_t>(m_) * (m_ + 1) / 2;
    dense_pattern_ = 4 * pattern_entries_ > triangular_capacity;
    cross_.assign(m_, 0.0);
    for (auto row : rows) {
      const double rhs = fixture_.b[row];
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        const auto position = inverse_permutation_[fixture_.indices[p]];
        cross_[position] += fixture_.values[p] * rhs;
      }
    }
    rows_ = rows;
    updates_since_refactor_ = 0;
    valid_ = diagonal_ratio() >= 1e-5;
    if (valid_) ++stats_.qr_update_refactors;
  }

  bool update_row(std::uint32_t row, int sign) {
    std::vector<double> x(m_, 0.0);
    std::vector<std::uint64_t> x_pattern(pattern_words_, 0);
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      if (const auto position = inverse_permutation_[fixture_.indices[p]];
          fixture_.values[p] != 0.0) {
        x[position] = fixture_.values[p];
        x_pattern[position >> 6] |= std::uint64_t{1} << (position & 63);
      }
    const double rhs = fixture_.b[row];
    for (int j = 0; j < m_; ++j)
      cross_[j] += sign * x[j] * rhs;
    for (int k = 0; k < m_; ++k) {
      if (x[k] == 0.0) continue;
      const auto diagonal_index = k + static_cast<std::size_t>(m_) * k;
      const double diagonal = R_[diagonal_index];
      if (!(diagonal > 0.0) || !std::isfinite(diagonal)) return false;
      const long double radicand =
          static_cast<long double>(diagonal) * diagonal
          + sign * static_cast<long double>(x[k]) * x[k];
      if (!(radicand > 0.0L)) return false;
      const double updated_diagonal = std::sqrt(static_cast<double>(radicand));
      const double c = updated_diagonal / diagonal;
      const double s = x[k] / diagonal;
      if (!(c > 0.0) || !std::isfinite(c) || !std::isfinite(s)) return false;
      R_[diagonal_index] = updated_diagonal;
      x_pattern[k >> 6] &= ~(std::uint64_t{1} << (k & 63));
      if (dense_pattern_) {
        for (int column = k + 1; column < m_; ++column) {
          const auto index = k + static_cast<std::size_t>(m_) * column;
          const double updated = (R_[index] + sign * s * x[column]) / c;
          x[column] = c * x[column] - s * updated;
          R_[index] = updated;
        }
        continue;
      }
      const auto row_pattern_offset =
          static_cast<std::size_t>(k) * pattern_words_;
      for (int word = k >> 6; word < pattern_words_; ++word) {
        std::uint64_t bits = row_pattern_[row_pattern_offset + word]
                             | x_pattern[word];
        while (bits) {
          const int bit = __builtin_ctzll(bits);
          bits &= bits - 1;
          const int column = (word << 6) + bit;
          if (column <= k || column >= m_) continue;
          const auto index = k + static_cast<std::size_t>(m_) * column;
          const double updated = (R_[index] + sign * s * x[column]) / c;
          x[column] = c * x[column] - s * updated;
          R_[index] = updated;
          const auto mask = std::uint64_t{1} << bit;
          if (updated != 0.0
              && !(row_pattern_[row_pattern_offset + word] & mask)) {
            row_pattern_[row_pattern_offset + word] |=
                mask;
            ++pattern_entries_;
          }
          if (x[column] != 0.0)
            x_pattern[word] |= std::uint64_t{1} << bit;
          else
            x_pattern[word] &= ~(std::uint64_t{1} << bit);
        }
      }
      const auto triangular_capacity =
          static_cast<std::size_t>(m_) * (m_ + 1) / 2;
      dense_pattern_ = 4 * pattern_entries_ > triangular_capacity;
    }
    return true;
  }

  double diagonal_ratio() const {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = 0.0;
    if (R_.size() != static_cast<std::size_t>(m_) * m_) return 0.0;
    for (int i = 0; i < m_; ++i) {
      const double value = std::abs(R_[i + static_cast<std::size_t>(m_) * i]);
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
    return maximum > 0.0 ? minimum / maximum : 0.0;
  }

  void rebuild_pattern() {
    pattern_words_ = (m_ + 63) / 64;
    row_pattern_.assign(static_cast<std::size_t>(m_) * pattern_words_, 0);
    pattern_entries_ = 0;
    for (int row = 0; row < m_; ++row)
      for (int column = row; column < m_; ++column)
        if (R_[row + static_cast<std::size_t>(m_) * column] != 0.0) {
          row_pattern_[static_cast<std::size_t>(row) * pattern_words_
                       + (column >> 6)] |=
              std::uint64_t{1} << (column & 63);
          ++pattern_entries_;
        }
    const auto triangular_capacity =
        static_cast<std::size_t>(m_) * (m_ + 1) / 2;
    dense_pattern_ = 4 * pattern_entries_ > triangular_capacity;
  }

  // Move one strong trailing column into the weakest local position without
  // rebuilding A or calling SPQR.  Each adjacent column interchange creates
  // one subdiagonal entry; a single Givens rotation restores triangular form.
  // Thus R'R and the represented face are preserved to working precision.
  bool local_repivot(bool force = false) {
    if (!std::getenv("TWALKER_QR_UPDATE_LOCAL_REPIVOT")) return false;
    ++stats_.qr_update_local_repivot_attempts;
    const auto started = Clock::now();
    int window = 16;
    if (const char *raw = std::getenv("TWALKER_QR_UPDATE_REPIVOT_WINDOW"))
      window = std::max(2, std::min(64, std::atoi(raw)));
    const double initial_ratio = diagonal_ratio();
    bool changed = false;
    for (int pass = 0;
         pass < 4 && (force ? pass == 0 : diagonal_ratio() < 1e-5);
         ++pass) {
      int weak = 0;
      double weak_diagonal = std::numeric_limits<double>::infinity();
      for (int i = 0; i < m_; ++i) {
        const double value =
            std::abs(R_[i + static_cast<std::size_t>(m_) * i]);
        if (value < weak_diagonal) {
          weak_diagonal = value;
          weak = i;
        }
      }
      const int stop = std::min(m_, weak + window);
      int best = weak;
      long double best_square =
          static_cast<long double>(weak_diagonal) * weak_diagonal;
      for (int column = weak + 1; column < stop; ++column) {
        long double square = 0.0L;
        for (int row = weak; row <= column; ++row) {
          const long double value =
              R_[row + static_cast<std::size_t>(m_) * column];
          square += value * value;
        }
        if (square > best_square) {
          best_square = square;
          best = column;
        }
      }
      if (best == weak) break;
      changed = true;
      for (int position = best - 1; position >= weak; --position) {
        for (int row = 0; row < m_; ++row)
          std::swap(R_[row + static_cast<std::size_t>(m_) * position],
                    R_[row + static_cast<std::size_t>(m_) * (position + 1)]);
        std::swap(permutation_[position], permutation_[position + 1]);
        std::swap(cross_[position], cross_[position + 1]);
        const auto diagonal =
            position + static_cast<std::size_t>(m_) * position;
        const auto subdiagonal =
            position + 1 + static_cast<std::size_t>(m_) * position;
        const double a = R_[diagonal], b = R_[subdiagonal];
        const double radius = std::hypot(a, b);
        if (!(radius > 0.0) || !std::isfinite(radius)) {
          stats_.qr_update_local_repivot_ms += milliseconds_since(started);
          return false;
        }
        const double cosine = a / radius, sine = b / radius;
        for (int column = position; column < m_; ++column) {
          const auto upper =
              position + static_cast<std::size_t>(m_) * column;
          const auto lower =
              position + 1 + static_cast<std::size_t>(m_) * column;
          const double x = R_[upper], y = R_[lower];
          R_[upper] = cosine * x + sine * y;
          R_[lower] = -sine * x + cosine * y;
        }
        R_[subdiagonal] = 0.0;
      }
    }
    for (int row = 0; row < m_; ++row) {
      if (R_[row + static_cast<std::size_t>(m_) * row] < 0.0)
        for (int column = row; column < m_; ++column)
          R_[row + static_cast<std::size_t>(m_) * column] *= -1.0;
    }
    inverse_permutation_.assign(m_, -1);
    for (int position = 0; position < m_; ++position)
      inverse_permutation_[permutation_[position]] = position;
    rebuild_pattern();
    stats_.qr_update_local_repivot_ms += milliseconds_since(started);
    const double final_ratio = diagonal_ratio();
    if (changed && std::isfinite(final_ratio) && final_ratio >= 1e-5) {
      ++stats_.qr_update_local_repivot_successes;
      return true;
    }
    return final_ratio > initial_ratio && final_ratio >= 1e-5;
  }

  bool restore_diagonal_ratio() {
    return diagonal_ratio() >= 1e-5 || local_repivot();
  }

  template <typename Function>
  void for_upper_entries(int row, Function function) const {
    const auto offset = static_cast<std::size_t>(row) * pattern_words_;
    for (int word = row >> 6; word < pattern_words_; ++word) {
      std::uint64_t bits = row_pattern_[offset + word];
      while (bits) {
        const int bit = __builtin_ctzll(bits);
        bits &= bits - 1;
        const int column = (word << 6) + bit;
        if (column > row && column < m_) function(column);
      }
    }
  }

  bool triangular_pair(std::vector<double> &right, bool transpose) const {
    if (right.size() != static_cast<std::size_t>(m_) * 2) return false;
    if (dense_pattern_) {
      const char upper = 'U', trans = transpose ? 'T' : 'N', nonunit = 'N';
      const int two = 2;
      int info = 0;
      dtrtrs_(&upper, &trans, &nonunit, &m_, &two, R_.data(), &m_,
              right.data(), &m_, &info);
      return info == 0;
    }
    if (transpose) {
      for (int row = 0; row < m_; ++row) {
        const double diagonal = R_[row + static_cast<std::size_t>(m_) * row];
        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) return false;
        const double value0 = right[row] / diagonal;
        const double value1 = right[m_ + row] / diagonal;
        right[row] = value0;
        right[m_ + row] = value1;
        for_upper_entries(row, [&](int column) {
          const double value = R_[row + static_cast<std::size_t>(m_) * column];
          right[column] -= value * value0;
          right[m_ + column] -= value * value1;
        });
      }
    } else {
      for (int reverse = 0; reverse < m_; ++reverse) {
        const int row = m_ - 1 - reverse;
        double value0 = right[row], value1 = right[m_ + row];
        for_upper_entries(row, [&](int column) {
          const double value = R_[row + static_cast<std::size_t>(m_) * column];
          value0 -= value * right[column];
          value1 -= value * right[m_ + column];
        });
        const double diagonal = R_[row + static_cast<std::size_t>(m_) * row];
        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) return false;
        right[row] = value0 / diagonal;
        right[m_ + row] = value1 / diagonal;
      }
    }
    return true;
  }

  bool solve_candidate(const std::vector<std::uint32_t> &rows,
                       FaceSolution &solution) const {
    std::vector<double> adjusted_d = fixture_.d;
    if (!target_shift_.empty()) {
      for (auto row : rows)
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
          adjusted_d[fixture_.indices[p]] -=
              fixture_.values[p] * target_shift_[row];
    }
    std::vector<double> heads(static_cast<std::size_t>(m_) * 2);
    for (int position = 0; position < m_; ++position) {
      heads[position] = cross_[position];
      heads[m_ + position] = adjusted_d[permutation_[position]];
    }
    if (!triangular_pair(heads, true)) return false;
    for (int i = 0; i < m_; ++i) heads[i] = -heads[i];
    if (!triangular_pair(heads, false)) return false;

    solution.rank = m_;
    solution.rows = rows;
    solution.core_diagonal_ratio = diagonal_ratio();
    solution.ua.assign(m_, 0.0);
    solution.uc.assign(m_, 0.0);
    solution.g.resize(rows.size());
    solution.h.resize(rows.size());
    double previous_residual = std::numeric_limits<double>::infinity();
    double previous_correction[2] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    double tail_bound[2] = {std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity()};
    bool bounded = false;
    for (int refinement = 0; refinement <= 3; ++refinement) {
      for (int position = 0; position < m_; ++position) {
        solution.ua[permutation_[position]] = heads[position];
        solution.uc[permutation_[position]] = heads[m_ + position];
      }
      std::vector<long double> transpose_g(m_, 0.0L),
                               transpose_h(m_, 0.0L);
      double g_norm = 0.0;
      for (std::size_t local = 0; local < rows.size(); ++local) {
        const auto row = rows[local];
        long double bua = 0.0L, buc = 0.0L;
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
          const auto column = fixture_.indices[p];
          const long double value = fixture_.values[p];
          bua += value * static_cast<long double>(solution.ua[column]);
          buc += value * static_cast<long double>(solution.uc[column]);
        }
        const long double shift = target_shift_.empty()
                                      ? 0.0L : target_shift_[row];
        const long double g = fixture_.b[row] + bua;
        const long double h = shift + buc;
        solution.g[local] = static_cast<double>(g);
        solution.h[local] = static_cast<double>(h);
        g_norm = std::max(g_norm, std::abs(solution.g[local]));
        for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
          const auto column = fixture_.indices[p];
          const long double value = fixture_.values[p];
          transpose_g[column] += value * g;
          transpose_h[column] += value * h;
        }
      }
      long double d_scale = 0.0L, dres2 = 0.0L;
      double orthogonality = 0.0;
      for (int column = 0; column < m_; ++column) {
        transpose_h[column] -= fixture_.d[column];
        orthogonality = std::max(
            orthogonality, std::abs(static_cast<double>(transpose_g[column])));
        dres2 += transpose_h[column] * transpose_h[column];
        d_scale += static_cast<long double>(fixture_.d[column])
                   * fixture_.d[column];
      }
      solution.dres = static_cast<double>(
          std::sqrt(dres2) / std::max(1.0L, std::sqrt(d_scale)));
      solution.piece_residual = orthogonality / std::max(1.0, g_norm);
      const double residual = std::max(solution.dres,
                                       solution.piece_residual);
      if (!std::isfinite(residual)) break;
      if (residual <= 1e-10 && refinement > 0
          && std::isfinite(tail_bound[0])
          && std::isfinite(tail_bound[1])) {
        constexpr double roundoff =
            32.0 * std::numeric_limits<double>::epsilon();
        solution.ua_relative_error_bound = tail_bound[0] + roundoff;
        solution.uc_relative_error_bound = tail_bound[1] + roundoff;
        solution.used_extended_gram = true;
        bounded = true;
        break;
      }
      if (refinement == 3) break;
      if (residual >= previous_residual) break;
      previous_residual = residual;
      std::vector<double> correction(static_cast<std::size_t>(m_) * 2);
      for (int position = 0; position < m_; ++position) {
        const auto original = permutation_[position];
        correction[position] = -static_cast<double>(transpose_g[original]);
        correction[m_ + position] =
            -static_cast<double>(transpose_h[original]);
      }
      if (!triangular_pair(correction, true)) return false;
      if (!triangular_pair(correction, false)) return false;
      for (int right = 0; right < 2; ++right) {
        double correction_norm = 0.0, solution_norm = 0.0;
        for (int position = 0; position < m_; ++position) {
          correction_norm = std::max(
              correction_norm,
              std::abs(correction[static_cast<std::size_t>(right) * m_
                                  + position]));
          solution_norm = std::max(
              solution_norm,
              std::abs(heads[static_cast<std::size_t>(right) * m_
                             + position]));
        }
        const double relative = correction_norm
                                / std::max(1.0, solution_norm);
        if (relative <= 5e-15) {
          tail_bound[right] = relative;
        } else if (std::isfinite(previous_correction[right])
                   && previous_correction[right] > 0.0) {
          const double contraction = relative / previous_correction[right];
          tail_bound[right] = contraction < 0.5
                                  ? relative * contraction / (1.0 - contraction)
                                  : std::numeric_limits<double>::infinity();
        } else {
          tail_bound[right] = std::numeric_limits<double>::infinity();
        }
        previous_correction[right] = relative;
      }
      for (std::size_t entry = 0; entry < heads.size(); ++entry)
        heads[entry] += correction[entry];
    }
    if (!bounded) return false;
    cache_products(fixture_, solution);
    return std::isfinite(solution.dres)
           && std::isfinite(solution.piece_residual);
  }

  const Fixture &fixture_;
  const std::vector<double> &target_shift_;
  FaceSolveStats &stats_;
  int m_;
  bool valid_ = false;
  std::int64_t last_oracle_rank_ = -1;
  std::size_t updates_since_refactor_ = 0;
  std::vector<std::uint32_t> rows_;
  std::vector<std::int64_t> permutation_, inverse_permutation_;
  std::vector<double> R_, cross_;
  int pattern_words_ = 0;
  std::vector<std::uint64_t> row_pattern_;
  std::size_t pattern_entries_ = 0;
  bool dense_pattern_ = false;
  std::vector<double> candidate_times_us_, oracle_times_us_;
};

}  // namespace

std::size_t FaceSolver::RowHash::operator()(
    const std::vector<std::uint32_t> &rows) const {
  std::size_t hash = 1469598103934665603ULL;
  for (auto row : rows) {
    hash ^= static_cast<std::size_t>(row) + 0x9e3779b97f4a7c15ULL
            + (hash << 6) + (hash >> 2);
    hash *= 1099511628211ULL;
  }
  return hash;
}

FaceSolver::FaceSolver(const Fixture &fixture, bool enable_cache,
                       std::vector<double> target_shift, int spqr_ordering,
                       bool allow_unguarded_direct)
    : fixture_(fixture), target_shift_(std::move(target_shift)),
      enable_cache_(enable_cache),
      spqr_ordering_(spqr_ordering < 0 ? SPQR_ORDERING_DEFAULT
                                       : spqr_ordering),
      allow_unguarded_direct_(allow_unguarded_direct) {
  if (!target_shift_.empty() && target_shift_.size() != fixture_.n)
    throw FaceDecline("target shift has wrong dimension");
  if (!cholmod_l_start(&common_)) throw FaceDecline("cholmod start failed");
  if (enable_cache_) cache_.reserve(1024);
  const bool qr_audit = std::getenv("TWALKER_QR_UPDATE_AUDIT") != nullptr;
  const bool qr_live_explicit =
      std::getenv("TWALKER_QR_UPDATE_LIVE") != nullptr;
  std::size_t qr_min_columns = qr_live_explicit ? 0 : 500;
  if (const char *raw = std::getenv("TWALKER_QR_UPDATE_MIN_COLUMNS"))
    qr_min_columns = static_cast<std::size_t>(std::stoul(raw));
  if (qr_audit && qr_live_explicit)
    throw FaceDecline("QR update audit and live modes are mutually exclusive");
  qr_update_live_ = !qr_audit
                    && !std::getenv("TWALKER_DISABLE_QR_UPDATE_LIVE")
                    && fixture_.m >= qr_min_columns;
  if (enable_cache_ && (qr_audit || qr_update_live_))
    qr_update_audit_ = new UpdatedQrAudit(fixture_, target_shift_, stats_);
}

FaceSolver::~FaceSolver() {
  delete static_cast<UpdatedQrAudit *>(qr_update_audit_);
  cholmod_l_finish(&common_);
}

FaceSolution FaceSolver::solve(const std::vector<std::uint32_t> &rows) {
  const auto total_start = Clock::now();
  ++stats_.calls;
  if (rows.empty()) throw FaceDecline("empty face");
  if (enable_cache_ && !force_numerical_) {
    const auto cache_start = Clock::now();
    const auto found = cache_.find(rows);
    // A cached face intentionally omits the large one-shot maintenance
    // artifacts.  When a persistent solver asks to reseed, recompute this
    // face once instead of returning an artifact-free cache entry.
    if (found != cache_.end() && !recurrence_seed_needed_) {
      FaceSolution cached = found->second;
      if (cached.qr_update_audit_candidate) {
        cached.qr_update_audit_candidate = false;
        cached.qr_update_audit_g.clear();
        cached.qr_update_audit_h.clear();
        cached.qr_update_audit_bua.clear();
        cached.qr_update_audit_buc.clear();
      }
      ++stats_.cache_hits;
      stats_.cache_ms += milliseconds_since(cache_start);
      stats_.total_ms += milliseconds_since(total_start);
      return cached;
    }
    stats_.cache_ms += milliseconds_since(cache_start);
  }
  if (!force_numerical_ && qr_update_live_ && qr_update_audit_) {
    FaceSolution candidate;
    if (static_cast<UpdatedQrAudit *>(qr_update_audit_)
            ->try_live(rows, candidate)) {
      stats_.total_ms += milliseconds_since(total_start);
      return candidate;
    }
  }
  ++stats_.numerical_calls;
  std::vector<double> shared_pseudoinverse;
  std::int64_t shared_seed_rank = 0;
  std::vector<double> shared_factored_core;
  std::vector<double> shared_factored_rz_core;
  std::vector<double> shared_factored_rz_tau;
  std::vector<std::int64_t> shared_factored_permutation;
  std::int64_t shared_factored_rank = 0;
  std::vector<double> shared_svd_left, shared_svd_singular, shared_svd_right;
  auto finish = [&](FaceSolution solution,
                    const std::vector<double> *fresh_core,
                    const std::vector<std::int64_t> *fresh_permutation) {
    const double oracle_ms = milliseconds_since(total_start);
    stats_.total_ms += oracle_ms;
    if (qr_update_audit_) {
      if (qr_update_live_)
        static_cast<UpdatedQrAudit *>(qr_update_audit_)
            ->reseed(rows, solution, fresh_core, fresh_permutation);
      else {
        FaceSolution audit_candidate;
        static_cast<UpdatedQrAudit *>(qr_update_audit_)
            ->observe(rows, solution, fresh_core, fresh_permutation,
                      oracle_ms, &audit_candidate);
        if (!audit_candidate.rows.empty()) {
          solution.qr_update_audit_candidate = true;
          solution.qr_update_audit_g = std::move(audit_candidate.g);
          solution.qr_update_audit_h = std::move(audit_candidate.h);
          solution.qr_update_audit_bua = std::move(audit_candidate.bua);
          solution.qr_update_audit_buc = std::move(audit_candidate.buc);
        }
      }
    }
    if (!solution.recurrence_pseudoinverse.empty()) {
      shared_pseudoinverse =
          std::move(solution.recurrence_pseudoinverse);
      shared_seed_rank = solution.recurrence_seed_rank;
      solution.recurrence_seed_rank = 0;
    }
    if (!solution.svd_left_space.empty()) {
      shared_svd_left = std::move(solution.svd_left_space);
      shared_svd_singular = std::move(solution.svd_singular_values);
      shared_svd_right = std::move(solution.svd_row_space);
    }
    // The walk is bounded at 2000 accepted pivots.  Capping retained faces
    // makes the memory contract explicit while preserving all observed repeats.
    if (enable_cache_ && !force_numerical_ && cache_.size() < 2048) {
      const auto inserted = cache_.emplace(rows, solution);
      if (inserted.second) {
        auto &cached = inserted.first->second;
        if (cached.qr_update_audit_candidate) {
          cached.qr_update_audit_candidate = false;
          cached.qr_update_audit_g.clear();
          cached.qr_update_audit_h.clear();
          cached.qr_update_audit_bua.clear();
          cached.qr_update_audit_buc.clear();
        }
        ++stats_.cache_inserts;
      }
    }
    // Keep the large one-shot seed out of the ordinary face cache.  Walker
    // consumes it immediately; retaining another copy only increases memory.
    if (!shared_pseudoinverse.empty() && !solution.used_core_svd
        && solution.rank == shared_seed_rank) {
      solution.recurrence_pseudoinverse = std::move(shared_pseudoinverse);
      solution.recurrence_seed_rank = shared_seed_rank;
    }
    if (!shared_svd_left.empty() && solution.used_dense_fallback) {
      solution.svd_left_space = std::move(shared_svd_left);
      solution.svd_singular_values = std::move(shared_svd_singular);
      solution.svd_row_space = std::move(shared_svd_right);
    }
    const bool retain_dense_factored_seed =
        !std::getenv("TWALKER_DISABLE_REVISED_COST_AWARE_SEED")
        && solution.used_dense_fallback
        && solution.rank == shared_factored_rank;
    if (!shared_factored_core.empty()
        && (!solution.used_dense_fallback || retain_dense_factored_seed)
        && !solution.used_core_svd
        && solution.rank == shared_factored_rank) {
      solution.factored_qr_core = std::move(shared_factored_core);
      solution.factored_rz_core = std::move(shared_factored_rz_core);
      solution.factored_rz_tau = std::move(shared_factored_rz_tau);
      solution.factored_permutation = std::move(shared_factored_permutation);
      solution.factored_seed_rank = shared_factored_rank;
    }
    return solution;
  };
  const int m = static_cast<int>(fixture_.m);
  std::size_t face_nnz = 0;
  for (auto row : rows) {
    if (row >= fixture_.n) throw FaceDecline("face row out of range");
    face_nnz += fixture_.indptr[row + 1] - fixture_.indptr[row];
  }

  const auto assembly_start = Clock::now();
  auto *triplet = cholmod_l_allocate_triplet(
      rows.size(), fixture_.m, face_nnz, 0, CHOLMOD_REAL, &common_);
  if (!triplet) throw FaceDecline("triplet allocation failed");
  auto *ti = static_cast<std::int64_t *>(triplet->i);
  auto *tj = static_cast<std::int64_t *>(triplet->j);
  auto *tx = static_cast<double *>(triplet->x);
  std::size_t cursor = 0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      ti[cursor] = static_cast<std::int64_t>(local);
      tj[cursor] = fixture_.indices[p];
      tx[cursor] = fixture_.values[p];
      ++cursor;
    }
  }
  triplet->nnz = cursor;
  auto *A = cholmod_l_triplet_to_sparse(triplet, cursor, &common_);
  cholmod_l_free_triplet(&triplet, &common_);
  const bool export_recurrence_seed = recurrence_seed_needed_;
  const std::size_t rhs_columns = export_recurrence_seed ? rows.size() + 1 : 1;
  auto *rhs = cholmod_l_allocate_dense(rows.size(), rhs_columns, rows.size(),
                                       CHOLMOD_REAL, &common_);
  if (!A || !rhs) {
    if (A) cholmod_l_free_sparse(&A, &common_);
    if (rhs) cholmod_l_free_dense(&rhs, &common_);
    throw FaceDecline("SPQR input allocation failed");
  }
  auto *rhs_values = static_cast<double *>(rhs->x);
  for (std::size_t i = 0; i < rows.size(); ++i)
    rhs_values[i] = fixture_.b[rows[i]];
  if (export_recurrence_seed)
    for (std::size_t i = 0; i < rows.size(); ++i)
      rhs_values[i + rows.size() * (i + 1)] = 1.0;
  stats_.assembly_ms += milliseconds_since(assembly_start);

  const auto spqr_start = Clock::now();
  cholmod_dense *qtb_dense = nullptr;
  cholmod_sparse *R = nullptr;
  std::int64_t *permutation_raw = nullptr;
  const auto rank64 = SuiteSparseQR_C(
      spqr_ordering_, SPQR_DEFAULT_TOL, fixture_.m, 0, A, nullptr, rhs,
      nullptr, &qtb_dense, &R, &permutation_raw, nullptr, nullptr, nullptr,
      &common_);
  cholmod_l_free_sparse(&A, &common_);
  cholmod_l_free_dense(&rhs, &common_);
  stats_.spqr_ms += milliseconds_since(spqr_start);
  if (rank64 <= 0 || !qtb_dense || !R) {
    if (qtb_dense) cholmod_l_free_dense(&qtb_dense, &common_);
    if (R) cholmod_l_free_sparse(&R, &common_);
    if (permutation_raw)
      cholmod_l_free(fixture_.m, sizeof(std::int64_t), permutation_raw,
                     &common_);
    throw FaceDecline("SuiteSparseQR_C failed or returned zero rank");
  }
  if (rank64 > m) throw FaceDecline("SPQR rank exceeds column count");
  const int rank = static_cast<int>(rank64);
  std::vector<std::int64_t> permutation(m);
  if (permutation_raw) {
    std::copy(permutation_raw, permutation_raw + m, permutation.begin());
    permutation_raw = static_cast<std::int64_t *>(cholmod_l_free(
        fixture_.m, sizeof(std::int64_t), permutation_raw, &common_));
  } else {
    std::iota(permutation.begin(), permutation.end(), 0);
  }

  const auto core_start = Clock::now();
  const auto *rp = static_cast<const std::int64_t *>(R->p);
  const auto *ri = static_cast<const std::int64_t *>(R->i);
  const auto *rx = static_cast<const double *>(R->x);
  const auto nnz_R = static_cast<std::size_t>(rp[R->ncol]);
  const bool collect_core_svd = !allow_unguarded_direct_
      && std::getenv("TWALKER_CORE_SVD_AUDIT") != nullptr;
  const int economy_rows = collect_core_svd
      ? std::min({static_cast<int>(R->nrow),
                  static_cast<int>(qtb_dense->nrow), m,
                  static_cast<int>(rows.size())})
      : rank;
  if (economy_rows < rank) throw FaceDecline("SPQR economy core below rank");
  std::vector<double> economy_core(
      static_cast<std::size_t>(economy_rows) * m, 0.0);
  for (int column = 0; column < m; ++column) {
    for (auto p = rp[column]; p < rp[column + 1]; ++p) {
      if (ri[p] < economy_rows)
        economy_core[static_cast<std::size_t>(ri[p])
                     + static_cast<std::size_t>(economy_rows) * column] = rx[p];
    }
  }
  const auto *qtb_values = static_cast<const double *>(qtb_dense->x);
  const auto qtb_leading = static_cast<std::size_t>(qtb_dense->d);
  std::vector<double> economy_qtb(qtb_values, qtb_values + economy_rows);
  std::vector<double> qtb(qtb_values, qtb_values + rank);
  std::vector<double> qt_identity;
  if (export_recurrence_seed) {
    qt_identity.resize(static_cast<std::size_t>(rank) * rows.size());
    for (std::size_t column = 0; column < rows.size(); ++column)
      for (int row = 0; row < rank; ++row)
        qt_identity[row + static_cast<std::size_t>(rank) * column] =
            qtb_values[row + qtb_leading * (column + 1)];
  }
  std::vector<double> core(static_cast<std::size_t>(rank) * m, 0.0);
  for (int column = 0; column < m; ++column)
    for (int row = 0; row < rank; ++row)
      core[row + static_cast<std::size_t>(rank) * column] =
          economy_core[row + static_cast<std::size_t>(economy_rows) * column];
  if (factored_seed_needed_) {
    shared_factored_core = core;
    shared_factored_permutation = permutation;
    shared_factored_rank = rank;
  }
  cholmod_l_free_dense(&qtb_dense, &common_);
  cholmod_l_free_sparse(&R, &common_);
  stats_.core_extract_ms += milliseconds_since(core_start);

  std::vector<double> tau(rank);
  const auto rz_start = Clock::now();
  if (rank < m) {
    int info = 0;
    int lwork = -1;
    double query = 0.0;
    dtzrzf_(&rank, &m, core.data(), &rank, tau.data(), &query, &lwork, &info);
    require_lapack(info, "dtzrzf workspace query");
    lwork = workspace_size(query);
    std::vector<double> work(lwork);
    dtzrzf_(&rank, &m, core.data(), &rank, tau.data(), work.data(), &lwork,
            &info);
    require_lapack(info, "dtzrzf");
  }
  stats_.rz_ms += milliseconds_since(rz_start);

  if (factored_seed_needed_) {
    shared_factored_rz_core = core;
    shared_factored_rz_tau = tau;
  }

  // Bounded admission probe for the proposed strong/weak split.  RZ applies
  // an orthogonal right transform, so the leading r-by-r triangular block
  // has exactly the nonzero singular values of the face matrix.  Sampling
  // every tenth numerical face spans the path without turning the audit into
  // a second solver.
  if (std::getenv("TWALKER_WEAK_SPECTRUM_AUDIT")
      && stats_.weak_spectrum_samples < 64
      && (stats_.numerical_calls - 1) % 10 == 0) {
    const auto spectrum_start = Clock::now();
    std::vector<double> square(static_cast<std::size_t>(rank) * rank, 0.0);
    for (int column = 0; column < rank; ++column)
      for (int row = 0; row <= column; ++row)
        square[row + static_cast<std::size_t>(rank) * column] =
            core[row + static_cast<std::size_t>(rank) * column];
    std::vector<double> singular(rank);
    std::vector<int> spectrum_iwork(8 * std::max(1, rank));
    const char no_vectors = 'N';
    const int one = 1;
    double dummy_u = 0.0, dummy_vt = 0.0, query = 0.0;
    int info = 0, lwork = -1;
    dgesdd_(&no_vectors, &rank, &rank, square.data(), &rank,
            singular.data(), &dummy_u, &one, &dummy_vt, &one, &query,
            &lwork, spectrum_iwork.data(), &info);
    require_lapack(info, "dgesdd(weak spectrum query)");
    lwork = workspace_size(query);
    std::vector<double> spectrum_work(lwork);
    dgesdd_(&no_vectors, &rank, &rank, square.data(), &rank,
            singular.data(), &dummy_u, &one, &dummy_vt, &one,
            spectrum_work.data(), &lwork, spectrum_iwork.data(), &info);
    require_lapack(info, "dgesdd(weak spectrum)");
    const double sigma_max = singular.empty() ? 0.0 : singular.front();
    std::uint64_t q8 = 0, q10 = 0, q12 = 0;
    for (double sigma : singular) {
      q8 += sigma <= 1e-8 * sigma_max;
      q10 += sigma <= 1e-10 * sigma_max;
      q12 += sigma <= 1e-12 * sigma_max;
    }
    ++stats_.weak_spectrum_samples;
    stats_.weak_q8_over_16 += q8 > 16;
    stats_.weak_q8_max = std::max(stats_.weak_q8_max, q8);
    stats_.weak_q8_sum += q8;
    stats_.weak_q10_max = std::max(stats_.weak_q10_max, q10);
    stats_.weak_q10_sum += q10;
    stats_.weak_q12_max = std::max(stats_.weak_q12_max, q12);
    stats_.weak_q12_sum += q12;
    stats_.weak_spectrum_ms += milliseconds_since(spectrum_start);
  }

  if (export_recurrence_seed) {
    const int active = static_cast<int>(rows.size());
    std::vector<double> coefficients(
        static_cast<std::size_t>(m) * active, 0.0);
    for (int column = 0; column < active; ++column)
      for (int row = 0; row < rank; ++row)
        coefficients[row + static_cast<std::size_t>(m) * column] =
            qt_identity[row + static_cast<std::size_t>(rank) * column];
    {
      const char upper = 'U', no_transpose = 'N', nonunit = 'N';
      int info = 0;
      dtrtrs_(&upper, &no_transpose, &nonunit, &rank, &active, core.data(),
              &rank, coefficients.data(), &m, &info);
      require_lapack(info, "dtrtrs(T seed)");
    }
    if (rank < m) {
      const char side = 'L', transpose = 'T';
      const int reflector_tail = m - rank, ldc = m;
      int info = 0, lwork = -1;
      double query = 0.0;
      dormrz_(&side, &transpose, &m, &active, &rank, &reflector_tail,
              core.data(), &rank, tau.data(), coefficients.data(), &ldc,
              &query, &lwork, &info);
      require_lapack(info, "dormrz(Z' seed) workspace query");
      lwork = workspace_size(query);
      std::vector<double> work(lwork);
      dormrz_(&side, &transpose, &m, &active, &rank, &reflector_tail,
              core.data(), &rank, tau.data(), coefficients.data(), &ldc,
              work.data(), &lwork, &info);
      require_lapack(info, "dormrz(Z' seed)");
    }
    shared_pseudoinverse.assign(
        static_cast<std::size_t>(m) * active, 0.0);
    for (int column = 0; column < active; ++column)
      for (int position = 0; position < m; ++position)
        shared_pseudoinverse[
            permutation[position] + static_cast<std::size_t>(m) * column] =
            coefficients[position + static_cast<std::size_t>(m) * column];
    shared_seed_rank = rank;
  }

  double diagonal_min = std::numeric_limits<double>::infinity();
  double diagonal_max = 0.0;
  for (int i = 0; i < rank; ++i) {
    const double value = std::abs(core[static_cast<std::size_t>(i) * rank + i]);
    diagonal_min = std::min(diagonal_min, value);
    diagonal_max = std::max(diagonal_max, value);
  }
  const double core_ratio = diagonal_min / diagonal_max;
  const double cutoff = std::max(rank, m) * std::numeric_limits<double>::epsilon();
  const bool core_svd_audit = !allow_unguarded_direct_
      && std::getenv("TWALKER_CORE_SVD_AUDIT") != nullptr
      && (!std::isfinite(core_ratio) || core_ratio < 1e-5);
  if (core_svd_audit) {
    ++stats_.core_svd_audits;
    const auto core_svd_start = Clock::now();
    std::vector<double> core_spectrum;
    auto candidate = core_svd_face(
        fixture_, rows, target_shift_, economy_core, economy_rows,
        economy_qtb, permutation, &core_spectrum);
    if (!core_spectrum.empty()) {
      const double cutoff = core_spectrum.front()
          * std::max(static_cast<int>(rows.size()), m)
          * std::numeric_limits<double>::epsilon();
      std::uint64_t near = 0;
      for (std::size_t component = static_cast<std::size_t>(rank);
           component < core_spectrum.size(); ++component)
        near += core_spectrum[component] > 0.01 * cutoff;
      stats_.discarded_near_cutoff_max = std::max(
          stats_.discarded_near_cutoff_max, near);
      stats_.discarded_near_cutoff_sum += near;
    }
    stats_.core_svd_ms += milliseconds_since(core_svd_start);
    const auto svd_start = Clock::now();
    auto oracle = dense_svd_face(fixture_, rows, target_shift_,
                                 factored_seed_needed_,
                                 recurrence_seed_needed_);
    stats_.dense_svd_ms += milliseconds_since(svd_start);
    ++stats_.dense_fallbacks;
    candidate.nnz_R = oracle.nnz_R = nnz_R;
    candidate.core_diagonal_ratio = oracle.core_diagonal_ratio = core_ratio;
    const double ua_error = relative_inf_error(candidate.ua, oracle.ua);
    const double uc_error = relative_inf_error(candidate.uc, oracle.uc);
    const double g_error = relative_inf_error(candidate.g, oracle.g);
    const double h_error = relative_inf_error(candidate.h, oracle.h);
    const double error = std::max({ua_error, uc_error, g_error, h_error});
    stats_.core_svd_max_error = std::max(stats_.core_svd_max_error, error);
    stats_.core_svd_max_ua_error = std::max(
        stats_.core_svd_max_ua_error, ua_error);
    stats_.core_svd_max_uc_error = std::max(
        stats_.core_svd_max_uc_error, uc_error);
    stats_.core_svd_max_g_error = std::max(
        stats_.core_svd_max_g_error, g_error);
    stats_.core_svd_max_h_error = std::max(
        stats_.core_svd_max_h_error, h_error);
    if (candidate.rank != oracle.rank)
      ++stats_.core_svd_rank_mismatches;
    if (candidate.rank == oracle.rank && error <= 1e-10)
      ++stats_.core_svd_accurate;
    return finish(std::move(oracle), &core, &permutation);
  }
  if (!std::isfinite(core_ratio) || core_ratio <= cutoff) {
    const auto svd_start = Clock::now();
    auto fallback = dense_svd_face(fixture_, rows, target_shift_,
                                   factored_seed_needed_,
                                   recurrence_seed_needed_);
    stats_.dense_svd_ms += milliseconds_since(svd_start);
    ++stats_.dense_fallbacks;
    fallback.nnz_R = nnz_R;
    fallback.core_diagonal_ratio = core_ratio;
    return finish(std::move(fallback), &core, &permutation);
  }
  // A residual gate does not detect forward error in the minimum-norm
  // multipliers.  On boeing2, cores below 5.5e-6 passed every residual at
  // 1e-9 yet disagreed with the SVD oracle by 1.5e-7.  Decline conservatively
  // at 1e-5 to an independent SVD; this is a correctness fallback, not a
  // tuned answer path.
  const bool direct_guard_audit = core_ratio < 1e-5
      && !allow_unguarded_direct_
      && std::getenv("TWALKER_DIRECT_GUARD_AUDIT") != nullptr;
  bool unguarded_direct_allowed = allow_unguarded_direct_;
  if (const char *raw = std::getenv(
          "TWALKER_DIRECT_CANDIDATE_MAX_DEFICIT"))
    unguarded_direct_allowed = unguarded_direct_allowed
        && m - rank <= std::max(0, std::atoi(raw));
  if (core_ratio < 1e-5 && !direct_guard_audit
      && !unguarded_direct_allowed) {
    const auto svd_start = Clock::now();
    auto fallback = dense_svd_face(fixture_, rows, target_shift_,
                                   factored_seed_needed_,
                                   recurrence_seed_needed_);
    stats_.dense_svd_ms += milliseconds_since(svd_start);
    ++stats_.dense_fallbacks;
    fallback.nnz_R = nnz_R;
    fallback.core_diagonal_ratio = core_ratio;
    return finish(std::move(fallback), &core, &permutation);
  }

  const auto triangular_start = Clock::now();
  std::vector<double> adjusted_d = fixture_.d;
  if (!target_shift_.empty()) {
    for (auto row : rows)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        adjusted_d[fixture_.indices[p]] -=
            fixture_.values[p] * target_shift_[row];
  }
  std::vector<double> transformed_d(m);
  for (int j = 0; j < m; ++j)
    transformed_d[j] = adjusted_d[permutation[j]];
  if (rank < m) {
    const char side = 'L', trans = 'N';
    const int one = 1, reflector_tail = m - rank, ldc = m;
    int info = 0, lwork = -1;
    double query = 0.0;
    dormrz_(&side, &trans, &m, &one, &rank, &reflector_tail, core.data(),
            &rank, tau.data(), transformed_d.data(), &ldc, &query, &lwork,
            &info);
    require_lapack(info, "dormrz(Z*d) workspace query");
    lwork = workspace_size(query);
    std::vector<double> work(lwork);
    dormrz_(&side, &trans, &m, &one, &rank, &reflector_tail, core.data(),
            &rank, tau.data(), transformed_d.data(), &ldc, work.data(), &lwork,
            &info);
    require_lapack(info, "dormrz(Z*d)");
  }

  std::vector<double> z_rhs(transformed_d.begin(),
                            transformed_d.begin() + rank);
  std::vector<double> z = z_rhs;
  {
    const char upper = 'U', transpose = 'T', nonunit = 'N';
    const int one = 1;
    int info = 0;
    dtrtrs_(&upper, &transpose, &nonunit, &rank, &one, core.data(), &rank,
            z.data(), &rank, &info);
    require_lapack(info, "dtrtrs(T')");
  }

  const bool collect_direct_guard = !allow_unguarded_direct_
      && std::getenv("TWALKER_DIRECT_GUARD_AUDIT") != nullptr;
  double transpose_ferr = std::numeric_limits<double>::infinity();
  if (collect_direct_guard) {
    const char upper = 'U', transpose = 'T', nonunit = 'N';
    const int one = 1;
    int info = 0;
    double berr = 0.0;
    std::vector<double> work(3 * rank);
    std::vector<int> iwork(rank);
    dtrrfs_(&upper, &transpose, &nonunit, &rank, &one, core.data(), &rank,
            z_rhs.data(), &rank, z.data(), &rank, &transpose_ferr, &berr,
            work.data(), iwork.data(), &info);
    require_lapack(info, "dtrrfs(T')");
  }

  std::vector<double> heads(static_cast<std::size_t>(rank) * 2);
  for (int i = 0; i < rank; ++i) {
    heads[i] = -qtb[i];
    heads[rank + i] = z[i];
  }
  const auto heads_rhs = heads;
  {
    const char upper = 'U', no_transpose = 'N', nonunit = 'N';
    const int two = 2;
    int info = 0;
    dtrtrs_(&upper, &no_transpose, &nonunit, &rank, &two, core.data(), &rank,
            heads.data(), &rank, &info);
    require_lapack(info, "dtrtrs(T)");
  }

  double solve_ferr[2] = {std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity()};
  if (collect_direct_guard) {
    const char upper = 'U', no_transpose = 'N', nonunit = 'N';
    const int two = 2;
    int info = 0;
    double berr[2] = {0.0, 0.0};
    std::vector<double> work(3 * rank);
    std::vector<int> iwork(rank);
    dtrrfs_(&upper, &no_transpose, &nonunit, &rank, &two, core.data(), &rank,
            heads_rhs.data(), &rank, heads.data(), &rank, solve_ferr, berr,
            work.data(), iwork.data(), &info);
    require_lapack(info, "dtrrfs(T)");
  }
  const double cod_forward_estimate = collect_direct_guard
      ? std::max(solve_ferr[0], solve_ferr[1] + transpose_ferr
                                    + solve_ferr[1] * transpose_ferr)
      : std::numeric_limits<double>::infinity();

  std::vector<double> coefficients(static_cast<std::size_t>(m) * 2, 0.0);
  for (int i = 0; i < rank; ++i) {
    coefficients[i] = heads[i];
    coefficients[m + i] = heads[rank + i];
  }
  if (rank < m) {
    const char side = 'L', transpose = 'T';
    const int two = 2, reflector_tail = m - rank, ldc = m;
    int info = 0, lwork = -1;
    double query = 0.0;
    dormrz_(&side, &transpose, &m, &two, &rank, &reflector_tail, core.data(),
            &rank, tau.data(), coefficients.data(), &ldc, &query, &lwork,
            &info);
    require_lapack(info, "dormrz(Z') workspace query");
    lwork = workspace_size(query);
    std::vector<double> work(lwork);
    dormrz_(&side, &transpose, &m, &two, &rank, &reflector_tail, core.data(),
            &rank, tau.data(), coefficients.data(), &ldc, work.data(), &lwork,
            &info);
    require_lapack(info, "dormrz(Z')");
  }
  stats_.triangular_ms += milliseconds_since(triangular_start);

  FaceSolution solution;
  solution.rank = rank;
  solution.rows = rows;
  solution.nnz_R = nnz_R;
  solution.core_diagonal_ratio = core_ratio;
  solution.ua.assign(m, 0.0);
  solution.uc.assign(m, 0.0);
  for (int j = 0; j < m; ++j) {
    solution.ua[permutation[j]] = coefficients[j];
    solution.uc[permutation[j]] = coefficients[m + j];
  }
  const auto products_start = Clock::now();
  cache_products(fixture_, solution);
  stats_.products_ms += milliseconds_since(products_start);

  if (direct_guard_audit) {
    ++stats_.direct_guard_audits;
    const auto refinement_start = Clock::now();
    const auto raw_ua = solution.ua;
    const auto raw_uc = solution.uc;
    double final_tail = std::numeric_limits<double>::infinity();
    double maximum_contraction = std::numeric_limits<double>::infinity();
    bool refinement_completed = false;

    auto solve_gram_correction = [&](const std::vector<double> &rhs) {
      std::vector<double> transformed(static_cast<std::size_t>(m) * 2);
      for (int right = 0; right < 2; ++right)
        for (int j = 0; j < m; ++j)
          transformed[static_cast<std::size_t>(right) * m + j] =
              rhs[static_cast<std::size_t>(right) * m + permutation[j]];
      if (rank < m) {
        const char side = 'L', trans = 'N';
        const int two = 2, reflector_tail = m - rank, ldc = m;
        int info = 0, lwork = -1;
        double query = 0.0;
        dormrz_(&side, &trans, &m, &two, &rank, &reflector_tail,
                core.data(), &rank, tau.data(), transformed.data(), &ldc,
                &query, &lwork, &info);
        require_lapack(info, "dormrz(Z*r) workspace query");
        lwork = workspace_size(query);
        std::vector<double> work(lwork);
        dormrz_(&side, &trans, &m, &two, &rank, &reflector_tail,
                core.data(), &rank, tau.data(), transformed.data(), &ldc,
                work.data(), &lwork, &info);
        require_lapack(info, "dormrz(Z*r)");
      }
      {
        const char upper = 'U', transpose = 'T', nonunit = 'N';
        const int two = 2, ldb = m;
        int info = 0;
        dtrtrs_(&upper, &transpose, &nonunit, &rank, &two, core.data(),
                &rank, transformed.data(), &ldb, &info);
        require_lapack(info, "dtrtrs(T' correction)");
      }
      {
        const char upper = 'U', no_transpose = 'N', nonunit = 'N';
        const int two = 2, ldb = m;
        int info = 0;
        dtrtrs_(&upper, &no_transpose, &nonunit, &rank, &two, core.data(),
                &rank, transformed.data(), &ldb, &info);
        require_lapack(info, "dtrtrs(T correction)");
      }
      for (int right = 0; right < 2; ++right)
        std::fill(transformed.begin() + static_cast<std::size_t>(right) * m
                      + rank,
                  transformed.begin() + static_cast<std::size_t>(right + 1) * m,
                  0.0);
      if (rank < m) {
        const char side = 'L', transpose = 'T';
        const int two = 2, reflector_tail = m - rank, ldc = m;
        int info = 0, lwork = -1;
        double query = 0.0;
        dormrz_(&side, &transpose, &m, &two, &rank, &reflector_tail,
                core.data(), &rank, tau.data(), transformed.data(), &ldc,
                &query, &lwork, &info);
        require_lapack(info, "dormrz(Z' correction) workspace query");
        lwork = workspace_size(query);
        std::vector<double> work(lwork);
        dormrz_(&side, &transpose, &m, &two, &rank, &reflector_tail,
                core.data(), &rank, tau.data(), transformed.data(), &ldc,
                work.data(), &lwork, &info);
        require_lapack(info, "dormrz(Z' correction)");
      }
      std::vector<double> correction(static_cast<std::size_t>(m) * 2);
      for (int right = 0; right < 2; ++right)
        for (int j = 0; j < m; ++j)
          correction[static_cast<std::size_t>(right) * m + permutation[j]] =
              transformed[static_cast<std::size_t>(right) * m + j];
      return correction;
    };

    try {
      double previous[2] = {std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity()};
      for (int iteration = 0; iteration < 5; ++iteration) {
        std::vector<long double> residual(static_cast<std::size_t>(m) * 2,
                                          0.0L);
        for (int column = 0; column < m; ++column)
          residual[m + column] = static_cast<long double>(adjusted_d[column]);
        for (auto row : rows) {
          const long double bw = static_cast<long double>(fixture_.b[row]);
          long double ax[2] = {0.0L, 0.0L};
          for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1];
               ++p) {
            const auto column = fixture_.indices[p];
            const long double value = fixture_.values[p];
            residual[column] -= value * bw;
            ax[0] += value * static_cast<long double>(solution.ua[column]);
            ax[1] += value * static_cast<long double>(solution.uc[column]);
          }
          for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1];
               ++p) {
            const auto column = fixture_.indices[p];
            const long double value = fixture_.values[p];
            residual[column] -= value * ax[0];
            residual[m + column] -= value * ax[1];
          }
        }
        std::vector<double> rhs(static_cast<std::size_t>(m) * 2);
        for (std::size_t entry = 0; entry < rhs.size(); ++entry)
          rhs[entry] = static_cast<double>(residual[entry]);
        const auto correction = solve_gram_correction(rhs);
        double relative[2] = {0.0, 0.0};
        for (int right = 0; right < 2; ++right) {
          const auto &current = right == 0 ? solution.ua : solution.uc;
          double correction_norm = 0.0, current_norm = 0.0;
          for (int column = 0; column < m; ++column) {
            correction_norm = std::max(
                correction_norm,
                std::abs(correction[static_cast<std::size_t>(right) * m
                                    + column]));
            current_norm = std::max(current_norm, std::abs(current[column]));
          }
          relative[right] = correction_norm / std::max(1.0, current_norm);
        }
        for (int column = 0; column < m; ++column) {
          solution.ua[column] += correction[column];
          solution.uc[column] += correction[m + column];
        }
        double tail[2] = {std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity()};
        double contraction[2] = {std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity()};
        for (int right = 0; right < 2; ++right) {
          if (relative[right] <= 1e-14) {
            contraction[right] = 0.0;
            tail[right] = relative[right];
          } else if (std::isfinite(previous[right]) && previous[right] > 0.0) {
            contraction[right] = relative[right] / previous[right];
            if (contraction[right] < 0.5)
              tail[right] = relative[right] * contraction[right]
                            / (1.0 - contraction[right]);
          }
          previous[right] = relative[right];
        }
        final_tail = std::max(tail[0], tail[1]);
        maximum_contraction = std::max(contraction[0], contraction[1]);
        if (iteration > 0 && final_tail <= 5e-11) {
          refinement_completed = true;
          break;
        }
      }
    } catch (const FaceDecline &) {
      refinement_completed = false;
    }
    stats_.direct_guard_refinement_ms += milliseconds_since(refinement_start);

    const auto svd_start = Clock::now();
    auto oracle = dense_svd_face(fixture_, rows, target_shift_,
                                 factored_seed_needed_,
                                 recurrence_seed_needed_);
    stats_.dense_svd_ms += milliseconds_since(svd_start);
    ++stats_.dense_fallbacks;
    oracle.nnz_R = nnz_R;
    oracle.core_diagonal_ratio = core_ratio;
    const double raw_error = std::max(relative_inf_error(raw_ua, oracle.ua),
                                      relative_inf_error(raw_uc, oracle.uc));
    const double refined_error = std::max(
        relative_inf_error(solution.ua, oracle.ua),
        relative_inf_error(solution.uc, oracle.uc));
    const auto consensus_start = Clock::now();
    if (!alternate_solver_)
      alternate_solver_ = std::make_unique<FaceSolver>(
          fixture_, false, target_shift_, SPQR_ORDERING_AMD, true);
    FaceSolution alternate;
    bool alternate_usable = false;
    try {
      alternate = alternate_solver_->solve(rows);
      alternate_usable = !alternate.used_dense_fallback;
    } catch (const FaceDecline &) {
      alternate_usable = false;
    }
    stats_.direct_guard_consensus_ms += milliseconds_since(consensus_start);
    ++stats_.direct_guard_consensus_attempts;
    double consensus_error = std::numeric_limits<double>::infinity();
    if (alternate_usable && alternate.rank == solution.rank)
      consensus_error = std::max(
          relative_inf_error(raw_ua, alternate.ua),
          relative_inf_error(raw_uc, alternate.uc));
    const bool consensus_accept = consensus_error <= 1e-10;
    const bool tight_consensus_accept = consensus_error <= 1e-14;
    const bool rank_matches = solution.rank == oracle.rank;
    if (!rank_matches) ++stats_.direct_guard_rank_mismatches;
    stats_.direct_guard_raw_max_error = std::max(
        stats_.direct_guard_raw_max_error, raw_error);
    stats_.direct_guard_refined_max_error = std::max(
        stats_.direct_guard_refined_max_error, refined_error);
    if (raw_error <= 1e-10) ++stats_.direct_guard_raw_accurate;
    if (refined_error <= 1e-10) {
      ++stats_.direct_guard_refined_accurate;
      if (rank_matches)
        ++stats_.direct_guard_rank_match_accurate;
      else
        ++stats_.direct_guard_rank_mismatch_accurate;
    }
    const bool tail_accept = refinement_completed && final_tail <= 5e-11
                             && maximum_contraction < 0.5;
    const bool ferr_accept = std::isfinite(cod_forward_estimate)
                             && cod_forward_estimate <= 1e-10;
    if (std::getenv("TWALKER_DIRECT_GUARD_TRACE")
        && refined_error > 1e-10) {
      std::cerr << "DIRECT_GUARD_BAD core=" << core_ratio
                << " spqr_rank=" << solution.rank
                << " svd_rank=" << oracle.rank
                << " raw=" << raw_error
                << " refined=" << refined_error
                << " tail=" << final_tail
                << " contraction=" << maximum_contraction
                << " ferr=" << cod_forward_estimate
                << " consensus=" << consensus_error
                << " tail_accept=" << tail_accept << '\n';
    }
    if (tail_accept) {
      ++stats_.direct_guard_tail_accepts;
      stats_.direct_guard_tail_worst_error = std::max(
          stats_.direct_guard_tail_worst_error, refined_error);
      if (refined_error > 1e-10)
        ++stats_.direct_guard_tail_false_accepts;
    }
    if (ferr_accept) {
      ++stats_.direct_guard_ferr_accepts;
      stats_.direct_guard_ferr_worst_error = std::max(
          stats_.direct_guard_ferr_worst_error, refined_error);
      if (refined_error > 1e-10)
        ++stats_.direct_guard_ferr_false_accepts;
    }
    if (consensus_accept) {
      ++stats_.direct_guard_consensus_accepts;
      stats_.direct_guard_consensus_worst_error = std::max(
          stats_.direct_guard_consensus_worst_error, raw_error);
      if (raw_error > 1e-10)
        ++stats_.direct_guard_consensus_false_accepts;
    }
    if (tight_consensus_accept) {
      ++stats_.direct_guard_tight_consensus_accepts;
      stats_.direct_guard_tight_consensus_worst_error = std::max(
          stats_.direct_guard_tight_consensus_worst_error, raw_error);
      if (raw_error > 1e-10)
        ++stats_.direct_guard_tight_consensus_false_accepts;
    }
    return finish(std::move(oracle), &core, &permutation);
  }

  const auto residual_start = Clock::now();
  solution.g.resize(rows.size());
  solution.h.resize(rows.size());
  std::vector<double> transpose_h(m, 0.0), transpose_g(m, 0.0);
  double slope_error = 0.0, constant_error = 0.0;
  double slope_scale = 1.0, constant_scale = 1.0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    const double bua = solution.bua[row];
    const double buc = solution.buc[row];
    solution.g[local] = fixture_.b[row] + bua;
    const double shift = target_shift_.empty() ? 0.0 : target_shift_[row];
    solution.h[local] = shift + buc;
    const double expected_slope = solution.g[local] - fixture_.b[row];
    slope_error = std::max(slope_error, std::abs(bua - expected_slope));
    constant_error = std::max(
        constant_error, std::abs(buc - (solution.h[local] - shift)));
    slope_scale = std::max(slope_scale, std::abs(expected_slope));
    constant_scale = std::max(constant_scale, std::abs(solution.h[local]));
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      const auto value = fixture_.values[p];
      transpose_h[column] += value * solution.h[local];
      transpose_g[column] += value * solution.g[local];
    }
  }
  std::vector<double> dual_residual(m);
  for (int j = 0; j < m; ++j)
    dual_residual[j] = transpose_h[j] - fixture_.d[j];
  solution.dres = norm2(dual_residual) / std::max(1.0, norm2(fixture_.d));
  const double orthogonality = inf_norm(transpose_g)
                               / std::max(1.0, inf_norm(solution.g));
  solution.piece_residual = std::max(
      {orthogonality, slope_error / slope_scale,
       constant_error / constant_scale});
  if (!std::isfinite(solution.dres) || !std::isfinite(solution.piece_residual))
    throw FaceDecline("nonfinite piece residual");
  if (solution.piece_residual > 1e-7) {
    stats_.residual_ms += milliseconds_since(residual_start);
    const auto svd_start = Clock::now();
    auto fallback = dense_svd_face(fixture_, rows, target_shift_,
                                   factored_seed_needed_,
                                   recurrence_seed_needed_);
    stats_.dense_svd_ms += milliseconds_since(svd_start);
    ++stats_.dense_fallbacks;
    fallback.nnz_R = nnz_R;
    fallback.core_diagonal_ratio = core_ratio;
    return finish(std::move(fallback), &core, &permutation);
  }
  for (const auto *vector : {&solution.g, &solution.h, &solution.ua,
                             &solution.uc})
    if (!std::all_of(vector->begin(), vector->end(),
                     [](double value) { return std::isfinite(value); }))
      throw FaceDecline("nonfinite coefficient");
  stats_.residual_ms += milliseconds_since(residual_start);
  return finish(std::move(solution), &core, &permutation);
}

FaceSolution FaceSolver::solve_uncached(
    const std::vector<std::uint32_t> &rows) {
  const bool was_forced = force_numerical_;
  force_numerical_ = true;
  try {
    auto solution = solve(rows);
    force_numerical_ = was_forced;
    return solution;
  } catch (...) {
    force_numerical_ = was_forced;
    throw;
  }
}

double relative_inf_error(const std::vector<double> &actual,
                          const std::vector<double> &expected) {
  if (actual.size() != expected.size())
    return std::numeric_limits<double>::infinity();
  double difference = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i)
    difference = std::max(difference, std::abs(actual[i] - expected[i]));
  return difference / std::max(1.0, inf_norm(expected));
}

}  // namespace twalker
