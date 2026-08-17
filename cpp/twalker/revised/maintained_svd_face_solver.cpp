#include "maintained_svd_face_solver.hpp"

#include <vecLib/cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>

extern "C" {
void dgesdd_(const char *jobz, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt,
             const int *ldvt, double *work, const int *lwork, int *iwork,
             int *info);
void dgeqrf_(const int *m, const int *n, double *a, const int *lda,
             double *tau, double *work, const int *lwork, int *info);
void dorgqr_(const int *m, const int *n, const int *k, double *a,
             const int *lda, const double *tau, double *work,
             const int *lwork, int *info);
}

namespace twalker::revised {
namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double inf_norm(const std::vector<double> &x) {
  double value = 0.0;
  for (double entry : x) value = std::max(value, std::abs(entry));
  return value;
}

double norm2(const std::vector<double> &x) {
  long double square = 0.0L;
  for (double entry : x)
    square += static_cast<long double>(entry) * entry;
  return std::sqrt(static_cast<double>(square));
}

int workspace_size(double query) {
  return std::max(1, static_cast<int>(std::ceil(query)));
}

bool thin_svd(int rows, int columns, std::vector<double> matrix,
              std::vector<double> &left, std::vector<double> &singular,
              std::vector<double> &right_transpose) {
  const int thin = std::min(rows, columns);
  if (rows <= 0 || columns <= 0 || thin <= 0) return false;
  left.resize(static_cast<std::size_t>(rows) * thin);
  singular.resize(thin);
  right_transpose.resize(static_cast<std::size_t>(thin) * columns);
  std::vector<int> iwork(8 * thin);
  const char job = 'S';
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &rows, &columns, matrix.data(), &rows, singular.data(),
          left.data(), &rows, right_transpose.data(), &thin, &query, &lwork,
          iwork.data(), &info);
  if (info != 0 || !std::isfinite(query)) return false;
  lwork = workspace_size(query);
  std::vector<double> work(lwork);
  dgesdd_(&job, &rows, &columns, matrix.data(), &rows, singular.data(),
          left.data(), &rows, right_transpose.data(), &thin, work.data(),
          &lwork, iwork.data(), &info);
  return info == 0;
}

bool numerical_rank_unchanged(const std::vector<double> &singular,
                              int ambient) {
  if (singular.empty() || !(singular.front() > 0.0)) return false;
  const double cutoff = singular.front() * ambient
                        * std::numeric_limits<double>::epsilon();
  return std::isfinite(singular.back()) && singular.back() > cutoff;
}

}  // namespace

MaintainedSvdFaceSolver::MaintainedSvdFaceSolver(
    const Fixture &fixture, std::vector<double> target_shift)
    : fixture_(fixture), target_shift_(std::move(target_shift)) {
  if (std::all_of(target_shift_.begin(), target_shift_.end(),
                  [](double value) { return value == 0.0; }))
    target_shift_.clear();
}

void MaintainedSvdFaceSolver::invalidate() {
  valid_ = false;
  rows_.clear();
  U_.clear();
  singular_.clear();
  Vt_.clear();
}

bool MaintainedSvdFaceSolver::seed(
    const std::vector<std::uint32_t> &rows, const FaceSolution &direct) {
  const auto start = Clock::now();
  invalidate();
  const auto active = rows.size();
  const auto rank = static_cast<std::size_t>(direct.rank);
  const auto m = fixture_.m;
  if (rows.empty() || direct.rows != rows || rank == 0 || rank > m
      || direct.svd_left_space.size() != active * rank
      || direct.svd_singular_values.size() != rank
      || direct.svd_row_space.size() != rank * m)
    return false;
  rows_ = rows;
  U_ = direct.svd_left_space;
  singular_ = direct.svd_singular_values;
  Vt_ = direct.svd_row_space;
  valid_ = numerical_rank_unchanged(
      singular_, std::max({static_cast<int>(active), static_cast<int>(m), 1}));
  if (!valid_) return false;
  RevisedFaceSolution check;
  if (!form_solution(check)) {
    invalidate();
    return false;
  }
  ++stats_.seeds;
  stats_.seed_ms += ms(start);
  return true;
}

bool MaintainedSvdFaceSolver::add_row(std::uint32_t row) {
  const int active = static_cast<int>(rows_.size());
  const int rank = static_cast<int>(singular_.size());
  const int m = static_cast<int>(fixture_.m);
  std::vector<double> dense(m, 0.0);
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
    dense[fixture_.indices[p]] = fixture_.values[p];

  std::vector<double> coordinate(rank, 0.0), reconstructed(m, 0.0);
  cblas_dgemv(CblasColMajor, CblasNoTrans, rank, m, 1.0, Vt_.data(),
              rank, dense.data(), 1, 0.0, coordinate.data(), 1);
  cblas_dgemv(CblasColMajor, CblasTrans, rank, m, 1.0, Vt_.data(), rank,
              coordinate.data(), 1, 0.0, reconstructed.data(), 1);
  for (int column = 0; column < m; ++column)
    reconstructed[column] = dense[column] - reconstructed[column];
  const double relative = norm2(reconstructed) / std::max(1.0, norm2(dense));
  stats_.worst_representation_residual = std::max(
      stats_.worst_representation_residual, relative);
  if (!std::isfinite(relative) || relative > 2e-10) {
    ++stats_.rank_change_declines;
    return false;
  }

  std::vector<double> K(static_cast<std::size_t>(rank + 1) * rank, 0.0);
  for (int component = 0; component < rank; ++component) {
    K[component + static_cast<std::size_t>(rank + 1) * component] =
        singular_[component];
    K[rank + static_cast<std::size_t>(rank + 1) * component] =
        coordinate[component];
  }
  std::vector<double> left, new_singular, right;
  if (!thin_svd(rank + 1, rank, std::move(K), left, new_singular, right)
      || !numerical_rank_unchanged(new_singular,
          std::max({active + 1, m, 1}))) {
    ++stats_.numerical_declines;
    return false;
  }

  std::vector<double> transformed(static_cast<std::size_t>(active) * rank);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, active, rank, rank,
              1.0, U_.data(), active, left.data(), rank + 1, 0.0,
              transformed.data(), active);
  const auto insertion = std::lower_bound(rows_.begin(), rows_.end(), row);
  const int local = static_cast<int>(insertion - rows_.begin());
  std::vector<double> new_u(static_cast<std::size_t>(active + 1) * rank);
  for (int component = 0; component < rank; ++component) {
    for (int target = 0; target < active + 1; ++target) {
      if (target == local)
        new_u[target + static_cast<std::size_t>(active + 1) * component] =
            left[rank + static_cast<std::size_t>(rank + 1) * component];
      else {
        const int source = target < local ? target : target - 1;
        new_u[target + static_cast<std::size_t>(active + 1) * component] =
            transformed[source + static_cast<std::size_t>(active) * component];
      }
    }
  }
  std::vector<double> new_vt(static_cast<std::size_t>(rank) * m);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, m, rank, 1.0,
              right.data(), rank, Vt_.data(), rank, 0.0, new_vt.data(), rank);
  rows_.insert(insertion, row);
  U_ = std::move(new_u);
  singular_ = std::move(new_singular);
  Vt_ = std::move(new_vt);
  ++stats_.additions;
  return true;
}

bool MaintainedSvdFaceSolver::remove_row(std::uint32_t row) {
  const int active = static_cast<int>(rows_.size());
  const int rank = static_cast<int>(singular_.size());
  const int m = static_cast<int>(fixture_.m);
  const auto found = std::lower_bound(rows_.begin(), rows_.end(), row);
  if (found == rows_.end() || *found != row || active - 1 < rank) {
    ++stats_.rank_change_declines;
    return false;
  }
  const int removed = static_cast<int>(found - rows_.begin());
  std::vector<double> q(static_cast<std::size_t>(active - 1) * rank);
  for (int component = 0; component < rank; ++component)
    for (int target = 0; target < active - 1; ++target) {
      const int source = target < removed ? target : target + 1;
      q[target + static_cast<std::size_t>(active - 1) * component] =
          U_[source + static_cast<std::size_t>(active) * component];
    }
  std::vector<double> tau(rank);
  int rows_q = active - 1, info = 0, lwork = -1;
  double query = 0.0;
  dgeqrf_(&rows_q, &rank, q.data(), &rows_q, tau.data(), &query, &lwork,
          &info);
  if (info != 0 || !std::isfinite(query)) return false;
  lwork = workspace_size(query);
  std::vector<double> work(lwork);
  dgeqrf_(&rows_q, &rank, q.data(), &rows_q, tau.data(), work.data(), &lwork,
          &info);
  if (info != 0) return false;
  std::vector<double> core(static_cast<std::size_t>(rank) * rank, 0.0);
  for (int column = 0; column < rank; ++column)
    for (int source = 0; source <= column; ++source)
      core[source + static_cast<std::size_t>(rank) * column] =
          q[source + static_cast<std::size_t>(rows_q) * column]
          * singular_[column];
  lwork = -1;
  dorgqr_(&rows_q, &rank, &rank, q.data(), &rows_q, tau.data(), &query,
          &lwork, &info);
  if (info != 0 || !std::isfinite(query)) return false;
  lwork = workspace_size(query);
  work.resize(lwork);
  dorgqr_(&rows_q, &rank, &rank, q.data(), &rows_q, tau.data(), work.data(),
          &lwork, &info);
  if (info != 0) return false;
  std::vector<double> left, new_singular, right;
  if (!thin_svd(rank, rank, std::move(core), left, new_singular, right)
      || !numerical_rank_unchanged(new_singular,
          std::max({rows_q, m, 1}))) {
    ++stats_.rank_change_declines;
    return false;
  }
  std::vector<double> new_u(static_cast<std::size_t>(rows_q) * rank);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows_q, rank, rank,
              1.0, q.data(), rows_q, left.data(), rank, 0.0, new_u.data(),
              rows_q);
  std::vector<double> new_vt(static_cast<std::size_t>(rank) * m);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, m, rank, 1.0,
              right.data(), rank, Vt_.data(), rank, 0.0, new_vt.data(), rank);
  rows_.erase(found);
  U_ = std::move(new_u);
  singular_ = std::move(new_singular);
  Vt_ = std::move(new_vt);
  ++stats_.removals;
  return true;
}

bool MaintainedSvdFaceSolver::transition(
    const std::vector<std::uint32_t> &rows) {
  const auto start = Clock::now();
  if (!valid_ || !std::is_sorted(rows.begin(), rows.end())) return false;
  if (rows == rows_) return true;
  std::vector<std::uint32_t> additions, removals;
  std::set_difference(rows.begin(), rows.end(), rows_.begin(), rows_.end(),
                      std::back_inserter(additions));
  std::set_difference(rows_.begin(), rows_.end(), rows.begin(), rows.end(),
                      std::back_inserter(removals));
  if (additions.size() + removals.size() > 2) return false;
  const auto saved_rows = rows_;
  const auto saved_u = U_;
  const auto saved_s = singular_;
  const auto saved_vt = Vt_;
  for (auto row : additions)
    if (!add_row(row)) {
      rows_ = saved_rows; U_ = saved_u; singular_ = saved_s; Vt_ = saved_vt;
      return false;
    }
  for (auto row : removals)
    if (!remove_row(row)) {
      rows_ = saved_rows; U_ = saved_u; singular_ = saved_s; Vt_ = saved_vt;
      return false;
    }
  if (rows_ != rows) {
    rows_ = saved_rows; U_ = saved_u; singular_ = saved_s; Vt_ = saved_vt;
    return false;
  }
  ++stats_.transitions;
  stats_.transition_ms += ms(start);
  return true;
}

bool MaintainedSvdFaceSolver::form_solution(RevisedFaceSolution &solution) {
  const auto start = Clock::now();
  const int active = static_cast<int>(rows_.size());
  const int rank = static_cast<int>(singular_.size());
  const int m = static_cast<int>(fixture_.m);
  if (!valid_ || active <= 0 || rank <= 0) return false;
  std::vector<double> b(active), adjusted_d = fixture_.d;
  for (int local = 0; local < active; ++local) {
    const auto row = rows_[local];
    b[local] = fixture_.b[row];
    if (!target_shift_.empty())
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        adjusted_d[fixture_.indices[p]] -=
            fixture_.values[p] * target_shift_[row];
  }
  std::vector<double> ub(rank, 0.0), vd(rank, 0.0);
  cblas_dgemv(CblasColMajor, CblasTrans, active, rank, 1.0, U_.data(),
              active, b.data(), 1, 0.0, ub.data(), 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, rank, m, 1.0, Vt_.data(), rank,
              adjusted_d.data(), 1, 0.0, vd.data(), 1);
  std::vector<double> slope_coordinate(rank), constant_coordinate(rank);
  for (int component = 0; component < rank; ++component) {
    slope_coordinate[component] = -ub[component] / singular_[component];
    constant_coordinate[component] =
        vd[component] / (singular_[component] * singular_[component]);
  }
  solution = RevisedFaceSolution{};
  solution.rows = rows_;
  solution.rank = rank;
  solution.ua.assign(m, 0.0);
  solution.uc.assign(m, 0.0);
  cblas_dgemv(CblasColMajor, CblasTrans, rank, m, 1.0, Vt_.data(), rank,
              slope_coordinate.data(), 1, 0.0, solution.ua.data(), 1);
  cblas_dgemv(CblasColMajor, CblasTrans, rank, m, 1.0, Vt_.data(), rank,
              constant_coordinate.data(), 1, 0.0, solution.uc.data(), 1);
  solution.g = b;
  cblas_dgemv(CblasColMajor, CblasNoTrans, active, rank, -1.0, U_.data(),
              active, ub.data(), 1, 1.0, solution.g.data(), 1);
  std::vector<double> projection(rank, 0.0);
  cblas_dgemv(CblasColMajor, CblasTrans, active, rank, 1.0, U_.data(),
              active, solution.g.data(), 1, 0.0, projection.data(), 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, active, rank, -1.0, U_.data(),
              active, projection.data(), 1, 1.0, solution.g.data(), 1);
  std::vector<double> h_coordinate(rank);
  for (int component = 0; component < rank; ++component)
    h_coordinate[component] = vd[component] / singular_[component];
  solution.h.assign(active, 0.0);
  cblas_dgemv(CblasColMajor, CblasNoTrans, active, rank, 1.0, U_.data(),
              active, h_coordinate.data(), 1, 0.0, solution.h.data(), 1);
  for (int local = 0; local < active; ++local)
    if (!target_shift_.empty()) solution.h[local] += target_shift_[rows_[local]];
  solution.bua.assign(fixture_.n, 0.0);
  solution.buc.assign(fixture_.n, 0.0);
  for (std::size_t row = 0; row < fixture_.n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      solution.bua[row] += fixture_.values[p] * solution.ua[column];
      solution.buc[row] += fixture_.values[p] * solution.uc[column];
    }
  std::vector<double> transpose_g(m, 0.0), dres(fixture_.d);
  for (int local = 0; local < active; ++local) {
    const auto row = rows_[local];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const auto column = fixture_.indices[p];
      transpose_g[column] += fixture_.values[p] * solution.g[local];
      dres[column] -= fixture_.values[p] * solution.h[local];
    }
  }
  solution.piece_residual = inf_norm(transpose_g)
                            / std::max(1.0, inf_norm(solution.g));
  solution.dres = norm2(dres) / std::max(1.0, norm2(fixture_.d));
  stats_.worst_piece_residual = std::max(
      stats_.worst_piece_residual, solution.piece_residual);
  stats_.solve_ms += ms(start);
  ++stats_.solves;
  return std::isfinite(solution.piece_residual) && std::isfinite(solution.dres)
      && solution.piece_residual <= 2e-10 && solution.dres <= 2e-10;
}

bool MaintainedSvdFaceSolver::solve(
    const std::vector<std::uint32_t> &rows, RevisedFaceSolution &solution) {
  ++stats_.calls;
  if (!valid_ || !transition(rows) || !form_solution(solution)) {
    ++stats_.numerical_declines;
    invalidate();
    return false;
  }
  return true;
}

}  // namespace twalker::revised
