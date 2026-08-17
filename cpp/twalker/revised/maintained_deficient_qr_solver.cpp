#include "maintained_deficient_qr_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <iostream>
#include <limits>
#include <numeric>

extern "C" {
void dtrtrs_(const char *, const char *, const char *, const int *,
             const int *, const double *, const int *, double *, const int *,
             int *);
void dtzrzf_(const int *, const int *, double *, const int *, double *,
             double *, const int *, int *);
void dormrz_(const char *, const char *, const int *, const int *,
             const int *, const int *, const double *, const int *,
             const double *, double *, const int *, double *, const int *,
             int *);
void dgeqrf_(const int *, const int *, double *, const int *, double *,
             double *, const int *, int *);
}

namespace twalker::revised {
namespace {
using Clock = std::chrono::steady_clock;
// This is only the local admission gate.  The walker applies the stronger
// horizon-scaled gate after the next event distance is known.
constexpr double kMaintainedSlopeTolerance = 2e-10;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
double inf_norm(const std::vector<double> &x) {
  double value = 0.0;
  for (double entry : x) value = std::max(value, std::abs(entry));
  return value;
}
double relative_error(const std::vector<double> &left,
                      const std::vector<double> &right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double difference = 0.0, scale = 1.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    difference = std::max(difference, std::abs(left[i] - right[i]));
    scale = std::max(scale, std::abs(right[i]));
  }
  return difference / scale;
}
}  // namespace

void MaintainedDeficientQrSolver::invalidate() {
  valid_ = cached_valid_ = cached_face_valid_ = false;
  orthonormal_coordinates_ = false;
  rows_.clear(); basis_columns_.clear(); permutation_.clear();
  R_.clear(); transform_.clear(); transform_rz_.clear();
  transform_tau_.clear(); cross_.clear();
  rank_ = updates_ = 0;
}

double MaintainedDeficientQrSolver::diagonal_ratio() const {
  double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
  for (int i = 0; i < rank_; ++i) {
    const double d = std::abs(R_[i + static_cast<std::size_t>(rank_) * i]);
    lo = std::min(lo, d); hi = std::max(hi, d);
  }
  return hi > 0.0 ? lo / hi : 0.0;
}

bool MaintainedDeficientQrSolver::seed(
    const std::vector<std::uint32_t> &rows, const FaceSolution &direct) {
  const auto started = Clock::now();
  invalidate();
  const int m = static_cast<int>(fixture_.m);
  const int r = static_cast<int>(direct.rank);
  if (rows.empty() || direct.rows != rows || r <= 0 || r > m)
    return false;
  rows_ = rows;
  rank_ = r;
  const bool factored_seed = direct.factored_seed_rank == r
      && direct.factored_qr_core.size() == static_cast<std::size_t>(r) * m
      && direct.factored_permutation.size() == fixture_.m;
  const bool orthogonal_seed = direct.svd_row_space.size()
      == static_cast<std::size_t>(r) * m;
  if (!factored_seed && !orthogonal_seed) return false;

  if (factored_seed) {
    permutation_ = direct.factored_permutation;
    basis_columns_.resize(r);
    for (int i = 0; i < r; ++i) {
      if (permutation_[i] < 0 || permutation_[i] >= m) return false;
      basis_columns_[i] = static_cast<std::uint32_t>(permutation_[i]);
    }
    R_.assign(static_cast<std::size_t>(r) * r, 0.0);
    for (int column = 0; column < r; ++column)
      for (int row = 0; row <= column; ++row)
        R_[row + static_cast<std::size_t>(r) * column] =
            direct.factored_qr_core[
                row + static_cast<std::size_t>(r) * column];
    transform_ = direct.factored_qr_core;
    const char upper = 'U', no = 'N', nonunit = 'N';
    int info = 0;
    dtrtrs_(&upper, &no, &nonunit, &r, &m,
            direct.factored_qr_core.data(), &r, transform_.data(), &r,
            &info);
    if (info != 0) return false;
  } else {
    // Dense SVD has already paid to reveal an orthonormal row space.  Use it
    // as T in A=C*T, where C=A*T'.  QR only the active-by-r coordinates C;
    // this preserves the singular values without forming normal equations.
    orthonormal_coordinates_ = true;
    permutation_.resize(m);
    std::iota(permutation_.begin(), permutation_.end(), 0);
    basis_columns_.resize(r);
    std::iota(basis_columns_.begin(), basis_columns_.end(), 0);
    transform_ = direct.svd_row_space;
    const int active = static_cast<int>(rows.size());
    std::vector<double> coordinates(
        static_cast<std::size_t>(active) * r, 0.0);
    for (int local = 0; local < active; ++local) {
      const auto fixture_row = rows[local];
      for (auto p = fixture_.indptr[fixture_row];
           p < fixture_.indptr[fixture_row + 1]; ++p) {
        const auto column = fixture_.indices[p];
        const double value = fixture_.values[p];
        for (int component = 0; component < r; ++component)
          coordinates[local + static_cast<std::size_t>(active) * component]
              += value * transform_[
                  component + static_cast<std::size_t>(r) * column];
      }
    }
    std::vector<double> tau(std::min(active, r));
    int info = 0, lwork = -1;
    double query = 0.0;
    dgeqrf_(&active, &r, coordinates.data(), &active, tau.data(), &query,
            &lwork, &info);
    if (info != 0 || !std::isfinite(query)) return false;
    lwork = std::max(1, static_cast<int>(std::ceil(query)));
    std::vector<double> work(lwork);
    dgeqrf_(&active, &r, coordinates.data(), &active, tau.data(),
            work.data(), &lwork, &info);
    if (info != 0) return false;
    R_.assign(static_cast<std::size_t>(r) * r, 0.0);
    for (int column = 0; column < r; ++column)
      for (int row = 0; row <= column; ++row)
        R_[row + static_cast<std::size_t>(r) * column] =
            coordinates[row + static_cast<std::size_t>(active) * column];
  }
  if (!refactor_transform_rz()) return false;
  for (int row = 0; row < r; ++row)
    if (R_[row + static_cast<std::size_t>(r) * row] < 0.0)
      for (int column = row; column < r; ++column)
        R_[row + static_cast<std::size_t>(r) * column] *= -1.0;
  cross_.assign(r, 0.0);
  for (auto row : rows_) {
    std::vector<double> coordinate;
    if (!row_coordinate(row, coordinate)) return false;
    for (int component = 0; component < r; ++component)
      cross_[component] += coordinate[component] * fixture_.b[row];
  }
  valid_ = diagonal_ratio() > 5e-15;
  if (!valid_) return false;
  RevisedSlopeSolution check;
  if (!form_solution(check)
      || std::max({relative_error(check.g, direct.g),
                   relative_error(check.ua, direct.ua),
                   relative_error(check.bua, direct.bua)}) > 2e-10) {
    invalidate();
    return false;
  }
  RevisedFaceSolution face_check;
  if (!form_face_solution(face_check, 1)
      || std::max({relative_error(face_check.g, direct.g),
                   relative_error(face_check.h, direct.h),
                   relative_error(face_check.ua, direct.ua),
                   relative_error(face_check.uc, direct.uc),
                   relative_error(face_check.bua, direct.bua),
                   relative_error(face_check.buc, direct.buc)}) > 2e-10) {
    invalidate();
    return false;
  }
  solution_cache_[rows] = check;
  face_cache_[rows] = face_check;
  ++stats_.seeds; stats_.seed_ms += ms(started);
  return true;
}

bool MaintainedDeficientQrSolver::refactor_transform_rz() {
  const int m = static_cast<int>(fixture_.m);
  if (rank_ <= 0
      || transform_.size() != static_cast<std::size_t>(rank_) * m)
    return false;
  transform_rz_ = transform_;
  transform_tau_.assign(rank_, 0.0);
  if (rank_ >= m) return true;
  int info = 0, lwork = -1;
  double query = 0.0;
  dtzrzf_(&rank_, &m, transform_rz_.data(), &rank_,
          transform_tau_.data(), &query, &lwork, &info);
  if (info != 0 || !std::isfinite(query)) return false;
  lwork = std::max(1, static_cast<int>(std::ceil(query)));
  std::vector<double> work(lwork);
  dtzrzf_(&rank_, &m, transform_rz_.data(), &rank_,
          transform_tau_.data(), work.data(), &lwork, &info);
  return info == 0;
}

bool MaintainedDeficientQrSolver::row_coordinate(
    std::uint32_t row, std::vector<double> &coordinate,
    std::vector<double> *residual, double *relative_residual) const {
  if (row >= fixture_.n || rank_ <= 0
      || transform_.size() != static_cast<std::size_t>(rank_) * fixture_.m
      || permutation_.size() != fixture_.m)
    return false;
  std::vector<double> permuted(fixture_.m, 0.0);
  std::vector<int> inverse(fixture_.m, -1);
  for (int position = 0; position < static_cast<int>(fixture_.m); ++position) {
    if (permutation_[position] < 0
        || permutation_[position] >= static_cast<std::int64_t>(fixture_.m))
      return false;
    inverse[permutation_[position]] = position;
  }
  for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
    const int position = inverse[fixture_.indices[p]];
    if (position < 0) return false;
    permuted[position] = fixture_.values[p];
  }
  coordinate.assign(rank_, 0.0);
  if (orthonormal_coordinates_) {
    for (int component = 0; component < rank_; ++component)
      for (int column = 0; column < static_cast<int>(fixture_.m); ++column)
        coordinate[component] +=
            transform_[component
                       + static_cast<std::size_t>(rank_) * column]
            * permuted[column];
  } else {
    std::copy(permuted.begin(), permuted.begin() + rank_,
              coordinate.begin());
  }
  if (!residual && !relative_residual) return true;
  std::vector<double> local_residual = permuted;
  double square = 0.0, residual_square = 0.0;
  for (int column = 0; column < static_cast<int>(fixture_.m); ++column) {
    double predicted = 0.0;
    for (int k = 0; k < rank_; ++k)
      predicted += coordinate[k]
          * transform_[k + static_cast<std::size_t>(rank_) * column];
    local_residual[column] -= predicted;
    square += permuted[column] * permuted[column];
    residual_square += local_residual[column] * local_residual[column];
  }
  const double relative = std::sqrt(residual_square / std::max(1.0, square));
  if (residual) *residual = std::move(local_residual);
  if (relative_residual) *relative_residual = relative;
  return std::isfinite(relative);
}

bool MaintainedDeficientQrSolver::entering_row_is_dependent(
    std::uint32_t row) {
  std::vector<double> coordinate;
  double relative = std::numeric_limits<double>::infinity();
  if (!row_coordinate(row, coordinate, nullptr, &relative)) return false;
  stats_.worst_representation_residual =
      std::max(stats_.worst_representation_residual, relative);
  return std::isfinite(relative) && relative <= 2e-10;
}

bool MaintainedDeficientQrSolver::update_row(std::uint32_t row, int sign) {
  std::vector<double> x;
  if (!row_coordinate(row, x)) return false;
  for (int j = 0; j < rank_; ++j)
    cross_[j] += sign * x[j] * fixture_.b[row];
  for (int k = 0; k < rank_; ++k) {
    if (x[k] == 0.0) continue;
    const auto index = k + static_cast<std::size_t>(rank_) * k;
    const double diagonal = R_[index];
    const long double radicand = static_cast<long double>(diagonal) * diagonal
        + sign * static_cast<long double>(x[k]) * x[k];
    if (!(diagonal > 0.0) || !(radicand > 0.0L)) return false;
    const double replacement = std::sqrt(static_cast<double>(radicand));
    const double c = replacement / diagonal, s = x[k] / diagonal;
    if (!(c > 0.0) || !std::isfinite(c) || !std::isfinite(s)) return false;
    R_[index] = replacement;
    for (int column = k + 1; column < rank_; ++column) {
      const auto target = k + static_cast<std::size_t>(rank_) * column;
      const double updated = (R_[target] + sign * s * x[column]) / c;
      x[column] = c * x[column] - s * updated;
      R_[target] = updated;
    }
  }
  return diagonal_ratio() > 1e-13;
}

bool MaintainedDeficientQrSolver::transition(
    const std::vector<std::uint32_t> &rows) {
  const auto started = Clock::now();
  if (!valid_ || !std::is_sorted(rows.begin(), rows.end())) return false;
  if (rows == rows_) { ++stats_.unchanged; return true; }
  std::vector<std::uint32_t> add, remove;
  std::set_difference(rows.begin(), rows.end(), rows_.begin(), rows_.end(),
                      std::back_inserter(add));
  std::set_difference(rows_.begin(), rows_.end(), rows.begin(), rows.end(),
                      std::back_inserter(remove));
  if (add.size() + remove.size() > 8 || updates_ >= 32) return false;
  bool needs_weak_exchange = false;
  for (auto row : add)
    needs_weak_exchange = !entering_row_is_dependent(row)
                          || needs_weak_exchange;
  if (needs_weak_exchange) {
    if (std::getenv("TWALKER_QUOTIENT_WEAK_EXCHANGE")
        && add.size() == remove.size() && add.size() <= 4
        && weak_exchange(rows, add, remove)) {
      ++stats_.transitions;
      stats_.additions += add.size(); stats_.removals += remove.size();
      stats_.transition_ms += ms(started);
      return true;
    }
    ++stats_.rank_change_declines;
    return false;
  }
  for (auto row : add) if (!update_row(row, +1)) return false;
  for (auto row : remove) if (!update_row(row, -1)) {
    ++stats_.rank_change_declines; return false;
  }
  rows_ = rows; updates_ += static_cast<int>(add.size() + remove.size());
  cached_valid_ = cached_face_valid_ = false; ++stats_.transitions;
  stats_.additions += add.size(); stats_.removals += remove.size();
  stats_.transition_ms += ms(started);
  return true;
}

bool MaintainedDeficientQrSolver::weak_exchange(
    const std::vector<std::uint32_t> &rows,
    const std::vector<std::uint32_t> &add,
    const std::vector<std::uint32_t> &remove) {
  const auto started = Clock::now();
  ++stats_.weak_exchange_attempts;
  auto decline = [&](const char *stage, double residual = 0.0) {
    ++stats_.weak_exchange_declines;
    if (std::getenv("TWALKER_QUOTIENT_WEAK_TRACE"))
      std::cerr << "quotient weak decline stage=" << stage
                << " k=" << add.size() << " residual=" << residual
                << " rows=" << rows.size() << '\n';
    return false;
  };
  const int m = static_cast<int>(fixture_.m);
  const int k = static_cast<int>(add.size());
  if (!valid_ || k <= 0 || k > 4 || add.size() != remove.size()
      || transform_.size() != static_cast<std::size_t>(rank_) * m) {
    return decline("shape");
  }

  // A_old=C_old*T.  The changed rows violate this relation only through the
  // k entering residuals E.  Update the final square factor first (add before
  // remove keeps the atomic exchange nonsingular), then solve
  //
  //   Z=(C_new'C_new)^-1 X_add',   T_new=T+Z*E.
  //
  // This is the exact rank-k quotient repair.  The unchanged rows certify
  // that Z*E lies in the lost weak coordinate space; no ambient face
  // factorization is reconstructed.
  std::vector<double> entering_x(static_cast<std::size_t>(k) * rank_, 0.0);
  std::vector<double> entering_e(static_cast<std::size_t>(k) * m, 0.0);
  for (int local = 0; local < k; ++local) {
    std::vector<double> coordinate, residual;
    if (!row_coordinate(add[local], coordinate, &residual, nullptr))
      return decline("coordinate");
    std::copy(coordinate.begin(), coordinate.end(),
              entering_x.begin()
                  + static_cast<std::size_t>(local) * rank_);
    std::copy(residual.begin(), residual.end(),
              entering_e.begin() + static_cast<std::size_t>(local) * m);
  }

  for (auto row : add)
    if (!update_row(row, +1)) {
      return decline("factor-add");
    }
  for (auto row : remove)
    if (!update_row(row, -1)) {
      return decline("factor-remove");
    }

  const char upper = 'U', trans = 'T', no = 'N', nonunit = 'N';
  int info = 0;
  std::vector<double> z = entering_x;  // k contiguous rank-vectors.
  for (int local = 0; local < k; ++local) {
    double *rhs = z.data() + static_cast<std::size_t>(local) * rank_;
    const int one = 1;
    dtrtrs_(&upper, &trans, &nonunit, &rank_, &one, R_.data(), &rank_,
            rhs, &rank_, &info);
    if (info != 0) {
      return decline("solve-transpose");
    }
    dtrtrs_(&upper, &no, &nonunit, &rank_, &one, R_.data(), &rank_,
            rhs, &rank_, &info);
    if (info != 0) {
      return decline("solve");
    }
  }

  std::vector<double> repaired = transform_;
  for (int column = 0; column < m; ++column)
    for (int component = 0; component < rank_; ++component) {
      long double correction = 0.0L;
      for (int local = 0; local < k; ++local)
        correction += static_cast<long double>(
            z[static_cast<std::size_t>(local) * rank_ + component])
            * entering_e[static_cast<std::size_t>(local) * m + column];
      repaired[component + static_cast<std::size_t>(rank_) * column] +=
          static_cast<double>(correction);
    }

  // Validate A_new=C_new*T_new against the original sparse rows.  This is a
  // rare repair (k<=4), so an O(nnz(A_new)*r) fail-closed audit is deliberate;
  // it is still far below a cold sparse rank reveal and core extraction.
  std::vector<int> permuted_position(m, -1);
  for (int position = 0; position < m; ++position)
    permuted_position[permutation_[position]] = position;
  double worst = 0.0;
  for (auto row : rows) {
    std::vector<double> x;
    if (!row_coordinate(row, x)) return decline("audit-coordinate");
    std::vector<double> original(m, 0.0);
    double scale = 1.0;
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const int position = permuted_position[fixture_.indices[p]];
      scale = std::max(scale, std::abs(fixture_.values[p]));
      original[position] = fixture_.values[p];
    }
    std::vector<double> predicted(m, 0.0);
    for (int column = 0; column < m; ++column)
      for (int component = 0; component < rank_; ++component)
        predicted[column] += x[component]
            * repaired[component + static_cast<std::size_t>(rank_) * column];
    for (int column = 0; column < m; ++column)
      predicted[column] -= original[column];
    worst = std::max(worst, inf_norm(predicted) / scale);
  }
  stats_.worst_weak_exchange_residual =
      std::max(stats_.worst_weak_exchange_residual, worst);
  if (!std::isfinite(worst) || worst > 2e-10) {
    return decline("representation", worst);
  }

  transform_ = std::move(repaired);
  orthonormal_coordinates_ = false;
  if (!refactor_transform_rz()) {
    return decline("rz");
  }
  rows_ = rows;
  updates_ += 2 * k;
  cached_valid_ = cached_face_valid_ = false;
  ++stats_.weak_exchange_accepts;
  stats_.weak_exchange_dimension += k;
  stats_.weak_exchange_ms += ms(started);
  if (std::getenv("TWALKER_QUOTIENT_WEAK_TRACE"))
    std::cerr << "quotient weak accept k=" << k
              << " residual=" << worst << " rows=" << rows.size()
              << " ms=" << ms(started) << '\n';
  return true;
}

bool MaintainedDeficientQrSolver::form_solution(
    RevisedSlopeSolution &solution, int refinement_steps) {
  const auto started = Clock::now();
  const char upper = 'U', trans = 'T', no = 'N', nonunit = 'N';
  const int one = 1, m = static_cast<int>(fixture_.m);
  int info = 0;
  auto solve_core = [&](std::vector<double> &rhs) {
    dtrtrs_(&upper, &trans, &nonunit, &rank_, &one, R_.data(), &rank_,
            rhs.data(), &rank_, &info);
    if (info != 0) return false;
    dtrtrs_(&upper, &no, &nonunit, &rank_, &one, R_.data(), &rank_,
            rhs.data(), &rank_, &info);
    return info == 0;
  };

  int rz_lwork = 1;
  std::vector<double> rz_work(1);
  if (rank_ < m) {
    const char side = 'L';
    const int tail = m - rank_, ldc = m, query_lwork = -1;
    double query = 0.0;
    std::vector<double> query_rhs(m, 0.0);
    dormrz_(&side, &trans, &m, &one, &rank_, &tail, transform_rz_.data(),
            &rank_, transform_tau_.data(), query_rhs.data(), &ldc,
            &query, &query_lwork, &info);
    if (info != 0 || !std::isfinite(query)) return false;
    rz_lwork = std::max(1, static_cast<int>(std::ceil(query)));
    rz_work.resize(rz_lwork);
  }
  auto solve_min_norm = [&](const std::vector<double> &coordinate,
                            std::vector<double> &coefficient) {
    if (orthonormal_coordinates_) {
      coefficient.assign(m, 0.0);
      for (int column = 0; column < m; ++column)
        for (int component = 0; component < rank_; ++component)
          coefficient[column] +=
              transform_[component
                         + static_cast<std::size_t>(rank_) * column]
              * coordinate[component];
      return true;
    }
    coefficient.assign(m, 0.0);
    std::copy(coordinate.begin(), coordinate.end(), coefficient.begin());
    dtrtrs_(&upper, &no, &nonunit, &rank_, &one, transform_rz_.data(),
            &rank_, coefficient.data(), &rank_, &info);
    if (info != 0) return false;
    if (rank_ < m) {
      const char side = 'L';
      const int tail = m - rank_, ldc = m;
      dormrz_(&side, &trans, &m, &one, &rank_, &tail,
              transform_rz_.data(), &rank_, transform_tau_.data(),
              coefficient.data(), &ldc, rz_work.data(), &rz_lwork, &info);
      if (info != 0) return false;
    }
    return true;
  };
  auto reconstruct = [&](const std::vector<double> &coefficient,
                         std::vector<double> &transpose_g) {
    solution = RevisedSlopeSolution{};
    solution.rows = rows_; solution.rank = rank_;
    solution.ua.assign(fixture_.m, 0.0);
    for (int position = 0; position < m; ++position)
      solution.ua[permutation_[position]] = coefficient[position];
    solution.g.resize(rows_.size());
    for (std::size_t local = 0; local < rows_.size(); ++local) {
      solution.g[local] = fixture_.b[rows_[local]];
      for (auto p = fixture_.indptr[rows_[local]];
           p < fixture_.indptr[rows_[local] + 1]; ++p)
        solution.g[local] += fixture_.values[p]
                             * solution.ua[fixture_.indices[p]];
    }
    solution.bua.assign(fixture_.n, 0.0);
    for (std::size_t row = 0; row < fixture_.n; ++row)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        solution.bua[row] += fixture_.values[p]
                              * solution.ua[fixture_.indices[p]];
    transpose_g.assign(fixture_.m, 0.0);
    for (std::size_t local = 0; local < rows_.size(); ++local)
      for (auto p = fixture_.indptr[rows_[local]];
           p < fixture_.indptr[rows_[local] + 1]; ++p)
        transpose_g[fixture_.indices[p]] +=
            fixture_.values[p] * solution.g[local];
    solution.slope_residual =
        inf_norm(transpose_g) / std::max(1.0, inf_norm(solution.g));
  };

  // R'R w=-C'b gives the least-squares coordinates in the square core.
  std::vector<double> coordinate = cross_;
  for (double &value : coordinate) value = -value;
  if (!solve_core(coordinate)) return false;
  std::vector<double> coefficient, transpose_g;
  if (!solve_min_norm(coordinate, coefficient)) return false;
  reconstruct(coefficient, transpose_g);

  double previous_residual = solution.slope_residual;
  for (int step = 0; step < refinement_steps; ++step) {
    // C'g is available as the pivot-coordinate entries of the original
    // sparse residual A'g.  In an SVD-seeded epoch C=A*T', so C'g=T*A'g;
    // in a pivot-column epoch it is the selected entries.  Use the retained
    // square factor only as the correction solver; acceptance is still
    // measured with the original A.
    std::vector<double> correction(rank_);
    if (orthonormal_coordinates_) {
      for (int component = 0; component < rank_; ++component) {
        long double value = 0.0L;
        for (int column = 0; column < m; ++column)
          value += static_cast<long double>(
                       transform_[component
                                  + static_cast<std::size_t>(rank_) * column])
                   * transpose_g[permutation_[column]];
        correction[component] = -static_cast<double>(value);
      }
    } else {
      for (int j = 0; j < rank_; ++j)
        correction[j] = -transpose_g[basis_columns_[j]];
    }
    if (!solve_core(correction)) return false;
    for (int j = 0; j < rank_; ++j) coordinate[j] += correction[j];
    if (!solve_min_norm(coordinate, coefficient)) return false;
    reconstruct(coefficient, transpose_g);
    ++stats_.refinement_steps;
    if (!(solution.slope_residual < previous_residual)) break;
    // A second correction is worthwhile only when the first one materially
    // improves the original sparse residual.  Stale factors such as Capri's
    // otherwise spend another O(r^2+rm) pass for rounding-level progress.
    if (solution.slope_residual > 0.75 * previous_residual) break;
    previous_residual = solution.slope_residual;
  }
  stats_.worst_slope_residual =
      std::max(stats_.worst_slope_residual, solution.slope_residual);
  stats_.solve_ms += ms(started);
  if (!std::isfinite(solution.slope_residual)
      || solution.slope_residual > kMaintainedSlopeTolerance)
    return false;
  ++stats_.solves;
  return true;
}

bool MaintainedDeficientQrSolver::refine(
    const std::vector<std::uint32_t> &rows,
    RevisedSlopeSolution &solution, int max_steps) {
  if (!valid_ || rows != rows_ || max_steps <= 0) return false;
  ++stats_.refinement_attempts;
  if (!form_solution(solution, std::min(max_steps, 2))) return false;
  cached_ = solution;
  cached_valid_ = true;
  solution_cache_[rows] = solution;
  return true;
}

bool MaintainedDeficientQrSolver::form_face_solution(
    RevisedFaceSolution &solution, int refinement_steps) {
  RevisedSlopeSolution slope;
  if (!form_solution(slope, std::max(0, refinement_steps))) return false;
  const int m = static_cast<int>(fixture_.m);
  const char upper = 'U', trans = 'T', no = 'N', nonunit = 'N';
  const int one = 1;
  int info = 0;
  auto solve_core = [&](std::vector<double> &rhs) {
    dtrtrs_(&upper, &trans, &nonunit, &rank_, &one, R_.data(), &rank_,
            rhs.data(), &rank_, &info);
    if (info != 0) return false;
    dtrtrs_(&upper, &no, &nonunit, &rank_, &one, R_.data(), &rank_,
            rhs.data(), &rank_, &info);
    return info == 0;
  };

  int rz_lwork = 1;
  std::vector<double> rz_work(1);
  if (rank_ < m) {
    const char side = 'L';
    const int tail = m - rank_, ldc = m, query_lwork = -1;
    double query = 0.0;
    std::vector<double> query_rhs(m, 0.0);
    dormrz_(&side, &no, &m, &one, &rank_, &tail, transform_rz_.data(),
            &rank_, transform_tau_.data(), query_rhs.data(), &ldc, &query,
            &query_lwork, &info);
    if (info != 0 || !std::isfinite(query)) return false;
    rz_lwork = std::max(1, static_cast<int>(std::ceil(query)));
    rz_work.resize(rz_lwork);
  }
  auto solve_min_norm = [&](const std::vector<double> &coordinate,
                            std::vector<double> &coefficient) {
    if (orthonormal_coordinates_) {
      coefficient.assign(m, 0.0);
      for (int column = 0; column < m; ++column)
        for (int component = 0; component < rank_; ++component)
          coefficient[column] +=
              transform_[component
                         + static_cast<std::size_t>(rank_) * column]
              * coordinate[component];
      return true;
    }
    coefficient.assign(m, 0.0);
    std::copy(coordinate.begin(), coordinate.end(), coefficient.begin());
    dtrtrs_(&upper, &no, &nonunit, &rank_, &one, transform_rz_.data(),
            &rank_, coefficient.data(), &rank_, &info);
    if (info != 0) return false;
    if (rank_ < m) {
      const char side = 'L';
      const int tail = m - rank_, ldc = m;
      dormrz_(&side, &trans, &m, &one, &rank_, &tail,
              transform_rz_.data(), &rank_, transform_tau_.data(),
              coefficient.data(), &ldc, rz_work.data(), &rz_lwork, &info);
      if (info != 0) return false;
    }
    return true;
  };
  auto normal_solution = [&](const std::vector<double> &rhs,
                             std::vector<double> &coefficient) {
    if (rhs.size() != fixture_.m) return false;
    std::vector<double> transformed(m);
    for (int position = 0; position < m; ++position)
      transformed[position] = rhs[permutation_[position]];
    if (orthonormal_coordinates_) {
      std::vector<double> target(rank_, 0.0);
      for (int component = 0; component < rank_; ++component)
        for (int column = 0; column < m; ++column)
          target[component] +=
              transform_[component
                         + static_cast<std::size_t>(rank_) * column]
              * transformed[column];
      if (!solve_core(target)) return false;
      std::vector<double> permuted_coefficient;
      if (!solve_min_norm(target, permuted_coefficient)) return false;
      coefficient.assign(m, 0.0);
      for (int position = 0; position < m; ++position)
        coefficient[permutation_[position]] = permuted_coefficient[position];
      return true;
    }
    if (rank_ < m) {
      const char side = 'L';
      const int tail = m - rank_, ldc = m;
      dormrz_(&side, &no, &m, &one, &rank_, &tail,
              transform_rz_.data(), &rank_, transform_tau_.data(),
              transformed.data(), &ldc, rz_work.data(), &rz_lwork, &info);
      if (info != 0) return false;
    }
    std::vector<double> target(transformed.begin(),
                               transformed.begin() + rank_);
    dtrtrs_(&upper, &trans, &nonunit, &rank_, &one,
            transform_rz_.data(), &rank_, target.data(), &rank_, &info);
    if (info != 0 || !solve_core(target)) return false;
    std::vector<double> permuted_coefficient;
    if (!solve_min_norm(target, permuted_coefficient)) return false;
    coefficient.assign(m, 0.0);
    for (int position = 0; position < m; ++position)
      coefficient[permutation_[position]] = permuted_coefficient[position];
    return true;
  };

  std::vector<double> adjusted_d = fixture_.d;
  if (!target_shift_.empty())
    for (auto row : rows_)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        adjusted_d[fixture_.indices[p]] -=
            fixture_.values[p] * target_shift_[row];
  std::vector<double> uc;
  if (!normal_solution(adjusted_d, uc)) return false;

  auto product = [&](const std::vector<double> &coefficient,
                     std::vector<double> &values) {
    values.assign(fixture_.n, 0.0);
    for (std::size_t row = 0; row < fixture_.n; ++row)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        values[row] += fixture_.values[p]
                       * coefficient[fixture_.indices[p]];
  };
  std::vector<double> buc;
  product(uc, buc);
  for (int step = 0; step < std::max(0, refinement_steps); ++step) {
    std::vector<double> residual = adjusted_d;
    for (auto row : rows_)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        residual[fixture_.indices[p]] -= fixture_.values[p] * buc[row];
    if (inf_norm(residual) <= 2e-14 * std::max(1.0, inf_norm(adjusted_d)))
      break;
    std::vector<double> correction;
    if (!normal_solution(residual, correction)) return false;
    for (int column = 0; column < m; ++column) uc[column] += correction[column];
    product(uc, buc);
  }

  solution = RevisedFaceSolution{};
  solution.rows = rows_;
  solution.rank = rank_;
  solution.g = std::move(slope.g);
  solution.ua = std::move(slope.ua);
  solution.bua = std::move(slope.bua);
  solution.uc = std::move(uc);
  solution.buc = std::move(buc);
  solution.h.resize(rows_.size());
  std::vector<double> dual_residual = fixture_.d;
  for (double &value : dual_residual) value = -value;
  for (std::size_t local = 0; local < rows_.size(); ++local) {
    const auto row = rows_[local];
    const double shift = target_shift_.empty() ? 0.0 : target_shift_[row];
    solution.h[local] = shift + solution.buc[row];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      dual_residual[fixture_.indices[p]] +=
          fixture_.values[p] * solution.h[local];
  }
  double dual_scale = std::max(1.0, inf_norm(fixture_.d));
  solution.dres = inf_norm(dual_residual) / dual_scale;
  solution.piece_residual = slope.slope_residual;
  solution.basis_diagonal_ratio = diagonal_ratio();
  solution.coordinate_diagonal_ratio = diagonal_ratio();
  return std::isfinite(solution.dres)
      && std::isfinite(solution.piece_residual)
      && solution.dres <= 2e-10 && solution.piece_residual <= 2e-10;
}

bool MaintainedDeficientQrSolver::solve_face(
    const std::vector<std::uint32_t> &rows, RevisedFaceSolution &solution) {
  ++stats_.calls;
  if (const auto found = face_cache_.find(rows); found != face_cache_.end()) {
    solution = found->second;
    ++stats_.unchanged;
    return true;
  }
  if (!valid_) return false;
  if (rows == rows_ && cached_face_valid_) {
    solution = cached_face_;
    return true;
  }
  if (!transition(rows) || !form_face_solution(solution, 1)) {
    ++stats_.numerical_declines;
    invalidate();
    return false;
  }
  cached_face_ = solution;
  cached_face_valid_ = true;
  if (face_cache_.size() < 2048) face_cache_[rows] = solution;
  return true;
}

bool MaintainedDeficientQrSolver::solve(
    const std::vector<std::uint32_t> &rows,
    RevisedSlopeSolution &solution) {
  ++stats_.calls;
  if (const auto found = solution_cache_.find(rows);
      found != solution_cache_.end()) {
    solution = found->second;
    ++stats_.unchanged;
    return true;
  }
  if (!valid_) return false;
  if (rows == rows_ && cached_valid_) {
    solution = cached_; return true;
  }
  if (!transition(rows) || !form_solution(solution)) {
    ++stats_.numerical_declines; invalidate(); return false;
  }
  cached_ = solution; cached_valid_ = true;
  if (solution_cache_.size() < 2048) solution_cache_[rows] = solution;
  return true;
}

}  // namespace twalker::revised
