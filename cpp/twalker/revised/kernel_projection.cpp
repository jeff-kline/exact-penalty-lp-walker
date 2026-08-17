#include "kernel_projection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

extern "C" {
void dgesdd_(const char *jobz, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt,
             const int *ldvt, double *work, const int *lwork, int *iwork,
             int *info);
void dgemv_(const char *trans, const int *m, const int *n,
            const double *alpha, const double *a, const int *lda,
            const double *x, const int *incx, const double *beta, double *y,
            const int *incy);
}

namespace twalker::revised {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds_since(const Clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double inf_norm(const std::vector<double> &values) {
  double result = 0.0;
  for (double value : values) result = std::max(result, std::abs(value));
  return result;
}

class ActiveQr {
 public:
  ActiveQr(const std::vector<double> &q, int rows, int dimension,
           bool maintained, KernelProjectionStats &stats)
      : q_(q), rows_(rows), dimension_(dimension), maintained_(maintained),
        stats_(stats), orthogonal_(static_cast<std::size_t>(dimension)
                                  * dimension, 0.0),
        triangular_(static_cast<std::size_t>(dimension) * dimension, 0.0) {}

  int size() const { return static_cast<int>(indices_.size()); }
  const std::vector<int> &indices() const { return indices_; }

  bool coefficients(int row, std::vector<double> &gamma) const {
    const int active = size();
    if (!active) return false;
    std::vector<double> product(active, 0.0), residual(dimension_);
    for (int component = 0; component < dimension_; ++component)
      residual[component] = q_[row + static_cast<std::size_t>(rows_) * component];
    for (int column = 0; column < active; ++column) {
      double value = 0.0;
      for (int component = 0; component < dimension_; ++component)
        value += orthogonal_[component
                             + static_cast<std::size_t>(dimension_) * column]
                 * residual[component];
      product[column] = value;
      for (int component = 0; component < dimension_; ++component)
        residual[component] -= value
            * orthogonal_[component
                          + static_cast<std::size_t>(dimension_) * column];
    }
    double residual2 = 0.0, row2 = 0.0;
    for (int component = 0; component < dimension_; ++component) {
      residual2 += residual[component] * residual[component];
      const double value =
          q_[row + static_cast<std::size_t>(rows_) * component];
      row2 += value * value;
    }
    if (std::sqrt(residual2) > 1e-11 * std::max(1.0, std::sqrt(row2)))
      return false;
    gamma = product;
    for (int i = active - 1; i >= 0; --i) {
      for (int j = i + 1; j < active; ++j)
        gamma[i] -= triangular_[i
                                + static_cast<std::size_t>(dimension_) * j]
                    * gamma[j];
      const double diagonal = triangular_[i
          + static_cast<std::size_t>(dimension_) * i];
      if (!(std::abs(diagonal) > 1e-14)) return false;
      gamma[i] /= diagonal;
    }
    return true;
  }

  bool insert(int row) {
    const auto started = Clock::now();
    const int active = size();
    if (active >= dimension_) return false;
    std::vector<double> residual(dimension_), product(active, 0.0);
    for (int component = 0; component < dimension_; ++component)
      residual[component] = q_[row + static_cast<std::size_t>(rows_) * component];
    // Two-pass modified Gram--Schmidt is the stable insertion update.  It is
    // O(qp), versus O(qp^2) for a cold factorization.
    for (int pass = 0; pass < 2; ++pass) {
      for (int column = 0; column < active; ++column) {
        double value = 0.0;
        for (int component = 0; component < dimension_; ++component)
          value += orthogonal_[component
                               + static_cast<std::size_t>(dimension_) * column]
                   * residual[component];
        product[column] += value;
        for (int component = 0; component < dimension_; ++component)
          residual[component] -= value
              * orthogonal_[component
                            + static_cast<std::size_t>(dimension_) * column];
      }
    }
    double residual2 = 0.0, row2 = 0.0;
    for (int component = 0; component < dimension_; ++component) {
      residual2 += residual[component] * residual[component];
      const double value =
          q_[row + static_cast<std::size_t>(rows_) * component];
      row2 += value * value;
    }
    const double diagonal = std::sqrt(residual2);
    if (!(diagonal > 1e-11 * std::max(1.0, std::sqrt(row2)))) {
      stats_.active_update_ms += milliseconds_since(started);
      return false;
    }
    for (int column = 0; column < active; ++column)
      triangular_[column
                  + static_cast<std::size_t>(dimension_) * active] =
          product[column];
    triangular_[active
                + static_cast<std::size_t>(dimension_) * active] = diagonal;
    for (int component = 0; component < dimension_; ++component)
      orthogonal_[component
                  + static_cast<std::size_t>(dimension_) * active] =
          residual[component] / diagonal;
    indices_.push_back(row);
    ++stats_.qr_insertions;
    stats_.active_update_ms += milliseconds_since(started);
    if (!maintained_) rebuild();
    return true;
  }

  bool erase(int position) {
    const auto started = Clock::now();
    int active = size();
    if (position < 0 || position >= active) return false;
    for (int column = position; column < active - 1; ++column)
      for (int row = 0; row < active; ++row)
        triangular_[row + static_cast<std::size_t>(dimension_) * column] =
            triangular_[row
                        + static_cast<std::size_t>(dimension_) * (column + 1)];
    for (int j = position; j < active - 1; ++j) {
      const double x = triangular_[j
          + static_cast<std::size_t>(dimension_) * j];
      const double y = triangular_[j + 1
          + static_cast<std::size_t>(dimension_) * j];
      const double radius = std::hypot(x, y);
      const double cosine = radius > 0.0 ? x / radius : 1.0;
      const double sine = radius > 0.0 ? y / radius : 0.0;
      for (int column = j; column < active - 1; ++column) {
        const auto top_index = j
            + static_cast<std::size_t>(dimension_) * column;
        const auto bottom_index = j + 1
            + static_cast<std::size_t>(dimension_) * column;
        const double top = triangular_[top_index];
        const double bottom = triangular_[bottom_index];
        triangular_[top_index] = cosine * top + sine * bottom;
        triangular_[bottom_index] = -sine * top + cosine * bottom;
      }
      for (int component = 0; component < dimension_; ++component) {
        const auto left_index = component
            + static_cast<std::size_t>(dimension_) * j;
        const auto right_index = component
            + static_cast<std::size_t>(dimension_) * (j + 1);
        const double left = orthogonal_[left_index];
        const double right = orthogonal_[right_index];
        orthogonal_[left_index] = cosine * left + sine * right;
        orthogonal_[right_index] = -sine * left + cosine * right;
      }
    }
    indices_.erase(indices_.begin() + position);
    --active;
    for (int row = 0; row < dimension_; ++row) {
      orthogonal_[row + static_cast<std::size_t>(dimension_) * active] = 0.0;
      triangular_[row + static_cast<std::size_t>(dimension_) * active] = 0.0;
      triangular_[active + static_cast<std::size_t>(dimension_) * row] = 0.0;
    }
    ++stats_.qr_deletions;
    stats_.active_update_ms += milliseconds_since(started);
    if (!maintained_) rebuild();
    return true;
  }

  bool solve(const std::vector<double> &c, std::vector<double> &solution) {
    const auto started = Clock::now();
    const int active = size();
    solution.assign(active, 0.0);
    if (!active) {
      stats_.active_solve_ms += milliseconds_since(started);
      return true;
    }
    // R' intermediate=c_P; R lambda=intermediate.
    for (int i = 0; i < active; ++i) {
      double value = c[indices_[i]];
      for (int j = 0; j < i; ++j)
        value -= triangular_[j
                             + static_cast<std::size_t>(dimension_) * i]
                 * solution[j];
      const double diagonal = triangular_[i
          + static_cast<std::size_t>(dimension_) * i];
      if (!(std::abs(diagonal) > 1e-14)) return false;
      solution[i] = value / diagonal;
    }
    for (int i = active - 1; i >= 0; --i) {
      for (int j = i + 1; j < active; ++j)
        solution[i] -= triangular_[i
                                   + static_cast<std::size_t>(dimension_) * j]
                       * solution[j];
      const double diagonal = triangular_[i
          + static_cast<std::size_t>(dimension_) * i];
      solution[i] /= diagonal;
    }
    stats_.active_solve_ms += milliseconds_since(started);
    return true;
  }

 private:
  const std::vector<double> &q_;
  int rows_ = 0;
  int dimension_ = 0;
  bool maintained_ = true;
  KernelProjectionStats &stats_;
  std::vector<int> indices_;
  std::vector<double> orthogonal_, triangular_;

  bool rebuild() {
    const auto rows = indices_;
    indices_.clear();
    std::fill(orthogonal_.begin(), orthogonal_.end(), 0.0);
    std::fill(triangular_.begin(), triangular_.end(), 0.0);
    const bool saved = maintained_;
    maintained_ = true;
    bool good = true;
    for (int row : rows) good = insert(row) && good;
    maintained_ = saved;
    ++stats_.qr_rebuilds;
    return good;
  }
};

std::string fingerprint(const ActiveQr &qr,
                        const std::vector<double> &multiplier) {
  std::ostringstream stream;
  for (int row : qr.indices()) {
    const double rounded = std::round(multiplier[row] * 1e13) / 1e13;
    stream << row << ':' << rounded << ';';
  }
  return stream.str();
}

}  // namespace

KernelProjector::KernelProjector(const Fixture &fixture) : fixture_(fixture) {}

bool KernelProjector::ensure_factor() {
  if (factor_ready_) return true;
  factor_failure_.clear();
  const auto started = Clock::now();
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  std::vector<double> matrix(static_cast<std::size_t>(m) * n, 0.0);
  for (int row = 0; row < n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      matrix[fixture_.indices[p] + static_cast<std::size_t>(m) * row] =
          fixture_.values[p];
  const int thin = std::min(m, n);
  singular_.assign(thin, 0.0);
  left_.assign(static_cast<std::size_t>(m) * m, 0.0);
  right_t_.assign(static_cast<std::size_t>(n) * n, 0.0);
  std::vector<int> iwork(8 * std::max(1, thin));
  const char job = 'A';
  const int lda = m, ldu = m, ldvt = n;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&job, &m, &n, matrix.data(), &lda, singular_.data(), left_.data(),
          &ldu, right_t_.data(), &ldvt, &query, &lwork, iwork.data(), &info);
  if (info != 0 || !std::isfinite(query) || query < 1.0) {
    factor_failure_ = "SVD workspace query failed";
    return false;
  }
  lwork = static_cast<int>(std::ceil(query));
  std::vector<double> work(lwork);
  dgesdd_(&job, &m, &n, matrix.data(), &lda, singular_.data(), left_.data(),
          &ldu, right_t_.data(), &ldvt, work.data(), &lwork, iwork.data(),
          &info);
  if (info != 0) {
    factor_failure_ = "SVD factorization failed";
    return false;
  }
  const double cutoff = thin && singular_.front() > 0.0
      ? singular_.front() * std::max(m, n)
            * std::numeric_limits<double>::epsilon()
      : 0.0;
  rank_ = 0;
  while (rank_ < thin && singular_[rank_] > cutoff) ++rank_;
  nullity_ = n - rank_;
  range_.assign(static_cast<std::size_t>(n) * rank_, 0.0);
  q_.assign(static_cast<std::size_t>(n) * nullity_, 0.0);
  for (int row = 0; row < n; ++row) {
    for (int component = 0; component < rank_; ++component)
      range_[row + static_cast<std::size_t>(n) * component] =
          right_t_[component + static_cast<std::size_t>(n) * row];
    for (int component = 0; component < nullity_; ++component)
      q_[row + static_cast<std::size_t>(n) * component] =
          right_t_[rank_ + component + static_cast<std::size_t>(n) * row];
  }
  particular_.assign(n, 0.0);
  for (int component = 0; component < rank_; ++component) {
    double coefficient = 0.0;
    for (int column = 0; column < m; ++column)
      coefficient += left_[column + static_cast<std::size_t>(m) * component]
                     * fixture_.d[column];
    coefficient /= singular_[component];
    for (int row = 0; row < n; ++row)
      particular_[row] +=
          range_[row + static_cast<std::size_t>(n) * component] * coefficient;
  }
  factor_ms_ = milliseconds_since(started);
  factor_ready_ = true;
  return true;
}

KernelProjectionResult KernelProjector::solve(
    double t, double target_sign, bool maintained, std::uint64_t max_events) {
  KernelProjectionResult result;
  result.t = t;
  const auto total_started = Clock::now();
  if (!ensure_factor()) {
    result.status = factor_failure_;
    return result;
  }
  result.rank = rank_;
  result.nullity = nullity_;
  result.stats.nullspace_ms = factor_ms_;
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  const double scale = std::max(1.0, std::abs(t));
  std::vector<double> scaled_particular(n), scaled_target(n), z0(nullity_, 0.0);
  for (int row = 0; row < n; ++row) {
    scaled_particular[row] = particular_[row] / scale;
    scaled_target[row] = target_sign * t * fixture_.b[row] / scale;
  }
  for (int component = 0; component < nullity_; ++component)
    for (int row = 0; row < n; ++row)
      z0[component] += q_[row + static_cast<std::size_t>(n) * component]
                       * (scaled_target[row] - scaled_particular[row]);

  std::vector<double> bound(n), linear(n), multiplier(n, 0.0), z(z0);
  std::vector<double> absolute_q(q_.size());
  std::transform(q_.begin(), q_.end(), absolute_q.begin(),
                 [](double value) { return std::abs(value); });
  std::vector<double> slack(n), absolute_product(n), absolute_z(nullity_);
  for (int row = 0; row < n; ++row) {
    bound[row] = -scaled_particular[row];
    double product = 0.0;
    for (int component = 0; component < nullity_; ++component)
      product += q_[row + static_cast<std::size_t>(n) * component]
                 * z0[component];
    linear[row] = bound[row] - product;
  }
  ActiveQr qr(q_, n, nullity_, maintained, result.stats);
  std::set<std::string> seen;
  constexpr double tolerance = 1e-10;
  constexpr double rank_tolerance = 1e-11;
  constexpr double one = 1.0;
  const int increment = 1;
  const char transpose = 'T', no_transpose = 'N';
  while (result.stats.events < max_events) {
    const auto state_started = Clock::now();
    z = z0;
    dgemv_(&transpose, &n, &nullity_, &one, q_.data(), &n,
           multiplier.data(), &increment, &one, z.data(), &increment);
    std::vector<std::uint8_t> active(n, 0);
    for (int row : qr.indices()) active[row] = 1;
    result.stats.state_ms += milliseconds_since(state_started);
    const auto pricing_started = Clock::now();
    const double zero = 0.0;
    dgemv_(&no_transpose, &n, &nullity_, &one, q_.data(), &n, z.data(),
           &increment, &zero, slack.data(), &increment);
    std::transform(z.begin(), z.end(), absolute_z.begin(),
                   [](double value) { return std::abs(value); });
    dgemv_(&no_transpose, &n, &nullity_, &one, absolute_q.data(), &n,
           absolute_z.data(), &increment, &zero, absolute_product.data(),
           &increment);
    int entering = -1;
    double largest = 0.0;
    for (int row = 0; row < n; ++row) {
      if (active[row]) continue;
      const double absolute =
          1.0 + std::abs(bound[row]) + absolute_product[row];
      const double violation = (bound[row] - slack[row]) / absolute;
      if (violation > tolerance
          && (entering < 0 || violation > largest
              + 32.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(largest))
              || (std::abs(violation - largest)
                      <= 32.0 * std::numeric_limits<double>::epsilon()
                             * std::max(1.0, std::abs(largest))
                  && row < entering))) {
        entering = row;
        largest = violation;
      }
    }
    result.stats.pricing_ms += milliseconds_since(pricing_started);
    if (entering < 0) {
      result.status = "PROJECTED";
      break;
    }

    std::vector<double> gamma;
    const bool dependent = qr.coefficients(entering, gamma);
    if (dependent) {
      int leaving_position = -1;
      double minimum = std::numeric_limits<double>::infinity();
      for (int position = 0; position < qr.size(); ++position) {
        const double direction = -gamma[position];
        if (!(direction < -rank_tolerance)) continue;
        const int row = qr.indices()[position];
        const double ratio = multiplier[row] / (-direction);
        if (ratio < minimum - 64.0 * std::numeric_limits<double>::epsilon()
                                  * std::max(1.0, std::abs(minimum))
            || (std::abs(ratio - minimum)
                    <= 64.0 * std::numeric_limits<double>::epsilon()
                           * std::max(1.0, std::abs(minimum))
                && (leaving_position < 0
                    || row < qr.indices()[leaving_position]))) {
          minimum = ratio;
          leaving_position = position;
        }
      }
      if (leaving_position < 0 || !std::isfinite(minimum)) {
        result.status = "DEPENDENT_UNBOUNDED";
        break;
      }
      for (int position = 0; position < qr.size(); ++position)
        multiplier[qr.indices()[position]] += minimum * (-gamma[position]);
      multiplier[entering] = minimum;
      const int leaving = qr.indices()[leaving_position];
      multiplier[leaving] = 0.0;
      qr.erase(leaving_position);
      if (!qr.insert(entering)) {
        result.status = "EXCHANGE_RANK_FAILURE";
        break;
      }
      ++result.stats.dependent_exchanges;
    } else if (!qr.insert(entering)) {
      result.status = "INSERT_FAILURE";
      break;
    }

    while (true) {
      std::vector<double> trial;
      if (!qr.solve(linear, trial)) {
        result.status = "ACTIVE_SOLVE_FAILURE";
        break;
      }
      bool positive = true;
      for (double value : trial)
        positive = positive
                   && value > tolerance * std::max(1.0, std::abs(value));
      if (positive) {
        std::fill(multiplier.begin(), multiplier.end(), 0.0);
        for (int position = 0; position < qr.size(); ++position)
          multiplier[qr.indices()[position]] = trial[position];
        break;
      }
      int leaving_position = -1;
      double minimum = std::numeric_limits<double>::infinity();
      for (int position = 0; position < qr.size(); ++position) {
        if (trial[position]
            > tolerance * std::max(1.0, std::abs(trial[position])))
          continue;
        const int row = qr.indices()[position];
        const double current = multiplier[row];
        const double denominator = current - trial[position];
        const double ratio = denominator > rank_tolerance
                                 ? current / denominator : 0.0;
        if (ratio < minimum - 64.0 * std::numeric_limits<double>::epsilon()
                                  * std::max(1.0, std::abs(minimum))
            || (std::abs(ratio - minimum)
                    <= 64.0 * std::numeric_limits<double>::epsilon()
                           * std::max(1.0, std::abs(minimum))
                && (leaving_position < 0
                    || row < qr.indices()[leaving_position]))) {
          minimum = std::max(0.0, ratio);
          leaving_position = position;
        }
      }
      if (leaving_position < 0) {
        result.status = "NO_DUAL_BLOCKER";
        break;
      }
      for (int position = 0; position < qr.size(); ++position) {
        const int row = qr.indices()[position];
        multiplier[row] = std::max(
            0.0, multiplier[row]
                     + minimum * (trial[position] - multiplier[row]));
      }
      const int leaving = qr.indices()[leaving_position];
      multiplier[leaving] = 0.0;
      qr.erase(leaving_position);
      ++result.stats.drops;
      if (qr.size() == 0) break;
    }
    if (!result.status.empty()) break;
    ++result.stats.events;
    const auto state = fingerprint(qr, multiplier);
    if (!seen.insert(state).second) {
      result.status = "CYCLE";
      break;
    }
  }
  if (result.status.empty()) result.status = "EVENT_LIMIT";
  if (result.status != "PROJECTED") {
    result.stats.total_ms = milliseconds_since(total_started);
    return result;
  }

  const auto certificate_started = Clock::now();
  z = z0;
  dgemv_(&transpose, &n, &nullity_, &one, q_.data(), &n,
         multiplier.data(), &increment, &one, z.data(), &increment);
  std::vector<double> scaled_y = scaled_particular;
  for (int row = 0; row < n; ++row)
    for (int component = 0; component < nullity_; ++component)
      scaled_y[row] += q_[row + static_cast<std::size_t>(n) * component]
                       * z[component];
  result.y.resize(n);
  for (int row = 0; row < n; ++row) result.y[row] = scale * scaled_y[row];

  for (int refinement = 0; refinement < 2; ++refinement) {
    std::vector<double> residual = fixture_.d, absolute(m, 1.0);
    for (int row = 0; row < n; ++row)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        residual[fixture_.indices[p]] -= fixture_.values[p] * result.y[row];
        absolute[fixture_.indices[p]] +=
            std::abs(fixture_.values[p] * result.y[row]);
      }
    double checked = 0.0;
    for (int column = 0; column < m; ++column)
      checked = std::max(checked, std::abs(residual[column])
                                     / (absolute[column]
                                        + std::abs(fixture_.d[column])));
    if (checked <= std::min(tolerance, 1e-12)) break;
    std::vector<double> coefficient(rank_, 0.0);
    for (int component = 0; component < rank_; ++component) {
      for (int column = 0; column < m; ++column)
        coefficient[component] +=
            left_[column + static_cast<std::size_t>(m) * component]
            * residual[column];
      coefficient[component] /= singular_[component];
    }
    for (int row = 0; row < n; ++row)
      for (int component = 0; component < rank_; ++component)
        result.y[row] +=
            range_[row + static_cast<std::size_t>(n) * component]
            * coefficient[component];
    ++result.stats.equality_refinements;
  }
  for (int row = 0; row < n; ++row) scaled_y[row] = result.y[row] / scale;

  std::vector<double> equality = fixture_.d, equality_absolute(m, 1.0);
  for (int row = 0; row < n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      equality[fixture_.indices[p]] -= fixture_.values[p] * result.y[row];
      equality_absolute[fixture_.indices[p]] +=
          std::abs(fixture_.values[p] * result.y[row]);
    }
  result.equality_error = 0.0;
  for (int column = 0; column < m; ++column)
    result.equality_error = std::max(
        result.equality_error,
        std::abs(equality[column])
            / (equality_absolute[column] + std::abs(fixture_.d[column])));
  const double yscale = std::max(1.0, inf_norm(scaled_y));
  result.nonnegative_error = 0.0;
  for (double value : scaled_y)
    result.nonnegative_error =
        std::max(result.nonnegative_error, std::max(0.0, -value) / yscale);
  result.stationarity_error = 0.0;
  for (int component = 0; component < nullity_; ++component) {
    double residual = 0.0, absolute = 1.0;
    for (int row = 0; row < n; ++row) {
      const double coefficient =
          q_[row + static_cast<std::size_t>(n) * component];
      residual += coefficient
                  * (scaled_y[row] - scaled_target[row] - multiplier[row]);
      absolute += std::abs(coefficient)
                  * (std::abs(scaled_y[row]) + std::abs(scaled_target[row])
                     + std::abs(multiplier[row]));
    }
    result.stationarity_error =
        std::max(result.stationarity_error, std::abs(residual) / absolute);
  }
  result.complementarity_error = 0.0;
  result.dual_error = 0.0;
  const double multiplier_scale = std::max(1.0, inf_norm(multiplier));
  for (int row = 0; row < n; ++row) {
    result.complementarity_error = std::max(
        result.complementarity_error,
        std::abs(multiplier[row] * scaled_y[row])
            / (1.0 + std::abs(multiplier[row])
                     * (1.0 + std::abs(scaled_y[row]))));
    result.dual_error = std::max(
        result.dual_error, std::max(0.0, -multiplier[row]) / multiplier_scale);
  }
  result.support = 0;
  for (double value : scaled_y)
    result.support += value > 1e-9 * yscale;
  result.active_core = qr.size();
  result.certified = std::max(
      {result.equality_error, result.nonnegative_error,
       result.stationarity_error, result.complementarity_error,
       result.dual_error}) <= 1e-8;
  result.status = result.certified ? "CERTIFIED" : "FALSE_KKT";
  result.stats.certificate_ms = milliseconds_since(certificate_started);
  result.stats.total_ms = milliseconds_since(total_started);
  return result;
}

}  // namespace twalker::revised
