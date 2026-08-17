#include "triangular_projection.hpp"
#include "face_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

extern "C" {
void dgeqp3_(const int *m, const int *n, double *a, const int *lda, int *jpvt,
             double *tau, double *work, const int *lwork, int *info);
void dtrtrs_(const char *uplo, const char *trans, const char *diag,
             const int *n, const int *nrhs, const double *a, const int *lda,
             double *b, const int *ldb, int *info);
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

class ActiveProjectorFactor {
 public:
  struct Candidate {
    bool valid = false;
    bool dependent = false;
    int row = -1;
    double diagonal = 0.0;
    double schur = 0.0;
    std::vector<double> r_column;
    std::vector<double> gamma;
  };

  using Column = std::function<bool(int, std::vector<double> &)>;

  ActiveProjectorFactor(int maximum, Column column,
                        TriangularProjectionStats &stats)
      : maximum_(maximum), column_(std::move(column)), stats_(stats),
        triangular_(static_cast<std::size_t>(std::max(0, maximum))
                    * std::max(0, maximum), 0.0) {}

  int size() const { return static_cast<int>(indices_.size()); }
  const std::vector<int> &indices() const { return indices_; }

  Candidate analyze(int row) {
    const auto started = Clock::now();
    Candidate candidate;
    candidate.row = row;
    std::vector<double> projected;
    if (!column_(row, projected) || row < 0
        || row >= static_cast<int>(projected.size())) {
      stats_.active_update_ms += milliseconds_since(started);
      return candidate;
    }
    const int active = size();
    candidate.diagonal = projected[row];
    candidate.r_column.assign(active, 0.0);
    candidate.gamma.assign(active, 0.0);
    for (int i = 0; i < active; ++i)
      candidate.r_column[i] = projected[indices_[i]];

    // R' r = P_Ai, then R gamma = r.  Consequently
    // P_AA gamma=P_Ai and the Schur complement is ||q_i-Q_A'gamma||^2.
    for (int i = 0; i < active; ++i) {
      for (int j = 0; j < i; ++j)
        candidate.r_column[i] -=
            triangular_[j + static_cast<std::size_t>(maximum_) * i]
            * candidate.r_column[j];
      const double diagonal =
          triangular_[i + static_cast<std::size_t>(maximum_) * i];
      if (!(std::abs(diagonal) > 1e-14)) {
        stats_.active_update_ms += milliseconds_since(started);
        return candidate;
      }
      candidate.r_column[i] /= diagonal;
    }
    candidate.gamma = candidate.r_column;
    for (int i = active - 1; i >= 0; --i) {
      for (int j = i + 1; j < active; ++j)
        candidate.gamma[i] -=
            triangular_[i + static_cast<std::size_t>(maximum_) * j]
            * candidate.gamma[j];
      const double diagonal =
          triangular_[i + static_cast<std::size_t>(maximum_) * i];
      if (!(std::abs(diagonal) > 1e-14)) {
        stats_.active_update_ms += milliseconds_since(started);
        return candidate;
      }
      candidate.gamma[i] /= diagonal;
    }
    long double norm2 = 0.0L;
    for (double value : candidate.r_column)
      norm2 += static_cast<long double>(value) * value;
    candidate.schur = candidate.diagonal - static_cast<double>(norm2);
    const double scale = std::max(1.0, std::abs(candidate.diagonal));
    candidate.dependent = candidate.schur <= 2e-11 * scale;
    candidate.valid = std::isfinite(candidate.schur)
                      && candidate.schur >= -2e-9 * scale;
    stats_.active_update_ms += milliseconds_since(started);
    return candidate;
  }

  bool insert(const Candidate &candidate) {
    const auto started = Clock::now();
    const int active = size();
    if (!candidate.valid || candidate.dependent || candidate.row < 0
        || active >= maximum_
        || static_cast<int>(candidate.r_column.size()) != active) {
      stats_.active_update_ms += milliseconds_since(started);
      return false;
    }
    for (int i = 0; i < active; ++i)
      triangular_[i + static_cast<std::size_t>(maximum_) * active] =
          candidate.r_column[i];
    triangular_[active + static_cast<std::size_t>(maximum_) * active] =
        std::sqrt(std::max(0.0, candidate.schur));
    indices_.push_back(candidate.row);
    ++stats_.insertions;
    stats_.active_update_ms += milliseconds_since(started);
    return true;
  }

  bool insert(int row) { return insert(analyze(row)); }

  bool erase(int position) {
    const auto started = Clock::now();
    int active = size();
    if (position < 0 || position >= active) return false;
    for (int column = position; column < active - 1; ++column)
      for (int row = 0; row < active; ++row)
        triangular_[row + static_cast<std::size_t>(maximum_) * column] =
            triangular_[row
                        + static_cast<std::size_t>(maximum_) * (column + 1)];
    for (int j = position; j < active - 1; ++j) {
      const double x =
          triangular_[j + static_cast<std::size_t>(maximum_) * j];
      const double y =
          triangular_[j + 1 + static_cast<std::size_t>(maximum_) * j];
      const double radius = std::hypot(x, y);
      const double cosine = radius > 0.0 ? x / radius : 1.0;
      const double sine = radius > 0.0 ? y / radius : 0.0;
      for (int column = j; column < active - 1; ++column) {
        const auto top = j + static_cast<std::size_t>(maximum_) * column;
        const auto bottom = j + 1
                            + static_cast<std::size_t>(maximum_) * column;
        const double a = triangular_[top];
        const double b = triangular_[bottom];
        triangular_[top] = cosine * a + sine * b;
        triangular_[bottom] = -sine * a + cosine * b;
      }
    }
    indices_.erase(indices_.begin() + position);
    --active;
    for (int row = 0; row < maximum_; ++row) {
      triangular_[row + static_cast<std::size_t>(maximum_) * active] = 0.0;
      triangular_[active + static_cast<std::size_t>(maximum_) * row] = 0.0;
    }
    ++stats_.deletions;
    stats_.active_update_ms += milliseconds_since(started);
    return true;
  }

  bool solve(const std::vector<double> &right,
             std::vector<double> &solution) {
    const auto started = Clock::now();
    const int active = size();
    solution.assign(active, 0.0);
    for (int i = 0; i < active; ++i) {
      double value = right[indices_[i]];
      for (int j = 0; j < i; ++j)
        value -= triangular_[j + static_cast<std::size_t>(maximum_) * i]
                 * solution[j];
      const double diagonal =
          triangular_[i + static_cast<std::size_t>(maximum_) * i];
      if (!(std::abs(diagonal) > 1e-14)) return false;
      solution[i] = value / diagonal;
    }
    for (int i = active - 1; i >= 0; --i) {
      for (int j = i + 1; j < active; ++j)
        solution[i] -=
            triangular_[i + static_cast<std::size_t>(maximum_) * j]
            * solution[j];
      const double diagonal =
          triangular_[i + static_cast<std::size_t>(maximum_) * i];
      solution[i] /= diagonal;
    }
    stats_.active_solve_ms += milliseconds_since(started);
    return true;
  }

 private:
  int maximum_ = 0;
  Column column_;
  TriangularProjectionStats &stats_;
  std::vector<int> indices_;
  std::vector<double> triangular_;
};

std::string fingerprint(const ActiveProjectorFactor &factor,
                        const std::vector<double> &multiplier) {
  std::ostringstream stream;
  for (int row : factor.indices()) {
    const double rounded = std::round(multiplier[row] * 1e13) / 1e13;
    stream << row << ':' << rounded << ';';
  }
  return stream.str();
}

}  // namespace

TriangularProjector::TriangularProjector(const Fixture &fixture)
    : fixture_(fixture) {}

bool TriangularProjector::ensure_factor() {
  if (factor_ready_) return true;
  factor_failure_.clear();
  const auto started = Clock::now();
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  if (m <= 0 || n < m) {
    factor_failure_ = "triangular lane requires n>=m and full column rank";
    return false;
  }
  std::vector<double> matrix(static_cast<std::size_t>(n) * m, 0.0);
  column_scale_.assign(m, 0.0);
  for (int row = 0; row < n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const int column = static_cast<int>(fixture_.indices[p]);
      matrix[row + static_cast<std::size_t>(n) * column] = fixture_.values[p];
      column_scale_[column] = std::hypot(column_scale_[column],
                                         fixture_.values[p]);
    }
  for (int column = 0; column < m; ++column) {
    if (!(column_scale_[column] > 0.0)) {
      factor_failure_ = "zero equality column";
      return false;
    }
    column_scale_[column] = 1.0 / column_scale_[column];
    for (int row = 0; row < n; ++row)
      matrix[row + static_cast<std::size_t>(n) * column] *=
          column_scale_[column];
  }

  permutation_.assign(m, 0);
  std::vector<double> tau(m, 0.0);
  int info = 0, lwork = -1;
  double query = 0.0;
  dgeqp3_(&n, &m, matrix.data(), &n, permutation_.data(), tau.data(), &query,
          &lwork, &info);
  if (info != 0 || !std::isfinite(query) || query < 1.0) {
    factor_failure_ = "pivoted QR workspace query failed";
    return false;
  }
  lwork = static_cast<int>(std::ceil(query));
  std::vector<double> work(lwork);
  dgeqp3_(&n, &m, matrix.data(), &n, permutation_.data(), tau.data(),
          work.data(), &lwork, &info);
  if (info != 0) {
    factor_failure_ = "pivoted QR factorization failed";
    return false;
  }
  for (int &column : permutation_) --column;
  triangular_.assign(static_cast<std::size_t>(m) * m, 0.0);
  double largest = 0.0;
  for (int column = 0; column < m; ++column)
    for (int row = 0; row <= column; ++row) {
      const double value = matrix[row + static_cast<std::size_t>(n) * column];
      triangular_[row + static_cast<std::size_t>(m) * column] = value;
      if (row == column) largest = std::max(largest, std::abs(value));
    }
  const double cutoff = largest * std::max(n, m) * 32.0
                        * std::numeric_limits<double>::epsilon();
  rank_ = 0;
  double smallest = std::numeric_limits<double>::infinity();
  for (int i = 0; i < m; ++i) {
    const double diagonal =
        std::abs(triangular_[i + static_cast<std::size_t>(m) * i]);
    if (diagonal > cutoff) ++rank_;
    smallest = std::min(smallest, diagonal);
  }
  factor_diagonal_ratio_ = largest > 0.0 ? smallest / largest : 0.0;
  if (rank_ != m) {
    factor_failure_ = "rank-deficient equality operator";
    return false;
  }
  nullity_ = n - m;
  factor_ms_ = milliseconds_since(started);
  factor_ready_ = true;
  return true;
}

bool TriangularProjector::solve_normal(
    const std::vector<double> &rhs, std::vector<double> &solution,
    TriangularProjectionStats &stats) const {
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  if (static_cast<int>(rhs.size()) != m) return false;
  const char upper = 'U', transpose = 'T', no_transpose = 'N', diagonal = 'N';
  const int one = 1;
  auto triangular_solve = [&](const std::vector<double> &right,
                              std::vector<double> &answer) {
    answer.assign(m, 0.0);
    for (int position = 0; position < m; ++position) {
      const int original = permutation_[position];
      answer[position] = column_scale_[original] * right[original];
    }
    int info = 0;
    dtrtrs_(&upper, &transpose, &diagonal, &m, &one, triangular_.data(), &m,
            answer.data(), &m, &info);
    if (info != 0) return false;
    dtrtrs_(&upper, &no_transpose, &diagonal, &m, &one,
            triangular_.data(), &m, answer.data(), &m, &info);
    if (info != 0) return false;
    std::vector<double> unpermuted(m, 0.0);
    for (int position = 0; position < m; ++position)
      unpermuted[permutation_[position]] = answer[position];
    for (int column = 0; column < m; ++column)
      answer[column] = column_scale_[column] * unpermuted[column];
    return true;
  };

  if (!triangular_solve(rhs, solution)) return false;
  ++stats.normal_solves;
  for (int refinement = 0; refinement < 2; ++refinement) {
    std::vector<double> product(n, 0.0), normal(m, 0.0), residual(m);
    for (int row = 0; row < n; ++row)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        product[row] += fixture_.values[p] * solution[fixture_.indices[p]];
    for (int row = 0; row < n; ++row)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        normal[fixture_.indices[p]] += fixture_.values[p] * product[row];
    double error = 0.0;
    for (int column = 0; column < m; ++column) {
      residual[column] = rhs[column] - normal[column];
      error = std::max(error, std::abs(residual[column])
                                / (1.0 + std::abs(rhs[column])
                                   + std::abs(normal[column])));
    }
    if (error <= 2e-13) break;
    std::vector<double> correction;
    if (!triangular_solve(residual, correction)) return false;
    for (int column = 0; column < m; ++column)
      solution[column] += correction[column];
    ++stats.normal_solves;
    ++stats.refinements;
  }
  return true;
}

bool TriangularProjector::apply_projector(
    const std::vector<double> &input, std::vector<double> &output,
    TriangularProjectionStats &stats) const {
  const auto started = Clock::now();
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  if (static_cast<int>(input.size()) != n) return false;
  std::vector<double> rhs(m, 0.0), coefficient;
  for (int row = 0; row < n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      rhs[fixture_.indices[p]] += fixture_.values[p] * input[row];
  if (!solve_normal(rhs, coefficient, stats)) return false;
  output = input;
  for (int row = 0; row < n; ++row)
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      output[row] -= fixture_.values[p] * coefficient[fixture_.indices[p]];
  stats.operator_ms += milliseconds_since(started);
  return true;
}

TriangularProjectionResult TriangularProjector::solve(
    double t, double target_sign, std::uint64_t max_events) {
  TriangularProjectionResult result;
  result.t = t;
  const auto total_started = Clock::now();
  if (!ensure_factor()) {
    result.status = factor_failure_;
    result.stats.total_ms = milliseconds_since(total_started);
    return result;
  }
  result.rank = rank_;
  result.nullity = nullity_;
  result.factor_diagonal_ratio = factor_diagonal_ratio_;
  result.stats.factor_ms = factor_ms_;
  const int n = static_cast<int>(fixture_.n);
  const int m = static_cast<int>(fixture_.m);
  const double scale = std::max(1.0, std::abs(t));

  std::vector<double> scaled_d(m), particular_coefficient;
  for (int column = 0; column < m; ++column)
    scaled_d[column] = fixture_.d[column] / scale;
  if (!solve_normal(scaled_d, particular_coefficient, result.stats)) {
    result.status = "PARTICULAR_SOLVE_FAILURE";
    result.stats.total_ms = milliseconds_since(total_started);
    return result;
  }
  std::vector<double> particular(n, 0.0), target(n), difference(n), projected;
  for (int row = 0; row < n; ++row) {
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      particular[row] += fixture_.values[p]
                         * particular_coefficient[fixture_.indices[p]];
    target[row] = target_sign * t * fixture_.b[row] / scale;
    difference[row] = target[row] - particular[row];
  }
  if (!apply_projector(difference, projected, result.stats)) {
    result.status = "BASE_PROJECTOR_FAILURE";
    result.stats.total_ms = milliseconds_since(total_started);
    return result;
  }
  std::vector<double> base(n), linear(n), multiplier(n, 0.0), y(n);
  for (int row = 0; row < n; ++row) {
    base[row] = particular[row] + projected[row];
    linear[row] = -base[row];
  }

  auto column = [&](int row, std::vector<double> &answer) {
    std::vector<double> unit(n, 0.0);
    unit[row] = 1.0;
    return apply_projector(unit, answer, result.stats);
  };
  ActiveProjectorFactor active(nullity_, column, result.stats);
  std::set<std::string> seen;
  constexpr double tolerance = 1e-10;
  constexpr double rank_tolerance = 1e-11;
  while (result.stats.events < max_events) {
    const auto pricing_started = Clock::now();
    std::vector<double> correction;
    if (!apply_projector(multiplier, correction, result.stats)) {
      result.status = "STATE_PROJECTOR_FAILURE";
      break;
    }
    for (int row = 0; row < n; ++row) y[row] = base[row] + correction[row];
    std::vector<std::uint8_t> is_active(n, 0);
    for (int row : active.indices()) is_active[row] = 1;
    int entering = -1;
    double largest = 0.0;
    for (int row = 0; row < n; ++row) {
      if (is_active[row]) continue;
      const double violation = -y[row]
          / (1.0 + std::abs(base[row]) + std::abs(correction[row]));
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

    auto candidate = active.analyze(entering);
    if (!candidate.valid) {
      result.status = "PROJECTOR_COLUMN_FAILURE";
      break;
    }
    if (candidate.dependent) {
      int leaving_position = -1;
      double minimum = std::numeric_limits<double>::infinity();
      for (int position = 0; position < active.size(); ++position) {
        const double direction = -candidate.gamma[position];
        if (!(direction < -rank_tolerance)) continue;
        const int row = active.indices()[position];
        const double ratio = multiplier[row] / (-direction);
        if (ratio < minimum
            || (ratio == minimum
                && (leaving_position < 0
                    || row < active.indices()[leaving_position]))) {
          minimum = ratio;
          leaving_position = position;
        }
      }
      if (leaving_position < 0 || !std::isfinite(minimum)) {
        result.status = "DEPENDENT_UNBOUNDED";
        break;
      }
      for (int position = 0; position < active.size(); ++position)
        multiplier[active.indices()[position]] +=
            minimum * (-candidate.gamma[position]);
      multiplier[entering] = minimum;
      const int leaving = active.indices()[leaving_position];
      multiplier[leaving] = 0.0;
      active.erase(leaving_position);
      if (!active.insert(entering)) {
        result.status = "EXCHANGE_RANK_FAILURE";
        break;
      }
      ++result.stats.dependent_exchanges;
    } else if (!active.insert(candidate)) {
      result.status = "INSERT_FAILURE";
      break;
    }

    while (true) {
      std::vector<double> trial;
      if (!active.solve(linear, trial)) {
        result.status = "ACTIVE_SOLVE_FAILURE";
        break;
      }
      bool positive = true;
      for (double value : trial)
        positive = positive
                   && value > tolerance * std::max(1.0, std::abs(value));
      if (positive) {
        std::fill(multiplier.begin(), multiplier.end(), 0.0);
        for (int position = 0; position < active.size(); ++position)
          multiplier[active.indices()[position]] = trial[position];
        break;
      }
      int leaving_position = -1;
      double minimum = std::numeric_limits<double>::infinity();
      for (int position = 0; position < active.size(); ++position) {
        if (trial[position]
            > tolerance * std::max(1.0, std::abs(trial[position])))
          continue;
        const int row = active.indices()[position];
        const double denominator = multiplier[row] - trial[position];
        const double ratio = denominator > rank_tolerance
                                 ? multiplier[row] / denominator : 0.0;
        if (ratio < minimum
            || (ratio == minimum
                && (leaving_position < 0
                    || row < active.indices()[leaving_position]))) {
          minimum = std::max(0.0, ratio);
          leaving_position = position;
        }
      }
      if (leaving_position < 0) {
        result.status = "NO_DUAL_BLOCKER";
        break;
      }
      for (int position = 0; position < active.size(); ++position) {
        const int row = active.indices()[position];
        multiplier[row] = std::max(
            0.0, multiplier[row]
                     + minimum * (trial[position] - multiplier[row]));
      }
      const int leaving = active.indices()[leaving_position];
      multiplier[leaving] = 0.0;
      active.erase(leaving_position);
      ++result.stats.drops;
      if (active.size() == 0) break;
    }
    if (!result.status.empty()) break;
    ++result.stats.events;
    const auto state = fingerprint(active, multiplier);
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
  std::vector<double> correction;
  if (!apply_projector(multiplier, correction, result.stats)) {
    result.status = "FINAL_PROJECTOR_FAILURE";
    result.stats.total_ms = milliseconds_since(total_started);
    return result;
  }
  result.y.resize(n);
  for (int row = 0; row < n; ++row)
    result.y[row] = scale * (base[row] + correction[row]);

  // At the seed origin, a degenerate positive candidate can differ from the
  // canonical support-face minimum-norm point by much more than its objective
  // error (Lotfi is the discriminator).  Pay for one direct linear-algebra
  // face solve only at the end.  This is not an optimization side call: it
  // solves B_W' y_W=d and reconstructs the bound multiplier -B*uc.
  if (t == 0.0) {
    const auto polish_started = Clock::now();
    std::vector<std::uint8_t> bound_active(n, 0);
    for (int row : active.indices()) bound_active[row] = 1;
    std::vector<std::uint32_t> rows;
    for (int row = 0; row < n; ++row)
      if (!bound_active[row]) rows.push_back(row);
    try {
      FaceSolver face_solver(fixture_, false);
      const auto face = face_solver.solve(rows);
      std::vector<double> polished(n, 0.0), polished_multiplier(n, 0.0);
      std::vector<std::uint8_t> on_face(n, 0);
      for (std::size_t local = 0; local < face.rows.size(); ++local) {
        polished[face.rows[local]] = face.h[local];
        on_face[face.rows[local]] = 1;
      }
      double feasibility = 0.0, dual_violation = 0.0;
      const double polish_scale = std::max(1.0, inf_norm(polished));
      for (int row = 0; row < n; ++row) {
        feasibility = std::max(feasibility,
                               std::max(0.0, -polished[row]) / polish_scale);
        if (!on_face[row]) {
          polished_multiplier[row] = -face.buc[row];
          dual_violation = std::max(
              dual_violation,
              std::max(0.0, -polished_multiplier[row])
                  / (1.0 + std::abs(face.buc[row])));
        }
      }
      if (feasibility <= 1e-9 && dual_violation <= 1e-9
          && face.dres <= 1e-9) {
        result.y = std::move(polished);
        multiplier = std::move(polished_multiplier);
        ++result.stats.face_polishes;
      }
    } catch (const FaceDecline &) {
      // The active-set answer remains subject to the full KKT certificate.
    }
    result.stats.face_polish_ms += milliseconds_since(polish_started);
  }

  // Late equality polishing is allowed only when the original sparse residual
  // requests it.  It never participates in event decisions.
  for (int refinement = 0; refinement < 2; ++refinement) {
    std::vector<double> residual = fixture_.d, absolute(m, 1.0);
    for (int row = 0; row < n; ++row)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
        residual[fixture_.indices[p]] -= fixture_.values[p] * result.y[row];
        absolute[fixture_.indices[p]] +=
            std::abs(fixture_.values[p] * result.y[row]);
      }
    double checked = 0.0;
    for (int column_index = 0; column_index < m; ++column_index)
      checked = std::max(
          checked, std::abs(residual[column_index])
                       / (absolute[column_index]
                          + std::abs(fixture_.d[column_index])));
    if (checked <= 1e-12) break;
    std::vector<double> coefficient;
    if (!solve_normal(residual, coefficient, result.stats)) break;
    for (int row = 0; row < n; ++row)
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        result.y[row] += fixture_.values[p]
                         * coefficient[fixture_.indices[p]];
  }

  std::vector<double> scaled_y(n), equality = fixture_.d,
      equality_absolute(m, 1.0);
  for (int row = 0; row < n; ++row) {
    scaled_y[row] = result.y[row] / scale;
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      equality[fixture_.indices[p]] -= fixture_.values[p] * result.y[row];
      equality_absolute[fixture_.indices[p]] +=
          std::abs(fixture_.values[p] * result.y[row]);
    }
  }
  for (int column_index = 0; column_index < m; ++column_index)
    result.equality_error = std::max(
        result.equality_error,
        std::abs(equality[column_index])
            / (equality_absolute[column_index]
               + std::abs(fixture_.d[column_index])));
  const double y_scale = std::max(1.0, inf_norm(scaled_y));
  for (double value : scaled_y)
    result.nonnegative_error = std::max(
        result.nonnegative_error, std::max(0.0, -value) / y_scale);
  std::vector<double> stationarity(n), projected_stationarity;
  for (int row = 0; row < n; ++row)
    stationarity[row] = scaled_y[row] - target[row] - multiplier[row];
  if (!apply_projector(stationarity, projected_stationarity, result.stats)) {
    result.status = "STATIONARITY_PROJECTOR_FAILURE";
    result.stats.total_ms = milliseconds_since(total_started);
    return result;
  }
  result.stationarity_error = inf_norm(projected_stationarity)
                              / (1.0 + inf_norm(stationarity));
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
    result.support += value > 1e-9 * y_scale;
  result.active_core = active.size();
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
