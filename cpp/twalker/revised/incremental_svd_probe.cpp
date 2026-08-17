#include "fixture.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double microseconds_since(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start)
      .count();
}

int workspace_size(double query) {
  if (!std::isfinite(query) || query < 1.0)
    throw std::runtime_error("invalid dgesdd workspace query");
  return static_cast<int>(std::ceil(query));
}

void require_lapack(int info, const char *operation) {
  if (info != 0)
    throw std::runtime_error(std::string(operation) + " failed, info="
                             + std::to_string(info));
}

double relative_inf_error(const std::vector<double> &got,
                          const std::vector<double> &want) {
  if (got.size() != want.size()) return std::numeric_limits<double>::infinity();
  double error = 0.0, scale = 1.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    error = std::max(error, std::abs(got[i] - want[i]));
    scale = std::max(scale, std::abs(want[i]));
  }
  return error / scale;
}

std::string stem(const std::string &path) {
  const auto slash = path.find_last_of('/');
  const auto dot = path.find_last_of('.');
  const auto begin = slash == std::string::npos ? 0 : slash + 1;
  const auto end = dot == std::string::npos || dot < begin ? path.size() : dot;
  return path.substr(begin, end - begin);
}

struct FaceValues {
  std::vector<double> g, h, ua, uc;
};

// Research-only replay prototype.  It maintains the compact SVD of the
// n-by-m matrix whose active rows equal B[W,:] and whose inactive rows are
// zero.  A support insertion/deletion is exactly a rank-one matrix update
// e_i (+/- B_i), so rank changes are exposed by the tiny update core instead
// of rebuilding a factorization of the whole face.
class IncrementalSvd {
 public:
  explicit IncrementalSvd(const twalker::Fixture &fixture)
      : fixture_(fixture), active_(fixture.n, false) {}

  void seed(const std::vector<std::uint32_t> &rows) {
    const int face_rows = static_cast<int>(rows.size());
    const int m = static_cast<int>(fixture_.m);
    const int thin = std::min(face_rows, m);
    if (face_rows == 0 || thin == 0) throw std::runtime_error("empty seed");
    std::vector<double> matrix(static_cast<std::size_t>(face_rows) * m, 0.0);
    for (int local = 0; local < face_rows; ++local) {
      const auto row = rows[local];
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        matrix[local + static_cast<std::size_t>(face_rows)
                           * fixture_.indices[p]] = fixture_.values[p];
    }
    std::vector<double> singular, left, vt;
    svd(std::move(matrix), face_rows, m, singular, left, vt);
    const int keep = numerical_rank(singular, face_rows);
    if (keep == 0) throw std::runtime_error("zero-rank seed");
    singular_.assign(singular.begin(), singular.begin() + keep);
    U_.assign(static_cast<std::size_t>(fixture_.n) * keep, 0.0);
    V_.assign(static_cast<std::size_t>(fixture_.m) * keep, 0.0);
    for (int component = 0; component < keep; ++component) {
      for (int local = 0; local < face_rows; ++local)
        U_[rows[local] + static_cast<std::size_t>(fixture_.n) * component] =
            left[local + static_cast<std::size_t>(face_rows) * component];
      for (int column = 0; column < m; ++column)
        V_[column + static_cast<std::size_t>(m) * component] =
            vt[component + static_cast<std::size_t>(thin) * column];
    }
    rows_ = rows;
    active_count_ = rows.size();
    updates_since_seed_ = 0;
    std::fill(active_.begin(), active_.end(), false);
    for (auto row : rows_) active_[row] = true;
  }

  void transition(const std::vector<std::uint32_t> &rows) {
    std::vector<std::uint32_t> entering, leaving;
    std::set_difference(rows.begin(), rows.end(), rows_.begin(), rows_.end(),
                        std::back_inserter(entering));
    std::set_difference(rows_.begin(), rows_.end(), rows.begin(), rows.end(),
                        std::back_inserter(leaving));
    // Add first so exchanges do not pass through an unnecessarily deficient
    // intermediate face.  This matches the existing factor-update discipline.
    for (auto row : entering) update_row(row, +1.0);
    for (auto row : leaving) update_row(row, -1.0);
    rows_ = rows;
    std::fill(active_.begin(), active_.end(), false);
    for (auto row : rows_) active_[row] = true;
  }

  FaceValues face() const {
    const int n = static_cast<int>(fixture_.n);
    const int m = static_cast<int>(fixture_.m);
    const int rank = static_cast<int>(singular_.size());
    std::vector<double> left_b(rank, 0.0), right_d(rank, 0.0);
    std::vector<double> adjusted_d = fixture_.d;
    for (auto row : rows_) {
      for (int component = 0; component < rank; ++component)
        left_b[component] +=
            U_[row + static_cast<std::size_t>(n) * component]
            * fixture_.b[row];
    }
    for (int component = 0; component < rank; ++component)
      for (int column = 0; column < m; ++column)
        right_d[component] +=
            V_[column + static_cast<std::size_t>(m) * component]
            * adjusted_d[column];

    FaceValues result;
    result.ua.assign(m, 0.0);
    result.uc.assign(m, 0.0);
    for (int component = 0; component < rank; ++component) {
      const double inverse = 1.0 / singular_[component];
      const double ua_scale = -left_b[component] * inverse;
      const double uc_scale = right_d[component] * inverse * inverse;
      for (int column = 0; column < m; ++column) {
        const double value =
            V_[column + static_cast<std::size_t>(m) * component];
        result.ua[column] += value * ua_scale;
        result.uc[column] += value * uc_scale;
      }
    }
    result.g.resize(rows_.size());
    result.h.resize(rows_.size());
    for (std::size_t local = 0; local < rows_.size(); ++local) {
      const auto row = rows_[local];
      double projected_b = 0.0, projected_d = 0.0;
      for (int component = 0; component < rank; ++component) {
        const double left =
            U_[row + static_cast<std::size_t>(n) * component];
        projected_b += left * left_b[component];
        projected_d += left * right_d[component] / singular_[component];
      }
      result.g[local] = fixture_.b[row] - projected_b;
      result.h[local] = projected_d;
    }
    return result;
  }

  int rank() const { return static_cast<int>(singular_.size()); }
  double singular_ratio() const {
    return singular_.empty() ? 0.0 : singular_.back() / singular_.front();
  }

 private:
  int numerical_rank(const std::vector<double> &singular,
                     int active_rows) const {
    if (singular.empty() || singular.front() <= 0.0) return 0;
    // A sequence of tiny-core SVDs accumulates more orthogonality error than
    // a cold decomposition.  Scale the standard LAPACK rank threshold by the
    // journal length; this is audited against every stored direct face before
    // it can be considered for live use.
    const double journal_guard = std::max(1.0, 16.0 * updates_since_seed_);
    const double cutoff = singular.front()
                          * std::max(active_rows, static_cast<int>(fixture_.m))
                          * std::numeric_limits<double>::epsilon()
                          * journal_guard;
    int rank = 0;
    while (rank < static_cast<int>(singular.size())
           && singular[rank] > cutoff)
      ++rank;
    return rank;
  }

  static void svd(std::vector<double> matrix, int rows, int columns,
                  std::vector<double> &singular, std::vector<double> &left,
                  std::vector<double> &vt) {
    const int thin = std::min(rows, columns);
    singular.resize(thin);
    left.resize(static_cast<std::size_t>(rows) * thin);
    vt.resize(static_cast<std::size_t>(thin) * columns);
    std::vector<int> iwork(std::max(1, 8 * thin));
    char job = 'S';
    int lda = rows, ldu = rows, ldvt = thin;
    int info = 0, lwork = -1;
    double query = 0.0;
    dgesdd_(&job, &rows, &columns, matrix.data(), &lda, singular.data(),
            left.data(), &ldu, vt.data(), &ldvt, &query, &lwork, iwork.data(),
            &info);
    require_lapack(info, "dgesdd query");
    lwork = workspace_size(query);
    std::vector<double> work(lwork);
    dgesdd_(&job, &rows, &columns, matrix.data(), &lda, singular.data(),
            left.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork,
            iwork.data(), &info);
    require_lapack(info, "dgesdd");
  }

  void update_row(std::uint32_t row, double sign) {
    if ((sign > 0.0) == active_[row])
      throw std::runtime_error("invalid support update");
    const int n = static_cast<int>(fixture_.n);
    const int m = static_cast<int>(fixture_.m);
    const int rank = static_cast<int>(singular_.size());
    std::vector<double> left_projection(rank), right_projection(rank);
    for (int component = 0; component < rank; ++component)
      left_projection[component] =
          U_[row + static_cast<std::size_t>(n) * component];
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p) {
      const int column = static_cast<int>(fixture_.indices[p]);
      const double value = sign * fixture_.values[p];
      for (int component = 0; component < rank; ++component)
        right_projection[component] +=
            V_[column + static_cast<std::size_t>(m) * component] * value;
    }

    std::vector<double> left_residual(n, 0.0), right_residual(m, 0.0);
    left_residual[row] = 1.0;
    for (int component = 0; component < rank; ++component) {
      const double weight = left_projection[component];
      for (int i = 0; i < n; ++i)
        left_residual[i] -=
            U_[i + static_cast<std::size_t>(n) * component] * weight;
    }
    for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
      right_residual[fixture_.indices[p]] = sign * fixture_.values[p];
    for (int component = 0; component < rank; ++component) {
      const double weight = right_projection[component];
      for (int column = 0; column < m; ++column)
        right_residual[column] -=
            V_[column + static_cast<std::size_t>(m) * component] * weight;
    }
    const double left_norm = cblas_dnrm2(n, left_residual.data(), 1);
    const double right_norm = cblas_dnrm2(m, right_residual.data(), 1);
    const double row_norm = [&] {
      long double sum = 0.0L;
      for (auto p = fixture_.indptr[row]; p < fixture_.indptr[row + 1]; ++p)
        sum += static_cast<long double>(fixture_.values[p])
               * fixture_.values[p];
      return std::sqrt(static_cast<double>(sum));
    }();
    const double orthogonality_gate =
        128.0 * std::numeric_limits<double>::epsilon();
    const bool extend_left = left_norm > orthogonality_gate;
    const bool extend_right =
        right_norm > orthogonality_gate * std::max(1.0, row_norm);
    const int core_rows = rank + static_cast<int>(extend_left);
    const int core_columns = rank + static_cast<int>(extend_right);
    std::vector<double> core(static_cast<std::size_t>(core_rows)
                                 * core_columns,
                             0.0);
    for (int component = 0; component < rank; ++component)
      core[component + static_cast<std::size_t>(core_rows) * component] =
          singular_[component];
    for (int column = 0; column < core_columns; ++column) {
      const double right = column < rank ? right_projection[column]
                                         : right_norm;
      for (int i = 0; i < core_rows; ++i) {
        const double left = i < rank ? left_projection[i] : left_norm;
        core[i + static_cast<std::size_t>(core_rows) * column] += left * right;
      }
    }

    std::vector<double> next_singular, core_left, core_vt;
    svd(std::move(core), core_rows, core_columns, next_singular, core_left,
        core_vt);
    ++updates_since_seed_;
    active_count_ += sign > 0.0 ? 1 : -1;
    const int keep = numerical_rank(next_singular,
                                    static_cast<int>(active_count_));
    if (keep == 0) throw std::runtime_error("rank-one update reached rank zero");

    std::vector<double> left_basis(static_cast<std::size_t>(n) * core_rows);
    for (int component = 0; component < rank; ++component)
      std::copy_n(U_.data() + static_cast<std::size_t>(n) * component, n,
                  left_basis.data() + static_cast<std::size_t>(n) * component);
    if (extend_left)
      for (int i = 0; i < n; ++i)
        left_basis[i + static_cast<std::size_t>(n) * rank] =
            left_residual[i] / left_norm;
    std::vector<double> right_basis(static_cast<std::size_t>(m)
                                    * core_columns);
    for (int component = 0; component < rank; ++component)
      std::copy_n(V_.data() + static_cast<std::size_t>(m) * component, m,
                  right_basis.data()
                      + static_cast<std::size_t>(m) * component);
    if (extend_right)
      for (int column = 0; column < m; ++column)
        right_basis[column + static_cast<std::size_t>(m) * rank] =
            right_residual[column] / right_norm;

    std::vector<double> next_u(static_cast<std::size_t>(n) * keep);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, keep, core_rows,
                1.0, left_basis.data(), n, core_left.data(), core_rows, 0.0,
                next_u.data(), n);
    std::vector<double> core_right(static_cast<std::size_t>(core_columns)
                                   * keep);
    const int thin = std::min(core_rows, core_columns);
    for (int component = 0; component < keep; ++component)
      for (int column = 0; column < core_columns; ++column)
        core_right[column + static_cast<std::size_t>(core_columns)
                                  * component] =
            core_vt[component + static_cast<std::size_t>(thin) * column];
    std::vector<double> next_v(static_cast<std::size_t>(m) * keep);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, keep,
                core_columns, 1.0, right_basis.data(), m, core_right.data(),
                core_columns, 0.0, next_v.data(), m);
    singular_.assign(next_singular.begin(), next_singular.begin() + keep);
    U_ = std::move(next_u);
    V_ = std::move(next_v);
    active_[row] = sign > 0.0;
  }

  const twalker::Fixture &fixture_;
  std::vector<std::uint32_t> rows_;
  std::vector<bool> active_;
  std::vector<double> singular_, U_, V_;
  std::size_t active_count_ = 0;
  std::size_t updates_since_seed_ = 0;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: incremental_svd_probe fixture.twfx...\n";
    return 2;
  }
  bool all_accurate = true;
  std::cout << std::setprecision(17);
  for (int argument = 1; argument < argc; ++argument) {
    const std::string path = argv[argument];
    try {
      const auto fixture = twalker::read_fixture(path);
      if (fixture.faces.empty()) throw std::runtime_error("fixture has no faces");
      IncrementalSvd solver(fixture);
      std::vector<double> elapsed;
      std::size_t accurate = 0;
      double max_error = 0.0;
      int min_rank = std::numeric_limits<int>::max();
      int max_rank = 0;
      for (std::size_t index = 0; index < fixture.faces.size(); ++index) {
        const auto start = Clock::now();
        if (index == 0)
          solver.seed(fixture.faces[index].rows);
        else
          solver.transition(fixture.faces[index].rows);
        const auto got = solver.face();
        const double micros = microseconds_since(start);
        if (index > 0) elapsed.push_back(micros);
        const auto &want = fixture.faces[index];
        const double g_error = relative_inf_error(got.g, want.g);
        const double h_error = relative_inf_error(got.h, want.h);
        const double ua_error = relative_inf_error(got.ua, want.ua);
        const double uc_error = relative_inf_error(got.uc, want.uc);
        const double error =
            std::max({g_error, h_error, ua_error, uc_error});
        max_error = std::max(max_error, error);
        accurate += std::isfinite(error) && error <= 1e-10;
        min_rank = std::min(min_rank, solver.rank());
        max_rank = std::max(max_rank, solver.rank());
        if (!(std::isfinite(error) && error <= 1e-10))
          std::cerr << stem(path) << " face=" << index << " error=" << error
                    << " components=" << g_error << ',' << h_error << ','
                    << ua_error << ',' << uc_error
                    << " rank=" << solver.rank()
                    << " ratio=" << solver.singular_ratio() << '\n';
      }
      std::sort(elapsed.begin(), elapsed.end());
      const double median = elapsed.empty()
                                ? std::numeric_limits<double>::infinity()
                                : elapsed[elapsed.size() / 2];
      all_accurate = all_accurate && accurate == fixture.faces.size();
      std::cout << "{\"model\":\"" << stem(path) << "\",\"faces\":"
                << fixture.faces.size() << ",\"accurate\":" << accurate
                << ",\"max_error\":" << max_error
                << ",\"min_rank\":" << min_rank
                << ",\"max_rank\":" << max_rank
                << ",\"median_update_face_us\":" << median << "}\n";
    } catch (const std::exception &error) {
      all_accurate = false;
      std::cerr << path << ": " << error.what() << '\n';
    }
  }
  return all_accurate ? 0 : 1;
}
