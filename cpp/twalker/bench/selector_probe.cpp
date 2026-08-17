#include "fixture.hpp"
#include "highs_c_api_minimal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double row_product(const twalker::Fixture &fixture, std::size_t row,
                   const std::vector<double> &u) {
  double value = 0.0;
  for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p)
    value += fixture.values[p] * u[fixture.indices[p]];
  return value;
}

void set_bounds(const twalker::Fixture &fixture,
                const twalker::OracleFace &face,
                std::vector<double> &lower, std::vector<double> &upper) {
  std::fill(lower.begin(), lower.end(), -1e30);
  for (std::size_t row = 0; row < fixture.n; ++row)
    upper[row] = -face.t * fixture.b[row];
  for (std::size_t local = 0; local < face.rows.size(); ++local) {
    const auto row = face.rows[local];
    const double equality = face.t * face.g[local] + face.h[local]
                            - face.t * fixture.b[row];
    lower[row] = upper[row] = equality;
  }
}

void probe(const std::string &path) {
  const auto fixture = twalker::read_fixture(path);
  const int n = static_cast<int>(fixture.n);
  const int m = static_cast<int>(fixture.m);
  std::vector<int> starts(n + 1), indices(fixture.nnz);
  for (int row = 0; row <= n; ++row)
    starts[row] = static_cast<int>(fixture.indptr[row]);
  for (std::size_t p = 0; p < fixture.nnz; ++p)
    indices[p] = static_cast<int>(fixture.indices[p]);
  std::vector<double> cost(m, 0.0), col_lower(m, -1e30),
      col_upper(m, 1e30), lower(n), upper(n), u(m);

  void *highs = Highs_create();
  if (!highs) throw std::runtime_error("Highs_create failed");
  Highs_setBoolOptionValue(highs, "output_flag", 0);
  Highs_setIntOptionValue(highs, "threads", 1);
  Highs_setStringOptionValue(highs, "presolve", "off");
  Highs_setStringOptionValue(highs, "solver", "simplex");

  std::vector<double> microseconds;
  double worst = 0.0;
  int passed = 0;
  for (std::size_t k = 0; k < fixture.faces.size(); ++k) {
    const auto &face = fixture.faces[k];
    set_bounds(fixture, face, lower, upper);
    int status = 0;
    if (k == 0) {
      status = Highs_passLp(highs, m, n, static_cast<int>(fixture.nnz), 2, 1,
                            0.0, cost.data(), col_lower.data(),
                            col_upper.data(), lower.data(), upper.data(),
                            starts.data(), indices.data(),
                            fixture.values.data());
    } else {
      status = Highs_changeRowsBoundsByRange(
          highs, 0, n - 1, lower.data(), upper.data());
    }
    const auto begin = std::chrono::steady_clock::now();
    if (status == 0) status = Highs_run(highs);
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
    microseconds.push_back(us);
    if (status != 0 || Highs_getModelStatus(highs) != 7
        || Highs_getSolution(highs, u.data(), nullptr, nullptr, nullptr) != 0)
      continue;
    double error = 0.0;
    for (int row = 0; row < n; ++row) {
      const double value = row_product(fixture, row, u);
      const double scale = 1.0 + std::abs(value) + std::abs(upper[row]);
      if (lower[row] == upper[row])
        error = std::max(error, std::abs(value - upper[row]) / scale);
      else
        error = std::max(error, std::max(0.0, value - upper[row]) / scale);
    }
    worst = std::max(worst, error);
    passed += error <= 1e-7;
  }
  Highs_destroy(highs);
  const double first = microseconds.front();
  std::vector<double> warm(microseconds.begin() + 1, microseconds.end());
  std::sort(microseconds.begin(), microseconds.end());
  std::sort(warm.begin(), warm.end());
  const double median = microseconds[microseconds.size() / 2];
  const double warm_median = warm.empty() ? median : warm[warm.size() / 2];
  std::cout << std::setprecision(17)
            << "{\"model\":\"" << path << "\",\"faces\":"
            << fixture.faces.size() << ",\"passed\":" << passed
            << ",\"median_us\":" << median
            << ",\"warm_median_us\":" << warm_median
            << ",\"first_us\":" << first
            << ",\"worst_error\":" << worst << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  try {
    for (int i = 1; i < argc; ++i) probe(argv[i]);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
