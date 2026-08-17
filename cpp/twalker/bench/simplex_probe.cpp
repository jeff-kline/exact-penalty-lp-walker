#include "fixture.hpp"
#include "highs_c_api_minimal.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void probe(const std::string &path) {
  const auto fixture = twalker::read_fixture(path);
  const int n = static_cast<int>(fixture.n);
  const int m = static_cast<int>(fixture.m);
  std::vector<int> starts(n + 1), indices(fixture.nnz);
  for (int row = 0; row <= n; ++row)
    starts[row] = static_cast<int>(fixture.indptr[row]);
  for (std::size_t p = 0; p < fixture.nnz; ++p)
    indices[p] = static_cast<int>(fixture.indices[p]);
  std::vector<double> col_lower(m, -1e30), col_upper(m, 1e30),
      row_upper(n, 1e30), times;
  std::vector<int> iterations;
  for (int repeat = 0; repeat < 3; ++repeat) {
    void *highs = Highs_create();
    if (!highs) throw std::runtime_error("Highs_create failed");
    Highs_setBoolOptionValue(highs, "output_flag", 0);
    Highs_setIntOptionValue(highs, "threads", 1);
    Highs_setStringOptionValue(highs, "presolve", "off");
    Highs_setStringOptionValue(highs, "solver", "simplex");
    const int pass = Highs_passLp(
        highs, m, n, static_cast<int>(fixture.nnz), 2, 1, 0.0,
        fixture.d.data(), col_lower.data(), col_upper.data(), fixture.b.data(),
        row_upper.data(), starts.data(), indices.data(), fixture.values.data());
    const auto begin = std::chrono::steady_clock::now();
    const int run = pass == 0 ? Highs_run(highs) : -1;
    times.push_back(std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - begin)
                        .count());
    int count = -1;
    if (run == 0 && Highs_getModelStatus(highs) == 7)
      Highs_getIntInfoValue(highs, "simplex_iteration_count", &count);
    iterations.push_back(count);
    Highs_destroy(highs);
  }
  std::sort(times.begin(), times.end());
  std::sort(iterations.begin(), iterations.end());
  const double wall = times[1];
  const int pivots = iterations[1];
  std::cout << std::setprecision(17) << "{\"model\":\"" << path
            << "\",\"simplex_iterations\":" << pivots
            << ",\"median_run_us\":" << wall
            << ",\"us_per_iteration\":" << wall / pivots << "}\n";
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
