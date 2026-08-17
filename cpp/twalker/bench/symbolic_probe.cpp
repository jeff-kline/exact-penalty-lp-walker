#include "fixture.hpp"

#include <SuiteSparseQR.hpp>
#include <SuiteSparseQR_C.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void probe(const std::string &path) {
  const auto fixture = twalker::read_fixture(path);
  cholmod_common common{};
  if (!cholmod_l_start(&common)) throw std::runtime_error("cholmod start");
  auto *triplet = cholmod_l_allocate_triplet(
      fixture.n, fixture.m, fixture.nnz, 0, CHOLMOD_REAL, &common);
  if (!triplet) throw std::runtime_error("triplet allocation");
  auto *ti = static_cast<std::int64_t *>(triplet->i);
  auto *tj = static_cast<std::int64_t *>(triplet->j);
  auto *tx = static_cast<double *>(triplet->x);
  std::size_t cursor = 0;
  for (std::size_t row = 0; row < fixture.n; ++row)
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      ti[cursor] = static_cast<std::int64_t>(row);
      tj[cursor] = static_cast<std::int64_t>(fixture.indices[p]);
      tx[cursor++] = fixture.values[p];
    }
  triplet->nnz = cursor;
  auto *matrix = cholmod_l_triplet_to_sparse(triplet, cursor, &common);
  cholmod_l_free_triplet(&triplet, &common);
  if (!matrix) throw std::runtime_error("sparse conversion");
  auto *rows = static_cast<std::int64_t *>(matrix->i);
  auto *values = static_cast<double *>(matrix->x);
  const auto *starts = static_cast<const std::int64_t *>(matrix->p);
  const std::size_t nnz = static_cast<std::size_t>(starts[matrix->ncol]);
  std::vector<double> base(values, values + nnz);
  auto *factor = SuiteSparseQR_C_symbolic(
      SPQR_ORDERING_AMD, 1, matrix, &common);
  if (!factor) throw std::runtime_error("symbolic factorization");

  std::vector<double> times;
  std::vector<std::uint8_t> support(fixture.n);
  int valid = 0;
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (const auto &face : fixture.faces) {
      std::fill(support.begin(), support.end(), 0);
      for (auto row : face.rows) support[row] = 1;
      for (std::size_t p = 0; p < nnz; ++p)
        values[p] = support[rows[p]] ? base[p] : 0.0;
      const auto begin = std::chrono::steady_clock::now();
      const int ok = SuiteSparseQR_C_numeric(
          SPQR_DEFAULT_TOL, matrix, factor, &common);
      const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
      times.push_back(us);
      if (ok) {
        auto *internal = static_cast<
            SuiteSparseQR_factorization<double, std::int64_t> *>(
            factor->factors);
        valid += internal && internal->rank > 0;
      }
    }
  }
  SuiteSparseQR_C_free(&factor, &common);
  cholmod_l_free_sparse(&matrix, &common);
  cholmod_l_finish(&common);
  std::sort(times.begin(), times.end());
  std::cout << std::setprecision(17) << "{\"model\":\"" << path
            << "\",\"numeric_calls\":" << times.size()
            << ",\"valid\":" << valid << ",\"median_us\":"
            << times[times.size() / 2] << ",\"p90_us\":"
            << times[9 * times.size() / 10] << "}\n";
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
