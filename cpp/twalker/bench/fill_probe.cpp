#include "fixture.hpp"

#include <SuiteSparseQR_C.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double micros(Clock::time_point start, Clock::time_point finish) {
  return std::chrono::duration<double, std::micro>(finish - start).count();
}

std::string stem(const std::string &path) {
  const auto slash = path.find_last_of('/');
  const auto dot = path.find_last_of('.');
  const auto begin = slash == std::string::npos ? 0 : slash + 1;
  const auto end = dot == std::string::npos || dot < begin ? path.size() : dot;
  return path.substr(begin, end - begin);
}

struct Common {
  cholmod_common value{};
  Common() {
    if (!cholmod_l_start(&value)) throw std::runtime_error("cholmod start failed");
  }
  ~Common() { cholmod_l_finish(&value); }
};

struct Factor {
  cholmod_sparse *R = nullptr;
  std::int64_t rank = -1;
  std::size_t nnz = 0;
  std::vector<std::int64_t> permutation;
  double assemble_us = 0.0;
  double factor_us = 0.0;
};

void free_factor(Factor &factor, cholmod_common *cc) {
  if (factor.R) cholmod_l_free_sparse(&factor.R, cc);
  factor.R = nullptr;
}

std::vector<std::uint32_t> permute_rows(const twalker::Fixture &fixture,
                                        const twalker::OracleFace &face,
                                        const std::string &kind) {
  auto rows = face.rows;
  if (kind == "original") return rows;
  auto key = [&](std::uint32_t row) {
    const auto first = fixture.indptr[row];
    const auto last = fixture.indptr[row + 1];
    const auto degree = last - first;
    const auto lo = degree ? fixture.indices[first] : fixture.m;
    const auto hi = degree ? fixture.indices[last - 1] : fixture.m;
    if (kind == "degree") return std::tuple<std::uint64_t, std::uint64_t,
                                             std::uint64_t>(degree, lo, row);
    if (kind == "span") return std::tuple<std::uint64_t, std::uint64_t,
                                           std::uint64_t>(lo, hi, row);
    return std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>(row, 0, 0);
  };
  std::stable_sort(rows.begin(), rows.end(),
                   [&](auto left, auto right) { return key(left) < key(right); });
  return rows;
}

Factor factor_face(const twalker::Fixture &fixture,
                   const std::vector<std::uint32_t> &rows, int ordering,
                   cholmod_common *cc,
                   const std::vector<std::int64_t> *column_map = nullptr) {
  Factor result;
  const auto assembly_start = Clock::now();
  std::size_t face_nnz = 0;
  for (auto row : rows)
    face_nnz += fixture.indptr[row + 1] - fixture.indptr[row];
  auto *triplet = cholmod_l_allocate_triplet(
      rows.size(), fixture.m, face_nnz, 0, CHOLMOD_REAL, cc);
  if (!triplet) throw std::runtime_error("triplet allocation failed");
  auto *ti = static_cast<std::int64_t *>(triplet->i);
  auto *tj = static_cast<std::int64_t *>(triplet->j);
  auto *tx = static_cast<double *>(triplet->x);
  std::size_t cursor = 0;
  for (std::size_t local = 0; local < rows.size(); ++local) {
    const auto row = rows[local];
    for (auto p = fixture.indptr[row]; p < fixture.indptr[row + 1]; ++p) {
      ti[cursor] = static_cast<std::int64_t>(local);
      const auto original_column = fixture.indices[p];
      tj[cursor] = column_map ? (*column_map)[original_column]
                              : original_column;
      tx[cursor] = fixture.values[p];
      ++cursor;
    }
  }
  triplet->nnz = cursor;
  auto *A = cholmod_l_triplet_to_sparse(triplet, cursor, cc);
  cholmod_l_free_triplet(&triplet, cc);
  auto *rhs = cholmod_l_allocate_dense(rows.size(), 1, rows.size(),
                                       CHOLMOD_REAL, cc);
  if (!A || !rhs) throw std::runtime_error("cholmod input allocation failed");
  auto *rhs_x = static_cast<double *>(rhs->x);
  for (std::size_t i = 0; i < rows.size(); ++i) rhs_x[i] = fixture.b[rows[i]];
  result.assemble_us = micros(assembly_start, Clock::now());

  cholmod_dense *qtb = nullptr;
  std::int64_t *permutation = nullptr;
  const auto factor_start = Clock::now();
  result.rank = SuiteSparseQR_C(ordering, SPQR_DEFAULT_TOL, fixture.m, 0, A,
                               nullptr, rhs, nullptr, &qtb, &result.R,
                               &permutation,
                               nullptr, nullptr, nullptr, cc);
  result.factor_us = micros(factor_start, Clock::now());
  cholmod_l_free_sparse(&A, cc);
  cholmod_l_free_dense(&rhs, cc);
  if (qtb) cholmod_l_free_dense(&qtb, cc);
  if (permutation) {
    result.permutation.assign(permutation, permutation + fixture.m);
    permutation = static_cast<std::int64_t *>(cholmod_l_free(
        fixture.m, sizeof(std::int64_t), permutation, cc));
  }
  if (result.rank < 0 || !result.R) throw std::runtime_error("SPQR failed");
  const auto *rp = static_cast<const std::int64_t *>(result.R->p);
  result.nnz = static_cast<std::size_t>(rp[result.R->ncol]);
  return result;
}

double triangular_pair_us(const Factor &factor, int batches) {
  const auto r = static_cast<std::size_t>(factor.rank);
  if (!r) return 0.0;
  const auto *rp = static_cast<const std::int64_t *>(factor.R->p);
  const auto *ri = static_cast<const std::int64_t *>(factor.R->i);
  const auto *rx = static_cast<const double *>(factor.R->x);
  std::vector<double> rhs(2 * r), work(2 * r);
  for (std::size_t i = 0; i < r; ++i) {
    rhs[i] = 1.0 / static_cast<double>(i + 1);
    rhs[r + i] = 1.0 / static_cast<double>(i + 2);
  }
  volatile double checksum = 0.0;
  const auto start = Clock::now();
  for (int batch = 0; batch < batches; ++batch) {
    std::copy(rhs.begin(), rhs.end(), work.begin());
    for (std::size_t reverse = 0; reverse < r; ++reverse) {
      const auto col = r - 1 - reverse;
      const auto diagonal_position = rp[col + 1] - 1;
      const double diagonal = (diagonal_position >= rp[col]
                               && static_cast<std::size_t>(
                                      ri[diagonal_position]) == col)
                                  ? rx[diagonal_position]
                                  : 0.0;
      if (!std::isfinite(diagonal) || diagonal == 0.0)
        return std::numeric_limits<double>::quiet_NaN();
      const double x0 = work[col] / diagonal;
      const double x1 = work[r + col] / diagonal;
      work[col] = x0;
      work[r + col] = x1;
      for (auto p = rp[col]; p < diagonal_position; ++p) {
        const auto row = static_cast<std::size_t>(ri[p]);
        if (row < col && row < r) {
          work[row] -= rx[p] * x0;
          work[r + row] -= rx[p] * x1;
        }
      }
    }
    checksum += work[batch % (2 * r)];
  }
  const auto elapsed = micros(start, Clock::now()) / batches;
  if (!std::isfinite(checksum)) std::cerr << "nonfinite checksum\n";
  return elapsed;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void emit(const std::string &model, std::size_t face_index,
          const std::string &ordering_name, const std::string &row_order,
          const Factor &factor, double assemble_us, double factor_us,
          double solve_us, std::size_t m, std::ostream &output) {
  const auto r = static_cast<double>(factor.rank);
  const auto capacity = r * m - r * (r - 1.0) / 2.0;
  const auto density = capacity ? factor.nnz / capacity : 0.0;
  output << "FILL," << model << ',' << face_index << ',' << ordering_name
         << ',' << row_order << ',' << factor.rank << ',' << factor.nnz << ','
         << density << ',' << assemble_us << ',' << factor_us << ',' << solve_us
         << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: fill_probe fixture.twfx... [--faces N] [--repeats N]\n";
    return 2;
  }
  int face_cap = 60;
  int repeats = 5;
  std::string output_path;
  std::vector<std::string> paths;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--faces" && i + 1 < argc) face_cap = std::atoi(argv[++i]);
    else if (arg == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
    else if (arg == "--out" && i + 1 < argc) output_path = argv[++i];
    else paths.push_back(arg);
  }
  const std::vector<std::pair<std::string, int>> orderings = {
      {"natural", SPQR_ORDERING_NATURAL}, {"colamd", SPQR_ORDERING_COLAMD},
      {"amd", SPQR_ORDERING_AMD}, {"metis", SPQR_ORDERING_METIS},
      {"default", SPQR_ORDERING_DEFAULT}, {"best", SPQR_ORDERING_BEST}};
  Common common;
  std::ofstream file_output;
  std::ostream *output = &std::cout;
  if (!output_path.empty()) {
    file_output.open(output_path);
    if (!file_output) {
      std::cerr << "cannot open output: " << output_path << '\n';
      return 2;
    }
    output = &file_output;
  }
  *output << "kind,model,face,ordering,row_order,rank,nnz_R,density,"
              "assemble_us,factor_us,tri_pair_us\n";
  try {
    for (const auto &path : paths) {
      const auto model = stem(path);
      const auto fixture = twalker::read_fixture(path);
      const auto count = std::min<std::size_t>(fixture.faces.size(), face_cap);
      std::vector<std::pair<std::string, std::vector<std::int64_t>>> fixed_maps;
      for (const auto &[name, ordering] :
           std::vector<std::pair<std::string, int>>{
               {"fixed_colamd", SPQR_ORDERING_COLAMD},
               {"fixed_amd", SPQR_ORDERING_AMD},
               {"fixed_default", SPQR_ORDERING_DEFAULT}}) {
        auto seed_factor = factor_face(
            fixture, permute_rows(fixture, fixture.faces.front(), "original"),
            ordering, &common.value);
        if (seed_factor.permutation.size() != fixture.m)
          throw std::runtime_error("seed factor did not return a permutation");
        std::vector<std::int64_t> inverse(fixture.m);
        for (std::size_t j = 0; j < fixture.m; ++j)
          inverse[seed_factor.permutation[j]] = static_cast<std::int64_t>(j);
        fixed_maps.push_back({name, std::move(inverse)});
        free_factor(seed_factor, &common.value);
      }
      for (std::size_t face_index = 0; face_index < count; ++face_index) {
        const auto &face = fixture.faces[face_index];
        for (const auto &[ordering_name, ordering] : orderings) {
          const auto rows = permute_rows(fixture, face, "original");
          std::vector<double> assembly, factor_times;
          Factor kept;
          for (int repeat = 0; repeat < repeats; ++repeat) {
            auto current = factor_face(fixture, rows, ordering, &common.value);
            assembly.push_back(current.assemble_us);
            factor_times.push_back(current.factor_us);
            if (repeat + 1 == repeats) kept = current;
            else free_factor(current, &common.value);
          }
          const auto solve_us = triangular_pair_us(kept, 1000);
          emit(model, face_index, ordering_name, "original", kept,
               median(assembly), median(factor_times), solve_us, fixture.m,
               *output);
          free_factor(kept, &common.value);
        }
        for (const auto &[ordering_name, column_map] : fixed_maps) {
          const auto rows = permute_rows(fixture, face, "original");
          std::vector<double> assembly, factor_times;
          Factor kept;
          for (int repeat = 0; repeat < repeats; ++repeat) {
            auto current = factor_face(fixture, rows, SPQR_ORDERING_FIXED,
                                       &common.value, &column_map);
            assembly.push_back(current.assemble_us);
            factor_times.push_back(current.factor_us);
            if (repeat + 1 == repeats) kept = current;
            else free_factor(current, &common.value);
          }
          emit(model, face_index, ordering_name, "original", kept,
               median(assembly), median(factor_times),
               triangular_pair_us(kept, 1000), fixture.m, *output);
          free_factor(kept, &common.value);
        }
        for (const std::string row_order : {"degree", "span"}) {
          const auto rows = permute_rows(fixture, face, row_order);
          std::vector<double> assembly, factor_times;
          Factor kept;
          for (int repeat = 0; repeat < repeats; ++repeat) {
            auto current = factor_face(fixture, rows, SPQR_ORDERING_DEFAULT,
                                       &common.value);
            assembly.push_back(current.assemble_us);
            factor_times.push_back(current.factor_us);
            if (repeat + 1 == repeats) kept = current;
            else free_factor(current, &common.value);
          }
          emit(model, face_index, "default", row_order, kept,
               median(assembly), median(factor_times),
               triangular_pair_us(kept, 1000), fixture.m, *output);
          free_factor(kept, &common.value);
        }
      }
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
