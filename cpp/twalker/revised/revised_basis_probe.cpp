#include "fixture.hpp"
#include "revised_basis_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

std::string stem(const std::string &path) {
  const auto slash = path.find_last_of('/');
  const auto dot = path.find_last_of('.');
  const auto begin = slash == std::string::npos ? 0 : slash + 1;
  const auto end = dot == std::string::npos || dot < begin ? path.size() : dot;
  return path.substr(begin, end - begin);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: revised_basis_probe fixture.twfx...\n";
    return 2;
  }
  constexpr double oracle_gate = 1e-10;
  bool all_good = true;
  std::cout << std::setprecision(17);
  for (int argument = 1; argument < argc; ++argument) {
    const std::string path = argv[argument];
    try {
      const auto fixture = twalker::read_fixture(path);
      twalker::revised::RevisedBasisSolver solver(fixture);
      std::vector<double> elapsed;
      std::size_t accurate = 0, solved = 0;
      double max_error = 0.0, max_residual = 0.0;
      double min_basis_ratio = std::numeric_limits<double>::infinity();
      double min_coordinate_ratio = std::numeric_limits<double>::infinity();
      std::int64_t min_rank = std::numeric_limits<std::int64_t>::max();
      std::int64_t max_rank = 0;
      for (const auto &face : fixture.faces) {
        twalker::revised::RevisedFaceSolution got;
        const auto start = std::chrono::steady_clock::now();
        if (!solver.solve(face.rows, got)) continue;
        elapsed.push_back(std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - start)
                              .count());
        ++solved;
        min_rank = std::min(min_rank, got.rank);
        max_rank = std::max(max_rank, got.rank);
        const double error = std::max(
            {twalker::revised::relative_inf_error(got.g, face.g),
             twalker::revised::relative_inf_error(got.h, face.h),
             twalker::revised::relative_inf_error(got.ua, face.ua),
             twalker::revised::relative_inf_error(got.uc, face.uc)});
        max_error = std::max(max_error, error);
        max_residual = std::max(
            max_residual, std::max(got.dres, got.piece_residual));
        min_basis_ratio = std::min(min_basis_ratio,
                                   got.basis_diagonal_ratio);
        min_coordinate_ratio = std::min(min_coordinate_ratio,
                                        got.coordinate_diagonal_ratio);
        accurate += std::isfinite(error) && error <= oracle_gate;
      }
      std::sort(elapsed.begin(), elapsed.end());
      const double median_us = elapsed.empty()
                                   ? std::numeric_limits<double>::infinity()
                                   : elapsed[elapsed.size() / 2];
      const auto &stats = solver.stats();
      const bool good = accurate == solved;
      all_good = all_good && good;
      std::cout << "{\"model\":\"" << stem(path) << "\",\"faces\":"
                << fixture.faces.size() << ",\"solved\":" << solved
                << ",\"accurate\":" << accurate
                << ",\"max_error\":" << max_error
                << ",\"max_residual\":" << max_residual
                << ",\"min_basis_ratio\":" << min_basis_ratio
                << ",\"min_coordinate_ratio\":" << min_coordinate_ratio
                << ",\"min_rank\":" << min_rank
                << ",\"max_rank\":" << max_rank
                << ",\"median_us\":" << median_us
                << ",\"rebuilds\":" << stats.rebuilds
                << ",\"unchanged_reuses\":" << stats.unchanged_reuses
                << ",\"local_transitions\":" << stats.local_transitions
                << ",\"row_additions\":" << stats.row_additions
                << ",\"row_removals\":" << stats.row_removals
                << ",\"rank_changes\":" << stats.rank_changes
                << ",\"basis_removals\":" << stats.basis_removals
                << ",\"basis_exchanges\":" << stats.basis_exchanges
                << ",\"update_failures\":" << stats.update_failures
                << ",\"condition_declines\":" << stats.condition_declines
                << ",\"residual_declines\":" << stats.residual_declines
                << ",\"declines\":" << stats.declines
                << ",\"worst_row_reconstruction\":"
                << stats.worst_row_reconstruction
                << ",\"phase_ms\":{\"rebuild\":" << stats.rebuild_ms
                << ",\"transition\":" << stats.transition_ms
                << ",\"solve\":" << stats.solve_ms
                << ",\"products\":" << stats.products_ms << "}}\n";
    } catch (const std::exception &error) {
      all_good = false;
      std::cerr << path << ": " << error.what() << '\n';
    }
  }
  return all_good ? 0 : 1;
}
