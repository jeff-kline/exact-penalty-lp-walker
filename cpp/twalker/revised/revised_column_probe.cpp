#include "fixture.hpp"
#include "face_solver.hpp"
#include "revised_column_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
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
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: revised_column_probe fixture.twfx...\n";
    return 2;
  }
  bool all_safe = true;
  std::cout << std::setprecision(17);
  for (int argument = 1; argument < argc; ++argument) {
    const std::string path = argv[argument];
    try {
      const auto fixture = twalker::read_fixture(path);
      twalker::revised::RevisedColumnSolver solver(fixture);
      std::unique_ptr<twalker::FaceSolver> direct;
      if (!std::getenv("TWALKER_DISABLE_REVISED_FACTORED_SEED"))
        direct = std::make_unique<twalker::FaceSolver>(fixture, false);
      std::vector<double> elapsed;
      std::size_t solved = 0, accurate = 0;
      std::int64_t last_rank = 0;
      double max_error = 0.0, max_residual = 0.0;
      double min_active_ratio = std::numeric_limits<double>::infinity();
      double min_transform_ratio = std::numeric_limits<double>::infinity();
      for (const auto &face : fixture.faces) {
        if (direct && solver.needs_factored_seed()) {
          direct->set_factored_seed_needed(true);
          auto seed = direct->solve(face.rows);
          solver.seed_from_direct(face.rows, seed);
          direct->set_factored_seed_needed(false);
        }
        twalker::revised::RevisedFaceSolution got;
        const auto start = std::chrono::steady_clock::now();
        if (!solver.solve(face.rows, got)) continue;
        elapsed.push_back(std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - start)
                              .count());
        ++solved;
        last_rank = got.rank;
        const double error = std::max(
            {twalker::revised::relative_inf_error(got.g, face.g),
             twalker::revised::relative_inf_error(got.h, face.h),
             twalker::revised::relative_inf_error(got.ua, face.ua),
             twalker::revised::relative_inf_error(got.uc, face.uc)});
        max_error = std::max(max_error, error);
        max_residual = std::max(
            max_residual, std::max(got.dres, got.piece_residual));
        min_active_ratio = std::min(min_active_ratio,
                                    got.basis_diagonal_ratio);
        min_transform_ratio = std::min(min_transform_ratio,
                                       got.coordinate_diagonal_ratio);
        accurate += std::isfinite(error) && error <= 1e-10;
      }
      std::sort(elapsed.begin(), elapsed.end());
      const double median_us = elapsed.empty()
                                   ? std::numeric_limits<double>::infinity()
                                   : elapsed[elapsed.size() / 2];
      const auto &s = solver.stats();
      all_safe = all_safe && accurate == solved;
      std::cout << "{\"model\":\"" << stem(path) << "\",\"faces\":"
                << fixture.faces.size() << ",\"solved\":" << solved
                << ",\"n\":" << fixture.n << ",\"m\":" << fixture.m
                << ",\"rank\":" << last_rank
                << ",\"accurate\":" << accurate
                << ",\"max_error\":" << max_error
                << ",\"max_residual\":" << max_residual
                << ",\"min_active_ratio\":" << min_active_ratio
                << ",\"min_transform_ratio\":" << min_transform_ratio
                << ",\"median_us\":" << median_us
                << ",\"rebuilds\":" << s.rebuilds
                << ",\"local_transitions\":" << s.local_transitions
                << ",\"rank_changes\":" << s.rank_changes
                << ",\"row_additions\":" << s.row_additions
                << ",\"row_removals\":" << s.row_removals
                << ",\"condition_declines\":" << s.condition_declines
                << ",\"residual_declines\":" << s.residual_declines
                << ",\"refinements\":" << s.refinements
                << ",\"declines\":" << s.declines
                << ",\"retirements\":" << s.retirements
                << ",\"worst_entering_representation\":"
                << s.worst_entering_representation
                << ",\"phase_ms\":{\"rebuild\":" << s.rebuild_ms
                << ",\"transition\":" << s.transition_ms
                << ",\"condition\":" << s.condition_ms
                << ",\"rank_test\":" << s.rank_test_ms
                << ",\"factor_update\":" << s.factor_update_ms
                << ",\"solve\":" << s.solve_ms
                << ",\"coefficient\":" << s.coefficient_ms
                << ",\"projection\":" << s.projection_ms
                << ",\"products\":" << s.products_ms
                << ",\"residual\":" << s.residual_ms << "}}\n";
    } catch (const std::exception &error) {
      all_safe = false;
      std::cerr << path << ": " << error.what() << '\n';
    }
  }
  return all_safe ? 0 : 1;
}
