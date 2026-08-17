#include "face_solver.hpp"
#include "fixture.hpp"

#include <algorithm>
#include <chrono>
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
    std::cerr << "usage: verify_face_solver fixture.twfx...\n";
    return 2;
  }
  constexpr double gate = 1e-10;
  bool all_good = true;
  std::cout << std::setprecision(17);
  for (int argument = 1; argument < argc; ++argument) {
    const std::string path = argv[argument];
    try {
      const auto fixture = twalker::read_fixture(path);
      twalker::FaceSolver solver(fixture);
      std::size_t accurate = 0, declines = 0;
      std::size_t dense_fallbacks = 0;
      std::int64_t min_rank = std::numeric_limits<std::int64_t>::max();
      std::int64_t max_rank = 0;
      double max_error = 0.0, max_dres = 0.0, max_piece = 0.0;
      double min_core_ratio = std::numeric_limits<double>::infinity();
      std::vector<double> elapsed;
      for (const auto &face : fixture.faces) {
        const auto start = std::chrono::steady_clock::now();
        try {
          const auto got = solver.solve(face.rows);
          elapsed.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - start)
                                .count());
          const double error = std::max(
              {twalker::relative_inf_error(got.g, face.g),
               twalker::relative_inf_error(got.h, face.h),
               twalker::relative_inf_error(got.ua, face.ua),
               twalker::relative_inf_error(got.uc, face.uc)});
          max_error = std::max(max_error, error);
          max_dres = std::max(max_dres, got.dres);
          max_piece = std::max(max_piece, got.piece_residual);
          min_core_ratio = std::min(min_core_ratio, got.core_diagonal_ratio);
          dense_fallbacks += got.used_dense_fallback;
          min_rank = std::min(min_rank, got.rank);
          max_rank = std::max(max_rank, got.rank);
          if (!(std::isfinite(error) && error <= gate))
            std::cerr << stem(path) << " face="
                      << (&face - fixture.faces.data()) << " error=" << error
                      << " core_ratio=" << got.core_diagonal_ratio
                      << " dres=" << got.dres
                      << " piece=" << got.piece_residual << '\n';
          accurate += std::isfinite(error) && error <= gate;
        } catch (const twalker::FaceDecline &) {
          ++declines;
        }
      }
      std::sort(elapsed.begin(), elapsed.end());
      const double median_us = elapsed.empty()
                                   ? std::numeric_limits<double>::infinity()
                                   : elapsed[elapsed.size() / 2];
      const bool good = accurate == fixture.faces.size() && declines == 0;
      all_good = all_good && good;
      std::cout << "{\"model\":\"" << stem(path) << "\",\"faces\":"
                << fixture.faces.size() << ",\"accurate\":" << accurate
                << ",\"declines\":" << declines << ",\"max_error\":"
                << max_error << ",\"dense_fallbacks\":" << dense_fallbacks
                << ",\"min_rank\":" << min_rank
                << ",\"max_rank\":" << max_rank
                << ",\"max_dres\":" << max_dres
                << ",\"max_piece_residual\":" << max_piece
                << ",\"min_core_ratio\":" << min_core_ratio
                << ",\"median_us\":" << median_us
                << ",\"phase_ms\":{\"total\":" << solver.stats().total_ms
                << ",\"calls\":" << solver.stats().calls
                << ",\"numerical_calls\":" << solver.stats().numerical_calls
                << ",\"cache_hits\":" << solver.stats().cache_hits
                << ",\"assembly\":" << solver.stats().assembly_ms
                << ",\"spqr\":" << solver.stats().spqr_ms
                << ",\"core_extract\":" << solver.stats().core_extract_ms
                << ",\"rz\":" << solver.stats().rz_ms
                << ",\"triangular\":" << solver.stats().triangular_ms
                << ",\"products\":" << solver.stats().products_ms
                << ",\"residual\":" << solver.stats().residual_ms
                << ",\"dense_svd\":" << solver.stats().dense_svd_ms
                << ",\"direct_guard_audits\":"
                << solver.stats().direct_guard_audits
                << ",\"direct_guard_raw_accurate\":"
                << solver.stats().direct_guard_raw_accurate
                << ",\"direct_guard_refined_accurate\":"
                << solver.stats().direct_guard_refined_accurate
                << ",\"direct_guard_tail_accepts\":"
                << solver.stats().direct_guard_tail_accepts
                << ",\"direct_guard_tail_false_accepts\":"
                << solver.stats().direct_guard_tail_false_accepts
                << ",\"direct_guard_rank_mismatches\":"
                << solver.stats().direct_guard_rank_mismatches
                << ",\"direct_guard_rank_match_accurate\":"
                << solver.stats().direct_guard_rank_match_accurate
                << ",\"direct_guard_rank_mismatch_accurate\":"
                << solver.stats().direct_guard_rank_mismatch_accurate
                << ",\"direct_guard_ferr_accepts\":"
                << solver.stats().direct_guard_ferr_accepts
                << ",\"direct_guard_ferr_false_accepts\":"
                << solver.stats().direct_guard_ferr_false_accepts
                << ",\"direct_guard_consensus_attempts\":"
                << solver.stats().direct_guard_consensus_attempts
                << ",\"direct_guard_consensus_accepts\":"
                << solver.stats().direct_guard_consensus_accepts
                << ",\"direct_guard_consensus_false_accepts\":"
                << solver.stats().direct_guard_consensus_false_accepts
                << ",\"direct_guard_tight_consensus_accepts\":"
                << solver.stats().direct_guard_tight_consensus_accepts
                << ",\"direct_guard_tight_consensus_false_accepts\":"
                << solver.stats().direct_guard_tight_consensus_false_accepts
                << ",\"core_svd_audits\":" << solver.stats().core_svd_audits
                << ",\"core_svd_accurate\":" << solver.stats().core_svd_accurate
                << ",\"core_svd_rank_mismatches\":"
                << solver.stats().core_svd_rank_mismatches
                << ",\"direct_guard_raw_max_error\":"
                << solver.stats().direct_guard_raw_max_error
                << ",\"direct_guard_refined_max_error\":"
                << solver.stats().direct_guard_refined_max_error
                << ",\"direct_guard_tail_worst_error\":"
                << solver.stats().direct_guard_tail_worst_error
                << ",\"direct_guard_ferr_worst_error\":"
                << solver.stats().direct_guard_ferr_worst_error
                << ",\"direct_guard_consensus_worst_error\":"
                << solver.stats().direct_guard_consensus_worst_error
                << ",\"direct_guard_consensus_ms\":"
                << solver.stats().direct_guard_consensus_ms
                << ",\"direct_guard_tight_consensus_worst_error\":"
                << solver.stats().direct_guard_tight_consensus_worst_error
                << ",\"core_svd_ms\":" << solver.stats().core_svd_ms
                << ",\"core_svd_max_error\":"
                << solver.stats().core_svd_max_error
                << ",\"direct_guard_refinement_ms\":"
                << solver.stats().direct_guard_refinement_ms
                << ",\"qr_update_observations\":"
                << solver.stats().qr_update_observations
                << ",\"qr_update_eligible\":"
                << solver.stats().qr_update_eligible
                << ",\"qr_update_attempts\":"
                << solver.stats().qr_update_attempts
                << ",\"qr_update_admitted\":"
                << solver.stats().qr_update_admitted
                << ",\"qr_update_accurate\":"
                << solver.stats().qr_update_accurate
                << ",\"qr_update_false_admits\":"
                << solver.stats().qr_update_false_admits
                << ",\"qr_update_live_returns\":"
                << solver.stats().qr_update_live_returns
                << ",\"qr_update_refactors\":"
                << solver.stats().qr_update_refactors
                << ",\"qr_update_additions\":"
                << solver.stats().qr_update_additions
                << ",\"qr_update_downdates\":"
                << solver.stats().qr_update_downdates
                << ",\"qr_update_large_changes\":"
                << solver.stats().qr_update_large_changes
                << ",\"qr_update_numerical_declines\":"
                << solver.stats().qr_update_numerical_declines
                << ",\"qr_update_update_ms\":"
                << solver.stats().qr_update_update_ms
                << ",\"qr_update_solve_ms\":"
                << solver.stats().qr_update_solve_ms
                << ",\"qr_update_median_us\":"
                << solver.stats().qr_update_median_us
                << ",\"qr_update_oracle_median_us\":"
                << solver.stats().qr_update_oracle_median_us
                << ",\"qr_update_max_error\":"
                << solver.stats().qr_update_max_error
                << ",\"qr_update_max_admitted_error\":"
                << solver.stats().qr_update_max_admitted_error
                << ",\"qr_update_max_residual\":"
                << solver.stats().qr_update_max_residual
                << ",\"cache\":" << solver.stats().cache_ms
                << "}}\n";
    } catch (const std::exception &error) {
      all_good = false;
      std::cerr << path << ": " << error.what() << '\n';
    }
  }
  return all_good ? 0 : 1;
}
