#pragma once

#include "face_solver.hpp"
#include "revised_basis_solver.hpp"

#include <cstdint>
#include <vector>

namespace twalker::revised {

struct MaintainedSvdFaceStats {
  std::uint64_t calls = 0, seeds = 0, transitions = 0, solves = 0;
  std::uint64_t additions = 0, removals = 0;
  std::uint64_t rank_change_declines = 0, numerical_declines = 0;
  double seed_ms = 0.0, transition_ms = 0.0, solve_ms = 0.0;
  double worst_representation_residual = 0.0;
  double worst_piece_residual = 0.0;
};

// Correctness-first Q-aware face updater.  It retains the authoritative thin
// SVD A=U*S*V' and updates U and V under rank-preserving row changes.  This is
// deliberately a bounded audit prototype: the small dense update can later
// be replaced by secular/Givens kernels if the representation is admitted.
class MaintainedSvdFaceSolver {
 public:
  explicit MaintainedSvdFaceSolver(
      const Fixture &fixture, std::vector<double> target_shift = {});

  bool seed(const std::vector<std::uint32_t> &rows,
            const FaceSolution &direct);
  bool solve(const std::vector<std::uint32_t> &rows,
             RevisedFaceSolution &solution);
  void invalidate();
  const MaintainedSvdFaceStats &stats() const { return stats_; }

 private:
  const Fixture &fixture_;
  std::vector<double> target_shift_;
  std::vector<std::uint32_t> rows_;
  std::vector<double> U_;       // active-by-r, column-major
  std::vector<double> singular_;
  std::vector<double> Vt_;      // r-by-m, column-major
  bool valid_ = false;
  MaintainedSvdFaceStats stats_;

  bool transition(const std::vector<std::uint32_t> &rows);
  bool add_row(std::uint32_t row);
  bool remove_row(std::uint32_t row);
  bool form_solution(RevisedFaceSolution &solution);
};

}  // namespace twalker::revised
