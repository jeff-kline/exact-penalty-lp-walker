#pragma once

#include "face_solver.hpp"
#include "revised_basis_solver.hpp"
#include "revised_column_solver.hpp"

#include <cstdint>
#include <map>
#include <vector>

namespace twalker::revised {

struct MaintainedDeficientQrStats {
  std::uint64_t calls = 0, seeds = 0, transitions = 0, unchanged = 0;
  std::uint64_t additions = 0, removals = 0;
  std::uint64_t rank_change_declines = 0, numerical_declines = 0, solves = 0;
  std::uint64_t weak_exchange_attempts = 0, weak_exchange_accepts = 0;
  std::uint64_t weak_exchange_declines = 0, weak_exchange_dimension = 0;
  std::uint64_t refinement_attempts = 0, refinement_steps = 0;
  double seed_ms = 0.0, transition_ms = 0.0, solve_ms = 0.0;
  double weak_exchange_ms = 0.0;
  double worst_representation_residual = 0.0, worst_slope_residual = 0.0;
  double worst_weak_exchange_residual = 0.0;
};

// Stable deficient-face analogue of a revised-simplex basis.  For A=B[W,:]
// retain pivot columns C=A[:,J], their QR factor R, and A=C*T.  Rank-
// preserving row changes update R in O(r^2); fixed T is factored once by
// TZRZF, so minimum-norm multipliers never use normal equations.
class MaintainedDeficientQrSolver {
 public:
  explicit MaintainedDeficientQrSolver(
      const Fixture &fixture, std::vector<double> target_shift = {})
      : fixture_(fixture), target_shift_(std::move(target_shift)) {}
  bool seed(const std::vector<std::uint32_t> &rows,
            const FaceSolution &direct);
  bool solve(const std::vector<std::uint32_t> &rows,
             RevisedSlopeSolution &solution);
  // Full affine face solve using the same retained independent core.  This
  // is the ordinary deficient-face analogue of a revised-simplex tableau:
  // both fixed right-hand sides are updated without reconstructing SPQR/RZ.
  bool solve_face(const std::vector<std::uint32_t> &rows,
                  RevisedFaceSolution &solution);
  // Rare late polish, called only after a known forward horizon amplifies the
  // raw slope residual beyond the endpoint budget.
  bool refine(const std::vector<std::uint32_t> &rows,
              RevisedSlopeSolution &solution, int max_steps = 2);
  void invalidate();
  const MaintainedDeficientQrStats &stats() const { return stats_; }

 private:
  const Fixture &fixture_;
  std::vector<double> target_shift_;
  std::vector<std::uint32_t> rows_, basis_columns_;
  std::vector<std::int64_t> permutation_;
  std::vector<double> R_, transform_, transform_rz_, transform_tau_, cross_;
  int rank_ = 0, updates_ = 0;
  bool valid_ = false, cached_valid_ = false;
  bool orthonormal_coordinates_ = false;
  RevisedSlopeSolution cached_;
  RevisedFaceSolution cached_face_;
  bool cached_face_valid_ = false;
  std::map<std::vector<std::uint32_t>, RevisedSlopeSolution> solution_cache_;
  std::map<std::vector<std::uint32_t>, RevisedFaceSolution> face_cache_;
  MaintainedDeficientQrStats stats_;
  bool transition(const std::vector<std::uint32_t> &rows);
  bool weak_exchange(const std::vector<std::uint32_t> &rows,
                     const std::vector<std::uint32_t> &add,
                     const std::vector<std::uint32_t> &remove);
  bool update_row(std::uint32_t row, int sign);
  bool entering_row_is_dependent(std::uint32_t row);
  bool row_coordinate(std::uint32_t row, std::vector<double> &coordinate,
                      std::vector<double> *residual = nullptr,
                      double *relative_residual = nullptr) const;
  bool refactor_transform_rz();
  bool form_solution(RevisedSlopeSolution &solution, int refinement_steps = 0);
  bool form_face_solution(RevisedFaceSolution &solution,
                          int refinement_steps = 1);
  double diagonal_ratio() const;
};

}  // namespace twalker::revised
