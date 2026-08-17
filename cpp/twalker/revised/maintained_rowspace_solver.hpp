#pragma once

#include "face_solver.hpp"
#include "revised_column_solver.hpp"

#include <cstdint>
#include <vector>

namespace twalker::revised {

struct MaintainedRowspaceStats {
  std::uint64_t calls = 0;
  std::uint64_t seeds = 0;
  std::uint64_t local_transitions = 0;
  std::uint64_t additions = 0;
  std::uint64_t removals = 0;
  std::uint64_t rank_increases = 0;
  std::uint64_t rank_decreases = 0;
  std::uint64_t rank_change_declines = 0;
  std::uint64_t numerical_declines = 0;
  std::uint64_t solves = 0;
  std::uint64_t refactors = 0;
  double seed_ms = 0.0;
  double transition_ms = 0.0;
  double solve_ms = 0.0;
  double worst_rowspace_residual = 0.0;
  double worst_orthogonality = 0.0;
  double worst_slope_residual = 0.0;
};

// Maintains the nonzero row space of A=B[rows,:] rather than A itself.
// Q has orthonormal rows, C=A*Q', and G=C'*C is positive definite even when
// A is strongly row/column rank deficient.  Ordinary dependent face changes
// are therefore Cholesky rank-one updates in the numerical rank r.
class MaintainedRowspaceSolver {
 public:
  explicit MaintainedRowspaceSolver(const Fixture &fixture)
      : fixture_(fixture) {}

  bool seed(const std::vector<std::uint32_t> &rows,
            const FaceSolution &direct);
  bool seed_from_rows(const std::vector<std::uint32_t> &rows,
                      std::int64_t expected_rank = -1);
  bool solve(const std::vector<std::uint32_t> &rows,
             RevisedSlopeSolution &solution);
  bool valid() const { return valid_; }
  void invalidate();
  const MaintainedRowspaceStats &stats() const { return stats_; }

 private:
  const Fixture &fixture_;
  std::vector<std::uint32_t> rows_;
  std::vector<double> rowspace_;    // rank-by-m, column-major
  std::vector<double> coordinates_; // active-by-rank, row-major
  std::vector<double> factor_;      // lower Cholesky, column-major
  std::vector<double> cross_b_;     // C'*b_W
  int rank_ = 0;
  int updates_since_refactor_ = 0;
  bool valid_ = false;
  bool cached_solution_valid_ = false;
  RevisedSlopeSolution cached_solution_;
  MaintainedRowspaceStats stats_;

  bool transition(const std::vector<std::uint32_t> &rows);
  bool add_row(std::uint32_t row);
  bool remove_row(std::uint32_t row);
  bool remove_rank_row(std::uint32_t row, std::size_t local,
                       const std::vector<double> &coordinate);
  bool row_coordinates(std::uint32_t row, std::vector<double> &coordinate,
                       std::vector<double> &residual,
                       double &relative_residual) const;
  bool refactor();
  bool form_solution(RevisedSlopeSolution &solution);
};

}  // namespace twalker::revised
