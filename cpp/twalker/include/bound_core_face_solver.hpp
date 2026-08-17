#pragma once

#include "face_solver.hpp"
#include "fixture.hpp"

#include <cstdint>
#include <vector>

namespace twalker {

struct BoundCoreSolveStats {
  int calls = 0;
  int accepted = 0;
  int structural_declines = 0;
  int numerical_declines = 0;
  int refinements = 0;
  int maximum_border = 0;
  double total_ms = 0.0;
  double factor_ms = 0.0;
  double products_ms = 0.0;
  double worst_dres = 0.0;
  double worst_piece_residual = 0.0;
  double minimum_rcond = 1.0;
};

// Exact full-column-rank face solver for operators consisting mostly of
// one-entry bound rows and a small collection of general rows.  If
//
//   B_W' B_W = D + C' C,
//
// D is diagonal.  Positive entries of D are eliminated analytically and its
// few zeros are retained as a border, leaving a dense system of order
// active_core_rows + zero_diagonal_entries.  Every candidate is checked
// against the original active operator and numerical failures decline to the
// general solver.
class BoundCoreFaceSolver {
 public:
  explicit BoundCoreFaceSolver(const Fixture &fixture,
                               std::vector<double> target_shift = {});

  bool structurally_eligible() const { return structurally_eligible_; }
  bool solve(const std::vector<std::uint32_t> &rows, FaceSolution &solution);
  const BoundCoreSolveStats &stats() const { return stats_; }

 private:
  const Fixture &fixture_;
  std::vector<double> target_shift_;
  std::vector<std::int32_t> unit_column_;
  std::vector<double> unit_value_;
  int general_row_count_ = 0;
  bool structurally_eligible_ = false;
  BoundCoreSolveStats stats_;
};

}  // namespace twalker
