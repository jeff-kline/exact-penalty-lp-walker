#pragma once

#include "fixture.hpp"

#include <cholmod.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace twalker::revised {

struct RevisedFaceSolution {
  std::vector<double> g, h, ua, uc, bua, buc;
  std::vector<std::uint32_t> rows;
  std::int64_t rank = 0;
  double dres = 0.0;
  double piece_residual = 0.0;
  double basis_diagonal_ratio = 0.0;
  double coordinate_diagonal_ratio = 0.0;
  bool forward_bound_valid = false;
  double ua_relative_error_bound = 0.0;
  double uc_relative_error_bound = 0.0;
  std::vector<double> reduced_residual_a, reduced_residual_c;
  std::vector<double> reduced_head_a, reduced_head_c;
  double reduced_gram_inf = 0.0;
  double reduced_rcond = 0.0;
  double projection_inf_norm = 0.0;
};

struct RevisedBasisStats {
  std::uint64_t calls = 0;
  std::uint64_t rebuilds = 0;
  std::uint64_t unchanged_reuses = 0;
  std::uint64_t local_transitions = 0;
  std::uint64_t row_additions = 0;
  std::uint64_t row_removals = 0;
  std::uint64_t rank_changes = 0;
  std::uint64_t basis_removals = 0;
  std::uint64_t basis_exchanges = 0;
  std::uint64_t update_failures = 0;
  std::uint64_t condition_declines = 0;
  std::uint64_t residual_declines = 0;
  std::uint64_t declines = 0;
  double rebuild_ms = 0.0;
  double transition_ms = 0.0;
  double solve_ms = 0.0;
  double products_ms = 0.0;
  double worst_row_reconstruction = 0.0;
};

// Quarantined prototype of a revised-simplex-style face state.
//
// For A = B[rows,:], retain independent support rows C and coordinates L such
// that A = L C.  The two positive-definite cores C C' and L' L are factored.
// Dependent-row insertion/deletion is therefore a rank-one update of L' L;
// SPQR is not required on those ordinary transitions.  Rank changes and basis
// removals deliberately rebuild in this first probe.
class RevisedBasisSolver {
 public:
  explicit RevisedBasisSolver(const Fixture &fixture,
                              std::vector<double> target_shift = {});
  ~RevisedBasisSolver();
  RevisedBasisSolver(const RevisedBasisSolver &) = delete;
  RevisedBasisSolver &operator=(const RevisedBasisSolver &) = delete;

  bool solve(const std::vector<std::uint32_t> &rows,
             RevisedFaceSolution &solution);
  const RevisedBasisStats &stats() const { return stats_; }

 private:
  const Fixture &fixture_;
  std::vector<double> target_shift_;
  std::vector<std::uint32_t> rows_;
  std::vector<std::uint32_t> basis_rows_;
  std::vector<double> basis_factor_;      // lower Cholesky of C C'
  std::vector<double> coordinate_factor_; // lower Cholesky of L' L
  std::unordered_map<std::uint32_t, std::vector<double>> coordinates_;
  cholmod_common common_{};
  RevisedBasisStats stats_;
  bool valid_ = false;

  bool rebuild(const std::vector<std::uint32_t> &rows);
  bool transition(const std::vector<std::uint32_t> &rows);
  bool form_solution(RevisedFaceSolution &solution);
  bool refactor_from_coordinates();
  bool row_coordinates(std::uint32_t row, std::vector<double> &coordinates,
                       double &relative_residual) const;
  double sparse_row_dot(std::uint32_t left, std::uint32_t right) const;
  double sparse_row_norm2(std::uint32_t row) const;
};

double relative_inf_error(const std::vector<double> &actual,
                          const std::vector<double> &expected);

}  // namespace twalker::revised
