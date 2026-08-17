#pragma once

#include "face_solver.hpp"
#include "revised_basis_solver.hpp"

#include <cholmod.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace twalker::revised {

struct RevisedColumnStats {
  std::uint64_t calls = 0;
  std::uint64_t rebuilds = 0;
  std::uint64_t local_transitions = 0;
  std::uint64_t unchanged_reuses = 0;
  std::uint64_t row_additions = 0;
  std::uint64_t row_removals = 0;
  std::uint64_t rank_changes = 0;
  std::uint64_t rank_increases = 0;
  std::uint64_t rank_decreases = 0;
  std::uint64_t update_failures = 0;
  std::uint64_t condition_declines = 0;
  std::uint64_t residual_declines = 0;
  std::uint64_t refinements = 0;
  std::uint64_t declines = 0;
  std::uint64_t retirements = 0;
  std::uint64_t active_refreshes = 0;
  std::uint64_t direct_seeds = 0;
  double rebuild_ms = 0.0;
  double transition_ms = 0.0;
  double condition_ms = 0.0;
  double rank_test_ms = 0.0;
  double factor_update_ms = 0.0;
  double solve_ms = 0.0;
  double products_ms = 0.0;
  double coefficient_ms = 0.0;
  double projection_ms = 0.0;
  double residual_ms = 0.0;
  double active_refresh_ms = 0.0;
  double direct_seed_ms = 0.0;
  double worst_entering_representation = 0.0;
};

// Centered path data for A=B[rows,:].  Unlike RevisedFaceSolution this does
// not contain the global affine intercept h=y-t*g.  Long paths make that
// intercept a numerically hostile representation; event progression needs
// only the least-squares slope
//
//   ua = argmin ||A*ua+b_rows||,  g=b_rows+A*ua,
//
// and B*ua for the inactive projection-slack derivatives.  Keeping this
// object separate also lets a rank-changing row update be audited and
// reseeded without invalidating the accepted centered endpoint.
struct RevisedSlopeSolution {
  std::vector<double> g, ua, bua;
  std::vector<std::uint32_t> rows;
  std::int64_t rank = 0;
  double slope_residual = 0.0;
};

// For A=B[rows,:], retain independent columns C=A[:,J] and T with A=C*T.
// A support-row add/drop leaves T unchanged whenever rank is unchanged and
// changes C'C by one rank-one term.  This is the support-pivot orientation
// analogous to a revised-simplex basis update.
class RevisedColumnSolver {
 public:
  explicit RevisedColumnSolver(const Fixture &fixture,
                               std::vector<double> target_shift = {},
                               bool centered_slope_mode = false,
                               bool force_shared_recurrence = false);
  ~RevisedColumnSolver();
  RevisedColumnSolver(const RevisedColumnSolver &) = delete;
  RevisedColumnSolver &operator=(const RevisedColumnSolver &) = delete;

  bool solve(const std::vector<std::uint32_t> &rows,
             RevisedFaceSolution &solution);
  bool seed_from_direct(const std::vector<std::uint32_t> &rows,
                        FaceSolution &direct_solution);
  bool solve_slope(const std::vector<std::uint32_t> &rows,
                   RevisedSlopeSolution &solution);
  bool seed_slope_from_direct(const std::vector<std::uint32_t> &rows,
                              FaceSolution &direct_solution);
  bool needs_direct_seed() const;
  bool needs_recurrence_seed() const {
    return coefficient_recurrence_ && shared_direct_seed_
           && needs_direct_seed();
  }
  bool needs_factored_seed() const {
    return factored_direct_seed_ && needs_direct_seed();
  }
  const RevisedColumnStats &stats() const { return stats_; }

 private:
  const Fixture &fixture_;
  std::vector<double> target_shift_;
  std::vector<std::uint32_t> rows_;
  std::vector<std::uint32_t> basis_columns_;
  std::vector<std::int32_t> basis_position_;
  std::vector<double> transform_;          // rank-by-m, column-major
  std::vector<double> coordinates_;        // active-by-rank, row-major
  std::vector<double> active_factor_;      // chol(C'C)
  std::vector<double> transform_factor_;   // chol(TT')
  std::vector<double> active_cross_b_;     // C'b
  std::vector<double> fixed_target_head_;  // (TT')^-1 T d, rank-epoch constant
  std::vector<double> representation_transform_; // T in A=C*T
  std::vector<double> rank_test_transform_; // orthonormal row basis V
  std::vector<std::uint32_t> dependent_columns_;
  std::vector<double> dependency_transform_; // D in T=[I,D], rank-by-k
  std::vector<double> dependency_factor_; // chol(I+D'D), k-by-k
  double projection_inf_norm_ = 0.0;       // ||T'(TT')^-1||inf
  cholmod_common common_{};
  cholmod_factor *reduced_factor_ = nullptr;
  cholmod_sparse *reduced_update_column_ = nullptr;
  std::vector<std::int64_t> reduced_inverse_perm_;
  std::vector<std::pair<std::int64_t, double>> reduced_update_entries_;
  // Fixed rank-epoch diagonal S for the mathematically equivalent system
  // (S C'C S)y=S rhs, x=S y.  Power-of-two entries avoid introducing
  // rounding in the scaling itself; original-operator residuals remain the
  // acceptance authority.
  std::vector<double> reduced_scale_;
  double reduced_rcond_ = 0.0;
  RevisedColumnStats stats_;
  bool valid_ = false;
  bool retired_ = false;
  bool persistent_rank_updates_ = false;
  bool coefficient_recurrence_ = false;
  bool shared_direct_seed_ = false;
  bool factored_direct_seed_ = false;
  bool centered_slope_mode_ = false;
  bool forced_shared_recurrence_ = false;
  bool orthonormal_factored_ = false;
  bool reduced_sparse_factored_ = false;
  bool reduced_equilibration_ = false;
  bool shadow_square_root_ = false;
  bool shadow_solve_only_ = false;
  bool rz_orthonormal_ = false;
  bool svd_orthonormal_ = false;
  bool direct_seeded_ = false;
  std::uint32_t direct_seed_count_ = 0;
  std::uint32_t max_direct_seeds_ = 16;
  std::uint32_t maximum_seed_deficiency_ = 0;
  std::uint64_t successful_faces_ = 0;
  std::int64_t recurrence_rank_ = 0;
  std::uint64_t recurrence_updates_ = 0;
  std::uint32_t recurrence_rebases_ = 0;
  std::vector<double> pseudoinverse_;      // m-by-active, column-major
  // Orthonormal row-space basis, rank-by-m column-major.  Centered rank
  // increases use this stable projector instead of A+, whose large entries
  // can destroy the small entering residual by cancellation.
  std::vector<double> recurrence_row_space_;
  std::vector<double> recurrence_ua_;
  // The dense oracle obtains this null-space slope from its orthogonal U
  // factors.  Retain that accurate vector and update it under Greville row
  // exchanges; recomputing b+A*ua through an ill-conditioned A+ discards the
  // very accuracy the shared seed was intended to preserve.
  std::vector<double> recurrence_g_;
  std::vector<double> recurrence_gamma_;
  std::vector<double> recurrence_uc_;
  bool recurrence_coefficients_valid_ = false;
  std::string last_update_failure_;

  bool rebuild(const std::vector<std::uint32_t> &rows);
  bool transition(const std::vector<std::uint32_t> &rows);
  bool form_solution(RevisedFaceSolution &solution);
  bool row_coordinates(std::uint32_t row, std::vector<double> &coordinates,
                       std::vector<double> &residual,
                       double &relative_residual) const;
  bool add_row(std::uint32_t row);
  bool remove_row(std::uint32_t row);
  bool refresh_active_factor();
  bool build_pseudoinverse();
  bool transition_recurrence(const std::vector<std::uint32_t> &rows);
  bool add_row_recurrence(std::uint32_t row);
  bool remove_row_recurrence(std::uint32_t row);
  bool form_recurrence_solution(RevisedFaceSolution &solution);
  bool form_slope_solution(RevisedSlopeSolution &solution);
  bool build_reduced_factor();
  bool update_reduced_factor(const std::vector<double> &coordinate, bool add);
  bool can_reseed_epoch() const;
  bool can_request_direct_seed() const;
};

}  // namespace twalker::revised
