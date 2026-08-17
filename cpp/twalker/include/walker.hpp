#pragma once

#include "bound_core_face_solver.hpp"
#include "face_solver.hpp"
#include "fixture.hpp"
#include "gram_face_solver.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace twalker {

struct CertificateErrors {
  double primal = 0.0;
  double dual = 0.0;
  double nonnegative = 0.0;
  double gap = 0.0;
};

// Shadow-only census for objective-preserving moves within the current dual
// objective-level face.  No value here influences the production walk.
struct RankLiftAuditStats {
  int global_rank = -1;
  int structurally_maximal_faces = 0;
  int audits = 0;
  int lp_solves = 0;
  int direction_successes = 0;
  int rank_gains = 0;
  int full_rank = 0;
  int weak_steps = 0;
  int activated_rows = 0;
  int source_dense_fallbacks = 0;
  int candidate_dense_fallbacks = 0;
  int live_applied = 0;
  std::int64_t total_rank_gain = 0;
  int max_rank_gain = 0;
  double lp_ms = 0.0;
  double global_factor_ms = 0.0;
  double factor_ms = 0.0;
  double max_dual_direction_residual = 0.0;
  double max_objective_direction_residual = 0.0;
  double max_objective_drift = 0.0;
  double live_shift_inf = 0.0;
  double live_shift_relative = 0.0;
};

// Quarantined shadow comparison for the native row-space updater.  These
// counters never influence support, t, or certification.
struct MaintainedRowspaceAuditStats {
  int checks = 0;
  int matches = 0;
  int seed_attempts = 0;
  int seed_successes = 0;
  int false_admissions = 0;
  std::uint64_t local_transitions = 0;
  std::uint64_t additions = 0;
  std::uint64_t removals = 0;
  std::uint64_t rank_increases = 0;
  std::uint64_t rank_decreases = 0;
  std::uint64_t rank_change_declines = 0;
  std::uint64_t numerical_declines = 0;
  std::uint64_t refactors = 0;
  double seed_ms = 0.0;
  double transition_ms = 0.0;
  double solve_ms = 0.0;
  double max_g_error = 0.0;
  double max_ua_error = 0.0;
  double max_bua_error = 0.0;
  double worst_rowspace_residual = 0.0;
  double worst_orthogonality = 0.0;
  double worst_slope_residual = 0.0;
  int cold_reveals = 0;
  double cold_reveal_ms = 0.0;
};

struct EventIntervalStats {
  int checks = 0;
  int certified = 0;
  int no_central_group = 0;
  int tie_disabled = 0;
  int winner_interval = 0;
  int tie_width = 0;
  int horizon = 0;
  int competitor_interval = 0;
  int competitor_margin = 0;
  int sticky_basic_zero_certified = 0;
  int uncertain_slope = 0;
  int final_cap = 0;
};

struct WalkResult {
  std::string status;
  bool native_seed = false;
  bool highs_seed = false;
  bool seed_converged = false;
  std::string seed_route;
  std::vector<std::uint8_t> seed_mask;
  int seed_iterations = 0;
  int seed_support = 0;
  double seed_ms = 0.0;
  double seed_dres = 0.0;
  int pivots = 0;
  double t = 0.0;
  std::vector<double> x, y;
  CertificateErrors certificate;
  int face_solves = 0;
  int dense_fallbacks = 0;
  int rank_deficient_solves = 0;
  std::int64_t rank_deficit_sum = 0;
  int max_rank_deficit = 0;
  int settle_rounds = 0;
  int tied_events = 0;
  int epsilon_repairs = 0;
  int tie_subset_repairs = 0;
  int corrector_repairs = 0;
  int qp_repairs = 0;
  int gram_fast_solves = 0;
  int gram_declines = 0;
  int bound_core_solves = 0;
  int bound_core_declines = 0;
  int bound_core_maximum_border = 0;
  int bound_core_refinements = 0;
  double bound_core_total_ms = 0.0;
  int bound_core_audits = 0;
  int bound_core_audit_violations = 0;
  double bound_core_audit_max_error = 0.0;
  int revised_column_solves = 0;
  int revised_column_declines = 0;
  int revised_column_rebuilds = 0;
  int revised_column_rank_changes = 0;
  int revised_column_retirements = 0;
  double revised_column_rebuild_ms = 0.0;
  double revised_column_seed_ms = 0.0;
  double revised_column_transition_ms = 0.0;
  double revised_column_solve_ms = 0.0;
  double revised_column_products_ms = 0.0;
  double revised_column_coefficient_ms = 0.0;
  double revised_column_projection_ms = 0.0;
  double revised_column_residual_ms = 0.0;
  int revised_column_direct_seeds = 0;
  int revised_bound_audits = 0;
  int revised_bound_violations = 0;
  double revised_bound_worst_ratio = 0.0;
  double revised_bound_max_width = 0.0;
  double revised_bound_max_actual = 0.0;
  int stability_refactors = 0;
  int settle_stability_refactors = 0;
  int terminal_stability_refactors = 0;
  int event_stability_refactors = 0;
  EventIntervalStats event_interval;
  int qr_settle_decision_checks = 0;
  int qr_settle_decision_matches = 0;
  int qr_event_decision_checks = 0;
  int qr_event_decision_matches = 0;
  int qr_event_tie_set_matches = 0;
  int retained_tie_polishes = 0;
  int retained_tie_certificates = 0;
  int retained_tie_declines = 0;
  double gram_final_rcond = 0.0;
  GramSolveStats gram_stats;
  int selector_calls = 0;
  double selector_ms = 0.0;
  int terminal_gates = 0;
  int recovery_calls = 0;
  double recovery_ms = 0.0;
  int terminal_support_repairs = 0;
  int terminal_support_repair_successes = 0;
  std::int64_t terminal_support_repair_iterations = 0;
  double terminal_support_repair_ms = 0.0;
  int qp_calls = 0;
  std::int64_t qp_iterations = 0;
  double qp_ms = 0.0;
  int rank_complete_calls = 0;
  int rank_complete_repairs = 0;
  double rank_complete_ms = 0.0;
  int critical_right_calls = 0;
  int critical_right_repairs = 0;
  double critical_right_ms = 0.0;
  RankLiftAuditStats rank_lift_audit;
  MaintainedRowspaceAuditStats maintained_rowspace_audit;
  FaceSolveStats direct_stats;
};

class Walker {
 public:
  explicit Walker(const Fixture &fixture, bool enable_face_cache = true,
                  double gram_min_rcond = 1e-8,
                  double target_nudge_epsilon = 0.0,
                  bool allow_revised_column = true,
                  bool enable_native_seed = true);
  ~Walker();
  WalkResult run(int max_pivots = 2000, double tmax = 1e8);

 private:
  const Fixture &fixture_;
  bool enable_native_seed_ = true;
  std::vector<double> target_shift_;
  BoundCoreFaceSolver bound_core_solver_;
  GramFaceSolver gram_solver_;
  FaceSolver solver_;
  int face_solves_ = 0;
  int dense_fallbacks_ = 0;
  int rank_deficient_solves_ = 0;
  std::int64_t rank_deficit_sum_ = 0;
  int max_rank_deficit_ = 0;
  int settle_rounds_ = 0;
  int gram_fast_solves_ = 0;
  int gram_declines_ = 0;
  int bound_core_solves_ = 0;
  int bound_core_declines_ = 0;
  int bound_core_audits_ = 0;
  int bound_core_audit_violations_ = 0;
  double bound_core_audit_max_error_ = 0.0;
  int revised_column_solves_ = 0;
  int revised_column_declines_ = 0;
  int revised_bound_audits_ = 0;
  int revised_bound_violations_ = 0;
  double revised_bound_worst_ratio_ = 0.0;
  double revised_bound_max_width_ = 0.0;
  double revised_bound_max_actual_ = 0.0;
#ifdef TWALKER_ENABLE_REVISED_COLUMN
  void *revised_column_solver_ = nullptr;
  void *centered_slope_solver_ = nullptr;
  void *maintained_rowspace_solver_ = nullptr;
  void *maintained_deficient_qr_solver_ = nullptr;
  void *deficient_recurrence_solver_ = nullptr;
  void *maintained_svd_face_solver_ = nullptr;
  FaceSolver *maintained_rowspace_seed_solver_ = nullptr;
  void *augmented_kkt_basis_ = nullptr;
  int augmented_kkt_audit_remaining_ = 0;
  int augmented_kkt_audit_consecutive_ = 0;
  double augmented_kkt_last_t_ = -1.0;
  std::uint64_t augmented_kkt_last_fingerprint_ = 0;
  bool revised_column_primary_ = false;
#endif
  bool gram_retired_ = false;
  // A verified full-rank direct face may re-arm the maintained Gram lane
  // once after a genuine numerical/rank decline.  A second genuine decline
  // is permanent, preventing rebuild churn on unsuitable paths.  Direct
  // audits of discrete tie decisions may re-arm the still-valid factor.
  bool gram_rearm_used_ = false;
  bool gram_stability_rearm_pending_ = false;
  bool extension_enabled_ = false;
  bool extension_used_ = false;
  bool trace_path_ = false;
  int stability_refactors_ = 0;
  int settle_stability_refactors_ = 0;
  int terminal_stability_refactors_ = 0;
  int event_stability_refactors_ = 0;
  EventIntervalStats event_interval_;
  int qr_settle_decision_checks_ = 0;
  int qr_settle_decision_matches_ = 0;
  int qr_event_decision_checks_ = 0;
  int qr_event_decision_matches_ = 0;
  int qr_event_tie_set_matches_ = 0;
  int retained_tie_polishes_ = 0;
  int retained_tie_certificates_ = 0;
  int retained_tie_declines_ = 0;
  int selector_calls_ = 0;
  double selector_ms_ = 0.0;
  std::vector<int> selector_col_status_;
  std::vector<int> selector_row_status_;
  std::vector<int> selector_basic_variables_;
  std::uint64_t selector_basis_fingerprint_ = 0;
  std::uint64_t selector_previous_basis_fingerprint_ = 0;
  int selector_basis_snapshots_ = 0;
  int selector_basis_repeats_ = 0;
  std::vector<int> horizon_col_status_;
  std::vector<int> horizon_row_status_;
  std::vector<int> horizon_basic_variables_;
  std::uint64_t horizon_basis_fingerprint_ = 0;
  std::uint64_t horizon_previous_basis_fingerprint_ = 0;
  int horizon_basis_snapshots_ = 0;
  int horizon_basis_repeats_ = 0;
  int terminal_gates_ = 0;
  int recovery_calls_ = 0;
  double recovery_ms_ = 0.0;
  int terminal_support_repairs_ = 0;
  int terminal_support_repair_successes_ = 0;
  std::int64_t terminal_support_repair_iterations_ = 0;
  double terminal_support_repair_ms_ = 0.0;
  std::vector<int> terminal_support_col_status_;
  std::vector<int> terminal_support_row_status_;
  void *selector_ = nullptr;
  void *horizon_selector_ = nullptr;
  void *primal_selector_ = nullptr;
  void *terminal_support_selector_ = nullptr;
  void *osqp_solver_ = nullptr;
  void *rank_lift_highs_ = nullptr;
  FaceSolver *rank_lift_solver_ = nullptr;
  FaceSolver *rank_lift_live_solver_ = nullptr;
  std::vector<double> rank_lift_row_weight_;
  bool rank_lift_audit_enabled_ = false;
  bool rank_lift_live_requested_ = false;
  int rank_lift_audit_max_faces_ = 24;
  RankLiftAuditStats rank_lift_audit_;
  bool maintained_rowspace_audit_enabled_ = false;
  bool maintained_rowspace_trace_ = false;
  bool deficient_face_audit_enabled_ = false;
  bool deficient_face_live_enabled_ = false;
  bool maintained_svd_audit_enabled_ = false;
  bool maintained_svd_live_enabled_ = false;
  int deficient_qr_seed_cooldown_ = 0;
  MaintainedRowspaceAuditStats maintained_rowspace_audit_;
  std::vector<std::uint8_t> rank_lift_candidate_support_;
  std::vector<double> rank_lift_candidate_y_;
  int qp_calls_ = 0;
  std::int64_t qp_iterations_ = 0;
  double qp_ms_ = 0.0;
  int rank_complete_calls_ = 0;
  int rank_complete_repairs_ = 0;
  double rank_complete_ms_ = 0.0;
  int critical_right_calls_ = 0;
  int critical_right_repairs_ = 0;
  double critical_right_ms_ = 0.0;
  // OBNR-1 quarantine: ordered basis/status fingerprints are retained across
  // repeated repair calls at one fixed t, so a later fallback cannot revisit
  // a representation already rejected at that endpoint.
  bool ordered_basis_t_valid_ = false;
  double ordered_basis_t_ = 0.0;
  std::vector<std::uint64_t> ordered_basis_fingerprints_;
  // Endpoint rank completion produces exactly the information needed for a
  // simplex-style crossover: a full-rank set of tight rows and a multiplier
  // representing the accepted projection.  Preserve it even when the generic
  // min-norm face reconstruction rejects the completed support.
  std::vector<std::uint8_t> completed_tight_support_;
  std::vector<double> completed_endpoint_u_;
  mutable std::string last_reject_;
  bool native_seed_converged_ = false;
  int native_seed_iterations_ = 0;
  int native_seed_support_ = 0;
  double native_seed_ms_ = 0.0;
  double native_seed_dres_ = 0.0;
  std::string native_seed_route_;

  bool native_newton_seed(std::vector<std::uint8_t> &support);
  bool native_newton_seed(std::vector<std::uint8_t> &support,
                          const std::vector<double> &initial_multiplier,
                          const std::string &route);
  bool kernel_projection_seed(std::vector<std::uint8_t> &support);
  bool triangular_projection_seed(std::vector<std::uint8_t> &support);
  bool highs_projection_seed(std::vector<std::uint8_t> &support);
  bool settle(std::vector<std::uint8_t> &support, double t,
              FaceSolution &solution, int rounds = 20,
              bool one_at_a_time = false);
  bool accept(const std::vector<std::uint8_t> &support, double t,
              const FaceSolution &solution);
  bool selector_feasible(const std::vector<std::uint8_t> &support, double t,
                         const std::vector<double> &face_y,
                         std::vector<double> *selected_u = nullptr);
  bool selector_affine_feasible(
      const std::vector<std::uint8_t> &support,
      const std::vector<double> &active_value,
      const std::vector<double> &offset,
      const std::vector<std::uint8_t> &selector_constraints,
      std::vector<double> *selected_u = nullptr,
      std::vector<double> *selected_product = nullptr,
      std::uint64_t *basis_fingerprint = nullptr,
      double equality_relative_band = 0.0);
  bool selector_maximize_forward_interval(
      const std::vector<std::uint8_t> &support,
      const std::vector<double> &endpoint_y,
      const std::vector<double> &endpoint_slack,
      const std::vector<double> &direction, double minimum_forward,
      double delta_cap, std::vector<double> &selected_ua,
      std::vector<double> &selected_bua, double &selected_delta,
      std::uint64_t *basis_fingerprint = nullptr);
  bool primal_face_candidate(const std::vector<double> &y,
                             std::vector<double> &x);
  FaceSolution solve(const std::vector<std::uint8_t> &support);
  FaceSolution solve_direct(const std::vector<std::uint8_t> &support,
                            bool force_refactor = false);
  FaceSolution solve_centered_slope(
      const std::vector<std::uint8_t> &support);
  bool audit_augmented_kkt_state(
      const std::vector<std::uint8_t> &support, double t,
      const FaceSolution &solution);
  bool audit_maintained_rowspace(
      const std::vector<std::uint8_t> &support,
      const FaceSolution &oracle);
  void audit_bound_core(const std::vector<std::uint32_t> &rows,
                        const FaceSolution &oracle);
  void record_face_rank(const FaceSolution &solution);
  void coefficient_product_bounds(const FaceSolution &solution,
                                  std::vector<double> &ua_error,
                                  std::vector<double> &uc_error) const;
  bool affine_product_bounds(const FaceSolution &solution, double t,
                             std::vector<double> &error) const;
  bool settle_decision_stable(const std::vector<std::uint8_t> &support,
                              double t, const FaceSolution &solution,
                              const std::vector<double> &y,
                              const std::vector<double> &q,
                              double y_scale, double q_scale,
                              bool one_at_a_time) const;
  bool event_decision_stable(double t, double tmax,
                             const FaceSolution &solution,
                             const std::vector<double> &slope,
                             const std::vector<double> &constant,
                             const std::vector<double> &candidate,
                             double &next,
                             bool *retained_tie_certificate = nullptr);
  void audit_qr_settle_decision(
      const std::vector<std::uint8_t> &support, double t,
      const FaceSolution &solution, bool one_at_a_time,
      const std::vector<std::uint8_t> &authoritative_next);
  void audit_qr_event_decision(
      double t, double tmax, const FaceSolution &solution, double next,
      const std::vector<std::size_t> &authoritative_ties);
  CertificateErrors certificate_pair(const std::vector<double> &x,
                                     const std::vector<double> &y) const;
  bool certificate_passes(const CertificateErrors &errors) const;
  bool recover_certificate(const std::vector<double> &y,
                           std::vector<double> &x,
                           CertificateErrors &errors);
  bool terminal_support_repair(const std::vector<std::uint8_t> &support,
                               std::vector<double> &y,
                               std::vector<double> &x,
                               CertificateErrors &errors);
  std::vector<std::uint8_t> corrector_support(double t) const;
  std::vector<double> qp_projection(double t);
  bool rank_complete_projection(double t,
                                const FaceSolution &endpoint,
                                std::vector<std::uint8_t> &support,
                                FaceSolution &solution);
  bool critical_right_transition(double t,
                                 const FaceSolution &endpoint,
                                 std::vector<std::uint8_t> &support,
                                 FaceSolution &solution,
                                 double &advanced_t);
  void audit_objective_preserving_rank_lift(
      double t, const FaceSolution &solution);
  bool apply_objective_preserving_rank_lift(
      std::vector<std::uint8_t> &support, double t,
      FaceSolution &solution);
};

}  // namespace twalker
