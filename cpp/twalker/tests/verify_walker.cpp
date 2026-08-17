#include "fixture.hpp"
#include "walker.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
std::string stem(const std::string &path) {
  const auto slash = path.find_last_of('/');
  const auto dot = path.find_last_of('.');
  const auto begin = slash == std::string::npos ? 0 : slash + 1;
  const auto end = dot == std::string::npos || dot < begin ? path.size() : dot;
  return path.substr(begin, end - begin);
}

void print_vector(const std::vector<double> &values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

void print_json_number(double value) {
  if (std::isfinite(value))
    std::cout << value;
  else
    std::cout << "null";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: verify_walker fixture.twfx...\n";
    return 2;
  }
  bool good = true;
  const bool enable_face_cache = std::getenv("TWALKER_DISABLE_FACE_CACHE") == nullptr;
  const bool emit_solution = std::getenv("TWALKER_EMIT_SOLUTION") != nullptr;
  const bool speculative_restart =
      std::getenv("TWALKER_DISABLE_REVISED_COLUMN") == nullptr
      && std::getenv("TWALKER_DISABLE_SPECULATIVE_RESTART") == nullptr;
  const bool force_speculative_restart =
      std::getenv("TWALKER_FORCE_SPECULATIVE_RESTART") != nullptr;
  double gram_min_rcond = 1e-8;
  if (const char *raw = std::getenv("TWALKER_GRAM_MIN_RCOND"))
    gram_min_rcond = std::stod(raw);
  double nudge_epsilon = 0.0;
  if (const char *raw = std::getenv("TWALKER_NUDGE_EPS"))
    nudge_epsilon = std::stod(raw);
  int max_pivots = 2000;
  if (const char *raw = std::getenv("TWALKER_MAX_PIVOTS"))
    max_pivots = std::stoi(raw);
  double tmax = 1e8;
  if (const char *raw = std::getenv("TWALKER_TMAX"))
    tmax = std::stod(raw);
  std::cout << std::setprecision(17);
  for (int i = 1; i < argc; ++i) {
    const std::string path = argv[i];
    try {
      const auto fixture = twalker::read_fixture(path);
      const auto start = std::chrono::steady_clock::now();
      auto execute = [&](const twalker::Fixture &input,
                         bool allow_revised_column,
                         bool enable_native_seed) {
        twalker::Walker walker(input, enable_face_cache, gram_min_rcond,
                               nudge_epsilon, allow_revised_column,
                               enable_native_seed);
        return walker.run(max_pivots, tmax);
      };
      auto result = execute(fixture, true, true);
      int speculative_restarts = 0;
      const bool revised_answer_influenced_walk =
          result.revised_column_solves > 0;
      if (speculative_restart
          && (result.status != "CERTIFIED" || force_speculative_restart)
          && (revised_answer_influenced_walk || force_speculative_restart)) {
        const auto seed_record = result;
        auto seeded_fixture = fixture;
        const bool generated_seed =
            seed_record.native_seed || seed_record.highs_seed;
        if (generated_seed && !seed_record.seed_mask.empty()) {
          seeded_fixture.post_seed_support = seed_record.seed_mask;
          seeded_fixture.t0 = 0.0;
        }
        result = execute(seeded_fixture, false, false);
        if (generated_seed) {
          result.native_seed = seed_record.native_seed;
          result.highs_seed = seed_record.highs_seed;
          result.seed_converged = seed_record.seed_converged;
          result.seed_route = seed_record.seed_route;
          result.seed_iterations = seed_record.seed_iterations;
          result.seed_support = seed_record.seed_support;
          result.seed_ms = seed_record.seed_ms;
          result.seed_dres = seed_record.seed_dres;
          result.seed_mask = seed_record.seed_mask;
        }
        speculative_restarts = 1;
      }
      const double milliseconds = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
      good = good && result.status == "CERTIFIED";
      std::cout << "{\"model\":\"" << stem(path) << "\",\"status\":\""
                << result.status << "\",\"native_seed\":"
                << (result.native_seed ? "true" : "false")
                << ",\"highs_seed\":"
                << (result.highs_seed ? "true" : "false")
                << ",\"seed_converged\":"
                << (result.seed_converged ? "true" : "false")
                << ",\"seed_route\":\"" << result.seed_route << "\""
                << ",\"seed_iterations\":" << result.seed_iterations
                << ",\"seed_support\":" << result.seed_support
                << ",\"seed_ms\":" << result.seed_ms
                << ",\"seed_dres\":";
      print_json_number(result.seed_dres);
      std::cout << ",\"pivots\":" << result.pivots
                << ",\"t\":" << result.t << ",\"face_solves\":"
                << result.face_solves << ",\"dense_fallbacks\":"
                << result.dense_fallbacks << ",\"rank_deficient_solves\":"
                << result.rank_deficient_solves
                << ",\"rank_deficit_sum\":" << result.rank_deficit_sum
                << ",\"max_rank_deficit\":" << result.max_rank_deficit
                << ",\"settle_rounds\":"
                << result.settle_rounds << ",\"tied_events\":"
                << result.tied_events << ",\"wall_ms\":" << milliseconds
                << ",\"speculative_restarts\":" << speculative_restarts
                << ",\"epsilon_repairs\":" << result.epsilon_repairs
                << ",\"tie_subset_repairs\":" << result.tie_subset_repairs
                << ",\"corrector_repairs\":" << result.corrector_repairs
                << ",\"qp_repairs\":" << result.qp_repairs
                << ",\"gram_fast_solves\":" << result.gram_fast_solves
                << ",\"gram_declines\":" << result.gram_declines
                << ",\"bound_core_solves\":" << result.bound_core_solves
                << ",\"bound_core_declines\":" << result.bound_core_declines
                << ",\"bound_core_maximum_border\":"
                << result.bound_core_maximum_border
                << ",\"bound_core_refinements\":"
                << result.bound_core_refinements
                << ",\"bound_core_total_ms\":"
                << result.bound_core_total_ms
                << ",\"bound_core_audits\":" << result.bound_core_audits
                << ",\"bound_core_audit_violations\":"
                << result.bound_core_audit_violations
                << ",\"bound_core_audit_max_error\":"
                << result.bound_core_audit_max_error
                << ",\"revised_column_solves\":"
                << result.revised_column_solves
                << ",\"revised_column_declines\":"
                << result.revised_column_declines
                << ",\"revised_column_rebuilds\":"
                << result.revised_column_rebuilds
                << ",\"revised_column_rank_changes\":"
                << result.revised_column_rank_changes
                << ",\"revised_column_retirements\":"
                << result.revised_column_retirements
                << ",\"revised_column_rebuild_ms\":"
                << result.revised_column_rebuild_ms
                << ",\"revised_column_seed_ms\":"
                << result.revised_column_seed_ms
                << ",\"revised_column_transition_ms\":"
                << result.revised_column_transition_ms
                << ",\"revised_column_solve_ms\":"
                << result.revised_column_solve_ms
                << ",\"revised_column_products_ms\":"
                << result.revised_column_products_ms
                << ",\"revised_column_coefficient_ms\":"
                << result.revised_column_coefficient_ms
                << ",\"revised_column_projection_ms\":"
                << result.revised_column_projection_ms
                << ",\"revised_column_residual_ms\":"
                << result.revised_column_residual_ms
                << ",\"revised_column_direct_seeds\":"
                << result.revised_column_direct_seeds
                << ",\"revised_bound_audits\":"
                << result.revised_bound_audits
                << ",\"revised_bound_violations\":"
                << result.revised_bound_violations
                << ",\"revised_bound_worst_ratio\":"
                << result.revised_bound_worst_ratio
                << ",\"revised_bound_max_width\":"
                << result.revised_bound_max_width
                << ",\"revised_bound_max_actual\":"
                << result.revised_bound_max_actual
                << ",\"stability_refactors\":"
                << result.stability_refactors
                << ",\"settle_stability_refactors\":"
                << result.settle_stability_refactors
                << ",\"terminal_stability_refactors\":"
                << result.terminal_stability_refactors
                << ",\"event_stability_refactors\":"
                << result.event_stability_refactors
                << ",\"event_interval\":{\"checks\":"
                << result.event_interval.checks
                << ",\"certified\":" << result.event_interval.certified
                << ",\"no_central_group\":"
                << result.event_interval.no_central_group
                << ",\"tie_disabled\":"
                << result.event_interval.tie_disabled
                << ",\"winner_interval\":"
                << result.event_interval.winner_interval
                << ",\"tie_width\":" << result.event_interval.tie_width
                << ",\"horizon\":" << result.event_interval.horizon
                << ",\"competitor_interval\":"
                << result.event_interval.competitor_interval
                << ",\"competitor_margin\":"
                << result.event_interval.competitor_margin
                << ",\"sticky_basic_zero_certified\":"
                << result.event_interval.sticky_basic_zero_certified
                << ",\"uncertain_slope\":"
                << result.event_interval.uncertain_slope
                << ",\"final_cap\":" << result.event_interval.final_cap
                << "}"
                << ",\"qr_settle_decision_checks\":"
                << result.qr_settle_decision_checks
                << ",\"qr_settle_decision_matches\":"
                << result.qr_settle_decision_matches
                << ",\"qr_event_decision_checks\":"
                << result.qr_event_decision_checks
                << ",\"qr_event_decision_matches\":"
                << result.qr_event_decision_matches
                << ",\"qr_event_tie_set_matches\":"
                << result.qr_event_tie_set_matches
                << ",\"retained_tie_polishes\":"
                << result.retained_tie_polishes
                << ",\"retained_tie_certificates\":"
                << result.retained_tie_certificates
                << ",\"retained_tie_declines\":"
                << result.retained_tie_declines
                << ",\"gram_final_rcond\":" << result.gram_final_rcond
                << ",\"gram_guarded_attempts\":"
                << result.gram_stats.guarded_attempts
                << ",\"gram_guarded_accepts\":"
                << result.gram_stats.guarded_accepts
                << ",\"gram_guarded_declines\":"
                << result.gram_stats.guarded_declines
                << ",\"gram_extended_refinements\":"
                << result.gram_stats.extended_refinements
                << ",\"gram_worst_forward_bound\":"
                << result.gram_stats.worst_accepted_forward_bound
                << ",\"gram_worst_tail_bound\":"
                << result.gram_stats.worst_accepted_tail_bound
                << ",\"selector_calls\":" << result.selector_calls
                << ",\"selector_ms\":" << result.selector_ms
                << ",\"terminal_gates\":" << result.terminal_gates
                << ",\"recovery_calls\":" << result.recovery_calls
                << ",\"recovery_ms\":" << result.recovery_ms
                << ",\"terminal_support_repairs\":"
                << result.terminal_support_repairs
                << ",\"terminal_support_repair_successes\":"
                << result.terminal_support_repair_successes
                << ",\"terminal_support_repair_iterations\":"
                << result.terminal_support_repair_iterations
                << ",\"terminal_support_repair_ms\":"
                << result.terminal_support_repair_ms
                << ",\"qp_calls\":" << result.qp_calls
                << ",\"qp_iterations\":" << result.qp_iterations
                << ",\"qp_ms\":" << result.qp_ms
                << ",\"rank_complete_calls\":"
                << result.rank_complete_calls
                << ",\"rank_complete_repairs\":"
                << result.rank_complete_repairs
                << ",\"rank_complete_ms\":"
                << result.rank_complete_ms
                << ",\"critical_right_calls\":"
                << result.critical_right_calls
                << ",\"critical_right_repairs\":"
                << result.critical_right_repairs
                << ",\"critical_right_ms\":"
                << result.critical_right_ms
                << ",\"maintained_rowspace_audit\":{\"checks\":"
                << result.maintained_rowspace_audit.checks
                << ",\"matches\":"
                << result.maintained_rowspace_audit.matches
                << ",\"seed_attempts\":"
                << result.maintained_rowspace_audit.seed_attempts
                << ",\"seed_successes\":"
                << result.maintained_rowspace_audit.seed_successes
                << ",\"false_admissions\":"
                << result.maintained_rowspace_audit.false_admissions
                << ",\"local_transitions\":"
                << result.maintained_rowspace_audit.local_transitions
                << ",\"additions\":"
                << result.maintained_rowspace_audit.additions
                << ",\"removals\":"
                << result.maintained_rowspace_audit.removals
                << ",\"rank_increases\":"
                << result.maintained_rowspace_audit.rank_increases
                << ",\"rank_decreases\":"
                << result.maintained_rowspace_audit.rank_decreases
                << ",\"rank_change_declines\":"
                << result.maintained_rowspace_audit.rank_change_declines
                << ",\"numerical_declines\":"
                << result.maintained_rowspace_audit.numerical_declines
                << ",\"refactors\":"
                << result.maintained_rowspace_audit.refactors
                << ",\"seed_ms\":"
                << result.maintained_rowspace_audit.seed_ms
                << ",\"transition_ms\":"
                << result.maintained_rowspace_audit.transition_ms
                << ",\"solve_ms\":"
                << result.maintained_rowspace_audit.solve_ms
                << ",\"max_g_error\":"
                << result.maintained_rowspace_audit.max_g_error
                << ",\"max_ua_error\":"
                << result.maintained_rowspace_audit.max_ua_error
                << ",\"max_bua_error\":"
                << result.maintained_rowspace_audit.max_bua_error
                << ",\"worst_rowspace_residual\":"
                << result.maintained_rowspace_audit.worst_rowspace_residual
                << ",\"worst_orthogonality\":"
                << result.maintained_rowspace_audit.worst_orthogonality
                << ",\"worst_slope_residual\":"
                << result.maintained_rowspace_audit.worst_slope_residual
                << ",\"cold_reveals\":"
                << result.maintained_rowspace_audit.cold_reveals
                << ",\"cold_reveal_ms\":"
                << result.maintained_rowspace_audit.cold_reveal_ms
                << "}"
                << ",\"rank_lift_audit\":{\"global_rank\":"
                << result.rank_lift_audit.global_rank
                << ",\"structurally_maximal_faces\":"
                << result.rank_lift_audit.structurally_maximal_faces
                << ",\"audits\":" << result.rank_lift_audit.audits
                << ",\"lp_solves\":" << result.rank_lift_audit.lp_solves
                << ",\"direction_successes\":"
                << result.rank_lift_audit.direction_successes
                << ",\"rank_gains\":" << result.rank_lift_audit.rank_gains
                << ",\"full_rank\":" << result.rank_lift_audit.full_rank
                << ",\"weak_steps\":" << result.rank_lift_audit.weak_steps
                << ",\"activated_rows\":"
                << result.rank_lift_audit.activated_rows
                << ",\"source_dense_fallbacks\":"
                << result.rank_lift_audit.source_dense_fallbacks
                << ",\"candidate_dense_fallbacks\":"
                << result.rank_lift_audit.candidate_dense_fallbacks
                << ",\"live_applied\":"
                << result.rank_lift_audit.live_applied
                << ",\"total_rank_gain\":"
                << result.rank_lift_audit.total_rank_gain
                << ",\"max_rank_gain\":"
                << result.rank_lift_audit.max_rank_gain
                << ",\"lp_ms\":" << result.rank_lift_audit.lp_ms
                << ",\"global_factor_ms\":"
                << result.rank_lift_audit.global_factor_ms
                << ",\"factor_ms\":" << result.rank_lift_audit.factor_ms
                << ",\"max_dual_direction_residual\":"
                << result.rank_lift_audit.max_dual_direction_residual
                << ",\"max_objective_direction_residual\":"
                << result.rank_lift_audit.max_objective_direction_residual
                << ",\"max_objective_drift\":"
                << result.rank_lift_audit.max_objective_drift
                << ",\"live_shift_inf\":"
                << result.rank_lift_audit.live_shift_inf
                << ",\"live_shift_relative\":"
                << result.rank_lift_audit.live_shift_relative << '}'
                << ",\"direct_phase_ms\":{\"total\":"
                << result.direct_stats.total_ms
                << ",\"calls\":" << result.direct_stats.calls
                << ",\"numerical_calls\":"
                << result.direct_stats.numerical_calls
                << ",\"cache_hits\":" << result.direct_stats.cache_hits
                << ",\"cache_inserts\":" << result.direct_stats.cache_inserts
                << ",\"assembly\":" << result.direct_stats.assembly_ms
                << ",\"spqr\":" << result.direct_stats.spqr_ms
                << ",\"core_extract\":"
                << result.direct_stats.core_extract_ms
                << ",\"rz\":" << result.direct_stats.rz_ms
                << ",\"triangular\":" << result.direct_stats.triangular_ms
                << ",\"products\":" << result.direct_stats.products_ms
                << ",\"residual\":" << result.direct_stats.residual_ms
                << ",\"dense_svd\":" << result.direct_stats.dense_svd_ms
                << ",\"direct_guard_audits\":"
                << result.direct_stats.direct_guard_audits
                << ",\"direct_guard_raw_accurate\":"
                << result.direct_stats.direct_guard_raw_accurate
                << ",\"direct_guard_refined_accurate\":"
                << result.direct_stats.direct_guard_refined_accurate
                << ",\"direct_guard_tail_accepts\":"
                << result.direct_stats.direct_guard_tail_accepts
                << ",\"direct_guard_tail_false_accepts\":"
                << result.direct_stats.direct_guard_tail_false_accepts
                << ",\"direct_guard_rank_mismatches\":"
                << result.direct_stats.direct_guard_rank_mismatches
                << ",\"direct_guard_rank_match_accurate\":"
                << result.direct_stats.direct_guard_rank_match_accurate
                << ",\"direct_guard_rank_mismatch_accurate\":"
                << result.direct_stats.direct_guard_rank_mismatch_accurate
                << ",\"direct_guard_ferr_accepts\":"
                << result.direct_stats.direct_guard_ferr_accepts
                << ",\"direct_guard_ferr_false_accepts\":"
                << result.direct_stats.direct_guard_ferr_false_accepts
                << ",\"direct_guard_consensus_attempts\":"
                << result.direct_stats.direct_guard_consensus_attempts
                << ",\"direct_guard_consensus_accepts\":"
                << result.direct_stats.direct_guard_consensus_accepts
                << ",\"direct_guard_consensus_false_accepts\":"
                << result.direct_stats.direct_guard_consensus_false_accepts
                << ",\"direct_guard_tight_consensus_accepts\":"
                << result.direct_stats.direct_guard_tight_consensus_accepts
                << ",\"direct_guard_tight_consensus_false_accepts\":"
                << result.direct_stats.direct_guard_tight_consensus_false_accepts
                << ",\"core_svd_audits\":"
                << result.direct_stats.core_svd_audits
                << ",\"core_svd_accurate\":"
                << result.direct_stats.core_svd_accurate
                << ",\"core_svd_rank_mismatches\":"
                << result.direct_stats.core_svd_rank_mismatches
                << ",\"weak_spectrum_samples\":"
                << result.direct_stats.weak_spectrum_samples
                << ",\"weak_q8_over_16\":"
                << result.direct_stats.weak_q8_over_16
                << ",\"weak_q8_max\":"
                << result.direct_stats.weak_q8_max
                << ",\"weak_q8_sum\":"
                << result.direct_stats.weak_q8_sum
                << ",\"weak_q10_max\":"
                << result.direct_stats.weak_q10_max
                << ",\"weak_q10_sum\":"
                << result.direct_stats.weak_q10_sum
                << ",\"weak_q12_max\":"
                << result.direct_stats.weak_q12_max
                << ",\"weak_q12_sum\":"
                << result.direct_stats.weak_q12_sum
                << ",\"discarded_near_cutoff_max\":"
                << result.direct_stats.discarded_near_cutoff_max
                << ",\"discarded_near_cutoff_sum\":"
                << result.direct_stats.discarded_near_cutoff_sum
                << ",\"weak_spectrum_ms\":"
                << result.direct_stats.weak_spectrum_ms
                << ",\"direct_guard_raw_max_error\":"
                << result.direct_stats.direct_guard_raw_max_error
                << ",\"direct_guard_refined_max_error\":"
                << result.direct_stats.direct_guard_refined_max_error
                << ",\"direct_guard_tail_worst_error\":"
                << result.direct_stats.direct_guard_tail_worst_error
                << ",\"direct_guard_ferr_worst_error\":"
                << result.direct_stats.direct_guard_ferr_worst_error
                << ",\"direct_guard_consensus_worst_error\":"
                << result.direct_stats.direct_guard_consensus_worst_error
                << ",\"direct_guard_consensus_ms\":"
                << result.direct_stats.direct_guard_consensus_ms
                << ",\"direct_guard_tight_consensus_worst_error\":"
                << result.direct_stats.direct_guard_tight_consensus_worst_error
                << ",\"core_svd_ms\":" << result.direct_stats.core_svd_ms
                << ",\"core_svd_max_error\":"
                << result.direct_stats.core_svd_max_error
                << ",\"core_svd_max_ua_error\":"
                << result.direct_stats.core_svd_max_ua_error
                << ",\"core_svd_max_uc_error\":"
                << result.direct_stats.core_svd_max_uc_error
                << ",\"core_svd_max_g_error\":"
                << result.direct_stats.core_svd_max_g_error
                << ",\"core_svd_max_h_error\":"
                << result.direct_stats.core_svd_max_h_error
                << ",\"direct_guard_refinement_ms\":"
                << result.direct_stats.direct_guard_refinement_ms
                << ",\"qr_update_observations\":"
                << result.direct_stats.qr_update_observations
                << ",\"qr_update_eligible\":"
                << result.direct_stats.qr_update_eligible
                << ",\"qr_update_attempts\":"
                << result.direct_stats.qr_update_attempts
                << ",\"qr_update_admitted\":"
                << result.direct_stats.qr_update_admitted
                << ",\"qr_update_accurate\":"
                << result.direct_stats.qr_update_accurate
                << ",\"qr_update_false_admits\":"
                << result.direct_stats.qr_update_false_admits
                << ",\"qr_update_live_returns\":"
                << result.direct_stats.qr_update_live_returns
                << ",\"qr_update_refactors\":"
                << result.direct_stats.qr_update_refactors
                << ",\"qr_update_additions\":"
                << result.direct_stats.qr_update_additions
                << ",\"qr_update_downdates\":"
                << result.direct_stats.qr_update_downdates
                << ",\"qr_update_large_changes\":"
                << result.direct_stats.qr_update_large_changes
                << ",\"qr_update_numerical_declines\":"
                << result.direct_stats.qr_update_numerical_declines
                << ",\"qr_update_full_rank_observations\":"
                << result.direct_stats.qr_update_full_rank_observations
                << ",\"qr_update_rank_deficient_observations\":"
                << result.direct_stats.qr_update_rank_deficient_observations
                << ",\"qr_update_rank_changes\":"
                << result.direct_stats.qr_update_rank_changes
                << ",\"qr_update_factor_failures\":"
                << result.direct_stats.qr_update_factor_failures
                << ",\"qr_update_ratio_declines\":"
                << result.direct_stats.qr_update_ratio_declines
                << ",\"qr_update_solve_failures\":"
                << result.direct_stats.qr_update_solve_failures
                << ",\"qr_update_repivot_retries\":"
                << result.direct_stats.qr_update_repivot_retries
                << ",\"qr_update_local_repivot_attempts\":"
                << result.direct_stats.qr_update_local_repivot_attempts
                << ",\"qr_update_local_repivot_successes\":"
                << result.direct_stats.qr_update_local_repivot_successes
                << ",\"qr_update_update_ms\":"
                << result.direct_stats.qr_update_update_ms
                << ",\"qr_update_local_repivot_ms\":"
                << result.direct_stats.qr_update_local_repivot_ms
                << ",\"qr_update_solve_ms\":"
                << result.direct_stats.qr_update_solve_ms
                << ",\"qr_update_replaceable_oracle_ms\":"
                << result.direct_stats.qr_update_replaceable_oracle_ms
                << ",\"qr_update_accurate_candidate_ms\":"
                << result.direct_stats.qr_update_accurate_candidate_ms
                << ",\"qr_update_median_us\":"
                << result.direct_stats.qr_update_median_us
                << ",\"qr_update_oracle_median_us\":"
                << result.direct_stats.qr_update_oracle_median_us
                << ",\"qr_update_max_error\":"
                << result.direct_stats.qr_update_max_error
                << ",\"qr_update_max_admitted_error\":"
                << result.direct_stats.qr_update_max_admitted_error
                << ",\"qr_update_max_residual\":"
                << result.direct_stats.qr_update_max_residual
                << ",\"cache\":" << result.direct_stats.cache_ms
                << "},\"nudge_epsilon\":" << nudge_epsilon
                << ",\"certificate\":{\"primal\":"
                << result.certificate.primal << ",\"dual\":"
                << result.certificate.dual << ",\"nonnegative\":"
                << result.certificate.nonnegative << ",\"gap\":"
                << result.certificate.gap << '}';
      if (emit_solution) {
        std::cout << ",\"solution\":{\"x\":";
        print_vector(result.x);
        std::cout << ",\"y\":";
        print_vector(result.y);
        std::cout << '}';
      }
      std::cout << "}\n";
    } catch (const std::exception &error) {
      good = false;
      std::cerr << path << ": " << error.what() << '\n';
    }
  }
  return good ? 0 : 1;
}
