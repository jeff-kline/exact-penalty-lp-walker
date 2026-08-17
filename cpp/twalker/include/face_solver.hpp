#pragma once

#include "fixture.hpp"

#include <cholmod.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace twalker {

class FaceDecline : public std::runtime_error {
 public:
  explicit FaceDecline(const std::string &message) : std::runtime_error(message) {}
};

struct FaceSolution {
  // bua = B*ua and buc = B*uc are part of the face artifact.  The walker
  // consumes them in settle, endpoint certification, and event selection;
  // recomputing them in each consumer was several sparse passes per pivot.
  std::vector<double> g, h, ua, uc, bua, buc;
  std::vector<std::uint32_t> rows;
  double dres = 0.0;
  double piece_residual = 0.0;
  double core_diagonal_ratio = 0.0;
  std::int64_t rank = 0;
  std::size_t nnz_R = 0;
  bool used_dense_fallback = false;
  bool used_core_svd = false;
  // Extension-band Gram solutions carry a posteriori coefficient bounds.
  // The walker propagates them to its discrete decisions and falls back to
  // direct SPQR whenever a decision interval crosses a boundary.
  bool used_extended_gram = false;
  bool used_maintained_gram = false;
  bool used_bound_core = false;
  double ua_relative_error_bound = 0.0;
  double uc_relative_error_bound = 0.0;
  // Correlated reduced-system artifacts for bounding -t*ua_head+uc_head
  // directly.  Separate coefficient intervals lose this cancellation at
  // large t and are unnecessarily pessimistic in settle decisions.
  bool affine_bound_valid = false;
  std::vector<double> reduced_residual_a, reduced_residual_c;
  std::vector<double> reduced_head_a, reduced_head_c;
  double reduced_gram_inf = 0.0;
  double reduced_rcond = 0.0;
  double projection_inf_norm = 0.0;
  // Row-specific infinity norm of the map from reduced-head error to B*x
  // error.  Structured bordered solvers can make this dramatically tighter
  // than bounding x first and multiplying by each row's l1 norm.
  std::vector<double> product_projection_inf_norm;
  std::vector<double> audit_oracle_g, audit_oracle_h;
  std::vector<double> audit_oracle_bua, audit_oracle_buc;
  // Audit-only maintained-QR answer for this exact numerical face.  The
  // direct answer above remains authoritative; Walker uses these vectors only
  // to compare settle and event decisions under TWALKER_QR_UPDATE_AUDIT.
  bool qr_update_audit_candidate = false;
  std::vector<double> qr_update_audit_g, qr_update_audit_h;
  std::vector<double> qr_update_audit_bua, qr_update_audit_buc;
  // Quarantined revised-column experiment.  When explicitly requested, the
  // direct SPQR/COD lane exports A^+ for its accepted face so the recurrence
  // can start from the factorization already paid for.  Empty in production.
  std::vector<double> recurrence_pseudoinverse;
  std::int64_t recurrence_seed_rank = 0;
  // Cheaper alternative seed: the accepted pre-RZ SPQR core R and its column
  // permutation.  The revised lane can recover C, T=R11^-1 R, and the two
  // fixed square-root factors without a second rank-revealing factorization.
  std::vector<double> factored_qr_core;
  std::vector<double> factored_rz_core;
  std::vector<double> factored_rz_tau;
  std::vector<double> svd_row_space;
  // Audit-only authoritative thin-SVD seed for Q-aware row updates.  These
  // vectors are moved around the ordinary face cache and returned only to the
  // immediate caller when the dedicated experiment is enabled.
  std::vector<double> svd_left_space;
  std::vector<double> svd_singular_values;
  std::vector<std::int64_t> factored_permutation;
  std::int64_t factored_seed_rank = 0;
};

// Cumulative wall-clock accounting for the direct correctness lane.  These
// timers are deliberately inside the native solver so wrapper/process startup
// cannot be mistaken for numerical work.
struct FaceSolveStats {
  std::uint64_t calls = 0;
  std::uint64_t numerical_calls = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t cache_inserts = 0;
  std::uint64_t dense_fallbacks = 0;
  std::uint64_t direct_guard_audits = 0;
  std::uint64_t direct_guard_raw_accurate = 0;
  std::uint64_t direct_guard_refined_accurate = 0;
  std::uint64_t direct_guard_tail_accepts = 0;
  std::uint64_t direct_guard_tail_false_accepts = 0;
  std::uint64_t direct_guard_rank_mismatches = 0;
  std::uint64_t direct_guard_rank_match_accurate = 0;
  std::uint64_t direct_guard_rank_mismatch_accurate = 0;
  std::uint64_t direct_guard_ferr_accepts = 0;
  std::uint64_t direct_guard_ferr_false_accepts = 0;
  std::uint64_t direct_guard_consensus_attempts = 0;
  std::uint64_t direct_guard_consensus_accepts = 0;
  std::uint64_t direct_guard_consensus_false_accepts = 0;
  std::uint64_t direct_guard_tight_consensus_accepts = 0;
  std::uint64_t direct_guard_tight_consensus_false_accepts = 0;
  std::uint64_t core_svd_audits = 0;
  std::uint64_t core_svd_accurate = 0;
  std::uint64_t core_svd_rank_mismatches = 0;
  std::uint64_t weak_spectrum_samples = 0;
  std::uint64_t weak_q8_over_16 = 0;
  std::uint64_t weak_q8_max = 0;
  std::uint64_t weak_q8_sum = 0;
  std::uint64_t weak_q10_max = 0;
  std::uint64_t weak_q10_sum = 0;
  std::uint64_t weak_q12_max = 0;
  std::uint64_t weak_q12_sum = 0;
  std::uint64_t discarded_near_cutoff_max = 0;
  std::uint64_t discarded_near_cutoff_sum = 0;
  double total_ms = 0.0;
  double assembly_ms = 0.0;
  double spqr_ms = 0.0;
  double core_extract_ms = 0.0;
  double rz_ms = 0.0;
  double triangular_ms = 0.0;
  double products_ms = 0.0;
  double residual_ms = 0.0;
  double dense_svd_ms = 0.0;
  double direct_guard_refinement_ms = 0.0;
  double direct_guard_raw_max_error = 0.0;
  double direct_guard_refined_max_error = 0.0;
  double direct_guard_tail_worst_error = 0.0;
  double direct_guard_ferr_worst_error = 0.0;
  double direct_guard_consensus_worst_error = 0.0;
  double direct_guard_consensus_ms = 0.0;
  double direct_guard_tight_consensus_worst_error = 0.0;
  double core_svd_ms = 0.0;
  double core_svd_max_error = 0.0;
  double core_svd_max_ua_error = 0.0;
  double core_svd_max_uc_error = 0.0;
  double core_svd_max_g_error = 0.0;
  double core_svd_max_h_error = 0.0;
  double weak_spectrum_ms = 0.0;
  std::uint64_t qr_update_observations = 0;
  std::uint64_t qr_update_eligible = 0;
  std::uint64_t qr_update_attempts = 0;
  std::uint64_t qr_update_admitted = 0;
  std::uint64_t qr_update_accurate = 0;
  std::uint64_t qr_update_false_admits = 0;
  std::uint64_t qr_update_live_returns = 0;
  std::uint64_t qr_update_refactors = 0;
  std::uint64_t qr_update_additions = 0;
  std::uint64_t qr_update_downdates = 0;
  std::uint64_t qr_update_large_changes = 0;
  std::uint64_t qr_update_numerical_declines = 0;
  std::uint64_t qr_update_full_rank_observations = 0;
  std::uint64_t qr_update_rank_deficient_observations = 0;
  std::uint64_t qr_update_rank_changes = 0;
  std::uint64_t qr_update_factor_failures = 0;
  std::uint64_t qr_update_ratio_declines = 0;
  std::uint64_t qr_update_solve_failures = 0;
  std::uint64_t qr_update_repivot_retries = 0;
  std::uint64_t qr_update_local_repivot_attempts = 0;
  std::uint64_t qr_update_local_repivot_successes = 0;
  double qr_update_update_ms = 0.0;
  double qr_update_local_repivot_ms = 0.0;
  double qr_update_solve_ms = 0.0;
  double qr_update_replaceable_oracle_ms = 0.0;
  double qr_update_accurate_candidate_ms = 0.0;
  double qr_update_median_us = 0.0;
  double qr_update_oracle_median_us = 0.0;
  double qr_update_max_error = 0.0;
  double qr_update_max_admitted_error = 0.0;
  double qr_update_max_residual = 0.0;
  double cache_ms = 0.0;
};

class FaceSolver {
 public:
  explicit FaceSolver(const Fixture &fixture, bool enable_cache = true,
                      std::vector<double> target_shift = {},
                      int spqr_ordering = -1,
                      bool allow_unguarded_direct = false);
  ~FaceSolver();
  FaceSolver(const FaceSolver &) = delete;
  FaceSolver &operator=(const FaceSolver &) = delete;

  FaceSolution solve(const std::vector<std::uint32_t> &rows);
  // Force a fresh numerical factorization while retaining this solver's
  // accounting and maintenance state.  This is reserved for rare path-event
  // corrections where returning the cached artifact would merely repeat the
  // coefficient error that triggered the correction.
  FaceSolution solve_uncached(const std::vector<std::uint32_t> &rows);
  const FaceSolveStats &stats() const { return stats_; }
  void set_recurrence_seed_needed(bool needed) {
    recurrence_seed_needed_ = needed;
  }
  void set_factored_seed_needed(bool needed) {
    factored_seed_needed_ = needed;
  }

 private:
  struct RowHash {
    std::size_t operator()(const std::vector<std::uint32_t> &rows) const;
  };

  const Fixture &fixture_;
  std::vector<double> target_shift_;
  bool enable_cache_;
  int spqr_ordering_;
  bool allow_unguarded_direct_;
  cholmod_common common_{};
  FaceSolveStats stats_;
  std::unordered_map<std::vector<std::uint32_t>, FaceSolution, RowHash> cache_;
  std::unique_ptr<FaceSolver> alternate_solver_;
  void *qr_update_audit_ = nullptr;
  bool qr_update_live_ = false;
  bool force_numerical_ = false;
  bool recurrence_seed_needed_ = false;
  bool factored_seed_needed_ = false;
};

double relative_inf_error(const std::vector<double> &actual,
                          const std::vector<double> &expected);

}  // namespace twalker
