#pragma once

#include "fixture.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace twalker::revised {

struct AugmentedKktStats {
  std::uint64_t calls = 0;
  std::uint64_t accepted = 0;
  std::uint64_t factorizations = 0;
  std::uint64_t factor_reuses = 0;
  std::uint64_t warm_starts = 0;
  std::uint64_t infeasible_constant_rows = 0;
  std::uint64_t projection_failures = 0;
  std::uint64_t audit_failures = 0;
  std::uint64_t projection_sweeps = 0;
  double factor_ms = 0.0;
  double projection_ms = 0.0;
  double audit_ms = 0.0;
  double worst_active_residual = 0.0;
  double worst_inactive_violation = 0.0;
  double worst_transpose_residual = 0.0;
};

struct AugmentedKktSolution {
  std::vector<std::uint32_t> rows;
  std::vector<double> g;
  std::vector<double> ua;
  std::vector<double> bua;
  std::vector<std::uint32_t> selector_active_rows;
  std::int64_t rank = 0;
  std::int64_t nullity = 0;
  std::uint64_t fingerprint = 0;
  std::uint64_t projection_sweeps = 0;
  double active_residual = 0.0;
  double inactive_violation = 0.0;
  double transpose_residual = 0.0;
};

// Native multiplier selector for a centered projection-path segment.
//
// The free-face direction g is unique, while its equality multiplier is not
// when B_F is rank deficient.  Cold minimum-norm reconstruction can therefore
// preserve B_F' g=0 yet choose the wrong signs for inactive slack slopes.
// This class keeps the unique direction fixed and selects only inside
// null(B_F):
//
//   B_F ua = g_F-b_F,
//   b_N+B_N ua <= 0.
//
// A rank-revealing SVD is currently the cold crash/audit authority.  The
// inequality phase is Dykstra's projection onto halfspaces in null-space
// coordinates, warm-started from the previously admitted multiplier.  No LP,
// QP, or external solver is called.  Every answer is rechecked against the
// original sparse B before it can be returned.
class AugmentedKktBasis {
 public:
  explicit AugmentedKktBasis(const Fixture &fixture) : fixture_(fixture) {}

  bool select(const std::vector<std::uint8_t> &support,
              const std::vector<double> &direction,
              const std::vector<std::uint8_t> &selector_constraints,
              AugmentedKktSolution &solution);
  bool select_affine(const std::vector<std::uint8_t> &support,
                     const std::vector<double> &active_value,
                     const std::vector<double> &offset,
                     const std::vector<std::uint8_t> &selector_constraints,
                     const std::vector<double> &warm_multiplier,
                     int lane, AugmentedKktSolution &solution);

  const AugmentedKktStats &stats() const { return stats_; }
  const std::string &last_failure() const { return last_failure_; }
  int blocking_row() const { return blocking_row_; }

 private:
  const Fixture &fixture_;
  std::vector<double> last_multiplier_[2];
  std::vector<std::uint8_t> factor_support_;
  std::vector<std::uint32_t> factor_rows_;
  std::vector<double> factor_singular_;
  std::vector<double> factor_u_;
  std::vector<double> factor_vt_;
  std::vector<double> factor_null_basis_;
  int factor_active_ = 0;
  int factor_rank_ = 0;
  int factor_nullity_ = 0;
  AugmentedKktStats stats_;
  std::string last_failure_;
  int blocking_row_ = -1;

  bool ensure_factor(const std::vector<std::uint8_t> &support);
};

}  // namespace twalker::revised
