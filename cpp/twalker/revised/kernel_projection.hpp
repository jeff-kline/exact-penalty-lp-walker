#pragma once

#include "fixture.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace twalker::revised {

struct KernelProjectionStats {
  std::uint64_t events = 0;
  std::uint64_t drops = 0;
  std::uint64_t dependent_exchanges = 0;
  std::uint64_t qr_insertions = 0;
  std::uint64_t qr_deletions = 0;
  std::uint64_t qr_rebuilds = 0;
  std::uint64_t equality_refinements = 0;
  double nullspace_ms = 0.0;
  double active_update_ms = 0.0;
  double active_solve_ms = 0.0;
  double state_ms = 0.0;
  double pricing_ms = 0.0;
  double certificate_ms = 0.0;
  double total_ms = 0.0;
};

struct KernelProjectionResult {
  bool certified = false;
  std::string status;
  int rank = 0;
  int nullity = 0;
  int support = 0;
  int active_core = 0;
  double t = 0.0;
  double equality_error = 0.0;
  double nonnegative_error = 0.0;
  double stationarity_error = 0.0;
  double complementarity_error = 0.0;
  double dual_error = 0.0;
  std::vector<double> y;
  KernelProjectionStats stats;
};

// Dense null-space lane for low nullity q=n-rank(B).  The fixed-t problem is
// reduced to
//
//   min_z 1/2 ||z-z0||^2  subject to Q z + yd >= 0,
//
// and solved by a native dual active set.  The active QR is updated by column
// insertion/deletion; `maintained=false` is an audit arm that deliberately
// rebuilds the identical factor after each event.
class KernelProjector {
 public:
  explicit KernelProjector(const Fixture &fixture);

  KernelProjectionResult solve(double t, double target_sign = -1.0,
                               bool maintained = true,
                               std::uint64_t max_events = 10000);

 private:
  const Fixture &fixture_;
  bool factor_ready_ = false;
  int rank_ = 0;
  int nullity_ = 0;
  std::vector<double> singular_;
  std::vector<double> left_;       // m x m, column major
  std::vector<double> right_t_;    // n x n, column major
  std::vector<double> range_;      // n x rank, column major
  std::vector<double> q_;          // n x nullity, column major
  std::vector<double> particular_;
  double factor_ms_ = 0.0;
  std::string factor_failure_;

  bool ensure_factor();
};

}  // namespace twalker::revised
