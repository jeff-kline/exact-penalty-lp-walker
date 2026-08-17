#pragma once

#include "fixture.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace twalker::revised {

struct TriangularProjectionStats {
  std::uint64_t events = 0;
  std::uint64_t drops = 0;
  std::uint64_t dependent_exchanges = 0;
  std::uint64_t insertions = 0;
  std::uint64_t deletions = 0;
  std::uint64_t normal_solves = 0;
  std::uint64_t refinements = 0;
  std::uint64_t face_polishes = 0;
  double factor_ms = 0.0;
  double operator_ms = 0.0;
  double active_update_ms = 0.0;
  double active_solve_ms = 0.0;
  double pricing_ms = 0.0;
  double certificate_ms = 0.0;
  double face_polish_ms = 0.0;
  double total_ms = 0.0;
};

struct TriangularProjectionResult {
  bool certified = false;
  std::string status;
  int rank = 0;
  int nullity = 0;
  int support = 0;
  int active_core = 0;
  double t = 0.0;
  double factor_diagonal_ratio = 0.0;
  double equality_error = 0.0;
  double nonnegative_error = 0.0;
  double stationarity_error = 0.0;
  double complementarity_error = 0.0;
  double dual_error = 0.0;
  std::vector<double> y;
  TriangularProjectionStats stats;
};

// Fixed-t projection without an explicit dense basis for ker(B').  For
// full-column-rank B, apply the orthogonal null projector as
//
//   P x = x - B (B'B)^-1 B'x,
//
// where the normal solve is carried out by a column-equilibrated, pivoted QR
// of B.  Only the triangular R and permutation are retained.  Bound events
// maintain a triangular factor of P_AA, so neither Q nor N'N is formed.
// Every candidate is certified against the original sparse B.
class TriangularProjector {
 public:
  explicit TriangularProjector(const Fixture &fixture);

  TriangularProjectionResult solve(double t, double target_sign = -1.0,
                                   std::uint64_t max_events = 10000);

 private:
  const Fixture &fixture_;
  bool factor_ready_ = false;
  int rank_ = 0;
  int nullity_ = 0;
  std::vector<double> triangular_;  // m x m upper triangular, column major
  std::vector<double> column_scale_;  // D_j=1/||B(:,j)||_2
  std::vector<int> permutation_;  // triangular position -> original column
  double factor_diagonal_ratio_ = 0.0;
  double factor_ms_ = 0.0;
  std::string factor_failure_;

  bool ensure_factor();
  bool solve_normal(const std::vector<double> &rhs,
                    std::vector<double> &solution,
                    TriangularProjectionStats &stats) const;
  bool apply_projector(const std::vector<double> &input,
                       std::vector<double> &output,
                       TriangularProjectionStats &stats) const;
};

}  // namespace twalker::revised
