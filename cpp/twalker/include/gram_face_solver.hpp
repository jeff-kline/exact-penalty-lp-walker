#pragma once

#include "face_solver.hpp"
#include "fixture.hpp"

#include <cholmod.h>

#include <cstdint>
#include <vector>

namespace twalker {

struct GramSolveStats {
  int rebuilds = 0;
  int updates = 0;
  int downdates = 0;
  int declines = 0;
  int refinements = 0;
  int guarded_attempts = 0;
  int guarded_accepts = 0;
  int guarded_declines = 0;
  int extended_refinements = 0;
  double worst_accepted_forward_bound = 0.0;
  double worst_accepted_tail_bound = 0.0;
  double worst_accepted_contraction = 0.0;
};

// Maintained full-column-rank face solver.  It is deliberately a fail-closed
// fast lane: rank-deficient or poorly conditioned faces remain SPQR work.
class GramFaceSolver {
 public:
  explicit GramFaceSolver(const Fixture &fixture, double min_rcond = 1e-8,
                          std::vector<double> target_shift = {});
  ~GramFaceSolver();
  GramFaceSolver(const GramFaceSolver &) = delete;
  GramFaceSolver &operator=(const GramFaceSolver &) = delete;

  bool solve(const std::vector<std::uint32_t> &rows, FaceSolution &solution,
             bool force_guarded_refinement = false);
  const GramSolveStats &stats() const { return stats_; }
  double rcond() const { return rcond_; }

 private:
  const Fixture &fixture_;
  std::vector<double> target_shift_;
  double min_rcond_;
  double rcond_ = 0.0;
  cholmod_common common_{};
  cholmod_factor *factor_ = nullptr;
  std::vector<std::uint8_t> support_;
  std::vector<double> atb_;
  std::vector<double> ats_;
  std::vector<std::int64_t> inverse_perm_;
  GramSolveStats stats_;

  bool rebuild(const std::vector<std::uint32_t> &rows);
  bool transition(const std::vector<std::uint32_t> &rows);
  bool apply_row(std::uint32_t row, bool add);
  bool refine_extended(const std::vector<std::uint32_t> &rows, double *x,
                       double &ua_error_bound, double &uc_error_bound);
  void clear_factor();
};

}  // namespace twalker
