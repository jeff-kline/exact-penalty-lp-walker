#pragma once

#include <cstdint>
#include <vector>

namespace twalker {

struct DenseNnlsResult {
  bool converged = false;
  std::vector<double> x;
  std::int64_t inner_iterations = 0;
  std::int64_t outer_iterations = 0;
  double kkt_error = 0.0;
};

// Lawson-Hanson NNLS for a dense column-major matrix, using a maintained thin
// QR with CGS2 insertion and Givens deletion.  Adapted from the repository's
// validated cpp/nnls.cpp corrector implementation.
DenseNnlsResult dense_nnls(const std::vector<double> &matrix, int rows,
                           int columns, const std::vector<double> &rhs,
                           int max_iterations);

}  // namespace twalker
