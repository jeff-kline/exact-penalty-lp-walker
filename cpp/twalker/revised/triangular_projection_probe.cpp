#include "fixture.hpp"
#include "face_solver.hpp"
#include "kernel_projection.hpp"
#include "triangular_projection.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

double relative_error(const std::vector<double> &left,
                      const std::vector<double> &right) {
  if (left.size() != right.size()) return INFINITY;
  double difference = 0.0, scale = 1.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    difference = std::max(difference, std::abs(left[i] - right[i]));
    scale = std::max(scale, std::abs(right[i]));
  }
  return difference / scale;
}

void emit(const std::string &path,
          const twalker::revised::TriangularProjectionResult &r,
          double oracle_error, double oracle_ms, int worst_index,
          double candidate_value, double oracle_value,
          double objective_difference, double face_candidate_error,
          double face_oracle_error) {
  std::cout << "{\"fixture\":\"" << path << "\",\"lane\":\"triangular\""
            << ",\"t\":" << r.t << ",\"status\":\"" << r.status
            << "\",\"rank\":" << r.rank << ",\"nullity\":"
            << r.nullity << ",\"support\":" << r.support
            << ",\"active_core\":" << r.active_core
            << ",\"events\":" << r.stats.events
            << ",\"drops\":" << r.stats.drops
            << ",\"exchanges\":" << r.stats.dependent_exchanges
            << ",\"diag_ratio\":" << r.factor_diagonal_ratio
            << ",\"oracle_error\":" << oracle_error
            << ",\"oracle_ms\":" << oracle_ms << ",\"errors\":["
            << r.equality_error << ',' << r.nonnegative_error << ','
            << r.stationarity_error << ',' << r.complementarity_error << ','
            << r.dual_error << "]"
            << ",\"ms\":{\"total\":" << r.stats.total_ms
            << ",\"factor\":" << r.stats.factor_ms
            << ",\"operator\":" << r.stats.operator_ms
            << ",\"update\":" << r.stats.active_update_ms
            << ",\"active_solve\":" << r.stats.active_solve_ms
            << ",\"pricing\":" << r.stats.pricing_ms
            << ",\"certificate\":" << r.stats.certificate_ms << "}"
            << ",\"normal_solves\":" << r.stats.normal_solves
            << ",\"refinements\":" << r.stats.refinements
            << ",\"face_polishes\":" << r.stats.face_polishes
            << ",\"face_polish_ms\":" << r.stats.face_polish_ms
            << ",\"comparison\":{\"index\":" << worst_index
            << ",\"candidate\":" << candidate_value
            << ",\"oracle\":" << oracle_value
            << ",\"objective_difference\":" << objective_difference
            << ",\"face_candidate_error\":" << face_candidate_error
            << ",\"face_oracle_error\":" << face_oracle_error << "}}\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: triangular_projection_probe fixture.twfx [t] [audit]\n";
    return 2;
  }
  const double t = argc > 2 ? std::stod(argv[2]) : 0.0;
  const bool audit = argc > 3 && std::string(argv[3]) == "audit";
  std::cout << std::setprecision(17);
  try {
    const auto fixture = twalker::read_fixture(argv[1]);
    twalker::revised::TriangularProjector projector(fixture);
    const auto candidate = projector.solve(t, -1.0, 10000);
    double error = 0.0, oracle_ms = 0.0, candidate_value = 0.0,
           oracle_value = 0.0, objective_difference = 0.0,
           face_candidate_error = 0.0, face_oracle_error = 0.0;
    int worst_index = -1;
    bool good = candidate.certified;
    if (audit) {
      twalker::revised::KernelProjector oracle(fixture);
      const auto reference = oracle.solve(t, -1.0, true, 10000);
      oracle_ms = reference.stats.total_ms;
      error = relative_error(candidate.y, reference.y);
      double worst = 0.0;
      long double candidate_objective = 0.0L, oracle_objective = 0.0L;
      for (std::size_t i = 0; i < candidate.y.size(); ++i) {
        const double difference = std::abs(candidate.y[i] - reference.y[i]);
        if (difference > worst) {
          worst = difference;
          worst_index = static_cast<int>(i);
          candidate_value = candidate.y[i];
          oracle_value = reference.y[i];
        }
        candidate_objective += 0.5L * candidate.y[i] * candidate.y[i];
        oracle_objective += 0.5L * reference.y[i] * reference.y[i];
      }
      objective_difference = static_cast<double>(candidate_objective
                                                 - oracle_objective);
      double y_scale = 1.0;
      for (double value : candidate.y)
        y_scale = std::max(y_scale, std::abs(value));
      std::vector<std::uint32_t> rows;
      for (std::size_t i = 0; i < candidate.y.size(); ++i)
        if (candidate.y[i] > 1e-9 * y_scale) rows.push_back(i);
      twalker::FaceSolver face_solver(fixture, false);
      const auto face = face_solver.solve(rows);
      std::vector<double> face_y(candidate.y.size(), 0.0);
      for (std::size_t local = 0; local < face.rows.size(); ++local)
        face_y[face.rows[local]] = face.h[local];
      face_candidate_error = relative_error(candidate.y, face_y);
      face_oracle_error = relative_error(reference.y, face_y);
      good = good && reference.certified && error <= 1e-9;
    }
    emit(argv[1], candidate, error, oracle_ms, worst_index, candidate_value,
         oracle_value, objective_difference, face_candidate_error,
         face_oracle_error);
    return good ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
