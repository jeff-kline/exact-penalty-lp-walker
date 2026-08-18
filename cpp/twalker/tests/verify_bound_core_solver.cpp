#include "bound_core_face_solver.hpp"
#include "fixture.hpp"

#include <algorithm>
#include <chrono>
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
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  bool good = true;
  // Structural eligibility keeps this route off most faces: on the 26-fixture
  // panel it serves exactly one, fit1d.  Exiting 0 therefore says almost
  // nothing on its own.  Pass --min-served=1 to assert the route is still
  // reached at all.
  long long min_served = -1;
  int first = 1;
  for (; first < argc; ++first) {
    const std::string arg = argv[first];
    if (arg.rfind("--min-served=", 0) != 0) break;
    min_served = std::stoll(arg.substr(13));
  }
  long long total_served = 0;
  std::cout << std::setprecision(17);
  for (int argument = first; argument < argc; ++argument) {
    const std::string path = argv[argument];
    const auto fixture = twalker::read_fixture(path);
    twalker::BoundCoreFaceSolver solver(fixture);
    int served = 0, accurate = 0;
    double worst = 0.0;
    for (const auto &face : fixture.faces) {
      twalker::FaceSolution got;
      if (!solver.solve(face.rows, got)) continue;
      ++served;
      const double error = std::max(
          {twalker::relative_inf_error(got.g, face.g),
           twalker::relative_inf_error(got.h, face.h),
           twalker::relative_inf_error(got.ua, face.ua),
           twalker::relative_inf_error(got.uc, face.uc)});
      worst = std::max(worst, error);
      accurate += std::isfinite(error) && error <= 1e-10;
      good = good && std::isfinite(error) && error <= 1e-10;
    }
    const auto &stats = solver.stats();
    std::cout << "{\"model\":\"" << stem(path)
              << "\",\"eligible\":" << solver.structurally_eligible()
              << ",\"faces\":" << fixture.faces.size()
              << ",\"served\":" << served << ",\"accurate\":" << accurate
              << ",\"max_error\":" << worst
              << ",\"maximum_border\":" << stats.maximum_border
              << ",\"refinements\":" << stats.refinements
              << ",\"total_ms\":" << stats.total_ms
              << ",\"factor_ms\":" << stats.factor_ms
              << ",\"products_ms\":" << stats.products_ms
              << ",\"minimum_rcond\":" << stats.minimum_rcond
              << ",\"max_dres\":" << stats.worst_dres
              << ",\"max_piece\":" << stats.worst_piece_residual << "}\n";
    total_served += served;
  }
  if (min_served >= 0 && total_served < min_served) {
    std::cerr << "coverage floor: served " << total_served
              << " faces, require " << min_served << '\n';
    good = false;
  }
  return good ? 0 : 1;
}
