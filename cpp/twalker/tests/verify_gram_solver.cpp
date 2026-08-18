#include "fixture.hpp"
#include "gram_face_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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
  double min_rcond = 1e-8;
  if (const char *raw = std::getenv("TWALKER_GRAM_MIN_RCOND"))
    min_rcond = std::stod(raw);
  // This solver declines faces it cannot serve within its guard, so exiting 0
  // proves only that whatever it did serve was accurate.  On the 26-fixture
  // panel it serves 12.  Pass --min-served=12 to make a silent drop in
  // coverage fail the run rather than pass it quietly.
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
    twalker::GramFaceSolver solver(fixture, min_rcond);
    int served = 0, accurate = 0;
    double max_error = 0.0, max_dres = 0.0, max_piece = 0.0;
    std::vector<double> times;
    for (const auto &face : fixture.faces) {
      twalker::FaceSolution got;
      const auto begin = std::chrono::steady_clock::now();
      const bool accepted = solver.solve(face.rows, got);
      const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - begin).count();
      if (!accepted) continue;
      times.push_back(us);
      ++served;
      const double error = std::max(
          {twalker::relative_inf_error(got.g, face.g),
           twalker::relative_inf_error(got.h, face.h),
           twalker::relative_inf_error(got.ua, face.ua),
           twalker::relative_inf_error(got.uc, face.uc)});
      max_error = std::max(max_error, error);
      max_dres = std::max(max_dres, got.dres);
      max_piece = std::max(max_piece, got.piece_residual);
      accurate += std::isfinite(error) && error <= 1e-10;
      if (!(std::isfinite(error) && error <= 1e-10)) {
        good = false;
        std::cerr << stem(path) << " face="
                  << (&face - fixture.faces.data()) << " error=" << error
                  << " rcond=" << solver.rcond() << " dres=" << got.dres
                  << " piece=" << got.piece_residual << '\n';
      }
    }
    std::sort(times.begin(), times.end());
    const double median = times.empty() ? 0.0 : times[times.size() / 2];
    const auto &stats = solver.stats();
    std::cout << "{\"model\":\"" << stem(path) << "\",\"faces\":"
              << fixture.faces.size() << ",\"served\":" << served
              << ",\"accurate\":" << accurate << ",\"max_error\":"
              << max_error << ",\"median_us\":" << median
              << ",\"max_dres\":" << max_dres
              << ",\"max_piece\":" << max_piece
              << ",\"rcond\":" << solver.rcond()
              << ",\"rebuilds\":" << stats.rebuilds
              << ",\"updates\":" << stats.updates
              << ",\"downdates\":" << stats.downdates
              << ",\"declines\":" << stats.declines
              << ",\"refinements\":" << stats.refinements
              << ",\"guarded_attempts\":" << stats.guarded_attempts
              << ",\"guarded_accepts\":" << stats.guarded_accepts
              << ",\"guarded_declines\":" << stats.guarded_declines
              << ",\"extended_refinements\":"
              << stats.extended_refinements
              << ",\"worst_forward_bound\":"
              << stats.worst_accepted_forward_bound
              << ",\"worst_tail_bound\":"
              << stats.worst_accepted_tail_bound
              << ",\"worst_contraction\":"
              << stats.worst_accepted_contraction << "}\n";
    total_served += served;
  }
  if (min_served >= 0 && total_served < min_served) {
    std::cerr << "coverage floor: served " << total_served
              << " faces, require " << min_served << '\n';
    good = false;
  }
  return good ? 0 : 1;
}
