#include "fixture.hpp"
#include "kernel_projection.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<double> parse_times(const std::string &text) {
  std::vector<double> result;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) result.push_back(std::stod(token));
  return result;
}

void emit(const std::string &path, const char *lane,
          const twalker::revised::KernelProjectionResult &r) {
  std::cout << "{\"fixture\":\"" << path << "\",\"lane\":\"" << lane
            << "\",\"t\":" << r.t << ",\"status\":\"" << r.status
            << "\",\"rank\":" << r.rank << ",\"nullity\":" << r.nullity
            << ",\"support\":" << r.support
            << ",\"active_core\":" << r.active_core
            << ",\"events\":" << r.stats.events
            << ",\"drops\":" << r.stats.drops
            << ",\"exchanges\":" << r.stats.dependent_exchanges
            << ",\"errors\":[" << r.equality_error << ','
            << r.nonnegative_error << ',' << r.stationarity_error << ','
            << r.complementarity_error << ',' << r.dual_error << ']'
            << ",\"ms\":{\"total\":" << r.stats.total_ms
            << ",\"nullspace\":" << r.stats.nullspace_ms
            << ",\"update\":" << r.stats.active_update_ms
            << ",\"solve\":" << r.stats.active_solve_ms
            << ",\"state\":" << r.stats.state_ms
            << ",\"pricing\":" << r.stats.pricing_ms
            << ",\"certificate\":" << r.stats.certificate_ms << "}"
            << ",\"qr\":{\"insertions\":" << r.stats.qr_insertions
            << ",\"deletions\":" << r.stats.qr_deletions
            << ",\"rebuilds\":" << r.stats.qr_rebuilds
            << "},\"refinements\":" << r.stats.equality_refinements
            << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: kernel_projection_probe fixture.twfx [t,...] [both]\n";
    return 2;
  }
  const auto times = parse_times(argc > 2 ? argv[2] : "0,256,1000000");
  const bool both = argc > 3 && std::string(argv[3]) == "both";
  std::cout << std::setprecision(17);
  try {
    const auto fixture = twalker::read_fixture(argv[1]);
    twalker::revised::KernelProjector projector(fixture);
    bool good = true;
    for (double t : times) {
      const auto maintained = projector.solve(t, -1.0, true);
      emit(argv[1], "maintained", maintained);
      good = good && maintained.certified;
      if (both) {
        const auto cold = projector.solve(t, -1.0, false);
        emit(argv[1], "cold", cold);
        good = good && cold.certified;
      }
    }
    return good ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
