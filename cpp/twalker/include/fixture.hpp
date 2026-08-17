#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace twalker {

struct OracleFace {
  double t = 0.0;
  std::vector<std::uint32_t> rows;
  std::vector<double> g, h, ua, uc;
  double dres = 0.0;
};

struct Fixture {
  std::uint32_t n = 0, m = 0, nnz = 0;
  std::vector<std::uint64_t> indptr;
  std::vector<std::uint32_t> indices;
  std::vector<double> values, b, d;
  std::vector<std::uint8_t> post_seed_support;
  double t0 = 0.0;
  std::vector<OracleFace> faces;
};

template <typename T>
inline T read_scalar(std::ifstream &stream) {
  T value{};
  stream.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!stream) throw std::runtime_error("truncated fixture scalar");
  return value;
}

template <typename T>
inline std::vector<T> read_vector(std::ifstream &stream, std::size_t count) {
  std::vector<T> result(count);
  if (count) {
    stream.read(reinterpret_cast<char *>(result.data()),
                static_cast<std::streamsize>(count * sizeof(T)));
  }
  if (!stream) throw std::runtime_error("truncated fixture vector");
  return result;
}

inline Fixture read_fixture(const std::string &path) {
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                "TWFX v1 requires a little-endian host");
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open fixture: " + path);
  char magic[4];
  stream.read(magic, 4);
  if (!stream || std::memcmp(magic, "TWFX", 4) != 0)
    throw std::runtime_error("bad fixture magic: " + path);
  const auto version = read_scalar<std::uint32_t>(stream);
  if (version != 1) throw std::runtime_error("unsupported fixture version");

  Fixture fixture;
  fixture.n = read_scalar<std::uint32_t>(stream);
  fixture.m = read_scalar<std::uint32_t>(stream);
  fixture.nnz = read_scalar<std::uint32_t>(stream);
  const auto face_count = read_scalar<std::uint32_t>(stream);
  fixture.indptr = read_vector<std::uint64_t>(stream, fixture.n + 1);
  fixture.indices = read_vector<std::uint32_t>(stream, fixture.nnz);
  fixture.values = read_vector<double>(stream, fixture.nnz);
  fixture.b = read_vector<double>(stream, fixture.n);
  fixture.d = read_vector<double>(stream, fixture.m);
  fixture.post_seed_support = read_vector<std::uint8_t>(stream, fixture.n);
  fixture.t0 = read_scalar<double>(stream);
  fixture.faces.reserve(face_count);
  for (std::uint32_t k = 0; k < face_count; ++k) {
    OracleFace face;
    face.t = read_scalar<double>(stream);
    const auto rows = read_scalar<std::uint32_t>(stream);
    face.rows = read_vector<std::uint32_t>(stream, rows);
    face.g = read_vector<double>(stream, rows);
    face.h = read_vector<double>(stream, rows);
    face.ua = read_vector<double>(stream, fixture.m);
    face.uc = read_vector<double>(stream, fixture.m);
    face.dres = read_scalar<double>(stream);
    fixture.faces.push_back(std::move(face));
  }
  if (stream.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("unexpected trailing fixture bytes");
  return fixture;
}

}  // namespace twalker
