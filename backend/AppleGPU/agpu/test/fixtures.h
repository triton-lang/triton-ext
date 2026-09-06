// Shapes that more than one suite builds.
#ifndef AGPU_TEST_FIXTURES_H
#define AGPU_TEST_FIXTURES_H

#include "agpu/emit/LayoutExpr.h"
#include "agpu/plan/AccessWidth.h"
#include "agpu/plan/RebindPlan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace agpu_test {

inline bool has(const std::string &hay, const std::string &needle) {
  return hay.find(needle) != std::string::npos;
}

// Overlapping count: advances by one.
inline int countOf(const std::string &s, const std::string &needle) {
  int n = 0;
  for (std::size_t i = s.find(needle); i != std::string::npos;
       i = s.find(needle, i + 1))
    ++n;
  return n;
}

// Register i sits at 1<<i along `dim`.
inline agpu::RegBases contiguousBases(int n, int dim = 1, int rank = 2) {
  agpu::RegBases b;
  for (int i = 0; i < n; ++i) {
    std::vector<std::int32_t> row((std::size_t)rank, 0);
    row[(std::size_t)dim] = 1 << i;
    b.push_back(row);
  }
  return b;
}

// Every coordinate of a shape, in row-major register order.
inline std::vector<agpu::RegCoord> coordsOfShape(const agpu::RegCoord &shape) {
  std::vector<agpu::RegCoord> out;
  agpu::RegCoord c(shape.size(), 0);
  for (;;) {
    out.push_back(c);
    int d = (int)shape.size() - 1;
    for (; d >= 0; --d) {
      if (++c[(std::size_t)d] < shape[(std::size_t)d])
        break;
      c[(std::size_t)d] = 0;
    }
    if (d < 0)
      break;
  }
  return out;
}

} // namespace agpu_test

#endif // AGPU_TEST_FIXTURES_H
