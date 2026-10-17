// Shapes that more than one suite builds.
#ifndef AGPU_TEST_FIXTURES_H
#define AGPU_TEST_FIXTURES_H

#include "agpu/plan/RebindPlan.h"

#include <cstdint>
#include <vector>

namespace agpu_test {

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
