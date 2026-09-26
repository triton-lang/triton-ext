// Shapes that more than one suite builds.
#ifndef AGPU_TEST_FIXTURES_H
#define AGPU_TEST_FIXTURES_H

#include "agpu/emit/LayoutExpr.h"
#include "agpu/plan/AccessWidth.h"
#include "agpu/plan/DotPlan.h"
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

// A half-precision GEMM with nothing elected: the shape every dot suite
// starts from before setting the one fact it is about.
inline agpu::DotFacts gemm(std::int64_t M, std::int64_t N, std::int64_t K,
                           std::int64_t warps = 4, std::int64_t Bd = 1) {
  agpu::DotFacts f;
  f.M = M;
  f.N = N;
  f.K = K;
  f.Bd = Bd;
  f.rank = Bd > 1 ? 3 : 2;
  f.aElemBytes = 2;
  f.bElemBytes = 2;
  f.numWarps = warps;
  return f;
}

// C under `apple_mma` with warpsPerCTA [gM, gN]: warps interleave at
// fragment granularity, low warp bits along columns, repetitions above.
inline void landIn(agpu::DotFacts &f, std::int64_t gM, std::int64_t gN) {
  agpu::LayoutBasis row, col;
  row.lane = {0, 1, 2, 0, 4};
  col.lane = {2, 0, 0, 4, 0};
  row.reg.push_back(0);
  col.reg.push_back(1);
  for (std::int64_t s = agpu::kSgFragDim; s < gN * agpu::kSgFragDim; s <<= 1) {
    row.warp.push_back(0);
    col.warp.push_back((std::int32_t)s);
  }
  for (std::int64_t s = agpu::kSgFragDim; s < gM * agpu::kSgFragDim; s <<= 1) {
    row.warp.push_back((std::int32_t)s);
    col.warp.push_back(0);
  }
  for (std::int64_t s = gN * agpu::kSgFragDim; s < f.nT() * agpu::kSgFragDim;
       s <<= 1) {
    row.reg.push_back(0);
    col.reg.push_back((std::int32_t)s);
  }
  for (std::int64_t s = gM * agpu::kSgFragDim; s < f.mT() * agpu::kSgFragDim;
       s <<= 1) {
    row.reg.push_back((std::int32_t)s);
    col.reg.push_back(0);
  }
  f.cDims = {row, col};
  f.cRegs = 2 * (f.mT() / gM) * (f.nT() / gN);
}

using Bits = std::vector<std::int32_t>;

// A basis named by the levels it uses, so a case spells only those.
inline agpu::LayoutBasis basis(Bits reg, Bits lane, Bits warp = {},
                               Bits block = {}) {
  agpu::LayoutBasis b;
  b.reg = std::move(reg);
  b.lane = std::move(lane);
  b.warp = std::move(warp);
  b.block = std::move(block);
  return b;
}

// One warp's worth of lanes laid along the last dim, registers above them.
inline agpu::LayoutBasis laneMajor(Bits reg = {32}) {
  return basis(std::move(reg), {1, 2, 4, 8, 16});
}

// A fragment read from threadgroup memory, as `loadFrag` spells its first
// element: from `addr` plus the lane's offset in a row of `ld`.
inline std::string laneLoad(const std::string &frag, const std::string &addr,
                            std::int64_t ld, const std::string &elem = "half") {
  return frag + ".thread_elements()[0] = (*(threadgroup " + elem + "2 *)(" +
         addr + " + (fragRow * " + std::to_string(ld) + " + fragCol))).x;";
}

// Once per fragment read from threadgroup memory.
inline constexpr const char *kLaneLoad =
    ".thread_elements()[0] = (*(threadgroup";

} // namespace agpu_test

#endif // AGPU_TEST_FIXTURES_H
