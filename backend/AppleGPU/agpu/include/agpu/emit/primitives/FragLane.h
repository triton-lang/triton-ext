// FragLane - which element of a simdgroup 8x8 fragment a lane holds.
//
// `simdgroup_float8x8::thread_elements()` exposes two scalars per lane, at a
// fixed lane -> (row, column) mapping the toolchain does not document.
// Measured on Apple silicon; same for `simdgroup_half8x8` and
// `simdgroup_bfloat8x8`:
//
//   row(lane)    = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2)
//   col(lane, i) = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2) | i
#ifndef AGPU_EMIT_FRAG_LANE_H
#define AGPU_EMIT_FRAG_LANE_H

#include "agpu/msl/Context.h"

#include <cstdint>

namespace agpu {

// Elements of an 8x8 fragment each lane of a 32-thread simdgroup holds.
inline constexpr int64_t kFragElemsPerLane = 2;

inline constexpr int64_t fragLaneRow(int64_t lane) {
  return ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
}

inline constexpr int64_t fragLaneCol(int64_t lane, int64_t elem) {
  return ((lane & 1) << 1) | (((lane >> 3) & 1) << 2) | elem;
}

inline msl::Expr *fragLaneRowExpr(msl::Context &c, const msl::Str &laneId) {
  return c.binary(
      msl::BinOp::Or,
      c.binary(msl::BinOp::And,
               c.binary(msl::BinOp::Shr, c.var(laneId), c.lit(1)), c.lit(3)),
      c.binary(msl::BinOp::Shl,
               c.binary(msl::BinOp::And,
                        c.binary(msl::BinOp::Shr, c.var(laneId), c.lit(4)),
                        c.lit(1)),
               c.lit(2)));
}

inline msl::Expr *fragLaneColExpr(msl::Context &c, const msl::Str &laneId,
                                  int64_t elem) {
  msl::Expr *base = c.binary(
      msl::BinOp::Or,
      c.binary(msl::BinOp::Shl,
               c.binary(msl::BinOp::And, c.var(laneId), c.lit(1)), c.lit(1)),
      c.binary(msl::BinOp::Shl,
               c.binary(msl::BinOp::And,
                        c.binary(msl::BinOp::Shr, c.var(laneId), c.lit(3)),
                        c.lit(1)),
               c.lit(2)));
  return elem ? c.binary(msl::BinOp::Or, base, c.lit(elem)) : base;
}

inline msl::Expr *fragElemExpr(msl::Context &c, const msl::Str &frag,
                               int64_t elem) {
  return c.subscript(c.member(c.var(frag), "thread_elements()"), c.lit(elem));
}

} // namespace agpu

#endif // AGPU_EMIT_FRAG_LANE_H
