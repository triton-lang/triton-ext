// OperandSource - where a dot operand's fragments are read from.
#ifndef AGPU_EMIT_OPERAND_SOURCE_H
#define AGPU_EMIT_OPERAND_SOURCE_H

#include "agpu/core/Units.h"
#include "agpu/emit/primitives/SlotExpr.h"
#include "agpu/emit/primitives/Stride.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/WarpSlots.h"

namespace agpu {

struct OperandSource {
  // A's fragment index selects a row band (scales by row stride); B's selects
  // a column band (scales by one element).
  enum class FragAxis { Rows, Cols };

  msl::Str buffer; // pool buffer or device pointer
  Stride leadingDim;
  FragAxis fragAxis = FragAxis::Rows;

  // Corner of the addressed window, in elements. Zero for a staged tile.
  int64_t rowOrigin = 0;
  int64_t colOrigin = 0;

  msl::AddrSpace space = msl::AddrSpace::Threadgroup;

  // Offset along this operand's fragment axis alone: no origins, no K term.
  msl::Expr *axisOffsetOf(msl::Context &c, SlotCoord pos,
                          const msl::Str &warpId) const {
    msl::Expr *frags =
        c.binary(msl::BinOp::Mul, coordOf(c, pos, warpId), c.lit(kSgFragDim));
    return fragAxis == FragAxis::Cols ? frags : leadingDim.scale(c, frags);
  }

  // Where fragment `pos` starts: the corner-adjusted offset along both axes.
  // The K term arrives separately, via kOffsetOf.
  msl::Expr *fragOffsetOf(msl::Context &c, SlotCoord pos,
                          const msl::Str &warpId) const {
    msl::Expr *frags =
        c.binary(msl::BinOp::Mul, coordOf(c, pos, warpId), c.lit(kSgFragDim));
    msl::Expr *off;
    if (fragAxis == FragAxis::Cols)
      off = c.binary(msl::BinOp::Add,
                     c.binary(msl::BinOp::Add, frags, c.lit(colOrigin)),
                     leadingDim.scale(c, rowOrigin));
    else
      off = c.binary(msl::BinOp::Add,
                     leadingDim.scale(
                         c, c.binary(msl::BinOp::Add, frags, c.lit(rowOrigin))),
                     c.lit(colOrigin));
    return off;
  }

  int64_t sliceStride = 0;

  // One element's offset, for a consumer that reads scalars over fragments.
  // `batch` (null for an unbatched dot) selects a slice by `sliceStride`.
  msl::Expr *elemOffsetOf(msl::Context &c, msl::Expr *row, msl::Expr *col,
                          msl::Expr *batch = nullptr) const {
    msl::Expr *off = c.binary(
        msl::BinOp::Add,
        leadingDim.scale(c, c.binary(msl::BinOp::Add, row, c.lit(rowOrigin))),
        c.binary(msl::BinOp::Add, col, c.lit(colOrigin)));
    if (!batch)
      return off;
    return c.binary(msl::BinOp::Add,
                    c.binary(msl::BinOp::Mul, batch, c.lit(sliceStride)), off);
  }

  // The K step's contribution, which moves along the other axis.
  msl::Expr *kOffsetOf(msl::Context &c, msl::Expr *kFrags) const {
    if (fragAxis == FragAxis::Cols)
      return leadingDim.scale(c, kFrags);
    return kFrags;
  }

  // Fragment rows per warp band, when the pointer already points at this
  // warp's band. Zero means the buffer covers the whole tile.
  int64_t bandFrags = 0;

  // The row this coordinate reads, which is band-relative when banded.
  SlotCoord rowOf(SlotCoord pos) const {
    if (bandFrags <= 0 || !pos.isConst())
      return pos;
    return SlotCoord::fixed(pos.constant % bandFrags);
  }
};

} // namespace agpu

#endif // AGPU_EMIT_OPERAND_SOURCE_H
