// SlotExpr - a SlotCoord spelled as an expression and the warp-block loop.
#ifndef AGPU_EMIT_SLOT_EXPR_H
#define AGPU_EMIT_SLOT_EXPR_H

#include "agpu/core/TileView.h"
#include "agpu/core/Units.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/WarpSlots.h"

#include <initializer_list>
#include <optional>
#include <vector>

namespace agpu {

// The one place a SlotCoord becomes an expression.
inline msl::Expr *coordOf(msl::Context &c, SlotCoord s,
                          const msl::Str &warpId) {
  if (s.isConst())
    return c.lit(s.constant);
  msl::Expr *t = c.var(warpId);
  if (s.warpDiv > 1)
    t = c.binary(msl::BinOp::Div, t, c.lit(s.warpDiv));
  if (s.warpMod > 0)
    t = c.binary(msl::BinOp::Rem, t, c.lit(s.warpMod));
  return c.binary(msl::BinOp::Add,
                  c.binary(msl::BinOp::Mul, t, c.lit(s.warpScale)),
                  c.lit(s.constant));
}

// A fragment position as an element offset, through the view that owns the
// strides: `frag(dim) * 8 * stride(dim)`, summed, plus an optional K term.
inline msl::Expr *fragOffset(msl::Context &c, const TileView &v,
                             std::initializer_list<SlotCoord> frag,
                             msl::Expr *kTerm, const msl::Str &warpId) {
  msl::Expr *off = nullptr;
  int d = 0;
  for (SlotCoord pos : frag) {
    msl::Expr *term = c.binary(
        msl::BinOp::Mul,
        c.binary(msl::BinOp::Mul, coordOf(c, pos, warpId), c.lit(kSgFragDim)),
        c.lit(v.strideAt(d)));
    off = off ? c.binary(msl::BinOp::Add, off, term) : term;
    ++d;
  }
  if (!off)
    off = c.lit(0);
  return kTerm ? c.binary(msl::BinOp::Add, off, kTerm) : off;
}

// One block per `blockCount()`, each emitted into its own guard. The program
// says which warp a block belongs to, or that it serves them all (null cond).
template <class EmitBlock>
inline void emitWarpBlocks(msl::Context &c, msl::Block &body,
                           const WarpProgram &prog, const WarpGrid &grid,
                           const msl::Str &warpId, EmitBlock emitBlock) {
  for (int64_t w = 0; w < prog.blockCount(grid.numWarps); ++w) {
    const std::vector<WarpSlot> slots =
        prog.slots(w, grid.mT, grid.nT, grid.numWarps);
    if (slots.empty())
      continue;

    msl::Block inner;
    emitBlock(inner, slots, w);

    // Idle hardware warps beyond what the program planned for are fenced off;
    // otherwise they compute a fragment the plan never assigned.
    const std::optional<int64_t> only = prog.guardWarp(w);
    msl::Expr *cond = nullptr;
    if (only)
      cond = c.binary(msl::BinOp::Eq, c.var(warpId), c.lit(*only));
    else if (grid.guardsIdleWarps())
      cond = c.binary(msl::BinOp::Lt, c.var(warpId), c.lit(grid.numWarps));
    c.guardedInto(body, cond, std::move(inner));
  }
}

} // namespace agpu

#endif // AGPU_EMIT_SLOT_EXPR_H
