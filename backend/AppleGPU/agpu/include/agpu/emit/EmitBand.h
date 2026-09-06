// EmitBand.h - the banded threadgroup round-trip: scatter the registers in a
// band, barrier, gather them back in the destination's order, repeat.

#ifndef AGPU_EMIT_BAND_H
#define AGPU_EMIT_BAND_H

#include "agpu/core/CoordGuard.h"
#include "agpu/emit/Emit.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Equal.h"
#include "agpu/plan/BandPlan.h"

namespace agpu {

// One register's participation in the round trip.
struct BandReg {
  int reg = 0;
  msl::Expr *offset = nullptr; // flat element offset within the tile
  CoordRange range;            // the offsets this register can take
};

struct BandNames : ThreadNames {
  msl::Str buffer = "sc";
  msl::Str flat = "f";
};

// Values a round trip moves. `srcValues[i]` is scattered from; `dstNames[i]`
// is gathered into. `scatterGuard` elects which threads write, or null for
// all of them; the gather is always unguarded.
struct BandIO {
  msl::SmallVec<BandReg, 8> src, dst;
  msl::SmallVec<msl::Expr *, 8> srcValues;
  msl::SmallVec<msl::Str, 8> dstNames;
  msl::Expr *scatterGuard = nullptr;
};

// Names the "all threads scatter" choice at a call site.
inline msl::Expr *const noScatterGuard = nullptr;

// Shared by the scatter and gather halves so they agree.
inline CoordGuard bandGuard(const CoordRange &range, BandPlan::Band b) {
  return planGuard({range}, {CoordWindow{range.dim, b.lo, b.hi}});
}

enum class BandDir {
  Scatter,
  Gather,
};

// `{ int f = off; if (f >= lo && f < hi) buf[f - lo] = v; }`. The guard picks
// the shape: Dead emits nothing (returns null), Unguarded emits the bare
// assignment, Needed emits the block.
inline msl::Stmt *bandAccess(msl::Context &c, const BandReg &r,
                             BandPlan::Band b, const CoordGuard &g,
                             const msl::Str &buf, const msl::Str &flatName,
                             msl::Expr *reg, BandDir dir) {
  if (g.isDead())
    return nullptr;

  auto move = [&](msl::Expr *slot) {
    return dir == BandDir::Scatter ? c.assign(slot, reg) : c.assign(reg, slot);
  };
  // `- 0` folds away in the builder, so band 0 needs no special case.
  auto slotAt = [&](msl::Expr *off) {
    return c.subscript(c.var(buf), c.binary(msl::BinOp::Sub, off, c.lit(b.lo)));
  };

  if (!g.needsTest())
    return move(slotAt(r.offset));

  // Name the offset once: the test and the index both use it.
  msl::Block blk;
  blk.push_back(c.declStmt(msl::Context::i32(), flatName, r.offset));

  msl::Expr *cond = guardCond(c, g, [&](int) { return c.var(flatName); });

  blk.push_back(c.guarded(cond, move(slotAt(c.var(flatName)))));
  return c.scope(std::move(blk));
}

// Every register reads back from the slot it wrote, so the round trip can be
// skipped and dst bound straight to src. Ask before banding.
inline bool roundTripIsIdentity(const BandIO &io) {
  if (io.src.size() != io.dst.size())
    return false;
  for (std::size_t i = 0; i < io.src.size(); ++i)
    if (!msl::exprsEqual(io.src[i].offset, io.dst[i].offset))
      return false;
  return true;
}

enum class RoundTrip {
  Emitted,
  Elided, // identity: bind dst to src, no pool traffic
};

// `plan.bandCount()` is 1 for a tile that fits: the unbanded case is one
// iteration with every guard Unguarded.
inline RoundTrip emitBandRoundTrip(msl::Context &c, msl::Block &body,
                                   const BandPlan &plan, const BandIO &io,
                                   const BandNames &nm) {
  if (roundTripIsIdentity(io))
    return RoundTrip::Elided;

  // Hard barriers on both sides of both halves. A band whose registers are
  // all Dead emits no access and a foldable barrier pair would then collapse
  // across the band boundary, letting a thread read a slot another is still
  // writing.
  auto half = [&](const msl::SmallVec<BandReg, 8> &regs, BandPlan::Band b,
                  BandDir dir, auto &&regExpr) {
    // Outside any election: a barrier inside a guard some threads fail hangs.
    body.push_back(c.hardBarrier());
    msl::Block accesses;
    msl::Block &into =
        (dir == BandDir::Scatter && io.scatterGuard) ? accesses : body;
    for (std::size_t i = 0; i < regs.size(); ++i) {
      const CoordGuard g = bandGuard(regs[i].range, b);
      if (msl::Stmt *s =
              bandAccess(c, regs[i], b, g, nm.buffer, nm.flat, regExpr(i), dir))
        into.push_back(s);
    }
    if (&into == &accesses && !accesses.empty())
      body.push_back(c.guarded(io.scatterGuard, c.scope(std::move(accesses))));
  };

  for (int64_t bi = 0; bi < plan.bandCount(); ++bi) {
    const BandPlan::Band b = plan.bandAt(bi);
    half(io.src, b, BandDir::Scatter,
         [&](std::size_t i) { return io.srcValues[i]; });
    half(io.dst, b, BandDir::Gather,
         [&](std::size_t i) { return c.var(io.dstNames[i]); });
  }
  body.push_back(c.hardBarrier());
  return RoundTrip::Emitted;
}

} // namespace agpu

#endif // AGPU_EMIT_BAND_H
