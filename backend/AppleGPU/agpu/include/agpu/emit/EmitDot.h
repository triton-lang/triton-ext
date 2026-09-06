// EmitDot.h - the dot entry point: facts in, statements out.
//
// Holds no policy: every decision below is read from the `Plan`.
#ifndef AGPU_EMIT_DOT_H
#define AGPU_EMIT_DOT_H

#include "agpu/emit/EmitDirect.h"
#include "agpu/emit/EmitPanel.h"
#include "agpu/emit/EmitPoison.h"
#include "agpu/emit/EmitScalar.h"
#include "agpu/plan/DotPlan.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/PanelSchedule.h"

#include <algorithm>

namespace agpu {

// What the caller supplies that the plan cannot know: the names of the pool
// pointers and the registers and how a coordinate is spelled.
struct DotInputs {
  DirectNames direct;
  PanelNames panel;

  // Where A and B are read from on the direct path.
  OperandSource a, b;

  // Per-tile staging and readback, for the panel path.
  std::function<PanelInputs(const PanelTile &)> tileInputs;

  // C's readback for the direct path, per band of rows. Null means the caller
  // addresses `poolC` itself and wants nothing read back.
  std::function<ReadbackInputs(const Range &)> readbackFor;

  // Where a fused dot's C stores straight to device (`Plan::storesCDirect`
  // says whether it did). Empty base means the drain goes through the pool.
  DeviceStoreTarget cStore;

  // The elementwise chain the walk folded between the loop result and that
  // store, applied by the drain per element. Only meaningful with a non-empty
  // `cStore`.
  std::vector<DrainStep> cSteps;

  // One coordinate source per operand: A, B and C have different layouts.
  // `PanelCoords::forAll` covers a shared layout.
  PanelCoords coords;

  // `emitKernel` decides this from the size budget and passes it down.
  bool rollK = false;
};

// Read off the plan.
inline WarpGrid gridOf(const Plan &p) {
  WarpGrid g =
      warpGridFor(p.facts, p.cBandRows() < p.cStagedView().extentAt(0));
  g.cover = p.cover;
  return g;
}

// A loop whose body accumulates into simdgroup fragments it does not own.
// Drained once after the loop: through the pool and a readback, or straight
// to the device tensor when the plan set `storesCDirect`.
//
// The warp program is planned once here and shared by the loop body's MMAs
// and the stores, so they refer to the same fragments.
inline Decision
emitFusedLoop(msl::Context &c, msl::Block &body, const Plan &p,
              const DirectNames &nm,
              const std::function<ReadbackInputs(const Range &)> &readbackFor,
              const CoordSource &cCoords, const DeviceStoreTarget &cStore,
              const std::vector<DrainStep> &cSteps,
              const std::function<Decision()> &emitLoop) {
  const bool direct = p.storesCDirect();
  if (direct ? !cStore.ok() : !readbackFor)
    return Decision::failed();

  const TileView cv = p.cStagedView();
  const WarpGrid grid = gridOf(p);
  const WarpProgram prog = planWarpProgram(grid);

  // A direct drain needs no result registers: the fragments are the result.
  ReadbackInputs back;
  if (!direct) {
    back = readbackFor(Range{0, cv.extentAt(0)});
    if (back.empty())
      return Decision::declined("emitFusedLoop",
                                "C's layout stopped resolving");

    // Zero-filled because a ragged tile's readback does not assign its edge
    // registers.
    for (const msl::Str &n : back.names)
      body.push_back(poisonDecl(c, n, back.regElem));
  }

  // Unguarded, once per accumulator index: a PerWarp program emits its blocks
  // inside `if (warp == k)` guards and a declaration inside one guard would
  // be invisible to the loop body's MMAs.
  {
    std::vector<WarpSlot> decls;
    for (int64_t w = 0; w < prog.blockCount(grid.numWarps); ++w)
      for (const WarpSlot &s : prog.slots(w, grid.mT, grid.nT, grid.numWarps))
        if (std::none_of(decls.begin(), decls.end(),
                         [&](const WarpSlot &d) { return d.acc == s.acc; }))
          decls.push_back(s);
    emitAccumDecls(c, body, decls, nm);
  }

  if (const Decision d = emitLoop(); !d.ok())
    return d;

  // Fragments straight to the device tensor. A barrier is needed only when
  // the drain borrows the pool for edge scratch, separating other warps'
  // final-iteration operand reads from the first scratch write. Outside the
  // warp blocks: a barrier under a warp guard is not reached by every thread.
  if (direct) {
    if (!cStore.edgeScratch.empty() && cStore.bounded())
      body.push_back(c.barrier());
    emitWarpBlocks(
        c, body, prog, grid, nm.warpId,
        [&](msl::Block &inner, const std::vector<WarpSlot> &slots, int64_t) {
          emitAccumDeviceStores(c, inner, slots, cStore, cSteps, nm);
        });
    return Decision::emitted();
  }

  if (p.readsBackByRename()) {
    emitWarpBlocks(
        c, body, prog, grid, nm.warpId,
        [&](msl::Block &inner, const std::vector<WarpSlot> &, int64_t) {
          emitFragmentReadback(c, inner, back.plan, back.names, back.bases,
                               back.regElem, directAccName(nm));
        });
    return Decision::emitted();
  }

  // A drain is the one pool write no staging barrier precedes, so it opens
  // its own epoch.
  body.push_back(c.barrier());

  emitWarpBlocks(c, body, prog, grid, nm.warpId,
                 [&](msl::Block &inner, const std::vector<WarpSlot> &slots,
                     int64_t) { emitAccumStores(c, inner, slots, cv, nm, 0); });

  // A thread's registers come from slots other warps stored.
  body.push_back(c.barrier());
  emitReadback(c, body, cv, nm.poolC, back.actions, back.names, back.bases,
               cCoords, back.elem, back.regElem);
  return Decision::emitted();
}

// One `switch` over the strategy the plan chose.
inline Decision emitDot(msl::Context &c, msl::Block &body, const Plan &p,
                        const DotInputs &in) {
  switch (p.kind) {
  case Plan::Kind::Unsupported:
    return dotDecision(p);

  case Plan::Kind::Scalar: {
    // `simdgroup_matrix` rejects integer elements outright, so no MMA is
    // available and the plan chose a per-thread K loop over the staged
    // operands.
    if (!in.readbackFor)
      return Decision::failed();
    const ReadbackInputs back =
        in.readbackFor(Range{0, p.cStagedView().extentAt(0)});
    if (back.empty())
      return Decision::declined("emitDot", "C's layout stopped resolving");
    emitScalarDot(c, body, p, in.a, in.b, back, in.coords.c, in.direct);
    return Decision::emitted();
  }

  case Plan::Kind::Direct:
  case Plan::Kind::Fused: {
    // Both use the same emitter: a fused dot is the accumulator declarations
    // hoisted out of a K loop with this inside it.
    const TileView cv = p.cStagedView();
    const int64_t bandRows = p.cBandRows();

    const WarpGrid grid = gridOf(p);
    const WarpProgram prog = planWarpProgram(grid);

    DirectInputs di;
    di.a = in.a;
    di.b = in.b;
    di.kT = p.facts.kT();
    di.rollK = in.rollK;

    // A Fused dot's accumulators belong to the loop around it: declared
    // before it, drained after it.
    emitDirectDot(c, body, prog, grid, di, cv, in.direct, bandRows,
                  in.readbackFor, in.coords.c, DotPassSchedule::of(p));
    return Decision::emitted();
  }

  case Plan::Kind::Panel: {
    if (!in.tileInputs)
      return Decision::failed();
    FragReuse reuse(body);
    // Per tile: the grid is the tile's fragment counts and a ragged final
    // tile has fewer.
    forEachPanelTile(p.facts, p.panel().panel, [&](const PanelTile &t) {
      const WarpGrid grid =
          panelWarpGrid(t, warpsFor(p.facts), p.facts.numWarps);
      // `rollK` is the kernel's: overwrite whatever the callback returned.
      PanelInputs ti = in.tileInputs(t);
      ti.rollK = in.rollK;
      emitPanelTile(c, body, t, in.panel, ti, in.coords, grid,
                    planWarpProgram(grid), &reuse);
    });
    return Decision::emitted();
  }
  }
  return Decision::failed();
}

} // namespace agpu

#endif // AGPU_EMIT_DOT_H
