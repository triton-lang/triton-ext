// Staging a dot: pool regions, accumulator registers, operand staging and
// the panel inputs.
#include "AgpuDotChain.h"
#include "AgpuEmitter.h"
#include "AgpuLog.h"

#include "agpu/emit/EmitPoison.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

void AgpuEmitter::declareAccumulatorRegisters(const DotOperands &ops,
                                              const agpu::Plan &plan) {
  // Declared up front for `cOut`: a ragged tile assigns under a guard and
  // leaves edge registers at their declared value. For a fused dot,
  // `emitFusedLoop` declares them before the loop instead.
  if (plan.accumulatorsOutlivePass())
    return;
  for (int64_t r = 0; r < registerCount(ops.cOutTy); ++r)
    cur_->push_back(agpu_.context().declStmt(
        agpu::mslTypeOf(ops.shape.cElem), accName(ops.cOut, r),
        agpu::poisonValue(agpu_.context(), ops.shape.cElem)));
}

void AgpuEmitter::tagDotNames(agpu::DotInputs &in) {
  // All dots declare into the same kernel scope, so tag names after the first.
  if (body_.dotSeq > 0) {
    const std::string tag = "d" + std::to_string(body_.dotSeq) + "_";
    in.direct.acc = in.panel.acc = tag + "acc";
    in.direct.frag = tag + "f";
    in.panel.fragA = tag + "fa";
    in.panel.fragB = tag + "fb";
  }
  ++body_.dotSeq;
}

agpu::Decision AgpuEmitter::namePoolRegions(const agpu::Plan &plan,
                                            agpu::DotInputs &in) {
  // B and C are regions of the threadgroup buffer carved by `walkOp`; this
  // handler only names them.
  const agpu::MmaNames mnm;
  const am::Str bBuf = body_.pool.use(mnm.poolB);
  const bool cThroughPool = plan.cThroughPool();
  const am::Str cBuf = cThroughPool ? body_.pool.use(mnm.poolC) : am::Str();
  if (bBuf.empty() || (cThroughPool && cBuf.empty()))
    return declined("tt.dot",
                    "a pool region this dot stages through was never carved");
  in.direct.poolB = in.panel.poolB = bBuf;
  in.direct.poolC = in.panel.poolC = cBuf;
  if (plan.edgeScratchFits()) {
    const am::Str eBuf = body_.pool.use(mnm.poolE);
    if (eBuf.empty())
      return declined("tt.dot", "the edge scratch region was never carved");
    in.direct.poolE = in.panel.poolE = eBuf;
  }
  return agpu::Decision::emitted();
}

agpu::ElemType AgpuEmitter::stagedElemOf(const agpu::Plan &plan,
                                         const agpu::ElemType &operand) {
  return plan.intThroughFloat ? agpu::f32() : operand;
}

agpu::Decision AgpuEmitter::stageAB(const DotOperands &ops,
                                    const agpu::Plan &plan,
                                    const agpu::ElemType &stagedAElem,
                                    const agpu::ElemType &stagedBElem,
                                    agpu::DotInputs &in) {
  // A is read in place or staged like B, per `f.aDirect`. Neither is staged
  // here when tiles stage per tile: the pool regions hold one tile, so a
  // whole-operand scatter would write past `pB`'s end.
  const bool stagesPerTile = plan.kind == agpu::Plan::Kind::Panel;
  const bool stagePad = plan.padStagedC();
  const agpu::DotFacts &pf = plan.facts;
  const agpu::TileView aStaged =
      agpu::stagedOperandView(pf, pf.M, pf.K, pf.aElemBytes, stagePad);
  const agpu::TileView bStaged = agpu::stagedOperandView(
      pf, pf.K, agpu::fragAlignedExtent(pf.N), pf.bElemBytes, stagePad);

  // Every pool user opens its epoch with a barrier: the previous one (a fused
  // pass's MMAs, a drained dot's readback) leaves its reads unfenced.
  if (!stagesPerTile)
    cur_->push_back(agpu_.context().barrier());

  if (pf.aDirect) {
    const agpu::Decision d = readADirect(ops, in);
    if (!d.ok())
      return d;
  } else {
    const agpu::MmaNames mnm;
    const am::Str aBuf = body_.pool.use(mnm.poolA);
    if (aBuf.empty())
      return declined("tt.dot",
                      "a pool region this dot stages through was never carved");
    in.direct.poolA = in.panel.poolA = aBuf;
    if (!stagesPerTile) {
      // `aStaged` is the plan's staged view; its row stride carries the bank
      // pad that simdgroup_load reads at.
      if (const agpu::Decision d =
              stageWholeTensor(ops.aStage, ops.aStageTy, aBuf, aStaged,
                               stagedAElem, "tt.dot", "an A");
          !d.ok())
        return d;
    }
    in.a.buffer = aBuf;
    in.a.leadingDim = agpu::Stride(aStaged.strideAt(aStaged.rank() - 2));
    if (pf.batched())
      in.a.sliceStride = aStaged.strideAt(0);
  }

  if (!stagesPerTile) {
    if (const agpu::Decision d =
            stageWholeTensor(ops.bStage, ops.bStageTy, in.panel.poolB, bStaged,
                             stagedBElem, "tt.dot", "a B");
        !d.ok())
      return d;
    cur_->push_back(agpu_.context().barrier());
  }

  in.b.buffer = in.panel.poolB;
  in.b.leadingDim = agpu::Stride(bStaged.strideAt(bStaged.rank() - 2));
  if (pf.batched())
    in.b.sliceStride = bStaged.strideAt(0);
  // B is indexed by N, its column axis, so fragment `ni` sits `ni*8`
  // elements across the row.
  in.b.fragAxis = agpu::OperandSource::FragAxis::Cols;
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::stageDotOperands(const DotOperands &ops,
                                             const agpu::Plan &plan,
                                             agpu::DotInputs &in) {
  declareAccumulatorRegisters(ops, plan);
  tagDotNames(in);

  if (const agpu::Decision d = namePoolRegions(plan, in); !d.ok())
    return d;

  const agpu::ElemType stagedAElem = stagedElemOf(plan, ops.shape.aElem);
  const agpu::ElemType stagedBElem = stagedElemOf(plan, ops.shape.bElem);

  if (const agpu::Decision d = stageAB(ops, plan, stagedAElem, stagedBElem, in);
      !d.ok())
    return d;

  // Fragment element follows the operands and overrides `MmaNames`'s half
  // default: a simdgroup_half8x8 loading from a `threadgroup float *` is a
  // type error.
  // `accElem` stays float; the accumulator is always fp32.
  const am::Type aMsl = agpu::mslTypeOf(stagedAElem);
  in.panel.opElem = am::spell(aMsl.scalarKind());
  in.direct.opElem = in.panel.opElem;
  // One source per operand. A/B use their staged types, where the staged
  // register sits.
  in.coords =
      agpu::PanelCoords{coordSourceOf(ops.aStageTy),
                        coordSourceOf(ops.bStageTy), coordSourceOf(ops.cOutTy)};

  setTileInputs(ops, plan, stagedAElem, stagedBElem, in);

  // A drain that stores straight to device has no readback, so C's register
  // layout is never asked for.
  if (plan.storesCDirect())
    return resolveDirectCStore(ops, plan, in);
  return setReadbackFor(ops, plan, in);
}

void AgpuEmitter::setTileInputs(const DotOperands &ops, const agpu::Plan &plan,
                                const agpu::ElemType &stagedAElem,
                                const agpu::ElemType &stagedBElem,
                                agpu::DotInputs &in) {
  const agpu::ValueId aId = ops.aStage, bId = ops.bStage;
  const agpu::ValueId cId = ops.cOut;
  const RankedTensorType aTy = ops.aStageTy, bTy = ops.bStageTy;
  const RankedTensorType cTy = ops.cOutTy;
  // `in.a` is the device source when A is read in place. The staged case
  // rebuilds a per-tile pool source inside the callback: a panel tile's pitch
  // is the tile's.
  // Resolved once here: narrowing a widened register mints a
  // temporary, which belongs in this block rather than the panel loop's.
  in.tileInputs = [this, plan, aId, aTy, bId, bTy, cId, cTy, deviceA = in.a,
                   poolA = in.panel.poolA, cIn = ops.cIn, stagedAElem,
                   stagedBElem, aNames = stagedNamesOf(aId, registerCount(aTy)),
                   bNames = stagedNamesOf(bId, registerCount(bTy)),
                   cElem = ops.shape.cElem](const agpu::PanelTile &t) {
    return panelInputsFor(t, plan, aId, aTy, bId, bTy, cId, cTy, deviceA, poolA,
                          cIn, stagedAElem, stagedBElem, cElem, aNames, bNames);
  };
}

agpu::Decision AgpuEmitter::setReadbackFor(const DotOperands &ops,
                                           const agpu::Plan &plan,
                                           agpu::DotInputs &in) {
  const agpu::ValueId cId = ops.cOut;
  const RankedTensorType cTy = ops.cOutTy;
  {
    am::SmallVec<agpu::StageAction, 8> probe;
    const am::SmallVec<am::Str, 8> bound =
        stagedNamesOf(cId, registerCount(cTy));
    if (const agpu::Decision d =
            planTileActions(cId, cTy, wholeWindowsOf(cTy), plan.cStagedView(),
                            (int)agpu::kAccBits, probe, bound, "tt.dot");
        !d.ok())
      return d;
  }
  in.readbackFor = [this, cId, cTy, cIn = ops.cIn, cView = plan.cStagedView(),
                    cNames = stagedNamesOf(cId, registerCount(cTy)),
                    rename = plan.readsBackByRename() ? plan.readback
                                                      : agpu::ReadbackPlan{},
                    // A fused dot's registers stay f32; other paths land in
                    // C's own element.
                    regElem = plan.accumulatorsOutlivePass()
                                  ? agpu::f32()
                                  : ops.shape.cElem](const agpu::Range &rows) {
    agpu::ReadbackInputs back;
    back.regElem = regElem;
    // A band clips the M axis and leaves the others whole.
    std::vector<agpu::CoordWindow> window = wholeWindowsOf(cTy);
    agpu::CoordWindow &mw = window[cTy.getRank() - 2];
    mw.lo = rows.lo;
    mw.hi = std::min(rows.hi, mw.hi);
    if (const agpu::Decision d =
            planTileActions(cId, cTy, window, cView, (int)agpu::kAccBits,
                            back.actions, cNames, "tt.dot");
        !d.ok()) {
      body_.notePending("a C band's layout stopped resolving mid-emission");
      return back;
    }
    // C is the dot's result and has no bound names yet; mint them by the
    // same `accName` convention the handler binds with.
    for (int64_t r = 0; r < registerCount(cTy); ++r)
      back.names.push_back(accName(cId, r));
    // Empty for a dot fed a zero (`readbackLoad` treats that as assign);
    // otherwise the incoming accumulator's names.
    back.bases = cIn;
    back.bases.resize(back.names.size());
    back.plan = rename;
    return back;
  };

  return agpu::Decision::emitted();
}

am::SmallVec<am::Str, 8> AgpuEmitter::stagedNamesOf(agpu::ValueId v,
                                                    int64_t regs) {
  am::SmallVec<am::Str, 8> names;
  for (int64_t r = 0; r < regs; ++r) {
    const am::Str *n = body_.sym.regAt(v, (std::size_t)r);
    names.push_back(n ? inIrType(v, r) : am::Str{});
  }
  return names;
}

agpu::Decision AgpuEmitter::planTileActions(
    agpu::ValueId v, RankedTensorType ty,
    const std::vector<agpu::CoordWindow> &windows, const agpu::TileView &dst,
    unsigned elemBits, am::SmallVec<agpu::StageAction, 8> &actions,
    const am::SmallVec<am::Str, 8> &names, std::string_view where) {
  const agpu::CoordSource cs = coordSourceOf(ty);
  // `rangeOf` indexes `cs.dims`; fewer output dims than rank reads past the
  // end.
  if ((int)cs.dims.size() != ty.getRank())
    return declined(where, "the layout has no coordinates");

  const int64_t regs = registerCount(ty);
  if ((int64_t)names.size() != regs)
    return declined(where, "a staged register has no name");

  for (int64_t r = 0; r < regs; ++r) {
    std::vector<agpu::CoordRange> ranges;
    for (int d = 0; d < ty.getRank(); ++d)
      ranges.push_back(cs.rangeOf((int)r, d, ty.getShape()[d]));

    // Use `registerCoordAt` here. `ranges[d].lo` is the reachable set's start
    // and is the same for every register along a lane-varying dim.
    const std::optional<agpu::TileView::Coord> at = registerCoordAt(ty, (int)r);
    if (!at)
      return declined(where, "the layout has no coordinates");

    agpu::TileView::Coord coord;
    for (std::size_t d = 0; d < windows.size() && d < at->size(); ++d)
      coord.push_back((*at)[d] - windows[d].lo);

    if (const std::optional<agpu::StageAction> a =
            agpu::planStage((int)r, ranges, windows, coord))
      actions.push_back(*a);
  }

  agpu::planStageRuns(actions, cs.dims, dst, elemBits);
  return agpu::Decision::emitted();
}

agpu::PanelInputs AgpuEmitter::panelInputsFor(
    const agpu::PanelTile &t, const agpu::Plan &plan, agpu::ValueId aId,
    RankedTensorType aTy, agpu::ValueId bId, RankedTensorType bTy,
    agpu::ValueId cId, RankedTensorType cTy, const agpu::OperandSource &deviceA,
    const am::Str &poolAName, const am::SmallVec<am::Str, 8> &cIn,
    const agpu::ElemType &aElem, const agpu::ElemType &bElem,
    const agpu::ElemType &cElem, const am::SmallVec<am::Str, 8> &aNames,
    const am::SmallVec<am::Str, 8> &bNames) {
  if (agpu_.gates.on(agpu::Gate::TraceOps)) {
    std::ostringstream os;
    os << "  panelInputsFor m=" << t.m.lo << ".." << t.m.hi << " n=" << t.n.lo
       << ".." << t.n.hi << " k=" << t.k.lo << ".." << t.k.hi << "\n";
    appendLog(agpu::Gate::TraceOps, os.str());
  }
  agpu::PanelInputs in;
  in.aElem = aElem;
  in.bElem = bElem;
  in.cRegElem = cElem;

  // A device-resident A plans no staging: the MMA reads it through `deviceA`
  // with this tile's corner as origin, since fragment indices are tile-local.
  bool ok = true;
  if (t.aDirect) {
    in.a = deviceA;
    in.a.rowOrigin = t.m.lo;
    in.a.colOrigin = t.k.lo;
  } else {
    in.a.buffer = poolAName;
    in.a.leadingDim = agpu::Stride(t.aView().strideAt(0));
    in.aNames = aNames;
    ok = planTileActions(aId, aTy, t.aWindows(), t.aStagedView(), aElem.bits,
                         in.aActions, aNames, "tt.dot")
             .ok();
  }
  in.bNames = bNames;
  ok = planTileActions(bId, bTy, t.bWindows(), t.bStagedView(), bElem.bits,
                       in.bActions, bNames, "tt.dot")
           .ok() &&
       ok;

  // C is the dot's result and has no bound names yet; staging would fill
  // `cNames` with empty strings, so they are minted below instead.
  ok = planTileActions(cId, cTy, t.cWindows(), t.cStagedView(),
                       (int)agpu::kAccBits, in.cActions,
                       stagedNamesOf(cId, registerCount(cTy)), "tt.dot")
           .ok() &&
       ok;
  in.cNames.clear();
  for (int64_t r = 0; r < registerCount(cTy); ++r)
    in.cNames.push_back(accName(cId, r));

  in.cBases = cIn;
  in.cBases.resize(in.cNames.size());
  if (plan.readsBackByRename())
    in.cRename = agpu::panelTileReadback(plan.facts, t);

  if (!ok) {
    body_.notePending(
        "an operand's layout does not resolve to tile coordinates");
  }
  return in;
}

agpu::Decision
AgpuEmitter::stageWholeTensor(agpu::ValueId v, RankedTensorType ty,
                              const am::Str &buffer, const agpu::TileView &dst,
                              const agpu::ElemType &elem,
                              std::string_view where, std::string_view what) {
  const int64_t regs = registerCount(ty);
  am::SmallVec<agpu::StageAction, 8> actions;
  const am::SmallVec<am::Str, 8> names = stagedNamesOf(v, regs);
  if (const agpu::Decision d = planTileActions(
          v, ty, wholeWindowsOf(ty), dst, elem.bits, actions, names, where);
      !d.ok())
    return d;
  int64_t covered = 0;
  for (const agpu::StageAction &a : actions)
    covered += a.width;
  if (covered < regs)
    return declined(where, std::string(what) + " register never lands");
  for (int64_t r = 0; r < regs; ++r)
    if (names[(std::size_t)r].empty())
      return declined(where, std::string(what) + " register has no name");

  agpu::emitStage(agpu_.context(), *cur_, dst, buffer, actions, names,
                  coordSourceOf(ty), elem);
  return agpu::Decision::emitted();
}

} // namespace mlir::triton::applegpu::bridge
