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
                                            Operation *dot,
                                            agpu::DotInputs &in) {
  // B and C are regions of the threadgroup buffer carved by `walkOp`; this
  // handler only names them.
  const agpu::MmaNames mnm;
  const bool bStaged =
      !plan.facts.bInPlace || plan.kind == agpu::Plan::Kind::Panel;
  const auto resident = body_.residentBuf.find({dot, 1});
  const am::Str bBuf = resident != body_.residentBuf.end() ? resident->second
                       : bStaged ? body_.pool.use(mnm.poolB)
                                 : am::Str();
  const bool cThroughPool = plan.cThroughPool();
  const am::Str cBuf = cThroughPool ? body_.pool.use(mnm.poolC) : am::Str();
  if ((bStaged && bBuf.empty()) || (cThroughPool && cBuf.empty()))
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
  const bool stagePad = plan.padStagedOperands();
  const agpu::DotFacts &pf = plan.facts;
  const agpu::TileView aStaged = agpu::stagedAView(pf, stagePad);
  const agpu::TileView bStaged = agpu::stagedBView(pf, stagePad);

  // Every pool user opens its epoch with a barrier: the previous one (a fused
  // pass's MMAs, a drained dot's readback) leaves its reads unfenced.
  if (!stagesPerTile)
    cur_->push_back(agpu_.context().barrier());

  // An operand already in a shared buffer is read where it lies. The origin
  // splits at the pitch into the window's corner.
  const auto inPlace = [&](Value md,
                           agpu::OperandSource &src) -> agpu::Decision {
    const auto it = body_.memDescOf.find(idOf(md));
    if (it == body_.memDescOf.end())
      return declined("tt.dot", "an in-place operand's buffer was never bound");
    const agpu::TileView &v = it->second.view;
    const int64_t pitch = v.rank() == 2 ? v.strideAt(0) : 0;
    if (pitch <= 0 || pitch % 2 != 0 || v.origin() % 2 != 0 ||
        v.swizzle().permutes())
      return declined("tt.dot", "an in-place operand's buffer changed shape");
    src.buffer = it->second.buffer;
    src.leadingDim = agpu::Stride(pitch);
    src.rowOrigin = v.origin() / pitch;
    src.colOrigin = v.origin() % pitch;
    return agpu::Decision::emitted();
  };
  const bool aInPlace = pf.aInPlace && !stagesPerTile;
  const bool bInPlace = pf.bInPlace && !stagesPerTile;

  if (pf.aDirect) {
    const agpu::Decision d = readADirect(ops, in);
    if (!d.ok())
      return d;
  } else if (pf.aFromRegs) {
    if (ops.aStageTy != ops.shape.aRegsTy)
      return agpu::Decision::failed("tt.dot", "A's registers were never bound");
    for (int64_t r = 0, n = registerCount(ops.aStageTy); r < n; ++r) {
      const am::Str *name = body_.sym.regAt(ops.aStage, (std::size_t)r);
      if (!name)
        return agpu::Decision::failed("tt.dot",
                                      "A's registers were never bound");
      in.aRegs.push_back(*name);
    }
  } else if (aInPlace) {
    if (const agpu::Decision d = inPlace(ops.shape.aShared, in.a); !d.ok())
      return d;
    in.direct.poolA = in.panel.poolA = in.a.buffer;
  } else {
    const agpu::MmaNames mnm;
    const auto resident = body_.residentBuf.find({ops.op, 0});
    const bool aResident = resident != body_.residentBuf.end();
    const am::Str aBuf =
        aResident ? resident->second : body_.pool.use(mnm.poolA);
    if (aBuf.empty())
      return declined("tt.dot",
                      "a pool region this dot stages through was never carved");
    in.direct.poolA = in.panel.poolA = aBuf;
    if (!stagesPerTile && !aResident) {
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
    if (!bInPlace && !body_.residentBuf.count({ops.op, 1}))
      if (const agpu::Decision d =
              stageWholeTensor(ops.bStage, ops.bStageTy, in.panel.poolB,
                               bStaged, stagedBElem, "tt.dot", "a B");
          !d.ok())
        return d;
    cur_->push_back(agpu_.context().barrier());
  }

  if (bInPlace) {
    if (const agpu::Decision d = inPlace(ops.shape.bShared, in.b); !d.ok())
      return d;
  } else {
    in.b.buffer = in.panel.poolB;
    in.b.leadingDim = agpu::Stride(bStaged.strideAt(bStaged.rank() - 2));
    if (pf.batched())
      in.b.sliceStride = bStaged.strideAt(0);
  }
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

  if (const agpu::Decision d = namePoolRegions(plan, ops.op, in); !d.ok())
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

  setTileInputs(ops, plan, in);

  // A drain that stores straight to device has no readback, so C's register
  // layout is never asked for.
  if (plan.storesCDirect())
    return resolveDirectCStore(ops, plan, in);
  return setReadbackFor(ops, plan, in);
}

void AgpuEmitter::setTileInputs(const DotOperands &ops, const agpu::Plan &plan,
                                agpu::DotInputs &in) {
  // Names are resolved here: narrowing a widened register mints a temporary,
  // which belongs in this block.
  const PanelStaging staged{
      in.a, in.panel.poolA,
      stagedNamesOf(ops.aStage, registerCount(ops.aStageTy)),
      stagedNamesOf(ops.bStage, registerCount(ops.bStageTy))};
  in.tileInputs = [this, ops, plan, staged](const agpu::PanelTile &t) {
    return panelInputsFor(t, ops, plan, staged);
  };
}

agpu::Decision AgpuEmitter::setReadbackFor(const DotOperands &ops,
                                           const agpu::Plan &plan,
                                           agpu::DotInputs &in) {
  const agpu::ValueId cId = ops.cOut;
  const RankedTensorType cTy = ops.cOutTy;
  // Asked here so the band planning inside `readbackFor`, which cannot return
  // a Decision, has a clean decline for the part that does not depend on the
  // band's window.
  if (const agpu::Decision d = tileCoordsResolvable(cTy, "tt.dot"); !d.ok())
    return d;
  in.readbackFor = [this, cId, cTy, cIn = ops.cIn, cView = plan.cStagedView(),
                    rename = plan.readsBackByRename() ? plan.readback
                                                      : agpu::ReadbackPlan{},
                    // A fused dot's registers stay f32; other paths land in
                    // C's own element.
                    regElem = plan.accumulatorsOutlivePass()
                                  ? agpu::f32()
                                  : ops.shape.cElem](const agpu::Range &rows) {
    using R = agpu::Result<agpu::ReadbackInputs>;
    agpu::ReadbackInputs back;
    back.regElem = regElem;
    // A band clips the M axis and leaves the others whole.
    std::vector<agpu::CoordWindow> window = wholeWindowsOf(cTy);
    agpu::CoordWindow &mw = window[cTy.getRank() - 2];
    mw.lo = rows.lo;
    mw.hi = std::min(rows.hi, mw.hi);
    if (const agpu::Decision d = planTileActions(
            cTy, window, cView, (int)agpu::kAccBits, back.actions, "tt.dot");
        !d.ok())
      return R::no(declined(
          "tt.dot", "a C band's layout stopped resolving mid-emission"));
    // C is the dot's result and has no bound names yet; mint them by the
    // same `accName` convention the handler binds with.
    for (int64_t r = 0; r < registerCount(cTy); ++r)
      back.names.push_back(accName(cId, r));
    // Empty for a dot fed a zero (`readbackLoad` treats that as assign);
    // otherwise the incoming accumulator's names.
    back.bases = cIn;
    back.bases.resize(back.names.size());
    back.plan = rename;
    return R::of(std::move(back));
  };

  return agpu::Decision::emitted();
}

am::SmallVec<am::Str, 8> AgpuEmitter::stagedNamesOf(agpu::ValueId v,
                                                    int64_t regs) {
  am::SmallVec<am::Str, 8> names;
  for (int64_t r = 0; r < regs; ++r) {
    const am::Str *n = body_.sym.regAt(v, (std::size_t)r);
    names.push_back(n ? inIrType(v, *n) : am::Str{});
  }
  return names;
}

agpu::Decision AgpuEmitter::tileCoordsResolvable(RankedTensorType ty,
                                                 std::string_view where) {
  // `rangeOf` indexes `cs.dims`; fewer output dims than rank reads past the
  // end.
  if ((int)coordSourceOf(ty).dims.size() != ty.getRank())
    return declined(where, "the layout has no coordinates");
  for (int64_t r = 0; r < registerCount(ty); ++r)
    if (!registerCoordAt(ty, (int)r))
      return declined(where, "the layout has no coordinates");
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::planTileActions(
    RankedTensorType ty, const std::vector<agpu::CoordWindow> &windows,
    const agpu::TileView &dst, unsigned elemBits,
    am::SmallVec<agpu::StageAction, 8> &actions, std::string_view where) {
  if (const agpu::Decision d = tileCoordsResolvable(ty, where); !d.ok())
    return d;

  const agpu::CoordSource cs = coordSourceOf(ty);
  const int64_t regs = registerCount(ty);

  for (int64_t r = 0; r < regs; ++r) {
    std::vector<agpu::CoordRange> ranges;
    for (int d = 0; d < ty.getRank(); ++d)
      ranges.push_back(cs.rangeOf((int)r, d, ty.getShape()[d]));

    // Use `registerCoordAt` here. `ranges[d].lo` is the reachable set's start
    // and is the same for every register along a lane-varying dim. Present for
    // every register, or `tileCoordsResolvable` would have declined above.
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

agpu::Result<agpu::PanelInputs>
AgpuEmitter::panelInputsFor(const agpu::PanelTile &t, const DotOperands &ops,
                            const agpu::Plan &plan,
                            const PanelStaging &staged) {
  using R = agpu::Result<agpu::PanelInputs>;
  if (agpu_.gates.on(agpu::Gate::TraceOps)) {
    std::ostringstream os;
    os << "  panelInputsFor m=" << t.m.lo << ".." << t.m.hi << " n=" << t.n.lo
       << ".." << t.n.hi << " k=" << t.k.lo << ".." << t.k.hi << "\n";
    appendLog(agpu::Gate::TraceOps, os.str());
  }
  agpu::PanelInputs in;
  in.aElem = stagedElemOf(plan, ops.shape.aElem);
  in.bElem = stagedElemOf(plan, ops.shape.bElem);
  in.cRegElem = ops.shape.cElem;

  // Every operand plans even after one declines, so each `Actions` vector is
  // filled; the first decline is kept for its reason.
  agpu::Decision bad = agpu::Decision::emitted();
  const auto plan1 = [&](const agpu::Decision &d, const char *operand) {
    if (!d.ok() && bad.ok())
      bad = agpu::Decision::declined("tt.dot",
                                     std::string(operand) + ": " + d.why());
  };

  // A device-resident A plans no staging: the MMA reads it through `deviceA`
  // with this tile's corner as origin, since fragment indices are tile-local.
  if (t.aDirect) {
    in.a = staged.deviceA;
    in.a.rowOrigin = t.m.lo;
    in.a.colOrigin = t.k.lo;
  } else {
    in.a.buffer = staged.poolA;
    in.a.leadingDim = agpu::Stride(t.aView().strideAt(0));
    in.aNames = staged.aNames;
    plan1(planTileActions(ops.aStageTy, t.aWindows(), t.aStagedView(),
                          in.aElem.bits, in.aActions, "tt.dot"),
          "A");
  }
  in.bNames = staged.bNames;
  plan1(planTileActions(ops.bStageTy, t.bWindows(), t.bStagedView(),
                        in.bElem.bits, in.bActions, "tt.dot"),
        "B");

  // C is the dot's result and has no bound names yet, so they are minted below.
  plan1(planTileActions(ops.cOutTy, t.cWindows(), t.cStagedView(),
                        (int)agpu::kAccBits, in.cActions, "tt.dot"),
        "C");
  for (int64_t r = 0; r < registerCount(ops.cOutTy); ++r)
    in.cNames.push_back(accName(ops.cOut, r));

  in.cBases = ops.cIn;
  in.cBases.resize(in.cNames.size());
  if (plan.readsBackByRename())
    in.cRename = agpu::panelTileReadback(plan.facts, t);

  if (!bad.ok())
    return R::no(declined("tt.dot", bad.why()));
  return R::of(std::move(in));
}

agpu::Decision
AgpuEmitter::stageWholeTensor(agpu::ValueId v, RankedTensorType ty,
                              const am::Str &buffer, const agpu::TileView &dst,
                              const agpu::ElemType &elem,
                              std::string_view where, std::string_view what) {
  const int64_t regs = registerCount(ty);
  am::SmallVec<agpu::StageAction, 8> actions;
  const am::SmallVec<am::Str, 8> names = stagedNamesOf(v, regs);
  if (const agpu::Decision d = planTileActions(ty, wholeWindowsOf(ty), dst,
                                               elem.bits, actions, where);
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

  // Metal sinks a fused dot's MMAs to the loop latch while this staging pins
  // their fragment loads above it. A never-taken select reading every
  // accumulator keeps the MMAs before this store.
  const std::optional<agpu::ElemType> regElem = elemTypeOf(ty.getElementType());
  if (!body_.anchorAccs.empty() && regElem &&
      regElem->kind == agpu::ElemType::Kind::Float && !actions.empty()) {
    am::Context &mc = agpu_.context();
    am::Expr *sum = nullptr;
    for (const am::Str &a : body_.anchorAccs) {
      am::Expr *e = agpu::fragElemExpr(mc, a, 0);
      sum = sum ? mc.binary(am::BinOp::Add, sum, e) : e;
    }
    const am::Str &x = names[(std::size_t)actions.front().reg];
    am::Expr *never = mc.binary(
        am::BinOp::Eq,
        mc.member(mc.var(agpu::KernelNames{}.gridSize), am::builtin::comp::X),
        mc.lit(0));
    cur_->push_back(mc.assign(
        mc.var(x),
        mc.ternary(never, mc.cast(agpu::mslTypeOf(*regElem), sum), mc.var(x))));
    body_.anchorAccs.clear();
  }

  agpu::emitStage(agpu_.context(), *cur_, dst, buffer, actions, names,
                  coordSourceOf(ty), elem);
  return agpu::Decision::emitted();
}

} // namespace mlir::triton::applegpu::bridge
