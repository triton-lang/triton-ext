// Shape op handlers: expand_dims, broadcast, convert_layout, trans, reshape,
// join/split/cat/unsplat.
#include "AgpuEmitter.h"
#include "AgpuOpTables.h"

#include "agpu/emit/EmitBand.h"
#include "agpu/emit/EmitElection.h"
#include "agpu/emit/EmitPoison.h"
#include "agpu/emit/EmitReshape.h"
#include "agpu/plan/RebindPlan.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

namespace {

std::optional<int64_t> elemThroughShapeOp(Value res, RankedTensorType srcTy,
                                          RankedTensorType resTy, int reg) {
  if (auto tr = res.getDefiningOp<triton::TransOp>())
    return elemThroughTranspose(srcTy, resTy, tr.getOrder(), reg);
  if (res.getDefiningOp<triton::ReshapeOp>())
    return elemThroughReshape(srcTy, resTy, reg);
  return elemThroughRebind(srcTy, resTy, reg);
}

} // namespace

// Whether a layout change can stay inside the warp (no pool, no barrier). The
// reservation pass and the emission pass must agree on this.
agpu::ShufflePlan AgpuEmitter::shuffleFor(RankedTensorType srcTy,
                                          RankedTensorType resTy,
                                          llvm::ArrayRef<int32_t> order) {
  if (tileElemCount(srcTy) != tileElemCount(resTy))
    return agpu::ShufflePlan();

  // A shuffle only moves values within a warp and the element maps below are
  // read at warp 0 only.
  if (!warpsAgree(srcTy, resTy, order))
    return agpu::ShufflePlan();

  std::vector<std::vector<int64_t>> src = elemsPerLaneOf(srcTy);
  const std::vector<std::vector<int64_t>> &dst = elemsPerLaneOf(resTy);
  if (src.empty() || dst.empty())
    return agpu::ShufflePlan();

  // Result element (i,j) is source element (j,i); renumber into the result's
  // order before comparing.
  if (!order.empty()) {
    const std::optional<std::vector<int64_t>> renumber =
        transposedElemMap(srcTy, resTy, order);
    if (!renumber)
      return agpu::ShufflePlan();
    for (std::vector<int64_t> &perReg : src)
      for (int64_t &e : perReg) {
        if (e < 0 || e >= (int64_t)renumber->size())
          return agpu::ShufflePlan();
        e = (*renumber)[(std::size_t)e];
      }
  }
  return agpu::planShuffleFromElems(src, dst);
}

agpu::Decision
AgpuEmitter::moveRegs(const agpu::OpView &o, RankedTensorType srcTy,
                      RankedTensorType resTy, llvm::ArrayRef<int32_t> order,
                      const agpu::ValueNames &srcNames, agpu::ElemType elem,
                      am::Expr *scatterGuard, agpu::ValueNames &moved) {
  if ((int64_t)srcNames.size() != registerCount(srcTy))
    return declined(o.name, "source register has no name");

  if (const agpu::ShufflePlan sp = shuffleFor(srcTy, resTy, order);
      sp.usable()) {
    const am::SmallVec<am::Str, 8> src(srcNames.begin(), srcNames.end());
    am::SmallVec<am::Str, 8> dstNames;
    for (int64_t r = 0, n = registerCount(resTy); r < n; ++r)
      dstNames.push_back(nameFor('w', o.results[0], r));

    // Suffixed per shuffle: both names are declared and a second shuffle in the
    // same kernel scope would redeclare them.
    agpu::ShuffleNames snm;
    const std::string sseq = std::to_string(body_.shuffleSeq++);
    snm.srcLane += sseq;
    snm.table += sseq;

    // Bind the returned names: an identity permutation emits nothing and hands
    // back the source names.
    const am::SmallVec<am::Str, 8> held =
        agpu::emitShuffle(agpu_.context(), *cur_, sp, src, dstNames, elem, snm);
    if (held.empty())
      return declined(o.name, "the shuffle produced no registers");
    moved.assign(held.begin(), held.end());
    return agpu::Decision::emitted();
  }

  // The slot index depends on the lane holding the element, so it must be a
  // runtime expression. The lane-0 answer is wrong for every other lane.
  const int64_t srcRegs = registerCount(srcTy);
  const int64_t resRegs = registerCount(resTy);

  // Equal element count: [N] -> [1,N] from a keep_dims reduction moves
  // no element.
  if (order.empty()) {
    if (tileElemCount(srcTy) != tileElemCount(resTy))
      return declined(o.name,
                      "a redistribution cannot change the element count");
  } else if (!isPermutationOf(srcTy.getShape(), resTy.getShape(), order)) {
    return declined(o.name, "the result is not that permutation of the source");
  }
  const agpu::TileView srcView = rowMajorViewOf(srcTy);
  agpu::TileView dstView = rowMajorViewOf(resTy);

  // Transpose: result coordinate (i,j) names source element (j,i).
  if (!order.empty())
    dstView = permutedView(srcView, order);
  const agpu::CoordSource srcCoords = coordSourceOf(srcTy);
  const agpu::CoordSource resCoords = coordSourceOf(resTy);
  if ((int)srcCoords.dims.size() != srcTy.getRank() ||
      (int)resCoords.dims.size() != resTy.getRank())
    return declined(o.name, "a layout has no per-thread coordinates");

  agpu::BandIO io;
  io.scatterGuard = scatterGuard;
  for (int64_t r = 0; r < srcRegs; ++r) {
    const int64_t elems = tileElemCount(srcTy);
    io.src.push_back(agpu::BandReg{
        (int)r, agpu::offsetExprOf(agpu_.context(), srcView, (int)r, srcCoords),
        agpu::CoordRange{0, 0, elems - 1}});
    io.srcValues.push_back(agpu_.context().var(srcNames[(std::size_t)r]));
  }

  agpu::ValueNames names;
  const int64_t elems = tileElemCount(resTy);
  for (int64_t r = 0; r < resRegs; ++r) {
    const am::Str n = nameFor('t', o.results[0], r);
    io.dst.push_back(agpu::BandReg{
        (int)r, agpu::offsetExprOf(agpu_.context(), dstView, (int)r, resCoords),
        agpu::CoordRange{0, 0, elems - 1}});
    io.dstNames.push_back(n);
    names.push_back(n);
  }

  const agpu::BandPlan plan = bandPlanFor(resTy, elem);

  // Built into its own block so an elided round trip does not declare the
  // scratch buffer.
  agpu::BandNames bnm;
  const am::Str scName = bnm.buffer;
  bnm.buffer = body_.pool.peek(scName);
  if (bnm.buffer.empty())
    return declined(o.name, "the scratch region was never carved");
  am::Block trip;
  const agpu::RoundTrip rt =
      agpu::emitBandRoundTrip(agpu_.context(), trip, plan, io, bnm);

  if (rt != agpu::RoundTrip::Elided) {
    body_.pool.use(scName);

    // Declared outside the gather's guard so a rejected lane still sees a
    // declared name.
    for (const am::Str &n : io.dstNames)
      cur_->push_back(agpu_.context().declStmt(
          agpu::mslTypeOf(elem), n, agpu::poisonValue(agpu_.context(), elem)));

    for (am::Stmt *s : trip)
      cur_->push_back(s);
  }

  if (rt == agpu::RoundTrip::Elided) {
    for (int64_t r = 0; r < resRegs && r < srcRegs; ++r)
      moved.push_back(srcNames[(std::size_t)r]);
    return agpu::Decision::emitted();
  }

  moved = std::move(names);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitRedistribute(const agpu::OpView &o,
                                             RankedTensorType srcTy,
                                             RankedTensorType resTy,
                                             llvm::ArrayRef<int32_t> order) {
  // A pointer tensor's per-element datum is its offset.
  if (isTensorOfPointers(srcTy))
    return emitRedistributeOffsets(o, srcTy, resTy, order);

  const agpu::ElemType *elemP = elemOf(o.results[0]);
  if (!elemP)
    return declined(o.name, "result type was never recorded");

  const Ready ready = readyForCounted(o, 0, 1, registerCount(srcTy),
                                      "source register has no name");
  if (!ready.ok())
    return ready.why;

  agpu::ValueNames srcNames;
  for (int64_t r = 0; r < ready.regs; ++r) {
    const am::Str narrowed = inIrType(o.operands[0], r);
    srcNames.push_back(narrowed.empty() ? ready.ops[0].at(r) : narrowed);
  }

  agpu::ValueNames moved;
  const agpu::Decision d = moveRegs(o, srcTy, resTy, order, srcNames, *elemP,
                                    agpu::noScatterGuard, moved);
  if (!d.ok())
    return d;
  body_.sym.bindRegs(o.results[0], std::move(moved));
  return d;
}

agpu::Decision AgpuEmitter::emitRedistributeOffsets(
    const agpu::OpView &o, RankedTensorType srcTy, RankedTensorType resTy,
    llvm::ArrayRef<int32_t> order) {
  // Sound only when every register names one base.
  const am::Str *baseP = body_.sym.uniformNameOf(o.operands[0]);
  if (!baseP)
    return declined(
        o.name,
        "a pointer tensor with more than one base cannot be redistributed");
  const am::Str base = *baseP;

  agpu::ValueNames offs;
  for (int64_t r = 0, n = registerCount(srcTy); r < n; ++r) {
    const auto off = body_.offsetOf.find({o.operands[0], r});
    if (off == body_.offsetOf.end())
      break;
    offs.push_back(off->second.name);
  }

  // No offsets means a complete address splatted across the tensor.
  if (offs.empty() && body_.basePtrs.count(o.operands[0])) {
    body_.sym.bindScalar(o.results[0], base);
    inheritBasePointer(o.operands[0], o.results[0]);
    return agpu::Decision::emitted();
  }
  if ((int64_t)offs.size() != registerCount(srcTy))
    return declined(o.name, "a pointer register has no recorded offset");

  const std::optional<agpu::ElemType> elem = movedTypeOf(srcTy);
  if (!elem)
    return declined(o.name, "result type was never recorded");

  // Every register must already be movedTypeOf's type.
  agpu::ValueNames wide;
  for (std::size_t r = 0; r < offs.size(); ++r) {
    const am::Str n = nameFor('o', o.results[0], (int64_t)r);
    cur_->push_back(agpu_.context().declStmt(agpu::mslTypeOf(*elem), n,
                                             agpu_.context().var(offs[r])));
    wide.push_back(n);
  }

  // Only the canonical holder scatters; duplicate lanes can hold stale copies.
  am::Expr *writers = agpu::electionExpr(
      agpu_.context(), agpu::electFor(spreadOf(srcTy)), agpu::BandNames{});

  agpu::ValueNames moved;
  const agpu::Decision d =
      moveRegs(o, srcTy, resTy, order, wide, *elem, writers, moved);
  if (!d.ok())
    return d;

  agpu::ValueNames bases;
  for (std::size_t r = 0; r < moved.size(); ++r) {
    bases.push_back(base);
    body_.offsetOf[{o.results[0], (int64_t)r}] =
        PtrOffset{moved[r], agpu::mslTypeOf(*elem), true};
  }
  body_.sym.bindRegs(o.results[0], std::move(bases));
  return d;
}

agpu::Decision AgpuEmitter::emitRebindOp(const agpu::OpView &o) {
  if (o.operands.size() != 1 || o.results.size() != 1)
    return declined(o.name, "unexpected operand or result count");

  // Convert/broadcast preserve the affine family; trans/reshape
  // permute the coordinate it is affine in, so they drop it.
  if (const auto it = body_.affine.find(o.operands[0]);
      it != body_.affine.end()) {
    agpu::AffineFamily fam = it->second;
    if (o.name == kExpandDims) {
      const Value r0 = mlirValueOf(o.results[0]);
      if (auto ed = r0 ? r0.getDefiningOp<ExpandDimsOp>() : ExpandDimsOp();
          ed && ed.getAxis() <= fam.scales.size()) {
        fam.scales.insert(fam.scales.begin() + (std::ptrdiff_t)ed.getAxis(), 0);
        body_.affine[o.results[0]] = fam;
      }
    } else if (o.name == kBroadcast || o.name == kConvertLayout) {
      body_.affine[o.results[0]] = fam;
    }
  }
  // Already bound by a dot's readback; no round trip is needed.
  if (body_.absorbedInto.count(o.results[0]))
    return agpu::Decision::emitted();
  const Value src = mlirValueOf(o.operands[0]);
  if (!src)
    return declined(o.name, "operand value was never recorded");
  const Value res = mlirValueOf(o.results[0]);
  if (!res)
    return declined(o.name, "result value was never recorded");
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  auto resTy = dyn_cast<RankedTensorType>(res.getType());
  if (!srcTy || !resTy)
    return declined(o.name, "an operand is not a ranked tensor");

  // The dot reads through a convert used only by it. Elided here,
  // since dead-code analysis assumes a threadgroup scatter is
  // observed.
  if (res.getDefiningOp<gpu::ConvertLayoutOp>() && usedOnlyByDot(res)) {
    agpu::ValueNames alias;
    for (int64_t r = 0, n = registerCount(srcTy); r < n; ++r) {
      const am::Str *s = body_.sym.regAt(o.operands[0], (std::size_t)r);
      if (!s)
        break;
      alias.push_back(*s);
    }
    if (!alias.empty()) {
      body_.sym.bindRegs(o.results[0], std::move(alias));
      return agpu::Decision::emitted();
    }
  }

  // A rename needs the layouts to agree for every thread.
  // Scoped to convert_layout: a square transpose is same-shape with
  // an interchangeable layout but still moves (i,j) to (j,i).
  if (res.getDefiningOp<gpu::ConvertLayoutOp>() &&
      srcTy.getShape() == resTy.getShape() &&
      !layoutsInterchangeable(srcTy, resTy))
    return emitRedistribute(o, srcTy, resTy);

  // A transpose whose two layouts are the same is still data
  // movement, which the comparison above can't see.
  if (auto tr = res.getDefiningOp<triton::TransOp>())
    return emitRedistribute(o, srcTy, resTy, tr.getOrder());

  // plan/RebindPlan.h decides which source register feeds each result
  // register; this layer supplies the coordinate sets.
  const int64_t srcRegs = registerCount(srcTy);
  std::vector<agpu::RegCoord> srcCoords;
  srcCoords.reserve((std::size_t)srcRegs);
  for (int64_t r = 0; r < srcRegs; ++r) {
    const std::optional<int64_t> e = flatElemAt(srcTy, (int)r);
    srcCoords.push_back(e ? agpu::RegCoord{(int32_t)*e}
                          : agpu::RegCoord{-1 - (int32_t)r});
  }

  const int64_t resRegs = registerCount(resTy);
  std::vector<agpu::RegCoord> resCoords;
  resCoords.reserve((std::size_t)resRegs);
  for (int64_t r = 0; r < resRegs; ++r) {
    const std::optional<int64_t> want =
        elemThroughShapeOp(res, srcTy, resTy, (int)r);
    if (!want)
      return declined(o.name, "cannot map a result register to a "
                              "source element");
    resCoords.push_back({(int32_t)*want});
  }

  const agpu::Rebind plan =
      agpu::rebind(resCoords, agpu::indexByCoord(srcCoords),
                   [](const agpu::RegCoord &rc, agpu::RegCoord &want) {
                     want = rc;
                     return true;
                   });

  // A result register no source feeds needs an element in another
  // thread.
  if (!plan.complete())
    return emitRedistribute(o, srcTy, resTy);

  const Ready ready =
      readyForCounted(o, 0, 1, srcRegs, "a source register was never bound");
  if (!ready.ok())
    return ready.why;

  agpu::ValueNames names;
  for (int64_t r = 0; r < resRegs; ++r) {
    const int from = plan.from[(std::size_t)r];
    names.push_back(ready.ops[0].at(from));

    // A pointer register is a name and an offset.
    inheritOffset(o.operands[0], (int64_t)from, o.results[0], r);
  }
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerRebindHandler() {
  table_.add("rebind", agpu::forOps({kExpandDims, kBroadcast, kConvertLayout,
                                     "tt.trans", "tt.reshape"},
                                    [this](const agpu::OpView &o) {
                                      return emitRebindOp(o);
                                    }));
}

bool AgpuEmitter::interleaveCoordsOf(Value v,
                                     std::vector<agpu::RegCoord> &out) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty)
    return false;
  for (int64_t r = 0, n = registerCount(ty); r < n; ++r) {
    const std::optional<std::vector<int64_t>> c = registerCoordAt(ty, (int)r);
    if (!c)
      return false;
    agpu::RegCoord rc;
    for (int64_t d : *c)
      rc.push_back((int32_t)d);
    out.push_back(std::move(rc));
  }
  return true;
}

agpu::Decision AgpuEmitter::emitUnsplatOp(const agpu::OpView &o) {
  // Bound as a scalar.
  if (o.operands.size() != 1 || o.results.size() != 1)
    return declined("tt.unsplat", "unexpected operand or result count");
  const Ready ready =
      readyForCounted(o, 0, 1, 1, "the operand was never bound");
  if (!ready.ok())
    return ready.why;
  body_.sym.bindScalar(o.results[0], ready.ops[0].at(0));
  return agpu::Decision::emitted();
}

// Two e2m1 values packed per i8, low nibble first along `axis`.
agpu::Decision AgpuEmitter::emitFp4ToFpOp(const agpu::OpView &o) {
  if (o.operands.size() != 1 || o.results.size() != 1)
    return declined("ttg.fp4_to_fp", "unexpected operand or result count");
  if (o.ints.empty())
    return declined("ttg.fp4_to_fp", "no axis attribute");

  const Value src = mlirValueOf(o.operands[0]);
  const Value res = mlirValueOf(o.results[0]);
  if (!src || !res)
    return declined("ttg.fp4_to_fp", "a value was never recorded");
  auto resTy = dyn_cast<RankedTensorType>(res.getType());
  if (!resTy)
    return declined("ttg.fp4_to_fp", "result is not a ranked tensor");
  const std::optional<agpu::ElemType> to = elemTypeOf(resTy.getElementType());
  if (!to)
    return declined("ttg.fp4_to_fp", "unsupported result element type");

  agpu::InterleaveFacts f;
  if (!interleaveCoordsOf(src, f.src) || !interleaveCoordsOf(res, f.dst))
    return declined("ttg.fp4_to_fp", "a register has no coordinate");

  const agpu::Fp4UnpackPlan plan = agpu::planFp4Unpack(f, (int)o.intAt(0));
  if (!plan.usable)
    return declined("ttg.fp4_to_fp",
                    "a result register has no source coordinate");

  agpu_.helpers.add(agpu::Helper::Fp4Unpack);

  auto &c = agpu_.context();
  return emitPerRegister(
      o, (int64_t)plan.from.size(), *to, 'q', [&](int64_t r) {
        RegValue v;
        const agpu::Fp4Pick &pick = plan.from[(std::size_t)r];
        const am::Str *n =
            body_.sym.regAt(o.operands[0], (std::size_t)pick.reg);
        if (!n)
          return v;
        // The helper masks to four bits, so the high nibble only
        // needs a shift.
        am::Expr *byte = c.var(*n);
        if (pick.high)
          byte = c.binary(am::BinOp::Shr, byte, c.lit(4));
        am::Expr *val =
            c.call(agpu::helperName(agpu::Helper::Fp4Unpack), {byte});
        // The helper answers in f32; narrow for f16/bf16 results.
        if (!(*to == agpu::f32()))
          val = c.cast(agpu::mslTypeOf(*to), val);
        v.value = val;
        return v;
      });
}

agpu::Decision AgpuEmitter::emitJoinSplitOp(const agpu::OpView &o) {
  const bool isJoin = o.name == kJoin;
  if (isJoin ? (o.operands.size() != 2 || o.results.size() != 1)
             : (o.operands.size() != 1 || o.results.size() != 2))
    return declined(o.name, "unexpected operand or result count");

  const Value src = mlirValueOf(o.operands[0]);
  if (!src)
    return declined(o.name, "operand value was never recorded");

  agpu::InterleaveFacts f;
  if (!interleaveCoordsOf(src, f.src))
    return declined(o.name, "an operand register has no coordinate");

  // One result for a join, two for a split, each planned against the
  // same source coordinates.
  for (std::size_t k = 0; k < o.results.size(); ++k) {
    const Value res = mlirValueOf(o.results[k]);
    if (!res)
      return declined(o.name, "result value was never recorded");
    f.dst.clear();
    if (!interleaveCoordsOf(res, f.dst))
      return declined(o.name, "a result register has no coordinate");

    const agpu::InterleavePlan plan =
        isJoin ? agpu::planJoinFrom(f) : agpu::planSplitFrom(f, (int)k);
    if (!plan.usable)
      return agpu::interleaveDecision(plan);

    const agpu::ValueNames *lhs = body_.sym.namesOf(o.operands[0]);
    const agpu::ValueNames *rhs =
        isJoin ? body_.sym.namesOf(o.operands[1]) : lhs;
    if (!lhs || !rhs)
      return declined(o.name, "an operand was never bound");

    agpu::ValueNames names = agpu::interleaveNames(plan, *lhs, *rhs);
    if (names.empty())
      return declined(o.name, "a source register was never bound");
    body_.sym.bindRegs(o.results[k], std::move(names));
  }
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerInterleaveHandler() {
  table_.add("unsplat", agpu::forOps({kUnsplat}, [this](const agpu::OpView &o) {
               return emitUnsplatOp(o);
             }));

  table_.add("fp4ToFp", agpu::forOps({kFp4ToFp}, [this](const agpu::OpView &o) {
               return emitFp4ToFpOp(o);
             }));

  table_.add("joinSplit",
             agpu::forOps({kJoin, "tt.split"}, [this](const agpu::OpView &o) {
               return emitJoinSplitOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
