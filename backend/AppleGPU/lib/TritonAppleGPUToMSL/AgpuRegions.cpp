// Ops carrying a user region the walk re-enters: tt.reduce, tt.scan,
// tt.map_elementwise.
#include "AgpuEmitter.h"

#include "agpu/emit/EmitMap.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

// The one MLIR-facing row of the combiner table; Combiner.h owns the rest.
// The structural match is Triton's: `getSingleCombiner` returns null unless
// the region is a single op over the block arguments, in order or
// commutatively reversed.
static agpu::Combiner combinerOf(Operation *op) {
  if (!op)
    return agpu::Combiner::Generic;
  return llvm::TypeSwitch<Operation *, agpu::Combiner>(op)
      .Case<arith::AddFOp>([](auto) { return agpu::Combiner::AddF; })
      .Case<arith::MulFOp>([](auto) { return agpu::Combiner::MulF; })
      .Case<arith::AddIOp>([](auto) { return agpu::Combiner::AddI; })
      .Case<arith::MulIOp>([](auto) { return agpu::Combiner::MulI; })
      .Case<arith::MinNumFOp>([](auto) { return agpu::Combiner::MinF; })
      .Case<arith::MaxNumFOp>([](auto) { return agpu::Combiner::MaxF; })
      .Case<arith::MinSIOp>([](auto) { return agpu::Combiner::MinS; })
      .Case<arith::MaxSIOp>([](auto) { return agpu::Combiner::MaxS; })
      .Case<arith::MinUIOp>([](auto) { return agpu::Combiner::MinU; })
      .Case<arith::MaxUIOp>([](auto) { return agpu::Combiner::MaxU; })
      .Case<arith::AndIOp>([](auto) { return agpu::Combiner::AndI; })
      .Case<arith::OrIOp>([](auto) { return agpu::Combiner::OrI; })
      .Case<arith::XOrIOp>([](auto) { return agpu::Combiner::XorI; })
      .Default([](auto) { return agpu::Combiner::Generic; });
}

// tt.scan has no `getSingleCombiner`, so this is that check for a scan region.
static Operation *singleScanCombiner(triton::ScanOp scan) {
  if (scan.getNumOperands() != 1 || scan.getNumResults() != 1)
    return nullptr;
  Block &blk = scan.getCombineOp().front();
  Operation *term = blk.getTerminator();
  if (term->getNumOperands() != 1)
    return nullptr;
  Operation *op = term->getOperand(0).getDefiningOp();
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1)
    return nullptr;
  const Value a = blk.getArgument(0), b = blk.getArgument(1);
  const Value lhs = op->getOperand(0), rhs = op->getOperand(1);
  const bool reversed =
      lhs == b && rhs == a && op->hasTrait<OpTrait::IsCommutative>();
  if (!(lhs == a && rhs == b) && !reversed)
    return nullptr;
  return op;
}

agpu::Decision AgpuEmitter::reductionPlanOf(triton::ReduceOp red,
                                            RankedTensorType srcTy,
                                            agpu::ReductionPlan &out) {
  const int axis = (int)red.getAxis();
  const LinearLayout ll = gpu::toLinearLayout(srcTy);
  MLIRContext *ctx = srcTy.getContext();

  const std::optional<StringAttr> dim = outDimAt(ll, axis);
  if (!dim)
    return declined("tt.reduce", "the layout has no such axis");

  // Registers that fold together: coordinates agree on every dimension but the
  // reduced one.
  const int64_t regs = registerCount(srcTy);
  std::vector<agpu::CoordKey> regCoords;
  for (int64_t r = 0; r < regs; ++r) {
    const std::optional<std::vector<int64_t>> c =
        registerCoordAt(srcTy, (int)r);
    if (!c)
      return declined("tt.reduce",
                      "a register has no coordinate under this layout");
    agpu::CoordKey::Storage k;
    for (int64_t v : *c)
      k.push_back((int32_t)v);
    regCoords.push_back(agpu::CoordKey(std::move(k)));
  }
  out.groups = agpu::groupSurvivors(regCoords, axis);
  if (out.groups.empty())
    return declined("tt.reduce", "no survivor group for this layout");

  // Lane/warp bits that move the reduced axis. A non-zero basis bit spreads it
  // across threads, needing a shuffle (lane) or threadgroup memory (warp).
  const agpu::BasisRow laneB = basisRow(ll, ctx, lldim::Lane, *dim);
  const agpu::BasisRow warpB = basisRow(ll, ctx, lldim::Warp, *dim);
  out.laneSteps = agpu::laneStepsFromMask(agpu::reduceMaskFromBases(
      std::vector<int32_t>(laneB.begin(), laneB.end())));
  out.warpMask = agpu::reduceMaskFromBases(
      std::vector<int32_t>(warpB.begin(), warpB.end()));

  const int64_t warps = numWarps();
  out.warpSubset = agpu::subsetsOf(out.warpMask, (int)warps);
  out.reducedAxis = axis;
  out.combiner = combinerOf(red.getSingleCombiner());

  // numWarps slots: every warp publishes, including ones this reduction does
  // not span.
  if (out.crossWarp())
    out.scratch = agpu::ScratchLayout{warps * agpu::kWarpSize, agpu::kWarpSize};

  // Accumulator element type is carried per operand: an integer reduction
  // through float is exact only to 2^24.
  for (Value s : red.getSrcs()) {
    const std::optional<agpu::ElemType> e = elemTypeOf(s.getType());
    if (!e)
      return declined("tt.reduce", "an operand has no element type");
    out.elems.push_back(*e);
    auto ty = dyn_cast<RankedTensorType>(s.getType());
    if (!ty)
      return declined("tt.reduce", "an operand is not a ranked tensor");
    out.regsPerOperand.push_back(registerCount(ty));
  }
  if (!out.operandsShareLayout())
    return declined("tt.reduce",
                    "the operands are not addressed by the same registers");

  return agpu::Decision::emitted();
}

am::SmallVec<am::Str, 4>
AgpuEmitter::lowerCombine(Region &region, am::Block &body,
                          const am::SmallVec<am::Str, 4> &lhs,
                          const am::SmallVec<am::Str, 4> &rhs) {
  am::SmallVec<am::Str, 4> out;
  Block &blk = region.front();

  // A combine region is walked once per fold step with the same value ids each
  // time; the per-walk suffix avoids Metal redefinition errors.
  const ScopeMark scope(*this, "c" + std::to_string(body_.combineSeq++));

  // Block arguments arrive source-major (a0,a1,...,b0,b1,...): that is the
  // order Triton's verifier requires.
  if (blk.getNumArguments() != lhs.size() + rhs.size()) {
    body_.notePending("the combine region's arity does not match the operands");
    return out;
  }
  const std::size_t nOp = lhs.size();
  for (std::size_t k = 0; k < nOp; ++k) {
    const BlockArgument a = blk.getArgument(k);
    const BlockArgument b = blk.getArgument(nOp + k);
    agpu::CarriedValue cv;
    if (const std::optional<agpu::ElemType> e = elemTypeOf(a.getType()))
      cv.elem = *e;
    cv.regs = {lhs[k]};
    bindCarried(a, cv);
    cv.regs = {rhs[k]};
    bindCarried(b, cv);
  }

  // cur_ is swapped for the walk and restored after: a handler is a
  // std::function registered once and cannot take the block as an argument.
  const bool ok = [&] {
    CurBlock in(*this, body);
    return walkBlock(blk, body).ok();
  }();
  if (!ok) {
    body_.pendingOk = false;
    if (body_.pendingWhy.empty())
      body_.pendingWhy = "an op in the combine region";
    return out;
  }

  // walkBlock skips terminators, so handle the combine region's here. Either
  // tt.reduce.return or tt.scan.return: one function serves both.
  Operation *term = blk.getTerminator();
  const bool isCombineReturn =
      isa<triton::ReduceReturnOp, triton::ScanReturnOp>(term);
  if (!isCombineReturn || term->getNumOperands() != lhs.size()) {
    body_.notePending(
        "the combine region does not return one value per operand");
    return out;
  }
  // A narrowing temporary reads a value declared in the body, so it has to be
  // emitted there rather than in the block this returns into.
  const CurBlock in(*this, body);
  for (Value v : term->getOperands()) {
    const am::Str *n = body_.sym.regAt(idOf(v), 0);
    if (!n) {
      body_.notePending("a combine result has no name");
      return out;
    }
    const am::Str narrowed = inIrType(idOf(v), 0);
    out.push_back(narrowed.empty() ? *n : narrowed);
  }
  return out;
}

RegionSources AgpuEmitter::regionSourcesOf(ValueRange srcs, ResultRange results,
                                           std::string_view where) {
  RegionSources rs;
  if (srcs.empty() || results.size() != srcs.size()) {
    rs.why = declined(where, "unexpected operand or result count");
    return rs;
  }
  rs.srcTy = dyn_cast<RankedTensorType>(srcs[0].getType());
  if (!rs.srcTy) {
    rs.why = declined(where, "the source is not a ranked tensor");
    return rs;
  }
  rs.sourceRegisterCount = registerCount(rs.srcTy);
  return rs;
}

agpu::Decision AgpuEmitter::gatherRegionNames(ValueRange srcs,
                                              llvm::ArrayRef<int> order,
                                              std::string_view where,
                                              RegionSources &into) {
  for (Value s : srcs) {
    const Operand op(body_.sym, idOf(s), into.sourceRegisterCount);
    if (!op.ok())
      return declined(where, "an operand register has no name");
    const auto nameAt = [&](int64_t r) {
      const am::Str n = inIrType(idOf(s), r);
      return n.empty() ? op.at(r) : n;
    };
    am::SmallVec<am::Str, 8> names;
    if (order.empty())
      for (int64_t r = 0; r < into.sourceRegisterCount; ++r)
        names.push_back(nameAt(r));
    else
      for (int r : order)
        names.push_back(nameAt((int64_t)r));
    into.names.push_back(std::move(names));
  }
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::scratchRegionsInto(int operands,
                                               am::Str (*keyFor)(int),
                                               std::string_view where,
                                               am::SmallVec<am::Str, 4> &into) {
  for (int k = 0; k < operands; ++k) {
    const am::Str b = body_.pool.use(keyFor(k));
    if (b.empty())
      return declined(where, "the scratch region was never carved");
    into.push_back(b);
  }
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitReduceOp(triton::ReduceOp red) {
  RegionSources rs =
      regionSourcesOf(red.getSrcs(), red.getResults(), "tt.reduce");
  if (!rs.ok())
    return rs.why;
  const RankedTensorType srcTy = rs.srcTy;

  agpu::ReductionPlan plan;
  const agpu::Decision d = reductionPlanOf(red, srcTy, plan);
  if (!d.ok())
    return d;

  if (const agpu::Decision g =
          gatherRegionNames(red.getSrcs(), {}, "tt.reduce", rs);
      !g.ok())
    return g;
  const auto &srcNames = rs.names;

  // Names prefixed per reduction to avoid redefinition. Scratch is not tagged:
  // it is a pool region shared by temporally disjoint uses.
  const std::string tag = std::to_string(body_.reduceSeq++);
  agpu::ReduceNames nm;
  nm.acc = "racc" + tag + "_";
  nm.peer = "rpeer" + tag + "_";
  if (plan.crossWarp())
    if (const agpu::Decision s =
            scratchRegionsInto((int)red.getSrcs().size(),
                               agpu::reduceScratchKey, "tt.reduce", nm.scratch);
        !s.ok())
      return s;

  body_.armPending();
  const std::vector<am::SmallVec<am::Str, 4>> accs =
      agpu::emitReduce(agpu_.context(), *cur_, plan, numWarps(), srcNames, nm,
                       [&](am::Block &body, const am::SmallVec<am::Str, 4> &a,
                           const am::SmallVec<am::Str, 4> &b) {
                         return lowerCombine(red.getCombineOp(), body, a, b);
                       });

  if (!body_.pendingOk)
    return declined("tt.reduce", body_.pendingWhy);
  if ((int)accs.size() != plan.groupCount())
    return declined("tt.reduce", "the emitter refused the plan it was given");

  // Bind each result register to the group holding its survivor by coordinate
  // key; register order is not group order.
  for (auto [k, res] : llvm::enumerate(red.getResults())) {
    auto resTy = dyn_cast<RankedTensorType>(res.getType());
    // A rank-1 reduction yields a scalar: no layout, one group.
    if (!resTy) {
      if (plan.groupCount() != 1)
        return declined("tt.reduce",
                        "a scalar result from more than one survivor group");
      body_.sym.bindScalar(idOf(res), accs[0][k]);
      continue;
    }

    agpu::ValueNames names;
    for (int64_t r = 0, n = registerCount(resTy); r < n; ++r) {
      const std::optional<std::vector<int64_t>> c =
          registerCoordAt(resTy, (int)r);
      if (!c)
        return declined("tt.reduce", "a result register has no coordinate");
      agpu::CoordKey::Storage key;
      for (int64_t v : *c)
        key.push_back((int32_t)v);
      const int gi = plan.groupFor(agpu::CoordKey(std::move(key)));
      if (gi < 0)
        return declined("tt.reduce",
                        "a result register belongs to no survivor group");
      names.push_back(accs[(std::size_t)gi][k]);
    }
    body_.sym.bindRegs(idOf(res), std::move(names));
  }

  return agpu::Decision::emitted();
}

// Bits of one input dimension that move the scanned axis, with the stride each
// moves it by. A scan needs the strides to connect adjacent elements.
static std::vector<agpu::AxisBit> axisBitsOf(const LinearLayout &ll,
                                             MLIRContext *ctx,
                                             llvm::StringRef inDim,
                                             StringAttr outDim) {
  std::vector<agpu::AxisBit> bits;
  const agpu::BasisRow row = basisRow(ll, ctx, inDim, outDim);
  for (std::size_t b = 0; b < row.size(); ++b)
    if (row[b] != 0)
      bits.push_back(agpu::AxisBit{(int)b, (int32_t)row[b]});
  return bits;
}

agpu::Decision AgpuEmitter::scanPlanOf(triton::ScanOp scan,
                                       RankedTensorType srcTy,
                                       agpu::ScanFacts &out) {
  const int axis = (int)scan.getAxis();
  const LinearLayout ll = gpu::toLinearLayout(srcTy);
  MLIRContext *ctx = srcTy.getContext();

  const std::optional<StringAttr> dim = outDimAt(ll, axis);
  if (!dim)
    return declined("tt.scan", "the layout has no such axis");

  out.laneBits = axisBitsOf(ll, ctx, lldim::Lane, *dim);
  out.warpBits = axisBitsOf(ll, ctx, lldim::Warp, *dim);
  // A thread's registers need not run along the scanned axis: a 16x16 column
  // scan can put two independent scans' elements in registers 0 and 1.
  out.regBits = axisBitsOf(ll, ctx, lldim::Register, *dim);
  out.numWarps = numWarps();
  out.regCount = registerCount(srcTy);
  out.reverse = scan.getReverse();
  out.combiner = combinerOf(singleScanCombiner(scan));

  // Carried per operand for the same reason as the reduction: an integer scan
  // through f32 is exact only to 2^24.
  for (Value s : scan.getSrcs()) {
    const std::optional<agpu::ElemType> e = elemTypeOf(s.getType());
    if (!e)
      return declined("tt.scan", "an operand has no element type");
    out.elems.push_back(*e);
    auto ty = dyn_cast<RankedTensorType>(s.getType());
    if (!ty)
      return declined("tt.scan", "an operand is not a ranked tensor");
    out.regsPerOperand.push_back(registerCount(ty));
  }

  return agpu::Decision::emitted();
}

// The order a scan consumes registers in: increasing along the axis, decreasing
// for reverse. Register index order is not axis order and emitScan folds r into
// r+1.
agpu::Decision AgpuEmitter::scanRegisterOrder(RankedTensorType srcTy, int axis,
                                              bool reverse,
                                              std::vector<int> &out) {
  const int64_t regs = registerCount(srcTy);

  // Sorted by the other axes first, then along the scanned one: registers
  // differing only off-axis belong to different scans and must group into
  // contiguous windows.
  std::vector<ScanRegisterKey> byCoord;
  for (int64_t r = 0; r < regs; ++r) {
    const std::optional<std::vector<int64_t>> c =
        registerCoordAt(srcTy, (int)r);
    if (!c || axis < 0 || (std::size_t)axis >= c->size())
      return declined("tt.scan",
                      "a register has no coordinate under this layout");
    ScanRegisterKey k;
    for (std::size_t d = 0; d < c->size(); ++d)
      if ((int)d != axis)
        k.group.push_back((*c)[d]);
    k.position = (*c)[(std::size_t)axis];
    k.reg = (int)r;
    byCoord.push_back(std::move(k));
  }

  // Stable sort; only the on-axis key reverses for a reverse scan.
  std::stable_sort(
      byCoord.begin(), byCoord.end(),
      [reverse](const ScanRegisterKey &a, const ScanRegisterKey &b) {
        if (a.group != b.group)
          return a.group < b.group;
        return reverse ? a.position > b.position : a.position < b.position;
      });
  for (const ScanRegisterKey &k : byCoord)
    out.push_back(k.reg);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitScanOp(triton::ScanOp scan) {
  RegionSources rs =
      regionSourcesOf(scan.getSrcs(), scan.getResults(), "tt.scan");
  if (!rs.ok())
    return rs.why;
  const RankedTensorType srcTy = rs.srcTy;

  agpu::ScanFacts facts;
  if (const agpu::Decision d = scanPlanOf(scan, srcTy, facts); !d.ok())
    return d;

  const agpu::ScanPlan plan = agpu::planScan(facts);
  if (!plan.usable)
    // Report the plan's own reason: every layout planScan refuses is an integer
    // condition over the axis bits.
    return agpu::scanDecline(facts);

  std::vector<int> order;
  if (const agpu::Decision d =
          scanRegisterOrder(srcTy, (int)scan.getAxis(), plan.reverse, order);
      !d.ok())
    return d;

  // The operands' registers in axis order, which is what emitScan consumes.
  if (const agpu::Decision g =
          gatherRegionNames(scan.getSrcs(), order, "tt.scan", rs);
      !g.ok())
    return g;
  const auto &srcNames = rs.names;
  const int64_t regs = rs.sourceRegisterCount;

  // Names prefixed per scan to avoid redefinition. Scratch is not tagged: it is
  // a pool region shared by temporally disjoint uses.
  const std::string tag = std::to_string(body_.scanSeq++);
  agpu::ScanNames nm;
  nm.acc = "sacc" + tag + "_";
  nm.peer = "speer" + tag + "_";
  nm.carry = "scarry" + tag + "_";
  if (plan.crossWarp)
    if (const agpu::Decision s =
            scratchRegionsInto((int)scan.getSrcs().size(), agpu::scanScratchKey,
                               "tt.scan", nm.scratch);
        !s.ok())
      return s;

  body_.armPending();
  const am::SmallVec<am::SmallVec<am::Str, 8>, 4> accs =
      agpu::emitScan(agpu_.context(), *cur_, plan, numWarps(), srcNames, nm,
                     [&](am::Block &body, const am::SmallVec<am::Str, 4> &a,
                         const am::SmallVec<am::Str, 4> &b) {
                       return lowerCombine(scan.getCombineOp(), body, a, b);
                     });

  if (!body_.pendingOk)
    return declined("tt.scan", body_.pendingWhy);
  if (accs.size() != scan.getResults().size())
    return declined("tt.scan", "the emitter refused the plan it was given");

  // Bound back through the same permutation the names were gathered by:
  // accs[k][i] belongs to register order[i].
  for (auto [k, res] : llvm::enumerate(scan.getResults())) {
    if (accs[k].size() != order.size())
      return declined("tt.scan", "a result has the wrong number of registers");
    agpu::ValueNames names((std::size_t)regs);
    for (std::size_t i = 0; i < order.size(); ++i)
      names[(std::size_t)order[i]] = accs[k][i];
    body_.sym.bindRegs(idOf(res), std::move(names));
  }

  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitMapOp(triton::MapElementwiseOp map) {
  am::Context &mc = agpu_.context();
  Region &region = map.getScalarOp();
  if (region.empty())
    return declined("tt.map_elementwise", "the scalar region is empty");

  Block &blk = region.front();
  agpu::MapPlan plan;
  plan.f.numSources = (int)map.getNumOperands();
  plan.f.numResults = (int)map.getNumResults();
  plan.f.pack = (int)map.getPack();
  // A multi-block region cannot yield through its terminator, since only one of
  // them runs; MapPlan declares a capture per result element instead.
  plan.f.multiBlock = !region.hasOneBlock();

  if (plan.f.numSources == 0)
    return declined("tt.map_elementwise", "no sources");
  auto srcTy = dyn_cast<RankedTensorType>(map.getOperand(0).getType());
  if (!srcTy)
    return declined("tt.map_elementwise", "a source is not a ranked tensor");
  plan.f.numRegisters = (int)registerCount(srcTy);

  if (const agpu::Decision d = agpu::mapDecision(plan); !d.ok())
    return d;
  if ((int)blk.getNumArguments() != plan.numBlockArguments())
    return declined("tt.map_elementwise",
                    "the region's arity does not match the sources and pack");

  std::vector<am::SmallVec<am::Str, 8>> srcNames;
  for (Value s : map.getSrcs()) {
    auto ty = dyn_cast<RankedTensorType>(s.getType());
    if (!ty || registerCount(ty) != plan.f.numRegisters)
      return declined("tt.map_elementwise",
                      "the sources disagree on register count");
    const Operand op(body_.sym, idOf(s), plan.f.numRegisters);
    if (!op.ok())
      return declined("tt.map_elementwise", "a source register has no name");
    am::SmallVec<am::Str, 8> names;
    for (int r = 0; r < plan.f.numRegisters; ++r)
      names.push_back(op.at(r));
    srcNames.push_back(std::move(names));
  }

  std::vector<agpu::ElemType> resultTypes;
  for (unsigned k = 0; k < map->getNumResults(); ++k) {
    const std::optional<agpu::ElemType> e =
        elemTypeOf(map->getResult(k).getType());
    if (!e)
      return declined("tt.map_elementwise", "a result type was never recorded");
    resultTypes.push_back(*e);
  }

  std::vector<agpu::RegisterNames> sources;
  for (const am::SmallVec<am::Str, 8> &s : srcNames)
    sources.push_back(agpu::RegisterNames(s.begin(), s.end()));

  std::vector<agpu::RegisterNames> results;
  bool bodyOk = true;

  // emitMap owns the group loop and index arithmetic; this binds block
  // arguments and walks the region.
  const agpu::Decision d = agpu::emitMap(
      mc, *cur_, plan, sources, resultTypes, results, agpu::MapNames{},
      [&](const agpu::MapBody &in, am::Block &body) {
        std::vector<am::Str> yielded;
        if (!bodyOk)
          return yielded;

        // Walked once per group with the same value ids each time, so each
        // inlining needs its own name scope.
        const ScopeMark scope(*this, "m" + std::to_string(body_.mapSeq++));

        // in.arguments is already in block-argument order.
        for (std::size_t i = 0; i < in.arguments.size(); ++i) {
          const BlockArgument a = blk.getArgument((unsigned)i);
          const agpu::ValueId id = idOf(a);
          body_.sym.bindScalar(id, in.arguments[i]);
          valueFor_[id] = a;
          if (const std::optional<agpu::ElemType> e = elemTypeOf(a.getType()))
            elemFor_[id] = *e;
        }

        // A multi-block region's map_return assigns captures.
        const auto atReturn = [&](Block &b, am::Block &into) -> agpu::Decision {
          auto ret =
              dyn_cast<triton::MapElementwiseReturnOp>(b.getTerminator());
          if (!ret)
            return agpu::Decision::emitted();
          if ((int)ret->getNumOperands() != plan.numResultOperands())
            return declined("tt.map_elementwise",
                            "the region's return does not match its results");
          for (int i = 0; i < plan.numCaptures(); ++i) {
            const am::Str *n =
                body_.sym.regAt(idOf(ret->getOperand((unsigned)i)), 0);
            if (!n)
              return declined("tt.map_elementwise",
                              "a captured value has no register name");
            into.push_back(
                mc.assign(mc.var(in.captures[(std::size_t)i]), mc.var(*n)));
          }
          return agpu::Decision::emitted();
        };

        const bool walked = walkWholeRegion(region, body, atReturn).ok();
        if (!walked) {
          bodyOk = false;
          return yielded;
        }

        if (plan.needsCaptures())
          return yielded;

        Operation *term = blk.getTerminator();
        if (!isa<triton::MapElementwiseReturnOp>(term) ||
            (int)term->getNumOperands() != plan.numResultOperands()) {
          bodyOk = false;
          return yielded;
        }
        for (int i = 0; i < plan.numResultOperands(); ++i) {
          const am::Str *n =
              body_.sym.regAt(idOf(term->getOperand((unsigned)i)), 0);
          if (!n) {
            bodyOk = false;
            return yielded;
          }
          yielded.push_back(*n);
        }
        return yielded;
      });

  if (!bodyOk)
    return declined("tt.map_elementwise", "the scalar region did not lower");
  if (!d.ok())
    return d;

  for (int k = 0; k < plan.f.numResults; ++k)
    body_.sym.bindRegs(idOf(map->getResult((unsigned)k)),
                       agpu::ValueNames(results[(std::size_t)k].begin(),
                                        results[(std::size_t)k].end()));
  return agpu::Decision::emitted();
}

} // namespace mlir::triton::applegpu::bridge
