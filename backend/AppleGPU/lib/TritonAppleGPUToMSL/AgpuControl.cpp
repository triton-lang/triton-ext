// scf.for, scf.if and scf.while lowered to MSL loops and variables.
#include "AgpuEmitter.h"

#include "agpu/cost/Occupancy.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

static bool isLoop(Operation *op) { return isa<scf::ForOp, scf::WhileOp>(op); }

static bool nestedInLoop(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (isLoop(p))
      return true;
  return false;
}

static bool laterLoopInBlock(Operation *op) {
  for (Operation *n = op->getNextNode(); n; n = n->getNextNode()) {
    bool found = false;
    n->walk([&](Operation *inner) { found = found || isLoop(inner); });
    if (found)
      return true;
  }
  return false;
}

agpu::Decision AgpuEmitter::emitForOp(scf::ForOp forOp) {
  // Loop-carried names are minted from the result id and bound to init, body
  // arg and yield, so all three use the same MSL variable.
  std::vector<agpu::ValueNames> carriedNames, initNamesOf;
  agpu::Carried carriedShape;

  const std::size_t n = forOp.getNumRegionIterArgs();
  if (forOp.getInitArgs().size() != n || forOp.getResults().size() != n)
    return agpu::Decision::failed();

  for (std::size_t i = 0; i < n; ++i) {
    const Value res = forOp.getResult(i);
    const Value init = forOp.getInitArgs()[i];
    const Value arg = forOp.getRegionIterArg(i);

    const agpu::CarriedValue cv = carriedFresh(res);
    bindCarried(arg, cv);
    if (cv.elem.isPointer())
      markBasePointer(idOf(arg));

    const agpu::Result<agpu::CarriedValue> iv = carriedFrom(
        init, cv, "scf.for", "a loop's initial value has no register names");
    if (!iv.ok())
      return iv.why;

    carriedNames.push_back(cv.regs);
    initNamesOf.push_back(iv.value.regs);
    carriedShape.push_back(cv);
  }

  agpu::LoopBounds b;
  b.iv = "iv" + std::to_string(idOf(forOp.getInductionVar()));
  body_.sym.bindScalar(idOf(forOp.getInductionVar()), b.iv);
  valueFor_[idOf(forOp.getInductionVar())] = forOp.getInductionVar();
  if (const std::optional<agpu::ElemType> e =
          elemTypeOf(forOp.getInductionVar().getType())) {
    elemFor_[idOf(forOp.getInductionVar())] = *e;
    // Must match the IR's width: declaring `int` for an i64 IV truncates `lo`
    // while `hi` still compares at 64 bits.
    b.wideIv = e->bits > 32;
  }

  const am::Str *lo = body_.sym.scalarName(idOf(forOp.getLowerBound()));
  const am::Str *hi = body_.sym.scalarName(idOf(forOp.getUpperBound()));
  const am::Str *st = body_.sym.scalarName(idOf(forOp.getStep()));
  if (!lo || !hi || !st)
    return declined("scf.for", "a loop bound has no emitted name");
  b.lo = agpu_.context().var(*lo);
  b.hi = agpu_.context().var(*hi);
  b.step = agpu_.context().var(*st);

  stageResidentOperands(forOp);
  body_.anchorAccs.clear();
  const int64_t poolFloor = body_.poolFloor;

  // Saved/restored around the body walk so a nested loop's fused dot is hosted
  // by the nested loop.
  agpu::Carried yieldValues;
  am::Block body;
  std::vector<FusedDot> enclosing;
  enclosing.swap(body_.fusedDots);
  const agpu::Decision bodyDone = walkRegion(forOp.getRegion(), body, [&] {
    auto y = dyn_cast<scf::YieldOp>(forOp.getBody(0)->getTerminator());
    if (!y || y.getNumOperands() != n)
      return declined("scf.for",
                      "the loop's yield does not match its carried values");
    for (std::size_t i = 0; i < n; ++i) {
      // Fragment-carried values have no yield name; the placeholder keeps this
      // list indexed by iter-arg.
      if (inFragments(body_.fusedDots, idOf(forOp.getResult(i)))) {
        yieldValues.push_back({});
        continue;
      }

      const agpu::Result<agpu::CarriedValue> got =
          carriedFrom(y.getOperand(i), carriedShape[i], "scf.for",
                      "a yielded value has no register names");
      if (!got.ok())
        return got.why;
      yieldValues.push_back(got.value);
    }
    return agpu::Decision::emitted();
  });
  std::vector<FusedDot> fused;
  fused.swap(body_.fusedDots);
  body_.fusedDots.swap(enclosing);
  body_.anchorAccs.clear();
  body_.poolFloor = poolFloor;
  if (!bodyDone.ok())
    return bodyDone;

  // A fused dot's result lives in accumulator fragments.
  agpu::Carried carried, inits, yielded;
  for (std::size_t i = 0; i < n; ++i) {
    const Value res = forOp.getResult(i);
    if (inFragments(fused, idOf(res)))
      continue;

    const agpu::Result<agpu::CarriedValue> c = carriedFor(res, carriedNames[i]);
    if (!c.ok())
      return c.why;
    const agpu::Result<agpu::CarriedValue> in = carriedFor(res, initNamesOf[i]);
    if (!in.ok())
      return in.why;
    carried.push_back(c.value);
    inits.push_back(in.value);
    yielded.push_back(yieldValues[i]);
  }

  const auto loop = [&]() -> agpu::Decision {
    const agpu::Decision d = agpu::emitFor(agpu_.context(), *cur_, b, carried,
                                           inits, std::move(body), yielded);
    // Only after a block's last loop: a respelling between two loops was
    // measured to cost more than it saved.
    if (d.ok() && body_.declaresThreadgroup && !nestedInLoop(forOp) &&
        !laterLoopInBlock(forOp)) {
      am::Context &mc = agpu_.context();
      const agpu::KernelNames nm;
      body_.hoist.rebase(mc, *cur_, mc.var(nm.simdLaneId),
                         mc.var(nm.simdGroupId));
    }
    return d;
  };

  for (FusedDot &fd : fused) {
    if (const auto it = body_.continuedFrom.find(fd.result);
        it != body_.continuedFrom.end())
      fd.initFrom = it->second;
    fd.continued = continuesInto(forOp, fd);
  }

  // Each bracket wraps the loop the previous ones built.
  std::function<agpu::Decision()> bracketed = loop;
  for (const FusedDot &fd : fused)
    bracketed = [&fd, inner = bracketed, this]() {
      agpu::ReadbackFn readback = fd.readbackFor;
      // The handed-over fragments already hold the earlier result, so the
      // readback assigns rather than adds it.
      if (!fd.initFrom.empty() && readback)
        readback = [rb = fd.readbackFor](const agpu::Range &rows) {
          agpu::Result<agpu::ReadbackInputs> got = rb(rows);
          if (got.ok())
            for (am::Str &base : got.value.bases)
              base = {};
          return got;
        };
      return agpu::emitFusedLoop(agpu_.context(), *cur_, fd.plan, fd.names,
                                 readback, fd.cCoords, fd.cStore, fd.cSteps,
                                 inner, fd.initFrom, fd.continued);
    };
  return bracketed();
}

// Metal sinks a fused dot's MMAs to the loop latch; a later staging written
// over its operands would pin their fragment loads above it. Carving the rest
// of the body above them avoids that when it fits without costing a resident
// threadgroup.
bool AgpuEmitter::keepsOperandsApart(Operation *dot) {
  int64_t end = body_.poolFloor;
  for (const PoolNeed::Region &r : poolNeedOf(dot).regions)
    if (!r.atBase)
      end += r.alignedBytes();
  int64_t after = 0;
  for (Operation *n = dot->getNextNode(); n; n = n->getNextNode())
    n->walk(
        [&](Operation *o) { after = std::max(after, poolNeedOf(o).bytes()); });
  if (after == 0)
    return true;
  auto func = dot->getParentOfType<triton::FuncOp>();
  int64_t planned = 0;
  func.walk([&](Operation *o) {
    if (o != func.getOperation())
      planned = std::max(planned, poolNeedOf(o).bytes());
  });
  const int64_t live = agpu_.pool.plan().live.count();
  const int64_t apart = end + after + live;
  if (apart > agpu::kTGResidentBudgetBytes ||
      agpu::cost::losesResidency(planned + live, apart,
                                 agpu::threadsFor(numWarps())))
    return false;
  body_.poolFloor = end;
  return true;
}

// When this loop's result is only the next loop's matching accumulator input,
// the next loop continues the fragments and nothing drains in between.
bool AgpuEmitter::continuesInto(scf::ForOp forOp, const FusedDot &fd) {
  Value res;
  for (Value r : forOp.getResults())
    if (idOf(r) == fd.result)
      res = r;
  if (!res || !res.hasOneUse())
    return false;
  OpOperand &use = *res.getUses().begin();
  auto next = dyn_cast<scf::ForOp>(use.getOwner());
  if (!next || use.getOperandNumber() < next.getNumControlOperands())
    return false;
  const unsigned i = use.getOperandNumber() - next.getNumControlOperands();
  for (triton::DotOp dot : next.getBody()->getOps<triton::DotOp>()) {
    if (dot.getC() != next.getRegionIterArg(i))
      continue;
    const DotShape shape = dotShapeOf(dot);
    if (!shape.aTy)
      return false;
    if (!fragmentsCarryOver(fd.plan, agpu_.planFor(dotFactsOf(shape))))
      return false;
    body_.continuedFrom[idOf(next.getResult(i))] =
        agpu::fusedAccNames(fd.plan, fd.names);
    return true;
  }
  return false;
}

bool AgpuEmitter::fragmentsCarryOver(const agpu::Plan &from,
                                     const agpu::Plan &to) {
  // A nonzero init the fragments did not start from is only added at the
  // drain, which a handover skips.
  const bool holdsInit = !from.facts.cInitNonzero || from.facts.cInitContinues;
  const agpu::WarpGrid a = agpu::gridOf(from), b = agpu::gridOf(to);
  return holdsInit && from.accumulatorsOutlivePass() &&
         to.accumulatorsOutlivePass() && a.mT == b.mT && a.nT == b.nT &&
         a.numWarps == b.numWarps &&
         agpu::planWarpProgram(a).sameCover(agpu::planWarpProgram(b)) &&
         agpu::fusedAccNames(to, agpu::DirectNames{}).size() ==
             agpu::fusedAccNames(from, agpu::DirectNames{}).size();
}

agpu::Decision AgpuEmitter::emitIfOp(scf::IfOp ifOp) {
  const am::Str *cond = body_.sym.scalarName(idOf(ifOp.getCondition()));
  if (!cond)
    return declined("scf.if", "the condition has no emitted name");

  // Declared outside both arms: an arm that does not run must still leave the
  // result readable.
  agpu::Carried results;
  for (Value res : ifOp.getResults())
    results.push_back(carriedFresh(res));

  const auto walkArm = [&](Region &region, am::Block &into,
                           agpu::Carried &yielded) {
    return walkRegion(region, into, [&] {
      const agpu::Result<agpu::Carried> y =
          carriedOperands(region.front().getTerminator(), results, "scf.if");
      yielded = y.value;
      return y.why;
    });
  };

  am::Block thenArm, elseArm;
  agpu::Carried thenYield, elseYield;
  if (const agpu::Decision d =
          walkArm(ifOp.getThenRegion(), thenArm, thenYield);
      !d.ok())
    return d;

  const bool hasElse = !ifOp.getElseRegion().empty();
  if (hasElse)
    if (const agpu::Decision d =
            walkArm(ifOp.getElseRegion(), elseArm, elseYield);
        !d.ok())
      return d;

  // `emitIf` declares the result and leaves it alone on the missing path, so
  // it would be read uninitialised.
  if (!results.empty() && !hasElse)
    return declined("scf.if",
                    "a result with no else arm has no value on that path");

  return agpu::emitIf(agpu_.context(), *cur_, *cond, results,
                      std::move(thenArm), thenYield, hasElse,
                      std::move(elseArm), elseYield);
}

agpu::Decision AgpuEmitter::emitWhileOp(scf::WhileOp wh) {
  // Carried values are named after the before region's arguments, which is
  // what the condition reads.
  agpu::Carried carried, inits;
  for (BlockArgument arg : wh.getBeforeArguments())
    carried.push_back(carriedFresh(arg));
  for (std::size_t i = 0; i < carried.size(); ++i) {
    const agpu::Result<agpu::CarriedValue> iv =
        carriedFrom(wh.getInits()[i], carried[i], "scf.while",
                    "a loop's initial value has no register names");
    if (!iv.ok())
      return iv.why;
    inits.push_back(iv.value);
  }

  agpu::Carried results;
  for (Value res : wh.getResults())
    results.push_back(carriedFresh(res));

  am::Block beforeArm;
  if (const agpu::Decision d = walkRegion(wh.getBefore(), beforeArm); !d.ok())
    return d;

  auto condOp =
      dyn_cast<scf::ConditionOp>(wh.getBefore().front().getTerminator());
  if (!condOp || condOp.getArgs().size() != results.size())
    return agpu::Decision::failed();

  const am::Str *cond = body_.sym.scalarName(idOf(condOp.getCondition()));
  if (!cond)
    return declined("scf.while", "the condition has no emitted name");

  agpu::Carried forwarded;
  for (std::size_t i = 0; i < results.size(); ++i) {
    const agpu::Result<agpu::CarriedValue> cv =
        carriedFrom(condOp.getArgs()[i], results[i], "scf.while",
                    "a forwarded value has no register names");
    if (!cv.ok())
      return cv.why;
    forwarded.push_back(cv.value);
    bindCarried(wh.getAfterArguments()[i], cv.value);
  }

  agpu::Carried yielded;
  am::Block afterArm;
  if (const agpu::Decision d = walkRegion(
          wh.getAfter(), afterArm,
          [&] {
            const agpu::Result<agpu::Carried> y = carriedOperands(
                wh.getAfter().front().getTerminator(), carried, "scf.while");
            yielded = y.value;
            return y.why;
          });
      !d.ok())
    return d;

  return agpu::emitWhile(agpu_.context(), *cur_, carried, inits,
                         std::move(beforeArm), *cond, results, forwarded,
                         std::move(afterArm), yielded);
}

} // namespace mlir::triton::applegpu::bridge
