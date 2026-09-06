// scf.for, scf.if and scf.while lowered to MSL loops and variables.
#include "AgpuEmitter.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

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

    agpu::CarriedValue iv;
    if (const agpu::Decision d =
            carriedFrom(init, cv, iv, "scf.for",
                        "a loop's initial value has no register names");
        !d.ok())
      return d;

    carriedNames.push_back(cv.regs);
    initNamesOf.push_back(iv.regs);
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

      agpu::CarriedValue got;
      if (const agpu::Decision d =
              carriedFrom(y.getOperand(i), carriedShape[i], got, "scf.for",
                          "a yielded value has no register names");
          !d.ok())
        return d;
      yieldValues.push_back(got);
    }
    return agpu::Decision::emitted();
  });
  std::vector<FusedDot> fused;
  fused.swap(body_.fusedDots);
  body_.fusedDots.swap(enclosing);
  if (!bodyDone.ok())
    return bodyDone.isBug() ? declined("scf.for", "an op in the loop body")
                            : bodyDone;

  // A fused dot's result lives in accumulator fragments.
  agpu::Carried carried, inits, yielded;
  for (std::size_t i = 0; i < n; ++i) {
    const Value res = forOp.getResult(i);
    if (inFragments(fused, idOf(res)))
      continue;

    if (const agpu::Decision d = carriedFor(res, carried, carriedNames[i]);
        !d.ok())
      return d;
    if (const agpu::Decision d = carriedFor(res, inits, initNamesOf[i]);
        !d.ok())
      return d;
    yielded.push_back(yieldValues[i]);
  }

  const auto loop = [&]() -> agpu::Decision {
    return agpu::emitFor(agpu_.context(), *cur_, b, carried, inits,
                         std::move(body), yielded);
  };

  // Each bracket wraps the loop the previous ones built.
  std::function<agpu::Decision()> bracketed = loop;
  for (const FusedDot &fd : fused)
    bracketed = [&fd, inner = bracketed, this]() {
      return agpu::emitFusedLoop(agpu_.context(), *cur_, fd.plan, fd.names,
                                 fd.readbackFor, fd.cCoords, fd.cStore,
                                 fd.cSteps, inner);
    };
  return bracketed();
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
      return carriedOperands(region.front().getTerminator(), results, yielded,
                             "scf.if");
    });
  };

  am::Block thenArm, elseArm;
  agpu::Carried thenYield, elseYield;
  if (const agpu::Decision d =
          walkArm(ifOp.getThenRegion(), thenArm, thenYield);
      !d.ok())
    return d.isBug() ? declined("scf.if", "an op in the then arm") : d;

  const bool hasElse = !ifOp.getElseRegion().empty();
  if (hasElse)
    if (const agpu::Decision d =
            walkArm(ifOp.getElseRegion(), elseArm, elseYield);
        !d.ok())
      return d.isBug() ? declined("scf.if", "an op in the else arm") : d;

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
    agpu::CarriedValue iv;
    if (const agpu::Decision d =
            carriedFrom(wh.getInits()[i], carried[i], iv, "scf.while",
                        "a loop's initial value has no register names");
        !d.ok())
      return d;
    inits.push_back(iv);
  }

  agpu::Carried results;
  for (Value res : wh.getResults())
    results.push_back(carriedFresh(res));

  am::Block beforeArm;
  if (const agpu::Decision d = walkRegion(wh.getBefore(), beforeArm); !d.ok())
    return d.isBug() ? declined("scf.while", "an op in the condition") : d;

  auto condOp =
      dyn_cast<scf::ConditionOp>(wh.getBefore().front().getTerminator());
  if (!condOp || condOp.getArgs().size() != results.size())
    return agpu::Decision::failed();

  const am::Str *cond = body_.sym.scalarName(idOf(condOp.getCondition()));
  if (!cond)
    return declined("scf.while", "the condition has no emitted name");

  agpu::Carried forwarded;
  for (std::size_t i = 0; i < results.size(); ++i) {
    agpu::CarriedValue cv;
    if (const agpu::Decision d =
            carriedFrom(condOp.getArgs()[i], results[i], cv, "scf.while",
                        "a forwarded value has no register names");
        !d.ok())
      return d;
    forwarded.push_back(cv);
    bindCarried(wh.getAfterArguments()[i], cv);
  }

  agpu::Carried yielded;
  am::Block afterArm;
  if (const agpu::Decision d = walkRegion(
          wh.getAfter(), afterArm,
          [&] {
            return carriedOperands(wh.getAfter().front().getTerminator(),
                                   carried, yielded, "scf.while");
          });
      !d.ok())
    return d.isBug() ? declined("scf.while", "an op in the loop body") : d;

  return agpu::emitWhile(agpu_.context(), *cur_, carried, inits,
                         std::move(beforeArm), *cond, results, forwarded,
                         std::move(afterArm), yielded);
}

} // namespace mlir::triton::applegpu::bridge
