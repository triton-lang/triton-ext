// AgpuDot - the tt.dot handler.
#include "AgpuDotChain.h"
#include "AgpuEmitter.h"
#include "AgpuLog.h"

#include <sstream>

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

void AgpuEmitter::logDotPlan(const DotOperands &ops, const agpu::Plan &plan) {
  if (!agpu_.gates.on(agpu::Gate::DotPlanDebug))
    return;
  llvm::errs() << "[dot] " << agpu::dotPlanReport(plan.facts, plan.fit);
  if (!ops.shape.cStore && !ops.shape.cStoreWhy.empty())
    llvm::errs() << " cStore: " << ops.shape.cStoreWhy;
  if (ops.shape.cStore && !ops.shape.cSteps.empty())
    llvm::errs() << " cFold: " << ops.shape.cSteps.size() << " step(s)"
                 << (ops.shape.cRowBound.present || ops.shape.cColBound.present
                         ? ", bounded"
                         : "");
  llvm::errs() << "\n";
}

agpu::Decision AgpuEmitter::emitDotOp(const agpu::OpView &o) {
  DotOperands ops = dotOperandsOf(o);
  if (!ops.ok())
    return ops.why;

  const agpu::DotFacts f = dotFactsOf(ops.shape);
  const agpu::Plan plan = agpu_.planFor(f);
  logDotPlan(ops, plan);

  if (plan.readsBackByRename()) {
    ops.cOutTy = renameLandingTypeOf(ops.shape);
    if (ops.cOutTy == ops.shape.cTy)
      ops.cOut = o.results[0];
  }

  if (plan.kind == agpu::Plan::Kind::Unsupported)
    return declined("tt.dot", "no strategy for this shape");

  agpu::DotInputs in;
  in.rollK = rollK_;
  if (const agpu::Decision d = stageDotOperands(ops, plan, in); !d.ok())
    return d;

  body_.armPending();
  const agpu::Decision d = agpu_.dot(*cur_, f, in);

  // Checked before `d`: an unplanned tile makes `d.ok()` say nothing
  // about correctness.
  if (!body_.pendingOk)
    return declined("tt.dot", body_.pendingWhy);

  if (!d.ok()) {
    if (d.isBug())
      return declined("tt.dot", "emitDot refused plan kind " +
                                    std::to_string((int)plan.kind) + " for " +
                                    std::to_string(f.M) + "x" +
                                    std::to_string(f.N) + "x" +
                                    std::to_string(f.K));
    return d;
  }

  // A fused dot's accumulators are the enclosing loop's to declare and
  // drain; left here for `emitForOp` to find. `result` holds the loop result
  // while `cOut` may differ, once the readback has absorbed the post-loop
  // convert and the loop must stop carrying the loop result.
  if (plan.accumulatorsOutlivePass()) {
    body_.fusedDots.push_back(FusedDot{plan, in.direct, in.coords.c,
                                       in.readbackFor, in.cStore, in.cSteps,
                                       idOf(ops.shape.cCarried)});
    // The store and its feeding chain must not also emit on their own.
    if (plan.storesCDirect()) {
      body_.absorbedOps.insert(ops.shape.cStore);
      for (Operation *o : ops.shape.cChainOps)
        body_.absorbedOps.insert(o);
    }
  }

  agpu::ValueNames names;
  for (int64_t r = 0; r < registerCount(ops.cOutTy); ++r)
    names.push_back(accName(ops.cOut, r));
  body_.sym.bindRegs(ops.cOut, std::move(names));
  // Notify the absorbed convert's handler, when there is one.
  if (ops.cOut != o.results[0] &&
      (!ops.shape.cCarried || ops.cOut != idOf(ops.shape.cCarried)))
    body_.absorbedInto.insert(ops.cOut);
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerDotHandler() {
  table_.add("dot", agpu::forOps({"tt.dot"}, [this](const agpu::OpView &o) {
               return emitDotOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
