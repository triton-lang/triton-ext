// AgpuDotChain - the fused-dot accumulator cycle. The proof here reads IR
// shape and never register bindings. Whether the drain is taken is a separate
// question, `cDirect`, which does read bindings.
#ifndef AGPU_BRIDGE_DOT_CHAIN_H
#define AGPU_BRIDGE_DOT_CHAIN_H

#include "AgpuFacts.h"

#include "agpu/plan/EpilogueOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>

namespace mlir::triton::applegpu::bridge {

// The result of a single-use `convert_layout` consuming `v`, or null.
inline Value convertAfter(Value v) {
  if (!v || !v.hasOneUse())
    return {};
  auto cv = dyn_cast<gpu::ConvertLayoutOp>(*v.getUsers().begin());
  return cv ? cv.getResult() : Value{};
}

inline Value pastConverts(Value v, std::vector<Operation *> &chainOps) {
  while (const Value cv = convertAfter(v)) {
    chainOps.push_back(cv.getDefiningOp());
    v = cv;
  }
  return v;
}

// Whether a C operand is a loop-carried value: the block argument of an
// scf.for.
inline bool carriedAccumulator(Value c) {
  auto arg = dyn_cast_or_null<BlockArgument>(c);
  return arg && isa<scf::ForOp>(arg.getOwner()->getParentOp());
}

// The loop result a dot's accumulator can be carried in as fragments, or null.
// The dot must be the iter arg's only reader: another reader would need the
// carried variable, which a fused loop never declares.
inline Value fusedAccumulator(Value c, Value result) {
  auto arg = dyn_cast_or_null<BlockArgument>(c);
  if (!arg || !arg.hasOneUse())
    return {};
  auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!forOp)
    return {};
  // Iter arg and matching result share an index, after the induction
  // variable at argument 0.
  const unsigned i = arg.getArgNumber();
  if (i == 0 || i > forOp.getNumResults())
    return {};

  if (!result || !result.hasOneUse())
    return {};
  OpOperand &use = *result.getUses().begin();
  auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
  if (!yield || yield->getParentOp() != forOp.getOperation() ||
      use.getOperandNumber() != i - 1)
    return {};
  return forOp.getResult(i - 1);
}

inline Value fusedInitOf(Value loopResult) {
  auto res = dyn_cast_or_null<OpResult>(loopResult);
  auto forOp = res ? dyn_cast<scf::ForOp>(res.getOwner()) : nullptr;
  if (!forOp)
    return {};
  return forOp.getInitArgs()[res.getResultNumber()];
}

// IR-level so the pool pre-pass and the handler answer identically; the
// walk's constant table does not exist yet when the pre-pass asks.
inline bool zeroSplat(Value v) {
  auto k = v ? v.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
  auto d = k ? dyn_cast<DenseElementsAttr>(k.getValue()) : DenseElementsAttr();
  if (!d || !d.isSplat())
    return false;
  const Attribute e = d.getSplatValue<Attribute>();
  if (auto fa = dyn_cast<FloatAttr>(e))
    return fa.getValue().isZero();
  if (auto ia = dyn_cast<IntegerAttr>(e))
    return ia.getValue().isZero();
  return false;
}

// The `tt.store` a fused accumulator drains into, and the single-use chain of
// converts and epilogue-table ops it folds on the way. The operands bound the
// fold and the arithmetic does not: each folded op's second operand must be
// readable at the drain's own coordinates and a mask must be the window's own
// bounds. A fanned-out loop result is allowed only where every consumer folds
// back in (gelu reads its accumulator twice, via `DrainStepFact::branch`).
struct FusedDrain {
  triton::StoreOp store;
  DeviceTile window;
  AxisBound rowBound, colBound;
  llvm::SmallVector<WindowBounds::Clamp, 2> clamps;
  // A tile-wide predicate the whole drain sits under. Null: none.
  Value uniformGuard;
  agpu::msl::SmallVec<DrainStepFact, 4> steps;
  // Ops a direct drain keeps the walk from emitting: chain converts, folded
  // ops, operands' single-use converts, any solely-folded operand load. The
  // store itself is `store`.
  std::vector<Operation *> chainOps;

  explicit operator bool() const { return (bool)store; }
};

// The loop result a drain-chain value hangs off, or null.
Value drainChainRootOf(Value v, int depth = 12);

// One attempt at proving a drain, from one consumer of the root. The state the
// proof threads through its stages lives here so the stages can be named
// functions at top level. One instance proves one
// spine: a fanned-out root builds a fresh one per consumer.
class DrainProof {
public:
  DrainProof(Value carried, Value root, const std::vector<Operation *> &headOps)
      : carried_(carried), root_(root) {
    out_.chainOps = headOps;
  }

  // Null on failure, with `why()` set to the link of the proof that broke.
  FusedDrain run(Operation *firstUser);
  const std::string &why() const { return why_; }

private:
  // An op folded into the drain, and which of its operands is the addend.
  struct FoldedOp {
    Operation *op;
    Value operand;
    bool roundBefore;
  };

  // One link of an operand's own chain back to the accumulator, before it is
  // classified into a DrainBranchLinkFact.
  struct BranchLinkRaw {
    Operation *op;
    Value operand;
  };

  bool spine(Operation *firstUser);
  bool storeWindow();
  bool foldSteps();
  bool checkFanOut();

  // DrainStepFact and DrainBranchLinkFact each declare their own `kind` enum,
  // so the one classification serves both as a template.
  template <class Target>
  std::string classifyOperand(Value operand, Target &target);

  // Ops from `user` to a store along single uses, or -1 if none is reached.
  static int hopsToStore(Operation *user);

  bool accChain(Value operand, agpu::msl::SmallVec<BranchLinkRaw, 4> &links,
                int &base, std::string &creason);

  bool operandPeel(Operation *def) const;
  Value lookThroughPeels(Value x) const;
  Value absorbOperandPeels(Value x);
  bool rootedAt(Value x) const { return drainChainRootOf(x) == carried_; }

  bool fail(std::string reason) {
    why_ = std::move(reason);
    return false;
  }

  Value carried_, root_;
  FusedDrain out_;
  triton::StoreOp store_;
  agpu::msl::SmallVec<FoldedOp, 4> folded_;
  // The one folded value allowed several consumers, and the spine step count
  // at which it was reached.
  Value fanOut_;
  int fanOutStep_ = 0;
  // A truncf, or a step computed on f16 tensors, rounds the running value. A
  // rounding not yet consumed by a step is pending; the consuming step records
  // it so the drain replays it in place.
  bool pendingRound_ = false;
  int64_t rows_ = 0, cols_ = 0;
  std::string why_;
};

// The fused-dot drain proof: whether the accumulator reaches a tt.store
// through a chain this backend can replay at the drain site. `why` receives
// the reason when it does not.
FusedDrain fusedDrainOf(Value carried, std::string *why = nullptr);

// The dot whose fused accumulator `loopResult` is, or null.
triton::DotOp fusedDotBehind(Value loopResult);

// Fill the C-side device window, when the fused chain ends in a store whose
// window the proof recognises. One function so the pool pre-pass and the
// handler answer identically.
void fillFusedCStore(DotShape &d);

// How this dot's accumulator is carried: whether it arrives as a loop
// iter-arg, whether the loop carries it in fragments and where its fused
// chain stores. The pool pre-pass and the handler reach C and result
// differently, so only the derivation is shared here.
void fillAccumulatorCarry(DotShape &d, Value c, Value result);

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_DOT_CHAIN_H
