// Masks loads whose value only reaches one arm of a select with that arm's
// condition, so lanes that discard the arm skip the memory access.

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

namespace tt = mlir::triton;
using namespace mlir;

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_MASKSELECTARMLOADS
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

// Element i of the op's result reads only element i of its operands, so a
// mask on element i of a load in the cone lines up with the select's.
static bool isElementwise(Operation *op, Type selTy) {
  if (op->getNumResults() != 1)
    return false;
  auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto sty = dyn_cast<RankedTensorType>(selTy);
  if (bool(ty) != bool(sty) || (ty && ty.getShape() != sty.getShape()))
    return false;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return !load.getIsVolatile();
  const StringRef n = op->getName().getStringRef();
  if (n.starts_with("arith.") || n.starts_with("math."))
    return true;
  return llvm::is_contained({"tt.extern_elementwise", "tt.precise_divf",
                             "tt.precise_sqrt", "tt.mulhiui", "tt.fp_to_fp",
                             "tt.bitcast", "tt.clampf", "tt.addptr"},
                            n);
}

// Elementwise ops in `sel`'s block whose every use ends in operand `arm` of
// `sel`.
static SetVector<Operation *> exclusiveCone(arith::SelectOp sel, unsigned arm) {
  SetVector<Operation *> cone;
  SmallVector<Value> work{sel->getOperand(arm)};
  while (!work.empty()) {
    Operation *def = work.pop_back_val().getDefiningOp();
    if (!def || def->getBlock() != sel->getBlock() || def->getNumRegions() ||
        cone.contains(def) || !isElementwise(def, sel.getType()))
      continue;
    cone.insert(def);
    for (Value v : def->getOperands())
      work.push_back(v);
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::make_early_inc_range(cone)) {
      bool exclusive = llvm::all_of(op->getUses(), [&](OpOperand &use) {
        if (use.getOwner() == sel.getOperation())
          return use.getOperandNumber() == arm;
        return cone.contains(use.getOwner());
      });
      if (!exclusive) {
        cone.remove(op);
        changed = true;
      }
    }
  }
  return cone;
}

// Moves the pure ops defining `v` above `pt` until `v` dominates it.
static bool hoistAbove(Value v, Operation *pt, DominanceInfo &dom) {
  if (dom.properlyDominates(v, pt))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != pt->getBlock() || def->getNumRegions() ||
      !isMemoryEffectFree(def))
    return false;
  for (Value operand : def->getOperands())
    if (!hoistAbove(operand, pt, dom))
      return false;
  def->moveBefore(pt);
  return true;
}

class MaskSelectArmLoadsPass
    : public impl::MaskSelectArmLoadsBase<MaskSelectArmLoadsPass> {
public:
  void runOnOperation() override {
    SmallVector<arith::SelectOp> selects;
    getOperation().walk([&](arith::SelectOp sel) { selects.push_back(sel); });
    for (arith::SelectOp sel : selects)
      for (unsigned arm : {1u, 2u})
        maskArm(sel, arm);
  }

private:
  void maskArm(arith::SelectOp sel, unsigned arm) {
    SmallVector<tt::LoadOp> loads;
    for (Operation *op : exclusiveCone(sel, arm))
      if (auto load = dyn_cast<tt::LoadOp>(op))
        loads.push_back(load);
    if (loads.empty())
      return;
    DominanceInfo dom(getOperation());
    Value cond = sel.getCondition();
    for (tt::LoadOp load : loads) {
      // A masked-off lane needs a value, and a loaded pointer has no zero.
      TypedAttr zero;
      if (!load.getOther() &&
          !(zero = Builder(load.getContext()).getZeroAttr(load.getType())))
        continue;
      Type maskTy = tt::getI1SameShape(load.getType());
      bool splat = cond.getType() != maskTy;
      if (splat &&
          !(cond.getType().isInteger(1) && isa<RankedTensorType>(maskTy)))
        continue;
      if (!hoistAbove(cond, load, dom))
        continue;
      OpBuilder b(load);
      Location loc = load.getLoc();
      Value c = cond;
      if (splat)
        c = tt::SplatOp::create(b, loc, maskTy, c);
      if (arm == 2) {
        Value one =
            arith::ConstantOp::create(b, loc, maskTy, b.getOneAttr(maskTy));
        c = arith::XOrIOp::create(b, loc, c, one);
      }
      if (Value mask = load.getMask())
        c = arith::AndIOp::create(b, loc, mask, c);
      load.getMaskMutable().assign(c);
      if (zero)
        load.getOtherMutable().assign(
            arith::ConstantOp::create(b, loc, load.getType(), zero));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createMaskSelectArmLoadsPass() {
  return std::make_unique<MaskSelectArmLoadsPass>();
}

} // namespace mlir::triton::applegpu
