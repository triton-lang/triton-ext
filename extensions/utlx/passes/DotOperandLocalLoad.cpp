/// Load loop-carried dot operands from shared memory in their dot layout.
///
/// TLX kernels often software-pipeline by hand: a tile is read with
/// tlx.local_load before the loop, carried through scf.for, and refilled with
/// another local_load at the end of each iteration. tl.dot then consumes the
/// loop-carried value, so AccelerateMatmul puts the convert to the dot-operand
/// layout inside the loop, on the block argument.
///
/// Stock RemoveLayoutConversions never rematerializes a convert to a dot
/// operand layout backwards. The convert therefore survives, and
/// ReduceDataDuplication later stages it through a fresh shared-memory buffer.
/// That costs an extra LDS round trip per dot, plus scratch allocations that
/// can push an otherwise fitting kernel over the shared-memory limit. Fork TLX
/// rematerializes these converts outside pipelined loops.
///
/// This pass handles the case that matters for TLX. It considers each convert
/// to a dot-operand layout whose source can be traced through scf.for, scf.if
/// and scf.execute_region values back to local_loads alone. If every value on
/// those paths is used only by converts to the same layout, or by the
/// region-carrying edges themselves, the local_loads produce the dot layout
/// directly. The intermediate values are retyped and the converts disappear.
/// Anything else is left for the normal passes.

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

namespace ttg = mlir::triton::gpu;

namespace {

using mlir::Value;

class DotOperandSlice {
public:
  explicit DotOperandSlice(ttg::DotOperandEncodingAttr encoding)
      : encoding(encoding) {}

  // Grows the slice from `root`; returns false if it cannot be retyped.
  bool build(Value root) {
    worklist.push_back(root);
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!values.insert(v))
        continue;
      if (!visitDef(v) || !visitUses(v))
        return false;
    }
    return true;
  }

  // Retypes the slice and erases its converts, recording them in `erased`.
  void rewrite(llvm::DenseSet<mlir::Operation *> &erased) {
    for (Value v : values) {
      auto type = cast<mlir::RankedTensorType>(v.getType());
      v.setType(type.cloneWithEncoding(encoding));
    }
    for (ttg::ConvertLayoutOp convert : converts) {
      convert.getResult().replaceAllUsesWith(convert.getSrc());
      erased.insert(convert);
      convert.erase();
    }
  }

private:
  void add(Value v) {
    if (!values.contains(v))
      worklist.push_back(v);
  }

  static Value yieldedFrom(mlir::Region &region, unsigned idx) {
    return region.front().getTerminator()->getOperand(idx);
  }

  bool visitDef(Value v) {
    if (auto arg = dyn_cast<mlir::BlockArgument>(v)) {
      auto forOp = dyn_cast<mlir::scf::ForOp>(arg.getOwner()->getParentOp());
      if (!forOp || arg.getArgNumber() < forOp.getNumInductionVars())
        return false;
      add(forOp.getTiedLoopInit(arg)->get());
      add(forOp.getTiedLoopYieldedValue(arg)->get());
      add(forOp.getTiedLoopResult(arg));
      return true;
    }
    mlir::Operation *def = v.getDefiningOp();
    unsigned idx = cast<mlir::OpResult>(v).getResultNumber();
    if (auto load = dyn_cast<ttg::LocalLoadOp>(def))
      return load.getSrc().getType().getRank() ==
             cast<mlir::RankedTensorType>(v.getType()).getRank();
    if (auto forOp = dyn_cast<mlir::scf::ForOp>(def)) {
      add(forOp.getRegionIterArgs()[idx]);
      return true;
    }
    if (auto ifOp = dyn_cast<mlir::scf::IfOp>(def)) {
      add(yieldedFrom(ifOp.getThenRegion(), idx));
      add(yieldedFrom(ifOp.getElseRegion(), idx));
      return true;
    }
    if (auto region = dyn_cast<mlir::scf::ExecuteRegionOp>(def)) {
      for (mlir::Block &block : region.getRegion())
        if (auto yield = dyn_cast<mlir::scf::YieldOp>(block.getTerminator()))
          add(yield.getOperand(idx));
        else
          return false;
      return true;
    }
    return false;
  }

  bool visitUses(Value v) {
    for (mlir::OpOperand &use : v.getUses()) {
      mlir::Operation *user = use.getOwner();
      unsigned idx = use.getOperandNumber();
      if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(user)) {
        if (convert.getType().getEncoding() != encoding)
          return false;
        converts.insert(convert);
        continue;
      }
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(user)) {
        if (idx < forOp.getNumControlOperands())
          return false;
        add(forOp.getTiedLoopRegionIterArg(&use));
        continue;
      }
      if (auto yield = dyn_cast<mlir::scf::YieldOp>(user)) {
        mlir::Operation *parent = yield->getParentOp();
        if (isa<mlir::scf::ForOp>(parent)) {
          auto forOp = cast<mlir::scf::ForOp>(parent);
          add(forOp.getRegionIterArgs()[idx]);
          add(forOp->getResult(idx));
          continue;
        }
        if (isa<mlir::scf::IfOp, mlir::scf::ExecuteRegionOp>(parent)) {
          add(parent->getResult(idx));
          continue;
        }
      }
      return false;
    }
    return true;
  }

  ttg::DotOperandEncodingAttr encoding;
  llvm::SetVector<Value> values;
  llvm::SetVector<ttg::ConvertLayoutOp> converts;
  llvm::SmallVector<Value> worklist;
};

class UTLXDotOperandLocalLoadPass
    : public mlir::PassWrapper<UTLXDotOperandLocalLoadPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXDotOperandLocalLoadPass)

  llvm::StringRef getArgument() const override {
    return "utlx-dot-operand-local-load";
  }
  llvm::StringRef getDescription() const override {
    return "Load loop-carried dot operands from shared memory in their dot "
           "layout";
  }

  void runOnOperation() override {
    llvm::SmallVector<ttg::ConvertLayoutOp> candidates;
    getOperation().walk([&](ttg::ConvertLayoutOp convert) {
      if (isa<ttg::DotOperandEncodingAttr>(convert.getType().getEncoding()))
        candidates.push_back(convert);
    });
    llvm::DenseSet<mlir::Operation *> erased;
    for (ttg::ConvertLayoutOp convert : candidates) {
      if (erased.contains(convert))
        continue;
      DotOperandSlice slice(
          cast<ttg::DotOperandEncodingAttr>(convert.getType().getEncoding()));
      if (!slice.build(convert.getSrc()))
        continue;
      slice.rewrite(erased);
    }
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createDotOperandLocalLoadPass() {
  return std::make_unique<UTLXDotOperandLocalLoadPass>();
}

void registerDotOperandLocalLoadPass() {
  mlir::PassRegistration<UTLXDotOperandLocalLoadPass>();
}

} // namespace utlx
