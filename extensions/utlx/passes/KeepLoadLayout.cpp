/// Undo coalescing of loads whose register layout the kernel chose.
///
/// Fork TLX lowers tlx.buffer_load to amdg.buffer_load, a scalar base plus a
/// tensor of offsets. Coalesce only rewrites tensor-of-pointer accesses, so the
/// load keeps the layout of its offsets. Kernels rely on this to load straight
/// into a dot operand layout, e.g. a K or V tile held in MFMA registers.
///
/// uTLX emulates buffer_load as a tt.load of `base + offsets` and marks it
/// with `utlx.keep_layout`. Coalesce gives that load a blocked layout and
/// converts back. Stock RemoveLayoutConversions never folds a convert to a dot
/// operand layout into its producer, so ReduceDataDuplication stages it
/// through a shared-memory buffer the size of the tile. That can push the
/// kernel over the LDS limit.
///
/// This pass runs right after Coalesce. For each marked load whose operands
/// are all converted from one layout, it rebuilds the load in that layout and
/// removes the converts Coalesce added.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kKeepLayoutAttr = "utlx.keep_layout";

// The layout `load` had before Coalesce, or null if it cannot be recovered.
mlir::Attribute originalEncoding(tt::LoadOp load) {
  mlir::Attribute encoding;
  for (mlir::Value operand : load->getOperands()) {
    auto tensorTy = dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (!tensorTy)
      continue;
    auto convert = operand.getDefiningOp<ttg::ConvertLayoutOp>();
    if (!convert)
      return {};
    mlir::Attribute srcEncoding = convert.getSrc().getType().getEncoding();
    if (encoding && encoding != srcEncoding)
      return {};
    encoding = srcEncoding;
  }
  return encoding;
}

void restore(tt::LoadOp load, mlir::Attribute encoding) {
  mlir::OpBuilder b(load);
  llvm::SmallVector<mlir::Value> operands;
  for (mlir::Value operand : load->getOperands()) {
    if (auto convert = operand.getDefiningOp<ttg::ConvertLayoutOp>())
      operand = convert.getSrc();
    operands.push_back(operand);
  }
  auto loadTy = cast<mlir::RankedTensorType>(load.getType());
  mlir::Type resultTy = loadTy.cloneWithEncoding(encoding);
  mlir::Operation *newLoad =
      b.create(load.getLoc(), load->getName().getIdentifier(), operands,
               mlir::TypeRange{resultTy}, load->getAttrs());
  newLoad->removeAttr(kKeepLayoutAttr);
  mlir::Value result = newLoad->getResult(0);

  // Users that convert back to the original layout take the new load directly;
  // anything else keeps seeing the coalesced layout.
  mlir::Value coalesced;
  for (mlir::OpOperand &use :
       llvm::make_early_inc_range(load.getResult().getUses())) {
    auto convert = dyn_cast<ttg::ConvertLayoutOp>(use.getOwner());
    if (convert && convert.getType() == resultTy) {
      convert.getResult().replaceAllUsesWith(result);
      convert.erase();
      continue;
    }
    if (!coalesced)
      coalesced = ttg::ConvertLayoutOp::create(b, load.getLoc(), load.getType(),
                                               result);
    use.set(coalesced);
  }

  llvm::SmallVector<mlir::Operation *> inputs;
  for (mlir::Value operand : load->getOperands())
    if (auto convert = operand.getDefiningOp<ttg::ConvertLayoutOp>())
      inputs.push_back(convert);
  load.erase();
  for (mlir::Operation *convert : inputs)
    if (convert->use_empty())
      convert->erase();
}

class UTLXKeepLoadLayoutPass
    : public mlir::PassWrapper<UTLXKeepLoadLayoutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXKeepLoadLayoutPass)

  llvm::StringRef getArgument() const override {
    return "utlx-keep-load-layout";
  }
  llvm::StringRef getDescription() const override {
    return "Undo coalescing of loads whose register layout the kernel chose";
  }

  void runOnOperation() override {
    llvm::SmallVector<tt::LoadOp> loads;
    getOperation().walk([&](tt::LoadOp load) {
      if (load->hasAttr(kKeepLayoutAttr))
        loads.push_back(load);
    });
    for (tt::LoadOp load : loads) {
      mlir::Attribute encoding = originalEncoding(load);
      auto loadTy = dyn_cast<mlir::RankedTensorType>(load.getType());
      if (encoding && loadTy && encoding != loadTy.getEncoding())
        restore(load, encoding);
      else
        load->removeAttr(kKeepLayoutAttr);
    }
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createKeepLoadLayoutPass() {
  return std::make_unique<UTLXKeepLoadLayoutPass>();
}

void registerKeepLoadLayoutPass() {
  mlir::PassRegistration<UTLXKeepLoadLayoutPass>();
}

} // namespace utlx
