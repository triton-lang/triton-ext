/// LowerWarpBarrierArrive: lower tlx.warp_barrier_arrive to PTX.
///
/// tlx.warp_barrier_arrive is uTLX's stand-in for fbtriton's
/// `ttng.arrive_barrier {perThread}`. Upstream Triton has neither the attribute
/// nor a plugin hook into its TritonGPU-to-LLVM patterns, so the op passes
/// through that conversion untouched: its barrier operand becomes an
/// unrealized_conversion_cast of the lowered shared-memory object. This pass
/// runs right after the conversion and emits what fbtriton's
/// ArriveBarrierOpConversion emits for a perThread arrive:
///
///   mbarrier.arrive.shared::cta.b64 _, [addr](, count);
///
/// from every thread, with no leader predicate and no barrier before it.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "tlx/dialect/include/IR/Dialect.h"

namespace tlx = mlir::triton::tlx;

namespace {

// The base address of the barrier: field 0 of the lowered shared-memory
// object, as LLVM::getSharedMemoryObjectFromStruct reads it.
mlir::Value getBarrierAddress(mlir::OpBuilder &builder,
                              tlx::WarpBarrierArriveOp op) {
  auto cast = op.getAlloc().getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast.getNumOperands() != 1)
    return {};
  mlir::Value smemObj = cast.getOperand(0);
  auto structTy = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(smemObj.getType());
  if (!structTy || structTy.getBody().empty() ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(structTy.getBody()[0]))
    return {};
  return mlir::LLVM::ExtractValueOp::create(builder, op.getLoc(), smemObj, 0);
}

class UTLXLowerWarpBarrierArrivePass
    : public mlir::PassWrapper<UTLXLowerWarpBarrierArrivePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXLowerWarpBarrierArrivePass)

  llvm::StringRef getArgument() const override {
    return "utlx-lower-warp-barrier-arrive";
  }
  llvm::StringRef getDescription() const override {
    return "Lower tlx.warp_barrier_arrive to a per-thread mbarrier.arrive "
           "after TritonGPU-to-LLVM conversion";
  }

  void runOnOperation() override {
    llvm::SmallVector<tlx::WarpBarrierArriveOp> arrives;
    getOperation().walk(
        [&](tlx::WarpBarrierArriveOp op) { arrives.push_back(op); });

    for (auto op : arrives) {
      mlir::OpBuilder builder(op);
      mlir::Value addr = getBarrierAddress(builder, op);
      if (!addr) {
        op.emitError("utlx-lower-warp-barrier-arrive must run right after "
                     "TritonGPU-to-LLVM conversion; the barrier operand is "
                     "not a lowered shared-memory object");
        return signalPassFailure();
      }

      std::string ptx = "mbarrier.arrive.shared::cta.b64 _, [$0]";
      if (op.getCount() > 1)
        ptx += ", " + std::to_string(op.getCount());
      ptx += ";";
      auto asmDialect = mlir::LLVM::AsmDialectAttr::get(
          builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT);
      mlir::LLVM::InlineAsmOp::create(
          builder, op.getLoc(), mlir::TypeRange(), mlir::ValueRange{addr}, ptx,
          "r", /*has_side_effects=*/true, /*is_align_stack=*/false,
          mlir::LLVM::TailCallKind::None, asmDialect, mlir::ArrayAttr());

      mlir::Operation *cast = op.getAlloc().getDefiningOp();
      op.erase();
      if (cast->use_empty())
        cast->erase();
    }
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createLowerWarpBarrierArrivePass() {
  return std::make_unique<UTLXLowerWarpBarrierArrivePass>();
}

void registerLowerWarpBarrierArrivePass() {
  mlir::PassRegistration<UTLXLowerWarpBarrierArrivePass>();
}

} // namespace utlx
