#ifndef UTLX_PASSES_PASSES_H
#define UTLX_PASSES_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace utlx {

// NVIDIA passes
std::unique_ptr<mlir::Pass> createPruneUnusedBarriersPass();
void registerPruneUnusedBarriersPass();

std::unique_ptr<mlir::Pass> createPingPongPrepPass();
void registerPingPongPrepPass();

std::unique_ptr<mlir::Pass> createPingPongSyncPass();
void registerPingPongSyncPass();

// AMD passes
std::unique_ptr<mlir::Pass> createAMDLowerBarrierOpsPass();
void registerAMDLowerBarrierOpsPass();

// AMD conversion patterns (not a standalone pass)
void populateAMDBarrierOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns,
                                        mlir::PatternBenefit benefit);

} // namespace utlx

#endif // UTLX_PASSES_PASSES_H
