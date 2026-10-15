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

std::unique_ptr<mlir::Pass> createLowerWarpBarrierArrivePass();
void registerLowerWarpBarrierArrivePass();

std::unique_ptr<mlir::Pass> createOptimizeDescriptorEncodingPass();
void registerOptimizeDescriptorEncodingPass();

// Give the kernel-argument descriptor `desc` the NVMMA layout of the shared
// view a TMA op copies through, updating the function signature. Returns
// whether the descriptor now matches.
bool matchTensorDescToSharedLayout(mlir::Value desc,
                                   mlir::Attribute sharedEncoding);

// Layout passes
std::unique_ptr<mlir::Pass> createDotOperandLocalLoadPass();
void registerDotOperandLocalLoadPass();
std::unique_ptr<mlir::Pass> createKeepLoadLayoutPass();
void registerKeepLoadLayoutPass();
std::unique_ptr<mlir::Pass> createRematDotOperandSlicesPass();
void registerRematDotOperandSlicesPass();

// AMD passes
std::unique_ptr<mlir::Pass> createFallbackAsyncCopyPass();
void registerFallbackAsyncCopyPass();

std::unique_ptr<mlir::Pass> createAMDLowerBarrierOpsPass();
void registerAMDLowerBarrierOpsPass();

// AMD conversion patterns (not a standalone pass)
void populateAMDBarrierOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns,
                                        mlir::PatternBenefit benefit);

} // namespace utlx

#endif // UTLX_PASSES_PASSES_H
