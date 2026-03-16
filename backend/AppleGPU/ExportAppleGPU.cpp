/// Export Apple GPU backend passes and dialect via triton-ext plugin API.
/// Registers 5 passes + 1 dialect in a single shared library.

#include "Export.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/Conversion/Passes.h"

using namespace triton::ext::plugin;

static void addAccelerateMatmul(mlir::PassManager *pm) {
  pm->addPass(mlir::triton::applegpu::createAccelerateAppleMatmulPass());
}
static void registerAccelerateMatmul() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createAccelerateAppleMatmulPass();
  });
}

static void addSimplifyGather(mlir::PassManager *pm) {
  pm->addPass(mlir::triton::applegpu::createSimplifyGatherLayoutPass());
}
static void registerSimplifyGather() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createSimplifyGatherLayoutPass();
  });
}

static void addToLLVMIR(mlir::PassManager *pm) {
  pm->addPass(mlir::triton::applegpu::createConvertTritonAppleGPUToLLVMPass());
}
static void registerToLLVMIR() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createConvertTritonAppleGPUToLLVMPass();
  });
}

static void addLowerGPUToAIR(mlir::PassManager *pm) {
  pm->addPass(mlir::triton::applegpu::createLowerGPUToAirPass());
}
static void registerLowerGPUToAIR() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createLowerGPUToAirPass();
  });
}

static void addReconcileUnrealizedCasts(mlir::PassManager *pm) {
  pm->addPass(mlir::createReconcileUnrealizedCastsPass());
}
static void registerReconcileUnrealizedCasts() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });
}

static void insertAppleGPUDialect(mlir::DialectRegistry *registry) {
  registry->insert<mlir::triton::applegpu::TritonAppleGPUDialect>();
}

static auto _p1 = exportPass("add_accelerate_matmul",
                              registerAccelerateMatmul, addAccelerateMatmul);
static auto _p2 = exportPass("add_simplify_gather",
                              registerSimplifyGather, addSimplifyGather);
static auto _p3 = exportPass("add_to_llvmir",
                              registerToLLVMIR, addToLLVMIR);
static auto _p4 = exportPass("add_lower_gpu_to_air",
                              registerLowerGPUToAIR, addLowerGPUToAIR);
static auto _p5 = exportPass("add_reconcile_unrealized_casts",
                              registerReconcileUnrealizedCasts,
                              addReconcileUnrealizedCasts);

static auto _d1 = exportDialect("TritonAppleGPU", insertAppleGPUDialect);
