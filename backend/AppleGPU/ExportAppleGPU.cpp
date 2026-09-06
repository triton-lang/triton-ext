// Exports the Apple GPU passes and dialect through the Triton plugin API.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUToMSL/Passes.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "triton/Tools/PluginUtils.h"

#include <iterator>

static void addAccelerateMatmul(mlir::PassManager *pm,
                                const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createAccelerateAppleMatmulPass());
}
static void addStoreShuffleLayout(mlir::PassManager *pm,
                                  const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createStoreShuffleLayoutPass());
}
static void addEmitMSL(mlir::PassManager *pm,
                       const std::vector<std::string> &args) {
  pm->addPass(mlir::triton::applegpu::createEmitMSLPass(
      args.empty() ? std::string() : args[0]));
}

static void registerAccelerateMatmul() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createAccelerateAppleMatmulPass();
  });
}
static void registerStoreShuffleLayout() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createStoreShuffleLayoutPass();
  });
}
static void registerEmitMSL() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createEmitMSLPass();
  });
}

static void insertAppleGPUDialect(mlir::DialectRegistry *registry) {
  registry->insert<mlir::triton::applegpu::TritonAppleGPUDialect>();
}

using namespace mlir::triton;

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo passes[] = {
      {"accelerate_matmul", "0.1.0", addAccelerateMatmul,
       registerAccelerateMatmul},
      {"store_shuffle_layout", "0.1.0", addStoreShuffleLayout,
       registerStoreShuffleLayout},
      {"emit_msl", "0.1.0", addEmitMSL, registerEmitMSL},
  };

  static plugin::DialectInfo dialects[] = {
      {"TritonAppleGPU", "0.1.0", insertAppleGPUDialect},
  };

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "TritonAppleGPUBackend",
      "0.1.0",
      passes,
      std::size(passes),
      dialects,
      std::size(dialects),
      nullptr,
      0, // numOps
      TRITON_VERSION,
  };
  return &info;
}
