// Exports the Apple GPU passes and dialect through the Triton plugin API.

#include "TritonAppleGPUToMSL/Passes.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "triton/Tools/PluginUtils.h"

#include <iterator>

static void addEmitMSL(mlir::PassManager *pm,
                       const std::vector<std::string> &args) {
  pm->addPass(mlir::triton::applegpu::createEmitMSLPass(
      args.empty() ? std::string() : args[0]));
}

static void registerEmitMSL() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createEmitMSLPass();
  });
}

using namespace mlir::triton;

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo passes[] = {
      {"emit_msl", "0.1.0", addEmitMSL, registerEmitMSL},
  };

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "TritonAppleGPUBackend",
      "0.1.0",
      passes,
      std::size(passes),
      nullptr,
      0, // numDialects
      nullptr,
      0, // numOps
      TRITON_VERSION,
  };
  return &info;
}
