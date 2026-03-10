/// Export a pass using the Triton plugin API. This file is meant to be compiled
/// in with the plugin implementation, exposing the necessary functions to
/// register the plugin with Triton.

#include "Export.h"
#include "StringMacros.h"

#define TRITON_EXT_PASS_CREATE_FUNC CONCAT(createTritonExt, TRITON_EXT_CLASS)
namespace triton::ext::plugin {
std::unique_ptr<Pass> TRITON_EXT_PASS_CREATE_FUNC() {
  return std::make_unique<TRITON_EXT_CLASS>();
}
} // namespace triton::ext::plugin

using namespace ::triton::ext::plugin;

// Plugin pass creation and registration functions
#define TRITON_EXT_PASS_ADD_FUNC CONCAT(addTritonExt, TRITON_EXT_CLASS)
#define TRITON_EXT_PASS_REGISTER_FUNC                                          \
  CONCAT(registerTritonExt, TRITON_EXT_CLASS)

// TritonExt pass creation function
static void TRITON_EXT_PASS_ADD_FUNC(mlir::PassManager *pm) {
  pm->addPass(TRITON_EXT_PASS_CREATE_FUNC());
}

// TritonExt pass registration function
static void TRITON_EXT_PASS_REGISTER_FUNC() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return TRITON_EXT_PASS_CREATE_FUNC();
  });
}

static TritonPluginResult initPlugin =
    exportPass(TOSTRING(TRITON_EXT_NAME), TRITON_EXT_PASS_REGISTER_FUNC,
               TRITON_EXT_PASS_ADD_FUNC);
