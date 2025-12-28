#include "TritonExtPassInfra.h"

#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define TRITON_EXT_PASS_CREATE_FUNC CONCAT(createTritonExt, TRITON_EXT_CLASS)

namespace mlir {
namespace triton_plugin {
std::unique_ptr<Pass> TRITON_EXT_PASS_CREATE_FUNC() {
  return std::make_unique<TRITON_EXT_CLASS>();
}
} // namespace triton_plugin
} // namespace mlir

using namespace ::mlir::triton_plugin;

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
    TritonExtLoadPass(TOSTRING(TRITON_EXT_NAME), TRITON_EXT_PASS_REGISTER_FUNC,
                      TRITON_EXT_PASS_ADD_FUNC);
