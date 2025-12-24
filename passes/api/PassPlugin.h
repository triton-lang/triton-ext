#ifndef TPAPI_PASSPLUGIN_H
#define TPAPI_PASSPLUGIN_H
// Set PLUGIN_NAME to the name of the plugin
// Set PLUGIN_PASS_FUNC to the pass creation function

#include "triton/Tools/PluginUtils.h"

// Key APIs: entry points to load and register the plugin
TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *passName);

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *passName);

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames);

// Plugin API registration
typedef void (*TPAPIRegisterPassFunc)();
typedef void (*TPAPIAddPassFunc)(mlir::PassManager *);

TRITON_PLUGIN_API
TPAPILoadPass(const char *passName,
    TPAPIRegisterPassFunc registerFunc,
    TPAPIAddPassFunc addFunc);

#ifdef PLUGIN_NAME
// User override
// Plugin pass creation and registration functions
#define PLUGIN_PASS_CREATE_FUNC addTritonPlugin##PLUGIN_NAME##Pass
#define PLUGIN_PASS_REGISTER_FUNC registerTritonPlugin##PLUGIN_NAME##Pass

// Plugin pass creation function
static void PLUGIN_PASS_CREATE_FUNC(mlir::PassManager *pm) {
    pm->addPass(PLUGIN_PASS_CLASS());
}

// Plugin pass registration function
static void PLUGIN_PASS_REGISTER_FUNC() {
    ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
        return PLUGIN_PASS_CLASS();
    });
}

static TritonPluginResult initPlugin = TPAPILoadPass(PLUGIN_NAME, PLUGIN_PASS_REGISTER_FUNC, PLUGIN_PASS_CREATE_FUNC);

#endif // PLUGIN_NAME

#endif // TPAPI_PASSPLUGIN_H