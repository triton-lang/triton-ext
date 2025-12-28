#ifndef TRITON_EXT_PASS_INFRA_H
#define TRITON_EXT_PASS_INFRA_H
// Set PLUGIN_NAME to the name of the plugin
// Set PLUGIN_PASS_FUNC to the pass creation function

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Tools/PluginUtils.h"

// Key APIs: entry points to load and register the plugin
TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *passName);

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *passName);

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames);

// Plugin API registration
typedef void (*TritonExtRegisterPassFunc)();
typedef void (*TritonExtAddPassFunc)(mlir::PassManager *);

TRITON_PLUGIN_API
TritonExtLoadPass(const char *passName,
    TritonExtRegisterPassFunc registerFunc,
    TritonExtAddPassFunc addFunc);

#endif // TRITON_EXT_PASS_INFRA_H