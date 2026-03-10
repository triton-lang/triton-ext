#ifndef TRITON_EXT_PASS_INFRA_H
#define TRITON_EXT_PASS_INFRA_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Tools/PluginUtils.h"
#include <string>

///
/// External APIs: entry points to load and register the plugin.
///
TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *passName);

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *passName);

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames);

TRITON_PLUGIN_API
tritonEnumeratePluginDialects(uint32_t *outDialectCount,
                              const char **outDialectNames);

TRITON_PLUGIN_API_TYPE(mlir::DialectPluginLibraryInfo)
tritonGetDialectPluginInfo(const char *name);

///
/// Internal APIs: for internal bookkeeping of what is exported above.
///
namespace triton::ext::plugin {

typedef void (*RegisterPassFunc)();
typedef void (*AddPassFunc)(mlir::PassManager *);
typedef void (*InsertDialect)(mlir::DialectRegistry *);

TritonPluginResult exportPass(const std::string passName,
                              RegisterPassFunc registerFunc,
                              AddPassFunc addFunc);

TritonPluginResult exportDialect(const std::string dialectName,
                                 InsertDialect insertFunc);

} // namespace triton::ext::plugin

#endif // TRITON_EXT_PASS_INFRA_H
