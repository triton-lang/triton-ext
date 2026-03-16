#ifndef TRITON_EXT_PASS_INFRA_H
#define TRITON_EXT_PASS_INFRA_H

#include "StringMacros.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Tools/PluginUtils.h"
#include <string>

using namespace mlir::triton;

///
/// External APIs: entry points to load the plugin.
///
TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo();

TRITON_PLUGIN_API void tritonReleasePluginInfo(plugin::PluginInfo *info);

///
/// Internal APIs: for internal bookkeeping of what is exported above.
///
namespace triton::ext::support {

/// A result code for plugin bookkeeping.
enum Result {
  TP_SUCCESS = 0,
  TP_GENERIC_FAILURE = 1,
};

Result exportPass(const std::string passName,
                  plugin::RegisterPassCallback registerFunc,
                  plugin::AddPassCallback addFunc);

Result exportDialect(const std::string dialectName,
                     plugin::RegisterDialectCallback registerFunc);

} // namespace triton::ext::support

#endif // TRITON_EXT_PASS_INFRA_H
