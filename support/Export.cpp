#include "Export.h"
#include "triton/Tools/PluginUtils.h" // For plugin:: callbacks and structs.
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-ext"

///
/// Internal API.
///
namespace triton::ext::support {

using namespace mlir::triton;

static std::unordered_map<std::string, std::pair<plugin::AddPassCallback,
                                                 plugin::RegisterPassCallback>>
    passMap;
static std::unordered_map<std::string, plugin::RegisterDialectCallback>
    dialectMap;

Result exportPass(const std::string passName,
                  plugin::RegisterPassCallback registerFunc,
                  plugin::AddPassCallback addFunc) {
  LLVM_DEBUG(llvm::dbgs() << "internally exporting pass: " << passName << "\n");
  passMap[passName] = {addFunc, registerFunc};
  return TP_SUCCESS;
}

Result exportDialect(const std::string dialectName,
                     plugin::RegisterDialectCallback insertFunc) {
  LLVM_DEBUG(llvm::dbgs() << "internally exporting dialect: " << dialectName
                          << "\n");
  dialectMap[dialectName] = insertFunc;
  return TP_SUCCESS;
}
} // namespace triton::ext::support

///
/// External API.
///
using namespace triton::ext::support;
using namespace mlir::triton;

// TODO: because TritonExtensionSupport is a separate library, it will not be
// built with TRITON_EXT_NAME or TRITON_EXT_CLASS defined. This means that we
// cannot statically generate the name here. (Register it with the internal
// API?)
static const char *PLUGIN_NAME = "triton-ext-todo";
static const char *VERSION = "0.1.0";

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  auto passes = new plugin::PassInfo[passMap.size()];
  size_t numPasses = 0;
  for (const auto &pair : passMap) {
    const std::string &passName = pair.first;
    auto registerFunc = pair.second.second;
    auto addFunc = pair.second.first;
    passes[numPasses++] =
        plugin::PassInfo{passName.c_str(), VERSION, addFunc, registerFunc};
  }

  auto dialects = new plugin::DialectInfo[dialectMap.size()];
  size_t numDialects = 0;
  for (const auto &pair : dialectMap) {
    const std::string &dialectName = pair.first;
    auto registerFunc = pair.second;
    dialects[numDialects++] =
        plugin::DialectInfo{dialectName.c_str(), VERSION, registerFunc};
  }

  auto info = new plugin::PluginInfo{TRITON_PLUGIN_API_VERSION,
                                     PLUGIN_NAME,
                                     VERSION,
                                     passes,
                                     numPasses,
                                     dialects,
                                     numDialects};
  return info;
}

#undef DEBUG_TYPE
