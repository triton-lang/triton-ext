#include "Export.h"
#include "mlir/Tools/Plugins/DialectPlugin.h" // For mlir::DialectPluginLibraryInfo.
#include "triton/Tools/PluginUtils.h"         // For TritonPluginResult.
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "triton-ext"

///
/// Internal API.
///
namespace triton::ext::plugin {

static std::unordered_map<std::string, std::pair<AddPassFunc, RegisterPassFunc>>
    passMap;
static std::unordered_map<std::string, InsertDialect> dialectMap;

TritonPluginResult exportPass(const std::string passName,
                              RegisterPassFunc registerFunc,
                              AddPassFunc addFunc) {
  LLVM_DEBUG(llvm::dbgs() << "internally exporting pass: " << passName << "\n");
  passMap[passName] = {addFunc, registerFunc};
  return TP_SUCCESS;
}

TritonPluginResult exportDialect(const std::string dialectName,
                                 InsertDialect insertFunc) {
  LLVM_DEBUG(llvm::dbgs() << "internally exporting dialect: " << dialectName
                          << "\n");
  dialectMap[dialectName] = insertFunc;
  return TP_SUCCESS;
}
} // namespace triton::ext::plugin

///
/// External pass API.
///
using namespace triton::ext::plugin;

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames) {
  if (!passCount)
    return TP_GENERIC_FAILURE;
  auto count = passMap.size();
  *passCount = count;
  if (!passNames) {
    LLVM_DEBUG(llvm::dbgs() << "found " << count << " passes\n");
    return TP_SUCCESS;
  }
  unsigned i = 0;
  for (const auto &pair : passMap) {
    const char *passName = pair.first.c_str();
    LLVM_DEBUG(llvm::dbgs() << "found pass: " << passName << "\n");
    passNames[i++] = passName;
  }
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *passName) {
  LLVM_DEBUG(llvm::dbgs() << "adding plugin pass: " << passName << "\n");
  std::string passNameStr(passName);
  if (passMap.find(passNameStr) == passMap.end())
    return TP_GENERIC_FAILURE;
  passMap[passNameStr].first(pm);
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *passName) {
  LLVM_DEBUG(llvm::dbgs() << "registering plugin pass: " << passName << "\n");
  std::string passNameStr(passName);
  if (passMap.find(passNameStr) == passMap.end())
    return TP_GENERIC_FAILURE;
  passMap[passNameStr].second();
  return TP_SUCCESS;
}

///
/// External dialect API.
///
TRITON_PLUGIN_API
tritonEnumeratePluginDialects(uint32_t *outDialectCount,
                              const char **outDialectNames) {
  *outDialectCount = dialectMap.size();
  if (!outDialectNames) {
    LLVM_DEBUG(llvm::dbgs() << "found " << *outDialectCount << " dialects\n");
    return TP_SUCCESS;
  }
  unsigned i = 0;
  for (const auto &pair : dialectMap) {
    LLVM_DEBUG(llvm::dbgs() << "found dialect: " << pair.first << "\n");
    outDialectNames[i++] = pair.first.c_str();
  }
  return TP_SUCCESS;
}

TRITON_PLUGIN_API_TYPE(mlir::DialectPluginLibraryInfo)
tritonGetDialectPluginInfo(const char *name) {
  LLVM_DEBUG(llvm::dbgs() << "get plugin info for dialect: " << name << "\n");
  std::string nameStr(name);
  if (dialectMap.find(nameStr) == dialectMap.end())
    llvm::report_fatal_error(llvm::Twine("unknown dialect: ") + nameStr);
  InsertDialect insertFunc = dialectMap[nameStr];
  return {MLIR_PLUGIN_API_VERSION, name, LLVM_VERSION_STRING, insertFunc};
}

#undef DEBUG_TYPE
