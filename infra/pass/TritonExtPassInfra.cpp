#include "TritonExtPassInfra.h"

static std::unordered_map<std::string, TritonExtAddPassFunc> passMap;
static std::unordered_map<std::string, TritonExtRegisterPassFunc> registryMap;
static std::vector<const char *> passNamesTable;

TRITON_PLUGIN_API
TritonExtLoadPass(const char *passName, TritonExtRegisterPassFunc registerFunc,
                  TritonExtAddPassFunc addFunc) {
  registryMap[passName] = registerFunc;
  passMap[passName] = addFunc;
  passNamesTable.push_back(passName);
  return TP_SUCCESS;
}

///
/// Plugin registration API
///
TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *passName) {
  std::string passNameStr(passName);
  if (passMap.find(passNameStr) == passMap.end())
    return TP_GENERIC_FAILURE;
  passMap[passNameStr](pm);
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *passName) {
  std::string passNameStr(passName);
  if (registryMap.find(passNameStr) == registryMap.end())
    return TP_GENERIC_FAILURE;
  registryMap[passNameStr]();
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames) {
  if (!passCount)
    return TP_GENERIC_FAILURE;
  auto count = passMap.size();
  assert(count == registryMap.size() &&
         "Expected register and add passes map size to match");
  *passCount = count;
  if (!passNames)
    return TP_SUCCESS;
  unsigned i = 0;
  for (auto passName : passNamesTable) {
    passNames[i] = passName;
  }
  return TP_SUCCESS;
}
