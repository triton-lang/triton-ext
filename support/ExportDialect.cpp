/// Export a pass using the Triton plugin API. This file is meant to be compiled
/// in with the plugin implementation, exposing the necessary functions to
/// register the plugin with Triton.

#include "Export.h"
#include "StringMacros.h"
#include "mlir/IR/DialectRegistry.h"

#define TRITON_EXT_DIALECT_INSERT_FUNC                                         \
  CONCAT(registerTritonExt, TRITON_EXT_CLASS)
static void TRITON_EXT_DIALECT_INSERT_FUNC(mlir::DialectRegistry *registry) {
  // TODO: add logging here for easier debugging.
  registry->insert<TRITON_EXT_CLASS>();
}

using namespace ::triton::ext;
static support::Result initPlugin = support::exportDialect(
    TOSTRING(TRITON_EXT_NAME), TRITON_EXT_DIALECT_INSERT_FUNC);
