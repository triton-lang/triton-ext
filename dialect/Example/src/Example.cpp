#include "Example.h"

// Define the dialect types.
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "ExampleTypes.cpp.inc"

// Define the dialect itself; we need to define how it initializes and how types
// get registered.
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "ExampleDialect.cpp.inc"

namespace mlir::triton::example {

void ExampleDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ExampleTypes.cpp.inc"
      >();
}

void ExampleDialect::initialize() {
  {
    addOperations<
#define GET_OP_LIST
#include "Example.cpp.inc"
        >();
  }
  registerTypes();
}

} // namespace mlir::triton::example

// Define the dialect operations.
#define GET_OP_CLASSES
#include "Example.cpp.inc"

// Include the MLIR dialect plugin registry implementation.
using namespace mlir::triton::example;
#include "ExportDialect.cpp"
