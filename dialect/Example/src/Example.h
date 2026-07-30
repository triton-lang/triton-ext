#ifndef TRITON_EXT_DIALECT_EXAMPLE_H
#define TRITON_EXT_DIALECT_EXAMPLE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Declare the dialect types.
#define GET_TYPEDEF_CLASSES
#include "ExampleTypes.h.inc"

// Declare the dialect itself.
#include "ExampleDialect.h.inc"

// Declare the dialect operations.
#define GET_OP_CLASSES
#include "Example.h.inc"

#endif // TRITON_EXT_DIALECT_EXAMPLE_H
