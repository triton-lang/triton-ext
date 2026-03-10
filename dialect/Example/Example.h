#ifndef TRITON_EXT_DIALECT_EXAMPLE_H
#define TRITON_EXT_DIALECT_EXAMPLE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Example.h.inc"
#include "ExampleDialect.h.inc"
#include "ExampleTypes.h.inc"

#endif // TRITON_EXT_DIALECT_EXAMPLE_H
