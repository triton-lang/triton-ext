#ifndef TRITON_EXT_DIALECT_EXAMPLE_DIALECT
#define TRITON_EXT_DIALECT_EXAMPLE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // For `Pure` trait.
include "triton/Dialect/Triton/IR/TritonTypes.td" // For `TT_Tensor` type.
include "triton/Dialect/Triton/IR/TritonInterfaces.td" // For `TensorSizeTrait` and `VerifyTensorLayoutsTrait` traits.

def Example_Dialect : Dialect {
    let name = "example";
    let summary = "An example out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::triton::example";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

class Example_Op<string mnemonic, list<Trait> traits = [TensorSizeTrait, VerifyTensorLayoutsTrait]> :
    Op<Example_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def Example_Zero : Example_Op<"zero", [Pure]> {
    let summary = "Construct a zero-filled tensor of any type.";
    let description = [{
        The `example.zero` operation illustrates how to define a new operation
        in a dialect. Generating a zero-filled tensor of any type is not
        exciting, but illustrates how to create new dialects and operations.

        ```mlir
        %0 = example.zero -> tensor<4x4x8xf32>
        ```
    }];
    let arguments = (ins);
    let results = (outs TT_Tensor:$result);
    // let assemblyFormat = "`->` type($result)";
}

#endif // TRITON_EXT_DIALECT_EXAMPLE_DIALECT
