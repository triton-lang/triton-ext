#ifndef TRITON_EXT_DIALECT_EXAMPLE_DIALECT
#define TRITON_EXT_DIALECT_EXAMPLE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // For `Pure` trait.

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
// RankedTensorOf<[F32]>
def Example_Tile : TypeDef<Example_Dialect, "Tile"> {
    let mnemonic = "tile";
    let summary = "An example custom type: a tensor tile.";
    let description = [{
        TODO
    }];
}

class Example_Op<string mnemonic, list<Trait> traits = []> :
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
        %0 = example.zero -> example.tile
        ```
    }];
    let arguments = (ins);
    let results = (outs Example_Tile:$result);
    let assemblyFormat = "attr-dict `->` type($result)";
}

#endif // TRITON_EXT_DIALECT_EXAMPLE_DIALECT
