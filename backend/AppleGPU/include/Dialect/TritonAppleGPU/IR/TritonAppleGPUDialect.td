#ifndef TRITON_APPLE_GPU_DIALECT
#define TRITON_APPLE_GPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonAppleGPU_Dialect : Dialect {
  let name = "applegpu";
  let cppNamespace = "::mlir::triton::applegpu";

  let description = [{
    Apple-specific encodings at the TritonGPU abstraction level. The only one
    is AppleMmaEncodingAttr, for tensors distributed across
    simdgroup_multiply_accumulate tiles.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
}

#endif // TRITON_APPLE_GPU_DIALECT
