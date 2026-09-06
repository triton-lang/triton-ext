#ifndef TRITON_APPLE_GPU_TRANSFORMS_PASSES
#define TRITON_APPLE_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def EmitMSL : Pass<"triton-apple-emit-msl", "mlir::ModuleOp"> {
  let summary = "Emit Metal Shading Language from TTGIR";

  let description = [{
    Terminal codegen. Writes MSL text for the module to the path given in the
    pass options.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::applegpu::TritonAppleGPUDialect"
  ];
}

#endif // TRITON_APPLE_GPU_TRANSFORMS_PASSES
