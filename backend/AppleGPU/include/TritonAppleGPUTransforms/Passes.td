#ifndef TRITON_APPLE_GPU_TRANSFORMS_PASSES
#define TRITON_APPLE_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AccelerateAppleMatmul : Pass<"triton-apple-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "Replace blocked dot ops with Apple simdgroup MMA encoding";

  let description = [{
    Rewrites `tt.dot` to AppleMmaEncodingAttr so it lowers to
    `simdgroup_multiply_accumulate`.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::applegpu::TritonAppleGPUDialect"
  ];
}

def StoreShuffleLayout : Pass<"triton-apple-store-shuffle-layout", "mlir::ModuleOp"> {
  let summary = "Re-lay MMA epilogue stores into the simdgroup-shuffle layout";

  let description = [{
    The mma C result and the coalesced store layout use different warp tilings,
    so the `#mma -> #blocked` convert crosses warps and spills through
    threadgroup memory. This re-lays the store into the blocked layout that
    shares the mma warp tiling, making the convert a within-simdgroup shuffle.

    A store whose address math uses an op outside the relayable allowlist keeps
    the threadgroup convert.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::applegpu::TritonAppleGPUDialect"
  ];
}

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
