#ifndef TRITON_APPLE_GPU_TRANSFORMS_PASSES
#define TRITON_APPLE_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AccelerateAppleMatmul : Pass<"triton-apple-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "Replace blocked dot ops with Apple simdgroup MMA encoding";

  let description = [{
    Rewrites tt.dot operations to use AppleMmaEncodingAttr, enabling
    simdgroup_multiply_accumulate code generation on Apple GPUs.

    For each tt.dot with BlockedEncoding output:
      1. Select warpsPerCTA based on output shape and numWarps budget
      2. Create AppleMmaEncodingAttr with selected warpsPerCTA
      3. Insert ConvertLayoutOp on A, B, C operands
      4. Create new dot with AppleMma output encoding
      5. Convert output back to original encoding for downstream users
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::applegpu::TritonAppleGPUDialect"
  ];
}

def SimplifyGatherLayout : Pass<"triton-apple-simplify-gather", "mlir::ModuleOp"> {
  let summary = "Strip efficient_layout from large gather ops to avoid IR explosion";

  let description = [{
    The upstream OptimizeThreadLocality pass marks gather ops with
    efficient_layout, forcing warp-local shuffle lowering. For large tensors
    this generates thousands of unrolled shuffle+select instructions that
    exceed the Metal GPU JIT compiler's instruction limit. This pass clears
    the flag so the gather falls back to shared memory, and the subsequent
    RemoveLayoutConversions pass eliminates the now-unnecessary convert_layouts.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

#endif // TRITON_APPLE_GPU_TRANSFORMS_PASSES
