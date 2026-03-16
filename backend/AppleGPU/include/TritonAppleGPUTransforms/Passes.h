#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::applegpu {

// Rewrite tt.dot with BlockedEncoding → AppleMmaEncoding
std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass();

// Strip efficient_layout from large gather ops
std::unique_ptr<mlir::Pass> createSimplifyGatherLayoutPass();

} // namespace mlir::triton::applegpu

// Generated pass declarations
#include "TritonAppleGPUTransforms/Passes.h.inc"
