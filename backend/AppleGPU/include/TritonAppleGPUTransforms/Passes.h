#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::applegpu {

std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass();
std::unique_ptr<mlir::Pass> createStoreShuffleLayoutPass();

} // namespace mlir::triton::applegpu

#include "TritonAppleGPUTransforms/Passes.h.inc"
