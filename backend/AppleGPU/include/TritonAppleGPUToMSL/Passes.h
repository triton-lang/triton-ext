#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir::triton::applegpu {

std::unique_ptr<mlir::Pass> createEmitMSLPass(std::string outPath = {});

} // namespace mlir::triton::applegpu
