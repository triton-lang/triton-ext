// Input dimension names used by TritonGPU's LinearLayout. A misspelled name
// reports a dimension with no bases.
#ifndef TRITON_APPLE_GPU_LINEAR_LAYOUT_DIMS_H
#define TRITON_APPLE_GPU_LINEAR_LAYOUT_DIMS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::applegpu::lldim {

inline constexpr llvm::StringRef Register = "register";
inline constexpr llvm::StringRef Lane = "lane";
inline constexpr llvm::StringRef Warp = "warp";
inline constexpr llvm::StringRef Block = "block";

} // namespace mlir::triton::applegpu::lldim

#endif // TRITON_APPLE_GPU_LINEAR_LAYOUT_DIMS_H
