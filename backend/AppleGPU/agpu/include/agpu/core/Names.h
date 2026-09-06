// Names.h - the identifiers the kernel prologue declares.
#ifndef AGPU_NAMES_H
#define AGPU_NAMES_H

#include "agpu/core/Containers.h"

#include <cctype>
#include <string>
#include <string_view>

namespace agpu {

// No MSL keyword begins with this prefix.
inline constexpr std::string_view kKernelPrefix = "triton_";

// Idempotent, so an already-prefixed symbol is not prefixed twice.
inline msl::Str kernelSymbol(std::string_view name) {
  if (name.substr(0, kKernelPrefix.size()) == kKernelPrefix)
    return msl::Str(name);
  return msl::Str(kKernelPrefix) + msl::Str(name);
}

struct ThreadNames {
  msl::Str laneId = "lane";
  msl::Str warpId = "warp";
  msl::Str threadId = "tid";
  // threadgroup_position_in_grid. The `.x` is part of the name because the
  // grid is one-dimensional here.
  msl::Str blockId = "tgpos.x";
};
} // namespace agpu

#endif // AGPU_NAMES_H
