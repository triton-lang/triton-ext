// Names.h - the identifiers the kernel prologue declares.
#ifndef AGPU_NAMES_H
#define AGPU_NAMES_H

#include "agpu/core/Containers.h"

#include <cctype>
#include <string>
#include <string_view>

namespace agpu {

// No MSL keyword begins with either prefix.
inline constexpr std::string_view kKernelPrefix = "triton_";
inline constexpr std::string_view kDeviceFnPrefix = "fn_";

// Idempotent, so an already-prefixed symbol is not prefixed twice.
inline msl::Str kernelSymbol(std::string_view name) {
  if (name.substr(0, kKernelPrefix.size()) == kKernelPrefix)
    return msl::Str(name);
  return msl::Str(kKernelPrefix) + msl::Str(name);
}

// Distinct prefix: these share a namespace with the kernel and the launcher's
// entry-point search must not match one.
inline msl::Str deviceFnSymbol(std::string_view name) {
  msl::Str out{kDeviceFnPrefix};
  for (const char c : name)
    out += std::isalnum(static_cast<unsigned char>(c)) || c == '_' ? c : '_';
  return out;
}

struct ThreadNames {
  msl::Str laneId = "lane";
  msl::Str warpId = "warp";
  msl::Str threadId = "tid";
  // threadgroup_position_in_grid. The `.x` is part of the name because the
  // grid is one-dimensional here.
  msl::Str blockId = "tgpos.x";
};

// Names of pool regions. Metal admits threadgroup declarations only in a
// kernel, so the pool declares and the emitters only address.
struct ScratchNames : ThreadNames {
  msl::SmallVec<msl::Str, 4> scratch;
};

// Distinct prefixes so a reduce and a scan in one kernel take separate
// regions. A multi-operand combine takes one region per operand, hence `k`.
inline msl::Str reduceScratchKey(int k) {
  return msl::Str("rscr") + msl::Str(std::to_string(k));
}
inline msl::Str scanScratchKey(int k) {
  return msl::Str("sscr") + msl::Str(std::to_string(k));
}

// An elected atomic publishes its result here for the threads the election
// excluded, which need the value the winner read.
inline msl::Str atomicScratchKey() { return msl::Str("ascr"); }

// The per-warp staging array a direct dot writes its fragment through. Named
// after the fragment so two in one kernel do not collide.
inline msl::Str directScratchName(const msl::Str &frag) {
  return frag + msl::Str("scr");
}

} // namespace agpu

#endif // AGPU_NAMES_H
