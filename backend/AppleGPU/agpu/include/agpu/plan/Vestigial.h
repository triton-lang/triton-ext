// Ops that produce no MSL because there is nothing left to produce.
//
// Each of these carried information for a pass that has already run, so they
// report `emitted()` and never reach the decline log.
//
// An op belongs here only if emitting nothing for it changes nothing. An op
// this backend does not cover yet must decline where it is dispatched.
#ifndef AGPU_VESTIGIAL_H
#define AGPU_VESTIGIAL_H

#include "agpu/core/Decline.h"

#include <cstddef>
#include <string_view>

namespace agpu {

inline constexpr std::string_view kVestigial[] = {
    "scf.yield",
    "scf.condition",
    "ttg.local_dealloc",
    // MLIR spells this one `llvm.intr.assume`.
    "llvm.intr.assume",
};

inline bool isVestigial(std::string_view op) {
  for (const std::string_view v : kVestigial)
    if (v == op)
      return true;
  return false;
}

// `notMine()` for anything else, so the dispatcher falls through to the next
// family.
inline Decision vestigialDecision(std::string_view op) {
  if (isVestigial(op))
    return Decision::emitted();
  return Decision::notMine();
}

inline constexpr std::size_t vestigialCount() {
  return sizeof(kVestigial) / sizeof(kVestigial[0]);
}

} // namespace agpu

#endif // AGPU_VESTIGIAL_H
