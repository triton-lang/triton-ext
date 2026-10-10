// CodeSize.h - what the emitter spends code on: how large a function may grow
// before Metal's compiler fails, when K steps roll into a loop, how far a
// rolled loop unrolls again, and when a masked load earns a peeled fast path.
// Every code-size knob lives here.
#ifndef AGPU_COST_CODE_SIZE_H
#define AGPU_COST_CODE_SIZE_H

#include "agpu/msl/Analysis.h"

#include <cstdint>

namespace agpu::cost {

// Metal's compiler dies with std::bad_alloc under PromoteMemToReg/SROA on an
// oversized function. Not derived from anything; a guess biased toward rolling
// too early.
inline constexpr int64_t kDeclBudget = 10000;

// Below this many fragment declarations, rolling K frees too little of the
// budget to be worth a re-walk. Unmeasured.
inline constexpr int64_t kRollFragFloor = 1024;

// Past this many MMAs a function rolls its K steps whatever its size. On flex
// attention's backward (M1 Pro, 2026-09-24) rolling keeps 94 registers, two
// resident 256-thread threadgroups, where unrolled needs 120 and keeps one:
// 2.43 against 2.84-2.94 ms, and 2.19 against 2.75 once fragments load by lane.
// The residency is what it protects; the count is how it is spelled until the
// roll is judged in Occupancy.h.
inline constexpr int64_t kRollMmaFloor = 128;

// A rolled panel K loop unrolls until a trip holds about this many MMAs.
// Unmeasured.
inline constexpr int64_t kUnrollCount = 8;

// The fewest registers a masked load must fill to earn a peeled unconditional
// fast path beside its guarded one. Unmeasured.
inline constexpr int64_t kMaskFastPathMinRegs = 4;

inline bool withinDeclBudget(const msl::FuncSize &s) {
  return s.optimiserLoad() <= kDeclBudget;
}

// Over the budget, with enough fragments that rolling brings it back.
inline bool rollingFitsBudget(const msl::FuncSize &s) {
  return !withinDeclBudget(s) && s.fragDecls >= kRollFragFloor;
}

inline bool rollsK(const msl::FuncSize &s) {
  return rollingFitsBudget(s) || s.mma > kRollMmaFloor;
}

} // namespace agpu::cost

#endif // AGPU_COST_CODE_SIZE_H
