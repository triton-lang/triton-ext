// Occupancy.h - how many threadgroups a core keeps resident, and what moving
// the pool does to that. Every threadgroup-memory judgment in the planner is
// made here.
//
// A core keeps as many threadgroups as both its threadgroup memory and its
// register file hold. The pool is known when it is planned; the register count
// only once Metal compiles the kernel, so each judgment holds across a band of
// register counts.
#ifndef AGPU_COST_OCCUPANCY_H
#define AGPU_COST_OCCUPANCY_H

#include <algorithm>
#include <cstdint>

namespace agpu::cost {

// What a core hands out across concurrently resident threadgroups. Measured
// on M1 Pro: a 20480-byte pool keeps three resident, 20736 bytes two.
inline constexpr int64_t kTGCoreBudgetBytes = 61440;

// Threadgroups that stay resident when a pool of `bytes` is declared. A step
// function: 15 KB gives four, 22 KB gives two.
inline constexpr int64_t tgResidency(int64_t bytes) {
  return bytes > 0 ? kTGCoreBudgetBytes / bytes : kTGCoreBudgetBytes;
}

// The register file, in 32-bit registers per core, and the most one thread may
// hold. A pipeline's maxTotalThreadsPerThreadgroup is the threads the file
// holds at the kernel's register count, allocated four at a time, in steps of
// 64 threads. `test_occupancy` holds the M1 Pro kernels this was fitted to.
inline constexpr int64_t kRegisterFileWords = 53248;
inline constexpr int64_t kMaxRegsPerThread = 128;
inline constexpr int64_t kRegisterGranule = 4;
inline constexpr int64_t kThreadStep = 64;

inline constexpr int64_t threadsForRegs(int64_t regs) {
  const int64_t alloc =
      (regs + kRegisterGranule - 1) / kRegisterGranule * kRegisterGranule;
  return kRegisterFileWords / alloc / kThreadStep * kThreadStep;
}

// Widest launch Metal admits regardless of register appetite: a kernel holding
// every register it may compiles to this, and a wider launch is then rejected
// at dispatch as OutOfResources.
inline constexpr int64_t kAlwaysAdmittedThreads =
    threadsForRegs(kMaxRegsPerThread);

// Threadgroups the register file keeps at `regs` per thread. A launch wider
// than that is pinned, so the compiler fits one.
inline constexpr int64_t regResidency(int64_t regs, int64_t threadsPerTG) {
  return std::max<int64_t>(1, threadsForRegs(regs) / threadsPerTG);
}

inline constexpr int64_t residency(int64_t bytes, int64_t threadsPerTG,
                                   int64_t regs) {
  return std::min(tgResidency(bytes), regResidency(regs, threadsPerTG));
}

// The register counts a kernel may compile to. Nothing narrows it yet: source
// liveness misses the compiled count by about twelve registers across a
// 558-kernel corpus (M1 Pro, 2026-09-24), since Metal hoists fragment loads by
// its own heuristics. `typical` decides what the band leaves open: the median
// of the 483 corpus kernels that declare a pool.
struct RegisterBand {
  int64_t lo = 1;
  int64_t hi = kMaxRegsPerThread;
  int64_t typical = 67;
};

// Threadgroups every register count in the band keeps.
inline constexpr int64_t certainResidency(int64_t bytes, int64_t threadsPerTG,
                                          RegisterBand band = {}) {
  return residency(bytes, threadsPerTG, band.hi);
}

// Moving the pool from `from` to `to` bytes. Residency moves one way with the
// pool and moves most at the fewest registers, so a gain is certain once it
// holds at the band's top and ruled out once it fails at its bottom; between
// them the typical count decides.
inline constexpr bool gainsResidency(int64_t from, int64_t to,
                                     int64_t threadsPerTG,
                                     RegisterBand band = {}) {
  if (residency(to, threadsPerTG, band.hi) >
      residency(from, threadsPerTG, band.hi))
    return true;
  if (residency(to, threadsPerTG, band.lo) <=
      residency(from, threadsPerTG, band.lo))
    return false;
  return residency(to, threadsPerTG, band.typical) >
         residency(from, threadsPerTG, band.typical);
}

inline constexpr bool losesResidency(int64_t from, int64_t to,
                                     int64_t threadsPerTG,
                                     RegisterBand band = {}) {
  return gainsResidency(to, from, threadsPerTG, band);
}

// For a choice whose downside is a spill rather than a slower read.
inline constexpr bool certainlyGainsResidency(int64_t from, int64_t to,
                                              int64_t threadsPerTG,
                                              RegisterBand band = {}) {
  return certainResidency(to, threadsPerTG, band) >
         certainResidency(from, threadsPerTG, band);
}

} // namespace agpu::cost

#endif // AGPU_COST_OCCUPANCY_H
