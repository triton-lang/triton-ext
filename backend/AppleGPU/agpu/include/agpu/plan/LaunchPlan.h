// LaunchPlan.h - what the launcher must guarantee for a kernel to finish.
//
// Metal has no cooperative launch and does not preempt a spinning
// threadgroup, so a kernel that waits on a non-resident threadgroup hangs
// until the watchdog kills the command buffer. The launcher rejects such a
// grid up front.
#ifndef AGPU_LAUNCH_PLAN_H
#define AGPU_LAUNCH_PLAN_H

namespace agpu {

// Whether the kernel makes progress only if the whole grid is resident.
enum class GridResidency {
  Independent, // any grid completes: threadgroups never wait on each other
  CoResident,  // the kernel waits for threadgroups it cannot cause to run
};

// What the caller observed in the IR. The verdict is drawn in residencyFor.
struct LaunchFacts {
  // A poll with no timeout: ends only when another threadgroup publishes.
  bool blockingPoll = false;

  // The two halves of a grid barrier. Both are required: a mutex also spins
  // on a device atomic in a loop but is safe at any grid size.
  bool atomicInLoop = false;
  bool readsGridExtent = false;
};

inline GridResidency residencyFor(const LaunchFacts &f) {
  if (f.blockingPoll || (f.atomicInLoop && f.readsGridExtent))
    return GridResidency::CoResident;
  return GridResidency::Independent;
}

// Module attribute the launcher reads the verdict off, via
// `module.get_int_attr`.
inline constexpr const char *kGridResidencyAttr = "applegpu.grid_coresident";

} // namespace agpu

#endif // AGPU_LAUNCH_PLAN_H
