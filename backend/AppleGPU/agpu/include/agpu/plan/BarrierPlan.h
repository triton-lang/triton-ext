// BarrierPlan.h - what a barrier synchronises and what it does not.
//
// `threadgroup_barrier` orders memory only within the threadgroup, even with
// `mem_device` set. Spanning threadgroups needs a device fence as well.
#ifndef AGPU_BARRIER_PLAN_H
#define AGPU_BARRIER_PLAN_H

#include "agpu/msl/Ast.h"

#include <cstdint>

namespace agpu {

// Mirrors `ttg::AddrSpace`, which this layer cannot see. The bridge passes
// its bits through unchecked.
enum class BarrierSpace : uint32_t {
  Local = 1,
  GlobalRead = 2,
  GlobalWrite = 4,
};

struct BarrierPlan {
  msl::Barrier::Scope scope = msl::Barrier::Scope::Threadgroup;
  bool needsDeviceFence = false;
};

inline BarrierPlan planBarrier(uint32_t spaces) {
  const uint32_t device =
      (uint32_t)BarrierSpace::GlobalRead | (uint32_t)BarrierSpace::GlobalWrite;
  BarrierPlan p;
  if ((spaces & device) == 0)
    return p;

  p.scope = msl::Barrier::Scope::Device;
  p.needsDeviceFence = true;
  return p;
}

} // namespace agpu

#endif // AGPU_BARRIER_PLAN_H
