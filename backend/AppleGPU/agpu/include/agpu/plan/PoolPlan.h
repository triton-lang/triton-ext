// PoolPlan.h - what a whole function needs from threadgroup memory.
//
// The pool is one buffer shared in time by operations that do not overlap,
// so the requirement is the largest single ask. Requests recorded here are
// the asks known before kernel bodies are built (device functions, live
// buffers, direct callers); a kernel's own body adds what it actually used.
//
// A threadgroup declaration over the hardware budget compiles and links
// cleanly, then takes down MTLCompilerService at pipeline-state creation
// with XPC_ERROR_CONNECTION_INTERRUPTED.
#ifndef AGPU_POOL_PLAN_H
#define AGPU_POOL_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/core/Units.h"
#include "agpu/plan/BandPlan.h"

#include <cstdint>
#include <vector>

namespace agpu {

// `label` names the operation, for the over-budget diagnostic.
struct PoolRequest {
  const char *label = "";
  Bytes bytes;
};

// What a function needs and whether it can have it.
struct FunctionPool {
  // The largest single request.
  Bytes scratch;

  // Threadgroup memory declared outside the pool, live across the operations
  // that use it.
  Bytes live;

  const char *driver = "";

  Bytes total() const { return scratch + live; }

  Capacity capacityFor(Bytes budget) const { return Capacity(budget, live); }
};

// A max: summing would reserve for every operation at once.
inline FunctionPool planFunctionPool(const std::vector<PoolRequest> &requests,
                                     Bytes liveBuffers = Bytes(0)) {
  FunctionPool p;
  p.live = liveBuffers;
  for (const PoolRequest &r : requests)
    if (r.bytes > p.scratch) {
      p.scratch = r.bytes;
      p.driver = r.label;
    }
  return p;
}

// Tested against the hardware limit. Occupancy is the residency question
// (`tgResidency`).
inline Decision
poolDecision(const FunctionPool &p,
             Bytes hardwareLimit = Bytes(kTGResidentBudgetBytes)) {
  if (p.total() <= hardwareLimit)
    return Decision::emitted();
  return Decision::declined("pool",
                            "threadgroup memory past the hardware limit");
}

// What an over-budget pool asked for and what it was allowed. The host reads
// these off the module and raises `OutOfResources`, which an autotuner catches
// to prune a config.
inline constexpr const char *kPoolNeededAttr = "applegpu.pool_needed_bytes";
inline constexpr const char *kPoolLimitAttr = "applegpu.pool_limit_bytes";

// A module's pool is the largest any of its functions needs. MSL forbids
// declaring threadgroup memory outside a kernel, so a kernel calling device
// functions declares one buffer for all of them.
inline FunctionPool planModulePool(const std::vector<FunctionPool> &functions) {
  FunctionPool p;
  for (const FunctionPool &f : functions) {
    if (f.scratch > p.scratch) {
      p.scratch = f.scratch;
      p.driver = f.driver;
    }
    if (f.live > p.live)
      p.live = f.live;
  }
  return p;
}

// What the operations walked so far have asked for, accumulated.
class PoolRequests {
public:
  // Scratch regions are reused, so requests are maxed together. `label`
  // names the requester for diagnostics only; it plays no role as a key, so
  // two requests under one label coexist without colliding.
  void scratch(const char *label, Bytes bytes) {
    requests_.push_back({label, bytes});
  }

  // Declared outside the pool (`ttg.local_alloc`). Live buffers coexist with
  // each other and with scratch, so these sum.
  void live(Bytes bytes) { live_ = live_ + bytes; }

  // Answerable before the max is settled. A device function's signature needs
  // it, since a function in a module with a pool takes the pool pointer.
  bool anyScratch() const { return !requests_.empty(); }

  FunctionPool plan() const { return planFunctionPool(requests_, live_); }

private:
  std::vector<PoolRequest> requests_;
  Bytes live_{0};
};

} // namespace agpu

#endif // AGPU_POOL_PLAN_H
