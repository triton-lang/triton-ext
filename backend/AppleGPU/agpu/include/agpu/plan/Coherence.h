// Coherence.h - which buffers need coherent accesses.
//
// A plain device access may be served from a cache that another threadgroup's
// store never invalidated. Marking everything coherent defeats caching for
// every buffer in the kernel.
#ifndef AGPU_COHERENCE_H
#define AGPU_COHERENCE_H

#include "agpu/msl/Containers.h"

#include <vector>

namespace agpu {

enum class AccessKind { Load, Store };

struct BufferAccess {
  int buffer = 0; // index of the kernel argument it traces back to
  AccessKind kind = AccessKind::Load;
  int loopDepth = 0; // 0 = outside any loop

  // A per-lane address into a tile.
  bool isTensor = false;
};

// What the IR says about a function's scalar device traffic.
struct CoherenceFacts {
  std::vector<BufferAccess> accesses;

  // A barrier whose address space covers device memory. Every buffer both
  // stored and loaded then becomes coherent.
  bool hasDeviceBarrier = false;
};

// Buffers that must be accessed coherently.
class CoherencePlan {
public:
  bool needsCoherent(int buffer) const {
    for (int b : coherent_)
      if (b == buffer)
        return true;
    return false;
  }
  const std::vector<int> &buffers() const { return coherent_; }
  bool any() const { return !coherent_.empty(); }

  friend CoherencePlan planCoherence(const CoherenceFacts &);

private:
  void add(int b) {
    if (!needsCoherent(b))
      coherent_.push_back(b);
  }

  // Every buffer some store publishes to some load, where `qualifies`
  // decides which pairs count.
  template <typename Pairs>
  void addPublished(const CoherenceFacts &f, const Pairs &qualifies) {
    for (const BufferAccess &store : f.accesses) {
      if (store.kind != AccessKind::Store)
        continue;
      for (const BufferAccess &load : f.accesses)
        if (load.kind == AccessKind::Load && load.buffer == store.buffer &&
            qualifies(store, load))
          add(store.buffer);
    }
  }

  std::vector<int> coherent_;
};

inline CoherencePlan planCoherence(const CoherenceFacts &f) {
  CoherencePlan p;

  // 1. A loop that both stores and loads one buffer through a scalar address.
  //    Tensor accesses do not qualify: each iteration addresses a different
  //    window and no lane re-reads what another wrote.
  p.addPublished(f, [](const BufferAccess &store, const BufferAccess &load) {
    return store.loopDepth > 0 && !store.isTensor && !load.isTensor &&
           load.loopDepth == store.loopDepth;
  });

  // 2. A device barrier. Tensor accesses do qualify here: a section guarded
  //    by a device barrier reads its shared array as a tile and those lanes
  //    must see another threadgroup's stores.
  if (f.hasDeviceBarrier)
    p.addPublished(
        f, [](const BufferAccess &, const BufferAccess &) { return true; });

  return p;
}

} // namespace agpu

#endif // AGPU_COHERENCE_H
