// HistogramPlan.h - counting into threadgroup bins, decided.
#ifndef AGPU_HISTOGRAM_PLAN_H
#define AGPU_HISTOGRAM_PLAN_H

#include "agpu/core/Units.h"
#include "agpu/plan/AtomicPlan.h"

namespace agpu {

// Bins live in threadgroup memory and every thread increments atomically.
// The zeroing loop strides by thread count since there may be more bins than
// threads.
struct HistogramPlan {
  int64_t bins = 0;
  int64_t threads = kWarpSize;
  ThreadElection election; // which threads own a source element

  int64_t zeroSteps() const {
    return threads > 0 ? (bins + threads - 1) / threads : bins;
  }
};

inline HistogramPlan planHistogram(int64_t bins, int64_t numWarps,
                                   unsigned laneFree, unsigned warpFree) {
  HistogramPlan p;
  p.bins = bins;
  p.threads = threadsFor(numWarps);

  // Same free-variable election atomics use: threads differing only in bits
  // that don't move the source index hold the same element.
  AtomicFacts f;
  f.laneFree = laneFree;
  f.warpFree = warpFree;
  p.election = electFor(f);
  return p;
}

} // namespace agpu

#endif // AGPU_HISTOGRAM_PLAN_H
