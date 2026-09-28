// Pipeline.h - how many stages a prefetched loop runs for the num_stages it
// was asked for.
//
// Loads run ahead into registers the loop carries. There is no asynchronous
// copy, so each stage past the first holds another slice of every prefetched
// tile for a whole trip. Resident threadgroups hide latency beyond one trip.
#ifndef AGPU_COST_PIPELINE_H
#define AGPU_COST_PIPELINE_H

#include "agpu/cost/Occupancy.h"

namespace agpu::cost {

// Measured across GEMM tiles and attention: a third stage never ran faster
// than the second (the workplan's num_stages record).
inline constexpr int64_t kMaxPipelineStages = 2;

// The stages to run, at most `requested`. `liveRegs` is what a thread
// certainly holds across a trip at one stage. A stage that would take that, or
// the band's typical count, past the per-thread maximum would spill.
inline constexpr int64_t pipelineStages(int64_t requested, int64_t regsPerStage,
                                        int64_t liveRegs,
                                        RegisterBand band = {}) {
  const int64_t base = std::max(band.typical, liveRegs);
  int64_t stages = std::min(requested, kMaxPipelineStages);
  while (stages > 1 && base + (stages - 1) * regsPerStage > kMaxRegsPerThread)
    --stages;
  return stages;
}

} // namespace agpu::cost

#endif // AGPU_COST_PIPELINE_H
