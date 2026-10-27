#include "agpu/cost/Pipeline.h"
#include "harness.h"

using namespace agpu::cost;

int main() {
  CASE("num_stages is a ceiling, and no loop runs past the measured depth");
  {
    CHECK_EQ(pipelineStages(1, 16, 48), 1);
    CHECK_EQ(pipelineStages(2, 16, 48), 2);
    CHECK_EQ(pipelineStages(3, 16, 48), kMaxPipelineStages);
    CHECK_EQ(pipelineStages(5, 16, 48), kMaxPipelineStages);
  }

  CASE("a stage that would spill at the typical register count is not run");
  {
    const RegisterBand band;
    const int64_t headroom = kMaxRegsPerThread - band.typical;
    CHECK_EQ(pipelineStages(2, headroom, 0), 2);
    CHECK_EQ(pipelineStages(2, headroom + 1, 0), 1);
    CHECK_EQ(pipelineStages(4, headroom + 1, 0), 1);
  }

  CASE("what the loop certainly holds counts when it is above typical");
  {
    // An fp32 128x64 accumulator over 128 threads, and one 128x32 + 32x64
    // slice.
    const int64_t acc = 64, slice = 48;
    CHECK_EQ(pipelineStages(2, slice, acc + slice), 1);
    CHECK_EQ(pipelineStages(2, 16, 32 + 16), 2);
  }

  return ::agpu_test::report("Pipeline");
}
