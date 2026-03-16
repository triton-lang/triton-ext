// Strip efficient_layout from gather ops whose warp-local lowering would
// generate too many instructions for the Metal GPU JIT compiler.

#include "TritonAppleGPUTransforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_SIMPLIFYGATHERLAYOUT
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {
class SimplifyGatherLayoutPass
    : public impl::SimplifyGatherLayoutBase<SimplifyGatherLayoutPass> {
  void runOnOperation() override {
    getOperation().walk([](triton::GatherOp op) {
      if (!op.getEfficientLayout())
        return;

      auto srcTy = op.getSrc().getType();
      auto idxTy = op.getIndices().getType();
      auto srcEnc =
          dyn_cast<gpu::BlockedEncodingAttr>(srcTy.getEncoding());
      if (!srcEnc)
        return;

      unsigned axis = op.getAxis();
      unsigned rank = srcTy.getRank();
      auto spt = srcEnc.getSizePerThread();
      auto tpw = srcEnc.getThreadsPerWarp();
      auto wpc = srcEnc.getWarpsPerCTA();

      // Estimate registers per thread for src and idx tensors.
      unsigned srcRegs = 1;
      for (unsigned d = 0; d < rank; ++d)
        srcRegs *= spt[d];

      unsigned idxRegs = 1;
      for (unsigned d = 0; d < rank; ++d) {
        unsigned s = idxTy.getDimSize(d) / (tpw[d] * wpc[d]);
        idxRegs *= std::max(s, 1u);
      }

      // Warp-local gather emits idxRegs * spt[axis] shuffles + selects.
      // Convert_layouts emit ~10 instructions per element each.
      // 3 convert_layouts: src, idx, result.
      unsigned shuffleOps = idxRegs * spt[axis] * 3;
      unsigned convertOps = (srcRegs + idxRegs * 2) * 10;

      if (shuffleOps + convertOps > 1024)
        op.setEfficientLayout(false);
    });
  }
};
} // namespace

std::unique_ptr<Pass> createSimplifyGatherLayoutPass() {
  return std::make_unique<SimplifyGatherLayoutPass>();
}

} // namespace mlir::triton::applegpu
