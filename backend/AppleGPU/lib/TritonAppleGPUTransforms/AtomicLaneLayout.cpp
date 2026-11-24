// Re-lays tensor atomics so each lane owns one element of the fastest axis.
// Metal has no vector atomics, so coalesce's vector-width layout makes one
// atomic instruction stride across several cache lines instead of covering one.

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_ATOMICLANELAYOUT
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

class AtomicLaneLayoutPass
    : public impl::AtomicLaneLayoutBase<AtomicLaneLayoutPass> {
public:
  void runOnOperation() override {
    SmallVector<Operation *> atomics;
    getOperation().walk([&](Operation *op) {
      if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op))
        atomics.push_back(op);
    });
    for (Operation *op : atomics)
      relay(op);
  }

private:
  void relay(Operation *op) {
    auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty)
      return;
    auto blk = dyn_cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
    if (!blk || blk.getSizePerThread()[blk.getOrder()[0]] == 1)
      return;
    SmallVector<unsigned> ones(ty.getRank(), 1);
    auto enc = ttg::BlockedEncodingAttr::get(
        op->getContext(), ty.getShape(), ones, blk.getOrder(),
        ttg::lookupNumWarps(op), 32, blk.getCGALayout());

    OpBuilder b(op);
    for (OpOperand &operand : op->getOpOperands()) {
      auto oty = dyn_cast<RankedTensorType>(operand.get().getType());
      if (!oty || oty.getEncoding() != blk)
        continue;
      operand.set(ttg::ConvertLayoutOp::create(
          b, op->getLoc(), oty.cloneWithEncoding(enc), operand.get()));
    }
    Value res = op->getResult(0);
    res.setType(ty.cloneWithEncoding(enc));
    b.setInsertionPointAfter(op);
    auto back = ttg::ConvertLayoutOp::create(b, op->getLoc(), ty, res);
    res.replaceAllUsesExcept(back, back);
  }
};

} // namespace

std::unique_ptr<Pass> createAtomicLaneLayoutPass() {
  return std::make_unique<AtomicLaneLayoutPass>();
}

} // namespace mlir::triton::applegpu
