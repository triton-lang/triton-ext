// A reduction to scalars reads its operands in the layout they arrive in. The
// result has no layout, so a convert_layout in front of it only moves data, and
// across warps that costs a threadgroup round trip.

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_REDUCETHROUGHLAYOUTCHANGE
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

struct ReduceThroughConvert : public OpRewritePattern<tt::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::ReduceOp red,
                                PatternRewriter &rewriter) const override {
    for (Value res : red.getResults())
      if (isa<RankedTensorType>(res.getType()))
        return failure();
    SmallVector<Value> srcs;
    Attribute encoding;
    for (Value operand : red.getSrcs()) {
      auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>();
      if (!cvt)
        return failure();
      Attribute enc = cvt.getSrc().getType().getEncoding();
      if (encoding && enc != encoding)
        return failure();
      encoding = enc;
      srcs.push_back(cvt.getSrc());
    }
    rewriter.modifyOpInPlace(red, [&] { red->setOperands(srcs); });
    return success();
  }
};

struct ReduceThroughLayoutChangePass
    : public impl::ReduceThroughLayoutChangeBase<
          ReduceThroughLayoutChangePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceThroughConvert>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createReduceThroughLayoutChangePass() {
  return std::make_unique<ReduceThroughLayoutChangePass>();
}

} // namespace mlir::triton::applegpu
