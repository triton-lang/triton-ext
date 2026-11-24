// Rewrites tt.dot to AppleMmaEncoding. B stays BlockedEncoding; A takes the
// mma encoding when it comes from another dot, so a chain does not round-trip
// through blocked. The result converts back to the user's layout unless its
// only consumer is another dot.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "agpu/core/Units.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/SmallVector.h"

#include <tuple>

#define GEN_PASS_DEF_ACCELERATEAPPLEMATMUL
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::triton::applegpu;

namespace {
bool resultFeedsRowReduction(tt::DotOp dot) {
  SmallVector<Value, 4> work{dot.getResult()};
  SmallPtrSet<Operation *, 8> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *u : v.getUsers()) {
      if (!seen.insert(u).second)
        continue;
      if (auto red = dyn_cast<tt::ReduceOp>(u)) {
        auto srcTy = dyn_cast<RankedTensorType>(red.getOperands()[0].getType());
        if (srcTy && red.getAxis() == srcTy.getRank() - 1)
          return true;
        continue;
      }
      if (u->hasTrait<OpTrait::Elementwise>() ||
          isa<ttg::ConvertLayoutOp, tt::BroadcastOp, tt::ExpandDimsOp>(u))
        for (Value r : u->getResults())
          work.push_back(r);
    }
  }
  return false;
}

bool operandComesFromDot(tt::DotOp dot) {
  SmallVector<Value, 4> work{dot.getA()};
  SmallPtrSet<Operation *, 8> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !seen.insert(def).second)
      continue;
    if (isa<tt::DotOp>(def))
      return true;
    if (def->hasTrait<OpTrait::Elementwise>() ||
        isa<ttg::ConvertLayoutOp, tt::BroadcastOp, tt::ExpandDimsOp,
            tt::FpToFpOp>(def))
      for (Value o : def->getOperands())
        work.push_back(o);
  }
  return false;
}

// warpsPerCTA for a dot shape and warp count.
SmallVector<unsigned> warpsPerTileApple(int64_t M, int64_t N, int numWarps,
                                        bool preferRowOwnership = false) {
  unsigned tilesM = std::max<int64_t>(1, M / agpu::kSgFragDim);
  unsigned tilesN = std::max<int64_t>(1, N / agpu::kSgFragDim);

  // Compared lexicographically and maximized; fields are in priority order.
  struct Rank {
    unsigned warpsUsed = 0;
    unsigned ownsRows = 0;
    unsigned operandFrags = 0;
    unsigned ownershipSkew = 0;
    unsigned warpGridSquareness = 0;

    auto key() const {
      return std::tuple<unsigned, unsigned, int, int, unsigned>(
          warpsUsed, ownsRows, -static_cast<int>(operandFrags),
          -static_cast<int>(ownershipSkew), warpGridSquareness);
    }
    bool operator<(const Rank &o) const { return key() < o.key(); }
    bool operator<=(const Rank &o) const { return key() <= o.key(); }
  };

  unsigned bestM = 1, bestN = 1;
  Rank best{/*warpsUsed=*/1,
            /*ownsRows=*/preferRowOwnership ? 1u : 0u,
            /*operandFrags=*/tilesM + tilesN,
            /*ownershipSkew=*/std::max(tilesM, tilesN) -
                std::min(tilesM, tilesN),
            /*warpGridSquareness=*/1};

  for (unsigned wm = 1; wm <= tilesM; ++wm) {
    if (tilesM % wm != 0)
      continue;
    for (unsigned wn = 1; wn <= tilesN; ++wn) {
      if (tilesN % wn != 0)
        continue;
      unsigned product = wm * wn;
      if (product > (unsigned)numWarps)
        continue;

      unsigned ownM = tilesM / wm;
      unsigned ownN = tilesN / wn;
      Rank key{/*warpsUsed=*/product,
               /*ownsRows=*/(preferRowOwnership && wn == 1) ? 1u : 0u,
               /*operandFrags=*/ownM + ownN,
               /*ownershipSkew=*/std::max(ownM, ownN) - std::min(ownM, ownN),
               /*warpGridSquareness=*/std::min(wm, wn)};
      if (key <= best)
        continue;

      best = key;
      bestM = wm;
      bestN = wn;
    }
  }

  // Warps left out of the encoding make it invalid; grow the row axis to cover
  // all of them.
  while (bestM * bestN < (unsigned)numWarps)
    bestM *= 2;

  return {bestM, bestN};
}

bool isSupportedDotType(mlir::Type elemTy) {
  return elemTy.isF16() || elemTy.isBF16() || elemTy.isF32() ||
         elemTy.isInteger(8);
}

struct BlockedToAppleMma : public OpRewritePattern<tt::DotOp> {
  int numWarps;

  BlockedToAppleMma(MLIRContext *ctx, int numWarps, PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), numWarps(numWarps) {}

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    auto ctx = dot.getContext();
    auto cType = cast<RankedTensorType>(dot.getC().getType());
    auto aType = cast<RankedTensorType>(dot.getA().getType());

    if (isa<AppleMmaEncodingAttr>(cType.getEncoding()))
      return failure();

    if (!isSupportedDotType(aType.getElementType()))
      return failure();

    auto shape = cType.getShape();
    const int rank = (int)shape.size();
    if (rank != 2 && rank != 3)
      return failure();

    int64_t M = shape[rank - 2], N = shape[rank - 1];

    auto wpc = warpsPerTileApple(M, N, numWarps,
                                 resultFeedsRowReduction(dot) ||
                                     operandComesFromDot(dot));
    // Batch never splits the warp grid: every warp visits every slice.
    if (rank == 3)
      wpc.insert(wpc.begin(), 1);
    auto mmaEnc = AppleMmaEncodingAttr::get(ctx, wpc);

    auto newCType =
        RankedTensorType::get(shape, cType.getElementType(), mmaEnc);

    auto loc = dot.getLoc();

    auto stripDotOpEnc = [&](Value operand) -> Value {
      auto ty = cast<RankedTensorType>(operand.getType());
      if (auto dotEnc =
              dyn_cast<ttg::DotOperandEncodingAttr>(ty.getEncoding())) {
        auto parentTy = RankedTensorType::get(
            ty.getShape(), ty.getElementType(), dotEnc.getParent());
        if (auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>()) {
          if (cvt.getSrc().getType() == parentTy)
            return cvt.getSrc();
        }
        return ttg::ConvertLayoutOp::create(rewriter, loc, parentTy, operand);
      }
      return operand;
    };
    Value newA = stripDotOpEnc(dot.getA());
    Value newB = stripDotOpEnc(dot.getB());

    if (auto aTy = dyn_cast<RankedTensorType>(newA.getType())) {
      if (aTy.getEncoding() != mmaEnc && operandComesFromDot(dot)) {
        auto aMmaTy =
            RankedTensorType::get(aTy.getShape(), aTy.getElementType(), mmaEnc);
        newA = ttg::ConvertLayoutOp::create(rewriter, loc, aMmaTy, newA);
      }
    }

    Value newC = dot.getC();
    if (auto cvt = newC.getDefiningOp<ttg::ConvertLayoutOp>()) {
      if (cvt.getSrc().getType() == newCType)
        newC = cvt.getSrc();
      else
        newC =
            ttg::ConvertLayoutOp::create(rewriter, loc, newCType, dot.getC());
    } else {
      newC = ttg::ConvertLayoutOp::create(rewriter, loc, newCType, dot.getC());
    }

    auto newDot = tt::DotOp::create(rewriter, loc, newCType, newA, newB, newC,
                                    dot.getInputPrecisionAttr(),
                                    dot.getMaxNumImpreciseAccAttr());

    SmallVector<OpOperand *> mmaUses;
    SmallVector<OpOperand *> blockedUses;
    SmallVector<ttg::ConvertLayoutOp> passthroughCasts;
    for (OpOperand &use : dot->getUses()) {
      auto *owner = use.getOwner();
      if (isa<tt::DotOp>(owner) &&
          (use.getOperandNumber() == 2 || use.getOperandNumber() == 0)) {
        mmaUses.push_back(&use);
        continue;
      }
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(owner)) {
        if (cvt.getType() == newCType) {
          for (OpOperand &cvtUse : cvt->getUses())
            mmaUses.push_back(&cvtUse);
          passthroughCasts.push_back(cvt);
          continue;
        }
      }
      blockedUses.push_back(&use);
    }

    for (OpOperand *use : mmaUses)
      use->set(newDot.getResult());
    for (ttg::ConvertLayoutOp cvt : passthroughCasts)
      if (cvt->use_empty())
        rewriter.eraseOp(cvt);

    if (blockedUses.empty()) {
      rewriter.eraseOp(dot);
      return success();
    }

    auto result =
        ttg::ConvertLayoutOp::create(rewriter, loc, cType, newDot.getResult());
    for (OpOperand *use : blockedUses)
      use->set(result.getResult());

    rewriter.eraseOp(dot);
    return success();
  }
};

// BlockedToAppleMma will not revisit a dot whose C is already #mma, so a
// chained dot's A convert is dropped here instead.
struct DropMmaAConvert : public OpRewritePattern<tt::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    auto cTy = dyn_cast<RankedTensorType>(dot.getC().getType());
    if (!cTy || !isa<AppleMmaEncodingAttr>(cTy.getEncoding()))
      return failure();

    auto cvt = dot.getA().getDefiningOp<ttg::ConvertLayoutOp>();
    if (!cvt)
      return failure();
    auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
    if (!srcTy || !isa<AppleMmaEncodingAttr>(srcTy.getEncoding()) ||
        cast<AppleMmaEncodingAttr>(srcTy.getEncoding()).getWarpsPerCTA() !=
            cast<AppleMmaEncodingAttr>(cTy.getEncoding()).getWarpsPerCTA())
      return failure();

    rewriter.modifyOpInPlace(dot,
                             [&] { dot.getAMutable().assign(cvt.getSrc()); });
    return success();
  }
};

// Simdgroup matrices take f32/f16/bf16 only. f16 holds every e4m3/e5m2 value.
struct PromoteFp8DotOperands : public OpRewritePattern<tt::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    auto aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
    auto bTy = dyn_cast<RankedTensorType>(dot.getB().getType());
    if (!aTy || !bTy)
      return failure();
    bool aFp8 = isa<Float8E4M3FNType, Float8E5M2Type>(aTy.getElementType());
    bool bFp8 = isa<Float8E4M3FNType, Float8E5M2Type>(bTy.getElementType());
    if (!aFp8 && !bFp8)
      return failure();

    Type f16 = rewriter.getF16Type();
    auto widen = [&](Value v) -> Value {
      auto ty = cast<RankedTensorType>(v.getType());
      if (!isa<Float8E4M3FNType, Float8E5M2Type>(ty.getElementType()))
        return v;
      return tt::FpToFpOp::create(rewriter, dot.getLoc(),
                                  ty.cloneWith(std::nullopt, f16), v);
    };
    Value a = widen(dot.getA()), b = widen(dot.getB());
    rewriter.modifyOpInPlace(dot, [&] {
      dot.getAMutable().assign(a);
      dot.getBMutable().assign(b);
    });
    return success();
  }
};

struct AccelerateAppleMatmul
    : public ::impl::AccelerateAppleMatmulBase<AccelerateAppleMatmul> {

  void runOnOperation() override {
    auto mod = getOperation();

    int numWarps = ttg::lookupNumWarps(mod);

    RewritePatternSet patterns(&getContext());
    patterns.add<BlockedToAppleMma>(&getContext(), numWarps);
    patterns.add<DropMmaAConvert>(&getContext());
    patterns.add<ttg::DecomposeScaledBlocked>(&getContext(), 1);
    patterns.add<PromoteFp8DotOperands>(&getContext());

    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir::triton::applegpu {
std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass() {
  return ::createAccelerateAppleMatmul();
}
} // namespace mlir::triton::applegpu
