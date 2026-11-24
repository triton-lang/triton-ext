// Re-lays an MMA epilogue store into a blocked layout whose #mma -> #blocked
// convert is a within-warp shuffle.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "agpu/core/Units.h"
#include "mlir/IR/Builders.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::triton::applegpu;

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_STORESHUFFLELAYOUT
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

static bool cvtNeedsSharedMemoryForTypes(RankedTensorType srcTy,
                                         RankedTensorType dstTy) {
  if (cvtReordersRegisters(srcTy, dstTy))
    return false;
  auto layout = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  auto kRegister = StringAttr::get(ctx, lldim::Register);
  auto kLane = StringAttr::get(ctx, lldim::Lane);
  if (to_vector(layout.getOutDimNames()) ==
      SmallVector<StringAttr, 2>{kRegister, kLane}) {
    auto srcLayout =
        ttg::toLinearLayout(srcTy).removeZeroBasesAlongDim(kRegister);
    auto dstLayout =
        ttg::toLinearLayout(dstTy).removeZeroBasesAlongDim(kRegister);
    auto factors = getWarpLayoutConvertDecomposition(srcLayout, dstLayout, 32);
    if (factors.mixedTranspositions.size() < 2)
      return false;
  }
  return true;
}

static bool isRelayableLayoutParametricOp(Operation *op) {
  return isa<tt::MakeRangeOp, tt::SplatOp, tt::ExpandDimsOp, tt::BroadcastOp,
             tt::AddPtrOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
             arith::CmpIOp, arith::AndIOp, arith::OrIOp, arith::SelectOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::ConstantOp,
             ttg::ConvertLayoutOp>(op);
}

class StoreShuffleLayoutPass
    : public impl::StoreShuffleLayoutBase<StoreShuffleLayoutPass> {

  using RelayMemo = DenseMap<std::pair<Value, Attribute>, Value>;

  Value relayConvert(ttg::ConvertLayoutOp cv, Attribute wantEnc, OpBuilder &b,
                     RelayMemo &memo) {
    return relay(cv.getSrc(), wantEnc, b, memo);
  }

  Value relayConstant(arith::ConstantOp cst, RankedTensorType tt2,
                      Attribute wantEnc, OpBuilder &b) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat())
      return nullptr;
    auto newResTy =
        RankedTensorType::get(tt2.getShape(), tt2.getElementType(), wantEnc);
    auto newAttr =
        DenseElementsAttr::get(newResTy, dense.getSplatValue<Attribute>());
    return arith::ConstantOp::create(b, cst.getLoc(), newResTy, newAttr);
  }

  Value cloneRelayableOpWithEncoding(Operation *def, RankedTensorType tt2,
                                     Attribute wantEnc, OpBuilder &b,
                                     RelayMemo &memo) {
    assert(def->getNumResults() == 1 && def->getNumRegions() == 0 &&
           "relayable op must be single-result and region-free");

    bool hasTensorOperand = llvm::any_of(def->getOperands(), [](Value o) {
      return isa<RankedTensorType>(o.getType());
    });
    Attribute srcEnc;
    if (hasTensorOperand) {
      srcEnc = inferSrcEncoding(def, wantEnc);
      if (!srcEnc)
        return nullptr;
    }

    SmallVector<Value> newOperands;
    for (Value operand : def->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType())) {
        newOperands.push_back(operand);
        continue;
      }
      Value re = relay(operand, srcEnc, b, memo);
      if (!re)
        return nullptr;
      newOperands.push_back(re);
    }

    auto newResTy =
        RankedTensorType::get(tt2.getShape(), tt2.getElementType(), wantEnc);
    OperationState state(def->getLoc(), def->getName().getStringRef());
    state.addOperands(newOperands);
    state.addTypes({newResTy});
    state.addAttributes(def->getAttrs());
    Operation *cloned = b.create(state);
    return cloned->getResult(0);
  }

  Value relay(Value v, Attribute wantEnc, OpBuilder &b, RelayMemo &memo) {
    auto tt2 = dyn_cast<RankedTensorType>(v.getType());
    if (!tt2)
      return v;
    if (tt2.getEncoding() == wantEnc)
      return v;
    auto key = std::make_pair(v, wantEnc);
    if (auto it = memo.find(key); it != memo.end())
      return it->second;

    Operation *def = v.getDefiningOp();
    if (!def || !isRelayableLayoutParametricOp(def))
      return nullptr;

    Value res;
    if (auto cv = dyn_cast<ttg::ConvertLayoutOp>(def))
      res = relayConvert(cv, wantEnc, b, memo);
    else if (auto cst = dyn_cast<arith::ConstantOp>(def))
      res = relayConstant(cst, tt2, wantEnc, b);
    else
      res = cloneRelayableOpWithEncoding(def, tt2, wantEnc, b, memo);

    if (res)
      memo[key] = res;
    return res;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    SmallVector<tt::StoreOp> targets;
    mod.walk([&](tt::StoreOp store) { targets.push_back(store); });

    for (tt::StoreOp store : targets) {
      auto cvt = store.getValue().getDefiningOp<ttg::ConvertLayoutOp>();
      if (!cvt)
        continue;
      auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
      auto mmaEnc = dyn_cast<AppleMmaEncodingAttr>(srcTy.getEncoding());
      auto dstTy = cast<RankedTensorType>(cvt.getResult().getType());
      auto blkEnc = dyn_cast<ttg::BlockedEncodingAttr>(dstTy.getEncoding());
      if (!mmaEnc || !blkEnc)
        continue;
      if (srcTy.getRank() != 2)
        continue;
      auto shape = srcTy.getShape();
      if (shape[0] % agpu::kSgFragDim != 0 || shape[1] % agpu::kSgFragDim != 0)
        continue;

      // Same 8x8 cells as the mma layout, different lanes. Deriving these
      // from the mma bases would make the convert the identity.
      auto wpc = mmaEnc.getWarpsPerCTA();
      SmallVector<unsigned> spt{1, 2};
      SmallVector<unsigned> tpw{8, 4};
      SmallVector<unsigned> ord{1, 0};
      SmallVector<unsigned> warps(wpc.begin(), wpc.end());

      int numWarps = ttg::lookupNumWarps(mod);
      int curWarps = 1;
      for (unsigned w : warps)
        curWarps *= w;
      if (numWarps % curWarps == 0) {
        unsigned factor = numWarps / curWarps;
        if (factor > 1)
          warps[ord[0]] *= factor;
      }

      auto wEnc = ttg::BlockedEncodingAttr::get(ctx, spt, tpw, warps, ord,
                                                blkEnc.getCGALayout());
      auto wType = RankedTensorType::get(shape, dstTy.getElementType(), wEnc);

      if (cvtNeedsSharedMemoryForTypes(srcTy, wType))
        continue;

      OpBuilder b(store);
      DenseMap<std::pair<Value, Attribute>, Value> memo;

      Value newPtr = relay(store.getPtr(), wEnc, b, memo);
      if (!newPtr)
        continue;
      Value newMask;
      if (store.getMask()) {
        newMask = relay(store.getMask(), wEnc, b, memo);
        if (!newMask)
          continue;
      }

      auto newCvt =
          ttg::ConvertLayoutOp::create(b, cvt.getLoc(), wType, cvt.getSrc());
      store.getValueMutable().assign(newCvt.getResult());
      store.getPtrMutable().assign(newPtr);
      if (newMask)
        store.getMaskMutable().assign(newMask);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createStoreShuffleLayoutPass() {
  return std::make_unique<StoreShuffleLayoutPass>();
}

} // namespace mlir::triton::applegpu
