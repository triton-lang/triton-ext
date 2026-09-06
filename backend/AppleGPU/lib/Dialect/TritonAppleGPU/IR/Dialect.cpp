// TritonAppleGPU dialect registration

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/TypeSwitch.h"

// Pull in tablegen-generated definitions
#include "dialect/TritonAppleGPU/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.cpp.inc"

namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::triton;

namespace {

struct AppleGPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    resultEncoding = ttg::SliceEncodingAttr::get(
        getDialect()->getContext(), axis,
        cast<ttg::DistributedEncodingTrait>(operandEncoding));
    return success();
  }

  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    auto ll = ttg::toLinearLayout(shape, operandEncoding);
    auto transposedLl = transposeLinearLayout(ll, order);
    resultEncoding = ttg::LinearEncodingAttr::get(getDialect()->getContext(),
                                                  std::move(transposedLl));
    return success();
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> loc) const override {
    auto sliceEncoding =
        mlir::dyn_cast<ttg::SliceEncodingAttr>(operandEncoding);
    if (!sliceEncoding)
      return emitOptionalError(
          loc, "ExpandDimsOp operand encoding must be SliceEncodingAttr");
    if (sliceEncoding.getDim() != axis)
      return emitOptionalError(
          loc, "Incompatible slice dimension for ExpandDimsOp operand");
    resultEncoding = sliceEncoding.getParent();
    return success();
  }

  LogicalResult inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                                   Attribute retEncoding,
                                   std::optional<Location> loc) const override {
    if (auto dotOpEnc =
            mlir::dyn_cast<ttg::DotOperandEncodingAttr>(operandEncoding)) {
      if (opIdx != dotOpEnc.getOpIdx())
        return emitOptionalError(loc, "Wrong opIdx");
      if (retEncoding != dotOpEnc.getParent())
        return emitOptionalError(loc, "Incompatible parent encoding");
    }
    return success();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    auto aEnc = mlir::dyn_cast<ttg::DotOperandEncodingAttr>(operandEncodingA);
    auto bEnc = mlir::dyn_cast<ttg::DotOperandEncodingAttr>(operandEncodingB);
    if (!aEnc && !bEnc)
      return success();
    if (!aEnc || !bEnc)
      return op->emitError("mismatching encoding between A and B operands");
    if (aEnc.getKWidth() != bEnc.getKWidth())
      return op->emitError("mismatching kWidth between A and B operands");
    return success();
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         bool allowReorder,
                         std::optional<Location> loc) const override {
    if (!isa<ttg::DistributedEncodingTrait>(srcEnc))
      return emitOptionalError(loc,
                               "Failed MemDescReshapeOp encoding inference");
    auto ctx = srcEnc.getContext();
    auto i32Type = IntegerType::get(ctx, 32, IntegerType::Unsigned);
    auto srcTy = RankedTensorType::get(srcShape, i32Type, srcEnc);
    auto ll = ttg::inferReshapeLinearLayout(cast<ttg::TensorOrMemDesc>(srcTy),
                                            dstShape);
    dstEnc = ttg::LinearEncodingAttr::get(ctx, std::move(ll));
    return success();
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got, std::optional<Location> loc,
                        bool ignoreRegBroadcast = false) const override {
    if (expected == got)
      return success();
    if (!expected || !got)
      return failure();
    auto expectedLL =
        ttg::toLinearLayout(shape, cast<ttg::LayoutEncodingTrait>(expected));
    auto gotLL =
        ttg::toLinearLayout(shape, cast<ttg::LayoutEncodingTrait>(got));
    if (ignoreRegBroadcast) {
      auto kReg =
          StringAttr::get(expected.getContext(), applegpu::lldim::Register);
      expectedLL = expectedLL.removeZeroBasesAlongDim(kReg);
      gotLL = gotLL.removeZeroBasesAlongDim(kReg);
    }
    if (expectedLL != gotLL)
      return emitOptionalError(loc, "Expected result encoding ", expected,
                               " but was ", got);
    return success();
  }

  // These three refuse to derive an apple_mma result encoding: the emitter
  // handles tt.join, tt.split and ttg.fp4_to_fp under other encodings.
  // Upstream dispatches on the source encoding's dialect, so they fire only
  // on an apple_mma operand.
  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    return emitOptionalError(
        loc, "AppleMmaEncoding has no join result encoding to infer");
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    return emitOptionalError(
        loc, "AppleMmaEncoding has no split result encoding to infer");
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    return emitOptionalError(
        loc, "AppleMmaEncoding has no fp4_to_fp result encoding to infer");
  }
};

} // anonymous namespace

namespace mlir::triton::applegpu {

void TritonAppleGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.cpp.inc"
      >();
  addInterfaces<AppleGPUInferLayoutInterface>();
}

LogicalResult
AppleMmaEncodingAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                             llvm::ArrayRef<unsigned> warpsPerCTA) {
  if (warpsPerCTA.size() != 2 && warpsPerCTA.size() != 3)
    return emitError() << "AppleMmaEncoding requires 2 or 3 warpsPerCTA dims";
  for (unsigned w : warpsPerCTA)
    if (w == 0)
      return emitError() << "AppleMmaEncoding warpsPerCTA must be non-zero";
  // The batch never splits the warp grid: every warp visits every slice.
  if (warpsPerCTA.size() == 3 && warpsPerCTA[0] != 1)
    return emitError() << "AppleMmaEncoding batch warpsPerCTA dim must be 1";
  return success();
}

SmallVector<unsigned> AppleMmaEncodingAttr::getRepOrder() const {
  SmallVector<unsigned> order(getRank());
  for (unsigned i = 0; i < order.size(); ++i)
    order[i] = i;
  return order;
}

SmallVector<unsigned>
AppleMmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getRepOrder();
}

} // namespace mlir::triton::applegpu
