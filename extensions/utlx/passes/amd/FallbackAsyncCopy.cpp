/// Lower async copies that cannot be direct-to-LDS through registers.
///
/// On gfx9 a ttg.async_copy_global_to_local becomes a direct-to-LDS load,
/// where every lane writes a contiguous chunk of at least 32 bits. This only
/// works if the elements a thread owns contiguously in its source layout are
/// also contiguous in the destination's shared layout. TLX kernels pin shared
/// layouts explicitly, e.g. a K-contiguous padded_shared buffer for the B
/// operand, and still accept either operand orientation. For a row-major B
/// the copy becomes a global->LDS transpose. Each lane then gets one 16-bit
/// element. No source layout can make that a legal direct-to-LDS write, so
/// CoalesceAsyncCopy gives up and the backend fails to lower the op.
///
/// Fork TLX has the same lowering restriction. This pass keeps such copies
/// correct instead. It replaces each copy whose src->shared contiguity, or
/// padded min interval, cannot reach the minimum width with a tt.load followed
/// by a ttg.local_store. That trades the async write for a synchronous one. The
/// copy's token is dropped from its commit/wait groups, which is safe on AMD:
/// group counts are recomputed from the async instructions actually emitted.
/// Copies that can be lowered directly are left untouched.
///
/// The surviving copies also get their vector width recorded as the op's
/// contiguity. TLX's buffer_load_to_local becomes an async copy of `ptr +
/// offsets`, and the multiple_of/max_contiguous hints sit on the offsets.
/// CanonicalizePointers later splits that addition and drops the hints, so by
/// the time the copy is lowered axis analysis can only prove a width of one,
/// which is not a legal direct-to-LDS load.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// Smallest direct-to-LDS load width on gfx9 (CDNA3 and CDNA4).
constexpr unsigned kMinDirectToLdsBits = 32;

// Returns true if no source-layout choice can lower `copy` direct-to-LDS.
// Mirrors the src->shared contiguity checks in canLoadDirectToLDS without
// depending on the AMD backend's private headers.
bool cannotLoadDirectToLds(ttg::AsyncCopyGlobalToLocalOp copy,
                           unsigned warpSize) {
  auto srcTy = cast<mlir::RankedTensorType>(copy.getSrc().getType());
  auto dstTy = copy.getResult().getType();
  unsigned elemBits = tt::getPointeeBitWidth(srcTy);
  if (elemBits == 0 || elemBits >= kMinDirectToLdsBits)
    return false;

  tt::LinearLayout shared;
  auto encoding = dstTy.getEncoding();
  if (auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(encoding)) {
    // Without LDS scatter, padding may only fall on warp boundaries.
    if (padded.getMinInterval() / warpSize * elemBits < kMinDirectToLdsBits)
      return true;
    shared = padded.getLinearComponent();
  } else if (auto swizzled =
                 dyn_cast<ttg::SwizzledSharedEncodingAttr>(encoding)) {
    // Swizzling is applied to the source pointers, so check the flat layout.
    auto shape = dstTy.getShape();
    if (shape.size() != static_cast<size_t>(srcTy.getRank()))
      return false;
    auto flat = ttg::SwizzledSharedEncodingAttr::get(
        copy.getContext(), swizzled.getVec(), 1, 1, swizzled.getOrder(),
        swizzled.getCGALayout());
    shared = ttg::toLinearLayout(shape, flat);
  } else {
    return false;
  }

  // This runs after CoalesceAsyncCopy, so the source layout is already the
  // best one it could find for the destination.
  auto srcToShared = ttg::toLinearLayout(srcTy).invertAndCompose(shared);
  unsigned contig = srcToShared.getNumConsecutiveInOut();
  return contig * elemBits < kMinDirectToLdsBits;
}

void lowerThroughRegisters(ttg::AsyncCopyGlobalToLocalOp copy) {
  mlir::OpBuilder b(copy);
  auto loc = copy.getLoc();
  auto srcTy = cast<mlir::RankedTensorType>(copy.getSrc().getType());
  mlir::Value mask = copy.getMask();
  mlir::Value other = copy.getOther();
  if (mask && !other) {
    // Masked direct-to-LDS lanes leave zeros behind; keep that behaviour.
    auto valueTy = srcTy.cloneWith(std::nullopt,
                                   copy.getResult().getType().getElementType());
    other = mlir::arith::ConstantOp::create(b, loc, valueTy,
                                            b.getZeroAttr(valueTy));
  }
  auto load =
      tt::LoadOp::create(b, loc, copy.getSrc(), mask, other, copy.getCache(),
                         copy.getEvict(), copy.getIsVolatile());
  ttg::LocalStoreOp::create(b, loc, load.getResult(), copy.getResult());

  // Detach the token. Commit groups and waits take variadic tokens, so the
  // copy just leaves its group; any other user gets an empty group.
  mlir::Value token = copy.getToken();
  for (mlir::OpOperand &use : llvm::make_early_inc_range(token.getUses())) {
    mlir::Operation *user = use.getOwner();
    if (auto commit = dyn_cast<ttg::AsyncCommitGroupOp>(user)) {
      commit.getInputTokensMutable().erase(use.getOperandNumber());
    } else if (auto wait = dyn_cast<ttg::AsyncWaitOp>(user)) {
      wait.getAsyncTokenMutable().erase(use.getOperandNumber());
    } else {
      b.setInsertionPoint(copy);
      use.set(ttg::AsyncCommitGroupOp::create(b, loc, mlir::ValueRange{}));
    }
  }
  copy.erase();
}

class UTLXFallbackAsyncCopyPass
    : public mlir::PassWrapper<UTLXFallbackAsyncCopyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXFallbackAsyncCopyPass)

  llvm::StringRef getArgument() const override {
    return "utlx-fallback-async-copy";
  }
  llvm::StringRef getDescription() const override {
    return "Lower async copies that cannot be direct-to-LDS through registers";
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    auto target = mod->getAttrOfType<mlir::StringAttr>(ttg::AttrTargetName);
    // gfx1250 supports LDS scatter and has none of these restrictions.
    if (!target || !target.getValue().starts_with("hip:gfx9"))
      return;
    unsigned warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    tt::ModuleAxisInfoAnalysis axisInfo(mod);
    llvm::SmallVector<ttg::AsyncCopyGlobalToLocalOp> copies;
    mod.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
      if (cannotLoadDirectToLds(copy, warpSize)) {
        copies.push_back(copy);
        return;
      }
      unsigned contig = axisInfo.getContiguity(copy.getSrc());
      if (mlir::Value mask = copy.getMask())
        contig = std::min(contig, axisInfo.getMaskAlignment(mask));
      if (contig > copy.getContiguity())
        copy.setContiguity(contig);
    });
    for (auto copy : copies)
      lowerThroughRegisters(copy);
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createFallbackAsyncCopyPass() {
  return std::make_unique<UTLXFallbackAsyncCopyPass>();
}

void registerFallbackAsyncCopyPass() {
  mlir::PassRegistration<UTLXFallbackAsyncCopyPass>();
}

} // namespace utlx
