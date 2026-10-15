/// OptimizeDescriptorEncoding: upstream's descriptor layout assignment, plus
/// the layouts TMA copies need.
///
/// Upstream's ttng-optimize-descriptor-encoding derives a descriptor's shared
/// layout from tt.descriptor_load / tt.descriptor_store only. Every other use,
/// including the ttng.async_tma_copy_* ops tlx.async_descriptor_load / _store
/// build, gets the default layout for the block shape. That breaks a kernel
/// that copies through an explicit NVMMA view, e.g. a 128x64 f16 tile laid out
/// as two 64B-swizzled 32-column atoms: the descriptor comes back with the
/// default 128B swizzle and the copy fails verification. Meta's fork counts
/// those copies as uses in its own copy of the pass; this does the same after
/// running upstream's assignment, within one pass so the IR verifies.
///
/// The TensorMap is built from the descriptor's final layout (see
/// get_tensordesc_metadata), and the TMA lowering issues one copy per box, so
/// a narrower swizzle needs nothing else.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

// Upstream's NvidiaGPUAssignDescriptorMemoryLayouts, which no header declares.
// Copied from lib/Dialect/TritonNvidiaGPU/Transforms/
// OptimizeDescriptorEncoding.cpp.
class AssignDescriptorMemoryLayouts
    : public ttg::AssignDescriptorMemoryLayouts {
private:
  mlir::Attribute buildFallbackSharedEncoding(mlir::MLIRContext *ctx,
                                              llvm::ArrayRef<int64_t> shape,
                                              llvm::ArrayRef<unsigned> order,
                                              ttg::CGAEncodingAttr cgaLayout,
                                              mlir::Type elementType) override {
    return ttg::NVMMASharedEncodingAttr::get(ctx, shape, order, cgaLayout,
                                             elementType, /*fp4Padded*/ false);
  }

  mlir::Attribute getCompatibleSharedEncoding(mlir::Attribute enc,
                                              llvm::ArrayRef<int64_t> shape,
                                              mlir::Type elementType) override {
    if (isCompatibleSharedEncoding(enc))
      return enc;

    auto sharedLinear = mlir::dyn_cast<ttg::SharedLinearEncodingAttr>(enc);
    if (!sharedLinear)
      return {};

    auto *ctx = enc.getContext();
    auto cgaLayout = ttg::getCGALayout(sharedLinear);
    auto order = ttg::getOrder(sharedLinear, shape);
    auto sharedLinearLayout = ttg::toLinearLayout(shape, sharedLinear);
    auto isEquivalent = [&](ttg::NVMMASharedEncodingAttr candidate) {
      auto candidateLayout = ttg::nvmmaSharedToLinearLayout(
          shape, candidate, ttg::TMAMode::Tiled, /*disableSwizzle=*/false,
          /*emitErrors=*/false);
      return mlir::succeeded(candidateLayout) &&
             *candidateLayout == sharedLinearLayout;
    };

    llvm::SmallVector<ttg::NVMMASharedEncodingAttr> preferredCandidates;
    for (bool fp4Padded : {false, true}) {
      auto preferred = ttg::NVMMASharedEncodingAttr::get(
          ctx, shape, order, cgaLayout, elementType, fp4Padded);
      preferredCandidates.push_back(preferred);
      if (isEquivalent(preferred))
        return preferred;
    }

    unsigned elementBitWidth =
        std::max(8u, elementType.getIntOrFloatBitWidth());
    for (bool fp4Padded : {false, true}) {
      for (unsigned swizzle : {0u, 32u, 64u, 128u}) {
        auto candidate = ttg::NVMMASharedEncodingAttr::get(
            ctx, swizzle, /*transposed=*/false, elementBitWidth, fp4Padded,
            cgaLayout);
        if (llvm::is_contained(preferredCandidates, candidate))
          continue;
        if (isEquivalent(candidate))
          return candidate;
      }
    }
    return {};
  }

  bool isCompatibleSharedEncoding(mlir::Attribute enc) override {
    if (auto nvmma = mlir::dyn_cast<ttg::NVMMASharedEncodingAttr>(enc))
      return !nvmma.getTransposed();
    return false;
  }
};

// The fields upstream's TMA verifier compares; the encodings may differ in
// rank for rank-reducing copies.
bool sameTMAEncoding(ttg::NVMMASharedEncodingAttr a,
                     ttg::NVMMASharedEncodingAttr b) {
  return a.getTransposed() == b.getTransposed() &&
         a.getSwizzlingByteWidth() == b.getSwizzlingByteWidth() &&
         a.getElementBitWidth() == b.getElementBitWidth() &&
         a.getFp4Padded() == b.getFp4Padded();
}

class UTLXOptimizeDescriptorEncodingPass
    : public mlir::PassWrapper<UTLXOptimizeDescriptorEncodingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      UTLXOptimizeDescriptorEncodingPass)

  llvm::StringRef getArgument() const override {
    return "utlx-optimize-descriptor-encoding";
  }
  llvm::StringRef getDescription() const override {
    return "Assign tensor descriptor layouts, taking the shared layout TMA "
           "copies go through into account";
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    AssignDescriptorMemoryLayouts assign;
    assign.assignMemoryLayouts(mod);

    mod.walk([](mlir::Operation *op) {
      if (auto load = mlir::dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op))
        utlx::matchTensorDescToSharedLayout(
            load.getDesc(), load.getResult().getType().getEncoding());
      else if (auto store =
                   mlir::dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
        utlx::matchTensorDescToSharedLayout(
            store.getDesc(), store.getSrc().getType().getEncoding());
      else if (auto reduce = mlir::dyn_cast<ttng::AsyncTMAReduceOp>(op))
        utlx::matchTensorDescToSharedLayout(
            reduce.getDesc(), reduce.getSrc().getType().getEncoding());
    });
  }
};

} // namespace

namespace utlx {

bool matchTensorDescToSharedLayout(mlir::Value desc,
                                   mlir::Attribute sharedEncoding) {
  auto descTy = mlir::dyn_cast<tt::TensorDescType>(desc.getType());
  auto nvmma =
      mlir::dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(sharedEncoding);
  if (!descTy || !nvmma || nvmma.getTransposed())
    return false;
  if (auto current = mlir::dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(
          descTy.getSharedLayout());
      current && sameTMAEncoding(current, nvmma))
    return true;
  if (nvmma.getRank() != descTy.getShape().size())
    return false;

  // Only a kernel argument, i.e. a host-built descriptor whose TensorMap the
  // launcher fills in from this layout. A descriptor built on the device,
  // carried through a loop or passed to a device function keeps its layout.
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(desc);
  if (!arg || !arg.getOwner()->isEntryBlock())
    return false;
  auto func = mlir::dyn_cast<tt::FuncOp>(arg.getOwner()->getParentOp());
  if (!func || !tt::isKernel(func))
    return false;

  arg.setType(tt::TensorDescType::get(descTy.getShape(),
                                      descTy.getElementType(), nvmma));
  func.setFunctionType(mlir::FunctionType::get(
      func.getContext(), arg.getOwner()->getArgumentTypes(),
      func.getResultTypes()));
  return true;
}

std::unique_ptr<mlir::Pass> createOptimizeDescriptorEncodingPass() {
  return std::make_unique<UTLXOptimizeDescriptorEncodingPass>();
}

void registerOptimizeDescriptorEncodingPass() {
  mlir::PassRegistration<UTLXOptimizeDescriptorEncodingPass>();
}

} // namespace utlx
