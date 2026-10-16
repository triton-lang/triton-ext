/// Vectorize global loads whose base is not aligned to the vector width.
///
/// gfx942 and gfx950 implement byte-aligned global vector accesses in
/// hardware, and LLVM models this through FeatureUnalignedBufferAccess.
/// Triton, however, limits a load's vector width by the alignment axis
/// analysis can prove for its base pointer, both when Coalesce picks the
/// load's layout and when ConvertToBufferOps records the buffer_load's
/// contiguity. A matmul with an odd K row stride, for example, only proves
/// 4-byte row alignment for f16 and gets 32-bit loads.
///
/// Fork Triton relaxes that clamp inside Coalesce and ConvertToBufferOps
/// (canUseUnalignedVectorizedLoad). Stock Triton has neither change, so uTLX
/// adds two passes that run directly after the stock ones:
///
/// - utlx-widen-unaligned-loads, after Coalesce, widens the coalesced layout
///   of each eligible tt.load to the width its contiguity allows.
/// - utlx-widen-unaligned-buffer-loads, after ConvertToBufferOps, raises each
///   amdg.buffer_load's contiguity, which the AMD lowering uses as the
///   minimum vector width.
///
/// Only the base-alignment clamp is relaxed. Axis analysis must still prove
/// the elements contiguous, the logical extent and 128-bit width caps remain,
/// and a mask limits the width to groups that share one predicate. Both passes
/// only ever widen, so anything they do not recognize keeps the stock result.
///
/// Stores, atomics, volatile loads and direct-to-LDS copies are left alone,
/// as in the fork. One difference: stock ConvertToBufferOps drops a load's
/// volatile flag, so the buffer_load pass cannot exclude volatile loads.

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/MathExtras.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// Widest global vector access, in bits.
constexpr unsigned kMaxVecBits = 128;

constexpr llvm::StringLiteral kKeepLayoutAttr = "utlx.keep_layout";

// Whether the module targets an architecture validated for unaligned vector
// loads. Future targets must opt in explicitly.
bool targetSupportsUnalignedVectorLoads(mlir::ModuleOp mod) {
  auto target = mod->getAttrOfType<mlir::StringAttr>(ttg::AttrTargetName);
  if (!target || !target.getValue().starts_with("hip:"))
    return false;
  // Ignore target feature suffixes, e.g. hip:gfx942:sramecc+:xnack-.
  llvm::StringRef processor = target.getValue().drop_front(4).split(':').first;
  return processor == "gfx942" || processor == "gfx950";
}

// The fork leaves loads that feed a dot operand layout at the stock width.
// After Coalesce a load's users are the converts back to its old layout, so
// look through those too.
bool hasDotOperandLayoutUser(mlir::Operation *op) {
  llvm::SmallVector<mlir::Operation *> worklist(op->getUsers());
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (!worklist.empty()) {
    mlir::Operation *user = worklist.pop_back_val();
    if (!visited.insert(user).second)
      continue;
    for (mlir::Value result : user->getResults()) {
      auto resultTy = dyn_cast<mlir::RankedTensorType>(result.getType());
      if (resultTy && isa<ttg::DotOperandEncodingAttr>(resultTy.getEncoding()))
        return true;
    }
    if (isa<ttg::ConvertLayoutOp>(user))
      worklist.append(user->user_begin(), user->user_end());
  }
  return false;
}

bool isEligibleLoad(mlir::Operation *op) {
  auto load = dyn_cast<tt::LoadOp>(op);
  return load && !load.getIsVolatile() && !load->hasAttr(kKeepLayoutAttr) &&
         !hasDotOperandLayoutUser(load);
}

// Elements per thread along `order[0]` that Coalesce may give `load` when
// the base address only needs element alignment. Mirrors the fork's
// getNumElementsPerThread for loads accepted by isEligibleLoad.
unsigned unalignedElementsPerThread(tt::LoadOp load,
                                    llvm::ArrayRef<unsigned> order,
                                    tt::ModuleAxisInfoAnalysis &axisInfo,
                                    llvm::ArrayRef<int64_t> shapePerCTA) {
  auto ptrTy = cast<mlir::RankedTensorType>(load.getPtr().getType());
  tt::AxisInfo *ptrInfo = axisInfo.getAxisInfo(load.getPtr());
  if (!ptrInfo)
    return 1;
  unsigned elemBits = tt::getPointeeBitWidth(ptrTy);
  unsigned elemBytes = std::max(elemBits / 8, 1u);
  unsigned maxMultiple =
      std::max<unsigned>(ptrInfo->getDivisibility(order[0]) / elemBytes, 1);
  unsigned maxContig = std::min<unsigned>(ptrInfo->getContiguity(order[0]),
                                          shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);

  unsigned unaligned = maxContig;
  if (mlir::Value mask = load.getMask()) {
    tt::AxisInfo *maskInfo = axisInfo.getAxisInfo(mask);
    unsigned maskAlignment =
        maskInfo ? std::max<unsigned>(maskInfo->getConstancy(order[0]), 1) : 1;
    unaligned = std::min(unaligned, maskAlignment);
  }
  alignment = std::max(alignment, unaligned);
  return std::min(alignment, std::max(kMaxVecBits / elemBits, 1u));
}

// Rebuild `load` in `encoding`, converting its operands in and its result back
// out, the way Coalesce does. RemoveLayoutConversions folds the converts.
void retypeLoad(tt::LoadOp load, mlir::Attribute encoding) {
  mlir::OpBuilder b(load);
  mlir::Location loc = load.getLoc();
  llvm::SmallVector<mlir::Value> operands;
  for (mlir::Value operand : load->getOperands()) {
    if (auto tensorTy = dyn_cast<mlir::RankedTensorType>(operand.getType()))
      operand = ttg::ConvertLayoutOp::create(
          b, loc, tensorTy.cloneWithEncoding(encoding), operand);
    operands.push_back(operand);
  }
  auto loadTy = cast<mlir::RankedTensorType>(load.getType());
  mlir::Operation *newLoad = b.create(
      loc, load->getName().getIdentifier(), operands,
      mlir::TypeRange{loadTy.cloneWithEncoding(encoding)}, load->getAttrs());
  mlir::Value result =
      ttg::ConvertLayoutOp::create(b, loc, loadTy, newLoad->getResult(0));
  load.getResult().replaceAllUsesWith(result);
  load.erase();
}

class UTLXWidenUnalignedLoadsPass
    : public mlir::PassWrapper<UTLXWidenUnalignedLoadsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXWidenUnalignedLoadsPass)

  llvm::StringRef getArgument() const override {
    return "utlx-widen-unaligned-loads";
  }
  llvm::StringRef getDescription() const override {
    return "Widen coalesced layouts of loads with an unaligned base";
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    if (!targetSupportsUnalignedVectorLoads(mod))
      return;
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    tt::ModuleAxisInfoAnalysis axisInfo(mod);

    // Decide every layout first: rewriting invalidates the axis analysis.
    llvm::SmallVector<std::pair<tt::LoadOp, mlir::Attribute>> widened;
    mod.walk([&](tt::LoadOp load) {
      auto ptrTy = dyn_cast<mlir::RankedTensorType>(load.getPtr().getType());
      if (!ptrTy || !isa<tt::PointerType>(ptrTy.getElementType()) ||
          !isEligibleLoad(load))
        return;
      auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(ptrTy.getEncoding());
      tt::AxisInfo *ptrInfo = axisInfo.getAxisInfo(load.getPtr());
      if (!blocked || !ptrInfo)
        return;
      // Coalesce orders dimensions by contiguity; anything else is not a
      // layout it chose.
      llvm::SmallVector<unsigned> order =
          mlir::getOrderFromContiguity(ptrInfo->getContiguity());
      if (llvm::ArrayRef<unsigned>(order) != blocked.getOrder())
        return;

      llvm::SmallVector<int64_t> shapePerCTA = ttg::getShapePerCTA(ptrTy);
      unsigned current = blocked.getSizePerThread()[order[0]];
      unsigned own =
          unalignedElementsPerThread(load, order, axisInfo, shapePerCTA);

      // Coalesce gives a load the widest layout among the memory accesses of
      // the same shape and order in its slice. The stock widths are already
      // in `current`; add the relaxed widths of related eligible loads.
      unsigned perThread = std::max(current, own);
      bool hasRelatedUnalignedLoad = false;
      for (mlir::Operation *use : mlir::getSlice(load)) {
        auto related = dyn_cast<tt::LoadOp>(use);
        if (!related || related == load || !isEligibleLoad(related))
          continue;
        auto relatedTy =
            dyn_cast<mlir::RankedTensorType>(related.getPtr().getType());
        tt::AxisInfo *relatedInfo = axisInfo.getAxisInfo(related.getPtr());
        if (!relatedTy || relatedTy.getShape() != ptrTy.getShape() ||
            !relatedInfo ||
            mlir::getOrderFromContiguity(relatedInfo->getContiguity()) != order)
          continue;
        perThread = std::max(
            perThread,
            unalignedElementsPerThread(related, order, axisInfo, shapePerCTA));
        hasRelatedUnalignedLoad = true;
      }

      int numWarps = ttg::lookupNumWarps(load);
      int64_t numElems = mlir::product<int64_t>(shapePerCTA);
      int64_t numThreads = numWarps * threadsPerWarp;
      perThread = std::min<int64_t>(
          perThread, std::max<int64_t>(numElems / numThreads, 1));
      // A masked load keeps the width its own predicate groups allow even when
      // a related load selects a wider layout.
      if (load.getMask() && hasRelatedUnalignedLoad)
        perThread = std::min(perThread, own);
      if (perThread <= current)
        return;

      llvm::SmallVector<unsigned> sizePerThread(ptrTy.getRank(), 1);
      sizePerThread[order[0]] = perThread;
      widened.emplace_back(load, ttg::BlockedEncodingAttr::get(
                                     mod.getContext(), ptrTy.getShape(),
                                     sizePerThread, order, numWarps,
                                     threadsPerWarp, blocked.getCGALayout()));
    });
    for (auto [load, encoding] : widened)
      retypeLoad(load, encoding);
  }
};

// Operand segments of amdg.buffer_load: ptr, offsets, stride, mask, other.
constexpr unsigned kBufferLoadSegments = 5;
constexpr unsigned kBufferLoadOffsetsSegment = 1;
constexpr unsigned kBufferLoadMaskSegment = 3;

// The operand of `op` in `segment`, or null if that segment is empty. The
// uTLX wheel is built without the AMD dialect headers, so amdg.buffer_load is
// accessed generically.
mlir::Value getSegmentOperand(mlir::Operation *op, unsigned segment) {
  std::optional<mlir::Attribute> attr =
      op->getInherentAttr("operandSegmentSizes");
  auto sizes = dyn_cast_or_null<mlir::DenseI32ArrayAttr>(
      attr ? *attr : mlir::Attribute());
  if (!sizes || sizes.size() != kBufferLoadSegments)
    return {};
  llvm::ArrayRef<int32_t> segmentSizes = sizes.asArrayRef();
  if (segmentSizes[segment] != 1)
    return {};
  unsigned index = 0;
  for (unsigned i = 0; i < segment; ++i)
    index += segmentSizes[i];
  return op->getOperand(index);
}

class UTLXWidenUnalignedBufferLoadsPass
    : public mlir::PassWrapper<UTLXWidenUnalignedBufferLoadsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      UTLXWidenUnalignedBufferLoadsPass)

  llvm::StringRef getArgument() const override {
    return "utlx-widen-unaligned-buffer-loads";
  }
  llvm::StringRef getDescription() const override {
    return "Raise buffer_load contiguity past the base alignment";
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    if (!targetSupportsUnalignedVectorLoads(mod))
      return;
    mlir::MLIRContext *ctx = mod.getContext();
    auto contiguityName = mlir::StringAttr::get(ctx, "contiguity");
    tt::ModuleAxisInfoAnalysis axisInfo(mod);

    llvm::SmallVector<std::pair<mlir::Operation *, unsigned>> widened;
    mod.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() != "amdg.buffer_load" ||
          op->getNumResults() != 1)
        return;
      auto resultTy =
          dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
      mlir::Value offsets = getSegmentOperand(op, kBufferLoadOffsetsSegment);
      tt::AxisInfo *offsetsInfo =
          offsets ? axisInfo.getAxisInfo(offsets) : nullptr;
      if (!resultTy || !resultTy.getEncoding() || !offsetsInfo)
        return;

      // The base is a scalar, so the addresses are as contiguous as the
      // offsets. The registers a thread owns bound the vector too.
      llvm::SmallVector<unsigned> order = ttg::getOrder(resultTy);
      unsigned contig =
          std::min<unsigned>(ttg::getContigPerThread(resultTy)[order[0]],
                             offsetsInfo->getContiguity(order[0]));
      unsigned elemBits = resultTy.getElementTypeBitWidth();
      unsigned vec =
          std::min(std::max(kMaxVecBits / elemBits, 1u), std::max(contig, 1u));
      if (mlir::Value mask = getSegmentOperand(op, kBufferLoadMaskSegment))
        vec = std::min(vec, axisInfo.getMaskAlignment(mask));
      if (vec > 1)
        vec = llvm::bit_floor(vec);

      unsigned current = 1;
      if (std::optional<mlir::Attribute> attr =
              op->getInherentAttr(contiguityName))
        if (auto intAttr = dyn_cast_or_null<mlir::IntegerAttr>(*attr))
          current = intAttr.getInt();
      if (vec > current)
        widened.emplace_back(op, vec);
    });

    auto i32 = mlir::IntegerType::get(ctx, 32);
    for (auto [op, vec] : widened)
      op->setInherentAttr(contiguityName, mlir::IntegerAttr::get(i32, vec));
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createWidenUnalignedLoadsPass() {
  return std::make_unique<UTLXWidenUnalignedLoadsPass>();
}

void registerWidenUnalignedLoadsPass() {
  mlir::PassRegistration<UTLXWidenUnalignedLoadsPass>();
}

std::unique_ptr<mlir::Pass> createWidenUnalignedBufferLoadsPass() {
  return std::make_unique<UTLXWidenUnalignedBufferLoadsPass>();
}

void registerWidenUnalignedBufferLoadsPass() {
  mlir::PassRegistration<UTLXWidenUnalignedBufferLoadsPass>();
}

} // namespace utlx
