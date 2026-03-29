/// Runtime op wrappers for new/modified MLIR ops introduced in triton-fb.
///
/// All ops are created via runtime op lookup using
/// mlir::RegisteredOperationName::lookup() so the plugin can build against
/// unpatched triton. If the op is not registered (unpatched build), a warning
/// is printed and the call is a no-op.

#include "ops/NewOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tlx = mlir::triton::tlx;

// ---------------------------------------------------------------------------
// Helper: create an op by runtime name lookup
// ---------------------------------------------------------------------------

static mlir::Operation *
createRuntimeOp(mlir::OpBuilder &builder, mlir::Location loc,
                llvm::StringRef opName, mlir::TypeRange resultTypes,
                mlir::ValueRange operands,
                llvm::ArrayRef<mlir::NamedAttribute> attrs = {}) {
  auto *ctx = builder.getContext();
  auto registeredOp = mlir::RegisteredOperationName::lookup(opName, ctx);
  if (!registeredOp) {
    llvm::errs() << "utlx: op '" << opName
                 << "' not registered in this Triton build.\n";
    return nullptr;
  }
  mlir::OperationState state(loc, *registeredOp);
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addAttributes(attrs);
  return builder.create(state);
}

// ---------------------------------------------------------------------------
// TTG ops
// ---------------------------------------------------------------------------

/// utlx_remote_shmem_store(src, dst, ctaRank)
void utlx::createRemoteShmemStore(TritonOpBuilder &self,
                                  std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;
  createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                  "triton_gpu.remote_shmem_store", {},
                  {operands[0], operands[1], operands[2]});
}

/// utlx_async_remote_shmem_store(src, dst, ctaRank, barrier)
void utlx::createAsyncRemoteShmemStore(TritonOpBuilder &self,
                                       std::vector<mlir::Value> &operands) {
  if (operands.size() < 4)
    return;
  createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                  "triton_gpu.async_remote_shmem_store", {},
                  {operands[0], operands[1], operands[2], operands[3]});
}

/// utlx_clock64() -> i64
void utlx::createClock64(TritonOpBuilder &self,
                         std::vector<mlir::Value> &operands) {
  auto i64Ty = self.getBuilder().getI64Type();
  auto *op = createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                             "triton_gpu.clock64", {i64Ty}, {});
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

// ---------------------------------------------------------------------------
// TTNG ops
// ---------------------------------------------------------------------------

/// utlx_async_store(src, dst, size)
void utlx::createAsyncStore(TritonOpBuilder &self,
                            std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;
  createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                  "triton_nvidia_gpu.async_store", {},
                  {operands[0], operands[1], operands[2]});
}

/// utlx_fence(scope_str_as_i32_constant)
/// scope is passed as an i32 constant: 0="gpu", 1="sys"
void utlx::createFence(TritonOpBuilder &self,
                       std::vector<mlir::Value> &operands) {
  if (operands.size() < 1)
    return;
  auto &builder = self.getBuilder();
  auto loc = self.getLastLoc();

  // Decode scope from i32 constant
  llvm::StringRef scope = "gpu";
  if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(
          operands[0].getDefiningOp())) {
    if (constOp.value() == 1)
      scope = "sys";
  }

  auto scopeAttr = builder.getNamedAttr("scope", builder.getStringAttr(scope));
  createRuntimeOp(builder, loc, "triton_nvidia_gpu.fence", {}, {}, {scopeAttr});
}

/// utlx_map_to_remote_buffer(src, ctaRank) -> memdesc
void utlx::createMapToRemoteBuffer(TritonOpBuilder &self,
                                   std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;
  mlir::Value src = operands[1];
  mlir::Value ctaRank = operands[2];

  auto *op = createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                             "triton_nvidia_gpu.map_to_remote_buffer",
                             {src.getType()}, {src, ctaRank});
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

/// utlx_cluster_size_1d() -> i32
void utlx::createClusterSize1D(TritonOpBuilder &self,
                               std::vector<mlir::Value> &operands) {
  auto i32Ty = self.getBuilder().getI32Type();
  auto *op = createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                             "triton_nvidia_gpu.cluster_size_1d", {i32Ty}, {});
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

/// utlx_async_clc_try_cancel(mbarAlloc, clcResAlloc)
void utlx::createAsyncCLCTryCancel(TritonOpBuilder &self,
                                   std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;
  createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                  "triton_nvidia_gpu.async_clc_try_cancel", {},
                  {operands[0], operands[1]});
}

/// utlx_clc_query_cancel(clcResAlloc) -> i32
void utlx::createCLCQueryCancel(TritonOpBuilder &self,
                                std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;
  auto i32Ty = self.getBuilder().getI32Type();
  auto *op = createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                             "triton_nvidia_gpu.clc_query_cancel", {i32Ty},
                             {operands[1]});
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

/// utlx_vote_ballot_sync(mask, pred) -> i32 or tensor
void utlx::createVoteBallotSync(TritonOpBuilder &self,
                                std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;
  mlir::Value mask = operands[1];
  mlir::Value pred = operands[2];

  // Result type: i32 for scalar pred, tensor<i32> for tensor pred
  mlir::Type resultType;
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(pred.getType())) {
    resultType = mlir::RankedTensorType::get(tensorTy.getShape(),
                                             self.getBuilder().getI32Type());
  } else {
    resultType = self.getBuilder().getI32Type();
  }

  auto *op = createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                             "triton_nvidia_gpu.vote_ballot_sync", {resultType},
                             {mask, pred});
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

/// utlx_async_tma_prefetch(desc, coord0, coord1, ..., pred)
/// operands[0] = result slot (unused, void op)
/// operands[1] = desc
/// operands[2..N-1] = coordinates
/// operands[N-1] = pred (i1)
void utlx::createAsyncTMAPrefetch(TritonOpBuilder &self,
                                  std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;
  // Last operand is pred, middle are coordinates
  mlir::Value desc = operands[1];
  mlir::Value pred = operands.back();
  llvm::SmallVector<mlir::Value> allOperands;
  allOperands.push_back(desc);
  for (size_t i = 2; i < operands.size() - 1; ++i)
    allOperands.push_back(operands[i]);
  allOperands.push_back(pred);

  createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                  "triton_nvidia_gpu.async_tma_prefetch", {}, allOperands);
}

/// utlx_named_barrier_arrive(bar, numThreads)
void utlx::createNamedBarrierArrive(TritonOpBuilder &self,
                                    std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;
  createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                  "triton_nvidia_gpu.named_barrier_arrive", {},
                  {operands[0], operands[1]});
}

/// utlx_named_barrier_wait(bar, numThreads)
void utlx::createNamedBarrierWait(TritonOpBuilder &self,
                                  std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;
  createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                  "triton_nvidia_gpu.named_barrier_wait", {},
                  {operands[0], operands[1]});
}

// ---------------------------------------------------------------------------
// AMD ops
// ---------------------------------------------------------------------------

/// utlx_read_barrier_phase(alloc) -> i32
void utlx::createReadBarrierPhase(TritonOpBuilder &self,
                                  std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;
  auto i32Ty = self.getBuilder().getI32Type();
  auto *op = createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                             "triton_amdgpu.read_barrier_phase", {i32Ty},
                             {operands[1]});
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

// ---------------------------------------------------------------------------
// Modified ops
// ---------------------------------------------------------------------------

/// utlx_fp_to_fp_with_rbits(result_slot, src, rbits, rounding_mode)
/// rounding_mode: i32 constant encoding TT_RoundingMode
///   (runtime lookup of enum values; 0=RTNE, 1=RTZ, 2=RS for stochastic)
void utlx::createFpToFpWithRbits(TritonOpBuilder &self,
                                 std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;
  auto &builder = self.getBuilder();
  auto loc = self.getLastLoc();

  mlir::Value src = operands[1];
  mlir::Value rbits = operands.size() > 3 ? operands[2] : mlir::Value();

  // Build attributes
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  if (operands.size() > 3) {
    // Last operand encodes rounding mode as i32 constant
    if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(
            operands.back().getDefiningOp())) {
      auto *ctx = builder.getContext();
      // Look up the rounding mode enum attr
      auto roundingAttrName = mlir::StringAttr::get(ctx, "rounding");
      int64_t mode = constOp.value();
      // Try to create the enum attr string
      llvm::StringRef modeStr;
      switch (mode) {
      case 0:
        modeStr = "rtne";
        break;
      case 1:
        modeStr = "rtz";
        break;
      case 2:
        modeStr = "rs";
        break;
      default:
        modeStr = "rtne";
        break;
      }
      attrs.push_back(builder.getNamedAttr(
          "rounding", mlir::StringAttr::get(ctx, modeStr)));
    }
  }

  // Build operand list: src, optional rbits
  llvm::SmallVector<mlir::Value> opOperands = {src};
  if (rbits)
    opOperands.push_back(rbits);

  // Result type matches src type (same shape, but potentially different elem)
  // Caller must set up operands[0] type carrier for the desired result type
  mlir::Type resultType = operands[0].getType();

  auto *op = createRuntimeOp(builder, loc, "tt.fp_to_fp", {resultType},
                             opOperands, attrs);
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

// ---------------------------------------------------------------------------
// Helper: extract a constant integer from an arith.constant Value
// ---------------------------------------------------------------------------

static std::optional<int64_t> extractConstInt(mlir::Value v) {
  if (auto constIntOp =
          mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(v.getDefiningOp()))
    return constIntOp.value();
  if (auto constOp =
          mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(v.getDefiningOp())) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Combined require-layout ops
// ---------------------------------------------------------------------------

/// utlx_require_nv_mma_shared_layout(result_slot, src, shape..., order...,
///                                     fp4Padded, swizzled)
/// All shape/order/flags are i32 constants.
/// Layout: shape = [rank values], order = [rank values], then fp4Padded,
/// swizzled Total operands = 1(result) + 1(src) + rank(shape) + rank(order) +
/// 2(flags) The rank is inferred: (numOperands - 4) / 2
void utlx::createRequireNvMmaSharedLayout(TritonOpBuilder &self,
                                          std::vector<mlir::Value> &operands) {
  if (operands.size() < 6) // at least rank=1
    return;

  mlir::Value src = operands[1];
  auto srcType = mlir::dyn_cast<ttg::MemDescType>(src.getType());
  if (!srcType)
    return;

  // Last two operands are fp4Padded, swizzled flags
  auto fp4PaddedVal = extractConstInt(operands[operands.size() - 2]);
  auto swizzledVal = extractConstInt(operands[operands.size() - 1]);
  if (!fp4PaddedVal || !swizzledVal)
    return;
  bool fp4Padded = *fp4PaddedVal != 0;
  bool swizzled = *swizzledVal != 0;

  // Remaining operands between src and flags: shape + order
  size_t numShapeOrder =
      operands.size() - 4; // minus result, src, fp4, swizzled
  if (numShapeOrder % 2 != 0)
    return;
  unsigned rank = numShapeOrder / 2;

  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<unsigned> order;
  for (unsigned i = 0; i < rank; ++i) {
    auto v = extractConstInt(operands[2 + i]);
    if (!v)
      return;
    shape.push_back(*v);
  }
  for (unsigned i = 0; i < rank; ++i) {
    auto v = extractConstInt(operands[2 + rank + i]);
    if (!v)
      return;
    order.push_back(static_cast<unsigned>(*v));
  }

  auto *context = self.getBuilder().getContext();
  auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(context, rank);

  mlir::Attribute encoding;
  if (swizzled) {
    encoding = ttg::NVMMASharedEncodingAttr::get(
        context, shape, order, cgaLayout, srcType.getElementType(), fp4Padded);
  } else {
    encoding = ttg::NVMMASharedEncodingAttr::get(
        context, /*swizzlingByteWidth=*/0,
        /*transposed=*/order[0] == 0,
        srcType.getElementType().getIntOrFloatBitWidth(), fp4Padded, cgaLayout);
  }

  auto newType = ttg::MemDescType::get(
      srcType.getShape(), srcType.getElementType(), encoding,
      srcType.getMemorySpace(), srcType.getMutableMemory(),
      srcType.getAllocShape());
  operands[0] = self.create<tlx::RequireLayoutOp>(newType, src);
}

/// utlx_require_nv_mma_layout(result_slot, opndA, opndAcc, version,
///                              versionMinor, moduleNumWarps)
void utlx::createRequireNvMmaLayout(TritonOpBuilder &self,
                                    std::vector<mlir::Value> &operands) {
  if (operands.size() < 6)
    return;

  mlir::Value opndA = operands[1];
  mlir::Value opndAcc = operands[2];
  auto versionVal = extractConstInt(operands[3]);
  auto versionMinorVal = extractConstInt(operands[4]);
  auto moduleNumWarpsVal = extractConstInt(operands[5]);
  if (!versionVal || !versionMinorVal || !moduleNumWarpsVal)
    return;

  unsigned versionMajor = static_cast<unsigned>(*versionVal);
  unsigned versionMinor = static_cast<unsigned>(*versionMinorVal);
  unsigned moduleNumWarps = static_cast<unsigned>(*moduleNumWarpsVal);

  auto *context = self.getBuilder().getContext();

  // Get element type of A
  mlir::Type dtypeA;
  if (auto memDesc = mlir::dyn_cast<ttg::MemDescType>(opndA.getType()))
    dtypeA = memDesc.getElementType();
  else if (auto tensorTy =
               mlir::dyn_cast<mlir::RankedTensorType>(opndA.getType()))
    dtypeA = tensorTy.getElementType();
  else
    return;

  auto retType = mlir::dyn_cast<mlir::RankedTensorType>(opndAcc.getType());
  if (!retType)
    return;
  auto retShapePerCTA = retType.getShape();

  // Look up contextual num_warps
  mlir::Block *parentBlock = self.getBuilder().getInsertionBlock();
  unsigned numWarps = moduleNumWarps;
  if (parentBlock) {
    if (auto *parentOp = parentBlock->getParentOp()) {
      auto contextualWarps = ttg::maybeLookupNumWarps(parentOp);
      if (contextualWarps)
        numWarps = static_cast<unsigned>(*contextualWarps);
    }
  }

  auto instrShape = mlir::mmaVersionToInstrShape(versionMajor, retShapePerCTA,
                                                 dtypeA, numWarps);

  llvm::SmallVector<unsigned, 2> warpsPerCTA = {numWarps, 1};
  auto CTALayout = ttg::CGAEncodingAttr::get1CTALayout(context, 2);
  auto encoding = ttg::NvidiaMmaEncodingAttr::get(
      context, versionMajor, versionMinor, warpsPerCTA, CTALayout, instrShape);

  auto newType = mlir::RankedTensorType::get(
      retType.getShape(), retType.getElementType(), encoding);
  operands[0] = self.create<tlx::RequireLayoutOp>(newType, opndAcc);
}

/// utlx_require_dot_operand_layout(result_slot, opnd, opIdx, parent_carrier)
/// parent_carrier: a Value whose type has the parent encoding (from
/// require_nv_mma_layout)
void utlx::createRequireDotOperandLayout(TritonOpBuilder &self,
                                         std::vector<mlir::Value> &operands) {
  if (operands.size() < 4)
    return;

  mlir::Value opnd = operands[1];
  auto opIdxVal = extractConstInt(operands[2]);
  if (!opIdxVal)
    return;
  unsigned opIdx = static_cast<unsigned>(*opIdxVal);
  mlir::Value parentCarrier = operands[3];

  auto *context = self.getBuilder().getContext();

  // Get parent encoding from the carrier value's type
  mlir::Attribute parentEnc;
  if (auto tensorTy =
          mlir::dyn_cast<mlir::RankedTensorType>(parentCarrier.getType()))
    parentEnc = tensorTy.getEncoding();
  else
    return;

  if (!parentEnc)
    return;

  auto opndType = mlir::dyn_cast<mlir::RankedTensorType>(opnd.getType());
  if (!opndType)
    return;

  auto encoding = ttg::DotOperandEncodingAttr::get(context, opIdx, parentEnc,
                                                   opndType.getElementType());

  auto newType = mlir::RankedTensorType::get(
      opndType.getShape(), opndType.getElementType(), encoding);
  operands[0] = self.create<tlx::RequireLayoutOp>(newType, opnd);
}

/// utlx_require_tensor_memory_layout(result_slot, src, blockM, blockN,
///                                     colStride, CTASplitM, CTASplitN)
void utlx::createRequireTensorMemoryLayout(TritonOpBuilder &self,
                                           std::vector<mlir::Value> &operands) {
  if (operands.size() < 7)
    return;

  mlir::Value src = operands[1];
  auto blockMVal = extractConstInt(operands[2]);
  auto blockNVal = extractConstInt(operands[3]);
  auto colStrideVal = extractConstInt(operands[4]);
  // CTASplitM, CTASplitN are unused for now (always 1-CTA)
  if (!blockMVal || !blockNVal || !colStrideVal)
    return;

  auto *context = self.getBuilder().getContext();
  auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(context, 2);
  auto encoding = ttng::TensorMemoryEncodingAttr::get(
      context, static_cast<unsigned>(*blockMVal),
      static_cast<unsigned>(*blockNVal), static_cast<unsigned>(*colStrideVal),
      cgaLayout, /*twoCTAs=*/false);

  auto srcType = mlir::dyn_cast<ttg::MemDescType>(src.getType());
  if (!srcType)
    return;

  auto newType = ttg::MemDescType::get(
      srcType.getShape(), srcType.getElementType(), encoding,
      srcType.getMemorySpace(), srcType.getMutableMemory(),
      srcType.getAllocShape());
  operands[0] = self.create<tlx::RequireLayoutOp>(newType, src);
}

/// utlx_require_tensor_memory_scales_layout(result_slot, src)
void utlx::createRequireTensorMemoryScalesLayout(
    TritonOpBuilder &self, std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;

  mlir::Value src = operands[1];
  auto *context = self.getBuilder().getContext();
  auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(context, 2);
  auto encoding = ttng::TensorMemoryScalesEncodingAttr::get(context, cgaLayout);

  auto srcType = mlir::dyn_cast<ttg::MemDescType>(src.getType());
  if (!srcType)
    return;

  auto newType = ttg::MemDescType::get(
      srcType.getShape(), srcType.getElementType(), encoding,
      srcType.getMemorySpace(), srcType.getMutableMemory(),
      srcType.getAllocShape());
  operands[0] = self.create<tlx::RequireLayoutOp>(newType, src);
}

// ---------------------------------------------------------------------------
// Thread/cluster ops
// ---------------------------------------------------------------------------

/// utlx_cluster_cta_rank() -> i32
void utlx::createClusterCtaRank(TritonOpBuilder &self,
                                std::vector<mlir::Value> &operands) {
  auto i32Ty = self.getBuilder().getI32Type();
  // ClusterCTAIdOp is registered as "nvgpu.cluster_id"
  auto *op = createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                             "nvgpu.cluster_id", {i32Ty}, {});
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

// ---------------------------------------------------------------------------
// Memory ops
// ---------------------------------------------------------------------------

/// utlx_async_load(result_slot, src, result_memdesc, [mask, other,]
/// useBulk_flag,
///                  [bulk_size, barrier])
/// useBulk_flag is always present; if useBulk=1, bulk_size and barrier follow.
/// For non-bulk loads, mask and other may be present before the flag.
void utlx::createAsyncLoad(TritonOpBuilder &self,
                           std::vector<mlir::Value> &operands) {
  if (operands.size() < 4)
    return;
  // The op is "triton_gpu.async_copy_global_to_local"
  // For now, use runtime op creation since the signature may vary
  mlir::Value src = operands[1];
  mlir::Value result = operands[2];

  // Find the useBulk flag - it's always the last or second-to-last group
  // For bulk: [src, result, bulk_size, barrier, useBulk=1]
  // For non-bulk: [src, result, (mask)?, (other)?, useBulk=0]
  auto useBulkVal = extractConstInt(operands.back());
  if (!useBulkVal)
    return;
  bool useBulk = *useBulkVal != 0;

  if (useBulk) {
    // operands: [result_slot, src, result, bulk_size, barrier, useBulk=1]
    if (operands.size() < 6)
      return;
    // Bulk async load is effectively a barrier-based TMA copy
    // Use AsyncCopyGlobalToLocalOp with bulk parameters
    // For now, create as runtime op since bulk variant may differ
    llvm::SmallVector<mlir::Value> opOperands = {src, result};
    auto *op = createRuntimeOp(
        self.getBuilder(), self.getLastLoc(),
        "triton_gpu.async_copy_global_to_local",
        {self.getBuilder().getType<ttg::AsyncTokenType>()}, opOperands);
    if (op && op->getNumResults() > 0)
      operands[0] = op->getResult(0);
  } else {
    // Non-bulk: operands[1]=src, operands[2]=result, then optional mask/other,
    // then useBulk=0
    llvm::SmallVector<mlir::Value> opOperands = {src, result};
    // Add mask and other if present (operands between result and useBulk flag)
    for (size_t i = 3; i < operands.size() - 1; ++i)
      opOperands.push_back(operands[i]);
    auto *op = createRuntimeOp(
        self.getBuilder(), self.getLastLoc(),
        "triton_gpu.async_copy_global_to_local",
        {self.getBuilder().getType<ttg::AsyncTokenType>()}, opOperands);
    if (op && op->getNumResults() > 0)
      operands[0] = op->getResult(0);
  }
}

/// utlx_global_scratch_alloc(result_slot, nbytes, alignment) -> ptr
void utlx::createGlobalScratchAlloc(TritonOpBuilder &self,
                                    std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;

  auto nbytesVal = extractConstInt(operands[1]);
  auto alignmentVal = extractConstInt(operands[2]);
  if (!nbytesVal || !alignmentVal)
    return;

  auto *context = self.getBuilder().getContext();
  auto ptrType = mlir::triton::PointerType::get(self.getBuilder().getI8Type(),
                                                /*addressSpace=*/1);

  auto nbytesAttr =
      self.getBuilder().getI32IntegerAttr(static_cast<int32_t>(*nbytesVal));
  auto alignmentAttr =
      self.getBuilder().getI32IntegerAttr(static_cast<int32_t>(*alignmentVal));

  auto *op = createRuntimeOp(
      self.getBuilder(), self.getLastLoc(), "triton_gpu.global_scratch_alloc",
      {ptrType}, {},
      {self.getBuilder().getNamedAttr("nbytes", nbytesAttr),
       self.getBuilder().getNamedAttr("alignment", alignmentAttr)});
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

/// utlx_make_dummy_register_layout(result_slot, shape..., type_carrier,
/// tmemCompatible) Creates a DummyRegisterLayoutAttr and returns it as a
/// RequireLayoutOp result. NOTE: This returns an attribute-carrying value, not
/// a real layout transform. It's used only to create a type carrier for
/// create_tmem_load/create_tmem_store.
void utlx::createMakeDummyRegisterLayout(TritonOpBuilder &self,
                                         std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;

  // Last operand is tmemCompatible flag, second-to-last is type carrier
  auto tmemVal = extractConstInt(operands.back());
  if (!tmemVal)
    return;
  bool tmemCompatible = *tmemVal != 0;

  mlir::Type elementType = operands[operands.size() - 2].getType();

  llvm::SmallVector<int64_t> shape;
  for (size_t i = 1; i < operands.size() - 2; ++i) {
    auto v = extractConstInt(operands[i]);
    if (!v)
      return;
    shape.push_back(*v);
  }

  auto *context = self.getBuilder().getContext();
  auto encoding = tlx::DummyRegisterLayoutAttr::get(context, shape, elementType,
                                                    tmemCompatible);

  // Return a Value that carries the encoding as a RankedTensorType
  auto tensorType = mlir::RankedTensorType::get(shape, elementType, encoding);
  // Create an arith.constant as a type carrier - the value doesn't matter,
  // only the type matters for create_tmem_load/create_tmem_store
  auto nullVal = self.getBuilder().create<mlir::arith::ConstantOp>(
      self.getLastLoc(), mlir::DenseElementsAttr::get(
                             mlir::RankedTensorType::get(shape, elementType),
                             self.getBuilder().getZeroAttr(elementType)));
  // Wrap in RequireLayoutOp to attach the encoding
  operands[0] = self.create<tlx::RequireLayoutOp>(tensorType, nullVal);
}

/// utlx_require_with_layout_carrier(result_slot, src, layout_carrier)
/// Applies the layout encoding from layout_carrier's type to src via
/// RequireLayoutOp.
void utlx::createRequireWithLayoutCarrier(TritonOpBuilder &self,
                                          std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;

  mlir::Value src = operands[1];
  mlir::Value layoutCarrier = operands[2];

  // Extract encoding from the carrier's type
  mlir::Attribute encoding;
  if (auto tensorTy =
          mlir::dyn_cast<mlir::RankedTensorType>(layoutCarrier.getType()))
    encoding = tensorTy.getEncoding();
  else if (auto memTy =
               mlir::dyn_cast<ttg::MemDescType>(layoutCarrier.getType()))
    encoding = memTy.getEncoding();

  if (!encoding) {
    operands[0] = src;
    return;
  }

  // Apply encoding to src
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(src.getType())) {
    auto newType = mlir::RankedTensorType::get(
        tensorTy.getShape(), tensorTy.getElementType(), encoding);
    operands[0] = self.create<tlx::RequireLayoutOp>(newType, src);
  } else if (auto memTy = mlir::dyn_cast<ttg::MemDescType>(src.getType())) {
    auto newType =
        ttg::MemDescType::get(memTy.getShape(), memTy.getElementType(),
                              encoding, memTy.getMemorySpace(),
                              memTy.getMutableMemory(), memTy.getAllocShape());
    operands[0] = self.create<tlx::RequireLayoutOp>(newType, src);
  } else {
    operands[0] = src;
  }
}

/// utlx_alloc_clc_responses(result_slot, numResponses) -> MemDesc
/// Allocates a shared memory buffer for CLC responses (128-bit entries).
void utlx::createAllocClcResponses(TritonOpBuilder &self,
                                   std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;

  auto numResponsesVal = extractConstInt(operands[1]);
  if (!numResponsesVal)
    return;
  int64_t numResponses = *numResponsesVal;

  auto *context = self.getBuilder().getContext();
  auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
  // CLC responses are 128-bit (i128) entries
  auto i128Type = self.getBuilder().getIntegerType(128, /*signed=*/false);

  auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(context, 1);
  auto encoding =
      ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {0}, cgaLayout);

  auto memDescType = ttg::MemDescType::get({numResponses}, i128Type, encoding,
                                           memorySpace, /*mutableMemory=*/true);

  operands[0] = self.create<ttg::LocalAllocOp>(memDescType);
}

/// utlx_clc_query(result_slot, clcResAlloc) -> i32
/// Queries CLC response and offsets by cluster CTA rank.
void utlx::createClcQuery(TritonOpBuilder &self,
                          std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;

  auto &builder = self.getBuilder();
  auto loc = self.getLastLoc();
  auto i32Ty = builder.getI32Type();

  // First query the cancel status
  auto *queryOp =
      createRuntimeOp(builder, loc, "triton_nvidia_gpu.clc_query_cancel",
                      {i32Ty}, {operands[1]});
  if (!queryOp || queryOp->getNumResults() == 0)
    return;

  mlir::Value tileId = queryOp->getResult(0);

  // Get cluster CTA rank
  auto *rankOp = createRuntimeOp(builder, loc, "nvgpu.cluster_id", {i32Ty}, {});
  if (!rankOp || rankOp->getNumResults() == 0) {
    operands[0] = tileId;
    return;
  }

  mlir::Value ctaRank = rankOp->getResult(0);

  // tileId == -1 ? tileId : tileId + ctaRank
  mlir::Value negOne = mlir::arith::ConstantIntOp::create(builder, loc, -1, 32);
  mlir::Value isNegOne = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, tileId, negOne);
  mlir::Value offset =
      mlir::arith::AddIOp::create(builder, loc, tileId, ctaRank);
  mlir::Value result =
      mlir::arith::SelectOp::create(builder, loc, isNegOne, tileId, offset);

  operands[0] = result;
}

// ---------------------------------------------------------------------------

/// utlx_thread_id(axis_constant) -> i32
void utlx::createThreadId(TritonOpBuilder &self,
                          std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;

  auto axisVal = extractConstInt(operands[1]);
  if (!axisVal)
    return;
  unsigned axis = static_cast<unsigned>(*axisVal);

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
  if (axis > 2)
    return;

  auto &builder = self.getBuilder();
  auto loc = self.getLastLoc();
  auto indexType = builder.getIndexType();
  auto threadId =
      mlir::gpu::ThreadIdOp::create(builder, loc, indexType, dims[axis]);
  auto i32Val = mlir::arith::IndexCastOp::create(
      builder, loc, builder.getI32Type(), threadId);
  operands[0] = i32Val;
}

// ---------------------------------------------------------------------------
// Modified ops
// ---------------------------------------------------------------------------

/// utlx_make_tensor_desc_with_desc_ptr(result_slot, base, shape..., strides...,
///                                      descPtr, rank_constant)
/// The last operand is a constant encoding the rank (number of shape dims).
/// shape dims come first, then stride dims, then optional descPtr.
void utlx::createMakeTensorDescWithDescPtr(TritonOpBuilder &self,
                                           std::vector<mlir::Value> &operands) {
  if (operands.size() < 4)
    return;
  auto &builder = self.getBuilder();
  auto loc = self.getLastLoc();

  // operands layout:
  //   [0] = result slot (type carries TT_TensorDescType)
  //   [1] = base (ptr)
  //   [2..2+rank-1] = shape dims (i32)
  //   [2+rank..2+2*rank-1] = stride dims (i64)
  //   [2+2*rank] = descPtr (optional, ptr) or rank_constant
  //   last = rank_constant (i32)

  // Extract rank from last operand
  auto rankVal = operands.back();
  int64_t rank = 0;
  if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(
          rankVal.getDefiningOp())) {
    rank = constOp.value();
  } else {
    llvm::errs() << "utlx_make_tensor_desc_with_desc_ptr: "
                    "last operand must be rank constant\n";
    return;
  }

  if (static_cast<int64_t>(operands.size()) < 2 + 2 * rank + 1)
    return;

  mlir::Value base = operands[1];
  llvm::SmallVector<mlir::Value> allOperands = {base};

  // Shape dims
  for (int64_t i = 0; i < rank; ++i)
    allOperands.push_back(operands[2 + i]);
  // Stride dims
  for (int64_t i = 0; i < rank; ++i)
    allOperands.push_back(operands[2 + rank + i]);

  // Check if there's a descPtr (operands.size() == 2 + 2*rank + 1 + 1)
  bool hasDescPtr = static_cast<int64_t>(operands.size()) > 2 + 2 * rank + 1;
  if (hasDescPtr)
    allOperands.push_back(operands[2 + 2 * rank]);

  mlir::Type resultType = operands[0].getType();

  auto *op = createRuntimeOp(builder, loc, "tt.make_tensor_desc", {resultType},
                             allOperands);
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}
