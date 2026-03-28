/// Runtime op wrappers for new/modified MLIR ops introduced in triton-fb.
///
/// All ops are created via runtime op lookup using
/// mlir::RegisteredOperationName::lookup() so the plugin can build against
/// unpatched triton. If the op is not registered (unpatched build), a warning
/// is printed and the call is a no-op.

#include "ops/NewOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

namespace ttg = mlir::triton::gpu;

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
                              "triton_nvidia_gpu.clc_query_cancel",
                              {i32Ty}, {operands[1]});
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
  if (auto tensorTy =
          mlir::dyn_cast<mlir::RankedTensorType>(pred.getType())) {
    resultType = mlir::RankedTensorType::get(tensorTy.getShape(),
                                              self.getBuilder().getI32Type());
  } else {
    resultType = self.getBuilder().getI32Type();
  }

  auto *op = createRuntimeOp(self.getBuilder(), self.getLastLoc(),
                              "triton_nvidia_gpu.vote_ballot_sync",
                              {resultType}, {mask, pred});
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
                  "triton_nvidia_gpu.async_tma_prefetch", {},
                  allOperands);
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
                              "triton_amdgpu.read_barrier_phase",
                              {i32Ty}, {operands[1]});
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
      case 0: modeStr = "rtne"; break;
      case 1: modeStr = "rtz"; break;
      case 2: modeStr = "rs"; break;
      default: modeStr = "rtne"; break;
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

  auto *op = createRuntimeOp(builder, loc, "tt.fp_to_fp",
                              {resultType}, opOperands, attrs);
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}

/// utlx_make_tensor_desc_with_desc_ptr(result_slot, base, shape..., strides...,
///                                      descPtr, rank_constant)
/// The last operand is a constant encoding the rank (number of shape dims).
/// shape dims come first, then stride dims, then optional descPtr.
void utlx::createMakeTensorDescWithDescPtr(
    TritonOpBuilder &self, std::vector<mlir::Value> &operands) {
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
  bool hasDescPtr =
      static_cast<int64_t>(operands.size()) > 2 + 2 * rank + 1;
  if (hasDescPtr)
    allOperands.push_back(operands[2 + 2 * rank]);

  mlir::Type resultType = operands[0].getType();

  auto *op = createRuntimeOp(builder, loc, "tt.make_tensor_desc",
                              {resultType}, allOperands);
  if (op && op->getNumResults() > 0)
    operands[0] = op->getResult(0);
}
