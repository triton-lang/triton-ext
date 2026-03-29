/// uTLX plugin: standalone Triton Plugin Extensions entry point.
///
/// This plugin implements:
///   - TLX dialect registration (types, attrs, ops)
///   - TLX transform passes
///   - TLX custom ops for the TritonOpBuilder (memory ops, barriers, etc.)
///   - ConvertTritonToTritonGPU plugin pass
///
/// Exports tritonGetPluginInfo() for loading via TRITON_PLUGIN_PATHS.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/PluginUtils.h"

// TLX dialect headers
#include "tlx/dialect/include/IR/Dialect.h"
#include "tlx/dialect/include/Transforms/Passes.h"

// Ported passes
#include "passes/Passes.h"

// New/modified op wrappers
#include "ops/NewOps.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tlx = mlir::triton::tlx;

// ---------------------------------------------------------------------------
// Helper: extract a constant integer from an arith.constant Value
// ---------------------------------------------------------------------------

static std::optional<int64_t> extractConstantInt(mlir::Value v) {
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

// ===========================================================================
// Custom Ops
// ===========================================================================

// --- utlx_local_alloc: SMEM allocation with automatic layout encoding ---
static void createLocalAllocSmem(TritonOpBuilder &self,
                                 std::vector<mlir::Value> &operands) {
  if (operands.size() < 5)
    return;

  mlir::Type elemType = operands[1].getType();

  auto targetHintVal = extractConstantInt(operands.back());
  if (!targetHintVal)
    return;
  bool isAMD = *targetHintVal == 1;

  llvm::SmallVector<int64_t> fullShape;
  for (unsigned i = 2; i < operands.size() - 1; ++i) {
    auto dimVal = extractConstantInt(operands[i]);
    if (!dimVal)
      return;
    fullShape.push_back(*dimVal);
  }

  unsigned perBufferRank = fullShape.size() - 1;
  auto *context = self.getBuilder().getContext();
  auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(context, perBufferRank);

  mlir::Attribute encoding;
  if (perBufferRank == 1 || isAMD) {
    llvm::SmallVector<unsigned> order;
    for (int i = perBufferRank - 1; i >= 0; --i)
      order.push_back(static_cast<unsigned>(i));
    encoding = ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, order,
                                                    cgaLayout);
  } else {
    llvm::SmallVector<int64_t> perBufferShape(fullShape.begin() + 1,
                                              fullShape.end());
    llvm::SmallVector<unsigned> order;
    for (int i = perBufferRank - 1; i >= 0; --i)
      order.push_back(static_cast<unsigned>(i));
    encoding = ttg::NVMMASharedEncodingAttr::get(context, perBufferShape, order,
                                                 cgaLayout, elemType, false);
  }

  auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
  auto memDescType =
      ttg::MemDescType::get(fullShape, elemType, encoding, memorySpace, true);

  operands[0] = self.create<ttg::LocalAllocOp>(memDescType);
}

// --- utlx_local_alloc_tmem: TMEM allocation (Blackwell) ---
// operands[0] = result slot
// operands[1] = type carrier (element type)
// operands[2..N-2] = shape dims (full shape including num buffers)
// operands[N-1] = layout hint (0 = tensor_memory_layout, 1 = dummy_tmem_layout)
static void createLocalAllocTmem(TritonOpBuilder &self,
                                 std::vector<mlir::Value> &operands) {
  if (operands.size() < 4)
    return;

  mlir::Type elemType = operands[1].getType();

  auto layoutHintVal = extractConstantInt(operands.back());
  if (!layoutHintVal)
    return;
  bool useDummyTmemLayout = *layoutHintVal == 1;

  llvm::SmallVector<int64_t> fullShape;
  for (unsigned i = 2; i < operands.size() - 1; ++i) {
    auto dimVal = extractConstantInt(operands[i]);
    if (!dimVal)
      return;
    fullShape.push_back(*dimVal);
  }

  unsigned perBufferRank = fullShape.size() - 1;
  auto *context = self.getBuilder().getContext();

  mlir::Attribute encoding;
  if (useDummyTmemLayout) {
    encoding = tlx::DummyTMEMLayoutAttr::get(context);
  } else {
    // TensorMemoryEncodingAttr with blockM=shape[0], blockN=shape[1],
    // colStride=1
    auto cgaLayout =
        ttg::CGAEncodingAttr::get1CTALayout(context, perBufferRank);
    llvm::SmallVector<int64_t> perBufferShape(fullShape.begin() + 1,
                                              fullShape.end());
    unsigned blockM = perBufferShape.size() >= 1 ? perBufferShape[0] : 1;
    unsigned blockN = perBufferShape.size() >= 2 ? perBufferShape[1] : 1;
    encoding = ttng::TensorMemoryEncodingAttr::get(
        context, blockM, blockN, /*colStride=*/1, cgaLayout, /*twoCTAs=*/false);
  }

  auto memorySpace = ttng::TensorMemorySpaceAttr::get(context);
  auto memDescType =
      ttg::MemDescType::get(fullShape, elemType, encoding, memorySpace, true);

  operands[0] = self.create<ttng::TMEMAllocOp>(memDescType, nullptr);
}

// --- utlx_local_view: Index into a multi-buffer allocation ---
static void createLocalView(TritonOpBuilder &self,
                            std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;

  mlir::Value localAlloc = operands[1];
  mlir::Value bufferIdx = operands[2];

  auto localAllocType = mlir::dyn_cast<ttg::MemDescType>(localAlloc.getType());
  if (!localAllocType)
    return;

  auto localAllocShape = localAllocType.getShape();
  mlir::Type memDescType;
  if (localAllocShape.size() == 1) {
    memDescType = ttg::MemDescType::get(
        {1}, localAllocType.getElementType(), localAllocType.getEncoding(),
        localAllocType.getMemorySpace(), localAllocType.getMutableMemory());
  } else {
    memDescType = ttg::MemDescType::get(
        localAllocShape.drop_front(), localAllocType.getElementType(),
        localAllocType.getEncoding(), localAllocType.getMemorySpace(),
        localAllocType.getMutableMemory());
  }

  operands[0] =
      self.create<ttg::MemDescIndexOp>(memDescType, localAlloc, bufferIdx);
}

// --- utlx_local_store: Store register tensor into SMEM buffer ---
static void createLocalStore(TritonOpBuilder &self,
                             std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;

  mlir::Value dst = operands[1];
  mlir::Value src = operands[2];

  if (!mlir::isa<ttg::MemDescType>(dst.getType()))
    return;

  self.create<ttg::LocalStoreOp>(src, dst);
}

// --- utlx_local_load: Load from SMEM buffer into register tensor ---
static void createLocalLoad(TritonOpBuilder &self,
                            std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;

  mlir::Value subView = operands[1];

  auto subViewType = mlir::dyn_cast<ttg::MemDescType>(subView.getType());
  if (!subViewType)
    return;

  auto newType = mlir::RankedTensorType::get(subViewType.getShape(),
                                             subViewType.getElementType());

  mlir::Value asyncToken;
  if (operands.size() > 2)
    asyncToken = operands[2];

  operands[0] = self.create<ttg::LocalLoadOp>(newType, subView, asyncToken);
}

// --- utlx_alloc_barriers: Allocate mbarriers in shared memory ---
static void createAllocBarriers(TritonOpBuilder &self,
                                std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;

  auto numBarriersVal = extractConstantInt(operands[1]);
  auto arriveCountVal = extractConstantInt(operands[2]);
  if (!numBarriersVal || !arriveCountVal)
    return;

  int64_t numBarriers = *numBarriersVal;
  int arriveCount = static_cast<int>(*arriveCountVal);

  auto *context = self.getBuilder().getContext();
  auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
  auto i64Type = self.getBuilder().getI64Type();

  int numCTAs = 1;
  auto cgaLayout = ttg::CGAEncodingAttr::get1DLayout(context, numCTAs);
  auto encoding =
      ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {0}, cgaLayout);

  auto barriersMemDescType = ttg::MemDescType::get(
      {numBarriers, numCTAs}, i64Type, encoding, memorySpace, true);

  auto singleBarrierMemDescType =
      ttg::MemDescType::get({numCTAs}, i64Type, encoding, memorySpace, true);

  mlir::Value bufferViews = self.create<ttg::LocalAllocOp>(barriersMemDescType);

  for (int64_t i = 0; i < numBarriers; i++) {
    mlir::Value idx = mlir::arith::ConstantIntOp::create(
        self.getBuilder(), bufferViews.getLoc(), i, 32);
    mlir::Value buf = self.create<ttg::MemDescIndexOp>(singleBarrierMemDescType,
                                                       bufferViews, idx);
    self.create<ttng::InitBarrierOp>(buf, arriveCount);
  }

  operands[0] = bufferViews;
}

// --- utlx_barrier_wait: Wait on mbarrier ---
static void createBarrierWait(TritonOpBuilder &self,
                              std::vector<mlir::Value> &operands) {
  if (operands.size() < 4)
    return;

  mlir::Value mbarrierLoc = operands[1];
  mlir::Value phase = operands[2];
  mlir::Value pred = operands[3];

  self.create<ttng::WaitBarrierOp>(mbarrierLoc, phase, pred);
}

// --- utlx_barrier_arrive: Arrive at mbarrier ---
static void createBarrierArrive(TritonOpBuilder &self,
                                std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;

  mlir::Value mbarrierLoc = operands[1];
  auto arriveCountVal = extractConstantInt(operands[2]);
  if (!arriveCountVal)
    return;

  self.create<ttng::ArriveBarrierOp>(mbarrierLoc,
                                     static_cast<int>(*arriveCountVal));
}

// --- utlx_barrier_expect: Expect bytes on mbarrier ---
static void createBarrierExpect(TritonOpBuilder &self,
                                std::vector<mlir::Value> &operands) {
  if (operands.size() < 4)
    return;

  mlir::Value mbarrierLoc = operands[1];
  auto expectBytesVal = extractConstantInt(operands[2]);
  if (!expectBytesVal)
    return;
  mlir::Value pred = operands[3];

  self.create<ttng::BarrierExpectOp>(mbarrierLoc,
                                     static_cast<int>(*expectBytesVal), pred);
}

// Named barrier ops are now handled by runtime op wrappers in ops/NewOps.cpp

// --- utlx_storage_alias_spec: Create a storage alias specification ---
static void createStorageAliasSpec(TritonOpBuilder &self,
                                   std::vector<mlir::Value> &operands) {
  // operands[0] = result slot
  // operands[1] = storage kind (i32: 0=smem, 1=tmem)
  // operands[2] = optional buffer_size_bytes (i64, -1 if not set)
  if (operands.size() < 3)
    return;

  auto storageVal = extractConstantInt(operands[1]);
  auto sizeVal = extractConstantInt(operands[2]);
  if (!storageVal)
    return;

  auto *context = self.getBuilder().getContext();
  auto storageKind =
      *storageVal == 0 ? tlx::StorageKind::smem : tlx::StorageKind::tmem;

  std::optional<int64_t> bufferSizeBytes;
  if (sizeVal && *sizeVal >= 0)
    bufferSizeBytes = *sizeVal;

  auto resultType =
      tlx::StorageAliasSpecType::get(context, storageKind, bufferSizeBytes);
  auto storageAttr = tlx::StorageKindAttr::get(context, storageKind);
  mlir::IntegerAttr bufferSizeAttr = nullptr;
  if (bufferSizeBytes)
    bufferSizeAttr = self.getBuilder().getI64IntegerAttr(*bufferSizeBytes);
  mlir::DenseI64ArrayAttr bufferShapeAttr = nullptr;

  operands[0] = self.create<tlx::StorageAliasSpecOp>(
      resultType, storageAttr, bufferSizeAttr, bufferShapeAttr);
}

// --- utlx_storage_alias_local_alloc: Allocate referencing a storage alias ---
static void createStorageAliasLocalAlloc(TritonOpBuilder &self,
                                         std::vector<mlir::Value> &operands) {
  // operands[0] = result slot
  // operands[1] = storage_alias_spec Value
  // operands[2] = type carrier (for element type)
  // operands[3..N-1] = shape dims
  // operands[N-1] = storage hint (0=smem, 1=tmem)
  if (operands.size() < 5)
    return;

  mlir::Value storageAlias = operands[1];
  mlir::Type elemType = operands[2].getType();

  auto storageHintVal = extractConstantInt(operands.back());
  if (!storageHintVal)
    return;
  bool isTmem = *storageHintVal == 1;

  llvm::SmallVector<int64_t> fullShape;
  for (unsigned i = 3; i < operands.size() - 1; ++i) {
    auto dimVal = extractConstantInt(operands[i]);
    if (!dimVal)
      return;
    fullShape.push_back(*dimVal);
  }

  auto *context = self.getBuilder().getContext();
  unsigned perBufferRank = fullShape.size() - 1;

  mlir::Attribute encoding;
  mlir::Attribute memorySpace;

  if (isTmem) {
    memorySpace = ttng::TensorMemorySpaceAttr::get(context);
    // Use dummy TMEM layout for now; resolved during layout propagation
    encoding = tlx::DummyTMEMLayoutAttr::get(context);
  } else {
    memorySpace = ttg::SharedMemorySpaceAttr::get(context);
    auto cgaLayout =
        ttg::CGAEncodingAttr::get1CTALayout(context, perBufferRank);
    llvm::SmallVector<unsigned> order;
    for (int i = perBufferRank - 1; i >= 0; --i)
      order.push_back(static_cast<unsigned>(i));
    if (perBufferRank == 1) {
      encoding = ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, order,
                                                      cgaLayout);
    } else {
      llvm::SmallVector<int64_t> perBufferShape(fullShape.begin() + 1,
                                                fullShape.end());
      encoding = ttg::NVMMASharedEncodingAttr::get(
          context, perBufferShape, order, cgaLayout, elemType, false);
    }
  }

  auto memDescType =
      ttg::MemDescType::get(fullShape, elemType, encoding, memorySpace, true);

  operands[0] =
      self.create<tlx::StorageAliasLocalAllocOp>(memDescType, storageAlias);
}

// --- utlx_reuse_group: Create a reuse group ---
static void createReuseGroup(TritonOpBuilder &self,
                             std::vector<mlir::Value> &operands) {
  // operands[0] = result slot
  // operands[1] = group_kind (i32: 0=shared, 1=distinct)
  // operands[2] = group_size (i32)
  // operands[3..N] = elements
  if (operands.size() < 4)
    return;

  auto groupKindVal = extractConstantInt(operands[1]);
  auto groupSizeVal = extractConstantInt(operands[2]);
  if (!groupKindVal || !groupSizeVal)
    return;

  auto *context = self.getBuilder().getContext();
  auto groupKindEnum = *groupKindVal == 0 ? tlx::ReuseGroupKind::shared
                                          : tlx::ReuseGroupKind::distinct;

  std::vector<mlir::Value> elements(operands.begin() + 3, operands.end());
  auto resultType = tlx::ReuseGroupType::get(context, groupKindEnum);
  auto groupKindAttr = tlx::ReuseGroupKindAttr::get(context, groupKindEnum);
  auto groupSizeAttr = self.getBuilder().getI64IntegerAttr(*groupSizeVal);

  operands[0] = self.create<tlx::ReuseGroupOp>(resultType, elements,
                                               groupKindAttr, groupSizeAttr);
}

// --- utlx_set_buffer_overlap ---
static void createSetBufferOverlap(TritonOpBuilder &self,
                                   std::vector<mlir::Value> &operands) {
  // operands[0] = result slot (unused)
  // operands[1] = storage_alias_spec
  // operands[2] = overlap_def (reuse_group)
  if (operands.size() < 3)
    return;

  self.create<tlx::SetBufferOverlapOp>(operands[1], operands[2]);
}

// --- utlx_local_alias: Create a local alias ---
static void createLocalAlias(TritonOpBuilder &self,
                             std::vector<mlir::Value> &operands) {
  // operands[0] = result slot
  // operands[1] = source memdesc
  // operands[2] = type carrier (element type)
  // operands[3..N-1] = shape dims
  // operands[N-1] = storage hint (0=smem, 1=tmem)
  if (operands.size() < 5)
    return;

  mlir::Value src = operands[1];
  mlir::Type elemType = operands[2].getType();

  auto storageHintVal = extractConstantInt(operands.back());
  if (!storageHintVal)
    return;
  bool isTmem = *storageHintVal == 1;

  llvm::SmallVector<int64_t> shape;
  for (unsigned i = 3; i < operands.size() - 1; ++i) {
    auto dimVal = extractConstantInt(operands[i]);
    if (!dimVal)
      return;
    shape.push_back(*dimVal);
  }

  auto srcType = mlir::dyn_cast<ttg::MemDescType>(src.getType());
  if (!srcType)
    return;

  auto memDescType = ttg::MemDescType::get(
      shape, elemType, srcType.getEncoding(), srcType.getMemorySpace(), true);

  operands[0] = self.create<tlx::LocalAliasOp>(memDescType, src);
}

// --- utlx_require_layout ---
static void createRequireLayout(TritonOpBuilder &self,
                                std::vector<mlir::Value> &operands) {
  // operands[0] = result slot
  // operands[1] = source value
  // operands[2] = encoding (as type carrier — not used directly here)
  // The encoding must be constructed separately; for now this is a passthrough
  if (operands.size() < 2)
    return;

  // This custom op just wraps the source — actual layout is set by the
  // require_layout builder method in Python which calls builder methods
  // directly
  operands[0] = operands[1];
}

// --- utlx_release_layout ---
static void createReleaseLayout(TritonOpBuilder &self,
                                std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;

  auto type = mlir::dyn_cast<mlir::RankedTensorType>(operands[1].getType());
  if (!type)
    return;

  auto newType =
      mlir::RankedTensorType::get(type.getShape(), type.getElementType());
  operands[0] = self.create<tlx::ReleaseLayoutOp>(newType, operands[1]);
}

// --- utlx_async_commit_group: Commit async copies with token threading ---
// operands[0] = result slot
// operands[1..N] = input async tokens (optional)
static void createAsyncCommitGroup(TritonOpBuilder &self,
                                   std::vector<mlir::Value> &operands) {
  llvm::SmallVector<mlir::Value> tokens;
  for (unsigned i = 1; i < operands.size(); ++i)
    tokens.push_back(operands[i]);
  operands[0] = self.create<ttg::AsyncCommitGroupOp>(tokens);
}

// --- utlx_async_wait_group: Wait for async copies with token threading ---
// operands[0] = result slot (unused for void ops, but kept for consistency)
// operands[1] = pendings (i32 constant)
// operands[2..N] = input async tokens (optional)
static void createAsyncWaitGroup(TritonOpBuilder &self,
                                 std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return;

  auto pendingsVal = extractConstantInt(operands[1]);
  if (!pendingsVal)
    return;

  llvm::SmallVector<mlir::Value> tokens;
  for (unsigned i = 2; i < operands.size(); ++i)
    tokens.push_back(operands[i]);
  self.create<ttg::AsyncWaitOp>(tokens, static_cast<int>(*pendingsVal));
}

// --- utlx_warp_group_dot_wait: WarpGroupDotWaitOp with ReleaseLayoutOp unwrap
// --- operands[0] = result slot operands[1] = input value (possibly wrapped in
// ReleaseLayoutOp) operands[2] = pendings (i32 constant)
static void createWarpGroupDotWait(TritonOpBuilder &self,
                                   std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return;

  mlir::Value input = operands[1];
  auto pendingsVal = extractConstantInt(operands[2]);
  if (!pendingsVal)
    return;

  // Unwrap ReleaseLayoutOp if present
  mlir::Value realInput = input;
  tlx::ReleaseLayoutOp releaseOp = nullptr;
  if (auto defOp = input.getDefiningOp()) {
    if (auto release = mlir::dyn_cast<tlx::ReleaseLayoutOp>(defOp)) {
      releaseOp = release;
      realInput = release.getSrc();
    }
  }

  // Create WarpGroupDotWaitOp with the unwrapped input
  auto waitOp = self.create<ttng::WarpGroupDotWaitOp>(
      llvm::SmallVector<mlir::Value>{realInput},
      static_cast<unsigned>(*pendingsVal));

  if (releaseOp) {
    // Move ReleaseLayoutOp after the wait and rewire
    releaseOp->moveAfter(waitOp.getOperation());
    releaseOp.getOperation()->setOperand(0, waitOp.getResult(0));
    operands[0] = releaseOp.getResult();
  } else {
    operands[0] = waitOp.getResult(0);
  }
}

// ===========================================================================
// Pass registration
// ===========================================================================

// ConvertTritonToTritonGPU pass (defined in uTLXConversionPatterns.cpp)
extern void addUTLXConversionPass(mlir::PassManager *pm,
                                  const std::vector<std::string> &args);
extern void registerUTLXConversionPass();
extern void addUTLXInsertAndPropagatePass(mlir::PassManager *pm,
                                          const std::vector<std::string> &);
extern void registerUTLXInsertAndPropagatePass();

// TLX transform passes
static void addTLXFixupPass(mlir::PassManager *pm,
                            const std::vector<std::string> &args) {
  // args: [target, num_warps, threads_per_warp, num_ctas, cluster_dims...]
  tlx::TritonTLXFixupOptions options;
  if (args.size() >= 1)
    options.target = args[0];
  if (args.size() >= 2)
    options.numWarps = std::stoi(args[1]);
  if (args.size() >= 3)
    options.threadsPerWarp = std::stoi(args[2]);
  if (args.size() >= 4)
    options.numCTAs = std::stoi(args[3]);
  for (size_t i = 4; i < args.size(); ++i)
    options.clusterDims.push_back(std::stoi(args[i]));
  pm->addPass(tlx::createTritonTLXFixup(options));
}

static void registerTLXFixupPass() { tlx::registerPasses(); }

static void addPropagateLayoutPass(mlir::PassManager *pm,
                                   const std::vector<std::string> &) {
  pm->addPass(tlx::createTlxPropagateLayout());
}

static void registerPropagateLayoutPass() {
  // Already registered by registerTLXFixupPass / registerPasses
}

static void addInsertRequireLayoutPass(mlir::PassManager *pm,
                                       const std::vector<std::string> &) {
  pm->addPass(tlx::createTLXInsertRequireLayout());
}

static void addRewriteLocalAliasPass(mlir::PassManager *pm,
                                     const std::vector<std::string> &) {
  pm->addPass(tlx::createTLXRewriteLocalAlias());
}

static void addResolvePlaceholderLayoutsPass(mlir::PassManager *pm,
                                             const std::vector<std::string> &) {
  pm->addPass(tlx::createTLXResolvePlaceholderLayouts());
}

static void addPrintTTGIRToTLXPass(mlir::PassManager *pm,
                                   const std::vector<std::string> &) {
  pm->addPass(tlx::createTLXPrintTTGIRToTLX());
}

static void addStorageAliasLoweringPass(mlir::PassManager *pm,
                                        const std::vector<std::string> &) {
  pm->addPass(tlx::createTLXStorageAliasLowering());
}

static void noopRegister() {}

// --- Ported NVIDIA passes ---
static void addPruneUnusedBarriersPass(mlir::PassManager *pm,
                                       const std::vector<std::string> &) {
  pm->addPass(utlx::createPruneUnusedBarriersPass());
}

static void registerPruneUnusedBarriersPassFn() {
  utlx::registerPruneUnusedBarriersPass();
}

static void addPingPongPrepPass(mlir::PassManager *pm,
                                const std::vector<std::string> &) {
  pm->addPass(utlx::createPingPongPrepPass());
}

static void registerPingPongPrepPassFn() { utlx::registerPingPongPrepPass(); }

static void addPingPongSyncPass(mlir::PassManager *pm,
                                const std::vector<std::string> &) {
  pm->addPass(utlx::createPingPongSyncPass());
}

static void registerPingPongSyncPassFn() { utlx::registerPingPongSyncPass(); }

// --- Ported AMD passes ---
// NOTE: AMD barrier passes are disabled until triton-tlx-core-changes patch
// is applied. Uncomment when patched triton is available:
// static void addAMDLowerBarrierOpsPass(mlir::PassManager *pm,
//                                        const std::vector<std::string> &) {
//   pm->addPass(utlx::createAMDLowerBarrierOpsPass());
// }
// static void registerAMDLowerBarrierOpsPassFn() {
//   utlx::registerAMDLowerBarrierOpsPass();
// }

// ===========================================================================
// Dialect registration
// ===========================================================================

static void registerTLXDialect(mlir::DialectRegistry *registry) {
  registry->insert<tlx::TLXDialect>();
}

// ===========================================================================
// Plugin entry point
// ===========================================================================

using namespace mlir::triton;

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo passes[] = {
      {"utlx_convert_triton_to_tritongpu", "0.1.0", addUTLXConversionPass,
       registerUTLXConversionPass},
      {"utlx_insert_and_propagate_layout", "0.1.0",
       addUTLXInsertAndPropagatePass, registerUTLXInsertAndPropagatePass},
      {"utlx_fixup", "0.1.0", addTLXFixupPass, registerTLXFixupPass},
      {"utlx_propagate_layout", "0.1.0", addPropagateLayoutPass,
       registerPropagateLayoutPass},
      {"utlx_insert_require_layout", "0.1.0", addInsertRequireLayoutPass,
       noopRegister},
      {"utlx_rewrite_local_alias", "0.1.0", addRewriteLocalAliasPass,
       noopRegister},
      {"utlx_resolve_placeholder_layouts", "0.1.0",
       addResolvePlaceholderLayoutsPass, noopRegister},
      {"utlx_print_ttgir_to_tlx", "0.1.0", addPrintTTGIRToTLXPass,
       noopRegister},
      {"utlx_storage_alias_lowering", "0.1.0", addStorageAliasLoweringPass,
       noopRegister},
      // Ported NVIDIA passes
      {"utlx_prune_unused_barriers", "0.1.0", addPruneUnusedBarriersPass,
       registerPruneUnusedBarriersPassFn},
      {"utlx_ping_pong_prep", "0.1.0", addPingPongPrepPass,
       registerPingPongPrepPassFn},
      {"utlx_ping_pong_sync", "0.1.0", addPingPongSyncPass,
       registerPingPongSyncPassFn},
      // Ported AMD passes (disabled until patched triton is available)
      // {"utlx_amd_lower_barrier_ops", "0.1.0",
      //  addAMDLowerBarrierOpsPass, registerAMDLowerBarrierOpsPassFn},
  };

  static plugin::DialectInfo dialects[] = {
      {"TLXDialect", "0.1.0", registerTLXDialect},
  };

  static plugin::OpInfo ops[] = {
      // Original TLX ops
      {"utlx_local_alloc", createLocalAllocSmem},
      {"utlx_local_alloc_tmem", createLocalAllocTmem},
      {"utlx_local_view", createLocalView},
      {"utlx_local_store", createLocalStore},
      {"utlx_local_load", createLocalLoad},
      {"utlx_alloc_barriers", createAllocBarriers},
      {"utlx_barrier_wait", createBarrierWait},
      {"utlx_barrier_arrive", createBarrierArrive},
      {"utlx_barrier_expect", createBarrierExpect},
      {"utlx_storage_alias_spec", createStorageAliasSpec},
      {"utlx_storage_alias_local_alloc", createStorageAliasLocalAlloc},
      {"utlx_reuse_group", createReuseGroup},
      {"utlx_set_buffer_overlap", createSetBufferOverlap},
      {"utlx_local_alias", createLocalAlias},
      {"utlx_require_layout", createRequireLayout},
      {"utlx_release_layout", createReleaseLayout},
      // New TTG ops (runtime op creation)
      {"utlx_remote_shmem_store", utlx::createRemoteShmemStore},
      {"utlx_async_remote_shmem_store", utlx::createAsyncRemoteShmemStore},
      {"utlx_clock64", utlx::createClock64},
      // New TTNG ops (runtime op creation)
      {"utlx_async_store", utlx::createAsyncStore},
      {"utlx_fence", utlx::createFence},
      {"utlx_map_to_remote_buffer", utlx::createMapToRemoteBuffer},
      {"utlx_cluster_size_1d", utlx::createClusterSize1D},
      {"utlx_async_clc_try_cancel", utlx::createAsyncCLCTryCancel},
      {"utlx_clc_query_cancel", utlx::createCLCQueryCancel},
      {"utlx_vote_ballot_sync", utlx::createVoteBallotSync},
      {"utlx_async_tma_prefetch", utlx::createAsyncTMAPrefetch},
      {"utlx_named_barrier_arrive", utlx::createNamedBarrierArrive},
      {"utlx_named_barrier_wait", utlx::createNamedBarrierWait},
      // New AMD ops (runtime op creation)
      {"utlx_read_barrier_phase", utlx::createReadBarrierPhase},
      // Modified ops with extended signatures (runtime op creation)
      {"utlx_fp_to_fp_with_rbits", utlx::createFpToFpWithRbits},
      {"utlx_make_tensor_desc_with_desc_ptr",
       utlx::createMakeTensorDescWithDescPtr},
      // Combined require-layout ops (encoding + RequireLayoutOp)
      {"utlx_require_nv_mma_shared_layout",
       utlx::createRequireNvMmaSharedLayout},
      {"utlx_require_nv_mma_layout", utlx::createRequireNvMmaLayout},
      {"utlx_require_dot_operand_layout", utlx::createRequireDotOperandLayout},
      {"utlx_require_tensor_memory_layout",
       utlx::createRequireTensorMemoryLayout},
      {"utlx_require_tensor_memory_scales_layout",
       utlx::createRequireTensorMemoryScalesLayout},
      // Memory ops
      {"utlx_async_load", utlx::createAsyncLoad},
      {"utlx_global_scratch_alloc", utlx::createGlobalScratchAlloc},
      {"utlx_make_dummy_register_layout", utlx::createMakeDummyRegisterLayout},
      {"utlx_require_with_layout_carrier",
       utlx::createRequireWithLayoutCarrier},
      {"utlx_alloc_clc_responses", utlx::createAllocClcResponses},
      {"utlx_clc_query", utlx::createClcQuery},
      // Thread/cluster ops
      {"utlx_cluster_cta_rank", utlx::createClusterCtaRank},
      {"utlx_thread_id", utlx::createThreadId},
      // MMA ops
      {"utlx_warp_group_dot_wait", createWarpGroupDotWait},
      // Async copy ops with token threading
      {"utlx_async_commit_group", createAsyncCommitGroup},
      {"utlx_async_wait_group", createAsyncWaitGroup},
  };

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "uTLXPlugin",
      "0.1.0",
      passes,
      12, // numPasses
      dialects,
      1, // numDialects
      ops,
      48, // numOps
  };
  return &info;
}
