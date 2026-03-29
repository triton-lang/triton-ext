/// PingPong Barrier Insertion passes ported to uTLX plugin.
///
/// Two passes:
///   1. UTLXPingPongPrep: Groups expensive ops into pingpong regions
///   2. UTLXPingPongSync: Inserts named barrier arrive/wait for pingpong
///   regions
///
/// Ported from
/// third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/PingPong.cpp
///
/// NOTE: The PingPongSync pass requires NamedBarrierArriveOp /
/// NamedBarrierWaitOp which are only available when the triton-tlx-core-changes
/// patch is applied. When building against unpatched triton, the sync pass will
/// warn and skip barrier insertion.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "utlx-ping-pong"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

typedef int AsyncTaskId;
llvm::SmallVector<AsyncTaskId> getAsyncTaskIds(mlir::Operation *op) {
  llvm::SmallVector<AsyncTaskId> result;
  if (auto attr = op->getAttrOfType<mlir::DenseI32ArrayAttr>("async_task_id")) {
    for (auto id : attr.asArrayRef())
      result.push_back(id);
  }
  return result;
}

class CriticalRegionManager {
private:
  static constexpr unsigned MIN_BARRIER_ID = 7;
  static constexpr unsigned MAX_BARRIER_ID = 15;

public:
  unsigned barrierId = MIN_BARRIER_ID;
  llvm::DenseMap<int, std::pair<unsigned, unsigned>> pingpongIdToBarrierId;
  llvm::DenseMap<int, llvm::SmallVector<mlir::Operation *>> pingpongIdToKeyOps;
  llvm::DenseMap<int, llvm::SmallVector<mlir::Operation *>>
      pingpongIdToPingBoundaryOps;
  llvm::DenseMap<int, llvm::SmallVector<mlir::Operation *>>
      pingpongIdToPongBoundaryOps;
  llvm::DenseMap<int, int> pingpongIdToThreadNum;

  CriticalRegionManager() = default;

  bool isExpensiveOp(mlir::Operation *op, int computeCapability) const {
    switch (computeCapability) {
    case 90: // Hopper
      if (mlir::isa<ttng::WarpGroupDotOp>(op)) {
        LDBG("Encounter a " << op->getName() << " op on Hopper.");
        return true;
      }
      break;
    case 100: // Blackwell
      if (mlir::isa<mlir::math::ExpOp, mlir::math::Exp2Op>(op)) {
        LDBG("Encounter a " << op->getName() << " op on Blackwell.");
        mlir::Type resultType = op->getResult(0).getType();
        if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(resultType))
          return tensorTy.getRank() > 1;
      }
      break;
    }
    return false;
  }

  void assignBarrierId(int pingpongId) {
    if (pingpongIdToBarrierId.count(pingpongId) > 0)
      return;
    if (barrierId + 1 > MAX_BARRIER_ID)
      return;
    pingpongIdToBarrierId[pingpongId] = {barrierId, barrierId + 1};
    LDBG("Assigned barrier ID {" << barrierId << ", " << barrierId + 1
                                 << "} to pingpong region '" << pingpongId
                                 << "'.");
    barrierId += 2;
  }

  bool hasPingPongBoundary(int pingpongRegionId) const {
    return (pingpongIdToPingBoundaryOps.count(pingpongRegionId) > 0) &&
           (pingpongIdToPingBoundaryOps.at(pingpongRegionId).size() == 2) &&
           (pingpongIdToPongBoundaryOps.count(pingpongRegionId) > 0) &&
           (pingpongIdToPongBoundaryOps.at(pingpongRegionId).size() == 2);
  }
};

static int getSingleTaskId(mlir::Operation *op) {
  auto asyncTasks = getAsyncTaskIds(op);
  if (asyncTasks.size() != 1)
    return -1;
  return asyncTasks[0];
}

static unsigned getLoopDepth(mlir::Operation *op) {
  unsigned depth = 0;
  auto pOp = op->getParentOfType<mlir::scf::ForOp>();
  while (pOp) {
    ++depth;
    pOp = pOp->getParentOfType<mlir::scf::ForOp>();
  }
  return depth;
}

void getNestedFor(mlir::Region *partition,
                  llvm::DenseMap<unsigned, llvm::SmallVector<mlir::Operation *>>
                      &loopDepthMap) {
  partition->walk([&](mlir::Operation *subOp) {
    if (mlir::dyn_cast<mlir::scf::ForOp>(subOp)) {
      unsigned tDepth = getLoopDepth(subOp);
      loopDepthMap[tDepth].push_back(subOp);
    }
  });
}

bool areControlFlowEquivalent(mlir::Operation *op1, mlir::Operation *op2) {
  assert(op1 && op2);
  if (op1->getBlock() != op2->getBlock())
    return false;
  mlir::Operation *earlier = op1;
  mlir::Operation *later = op2;
  if (later->isBeforeInBlock(earlier))
    std::swap(earlier, later);
  for (mlir::Operation *cur = earlier->getNextNode(); cur && cur != later;
       cur = cur->getNextNode()) {
    if (mlir::isa<mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::IfOp>(cur))
      return false;
  }
  return true;
}

mlir::Operation *findEndOp(CriticalRegionManager &crManager,
                           mlir::Operation *keyOp,
                           mlir::Operation *stopOp = nullptr) {
  mlir::Operation *curOp = keyOp;
  mlir::Operation *later = stopOp;
  if (stopOp) {
    if (later->isBeforeInBlock(curOp))
      std::swap(curOp, later);
  }
  while (curOp) {
    if (mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::WhileOp>(curOp))
      return nullptr;
    if (!mlir::isMemoryEffectFree(curOp))
      return curOp;
    if (curOp == stopOp)
      return nullptr;
    mlir::Operation *nextOp = curOp->getNextNode();
    if (!nextOp || !areControlFlowEquivalent(curOp, nextOp))
      return nullptr;
    curOp = nextOp;
  }
  return nullptr;
}

mlir::Operation *firstOpInBlock(llvm::ArrayRef<mlir::Operation *> ops) {
  if (ops.empty())
    return nullptr;
  auto it = llvm::min_element(ops, [](mlir::Operation *a, mlir::Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return *it;
}

mlir::Operation *lastOpInBlock(llvm::ArrayRef<mlir::Operation *> ops) {
  if (ops.empty())
    return nullptr;
  auto it = llvm::max_element(ops, [](mlir::Operation *a, mlir::Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return *it;
}

int arrivesFirst(llvm::ArrayRef<mlir::Operation *> keyOps) {
  auto it =
      llvm::min_element(keyOps, [](mlir::Operation *a, mlir::Operation *b) {
        auto scA = ttg::getStageCluster(a);
        auto scB = ttg::getStageCluster(b);
        int stageA = scA->first, stageB = scB->first;
        int clusterA = scA->second, clusterB = scB->second;
        if (stageA == stageB) {
          if (clusterA == clusterB)
            return a->isBeforeInBlock(b);
          return clusterA < clusterB;
        }
        return stageA < stageB;
      });
  return getSingleTaskId(*it);
}

/// Create a named barrier arrive/wait op by name (runtime lookup).
/// This avoids compile-time dependency on
/// NamedBarrierArriveOp/NamedBarrierWaitOp.
static bool createNamedBarrierOp(mlir::OpBuilder &builder, mlir::Location loc,
                                 llvm::StringRef opName, mlir::Value barrier,
                                 mlir::Value numThreads) {
  auto *ctx = builder.getContext();
  auto registeredOp = mlir::RegisteredOperationName::lookup(opName, ctx);
  if (!registeredOp) {
    return false;
  }
  mlir::OperationState state(loc, *registeredOp);
  state.addOperands({barrier, numThreads});
  builder.create(state);
  return true;
}

static constexpr llvm::StringLiteral kNamedBarrierArrive =
    "triton_nvidia_gpu.arrive_barrier_named";
static constexpr llvm::StringLiteral kNamedBarrierWait =
    "triton_nvidia_gpu.wait_barrier_named";

static void handleWarpSpec(ttg::WarpSpecializeOp wsOp, int computeCapability) {
  llvm::SmallVector<
      llvm::DenseMap<unsigned, llvm::SmallVector<mlir::Operation *>>>
      partitionLoopDepths;
  llvm::SmallVector<mlir::Region *> computeRegions;

  for (mlir::Region *region : wsOp.getPartitionRegions()) {
    computeRegions.push_back(region);
    llvm::DenseMap<unsigned, llvm::SmallVector<mlir::Operation *>> loopDepths;
    getNestedFor(region, loopDepths);
    partitionLoopDepths.push_back(loopDepths);
  }

  unsigned numPartitionWithLoops = 0;
  bool hasSingleOuterLoop = true;
  for (auto &loopDepth : partitionLoopDepths) {
    if (!loopDepth.empty())
      numPartitionWithLoops += 1;
    if (loopDepth[0].size() != 1)
      hasSingleOuterLoop = false;
  }
  if (numPartitionWithLoops < 2 || !hasSingleOuterLoop)
    return;

  CriticalRegionManager crManager;

  for (unsigned iter = 0; iter < computeRegions.size(); ++iter) {
    mlir::Region *region = computeRegions[iter];
    region->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      if (auto pingpongIdAttr =
              op->getAttrOfType<mlir::IntegerAttr>("pingpong_id")) {
        int pingpongId = pingpongIdAttr.getInt();
        crManager.pingpongIdToKeyOps[pingpongId].push_back(op);
        crManager.assignBarrierId(pingpongId);
      }
    });
  }

  for (auto &[pingpongId, keyOps] : crManager.pingpongIdToKeyOps) {
    llvm::DenseMap<int, llvm::SmallVector<mlir::Operation *>> startOps;
    llvm::DenseMap<int, llvm::SmallVector<mlir::Operation *>> endOps;
    llvm::DenseMap<int, int> numWarps;

    bool foundNullEndOp = false;
    for (auto &keyOp : keyOps) {
      int partitionId = getSingleTaskId(keyOp);
      if (partitionId != -1) {
        mlir::Operation *endOp = findEndOp(crManager, keyOp, nullptr);
        if (!endOp) {
          foundNullEndOp = true;
          break;
        }
        startOps[partitionId].push_back(keyOp);
        endOps[partitionId].push_back(endOp);
        if (numWarps.count(partitionId) == 0)
          numWarps[partitionId] = ttg::lookupNumWarps(keyOp);
      }
    }
    if (foundNullEndOp)
      continue;
    if (startOps.size() != 2 || endOps.size() != 2 || numWarps.size() != 2)
      continue;

    int arrivesFirstPartitionId = arrivesFirst(keyOps);
    int numberOfThreads = 0;
    for (auto [partitionId, startOp] : startOps) {
      mlir::Operation *unionStartOp = firstOpInBlock(startOp);
      mlir::Operation *unionEndOp = lastOpInBlock(endOps[partitionId]);
      if (partitionId != arrivesFirstPartitionId) {
        crManager.pingpongIdToPingBoundaryOps[pingpongId].push_back(
            unionStartOp);
        crManager.pingpongIdToPingBoundaryOps[pingpongId].push_back(unionEndOp);
      } else {
        crManager.pingpongIdToPongBoundaryOps[pingpongId].push_back(
            unionStartOp);
        crManager.pingpongIdToPongBoundaryOps[pingpongId].push_back(unionEndOp);
      }
      numberOfThreads += numWarps[partitionId] * 32;
    }
    if (crManager.pingpongIdToThreadNum.count(pingpongId) == 0)
      crManager.pingpongIdToThreadNum[pingpongId] = numberOfThreads;
  }

  // Step 3: Insert pingpong barriers using runtime op lookup
  for (auto &[pingpongId, pingBoundOps] :
       crManager.pingpongIdToPingBoundaryOps) {
    if (!crManager.hasPingPongBoundary(pingpongId))
      continue;
    const llvm::SmallVector<mlir::Operation *> &pongBoundOps =
        crManager.pingpongIdToPongBoundaryOps[pingpongId];
    if (crManager.pingpongIdToBarrierId.count(pingpongId) == 0)
      continue;

    auto [pingBarrierId, pongBarrierId] =
        crManager.pingpongIdToBarrierId[pingpongId];
    int numThreads = crManager.pingpongIdToThreadNum[pingpongId];

    // Ping partition
    mlir::Operation *pingStart = pingBoundOps[0];
    mlir::Operation *pingEnd = pingBoundOps[1];
    mlir::Region *pingRegion = pingStart->getParentRegion();
    while (pingRegion) {
      mlir::Operation *parentOp = pingRegion->getParentOp();
      if (mlir::isa<ttg::WarpSpecializePartitionsOp>(parentOp))
        break;
      pingRegion = parentOp->getParentRegion();
    }
    if (!pingRegion)
      continue;

    mlir::Block &pingRegionBlock = pingRegion->front();
    mlir::OpBuilder builder(&pingRegionBlock, pingRegionBlock.begin());
    auto pingRegionLoc = pingRegionBlock.front().getLoc();
    mlir::Value pingBarrier = builder.create<mlir::arith::ConstantIntOp>(
        pingRegionLoc, pingBarrierId, 32);
    mlir::Value pongBarrier = builder.create<mlir::arith::ConstantIntOp>(
        pingRegionLoc, pongBarrierId, 32);
    mlir::Value pingNumThreads = builder.create<mlir::arith::ConstantIntOp>(
        pingRegionLoc, numThreads, 32);

    if (!createNamedBarrierOp(builder, pingRegionLoc, kNamedBarrierArrive,
                              pongBarrier, pingNumThreads)) {
      llvm::errs() << "utlx-ping-pong-sync: NamedBarrierArriveOp not "
                      "registered. Apply triton-tlx-core-changes patch.\n";
      return;
    }
    builder.setInsertionPoint(pingStart);
    createNamedBarrierOp(builder, pingStart->getLoc(), kNamedBarrierWait,
                         pingBarrier, pingNumThreads);
    builder.setInsertionPointAfter(pingEnd);
    createNamedBarrierOp(builder, pingEnd->getLoc(), kNamedBarrierArrive,
                         pongBarrier, pingNumThreads);

    // Pong partition
    mlir::Operation *pongStart = pongBoundOps[0];
    mlir::Operation *pongEnd = pongBoundOps[1];
    mlir::Region *pongRegion = pongStart->getParentRegion();
    mlir::Block &pongRegionBlock = pongRegion->front();
    mlir::OpBuilder builder2(&pongRegionBlock, pongRegionBlock.begin());
    auto pongRegionLoc = pongRegionBlock.front().getLoc();
    mlir::Value pingBarrier2 = builder2.create<mlir::arith::ConstantIntOp>(
        pongRegionLoc, pingBarrierId, 32);
    mlir::Value pongBarrier2 = builder2.create<mlir::arith::ConstantIntOp>(
        pongRegionLoc, pongBarrierId, 32);
    mlir::Value pingNumThreads2 = builder2.create<mlir::arith::ConstantIntOp>(
        pongRegionLoc, numThreads, 32);

    builder2.setInsertionPoint(pongStart);
    createNamedBarrierOp(builder2, pongStart->getLoc(), kNamedBarrierWait,
                         pongBarrier2, pingNumThreads2);
    builder2.setInsertionPointAfter(pongEnd);
    createNamedBarrierOp(builder2, pongEnd->getLoc(), kNamedBarrierArrive,
                         pingBarrier2, pingNumThreads2);
  }
}

// ---------------------------------------------------------------------------
// PingPongPrep pass
// ---------------------------------------------------------------------------

class UTLXPingPongPrepPass
    : public mlir::PassWrapper<UTLXPingPongPrepPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXPingPongPrepPass)

  llvm::StringRef getArgument() const override { return "utlx-ping-pong-prep"; }
  llvm::StringRef getDescription() const override {
    return "Group expensive ops into pingpong regions";
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    int capability = getNVIDIAComputeCapability(moduleOp);

    CriticalRegionManager crManager;

    moduleOp.walk([&](tt::FuncOp funcOp) {
      llvm::SmallVector<llvm::SmallVector<mlir::Operation *, 4>> expensiveOps;

      funcOp.walk([&](mlir::Operation *op) {
        if (!crManager.isExpensiveOp(op, capability))
          return;

        bool foundGroup = false;
        for (auto &group : expensiveOps) {
          bool matchType = true;
          for (auto &refOp : group) {
            if (op->getName() != refOp->getName()) {
              matchType = false;
              break;
            }
            if (!areControlFlowEquivalent(op, refOp)) {
              matchType = false;
              break;
            }
            int opTaskId = getSingleTaskId(op);
            int refTaskId = getSingleTaskId(refOp);
            if (opTaskId == -1 || refTaskId == -1)
              continue;
            if (opTaskId == refTaskId) {
              bool hasMemEffects = (findEndOp(crManager, op, refOp) != nullptr);
              if (hasMemEffects)
                matchType = false;
            }
          }
          foundGroup = matchType;
          if (foundGroup) {
            group.push_back(op);
            break;
          }
        }
        if (!foundGroup)
          expensiveOps.push_back({op});
      });

      unsigned pingpongID = 0;
      for (auto &group : expensiveOps) {
        llvm::SmallDenseSet<int, 4> partitionIds;
        for (auto *op : group) {
          int taskId = getSingleTaskId(op);
          if (taskId != -1)
            partitionIds.insert(taskId);
        }
        if (partitionIds.size() != 2)
          continue;
        for (auto *op : group) {
          op->setAttr(
              "pingpong_id",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32), pingpongID));
        }
        pingpongID++;
      }
    });
  }
};

// ---------------------------------------------------------------------------
// PingPongSync pass
// ---------------------------------------------------------------------------

class UTLXPingPongSyncPass
    : public mlir::PassWrapper<UTLXPingPongSyncPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXPingPongSyncPass)

  llvm::StringRef getArgument() const override { return "utlx-ping-pong-sync"; }
  llvm::StringRef getDescription() const override {
    return "Insert named barrier arrive/wait for pingpong regions";
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    int capability = getNVIDIAComputeCapability(moduleOp);

    moduleOp.walk([&](tt::FuncOp funcOp) {
      for (auto &block : funcOp.getBody().getBlocks()) {
        for (mlir::Operation &bodyOp : block.getOperations()) {
          if (auto wsOp = mlir::dyn_cast<ttg::WarpSpecializeOp>(&bodyOp))
            handleWarpSpec(wsOp, capability);
        }
      }
    });
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createPingPongPrepPass() {
  return std::make_unique<UTLXPingPongPrepPass>();
}

std::unique_ptr<mlir::Pass> createPingPongSyncPass() {
  return std::make_unique<UTLXPingPongSyncPass>();
}

void registerPingPongPrepPass() {
  mlir::PassRegistration<UTLXPingPongPrepPass>();
}

void registerPingPongSyncPass() {
  mlir::PassRegistration<UTLXPingPongSyncPass>();
}

} // namespace utlx
