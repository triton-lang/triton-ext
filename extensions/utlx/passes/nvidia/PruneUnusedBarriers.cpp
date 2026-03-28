/// PruneUnusedBarriers pass ported to uTLX plugin.
///
/// Removes barrier allocations that have no wait-like uses after
/// warp specialization materializes them.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

enum class UseKind {
  Wait,
  Pruneable,
  Unknown,
};

UseKind classifyUse(mlir::Operation *user) {
  if (mlir::isa<ttng::WaitBarrierOp>(user))
    return UseKind::Wait;
  if (mlir::isa<ttng::InitBarrierOp, ttng::InvalBarrierOp,
                ttng::ArriveBarrierOp, ttng::AsyncCopyMbarrierArriveOp>(user))
    return UseKind::Pruneable;
  return UseKind::Unknown;
}

void traceBarrierUses(mlir::Value barrierVal,
                      llvm::SmallVectorImpl<mlir::OpOperand *> &terminalUses) {
  for (mlir::OpOperand &use : barrierVal.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (user->hasTrait<mlir::OpTrait::MemDescViewTrait>()) {
      assert(user->getNumResults() == 1);
      traceBarrierUses(user->getResult(0), terminalUses);
      continue;
    }
    if (auto wsOp = mlir::dyn_cast<ttg::WarpSpecializeOp>(user)) {
      unsigned operandIdx = use.getOperandNumber();
      for (mlir::Region *region : wsOp.getPartitionRegions()) {
        mlir::Value blockArg = region->getArgument(operandIdx);
        traceBarrierUses(blockArg, terminalUses);
      }
      continue;
    }
    terminalUses.push_back(&use);
  }
}

bool isBarrierAlloc(ttg::LocalAllocOp alloc) {
  auto memDescType = alloc.getType();
  if (!memDescType.getElementType().isInteger(64))
    return false;
  return !alloc.getSrc();
}

void pruneBarrier(ttg::LocalAllocOp alloc,
                  llvm::SmallVectorImpl<mlir::OpOperand *> &terminalUses) {
  // Phase 1: Erase terminal uses.
  for (mlir::OpOperand *use : terminalUses) {
    mlir::Operation *user = use->getOwner();
    if (mlir::isa<ttng::InitBarrierOp, ttng::InvalBarrierOp,
                  ttng::ArriveBarrierOp, ttng::AsyncCopyMbarrierArriveOp>(
            user)) {
      user->erase();
      continue;
    }
  }

  // Phase 2: Clean up warp_specialize captures.
  llvm::SmallVector<std::pair<ttg::WarpSpecializeOp, unsigned>> wsCaptures;
  std::function<void(mlir::Value)> collectWSCaptures = [&](mlir::Value val) {
    for (mlir::OpOperand &use : val.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user->hasTrait<mlir::OpTrait::MemDescViewTrait>()) {
        collectWSCaptures(user->getResult(0));
        continue;
      }
      if (auto wsOp = mlir::dyn_cast<ttg::WarpSpecializeOp>(user)) {
        wsCaptures.push_back({wsOp, use.getOperandNumber()});
      }
    }
  };
  collectWSCaptures(alloc.getResult());

  for (auto [wsOp, idx] : wsCaptures) {
    bool allUnused = true;
    for (mlir::Region *region : wsOp.getPartitionRegions()) {
      if (!region->getArgument(idx).use_empty()) {
        allUnused = false;
        break;
      }
    }
    if (allUnused) {
      llvm::BitVector toRemove(wsOp.getPartitionOp().getNumOperands());
      toRemove.set(idx);
      for (mlir::Region *region : wsOp.getPartitionRegions())
        region->front().eraseArguments(toRemove);
      wsOp->eraseOperands(toRemove);
    }
  }

  // Phase 3: Clean up dead view ops.
  std::function<void(mlir::Value)> eraseDeadViews = [&](mlir::Value val) {
    llvm::SmallVector<mlir::Operation *> users;
    for (mlir::OpOperand &use : val.getUses())
      users.push_back(use.getOwner());
    for (mlir::Operation *user : users) {
      if (user->hasTrait<mlir::OpTrait::MemDescViewTrait>() &&
          user->getResult(0).use_empty()) {
        user->erase();
      }
    }
  };
  eraseDeadViews(alloc.getResult());

  // Phase 4: Erase the alloc if unused.
  if (alloc.use_empty())
    alloc.erase();
}

class UTLXPruneUnusedBarriersPass
    : public mlir::PassWrapper<UTLXPruneUnusedBarriersPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXPruneUnusedBarriersPass)

  llvm::StringRef getArgument() const override {
    return "utlx-prune-unused-barriers";
  }
  llvm::StringRef getDescription() const override {
    return "Remove barrier allocations with no wait uses";
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    llvm::SmallVector<ttg::LocalAllocOp> barrierAllocs;
    mod.walk([&](ttg::LocalAllocOp alloc) {
      if (isBarrierAlloc(alloc))
        barrierAllocs.push_back(alloc);
    });

    for (auto alloc : barrierAllocs) {
      llvm::SmallVector<mlir::OpOperand *> terminalUses;
      traceBarrierUses(alloc.getResult(), terminalUses);

      bool hasWaitUses = false;
      bool hasUnknownUses = false;

      for (mlir::OpOperand *use : terminalUses) {
        UseKind kind = classifyUse(use->getOwner());
        switch (kind) {
        case UseKind::Wait:
          hasWaitUses = true;
          break;
        case UseKind::Unknown:
          hasUnknownUses = true;
          break;
        case UseKind::Pruneable:
          break;
        }
      }

      if (hasWaitUses || hasUnknownUses)
        continue;

      pruneBarrier(alloc, terminalUses);
    }
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createPruneUnusedBarriersPass() {
  return std::make_unique<UTLXPruneUnusedBarriersPass>();
}

void registerPruneUnusedBarriersPass() {
  mlir::PassRegistration<UTLXPruneUnusedBarriersPass>();
}

} // namespace utlx
