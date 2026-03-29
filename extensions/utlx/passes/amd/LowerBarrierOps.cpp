/// LowerBarrierOps pass ported to uTLX plugin.
///
/// Lowers ttng::InitBarrierOp, ttng::ArriveBarrierOp, ttng::WaitBarrierOp
/// to AMD-specific barrier ops (triton::amdgpu::*).
///
/// Ported from third_party/amd/lib/TritonAMDGPUTransforms/LowerBarrierOps.cpp
///
/// NOTE: ReadBarrierPhaseOp is created via runtime op lookup since it may
/// only be available when the triton-tlx-core-changes patch is applied.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

// AMD dialect includes
#include "Dialect/TritonAMDGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;

#define DEBUG_TYPE "utlx-amd-lower-barrier-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

static const int THREADS_PER_WAVE = 64;

void lowerInitBarrierOps(mlir::ModuleOp m,
                         std::map<mlir::triton::gpu::LocalAllocOp, int>
                             &localAllocToBarrierExpectedCount) {
  llvm::SmallVector<mlir::Operation *> eraseOps;
  auto cond = mlir::Value();
  m.walk([&](ttng::InitBarrierOp op) {
    LDBG("Lowering InitBarrierOp: " << op << "\n");
    auto loc = op.getLoc();
    mlir::OpBuilder builder(op);
    if (!cond) {
      auto i32ty = builder.getIntegerType(32);
      auto threadId = builder.create<mlir::ROCDL::ThreadIdXOp>(loc, i32ty);
      auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
      cond = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, threadId, zero);
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    thenBuilder.create<mlir::triton::amdgpu::InitBarrierOp>(loc, op.getAlloc(),
                                                            op.getCount());
    if (auto defOp = mlir::dyn_cast<ttg::MemDescIndexOp>(
            op.getAlloc().getDefiningOp())) {
      if (auto srcOp = mlir::dyn_cast<ttg::LocalAllocOp>(
              defOp.getSrc().getDefiningOp())) {
        localAllocToBarrierExpectedCount[srcOp] = op.getCount();
      }
    }
    eraseOps.push_back(op);
  });
  for (auto op : eraseOps)
    op->erase();
}

void lowerArriveBarrierOps(mlir::ModuleOp m,
                           const std::map<mlir::triton::gpu::LocalAllocOp, int>
                               &localAllocToBarrierExpectedCount) {
  llvm::SmallVector<mlir::Operation *> eraseOps;
  auto cond = mlir::Value();
  m.walk([&](ttng::ArriveBarrierOp op) {
    LDBG("Lowering ArriveBarrierOp: " << op << "\n");
    auto loc = op.getLoc();
    mlir::OpBuilder builder(op);
    if (!cond) {
      auto i32ty = builder.getIntegerType(32);
      auto threadId = builder.create<mlir::ROCDL::ThreadIdXOp>(loc, i32ty);
      auto threadsPerWave =
          builder.create<mlir::arith::ConstantIntOp>(loc, THREADS_PER_WAVE, 32);
      auto mod =
          builder.create<mlir::arith::RemSIOp>(loc, threadId, threadsPerWave);
      auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
      cond = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, mod, zero);
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    if (auto defOp = mlir::dyn_cast<ttg::MemDescIndexOp>(
            op.getAlloc().getDefiningOp())) {
      if (auto srcOp = mlir::dyn_cast<ttg::LocalAllocOp>(
              defOp.getSrc().getDefiningOp())) {
        auto it = localAllocToBarrierExpectedCount.find(srcOp);
        if (it != localAllocToBarrierExpectedCount.end()) {
          auto expectedCount = it->second;
          auto incrementCount = op.getCount();
          thenBuilder.create<mlir::triton::amdgpu::ArriveBarrierOp>(
              loc, op.getAlloc(), incrementCount, expectedCount);
        } else {
          assert(false && "Cannot find LocalAllocOp for ArriveBarrierOp");
        }
      }
    } else {
      assert(false && "ArriveBarrierOp not connected to LocalAllocOp");
    }
    eraseOps.push_back(op);
  });
  for (auto op : eraseOps)
    op->erase();
}

/// Create a ReadBarrierPhaseOp via runtime op lookup, since it may
/// not exist in unpatched triton.
static mlir::Value createReadBarrierPhaseOp(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Type resultType,
                                            mlir::Value alloc) {
  auto *ctx = builder.getContext();
  // Try the AMD-specific op first
  llvm::StringRef opName = "triton_amdgpu.read_barrier_phase";
  auto registeredOp = mlir::RegisteredOperationName::lookup(opName, ctx);
  if (!registeredOp) {
    llvm::errs() << "utlx-amd-lower-barrier-ops: ReadBarrierPhaseOp not "
                    "registered. Apply triton-tlx-core-changes patch.\n";
    return {};
  }
  mlir::OperationState state(loc, *registeredOp);
  state.addTypes(resultType);
  state.addOperands(alloc);
  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

void lowerWaitBarrierOps(mlir::ModuleOp m) {
  llvm::SmallVector<mlir::Operation *> eraseOps;
  m.walk([&](ttng::WaitBarrierOp op) {
    LDBG("Lowering WaitBarrierOp: " << op << "\n");
    auto loc = op.getLoc();
    mlir::OpBuilder builder(op);
    auto waitPhase = op.getPhase();
    auto whileOp = builder.create<mlir::scf::WhileOp>(loc, mlir::TypeRange{},
                                                      mlir::ValueRange{});
    // Spin Wait - Before block
    mlir::Block *beforeBlock = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToEnd(beforeBlock);
    auto i32ty = builder.getIntegerType(32);

    mlir::Value barrierPhase =
        createReadBarrierPhaseOp(builder, loc, i32ty, op.getAlloc());
    if (!barrierPhase) {
      // ReadBarrierPhaseOp not available, abort lowering
      whileOp.erase();
      return;
    }

    mlir::Value phaseCond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, barrierPhase, waitPhase);
    builder.create<mlir::scf::ConditionOp>(loc, phaseCond, mlir::ValueRange{});
    // Spin Wait - After block
    mlir::Block *afterBlock = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToEnd(afterBlock);
    auto five = builder.create<mlir::arith::ConstantIntOp>(loc, 5, 32);
    mlir::LLVM::createLLVMIntrinsicCallOp(builder, loc, "llvm.amdgcn.s.sleep",
                                          mlir::TypeRange{},
                                          mlir::ValueRange{five});
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{});
    builder.setInsertionPointAfter(whileOp);

    auto asmDialectAttr = mlir::LLVM::AsmDialectAttr::get(
        builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT);
    builder.create<mlir::LLVM::InlineAsmOp>(
        loc, mlir::TypeRange(), mlir::ValueRange(), "s_wakeup", "",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        mlir::LLVM::TailCallKind::None, asmDialectAttr, mlir::ArrayAttr());

    eraseOps.push_back(op);
  });
  for (auto op : eraseOps)
    op->erase();
}

class UTLXAMDLowerBarrierOpsPass
    : public mlir::PassWrapper<UTLXAMDLowerBarrierOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXAMDLowerBarrierOpsPass)

  llvm::StringRef getArgument() const override {
    return "utlx-amd-lower-barrier-ops";
  }
  llvm::StringRef getDescription() const override {
    return "Lower barrier ops to AMD-specific barrier ops";
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    std::map<mlir::triton::gpu::LocalAllocOp, int>
        localAllocToBarrierExpectedCount;
    lowerInitBarrierOps(m, localAllocToBarrierExpectedCount);
    lowerArriveBarrierOps(m, localAllocToBarrierExpectedCount);
    lowerWaitBarrierOps(m);
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createAMDLowerBarrierOpsPass() {
  return std::make_unique<UTLXAMDLowerBarrierOpsPass>();
}

void registerAMDLowerBarrierOpsPass() {
  mlir::PassRegistration<UTLXAMDLowerBarrierOpsPass>();
}

} // namespace utlx
