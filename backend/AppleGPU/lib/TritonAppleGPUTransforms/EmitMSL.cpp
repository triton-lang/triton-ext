// The `emit-msl` pass: runs the emitter in lib/TritonAppleGPUToMSL/, records
// the launch facts the host needs and writes the text out.

#include "../TritonAppleGPUToMSL/AgpuEmitter.h"
#include "TritonAppleGPUToMSL/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>

#include <string>

using namespace mlir;

namespace mlir::triton::applegpu {

namespace {
class EmitMSLPass : public PassWrapper<EmitMSLPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitMSLPass)

  EmitMSLPass() = default;
  explicit EmitMSLPass(std::string outPath) : outPath(std::move(outPath)) {}
  EmitMSLPass(const EmitMSLPass &other)
      : PassWrapper(other), outPath(other.outPath) {}

  StringRef getArgument() const final { return "emit-msl"; }
  StringRef getDescription() const final {
    return "Emit MSL source from TritonGPU IR";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    std::string msl;
    llvm::raw_string_ostream ss(msl);

    bridge::AgpuEmitter emitter(mod, ss);
    if (failed(emitter.emit())) {
      signalPassFailure();
      return;
    }
    ss.flush();

    // Tells the host launcher whether the whole grid must be resident at once.
    const agpu::GridResidency residency =
        agpu::residencyFor(bridge::launchFactsOf(mod));
    mod->setAttr(
        agpu::kGridResidencyAttr,
        IntegerAttr::get(IntegerType::get(mod.getContext(), 1),
                         residency == agpu::GridResidency::CoResident));

    if (outPath.empty()) {
      llvm::errs() << msl;
      return;
    }
    std::error_code ec;
    llvm::raw_fd_ostream out(outPath, ec);
    if (ec) {
      mod.emitError("EmitMSL: cannot open '" + outPath + "': " + ec.message());
      signalPassFailure();
      return;
    }
    out << msl;

    // ~raw_fd_ostream aborts on an unclaimed error.
    if (const std::error_code werr = out.error()) {
      out.clear_error();
      mod.emitError("EmitMSL: cannot write '" + outPath +
                    "': " + werr.message());
      signalPassFailure();
    }
  }

private:
  std::string outPath;
};

} // namespace

std::unique_ptr<mlir::Pass> createEmitMSLPass(std::string outPath) {
  return std::make_unique<EmitMSLPass>(std::move(outPath));
}

} // namespace mlir::triton::applegpu
