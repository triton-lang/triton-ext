// The `emit-msl` pass: runs the emitter in lib/TritonAppleGPUToMSL/, records
// the launch facts the host needs and writes the text out.

#include "../TritonAppleGPUToMSL/AgpuEmitter.h"
#include "TritonAppleGPUToMSL/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <map>
#include <set>

#include <string>

using namespace mlir;

namespace mlir::triton::applegpu {

namespace {

// MSL_ENABLE_DUMP is the MSL counterpart of Triton's MLIR_ENABLE_DUMP, and
// lands on the same stream MLIR_ENABLE_DUMP's llvm::dbgs() does, so the two
// interleave in pipeline order. Read with getenv because
// triton::tools::getBoolEnv asserts against a whitelist in libtriton's own
// header, which an out-of-tree backend cannot add to. MLIR_DUMP_PATH does
// not redirect this one; TRITON_MSL_DUMP writes the MSL to a file.
bool mslDumpEnabled() {
  const char *v = std::getenv("MSL_ENABLE_DUMP");
  if (!v)
    return false;
  const StringRef s = StringRef(v).trim();
  return !s.empty() && !s.equals_insensitive("0") &&
         !s.equals_insensitive("false") && !s.equals_insensitive("off") &&
         !s.equals_insensitive("n") && !s.equals_insensitive("no");
}

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
    RewritePatternSet expand(mod.getContext());
    arith::populateCeilFloorDivExpandOpsPatterns(expand);
    if (failed(applyPatternsGreedily(mod, std::move(expand)))) {
      signalPassFailure();
      return;
    }

    std::string msl;
    llvm::raw_string_ostream ss(msl);

    bridge::AgpuEmitter emitter(mod, ss);
    if (failed(emitter.emit())) {
      signalPassFailure();
      return;
    }
    ss.flush();

    // Tells the host launcher whether the whole grid must be resident at once,
    // and which buffers it must make resident.
    const agpu::LaunchFacts facts = bridge::launchFactsOf(mod);
    const auto flag = [&](const char *name, bool value) {
      mod->setAttr(
          name, IntegerAttr::get(IntegerType::get(mod.getContext(), 1), value));
    };
    flag(agpu::kGridResidencyAttr,
         agpu::residencyFor(facts) == agpu::GridResidency::CoResident);
    flag(agpu::kExposesAddressesAttr, facts.exposesAddresses);
    flag(agpu::kReadsAddressesAttr, facts.readsAddresses);

    if (mslDumpEnabled())
      llvm::errs() << "// -----// MSL Dump After EmitMSL "
                      "('builtin.module' operation) //----- //\n"
                   << msl;

    if (outPath.empty())
      return;
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
