#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/PluginUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

// An mlir-opt-style driver for triton-ext extension passes. It resolves MLIR
// and the Triton dialect from the same libtriton the plugins link, so both see
// one TypeID per op and cast<triton::FuncOp> works across the driver/plugin
// boundary. It avoids MlirOptMain, which pulls in upstream dialects libtriton
// does not embed.

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));
static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));
static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into chunks on '// -----' and process "
                   "each independently"),
    llvm::cl::init(false));
static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match expected-* lines"),
    llvm::cl::init(false));

namespace {

// Load plugins from TRITON_PLUGIN_PATHS and register their passes and dialects.
void loadTritonExtPlugins(DialectRegistry &registry) {
  const char *paths = std::getenv("TRITON_PLUGIN_PATHS");
  if (!paths)
    return;
  std::string err;
  auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(paths, &err);
  if (!lib.isValid()) {
    llvm::errs() << "triton-ext-opt: cannot load plugin '" << paths
                 << "': " << err << "\n";
    return;
  }
  auto *getInfo = reinterpret_cast<triton::plugin::PluginInfo *(*)()>(
      lib.getAddressOfSymbol("tritonGetPluginInfo"));
  if (!getInfo) {
    llvm::errs() << "triton-ext-opt: '" << paths
                 << "' has no tritonGetPluginInfo symbol\n";
    return;
  }
  if (triton::plugin::PluginInfo *info = getInfo()) {
    for (size_t i = 0; i < info->numDialects; ++i)
      if (info->dialects[i].registerDialect)
        info->dialects[i].registerDialect(&registry);
    for (size_t i = 0; i < info->numPasses; ++i)
      if (info->passes[i].registerPass)
        info->passes[i].registerPass();
  }
}

void buildRegistry(DialectRegistry &registry) {
  registerPass([] { return createCanonicalizerPass(); });
  registerPass([] { return createCSEPass(); });

  registry.insert<triton::TritonDialect, triton::gpu::TritonGPUDialect,
                  triton::nvidia_gpu::TritonNvidiaGPUDialect,
                  triton::instrument::TritonInstrumentDialect,
                  triton::gluon::GluonDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, func::FuncDialect, gpu::GPUDialect,
                  math::MathDialect, scf::SCFDialect>();
  loadTritonExtPlugins(registry);
}

LogicalResult processChunk(std::unique_ptr<llvm::MemoryBuffer> chunk,
                           const PassPipelineCLParser &passPipeline,
                           const DialectRegistry &registry, raw_ostream &os) {
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(chunk), llvm::SMLoc());

  SourceMgrDiagnosticVerifierHandler verifyHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(!verifyDiagnostics);

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);

  auto run = [&]() -> LogicalResult {
    if (!module)
      return failure();
    PassManager pm(&context);
    auto errorHandler = [&](const llvm::Twine &msg) {
      emitError(UnknownLoc::get(&context)) << msg;
      return failure();
    };
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();
    return pm.run(*module);
  };

  if (verifyDiagnostics) {
    (void)run();
    return verifyHandler.verify();
  }
  if (failed(run()))
    return failure();
  module->print(os);
  os << "\n";
  return success();
}

} // namespace

int main(int argc, char **argv) {
  DialectRegistry registry;
  buildRegistry(registry);

  PassPipelineCLParser passPipeline("", "Pass pipeline to run");
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Triton extension optimizer driver\n");

  auto input = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!input) {
    llvm::errs() << "triton-ext-opt: cannot open " << inputFilename << ": "
                 << input.getError().message() << "\n";
    return 1;
  }

  std::error_code ec;
  auto output =
      llvm::ToolOutputFile(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "triton-ext-opt: " << ec.message() << "\n";
    return 1;
  }

  auto processOne = [&](std::unique_ptr<llvm::MemoryBuffer> buf,
                        raw_ostream &os) {
    return processChunk(std::move(buf), passPipeline, registry, os);
  };

  LogicalResult result = success();
  if (splitInputFile) {
    StringRef marker = "// -----";
    StringRef buffer = input.get()->getBuffer();
    bool first = true;
    while (true) {
      size_t pos = buffer.find(marker);
      StringRef chunk = buffer.substr(0, pos);
      if (!first)
        output.os() << "// -----\n";
      first = false;
      if (failed(processOne(llvm::MemoryBuffer::getMemBufferCopy(chunk),
                            output.os())))
        result = failure();
      if (pos == StringRef::npos)
        break;
      buffer = buffer.substr(pos + marker.size());
    }
  } else {
    result = processOne(std::move(input.get()), output.os());
  }

  if (failed(result))
    return 1;
  output.keep();
  return 0;
}
