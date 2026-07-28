#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/PluginUtils.h"

// Test pass that does cast<triton::FuncOp>, which only succeeds when the driver
// and plugin share libtriton's TypeID for the op. Tags each tt.func with an
// attribute so a lit test can check the cast ran.

using namespace mlir;

namespace {

struct TypeIDCastPass
    : public PassWrapper<TypeIDCastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeIDCastPass)

  StringRef getArgument() const override { return "test-triton-typeid-cast"; }
  StringRef getDescription() const override {
    return "Test: cast<triton::FuncOp> across the driver/plugin boundary";
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() != "tt.func")
        return;
      // The cross-boundary identity check. Aborts under a driver that does not
      // share libtriton's TypeID copy.
      auto func = dyn_cast<triton::FuncOp>(op);
      if (!func) {
        op->emitError("triton::FuncOp identity mismatch: cast failed "
                      "(TypeID conflict)");
        signalPassFailure();
        return;
      }
      func->setAttr("triton_ext.typeid_cast_ok", UnitAttr::get(&getContext()));
    });
  }
};

} // namespace

static void addTypeIDCastPass(PassManager *pm,
                              const std::vector<std::string> &) {
  pm->addPass(std::make_unique<TypeIDCastPass>());
}

static void registerTypeIDCastPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TypeIDCastPass>();
  });
}

using namespace mlir::triton;

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo pass = {"test-triton-typeid-cast", "0.1.0",
                                  addTypeIDCastPass, registerTypeIDCastPass};
  static plugin::PassInfo passes[] = {pass};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION,
                                    "TypeIDCastTestPlugin",
                                    "0.1.0",
                                    passes,
                                    1,
                                    nullptr,
                                    0,
                                    nullptr,
                                    0,
                                    TRITON_VERSION};
  return &info;
}
