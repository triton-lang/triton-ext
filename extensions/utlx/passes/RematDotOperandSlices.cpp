/// Compute values feeding a dot-operand convert directly in that layout.
///
/// A TLX kernel can ask for a computed tensor in a dot operand layout, e.g.
/// buffer_load offsets laid out so the loaded tile lands in MFMA registers:
///
///   offs = base + rows[:, None] * D + cols[None, :]
///   offs = tlx.require_layout(offs, dot_operand_layout(...))
///
/// Layout conversion gives the arithmetic the default blocked layout and
/// converts the result. Stock RemoveLayoutConversions never rematerializes a
/// convert to a dot operand layout backwards, so ReduceDataDuplication stages
/// it through shared memory -- for a 256x128 i32 tile, 128 KiB of it. Fork
/// TLX rematerializes such converts outside pipelined loops.
///
/// This pass handles the cheap and common case. It considers each convert to a
/// dot operand layout outside any loop whose backward slice consists only of
/// index arithmetic: constants, make_range, splat, expand_dims, broadcast,
/// pure elementwise ops and local_load. It clones that slice in the dot
/// operand layout and drops the convert. Anything else is left alone.

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

bool isIndexArithmetic(mlir::Operation *op) {
  if (isa<mlir::arith::ConstantOp, tt::MakeRangeOp, tt::SplatOp,
          tt::ExpandDimsOp, tt::BroadcastOp, ttg::ConvertLayoutOp>(op))
    return true;
  if (auto load = dyn_cast<ttg::LocalLoadOp>(op))
    return load.getSrc().getType().getRank() == load.getType().getRank();
  // Not op->hasTrait<OpTrait::Elementwise>(): libtriton and this plugin key
  // a template trait's TypeID by its compiler-spelled name, which differs
  // between e.g. a GCC-built plugin and a clang-built PyPI libtriton, so the
  // check is always false there. Ask libtriton, and list Triton's pure
  // elementwise ops, which lack the arith-style mappable traits.
  bool elementwise =
      mlir::OpTrait::hasElementwiseMappableTraits(op) ||
      isa<tt::AddPtrOp, tt::BitcastOp, tt::ClampFOp, tt::FpToFpOp,
          tt::IntToPtrOp, tt::MulhiUIOp, tt::PreciseDivFOp, tt::PreciseSqrtOp,
          tt::PtrToIntOp>(op);
  return elementwise && mlir::isMemoryEffectFree(op);
}

// Returns the slice feeding `convert` in topological order, or an empty set
// if it cannot be cloned in the target layout.
llvm::SetVector<mlir::Operation *>
getSlice(ttg::ConvertLayoutOp convert,
         llvm::DenseMap<mlir::Value, mlir::Attribute> &layout) {
  llvm::SetVector<mlir::Value> values;
  if (mlir::failed(mlir::getConvertBackwardSlice(
          convert.getSrcMutable(), values, convert.getType().getEncoding(),
          layout)))
    return {};
  llvm::SetVector<mlir::Operation *> ops;
  for (mlir::Value v : values) {
    mlir::Operation *def = v.getDefiningOp();
    if (!def || def->getNumResults() != 1 || !isIndexArithmetic(def))
      return {};
    ops.insert(def);
  }
  return mlir::topologicalSort(ops);
}

mlir::Operation *cloneWithLayout(mlir::OpBuilder &b, mlir::Operation *op,
                                 mlir::IRMapping &mapping,
                                 mlir::Attribute encoding) {
  auto oldTy = cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto newTy = oldTy.cloneWithEncoding(encoding);
  if (auto cst = dyn_cast<mlir::arith::ConstantOp>(op)) {
    auto value = dyn_cast<mlir::SplatElementsAttr>(cst.getValue());
    if (!value)
      return nullptr;
    return mlir::arith::ConstantOp::create(
        b, op->getLoc(), newTy,
        mlir::SplatElementsAttr::get(newTy,
                                     value.getSplatValue<mlir::Attribute>()));
  }
  mlir::Operation *clone = b.clone(*op, mapping);
  clone->getResult(0).setType(newTy);
  return clone;
}

bool rematerialize(ttg::ConvertLayoutOp convert) {
  llvm::DenseMap<mlir::Value, mlir::Attribute> layout;
  llvm::SetVector<mlir::Operation *> slice = getSlice(convert, layout);
  if (slice.empty())
    return false;
  // Non-splat constants would need their elements re-laid out; skip them.
  for (mlir::Operation *op : slice)
    if (auto cst = dyn_cast<mlir::arith::ConstantOp>(op))
      if (!isa<mlir::SplatElementsAttr>(cst.getValue()))
        return false;

  mlir::IRMapping mapping;
  mlir::OpBuilder b(convert.getContext());
  for (mlir::Operation *op : slice) {
    b.setInsertionPointAfter(op);
    mlir::Value result = op->getResult(0);
    mlir::Operation *clone = cloneWithLayout(b, op, mapping, layout[result]);
    mapping.map(result, clone->getResult(0));
  }
  convert.getResult().replaceAllUsesWith(mapping.lookup(convert.getSrc()));
  convert.erase();
  for (mlir::Operation *op : llvm::reverse(slice))
    if (mlir::isOpTriviallyDead(op))
      op->erase();
  return true;
}

class UTLXRematDotOperandSlicesPass
    : public mlir::PassWrapper<UTLXRematDotOperandSlicesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXRematDotOperandSlicesPass)

  llvm::StringRef getArgument() const override {
    return "utlx-remat-dot-operand-slices";
  }
  llvm::StringRef getDescription() const override {
    return "Compute index arithmetic feeding a dot-operand convert directly "
           "in that layout";
  }

  void runOnOperation() override {
    llvm::SmallVector<ttg::ConvertLayoutOp> candidates;
    getOperation().walk([&](ttg::ConvertLayoutOp convert) {
      // Inside a loop the convert may be what the pipeliner hoists.
      if (isa<ttg::DotOperandEncodingAttr>(convert.getType().getEncoding()) &&
          !convert->getParentOfType<mlir::LoopLikeOpInterface>())
        candidates.push_back(convert);
    });
    for (ttg::ConvertLayoutOp convert : candidates)
      rematerialize(convert);
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createRematDotOperandSlicesPass() {
  return std::make_unique<UTLXRematDotOperandSlicesPass>();
}

void registerRematDotOperandSlicesPass() {
  mlir::PassRegistration<UTLXRematDotOperandSlicesPass>();
}

} // namespace utlx
