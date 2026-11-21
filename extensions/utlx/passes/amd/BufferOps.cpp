/// Carry the amdg buffer ops uTLX creates at TTIR level through the stock
/// AMD TTGIR pipeline.
///
/// tlx.buffer_load, buffer_store and buffer_load_to_local lower to the amdg
/// buffer ops directly, as in the fork, so that the trusted `contiguity` and
/// the offsets' register layout reach the backend. Two stock passes need help.
///
/// Layout passes may rebuild a buffer op in another register layout (stock
/// Triton treats it as cheap to rematerialize), but `contiguity` is a promise
/// about the layout the kernel chose. The fork pins that layout
/// (tlx.preserve_layout). Here op creation records it in
/// `utlx.buffer_layout`, and utlx-keep-buffer-op-layout, run after the last
/// layout pass, rebuilds any op that moved in its recorded layout, converting
/// operands and results around it.
///
/// The fork's
/// CanonicalizePointers has patterns that rewrite those ops' scalar base when
/// it derives from a pointer argument the pass canonicalizes; stock Triton's
/// does not, and fails to legalize them. (Stock Triton only creates buffer
/// ops later, in ConvertToBufferOps.)
///
/// The pass collects the ops to rewrite by following pointer-typed values
/// from the arguments, so it stops at an integer:
///
/// - utlx-hide-buffer-op-bases, run directly before CanonicalizePointers,
///   routes each buffer op's base through tt.ptr_to_int / tt.int_to_ptr.
///   CanonicalizePointers then materializes the canonical pointer at the
///   ptr_to_int and leaves the buffer op alone.
/// - utlx-restore-buffer-op-bases, run directly after it, before anything
///   can fold the round trip, feeds the materialized pointer to the buffer op
///   again, so axis analysis sees the base as before.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kHiddenBaseAttr = "utlx.hidden_buffer_base";
// Set when ops/NewOps.cpp creates the op.
constexpr llvm::StringLiteral kBufferLayoutAttr = "utlx.buffer_layout";

// The buffer ops uTLX creates. Each takes exactly one scalar pointer, its base.
bool isUTLXBufferOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "amdg.buffer_load" || name == "amdg.buffer_store" ||
         name == "amdg.buffer_load_to_local";
}

mlir::OpOperand *getBaseOperand(mlir::Operation *op) {
  for (mlir::OpOperand &operand : op->getOpOperands())
    if (mlir::isa<tt::PointerType>(operand.get().getType()))
      return &operand;
  return nullptr;
}

// The offsets operand: the first tensor operand after the base.
mlir::Value getOffsets(mlir::Operation *op) {
  mlir::OpOperand *base = getBaseOperand(op);
  if (!base)
    return {};
  for (mlir::Value operand :
       op->getOperands().drop_front(base->getOperandNumber() + 1))
    if (mlir::isa<mlir::RankedTensorType>(operand.getType()))
      return operand;
  return {};
}

// Rebuild `op` with every tensor operand and result in `encoding`.
void rebuildInLayout(mlir::Operation *op, mlir::Attribute encoding) {
  mlir::OpBuilder b(op);
  mlir::Location loc = op->getLoc();
  auto relaid = [&](mlir::Type type) -> mlir::Type {
    if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type))
      return tensorTy.cloneWithEncoding(encoding);
    return type;
  };
  llvm::SmallVector<mlir::Value> operands;
  for (mlir::Value operand : op->getOperands()) {
    mlir::Type type = relaid(operand.getType());
    if (type != operand.getType())
      operand = ttg::ConvertLayoutOp::create(b, loc, type, operand);
    operands.push_back(operand);
  }
  llvm::SmallVector<mlir::Type> resultTypes;
  for (mlir::Type type : op->getResultTypes())
    resultTypes.push_back(relaid(type));
  mlir::Operation *newOp = b.create(loc, op->getName().getIdentifier(),
                                    operands, resultTypes, op->getAttrs());
  for (auto [oldResult, newResult] :
       llvm::zip(op->getResults(), newOp->getResults())) {
    mlir::Value replacement = newResult;
    if (newResult.getType() != oldResult.getType())
      replacement =
          ttg::ConvertLayoutOp::create(b, loc, oldResult.getType(), newResult);
    oldResult.replaceAllUsesWith(replacement);
  }
  op->erase();
}

class UTLXKeepBufferOpLayoutPass
    : public mlir::PassWrapper<UTLXKeepBufferOpLayoutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXKeepBufferOpLayoutPass)

  llvm::StringRef getArgument() const override {
    return "utlx-keep-buffer-op-layout";
  }
  llvm::StringRef getDescription() const override {
    return "Restore the register layout buffer op contiguity was given for";
  }

  void runOnOperation() override {
    llvm::SmallVector<std::pair<mlir::Operation *, mlir::Attribute>> moved;
    getOperation().walk([&](mlir::Operation *op) {
      mlir::Attribute encoding = op->getAttr(kBufferLayoutAttr);
      if (!encoding || !isUTLXBufferOp(op))
        return;
      mlir::Value offsets = getOffsets(op);
      auto offsetsTy =
          offsets ? mlir::dyn_cast<mlir::RankedTensorType>(offsets.getType())
                  : nullptr;
      if (offsetsTy && offsetsTy.getEncoding() != encoding)
        moved.emplace_back(op, encoding);
    });
    for (auto [op, encoding] : moved)
      rebuildInLayout(op, encoding);
  }
};

class UTLXHideBufferOpBasesPass
    : public mlir::PassWrapper<UTLXHideBufferOpBasesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXHideBufferOpBasesPass)

  llvm::StringRef getArgument() const override {
    return "utlx-hide-buffer-op-bases";
  }
  llvm::StringRef getDescription() const override {
    return "Route buffer op bases through an integer for CanonicalizePointers";
  }

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    auto i64 = builder.getI64Type();
    getOperation().walk([&](mlir::Operation *op) {
      if (!isUTLXBufferOp(op))
        return;
      mlir::OpOperand *base = getBaseOperand(op);
      if (!base)
        return;
      builder.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();
      auto asInt = builder.create<tt::PtrToIntOp>(loc, i64, base->get());
      auto asPtr =
          builder.create<tt::IntToPtrOp>(loc, base->get().getType(), asInt);
      asPtr->setAttr(kHiddenBaseAttr, builder.getUnitAttr());
      base->set(asPtr);
    });
  }
};

class UTLXRestoreBufferOpBasesPass
    : public mlir::PassWrapper<UTLXRestoreBufferOpBasesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UTLXRestoreBufferOpBasesPass)

  llvm::StringRef getArgument() const override {
    return "utlx-restore-buffer-op-bases";
  }
  llvm::StringRef getDescription() const override {
    return "Undo utlx-hide-buffer-op-bases";
  }

  void runOnOperation() override {
    llvm::SmallVector<tt::IntToPtrOp> hidden;
    getOperation().walk([&](tt::IntToPtrOp op) {
      if (op->hasAttr(kHiddenBaseAttr))
        hidden.push_back(op);
    });
    for (tt::IntToPtrOp asPtr : hidden) {
      auto asInt = asPtr.getSrc().getDefiningOp<tt::PtrToIntOp>();
      if (!asInt || asInt.getSrc().getType() != asPtr.getType()) {
        asPtr->removeAttr(kHiddenBaseAttr);
        continue;
      }
      asPtr.replaceAllUsesWith(asInt.getSrc());
      asPtr.erase();
      if (asInt->use_empty())
        asInt.erase();
    }
  }
};

} // namespace

namespace utlx {

std::unique_ptr<mlir::Pass> createKeepBufferOpLayoutPass() {
  return std::make_unique<UTLXKeepBufferOpLayoutPass>();
}

void registerKeepBufferOpLayoutPass() {
  mlir::PassRegistration<UTLXKeepBufferOpLayoutPass>();
}

std::unique_ptr<mlir::Pass> createHideBufferOpBasesPass() {
  return std::make_unique<UTLXHideBufferOpBasesPass>();
}

void registerHideBufferOpBasesPass() {
  mlir::PassRegistration<UTLXHideBufferOpBasesPass>();
}

std::unique_ptr<mlir::Pass> createRestoreBufferOpBasesPass() {
  return std::make_unique<UTLXRestoreBufferOpBasesPass>();
}

void registerRestoreBufferOpBasesPass() {
  mlir::PassRegistration<UTLXRestoreBufferOpBasesPass>();
}

} // namespace utlx
