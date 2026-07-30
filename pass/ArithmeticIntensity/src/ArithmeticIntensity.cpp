//===- ArithmeticIntensity.cpp - Kernel arithmetic-intensity analysis -===//
//
// Part of the Triton extensions project (text.0).
//
//===---------------------------------------------------------------------===//
//
// `-triton-arithmetic-intensity` annotates each `tt.func` argument with two
// string attributes describing the work done by the kernel against that
// argument:
//
//   - `tt.bandwidth`: an algebraic equation for the total bytes moved per
//     CTA across all loads/stores rooted at this argument.
//   - `tt.compute`:   an algebraic equation for the total FLOPs feeding the
//     stores rooted at this argument.
//
// ## Algorithm
//
// For each function:
//   1. Walk every block once and classify each op's contribution as
//      Load / Store / Compute / Other (see `BlockMetrics::calculateMetric`).
//   2. For every load/store, trace the address back to its originating
//      function-argument via `findPointerParam`, then aggregate the metric
//      bottom-up to function scope, multiplying by the symbolic trip count
//      of each enclosing `scf.for` and approximating `scf.if` by the
//      then-branch contribution.
//   3. Write the resulting `AffineExpr` back to the `tt.func` argument as
//      a string attribute.
//
// ## Symbolic equations
//
// Equations are built as MLIR `AffineExpr` over a per-function symbol
// table (`SymTable`) that maps each "leaf" `Value` (function arg, program
// id, num programs, opaque integer producer) to an `AffineSymbolExpr`, so
// constant folding, canonicalization, and `simplifyAffineExpr` come for
// free. At print time the simplified expression is post-processed:
// `s<N>` symbol tokens are rewritten to source-level names
// (`args[N]`, `program_id[N]`, `num_programs[N]`), and the affine
// printer's `floordiv`/`mod` keywords are rewritten to `/` and `%`.
// `ceildiv` is not handled: the pass never produces it and
// `simplifyAffineExpr` does not introduce it.
//
// ## Metric model
//
//   - Bytes (not element counts) are computed as
//     `numElements(type) * elementBits / 8`, rounded up to a whole byte so
//     sub-byte element types (e.g. `i1`) are accounted for in aggregate.
//   - `tt.dot` contributes `M * N * K * 2` FLOPs per block.
//   - Elementwise / reduce / `tt.addptr` ops contribute one op per output
//     element.
//
// ## Control flow
//
//   - `scf.for` trip count = `(upper - lower) floordiv step`.
//   - `scf.for` iter_args are substituted with their init value (exact
//     when the iter_arg is loop-invariant, safe otherwise).
//   - `scf.for` induction variables are substituted with the loop's upper
//     bound, giving a conservative upper-bound estimate when an inner
//     loop's bound references an outer IV.
//   - `scf.if` results are approximated by the then-branch yield.
//     `AffineExpr` has no `max(...)`; expressing that would need a
//     structured attribute.
//   - Unrecognised integer producers fall back to a fresh opaque symbol so
//     downstream arithmetic still produces a well-formed equation rather
//     than crashing the pass.
//
// ## Pointer-arg detection
//
// `findPointerParam` accepts scalar `!tt.ptr<>`, tensor-of-pointer
// function args (`tensor<NxMx!tt.ptr<>>`), and `!tt.tensordesc<>` args,
// and walks back through `tt.addptr`, `tt.splat`, and `scf.for` iter_args
// to find the originating function block argument.
//
// ## Store metric
//
// `tt.store` / `tt.descriptor_store` have no SSA results, so they cannot
// be keyed in the per-block `metricsMap` by `Value`;
// `BlockMetrics::getStoreOpMetric(Operation*)` builds the Store metric
// directly from the stored-value type at the point of use.
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/StrUtil.h"

#include <cctype>

#define DEBUG_TYPE "triton-arithmetic-intensity"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONARITHMETICINTENSITY
#include "Passes.h.inc"
} // namespace mlir::triton

namespace {

////////////////////////////////////////////////////////////////////////////////
// Metric class
//
// Holds the metric for a single value, including the kind of operation, the
// size of the operation, and the element type of the operation.
////////////////////////////////////////////////////////////////////////////////
class Metric {
public:
  enum MetricKind {
    Load,
    Store,
    Compute,
    Other,
  };

  Metric(MetricKind kind = MetricKind::Other, int64_t size = 0,
         Type elementType = nullptr)
      : kind(kind), size(size), elementType(elementType) {}

  MetricKind getKind() const { return kind; }
  int64_t getSize() const { return size; }
  Type getElementType() const { return elementType; }

  void addSize(int64_t sz) { size += sz; }
  void setElementType(Type etype) { elementType = etype; }

  std::string getKindString() const {
    switch (kind) {
    case MetricKind::Load:
      return "Load";
    case MetricKind::Store:
      return "Store";
    case MetricKind::Compute:
      return "Compute";
    }
    return "Other";
  }

private:
  MetricKind kind;
  int64_t size;
  Type elementType;
};

////////////////////////////////////////////////////////////////////////////////
// BlockMetrics class
//
// Holds the metrics for a single block, including the load and store ops,
// the metrics map, and the result metrics.
////////////////////////////////////////////////////////////////////////////////
class BlockMetrics {

  bool isLoadLikeOp(Operation *op) const {
    return isa<triton::LoadOp, triton::DescriptorLoadOp>(op);
  }

  bool isStoreLikeOp(Operation *op) const {
    return isa<triton::StoreOp, triton::DescriptorStoreOp>(op);
  }

  int64_t getNumElements(Type type) const {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
      auto elementSize = getNumElements(rankedType.getElementType());
      return rankedType.getNumElements() * elementSize;
    } else if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      return 1; // elements of pointee?
    } else if (auto tensorType = dyn_cast<triton::TensorDescType>(type)) {
      return getNumElements(tensorType.getBlockType());
    } else if (auto vectorType = dyn_cast<VectorType>(type)) {
      auto elementSize = getNumElements(vectorType.getElementType());
      return vectorType.getNumElements() * elementSize;
    }
    return 1;
  }

  Type getElementType(Type type) const {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
      return getElementType(rankedType.getElementType());
    } else if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      // should be int64_t for pointer
      return getElementType(ptrType.getPointeeType());
    } else if (auto vectorType = dyn_cast<VectorType>(type)) {
      return getElementType(vectorType.getElementType());
    } else if (auto tensorType = dyn_cast<triton::TensorDescType>(type)) {
      return getElementType(tensorType.getBlockType());
    }
    return type;
  }

  // Total number of bytes transferred when accessing a value of `type`.
  // Computed as `getNumElements(type) * bits_per_element`, rounded up to a
  // whole number of bytes so that sub-byte element types (e.g. `i1`) are
  // accounted for correctly in aggregate.
  int64_t getNumBytes(Type type) const {
    int64_t numElements = getNumElements(type);
    Type elemType = getElementType(type);
    int64_t bitsPerElement = 8;
    if (elemType.isIntOrFloat()) {
      bitsPerElement = elemType.getIntOrFloatBitWidth();
    } else if (isa<triton::PointerType>(elemType)) {
      bitsPerElement = 64;
    }
    return (numElements * bitsPerElement + 7) / 8;
  }

  Metric calculateMetric(Value value) {
    auto *op = value.getDefiningOp();
    if (isLoadLikeOp(op)) {
      auto type = op->getResult(0).getType();
      return Metric(Metric::MetricKind::Load, getNumBytes(type),
                    getElementType(type));
    } else if (isStoreLikeOp(op)) {
      auto type = op->getOperand(1).getType();
      return Metric(Metric::MetricKind::Store, getNumBytes(type),
                    getElementType(type));
    } else if (isa<scf::YieldOp>(op)) {
      return Metric();
    } else if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // FLOPS = M * N * K * 2
      auto aType = cast<RankedTensorType>(dotOp.getA().getType());
      auto K = aType.getShape().back();
      auto cSize = getNumElements(dotOp.getC().getType());
      auto flops = cSize * K * 2;
      return Metric(Metric::MetricKind::Compute, flops,
                    getElementType(value.getType()));
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      // Approximation: FLOPS = sum of input sizes
      // TODO: improve this
      int64_t flops = 0;
      for (auto inputTy : reduceOp.getInputTypes()) {
        flops += getNumElements(inputTy);
      }
      return Metric(Metric::MetricKind::Compute, flops,
                    getElementType(value.getType()));
    } else if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
      auto type = addPtrOp.getOffset().getType();
      return Metric(Metric::MetricKind::Compute, getNumElements(type),
                    getElementType(type));
    } else if (isa<triton::SplatOp, triton::BroadcastOp, triton::MakeRangeOp,
                   triton::ExpandDimsOp, triton::GetProgramIdOp,
                   triton::TransOp, triton::ReshapeOp>(op)) {
      return Metric(Metric::MetricKind::Compute, 0,
                    getElementType(value.getType()));
    } else if (isa<arith::SelectOp>(op)) {
      return Metric(Metric::MetricKind::Compute);
    } else if (op->hasTrait<OpTrait::Elementwise>()) {
      auto flops = getNumElements(value.getType());
      return Metric(Metric::MetricKind::Compute, flops,
                    getElementType(value.getType()));
    } else if (isa<arith::ConstantOp>(op)) {
      return Metric();
    } else if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op)) {
      return Metric();
    } else {
      LDBG("Value is not a dot or elementwise operation: " << value);
      auto flops = getNumElements(value.getType());
      return Metric(Metric::MetricKind::Compute, flops,
                    getElementType(value.getType()));
    }
  }

public:
  BlockMetrics(Block *block) : block(block) {
    for (auto &op : *block) {
      for (auto result : op.getResults()) {
        metricsMap.try_emplace(result, calculateMetric(result));
      }
      if (isLoadLikeOp(&op)) {
        loadOps.push_back(&op);
      } else if (isStoreLikeOp(&op)) {
        storeOps.push_back(&op);
      }
    }
  }

  std::optional<Metric> getMetric(Value value) const {
    auto it = metricsMap.find(value);
    if (it != metricsMap.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  // Build a Store metric for `storeOp`. `tt.store` has no results, so its
  // metric is not recorded in `metricsMap` by the constructor; the driver
  // calls this helper at the point of use instead.
  Metric getStoreOpMetric(Operation *storeOp) const {
    assert(isStoreLikeOp(storeOp) && "expected a store-like op");
    auto type = storeOp->getOperand(1).getType();
    return Metric(Metric::MetricKind::Store, getNumBytes(type),
                  getElementType(type));
  }

  const SmallVector<Operation *> &getLoadOps() const { return loadOps; }
  const SmallVector<Operation *> &getStoreOps() const { return storeOps; }

  Metric calculateChainMetric(Value value, DenseSet<Value> &visited,
                              SmallVector<Value> &edges) const {
    if (visited.contains(value)) {
      return Metric();
    }
    visited.insert(value);
    auto mval = getMetric(value);
    if (!mval || mval->getKind() != Metric::MetricKind::Compute) {
      edges.push_back(value);
      return Metric();
    }
    Metric totalMetric = *mval;
    for (auto operand : value.getDefiningOp()->getOperands()) {
      totalMetric.addSize(
          calculateChainMetric(operand, visited, edges).getSize());
    }
    return totalMetric;
  }

  void dump() const {
    llvm::errs() << "Block: ----------------------------------------------\n";
    llvm::errs() << "Block: " << *block << "\n";
    llvm::errs() << "Load Ops: " << loadOps.size() << "\n";
    llvm::errs() << "Store Ops: " << storeOps.size() << "\n";
    for (auto &metric : metricsMap) {
      llvm::errs() << "Metric: type= " << metric.second.getKindString()
                   << ", size= " << metric.second.getSize()
                   << ", elementType= " << metric.second.getElementType()
                   << "\n";
      if (metric.first.getDefiningOp()->getNumRegions() > 0) {
        llvm::errs() << "  - Value: " << metric.first.getDefiningOp()->getName()
                     << "\n";
      } else {
        llvm::errs() << "  - Value: " << metric.first << "\n";
      }
    }
  }

private:
  Block *block;
  SmallVector<Operation *> loadOps;
  SmallVector<Operation *> storeOps;
  Operation *yieldOp;
  DenseMap<Value, Metric> metricsMap;
  SmallVector<Metric> resultMetrics;
};

////////////////////////////////////////////////////////////////////////////////
// SymTable
//
// Holds the mapping from "leaf" `Value`s (function arguments, program IDs,
// opaque integer producers) to AffineSymbolExpr indices for a single function,
// and knows how to serialise an `AffineExpr` built over those symbols back to
// a human-readable string with source-level names.
////////////////////////////////////////////////////////////////////////////////
class SymTable {
public:
  SymTable(triton::FuncOp func)
      : func(func), ctx(func.getContext()),
        entryBlock(&func.getBody().front()) {}

  MLIRContext *getContext() const { return ctx; }

  AffineExpr get(Value v) {
    auto [it, inserted] = indexByValue.try_emplace(v, values.size());
    if (inserted)
      values.push_back(v);
    return getAffineSymbolExpr(it->second, ctx);
  }

  AffineExpr constant(int64_t c) const { return getAffineConstantExpr(c, ctx); }

  // Simplify and serialise `expr`, substituting `s<i>` tokens with the
  // source-level name of the value at index `i` and rewriting affine-printer
  // keywords (`floordiv`, `mod`) to `/` and `%`.
  std::string print(AffineExpr expr) const {
    if (!expr)
      return "";
    AffineExpr simplified = simplifyAffineExpr(expr, /*numDims=*/0,
                                               /*numSymbols=*/values.size());
    std::string raw;
    {
      llvm::raw_string_ostream os(raw);
      simplified.print(os);
    }
    return rewrite(raw);
  }

private:
  std::string nameForSymbol(unsigned idx) const {
    Value v = values[idx];
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      if (blockArg.getOwner() == entryBlock)
        return "args[" + std::to_string(blockArg.getArgNumber()) + "]";
    }
    if (auto *op = v.getDefiningOp()) {
      if (auto p = dyn_cast<triton::GetProgramIdOp>(op))
        return "program_id[" + std::to_string(p.getAxisAsInt()) + "]";
      if (auto p = dyn_cast<triton::GetNumProgramsOp>(op))
        return "num_programs[" + std::to_string(p.getAxisAsInt()) + "]";
    }
    return "s" + std::to_string(idx);
  }

  std::string rewrite(StringRef raw) const {
    auto isIdent = [](char c) {
      return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0, n = raw.size(); i < n;) {
      bool atBoundary = (i == 0) || !isIdent(raw[i - 1]);
      if (atBoundary) {
        auto tryKeyword = [&](StringRef kw, StringRef repl) {
          if (raw.substr(i, kw.size()) != kw)
            return false;
          if (i + kw.size() < n && isIdent(raw[i + kw.size()]))
            return false;
          out.append(repl.begin(), repl.end());
          i += kw.size();
          return true;
        };
        if (tryKeyword("floordiv", "/"))
          continue;
        // TODO: re-enable this when we have a way to represent ceildiv
        // if (tryKeyword("ceildiv", "/"))
        //  continue;
        if (tryKeyword("mod", "%"))
          continue;
        if (raw[i] == 's' && i + 1 < n &&
            std::isdigit(static_cast<unsigned char>(raw[i + 1]))) {
          size_t j = i + 1;
          unsigned idx = 0;
          while (j < n && std::isdigit(static_cast<unsigned char>(raw[j]))) {
            idx = idx * 10 + (raw[j] - '0');
            ++j;
          }
          if (j == n || !isIdent(raw[j])) {
            if (idx < values.size()) {
              out += nameForSymbol(idx);
              i = j;
              continue;
            }
          }
        }
      }
      out += raw[i++];
    }
    return out;
  }

  triton::FuncOp func;
  MLIRContext *ctx;
  Block *entryBlock;
  DenseMap<Value, unsigned> indexByValue;
  SmallVector<Value> values;
};

////////////////////////////////////////////////////////////////////////////////
// ArithmeticIntensityAnalysisDriver class
//
// Performs the arithmetic intensity analysis for a single function.
////////////////////////////////////////////////////////////////////////////////
class ArithmeticIntensityAnalysisDriver {

  // True for function arguments that can serve as the base of a memory
  // access: scalar pointers, tensors of pointers, or tensor descriptors.
  static bool isPointerLikeFuncArgType(Type type) {
    if (auto rtt = dyn_cast<RankedTensorType>(type))
      type = rtt.getElementType();
    return isa<triton::PointerType, triton::TensorDescType>(type);
  }

  BlockArgument findPointerParam(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (auto funcOp = dyn_cast<triton::FuncOp>(parentOp)) {
        assert(funcOp == func && "Expected function argument");
        if (isPointerLikeFuncArgType(blockArg.getType()))
          return blockArg;
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        unsigned numIVs = forOp.getNumInductionVars();
        unsigned argNumber = blockArg.getArgNumber();
        if (argNumber < numIVs) {
          // Induction variables are integer-typed, never pointers.
          return BlockArgument();
        }
        return findPointerParam(forOp.getInitArgs()[argNumber - numIVs]);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        assert(false && "Not implemented");
      } else {
        LDBG("Unsupported operation: " << parentOp->getName());
      }
    } else {
      auto defOp = value.getDefiningOp();
      for (auto operand : defOp->getOperands()) {
        auto blockArg = findPointerParam(operand);
        if (blockArg) {
          return blockArg;
        }
      }
    }
    return BlockArgument();
  }

  // Resolve `value` to an AffineExpr over symbolic leaves (function block
  // arguments, program_id / num_programs results, opaque integer producers).
  //
  // Approximations:
  //  - Loop induction variables are substituted with their upper bound so
  //    that any expression containing one becomes a conservative
  //    upper-bound estimate. This lets nested loops whose bounds depend on
  //    an outer IV produce a symbolic equation rather than crashing.
  //  - Loop-carried iter_args are substituted with their init value. This
  //    is exact when the iter_arg's symbolic value is invariant across
  //    iterations (common for shape / size / bound bookkeeping), and a
  //    safe approximation otherwise.
  //  - scf.if results are approximated by the then-branch's yielded value.
  //    AffineExpr does not represent max(...); a future structured
  //    attribute (#tt.density<...>) can express this directly.
  //  - Unrecognised integer producers are bound to a fresh opaque symbol so
  //    that downstream multiplication / addition still produces a
  //    well-formed equation rather than crashing the pass.
  AffineExpr getSymbolicValue(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<triton::FuncOp>(parentOp))
        return syms.get(value);
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        unsigned numIVs = forOp.getNumInductionVars();
        unsigned argNumber = blockArg.getArgNumber();
        if (argNumber < numIVs) {
          LDBG("Substituting induction variable with upper bound (worst "
               "case): "
               << value);
          return getSymbolicValue(forOp.getUpperBound());
        }
        return getSymbolicValue(forOp.getInitArgs()[argNumber - numIVs]);
      }
      LDBG("Treating block arg with unsupported parent as opaque symbol: "
           << value);
      return syms.get(value);
    }
    auto *defOp = value.getDefiningOp();
    if (auto constant = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constant.getValueAttr()))
        return syms.constant(intAttr.getInt());
      LDBG("Treating non-integer constant as opaque symbol: " << value);
      return syms.get(value);
    }
    if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(defOp))
      return syms.get(value);
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      unsigned resultIdx = cast<OpResult>(value).getResultNumber();
      auto *yieldOp = forOp.getBody()->getTerminator();
      return getSymbolicValue(yieldOp->getOperand(resultIdx));
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      unsigned resultIdx = cast<OpResult>(value).getResultNumber();
      auto *thenYield = ifOp.thenBlock()->getTerminator();
      return getSymbolicValue(thenYield->getOperand(resultIdx));
    }
    if (defOp->getNumOperands() == 2) {
      auto lhs = getSymbolicValue(defOp->getOperand(0));
      auto rhs = getSymbolicValue(defOp->getOperand(1));
      if (isa<arith::AddIOp>(defOp))
        return lhs + rhs;
      if (isa<arith::SubIOp>(defOp))
        return lhs - rhs;
      if (isa<arith::MulIOp>(defOp))
        return lhs * rhs;
      if (isa<arith::DivSIOp, arith::DivUIOp>(defOp))
        return lhs.floorDiv(rhs);
      if (isa<arith::RemSIOp, arith::RemUIOp>(defOp))
        return lhs % rhs;
    }
    LDBG("Treating unsupported integer producer as opaque symbol: " << *defOp);
    return syms.get(value);
  }

  AffineExpr getSymbolicIterations(scf::ForOp forOp) {
    auto upperBound = getSymbolicValue(forOp.getUpperBound());
    auto lowerBound = getSymbolicValue(forOp.getLowerBound());
    auto step = getSymbolicValue(forOp.getStep());
    return (upperBound - lowerBound).floorDiv(step);
  }

  AffineExpr calculateBandwidth(Operation *op, AffineExpr size) {
    auto parentOp = op->getParentOp();
    if (isa<FunctionOpInterface>(parentOp))
      return size;
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      size = getSymbolicIterations(forOp) * size;
    } else if (isa<scf::IfOp>(parentOp)) {
      // scf.if: keep `size` as the then-branch contribution. AffineExpr has
      // no max(); see SymTable header comment.
    } else {
      LDBG("Unsupported parent op: " << parentOp->getName());
    }
    return calculateBandwidth(parentOp, size);
  }

  AffineExpr calculateCompute(Value value, DenseSet<Value> &visited,
                              SmallVector<Value> &edges) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      if (auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        unsigned numIVs = forOp.getNumInductionVars();
        unsigned argNumber = blockArg.getArgNumber();
        if (argNumber >= numIVs)
          edges.push_back(forOp.getInitArgs()[argNumber - numIVs]);
        // else: induction variable; no compute contribution.
      } else {
        assert(isa<FunctionOpInterface>(blockArg.getOwner()->getParentOp()) &&
               "Expected function argument");
      }
      return AffineExpr();
    }

    auto *defOp = value.getDefiningOp();
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      unsigned idx = cast<OpResult>(value).getResultNumber();
      auto *yieldOp = forOp.getBody()->getTerminator();
      edges.push_back(yieldOp->getOperand(idx));
      return AffineExpr();
    }
    if (isa<triton::ReduceOp>(defOp)) {
      // Treat like a normal compute op (handled by the chain walk below).
    } else if (defOp->getNumRegions() > 0) {
      LDBG("Unsupported region-bearing op in compute chain: " << *defOp);
      return AffineExpr();
    }

    auto &blockMetrics = metrics.at(defOp->getBlock());
    auto mval = blockMetrics.calculateChainMetric(value, visited, edges);
    return syms.constant(mval.getSize());
  }

  AffineExpr calculateCompute(Value value, DenseSet<Value> &visited) {
    SmallVector<Value> edges;
    AffineExpr computeSize = calculateCompute(value, visited, edges);
    auto *valueOp = value.getDefiningOp();
    if (computeSize && valueOp != nullptr)
      computeSize = calculateBandwidth(valueOp, computeSize);
    for (auto edge : edges) {
      AffineExpr edgeMetric = calculateCompute(edge, visited);
      if (edgeMetric)
        computeSize = computeSize ? computeSize + edgeMetric : edgeMetric;
    }
    return computeSize;
  }

public:
  ArithmeticIntensityAnalysisDriver(triton::FuncOp func)
      : func(func), syms(func), bandwidthMetrics(func.getNumArguments()),
        computeMetrics(func.getNumArguments()) {}

  void run() {
    func.walk<WalkOrder::PostOrder>(
        [&](Block *block) { metrics.try_emplace(block, block); });

    auto add = [](AffineExpr &acc, AffineExpr addend) {
      acc = acc ? acc + addend : addend;
    };

    for (auto &[block, blockMetrics] : metrics) {
      for (auto *loadOp : blockMetrics.getLoadOps()) {
        auto param = findPointerParam(loadOp->getOperand(0));
        if (!param) {
          LDBG("Skipping load with no resolvable function-arg base: "
               << *loadOp);
          continue;
        }
        auto metric = blockMetrics.getMetric(loadOp->getResult(0));
        if (metric) {
          AffineExpr total =
              calculateBandwidth(loadOp, syms.constant(metric->getSize()));
          add(bandwidthMetrics[param.getArgNumber()], total);
        }
      }
      for (auto *storeOp : blockMetrics.getStoreOps()) {
        auto param = findPointerParam(storeOp->getOperand(0));
        if (!param) {
          LDBG("Skipping store with no resolvable function-arg base: "
               << *storeOp);
          continue;
        }
        Metric storeMetric = blockMetrics.getStoreOpMetric(storeOp);
        AffineExpr total =
            calculateBandwidth(storeOp, syms.constant(storeMetric.getSize()));
        add(bandwidthMetrics[param.getArgNumber()], total);
        assert(!computeMetrics[param.getArgNumber()]);
        DenseSet<Value> visited;
        AffineExpr compute = calculateCompute(storeOp->getOperand(1), visited);
        if (compute)
          add(computeMetrics[param.getArgNumber()], compute);
      }
    }
    LLVM_DEBUG(dump());
  }

  std::optional<std::string> getBandwidthMetric(unsigned index) const {
    if (index >= bandwidthMetrics.size() || !bandwidthMetrics[index])
      return std::nullopt;
    return syms.print(bandwidthMetrics[index]);
  }
  std::optional<std::string> getComputeMetric(unsigned index) const {
    if (index >= computeMetrics.size() || !computeMetrics[index])
      return std::nullopt;
    return syms.print(computeMetrics[index]);
  }

  void dump() {
    llvm::errs() << "Arithmetic Intensity Analysis Driver: "
                    "----------------------------------------------\n";
    llvm::errs() << "Function: " << func.getName() << "\n";
    for (auto [block, blockMetrics] : metrics)
      blockMetrics.dump();
    llvm::errs() << "Bandwidth Metrics: " << bandwidthMetrics.size() << "\n";
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      llvm::errs() << "Bandwidth Metric: index= " << i << ", size= "
                   << (bandwidthMetrics[i] ? syms.print(bandwidthMetrics[i])
                                           : std::string("<none>"))
                   << "\n";
    }
    llvm::errs() << "Compute Metrics: " << computeMetrics.size() << "\n";
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      llvm::errs() << "Compute Metric: index= " << i << ", size= "
                   << (computeMetrics[i] ? syms.print(computeMetrics[i])
                                         : std::string("<none>"))
                   << "\n";
    }
  }

private:
  triton::FuncOp func;
  SymTable syms;
  DenseMap<Block *, BlockMetrics> metrics;
  SmallVector<AffineExpr> bandwidthMetrics;
  SmallVector<AffineExpr> computeMetrics;
};

////////////////////////////////////////////////////////////////////////////////
// Pass ArithmeticIntensity
////////////////////////////////////////////////////////////////////////////////
struct ArithmeticIntensityPass
    : public triton::impl::TritonArithmeticIntensityBase<
          ArithmeticIntensityPass> {
  using TritonArithmeticIntensityBase::TritonArithmeticIntensityBase;

  // TODO: get callgraph (see Analysis/Allocation.h)
  void runOnOperation() override {
    for (auto func : getOperation().getOps<triton::FuncOp>()) {
      ArithmeticIntensityAnalysisDriver driver(func);
      driver.run();
      for (unsigned i = 0; i < func.getNumArguments(); i++) {
        if (auto bandwidth = driver.getBandwidthMetric(i)) {
          func.setArgAttr(
              i, "tt.bandwidth",
              StringAttr::get(func.getContext(), bandwidth.value()));
        }
        if (auto compute = driver.getComputeMetric(i)) {
          func.setArgAttr(i, "tt.compute",
                          StringAttr::get(func.getContext(), compute.value()));
        }
      }
    }
  }
};

} // namespace

// Include the MLIR pass plugin registry implementation
#include "ExportPass.cpp"
