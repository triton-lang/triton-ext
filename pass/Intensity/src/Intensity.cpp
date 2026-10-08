//===- Intensity.cpp - Kernel intensity analysis -===//
//
// Triton Intensity extension.
//
//===---------------------------------------------------------------------===//
//
// `-triton-intensity` annotates each `tt.func` argument with
// string attributes describing the work done by the kernel against that
// argument:
//
//   - `tint.load_bytes`:  an algebraic equation for the total bytes loaded
//     per CTA across all loads rooted at this argument.
//   - `tint.store_bytes`: an algebraic equation for the total bytes stored
//     per CTA across all stores rooted at this argument.
//   - `tint.op_count`:    an algebraic equation for the total op count
//     (FLOPs) feeding the stores rooted at this argument.
//
// ## Algorithm
//
// For each function:
//   1. Walk every block once and classify each op's contribution as
//      Load / Store / Compute / Other (see `BlockMetrics::calculateMetric`).
//   2. For every load/store, trace the address back to its originating
//      function-argument via `findPointerParam`, then aggregate the metric
//      bottom-up to function scope, multiplying by the symbolic trip count
//      of each enclosing `scf.for`. An `scf.if` result used as a bound is
//      the `max` of the two yields.
//   3. Write the resulting `SymExpr` back to the `tt.func` argument as a
//      string attribute.
//
// ## Symbolic equations
//
// Equations are `SymExpr` values (see `SymExpr.h`): an MLIR `AffineExpr`,
// or a `min` / `max` of such expressions. `SymTable` maps each leaf
// `Value` (function arg, program id, num programs, opaque integer
// producer) to an `AffineSymbolExpr`. Affine subtrees still constant-fold
// and pass through `simplifyAffineExpr`. `min` and `max` stay outside
// `AffineExpr`, which has no such kinds; `+`, `-`, and multiplication or
// division by a positive constant distribute over them.
//
// At print time each affine subtree is simplified. `s<N>` symbol tokens
// are rewritten to source-level names (`args[N]`, `program_id[N]`,
// `num_programs[N]`). `%` is modulo. `floordiv` and `ceildiv` print as
// function calls, like `min` and `max`.
//
// ## Metric model
//
//   - Bytes (not element counts) are computed as
//     `numElements(type) * elementBits / 8`, rounded up to a whole byte so
//     sub-byte element types (e.g. `i1`) are accounted for in aggregate.
//   - `tt.dot` contributes `M * N * K * 2` FLOPs per block.
//   - Elementwise / reduce / `tt.addptr` ops contribute one op per output
//     element.
//   - Layout ops (`tt.trans`, `tt.reshape`, `tt.split`) contribute nothing.
//   - Each compute op is counted once per function: it is attributed to the
//     first store (in program order) whose value chain reaches it, so an
//     accumulator written by several stores is not counted several times.
//
// ## Control flow
//
//   - `scf.for` trip count = `max(cdiv(upper - lower, step), 0)`, so a
//     descending span (`upper < lower`) contributes no iterations.
//   - `scf.for` iter_args are substituted with their init value (exact
//     when the iter_arg is loop-invariant, safe otherwise).
//   - `scf.for` induction variables are substituted with the loop's upper
//     bound, giving a conservative upper-bound estimate when an inner
//     loop's bound references an outer IV.
//   - `scf.if` results used as symbolic values are `max(then, else)`, an
//     upper bound on the yielded value. `arith.min*` / `arith.max*` are
//     recorded as `min` / `max`.
//   - An op nested in `scf.if` is counted at its full size. The function
//     walk still sums both sides, which over-approximates work that runs
//     on only one side.
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

#include "SymExpr.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/StrUtil.h"

#include <cctype>

#define DEBUG_TYPE "triton-intensity"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONINTENSITY
#include "Passes.h.inc"
} // namespace mlir::triton

namespace {

using tint::SymExpr;

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
public:
  static bool isLoadLikeOp(Operation *op) {
    return isa<triton::LoadOp, triton::DescriptorLoadOp>(op);
  }

  static bool isStoreLikeOp(Operation *op) {
    return isa<triton::StoreOp, triton::DescriptorStoreOp>(op);
  }

private:
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
                   triton::TransOp, triton::ReshapeOp, triton::SplitOp>(op)) {
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
// and knows how to serialise a `SymExpr` built over those symbols back to
// a human-readable string with source-level names.
////////////////////////////////////////////////////////////////////////////////
class SymTable {
public:
  SymTable(triton::FuncOp func)
      : func(func), ctx(func.getContext()), entryBlock(&func.getBody().front()),
        exprs(ctx) {}

  MLIRContext *getContext() const { return ctx; }

  SymExpr get(Value v) {
    auto [it, inserted] = indexByValue.try_emplace(v, values.size());
    if (inserted)
      values.push_back(v);
    return exprs.getAffine(getAffineSymbolExpr(it->second, ctx));
  }

  SymExpr constant(int64_t c) { return exprs.getConstant(c); }

  // Simplify and serialise `expr`, substituting `s<i>` tokens with the
  // source-level name of the value at index `i`.
  std::string print(SymExpr expr) const {
    if (!expr)
      return "";
    expr = expr.simplify(/*numSymbols=*/values.size());
    std::string raw;
    {
      llvm::raw_string_ostream os(raw);
      expr.print(os);
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
      if (atBoundary && raw[i] == 's' && i + 1 < n &&
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
      out += raw[i++];
    }
    return out;
  }

  triton::FuncOp func;
  MLIRContext *ctx;
  Block *entryBlock;
  SymExpr::Context exprs;
  DenseMap<Value, unsigned> indexByValue;
  SmallVector<Value> values;
};

////////////////////////////////////////////////////////////////////////////////
// IntensityAnalysisDriver class
//
// Performs the arithmetic intensity analysis for a single function.
////////////////////////////////////////////////////////////////////////////////
class IntensityAnalysisDriver {

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

  // Resolve `value` to a SymExpr over symbolic leaves (function block
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
  //  - scf.if results are `max(then, else)`, an upper bound on the yielded
  //    value. `arith.minsi` / `minui` / `maxsi` / `maxui` are recorded as
  //    `min` / `max`.
  //  - Unrecognised integer producers are bound to a fresh opaque symbol so
  //    that downstream multiplication / addition still produces a
  //    well-formed equation rather than crashing the pass.
  SymExpr getSymbolicValue(Value value) {
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
      SymExpr thenExpr =
          getSymbolicValue(ifOp.thenYield().getOperand(resultIdx));
      if (!ifOp.elseBlock())
        return thenExpr;
      return SymExpr::max(
          thenExpr, getSymbolicValue(ifOp.elseYield().getOperand(resultIdx)));
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
      if (isa<arith::MinSIOp, arith::MinUIOp>(defOp))
        return SymExpr::min(lhs, rhs);
      if (isa<arith::MaxSIOp, arith::MaxUIOp>(defOp))
        return SymExpr::max(lhs, rhs);
    }
    LDBG("Treating unsupported integer producer as opaque symbol: " << *defOp);
    return syms.get(value);
  }

  SymExpr getSymbolicIterations(scf::ForOp forOp) {
    auto upperBound = getSymbolicValue(forOp.getUpperBound());
    auto lowerBound = getSymbolicValue(forOp.getLowerBound());
    auto step = getSymbolicValue(forOp.getStep());
    // `scf.for` runs `ceil((ub - lb) / step)` iterations when `step > 0`
    // (e.g. a persistent loop `range(pid, num_tiles, NUM_SMS)`). A negative
    // span runs zero times.
    SymExpr iters = (upperBound - lowerBound).ceilDiv(step);
    return SymExpr::max(iters, syms.constant(0));
  }

  // Multiply the per-execution `size` of `op` by the symbolic trip count of
  // every enclosing `scf.for`, up to function scope.
  // TODO: calculate once for each block parent, this should be a lookup
  SymExpr scaleByTripCount(Operation *op, SymExpr size) {
    auto parentOp = op->getParentOp();
    if (isa<FunctionOpInterface>(parentOp))
      return size;
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      size = getSymbolicIterations(forOp) * size;
    } else if (isa<scf::IfOp>(parentOp)) {
      // Count the op at full size. It runs on at most one side; the walk
      // still sums both sides.
    } else {
      LDBG("Unsupported parent op: " << parentOp->getName());
    }
    return scaleByTripCount(parentOp, size);
  }

  SymExpr calculateCompute(Value value, DenseSet<Value> &visited,
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
      return SymExpr();
    }

    auto *defOp = value.getDefiningOp();
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      unsigned idx = cast<OpResult>(value).getResultNumber();
      auto *yieldOp = forOp.getBody()->getTerminator();
      edges.push_back(yieldOp->getOperand(idx));
      return SymExpr();
    }
    if (isa<triton::ReduceOp>(defOp)) {
      // Treat like a normal compute op (handled by the chain walk below).
    } else if (defOp->getNumRegions() > 0) {
      LDBG("Unsupported region-bearing op in compute chain: " << *defOp);
      return SymExpr();
    }

    auto &blockMetrics = metrics.at(defOp->getBlock());
    auto mval = blockMetrics.calculateChainMetric(value, visited, edges);
    return syms.constant(mval.getSize());
  }

  SymExpr calculateCompute(Value value, DenseSet<Value> &visited) {
    SmallVector<Value> edges;
    SymExpr computeSize = calculateCompute(value, visited, edges);
    auto *valueOp = value.getDefiningOp();
    if (computeSize && valueOp != nullptr)
      computeSize = scaleByTripCount(valueOp, computeSize);
    for (auto edge : edges) {
      SymExpr edgeMetric = calculateCompute(edge, visited);
      if (edgeMetric)
        computeSize = computeSize ? computeSize + edgeMetric : edgeMetric;
    }
    return computeSize;
  }

public:
  IntensityAnalysisDriver(triton::FuncOp func)
      : func(func), syms(func), loadBytesMetrics(func.getNumArguments()),
        storeBytesMetrics(func.getNumArguments()),
        computeMetrics(func.getNumArguments()) {
    run();
  }

  void run() {
    func.walk<WalkOrder::PostOrder>(
        [&](Block *block) { metrics.try_emplace(block, block); });

    auto add = [](SymExpr &acc, SymExpr addend) {
      acc = acc ? acc + addend : addend;
    };

    // Every compute op is counted once: the visited set is shared by all
    // stores of the function, so a value reaching several stores (e.g. an
    // accumulator written back in two epilogue sub-tiles, or one store per
    // output argument) is attributed to the first store, in program order,
    // whose value chain reaches it. Walking in program order keeps that
    // attribution deterministic.
    DenseSet<Value> visited;
    func.walk([&](Operation *op) {
      if (BlockMetrics::isLoadLikeOp(op)) {
        auto param = findPointerParam(op->getOperand(0));
        if (!param) {
          LDBG("Skipping load with no resolvable function-arg base: " << *op);
          return;
        }
        auto metric = metrics.at(op->getBlock()).getMetric(op->getResult(0));
        if (metric) {
          SymExpr total =
              scaleByTripCount(op, syms.constant(metric->getSize()));
          add(loadBytesMetrics[param.getArgNumber()], total);
        }
      } else if (BlockMetrics::isStoreLikeOp(op)) {
        auto param = findPointerParam(op->getOperand(0));
        if (!param) {
          LDBG("Skipping store with no resolvable function-arg base: " << *op);
          return;
        }
        Metric storeMetric = metrics.at(op->getBlock()).getStoreOpMetric(op);
        SymExpr total =
            scaleByTripCount(op, syms.constant(storeMetric.getSize()));
        add(storeBytesMetrics[param.getArgNumber()], total);
        SymExpr compute = calculateCompute(op->getOperand(1), visited);
        if (compute)
          add(computeMetrics[param.getArgNumber()], compute);
      }
    });
    LLVM_DEBUG(dump());
  }

  std::optional<std::string> getLoadBytesMetric(unsigned index) const {
    return printMetric(loadBytesMetrics, index);
  }
  std::optional<std::string> getStoreBytesMetric(unsigned index) const {
    return printMetric(storeBytesMetrics, index);
  }
  std::optional<std::string> getComputeMetric(unsigned index) const {
    return printMetric(computeMetrics, index);
  }

  void dump() {
    llvm::errs() << "Intensity Analysis Driver: "
                    "----------------------------------------------\n";
    llvm::errs() << "Function: " << func.getName() << "\n";
    for (auto [block, blockMetrics] : metrics)
      blockMetrics.dump();
    dumpMetrics("Load Bytes", loadBytesMetrics);
    dumpMetrics("Store Bytes", storeBytesMetrics);
    dumpMetrics("Compute", computeMetrics);
  }

private:
  std::optional<std::string> printMetric(ArrayRef<SymExpr> exprs,
                                         unsigned index) const {
    if (index >= exprs.size() || !exprs[index])
      return std::nullopt;
    return syms.print(exprs[index]);
  }

  void dumpMetrics(StringRef label, ArrayRef<SymExpr> exprs) const {
    llvm::errs() << label << " Metrics: " << exprs.size() << "\n";
    for (unsigned i = 0; i < exprs.size(); ++i) {
      llvm::errs() << label << " Metric: index= " << i << ", size= "
                   << (exprs[i] ? syms.print(exprs[i]) : std::string("<none>"))
                   << "\n";
    }
  }

  triton::FuncOp func;
  SymTable syms;
  DenseMap<Block *, BlockMetrics> metrics;
  // Per function argument: bytes loaded from / stored to memory rooted at
  // the argument, and FLOPs feeding the stores rooted at it.
  SmallVector<SymExpr> loadBytesMetrics;
  SmallVector<SymExpr> storeBytesMetrics;
  SmallVector<SymExpr> computeMetrics;
};

////////////////////////////////////////////////////////////////////////////////
// Pass Intensity
////////////////////////////////////////////////////////////////////////////////
struct IntensityPass : public triton::impl::TritonIntensityBase<IntensityPass> {
  using TritonIntensityBase::TritonIntensityBase;

  // TODO: get callgraph (see Analysis/Allocation.h)
  void runOnOperation() override {
    for (auto func : getOperation().getOps<triton::FuncOp>()) {
      IntensityAnalysisDriver driver(func);
      auto setAttr = [&](unsigned i, StringRef name,
                         const std::optional<std::string> &value) {
        if (value)
          func.setArgAttr(i, name,
                          StringAttr::get(func.getContext(), value.value()));
      };
      for (unsigned i = 0; i < func.getNumArguments(); i++) {
        setAttr(i, "tint.load_bytes", driver.getLoadBytesMetric(i));
        setAttr(i, "tint.store_bytes", driver.getStoreBytesMetric(i));
        setAttr(i, "tint.op_count", driver.getComputeMetric(i));
      }
    }
  }
};

} // namespace

// Include the MLIR pass plugin registry implementation
#include "ExportPass.cpp"
