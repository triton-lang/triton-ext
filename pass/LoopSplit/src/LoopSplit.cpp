#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will examine uses of induction variables to determine if a loop
// should be split into consecutive loops of [lo..midp) and [midp..hi].
// If the induction var is `<` or `>` a loop invariant value, it should be
// split.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "triton-loop-split"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;

namespace {

/// @brief Struct to collect characteristics of CmpI predicates.
struct CmpType {
  bool greater;
  bool equal;
  const char *str;
};

/// @brief Map to quickly determine if a supported predicate is present.
///        If present, quickly query characteristics.
/// Only signed predicates: the split point is clamped to the (signed) loop
/// bounds, which is not valid for an unsigned comparison.
static DenseMap<arith::CmpIPredicate, CmpType> CmpTypeMap = {
    {arith::CmpIPredicate::sge, {true, true, "sge"}},
    {arith::CmpIPredicate::sgt, {true, false, "sgt"}},
    {arith::CmpIPredicate::sle, {false, true, "sle"}},
    {arith::CmpIPredicate::slt, {false, false, "slt"}},
};

/// @brief Predicate with its operands swapped, e.g. `a < b` <=> `b > a`.
static arith::CmpIPredicate swapPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  default:
    return predicate;
  }
}

/// @brief  Capture the cmpi op and canonicalize (induction var on LHS).
class CanonCmp {
public:
  CanonCmp(arith::CmpIOp cmp, OpOperand &iter) : predicate(cmp.getPredicate()) {
    if (isValid()) {
      if (iter.getOperandNumber() == 1)
        predicate = swapPredicate(predicate);
      comparand = cmp.getOperand(iter.getOperandNumber() ^ 1);
    }
  }

  bool isValid() const {
    return CmpTypeMap.find(predicate) != CmpTypeMap.end();
  }

  bool isEqual() const {
    return isValid() ? CmpTypeMap[predicate].equal : false;
  }
  bool isGreater() const {
    return isValid() ? CmpTypeMap[predicate].greater : false;
  }

  Value getComparand() const { return comparand; }

  void dump() const {
    if (isValid()) {
      LDBG("  predicate: " << CmpTypeMap[predicate].str);
      LDBG("  comparand: " << comparand);
    }
  }

private:
  arith::CmpIPredicate predicate;
  Value comparand;
};

//===----------------------------------------------------------------------===//
/// @brief LoopBisect class to process an scf.for induction variable, and
///        split the loop when the right conditions are met.
class LoopBisect {
public:
  LoopBisect(scf::ForOp _forOp) : forOp(_forOp) {}

  LogicalResult bisect();

private:
  void getCmp(OpOperand &opr);

private:
  // Data members
  scf::ForOp forOp;

  DenseMap<Operation *, CanonCmp> cmpMap;
};

/// Test the use for:
///  1. Is a CmpI
///  2. Is >=, <=, >, <
///  3. The comparand is loop-invariant
/// Note: poor man's SCEV
/// TODO: add support for mask logic
void LoopBisect::getCmp(OpOperand &opr) {
  if (auto cmp = dyn_cast<arith::CmpIOp>(opr.getOwner())) {
    CanonCmp ccmp(cmp, opr);
    if (ccmp.isValid()) {
      // Other most be loop invariant, needs full DFG analysis
      Value other = ccmp.getComparand();
      auto *defOther = other.getDefiningOp();
      if (!defOther)
        defOther = dyn_cast<BlockArgument>(other).getOwner()->getParentOp();
      if (forOp->isAncestor(defOther)) {
        LDBG("Comparand not loop invariant");
        return;
      }
      cmpMap.insert(std::make_pair(cmp, ccmp));
    }
  }
}

LogicalResult LoopBisect::bisect() {
  auto lo = forOp.getLowerBound();
  auto hi = forOp.getUpperBound();
  auto step = forOp.getConstantStep();

  if (!step) {
    LDBG("Non-constant step");
    return failure();
  }
  int64_t stepVal = step->getSExtValue();
  if (stepVal <= 0) {
    LDBG("Step is not positive: " << stepVal);
    return failure();
  }

  // Collect comparators
  auto iter = forOp.getInductionVar();
  for (OpOperand &use : iter.getUses()) {
    getCmp(use);
  }

  // Split loop on the first comparison
  if (cmpMap.size() >= 1) {
    auto [cmp, ccmp] = *cmpMap.begin();

    LDBG("Split cmp:    " << *cmp);
    LLVM_DEBUG(ccmp.dump());

    auto loc = cmp->getLoc();
    OpBuilder b(forOp);

    // midp is the first induction value at which the comparison flips:
    // `iv < m` and `iv >= m` flip at m, `iv <= m` and `iv > m` at m + 1.
    // When m >= hi there is no flip inside the loop, so use hi; this also
    // avoids overflowing m + 1.
    Value midp = ccmp.getComparand();
    if (ccmp.isEqual() != ccmp.isGreater()) {
      Value one = arith::ConstantIntOp::create(b, loc, midp.getType(), 1);
      Value next = arith::AddIOp::create(b, loc, midp, one);
      Value pastEnd =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, midp, hi);
      midp = arith::SelectOp::create(b, loc, pastEnd, hi, next);
    }
    // Keep midp within [lo, hi] so that neither loop runs iterations the
    // original loop did not.
    midp = arith::MaxSIOp::create(b, loc, midp, lo);
    midp = arith::MinSIOp::create(b, loc, midp, hi);

    /// Handle midp not a multiple of step: round it up to the next iteration,
    /// midp = lo + ceil((midp - lo) / step) * step, which may pass hi.
    if (stepVal != 1) {
      Value step = forOp.getStep();
      Value diff = arith::SubIOp::create(b, loc, midp, lo);
      Value iters = arith::CeilDivSIOp::create(b, loc, diff, step);
      diff = arith::MulIOp::create(b, loc, iters, step);
      midp = arith::AddIOp::create(b, loc, lo, diff);
      midp = arith::MinSIOp::create(b, loc, midp, hi);
    }

    /// TODO(sjw): update upstream peelForLoop
    /// bisect loop [lo .. midp)
    /// bisect loop [midp .. hi)
    IRMapping mapping;
    b.setInsertionPointAfter(forOp);
    scf::ForOp newForOp = cast<scf::ForOp>(b.clone(*forOp, mapping));
    newForOp.setLowerBound(midp);
    forOp.replaceAllUsesWith(newForOp.getResults());
    newForOp.getInitArgsMutable().assign(forOp->getResults());
    forOp.setUpperBound(midp);

    // replace cmp with constant True/False for each loop
    b.setInsertionPoint(forOp);
    cmp->replaceAllUsesWith(
        arith::ConstantIntOp::create(b, loc, !ccmp.isGreater(), 1));
    auto *newCmp = mapping.lookup(cmp);
    newCmp->replaceAllUsesWith(
        arith::ConstantIntOp::create(b, loc, ccmp.isGreater(), 1));
  }

  return success();
}

// To make available the auto-generated base classes in the `impl`
// namespace, we drop in the generated headers from `Passes.td`.
#define GEN_PASS_DEF_TRITONLOOPSPLIT
#include "Passes.h.inc"

struct LoopBisectPass : public impl::TritonLoopSplitBase<LoopBisectPass> {
  LoopBisectPass() = default;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    SmallVector<scf::ForOp> loops;
    getOperation()->walk<WalkOrder::PostOrder>(
        [&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      LoopBisect sp(forOp);
      if (failed(sp.bisect()))
        continue;
    }
  }

private:
};
} // namespace

// Include the MLIR pass plugin registry implementation.
#include "ExportPass.cpp"
