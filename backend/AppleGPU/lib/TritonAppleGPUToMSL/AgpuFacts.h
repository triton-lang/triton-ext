// AgpuFacts - what the walk knows about a value. Read only: emission is
// elsewhere.
#ifndef AGPU_BRIDGE_FACTS_H
#define AGPU_BRIDGE_FACTS_H

#include "agpu/bind/SymbolTable.h"
#include "agpu/emit/EmitDot.h"
#include "agpu/msl/Ast.h"
#include "agpu/plan/Elementwise.h"

#include "AgpuDeviceTile.h"
#include "AgpuPool.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir::triton::applegpu::bridge {

// Carried beside an OpView: OpView's int64_t cannot hold a float.
struct ConstantValue {
  int64_t i = 0;
  double f = 0.0;
  bool isFloat = false;
  // False for an attribute type the backend has no representation for.
  bool known = false;
};

// One value's registers. Once ok(), every at(r) has a name.
class Operand {
public:
  Operand(const agpu::SymbolTable &sym, agpu::ValueId v, int64_t regs)
      : sym_(&sym), v_(v), regs_(regs) {
    for (int64_t r = 0; r < regs; ++r)
      if (sym.regAt(v, (std::size_t)r) == nullptr) {
        unnamed_ = r;
        break;
      }
  }

  bool ok() const { return unnamed_ < 0; }
  int64_t registers() const { return regs_; }

  int64_t firstUnnamedRegister() const { return unnamed_; }

  // Only meaningful once ok(), which is vacuously true for a zero-register
  // operand, hence the range check.
  const agpu::msl::Str &at(int64_t r) const {
    static const agpu::msl::Str kNone;
    if (r < 0 || r >= regs_)
      return kNone;
    const agpu::msl::Str *n = sym_->regAt(v_, (std::size_t)r);
    return n ? *n : kNone;
  }

private:
  const agpu::SymbolTable *sym_;
  agpu::ValueId v_;
  int64_t regs_;
  int64_t unnamed_ = -1;
};

// Everything a handler needs before it can emit: arity, result element type,
// register count and the operands resolved to names.
struct Ready {
  agpu::ElemType elem;
  int64_t regs = 0;
  agpu::msl::SmallVec<Operand, 3> ops;
  agpu::Decision why = agpu::Decision::emitted();

  bool ok() const { return why.ok(); }

  const Operand &operator[](std::size_t i) const { return ops[i]; }
};

struct RegionSources {
  RankedTensorType srcTy;
  int64_t sourceRegisterCount = 0;
  agpu::msl::SmallVec<agpu::msl::SmallVec<agpu::msl::Str, 8>, 4> names;
  agpu::Decision why = agpu::Decision::emitted();

  bool ok() const { return why.ok(); }
};

// What one register holds. With a guard, the declaration takes `init` and a
// guarded assignment of `value` follows.
struct RegValue {
  agpu::msl::Expr *value = nullptr; // null declines the whole op
  agpu::msl::Expr *guard = nullptr; // null: unconditional
  agpu::msl::Expr *init = nullptr;  // the value when the guard fails
};

// One link of an accumulator-rooted operand chain (`DrainStepFact::branch`).
// No rounding (branches are f32 end to end) and no nesting: a link's operand
// is a splat or a window.
struct DrainBranchLinkFact {
  Operation *op = nullptr;
  enum class Operand { None, Splat, Window } kind = Operand::None;
  Value splat;             // Splat: the scalar, null for a constant
  double splatConst = 0.0; // Splat with null `splat`: the number
  DrainAddend addend;      // Window: the load the drain absorbs
};

// One folded drain op: which op and where its second operand is readable at
// drain time. Mirrors the emit layer's `DrainStep`: Values here, names there.
struct DrainStepFact {
  Operation *op = nullptr;
  enum class Operand { None, Splat, Window, AccChain } kind = Operand::None;
  Value splat;             // Splat: the scalar, null for a constant
  double splatConst = 0.0; // Splat with null `splat`: the number
  DrainAddend addend;      // Window: the load the drain absorbs
  // AccChain: the operand is itself computed from the accumulator (e.g.
  // gelu's chain of epilogue ops). Empty: the operand IS the accumulator.
  agpu::msl::SmallVec<DrainBranchLinkFact, 4> branch;
  int branchBase = 0;
  bool roundBefore = false;
};

struct DotShape {
  RankedTensorType aTy, bTy, cTy;
  agpu::ElemType aElem, bElem, cElem;

  // A's device residency, when it has one: base pointer and row stride, so
  // the MMA reads fragments in place and never stages them into the pool.
  DeviceTile aDevice;

  Value cInput, cResult;

  // The incoming C is an scf.for iter arg: this dot runs once per iteration.
  bool cIterArg = false;

  // The loop result this dot's accumulator is carried in. Non-null only when
  // the accumulation cycle is this dot's alone: C is iter arg `i` with no
  // reader but this dot and the dot's result is what the yield hands back at
  // `i`. The loop must not also carry this value in variables.
  Value cCarried;

  bool accumulatorIsCarried() const { return cIterArg; }
  bool accumulatorOutlivesLoop() const { return cCarried != nullptr; }

  // The device window a fused C stores straight into: the loop result's
  // (possibly converted) single use is a `tt.store` through a row-major
  // window. Null base means the drain goes through the pool. Whether it is
  // used is the plan's call (`DotFacts::cDirect` -> `Plan::storesCDirect`).
  DeviceTile cDevice;
  Operation *cStore = nullptr;

  // The store's mask, reduced to per-axis bounds on the window's own
  // coordinates. Both absent for an unmasked store.
  AxisBound cRowBound, cColBound;
  llvm::SmallVector<WindowBounds::Clamp, 2> cClamps;

  // A conjunct of the same mask that is uniform across the tile: one guard
  // around the whole drain.
  Value cUniformGuard;

  // The elementwise ops between the loop result and the store, in order: the
  // chain the direct drain folds. Each step's second operand is readable at
  // the drain's own coordinates, a splat or a device window at the store's
  // starts.
  agpu::msl::SmallVec<DrainStepFact, 4> cSteps;

  // What a direct drain renders itself and the walk must not emit: the
  // chain's converts, the folded ops and any operand load the fold was the
  // only reader of. Consulted only when the plan chose the direct drain.
  std::vector<Operation *> cChainOps;

  // Which link of the C-store proof broke, when it did. Diagnostic only.
  std::string cStoreWhy;
};

struct DotOperands {
  DotShape shape;

  // Not the operand itself when a `convert_layout` fronts it: staging
  // addresses by element index, so that round trip is one the scatter undoes.
  agpu::ValueId aStage = 0, bStage = 0;
  RankedTensorType aStageTy, bStageTy;

  agpu::ValueId cOut = 0;
  RankedTensorType cOutTy;

  // The incoming accumulator, when it contributes. Empty means zero, which
  // `readbackLoad` reads as an assigning readback.
  agpu::msl::SmallVec<agpu::msl::Str, 8> cIn;

  agpu::Decision why = agpu::Decision::emitted();

  bool ok() const { return why.ok(); }
};

// What a fused dot needs its enclosing loop to emit. The loop declares and
// drains the accumulator fragments: the dot's handler records this and
// `emitForOp` finds it once the body is built.
struct FusedDot {
  agpu::Plan plan;
  agpu::DirectNames names;
  agpu::CoordSource cCoords;
  std::function<agpu::ReadbackInputs(const agpu::Range &)> readbackFor;
  // The device window the drain stores into, when the plan chose the direct
  // drain; empty otherwise.
  agpu::DeviceStoreTarget cStore;
  // The folded elementwise chain that drain applies per element.
  std::vector<agpu::DrainStep> cSteps;

  // The loop result the accumulators carry.
  agpu::ValueId result = 0;

  // Whether this dot's fragments carry `v` across the loop. If so, the loop
  // must not also carry it in variables.
  bool carries(agpu::ValueId v) const { return v == result; }
};

// Whether any of a loop's fused dots carries `v` in its fragments.
inline bool inFragments(const std::vector<FusedDot> &dots, agpu::ValueId v) {
  for (const FusedDot &fd : dots)
    if (fd.carries(v))
      return true;
  return false;
}

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_FACTS_H
