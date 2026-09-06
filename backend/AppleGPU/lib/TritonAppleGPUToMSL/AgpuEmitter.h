// AgpuEmitter - the walk. TTGIR in, agpu facts out, MSL text on a stream.
//
// Split across AgpuEmitter.cpp (walk core), AgpuValues.cpp (value identity,
// coherence, pointers and layout queries), AgpuPool.cpp (threadgroup pool),
// AgpuDot*.cpp (tt.dot facts, drain, staging and handler),
// AgpuDotChain.cpp (the fused drain proof),
// AgpuRegions.cpp (reduce/scan/map), AgpuCarried.cpp (values crossing a region
// boundary), AgpuControl.cpp (for/if/while), AgpuDeviceFns.cpp (the CFG walk,
// device fns and tt.call), AgpuDebug.cpp (assert/print) and
// AgpuHandlers*.cpp (dispatch-table families).
#ifndef AGPU_BRIDGE_EMITTER_H
#define AGPU_BRIDGE_EMITTER_H

#include "AgpuFacts.h"
#include "AgpuLayout.h"
#include "AgpuShape.h"
#include "AgpuTypes.h"
#include "agpu/core/MemDesc.h"

#include "agpu/Emitter.h"
#include "agpu/bind/Dispatch.h"
#include "agpu/bind/LayoutBind.h"
#include "agpu/bind/PointerBind.h"
#include "agpu/bind/SymbolTable.h"
#include "agpu/emit/EmitAtomic.h"
#include "agpu/emit/EmitControl.h"
#include "agpu/emit/EmitReduce.h"
#include "agpu/emit/EmitRegion.h"
#include "agpu/emit/EmitScan.h"
#include "agpu/emit/EmitShuffle.h"
#include "agpu/plan/AccessPlan.h"
#include "agpu/plan/BandPlan.h"
#include "agpu/plan/Coherence.h"
#include "agpu/plan/LaunchPlan.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <map>
#include <set>
#include <sstream>
#include <string>

namespace mlir::triton::applegpu::bridge {

agpu::LaunchFacts launchFactsOf(Operation *scope);

// A decline names the op it came from. The op already knows its own name, so
// respelling it at the call site is what pushed these calls past one line.
inline agpu::Decision declined(std::string_view where, agpu::msl::Str why) {
  return agpu::Decision::declined(std::string(where), std::move(why));
}

inline agpu::Decision declined(const agpu::OpView &o, agpu::msl::Str why) {
  return declined(o.name, std::move(why));
}

struct PtrOffset {
  agpu::msl::Str name;
  agpu::msl::Type type = agpu::msl::Context::i32();
  // Whether this pointer minted the variable or reuses the index operand's.
  bool owned = false;
};

// A threadgroup buffer declared outside the pool.
struct LiveBuffer {
  agpu::msl::Str name;
  agpu::msl::Type elem;
  agpu::msl::Str decl;
};

struct BodyState {
  BodyState() = default;
  BodyState(const agpu::SymbolTable &afterArgs,
            const std::set<agpu::ValueId> &argPtrs)
      : sym(afterArgs), basePtrs(argPtrs), declaresThreadgroup(true) {}

  agpu::CoordHoist hoist{agpu::ThreadNames{}};
  agpu::msl::Block liveDecls;
  std::vector<LiveBuffer> liveBuffers;
  PoolLedger pool;

  // Distinguishes the temporaries `offsetSum` and `castTo` mint. One counter
  // for both: what it guarantees is distinctness within a body.
  int64_t tempSeq = 0;
  int dotSeq = 0;
  int reduceSeq = 0;
  int scanSeq = 0;
  int combineSeq = 0;
  int mapSeq = 0;
  int shuffleSeq = 0;
  int callSeq = 0;

  agpu::msl::Str scope;

  // Convert results a dot's readback already landed in. Their handler emits
  // nothing: the round trip was absorbed into the readback.
  std::set<agpu::ValueId> absorbedInto;
  std::set<Operation *> absorbedOps;
  std::vector<FusedDot> fusedDots;

  agpu::SymbolTable sym;

  // addptr binds the base name and keeps the offset here: an access is
  // base[off].
  std::map<std::pair<agpu::ValueId, int64_t>, PtrOffset> offsetOf;
  std::map<agpu::ValueId, agpu::AffineFamily> affine;
  llvm::DenseSet<Value> clampApplied;

  // Written only via markBasePointer/inheritBasePointer.
  std::set<agpu::ValueId> basePtrs;

  std::map<agpu::ValueId, agpu::MemDesc> memDescOf;

  // Metal admits a threadgroup declaration only in a kernel.
  bool declaresThreadgroup = false;
  // A callback handed to an agpu::emitX function cannot return a Decision
  // through that signature, so a failure inside one is parked here and the
  // caller reads it after. Armed before each use, because a failure left over
  // from one op would otherwise decline the next.
  bool pendingOk = true;
  std::string pendingWhy;

  void armPending() {
    pendingOk = true;
    pendingWhy.clear();
  }
  void notePending(std::string why) {
    pendingOk = false;
    pendingWhy = std::move(why);
  }
};

class AgpuEmitter {
public:
  AgpuEmitter(ModuleOp mod, llvm::raw_ostream &os) : mod_(mod), os_(os) {
    agpu_.gates = agpu::GateSet::fromEnvironment();
  }

  LogicalResult emit();

private:
  // Insertion order: first match wins.
  void registerHandlers();
  void registerArithHandlers(); // elementwise .. cast
  void registerValueHandlers(); // grid, splat, poison, constant
  void registerRebindHandler();
  void registerCallHandler();
  void registerTileHandlers(); // histogram, gather
  void registerInterleaveHandler();
  void registerDotHandler();
  void registerRangeHandler();
  void registerAddPtrHandler();
  void registerAtomicHandlers(); // atomic_rmw, atomic_cas, atomic_poll
  void registerMemoryHandler();  // load, store
  void registerMemDescHandler(); // local_alloc/load, memdesc views
  void registerBarrierHandler(); // ttg.barrier, gpu.barrier

  agpu::Decision walkBlock(Block &block, agpu::msl::Block &out);

  agpu::RegionFacts regionFactsOf(Region &region);

  using TerminatorFn =
      std::function<agpu::Decision(Block &, agpu::msl::Block &)>;

  agpu::Decision walkRegionCFG(Region &region, agpu::msl::Block &out,
                               const TerminatorFn &atTerminator = nullptr);

  agpu::Decision walkWholeRegion(Region &region, agpu::msl::Block &out,
                                 const TerminatorFn &atTerminator = nullptr);

  LogicalResult
  addDeviceFnsInCallOrder(const llvm::DenseSet<StringRef> &callTargets);

  LogicalResult addDeviceFn(triton::FuncOp func);

  LogicalResult appendDeviceReturn(triton::FuncOp func,
                                   const agpu::DeviceFnAbi &abi,
                                   agpu::msl::Block &body);

  struct DeviceFnEntry {
    agpu::DeviceFnFacts facts;
    agpu::DeviceFnAbi abi;
  };
  std::map<std::string, DeviceFnEntry> deviceFns_;

  agpu::Decision walkOp(Operation *op);

  RegionSources regionSourcesOf(ValueRange srcs, ResultRange results,
                                std::string_view where);

  agpu::Decision gatherRegionNames(ValueRange srcs, llvm::ArrayRef<int> order,
                                   std::string_view where, RegionSources &into);

  agpu::Decision
  scratchRegionsInto(int operands, agpu::msl::Str (*keyFor)(int),
                     std::string_view where,
                     agpu::msl::SmallVec<agpu::msl::Str, 4> &into);

  agpu::Decision emitReduceOp(triton::ReduceOp red);

  agpu::Decision reductionPlanOf(triton::ReduceOp red, RankedTensorType srcTy,
                                 agpu::ReductionPlan &out);

  agpu::Decision emitScanOp(triton::ScanOp scan);

  agpu::Decision scanPlanOf(triton::ScanOp scan, RankedTensorType srcTy,
                            agpu::ScanFacts &out);

  // Register index order is not axis order; folding must walk axis order.
  agpu::Decision scanRegisterOrder(RankedTensorType srcTy, int axis,
                                   bool reverse, std::vector<int> &out);

  agpu::Decision emitMapOp(triton::MapElementwiseOp map);

  agpu::msl::SmallVec<agpu::msl::Str, 4>
  lowerCombine(Region &region, agpu::msl::Block &body,
               const agpu::msl::SmallVec<agpu::msl::Str, 4> &lhs,
               const agpu::msl::SmallVec<agpu::msl::Str, 4> &rhs);

  int64_t registersHeldByType(Type t) const;

  void bindCarried(Value v, const agpu::CarriedValue &cv);

  agpu::CarriedValue carriedFresh(Value v);

  agpu::Decision carriedFrom(Value v, const agpu::CarriedValue &like,
                             agpu::CarriedValue &out, std::string_view where,
                             std::string_view why);

  agpu::Decision carriedOperands(Operation *term, const agpu::Carried &like,
                                 agpu::Carried &out, std::string_view where);

  // Restores cur_, which a handler appends through.
  agpu::Decision walkRegion(Region &region, agpu::msl::Block &into);

  agpu::Decision walkRegion(Region &region, agpu::msl::Block &into,
                            const llvm::function_ref<agpu::Decision()> &atEnd);

  agpu::Decision emitForOp(scf::ForOp forOp);

  agpu::Decision emitIfOp(scf::IfOp ifOp);

  agpu::Decision emitWhileOp(scf::WhileOp wh);

  agpu::Decision carriedFor(Value v, agpu::Carried &out,
                            const agpu::ValueNames &names);

  agpu::Decision declineOp(Operation *op, const agpu::Decision &d,
                           std::string_view name);

  agpu::Decision emitted(const agpu::Decision &d, Operation *op,
                         std::string_view name) {
    return d.ok() ? d : declineOp(op, d, name);
  }

  // Pre-pass facts only: clamp targets and live buffer bytes. Pool
  // scratch is sized from what the built bodies used.
  void scanPool(triton::FuncOp func);

  agpu::DebugBinding printBindingOf(triton::FuncOp func);

  agpu::DebugBinding assertBindingOf(triton::FuncOp func);

  agpu::CoherenceFacts coherenceFactsOf(triton::FuncOp func);

  agpu::Decision emitAssertOp(triton::AssertOp as);

  agpu::Decision emitPrintOp(triton::PrintOp pr);

  PoolNeed poolNeedOf(Operation *op);

  agpu::msl::Str liveBuffer(const agpu::msl::Str &name, agpu::ElemType elem);

  agpu::BandPlan bandPlanFor(RankedTensorType ty,
                             const agpu::ElemType &elem) const;

  bool isZeroTensor(agpu::ValueId v) const;

  agpu::ValueId idOf(Value v);

  bool willHaveScalarName(Value v, Operation *useSite) const;

  agpu::ValueNames freshNames(Value v, int64_t count);

  LogicalResult bindArgs(triton::FuncOp func,
                         std::vector<agpu::KernelArg> &args);

  // .at() aborts in this exceptions-disabled build, so always go through
  // these accessors.
  const agpu::ElemType *elemOf(agpu::ValueId v) const {
    const auto it = elemFor_.find(v);
    return it == elemFor_.end() ? nullptr : &it->second;
  }
  agpu::ElemType declaredOf(agpu::ValueId v) const {
    const auto it = declaredFor_.find(v);
    if (it != declaredFor_.end())
      return it->second;
    const agpu::ElemType *e = elemOf(v);
    return e ? *e : agpu::i32();
  }
  Value mlirValueOf(agpu::ValueId v) const {
    const auto it = valueFor_.find(v);
    return it == valueFor_.end() ? Value{} : it->second;
  }

  // axis selects the output dimension; null when v is not a tensor or has
  // no such axis.
  agpu::msl::Expr *coordOf(Value v, int reg, int axis = 0);

  agpu::msl::Block poolDecls();

  Ready readyFor(const agpu::OpView &o, std::size_t operands,
                 std::size_t results = 1) {
    Ready r;
    if (o.operands.size() < operands || o.results.size() != results) {
      r.why = declined(o.name, "expected " + std::to_string(operands) +
                                   " operands and " + std::to_string(results) +
                                   " result" + (results == 1 ? "" : "s"));
      return r;
    }
    const agpu::ElemType *e = elemOf(o.results[0]);
    if (!e) {
      r.why = declined(o.name, "result type was never recorded");
      return r;
    }
    r.elem = *e;
    r.regs = registersHeldBy(o.results[0]);
    resolveOperands(o, 0, operands, r);
    return r;
  }

  Ready readyForCounted(const agpu::OpView &o, std::size_t first,
                        std::size_t operands, int64_t regs,
                        std::string_view noName) {
    Ready r;
    if (o.operands.size() < operands) {
      r.why = declined(o.name,
                       "expected " + std::to_string(operands) + " operands");
      return r;
    }
    if (regs <= 0) {
      r.why = declined(o.name, "pointer tensor has no readable layout");
      return r;
    }
    if (!o.results.empty())
      if (const agpu::ElemType *e = elemOf(o.results[0]))
        r.elem = *e;
    r.regs = regs;
    resolveOperands(o, first, operands, r);
    if (!r.ok())
      r.why = declined(o.name,
                       std::string(noName) + " (register " +
                           std::to_string(r.ops.back().firstUnnamedRegister()) +
                           " of " + std::to_string(r.regs) + ")");
    return r;
  }

  // Differs from SymbolTable::regCount, which counts bound names: a
  // pointer's registers share one base name.
  int64_t tensorRegisterCountOf(agpu::ValueId v) {
    const Value mv = mlirValueOf(v);
    if (!mv)
      return 0;
    auto t = dyn_cast<RankedTensorType>(mv.getType());
    if (!t)
      return 0;
    const auto it = regCountOf_.find(t);
    if (it != regCountOf_.end())
      return it->second;
    const int64_t n = registerCount(t);
    regCountOf_[t] = n;
    return n;
  }

  int64_t registersHeldBy(agpu::ValueId v) {
    const int64_t n = tensorRegisterCountOf(v);
    return n > 0 ? n : 1;
  }

  bool affineRegisterDeltas(RankedTensorType rt, int regs) {
    const auto key = std::make_pair(Type(rt), regs);
    const auto it = affineDeltasOf_.find(key);
    if (it != affineDeltasOf_.end())
      return it->second;
    const bool ok = registerDeltasAreAffine(rt, regs);
    affineDeltasOf_[key] = ok;
    return ok;
  }

  // Per-register offsets as scaled differences of each register's lane-0
  // coordinate from register 0's. Deltas are arithmetic differences but the
  // layout composes over GF(2), so the caller must have cleared
  // affineRegisterDeltas first.
  bool scaledRegisterDeltas(RankedTensorType rt, int64_t regs,
                            llvm::ArrayRef<int64_t> scales,
                            std::vector<int64_t> &deltas);

  Operand operandOf(const agpu::OpView &o, std::size_t i, int64_t regs) const {
    return Operand(body_.sym, o.operands[i], regs);
  }

  Operand unresolvedOperand() const { return Operand(body_.sym, 0, 0); }

  void resolveOperands(const agpu::OpView &o, std::size_t from, std::size_t to,
                       Ready &r) {
    for (std::size_t i = 0; i < from; ++i)
      r.ops.push_back(unresolvedOperand());
    for (std::size_t i = from; i < to; ++i) {
      r.ops.push_back(operandOf(o, i, r.regs));
      if (!r.ops.back().ok()) {
        r.why = declined(
            o.name, "operand register " +
                        std::to_string(r.ops.back().firstUnnamedRegister()) +
                        " of " + std::to_string(r.regs) + " has no name");
        return;
      }
    }
  }

  // Declares one value per register from build(r) and binds the result.
  template <typename BuildFn>
  agpu::Decision emitPerRegister(const agpu::OpView &o, int64_t regs,
                                 const agpu::ElemType &elem, char tag,
                                 BuildFn build) {
    agpu::ValueNames names;
    for (int64_t r = 0; r < regs; ++r) {
      const RegValue v = build(r);
      if (!v.value)
        return declined(o.name, "cannot build register " + std::to_string(r));

      const agpu::msl::Str n = nameFor(tag, o.results[0], r);
      const agpu::msl::Type ty = agpu::mslTypeOf(elem);

      if (!v.guard) {
        cur_->push_back(agpu_.context().declStmt(ty, n, v.value));
      } else {
        cur_->push_back(agpu_.context().declStmt(ty, n, v.init));
        agpu::msl::Block one;
        one.push_back(agpu_.context().assign(agpu_.context().var(n), v.value));
        agpu_.context().guardedInto(*cur_, v.guard, std::move(one));
      }
      names.push_back(n);
    }
    body_.sym.bindRegs(o.results[0], std::move(names));
    // Null for tt.poison, which reaches here with no `elemFor_` entry.
    const agpu::ElemType *ir = elemOf(o.results[0]);
    if (ir && agpu::widensToF32(*ir) && elem == agpu::f32())
      declaredFor_[o.results[0]] = elem;
    return agpu::Decision::emitted();
  }

  // body_.scope disambiguates a region walked multiple times (e.g. a combine
  // region walked once per fold step); without it two invocations mint the
  // same name for the same op id.
  agpu::msl::Str nameFor(char tag, agpu::ValueId v, int64_t reg) const {
    return std::string(1, tag) + std::to_string(v) + body_.scope + "_" +
           std::to_string(reg);
  }

  // An object so an early return (a combine region can decline mid-walk)
  // still restores body_.scope.
  class ScopeMark {
  public:
    ScopeMark(AgpuEmitter &e, const agpu::msl::Str &s)
        : e_(e), saved_(e.body_.scope) {
      e_.body_.scope = saved_ + s;
    }
    ~ScopeMark() { e_.body_.scope = saved_; }

  private:
    AgpuEmitter &e_;
    agpu::msl::Str saved_;
  };

  class CurBlock;

  class CurrentBlock {
  public:
    agpu::msl::Block &operator*() const { return *at_; }
    agpu::msl::Block *operator->() const { return at_; }
    explicit operator bool() const { return at_ != nullptr; }

  private:
    friend class AgpuEmitter::CurBlock;
    agpu::msl::Block *at_ = nullptr;
  };

  class CurBlock {
  public:
    CurBlock(AgpuEmitter &e, agpu::msl::Block &b) : e_(e), saved_(e.cur_.at_) {
      e_.cur_.at_ = &b;
    }
    ~CurBlock() { e_.cur_.at_ = saved_; }

    CurBlock(const CurBlock &) = delete;
    CurBlock &operator=(const CurBlock &) = delete;

  private:
    AgpuEmitter &e_;
    agpu::msl::Block *saved_ = nullptr;
  };

  agpu::msl::Str castTo(const agpu::ElemType &to, const agpu::msl::Str &src);

  // Empty if `v` has no name.
  agpu::msl::Str inIrType(agpu::ValueId v, int64_t r);

  // Declares `name` as a device pointer to `elem` at base + offset, and
  // returns it. A uniform offset is folded into the pointer once, so it does
  // not reappear at every access.
  agpu::msl::Str derivedDevicePointer(const agpu::msl::Str &base,
                                      agpu::msl::Expr *offset,
                                      const agpu::ElemType &elem,
                                      agpu::msl::Str name);

  DotOperands dotOperandsOf(const agpu::OpView &o);

  agpu::DotFacts dotFactsOf(const DotShape &shape);

  // The single-use convert_layout a non-fused dot's readback lands in instead
  // of the result, when its layout resolves and, if the readback adds C, is
  // interchangeable with C's. Null: the result itself. Facts and emission
  // both ask this, so the rename decision and the registers it names agree.
  Value readbackLandingOf(const DotShape &shape);
  RankedTensorType renameLandingTypeOf(const DotShape &shape);

  // aTy is null when a type could not be read.
  DotShape dotShapeOf(triton::DotOp dot) const;

  agpu::Decision readADirect(const DotOperands &ops, agpu::DotInputs &in);

  agpu::Decision strideOf(const DeviceTile &t, const char *which,
                          agpu::Stride &out);

  agpu::Decision baseOffsetExpr(const DeviceTile &t, const char *which,
                                agpu::msl::Expr *&out);

  void declareAccumulatorRegisters(const DotOperands &ops,
                                   const agpu::Plan &plan);

  void tagDotNames(agpu::DotInputs &in);

  agpu::Decision namePoolRegions(const agpu::Plan &plan, agpu::DotInputs &in);

  static agpu::ElemType stagedElemOf(const agpu::Plan &plan,
                                     const agpu::ElemType &operand);

  agpu::Decision stageAB(const DotOperands &ops, const agpu::Plan &plan,
                         const agpu::ElemType &stagedAElem,
                         const agpu::ElemType &stagedBElem,
                         agpu::DotInputs &in);

  void setTileInputs(const DotOperands &ops, const agpu::Plan &plan,
                     const agpu::ElemType &stagedAElem,
                     const agpu::ElemType &stagedBElem, agpu::DotInputs &in);

  agpu::Decision setReadbackFor(const DotOperands &ops, const agpu::Plan &plan,
                                agpu::DotInputs &in);

  agpu::Decision resolveDrainSteps(const DotOperands &ops, agpu::DotInputs &in);

  agpu::Decision resolveDirectCStore(const DotOperands &ops,
                                     const agpu::Plan &plan,
                                     agpu::DotInputs &in);

  agpu::Decision stageDotOperands(const DotOperands &ops,
                                  const agpu::Plan &plan, agpu::DotInputs &in);

  // Sums every step of a chained addptr (row + col, etc).
  PtrOffset offsetSum(agpu::ValueId basePtr, int64_t reg,
                      const agpu::msl::Str &added);

  void inheritOffset(agpu::ValueId from, int64_t fromReg, agpu::ValueId to,
                     int64_t toReg);

  agpu::msl::Expr *addressAt(agpu::ValueId ptr, int64_t reg);

  // A pointer's storage is its offset. A borrowed name
  // (splat/rename/addptr) is not storage.
  std::vector<std::pair<agpu::msl::Str, agpu::msl::Type>>
  storageOf(agpu::ValueId v) const;

  void markBasePointer(agpu::ValueId v);

  void inheritBasePointer(agpu::ValueId from, agpu::ValueId to);

  // Metal has no 16-bit atomic; both 16-bit paths operate on the containing
  // 32-bit word plus a flag for which half.
  bool declarePacked16Word(agpu::msl::Expr *addr,
                           const agpu::msl::Str &wordName,
                           const agpu::msl::Str &highName);

  agpu::msl::Expr *maskAt(const agpu::OpView &o, std::size_t maskIndex,
                          int64_t reg);

  agpu::AddressSpread spreadOf(Value ptr);
  agpu::AddressSpread spreadOf(RankedTensorType ty);

  agpu::Decision emitLoad(const agpu::OpView &o, std::size_t maskIndex);
  agpu::Decision emitStore(const agpu::OpView &o, std::size_t maskIndex);

  agpu::Decision emitLocalAlloc(const agpu::OpView &o);
  agpu::Decision emitLocalLoad(const agpu::OpView &o);

  agpu::Decision emitHistogramOp(const agpu::OpView &o);
  agpu::Decision emitGatherOp(const agpu::OpView &o);

  agpu::Decision emitAtomicRmwOp(const agpu::OpView &o);
  agpu::Decision emitAtomicCasOp(const agpu::OpView &o);
  agpu::Decision emitAtomicPollOp(const agpu::OpView &o);

  agpu::Decision emitGridQueryOp(const agpu::OpView &o);
  agpu::Decision emitSplatOp(const agpu::OpView &o);
  agpu::Decision emitPoisonOp(const agpu::OpView &o);
  agpu::Decision emitConstantOp(const agpu::OpView &o);
  agpu::Decision emitMakeRangeOp(const agpu::OpView &o);

  agpu::Decision emitElementwiseOp(const agpu::OpView &o);
  agpu::Decision emitCompareOp(const agpu::OpView &o);
  agpu::Decision emitCompareFOp(const agpu::OpView &o);
  agpu::Decision emitSelectOp(const agpu::OpView &o);
  agpu::Decision emitNegateOp(const agpu::OpView &o);
  agpu::Decision emitMath1Op(const agpu::OpView &o);
  agpu::Decision emitMath2Op(const agpu::OpView &o);
  agpu::Decision emitMath3Op(const agpu::OpView &o);
  agpu::Decision emitCastOp(const agpu::OpView &o);
  agpu::Decision emitReinterpretCast(const agpu::OpView &o, const Ready &ready,
                                     const Operand &a, agpu::ElemType from,
                                     agpu::ElemType to);
  agpu::Decision emitConvertCast(const agpu::OpView &o, const Ready &ready,
                                 const Operand &a, const agpu::ElemType &from,
                                 const agpu::ElemType &to,
                                 const agpu::ElemType &fromDeclared);

  agpu::Decision emitAddPtrOp(const agpu::OpView &o);
  agpu::Decision emitMemoryOp(const agpu::OpView &o);
  agpu::Decision emitBarrierOp(const agpu::OpView &o);
  agpu::Decision emitMemDescViewOp(const agpu::OpView &o);

  agpu::Decision emitRebindOp(const agpu::OpView &o);

  bool interleaveCoordsOf(Value v, std::vector<agpu::RegCoord> &out);
  agpu::Decision emitUnsplatOp(const agpu::OpView &o);
  agpu::Decision emitFp4ToFpOp(const agpu::OpView &o);
  agpu::Decision emitJoinSplitOp(const agpu::OpView &o);

  agpu::Decision emitCallOp(const agpu::OpView &o);
  agpu::Decision emitDotOp(const agpu::OpView &o);
  void logDotPlan(const DotOperands &ops, const agpu::Plan &plan);

  agpu::msl::RollPrediction predictRollFor(triton::FuncOp func);

  agpu::BuiltBody buildKernelBody(Region &region);

  agpu::PtrDims ptrDimsOf(Value ptr, const agpu::ElemType &elem);

  std::vector<agpu::LayoutBasis> layoutDimsOf(Value v);

  agpu::MaskBound maskBoundOf(Value mask, Value laidOut);

  // A wide access casts to an unqualified vector pointer, stripping the
  // coherent qualifier, so the fact travels on the access instead.
  bool coherentBuffer(Value ptr) const;

  ModuleAxisInfoAnalysis &axisInfo() {
    if (!axisInfo_)
      axisInfo_ = std::make_unique<ModuleAxisInfoAnalysis>(mod_);
    return *axisInfo_;
  }

  static agpu::msl::Str accName(agpu::ValueId v, int64_t reg) {
    return "acc" + std::to_string(v) + "_" + std::to_string(reg);
  }

  agpu::CoordSource coordSourceOf(RankedTensorType ty);

  // `names` is indexed by register (one entry always); `actions` one per
  // register that reaches the window. Declines when the layout cannot be read.
  // `names` is the caller's already-resolved register names, one per register
  // of `ty`, empty where the value has none.
  agpu::Decision
  planTileActions(agpu::ValueId v, RankedTensorType ty,
                  const std::vector<agpu::CoordWindow> &windows,
                  const agpu::TileView &dst, unsigned elemBits,
                  agpu::msl::SmallVec<agpu::StageAction, 8> &actions,
                  const agpu::msl::SmallVec<agpu::msl::Str, 8> &names,
                  std::string_view where);

  // The registers of `v` in their IR element type, narrowed where a wider
  // evaluation width left them declared as f32.
  agpu::msl::SmallVec<agpu::msl::Str, 8> stagedNamesOf(agpu::ValueId v,
                                                       int64_t regs);

  agpu::PanelInputs
  panelInputsFor(const agpu::PanelTile &t, const agpu::Plan &plan,
                 agpu::ValueId aId, RankedTensorType aTy, agpu::ValueId bId,
                 RankedTensorType bTy, agpu::ValueId cId, RankedTensorType cTy,
                 const agpu::OperandSource &deviceA,
                 const agpu::msl::Str &poolAName,
                 const agpu::msl::SmallVec<agpu::msl::Str, 8> &cIn,
                 const agpu::ElemType &aElem, const agpu::ElemType &bElem,
                 const agpu::ElemType &cElem,
                 const agpu::msl::SmallVec<agpu::msl::Str, 8> &aNames,
                 const agpu::msl::SmallVec<agpu::msl::Str, 8> &bNames);

  agpu::Decision stageWholeTensor(agpu::ValueId v, RankedTensorType ty,
                                  const agpu::msl::Str &buffer,
                                  const agpu::TileView &dst,
                                  const agpu::ElemType &elem,
                                  std::string_view where,
                                  std::string_view what);

  // Costs registers * 32 layout applications, and both the pool pre-pass and
  // the handlers ask it per op for types a whole function shares.
  const std::vector<std::vector<int64_t>> &elemsPerLaneOf(RankedTensorType rt);
  std::map<const void *, std::vector<std::vector<int64_t>>> elemsPerLane_;

  // `order[d]` names which source axis becomes result axis d (transpose);
  // empty for a pure layout change.
  agpu::ShufflePlan shuffleFor(RankedTensorType srcTy, RankedTensorType resTy,
                               llvm::ArrayRef<int32_t> order = {});

  agpu::Decision emitRedistribute(const agpu::OpView &o, RankedTensorType srcTy,
                                  RankedTensorType resTy,
                                  llvm::ArrayRef<int32_t> order = {});

  // Moves the offset only; declines if registers don't share one base, since
  // a moved offset would then index a different buffer.
  agpu::Decision emitRedistributeOffsets(const agpu::OpView &o,
                                         RankedTensorType srcTy,
                                         RankedTensorType resTy,
                                         llvm::ArrayRef<int32_t> order);

  agpu::Decision moveRegs(const agpu::OpView &o, RankedTensorType srcTy,
                          RankedTensorType resTy, llvm::ArrayRef<int32_t> order,
                          const agpu::ValueNames &srcNames, agpu::ElemType elem,
                          agpu::msl::Expr *scatterGuard,
                          agpu::ValueNames &moved);

  int64_t numWarps() const;

  ModuleOp mod_;
  llvm::raw_ostream &os_;

  std::unique_ptr<ModuleAxisInfoAnalysis> axisInfo_;

  std::set<int> coherentArgs_;

  agpu::Emitter agpu_;
  agpu::DispatchTable table_;

  llvm::DenseMap<Value, agpu::ValueId> ids_;
  agpu::ValueId nextId_ = 0;

  llvm::DenseMap<Type, int64_t> regCountOf_;
  llvm::DenseMap<std::pair<Type, int>, bool> affineDeltasOf_;

  std::map<agpu::ValueId, agpu::ElemType> elemFor_;

  // The type a value's registers are declared in, when wider than the IR type.
  // Absent means the two agree.
  std::map<agpu::ValueId, agpu::ElemType> declaredFor_;

  std::map<agpu::ValueId, Value> valueFor_;

  // Keyed by the mlir scalar whose one definition takes the min(); two
  // stores asking different clamps for one scalar poison it. Module-lived,
  // cleared once in emit(): bodies run during print(), after every function
  // has been scanned.
  llvm::DenseMap<Value, int64_t> clampOf_;
  llvm::DenseSet<Value> clampPoison_;

  llvm::DenseMap<Operation *, bool> cDirectOf_;

  std::map<agpu::ValueId, std::vector<ConstantValue>> constantFor_;

  CurrentBlock cur_;

  bool bodyOk_ = true;

  bool rollK_ = false;

  BodyState body_;
};
} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_EMITTER_H
