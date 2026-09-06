// AgpuEmitter - the walk. TTGIR in, agpu facts out, MSL text on a stream.
//
// Split across AgpuEmitter.cpp (walk core), AgpuValues.cpp (value identity,
// coherence, pointers and layout queries), AgpuPool.cpp (threadgroup pool) and
// AgpuHandlers*.cpp (dispatch-table families).
#ifndef AGPU_BRIDGE_EMITTER_H
#define AGPU_BRIDGE_EMITTER_H

#include "AgpuFacts.h"
#include "AgpuLayout.h"
#include "AgpuTypes.h"

#include "agpu/Emitter.h"
#include "agpu/bind/Dispatch.h"
#include "agpu/bind/LayoutBind.h"
#include "agpu/bind/SymbolTable.h"
#include "agpu/emit/Emit.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <map>
#include <set>
#include <sstream>
#include <string>

namespace mlir::triton::applegpu::bridge {

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

struct BodyState {
  BodyState() = default;
  BodyState(const agpu::SymbolTable &afterArgs,
            const std::set<agpu::ValueId> &argPtrs)
      : sym(afterArgs), basePtrs(argPtrs) {}

  agpu::CoordHoist hoist{agpu::ThreadNames{}};

  // Distinguishes the temporaries `offsetSum` mints: what it guarantees is
  // distinctness within a body.
  int64_t tempSeq = 0;

  agpu::msl::Str scope;

  agpu::SymbolTable sym;

  // addptr binds the base name and keeps the offset here: an access is
  // base[off].
  std::map<std::pair<agpu::ValueId, int64_t>, PtrOffset> offsetOf;

  // Written only via markBasePointer/inheritBasePointer.
  std::set<agpu::ValueId> basePtrs;
};

class AgpuEmitter {
public:
  AgpuEmitter(ModuleOp mod, llvm::raw_ostream &os) : mod_(mod), os_(os) {}

  LogicalResult emit();

private:
  // Insertion order: first match wins.
  void registerHandlers();
  void registerArithHandlers(); // elementwise, compare
  void registerValueHandlers(); // grid, splat, constant
  void registerRangeHandler();
  void registerAddPtrHandler();
  void registerMemoryHandler(); // load, store

  agpu::Decision walkBlock(Block &block, agpu::msl::Block &out);

  using TerminatorFn =
      std::function<agpu::Decision(Block &, agpu::msl::Block &)>;

  agpu::Decision walkWholeRegion(Region &region, agpu::msl::Block &out,
                                 const TerminatorFn &atTerminator = nullptr);

  agpu::Decision walkOp(Operation *op);

  int64_t registersHeldByType(Type t) const;

  agpu::Decision declineOp(Operation *op, const agpu::Decision &d,
                           std::string_view name);

  agpu::Decision emitted(const agpu::Decision &d, Operation *op,
                         std::string_view name) {
    return d.ok() ? d : declineOp(op, d, name);
  }

  // Pre-pass facts only: clamp targets and live buffer bytes. Pool
  // scratch is sized from what the built bodies used.
  agpu::ValueId idOf(Value v);

  LogicalResult bindArgs(triton::FuncOp func,
                         std::vector<agpu::KernelArg> &args);

  // .at() aborts in this exceptions-disabled build, so always go through
  // these accessors.
  const agpu::ElemType *elemOf(agpu::ValueId v) const {
    const auto it = elemFor_.find(v);
    return it == elemFor_.end() ? nullptr : &it->second;
  }
  Value mlirValueOf(agpu::ValueId v) const {
    const auto it = valueFor_.find(v);
    return it == valueFor_.end() ? Value{} : it->second;
  }

  // axis selects the output dimension; null when v is not a tensor or has
  // no such axis.
  agpu::msl::Expr *coordOf(Value v, int reg, int axis = 0);

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
    return agpu::Decision::emitted();
  }

  // body_.scope disambiguates a region walked multiple times (e.g. a combine
  // region walked once per fold step); without it two invocations mint the
  // same name for the same op id.
  agpu::msl::Str nameFor(char tag, agpu::ValueId v, int64_t reg) const {
    return std::string(1, tag) + std::to_string(v) + body_.scope + "_" +
           std::to_string(reg);
  }

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

  // Sums every step of a chained addptr (row + col, etc).
  PtrOffset offsetSum(agpu::ValueId basePtr, int64_t reg,
                      const agpu::msl::Str &added);

  void inheritOffset(agpu::ValueId from, int64_t fromReg, agpu::ValueId to,
                     int64_t toReg);

  agpu::msl::Expr *addressAt(agpu::ValueId ptr, int64_t reg);

  void markBasePointer(agpu::ValueId v);

  void inheritBasePointer(agpu::ValueId from, agpu::ValueId to);

  // Metal has no 16-bit atomic; both 16-bit paths operate on the containing
  // 32-bit word plus a flag for which half.
  bool declarePacked16Word(agpu::msl::Expr *addr,
                           const agpu::msl::Str &wordName,
                           const agpu::msl::Str &highName);

  agpu::msl::Expr *maskAt(const agpu::OpView &o, std::size_t maskIndex,
                          int64_t reg);

  bool addressesAreRedundant(Value ptr);

  agpu::Decision emitLoad(const agpu::OpView &o, std::size_t maskIndex);
  agpu::Decision emitStore(const agpu::OpView &o, std::size_t maskIndex);

  agpu::Decision emitGridQueryOp(const agpu::OpView &o);
  agpu::Decision emitSplatOp(const agpu::OpView &o);
  agpu::Decision emitConstantOp(const agpu::OpView &o);
  agpu::Decision emitMakeRangeOp(const agpu::OpView &o);

  agpu::Decision emitElementwiseOp(const agpu::OpView &o);
  agpu::Decision emitCompareOp(const agpu::OpView &o);
  agpu::Decision emitAddPtrOp(const agpu::OpView &o);
  agpu::Decision emitMemoryOp(const agpu::OpView &o);

  agpu::BuiltBody buildKernelBody(Region &region);

  std::vector<agpu::LayoutBasis> layoutDimsOf(Value v);

  agpu::CoordSource coordSourceOf(RankedTensorType ty);

  int64_t numWarps() const;

  ModuleOp mod_;
  llvm::raw_ostream &os_;

  agpu::Emitter agpu_;
  agpu::DispatchTable table_;

  llvm::DenseMap<Value, agpu::ValueId> ids_;
  agpu::ValueId nextId_ = 0;

  llvm::DenseMap<Type, int64_t> regCountOf_;

  std::map<agpu::ValueId, agpu::ElemType> elemFor_;

  std::map<agpu::ValueId, Value> valueFor_;

  std::map<agpu::ValueId, std::vector<ConstantValue>> constantFor_;

  CurrentBlock cur_;

  bool bodyOk_ = true;

  BodyState body_;
};
} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_EMITTER_H
