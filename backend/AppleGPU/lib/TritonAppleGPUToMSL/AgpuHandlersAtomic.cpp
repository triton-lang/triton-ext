// Atomic handlers: read-modify-write, compare-and-swap, poll.
#include "AgpuEmitter.h"
#include "AgpuEnums.h"

#include "agpu/emit/EmitCas.h"
#include "agpu/emit/EmitPoll.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

bool AgpuEmitter::declarePacked16Word(am::Expr *addr, const am::Str &wordName,
                                      const am::Str &highName) {
  if (!addr)
    return false;
  am::Context &c = agpu_.context();

  // A 16-bit atomic runs as a 32-bit atomic on the containing word: mask the
  // pointer to the word, set a flag for which half.
  const am::Type sizeT = am::Type::scalar(am::Scalar::U64);
  const am::Type wordPtr = am::Type::named(am::builtin::atomic::Uint)
                               .pointerTo(am::AddrSpace::Device);

  am::Expr *asInt = c.cast(sizeT, c.addrOf(addr));
  cur_->push_back(c.declStmt(am::Type::scalar(am::Scalar::Bool), highName,
                             c.binary(am::BinOp::Ne,
                                      c.binary(am::BinOp::And, asInt, c.lit(2)),
                                      c.lit(0))));
  cur_->push_back(c.declStmt(
      wordPtr, wordName,
      c.cast(wordPtr, c.binary(am::BinOp::And, asInt, c.lit(~(int64_t)3)))));
  return true;
}

agpu::Decision AgpuEmitter::emitAtomicPollOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const Value res = mlirValueOf(o.results.empty() ? 0 : o.results[0]);
  auto poll =
      res ? res.getDefiningOp<triton::AtomicPollOp>() : triton::AtomicPollOp{};
  if (!poll)
    return declined("tt.atomic_poll", "the op was never recorded");

  // Flag width comes from the expected value.
  const std::optional<agpu::ElemType> want =
      elemTypeOf(poll.getExpected().getType());
  if (!want)
    return declined("tt.atomic_poll", "the expected value has no element type");

  agpu::PollFacts f;
  f.bits = want->bits;
  f.acquire = poll.getSem() == triton::MemSemantic::ACQUIRE;
  // Presence only: a timeout poll tests once and reports what it saw. The
  // duration is not honoured.
  f.hasTimeout = poll.getTimeout() ? true : false;

  const agpu::PollPlan plan = agpu::planPoll(f);
  if (!plan.usable)
    return pollDecision(plan);

  const Ready ready =
      readyForCounted(o, 0, 2, 1, "an operand has no register name");
  if (!ready.ok())
    return ready.why;
  const Operand &expected = ready.ops[1];

  am::Expr *addr = addressAt(o.operands[0], 0);
  if (!addr)
    return declined("tt.atomic_poll", "pointer has no recorded offset");

  agpu::PollNames nm;
  const std::string tag = std::to_string(o.results[0]) + body_.scope;
  nm.ptr = "flagp" + tag;
  nm.expected = "want" + tag;
  nm.result = "ready" + tag;
  nm.flag = "seen" + tag;

  // Empty at every width but 16-bit; emitPoll reads empty as "flag
  // occupies the whole word".
  am::Str isHigh;
  if (plan.load == agpu::PollLoad::PackedHalf) {
    isHigh = "hi" + tag;
    if (!declarePacked16Word(addr, nm.ptr, isHigh))
      return declined("tt.atomic_poll", "pointer has no recorded offset");
  } else {
    const am::Type wordPtr = agpu::pollPtrType(plan);
    cur_->push_back(
        mc.declStmt(wordPtr, nm.ptr, mc.cast(wordPtr, mc.addrOf(addr))));
  }

  cur_->push_back(
      mc.declStmt(agpu::mslTypeOf(*want), nm.expected, mc.var(expected.at(0))));

  const agpu::Decision d = agpu::emitPoll(mc, *cur_, plan, nm, isHigh);
  if (!d.ok())
    return d;

  // Scalar bool regardless of pointer layout: the poll runs once per
  // threadgroup.
  body_.sym.bindScalar(o.results[0], nm.result);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitAtomicCasOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  if (o.operands.size() < 3)
    return declined("tt.atomic_cas", "expected a pointer, a compare and a "
                                     "value");

  const agpu::ElemType *elemP = elemOf(o.results[0]);
  if (!elemP)
    return declined("tt.atomic_cas", "result type was never recorded");

  const Value ptrV = mlirValueOf(o.operands[0]);
  agpu::CasFacts f;
  f.elem = elemP->kind == agpu::ElemType::Kind::Float ? agpu::ElemClass::Float
                                                      : agpu::ElemClass::Int;
  f.bits = elemP->bits;
  f.order = memOrderOf((triton::MemSemantic)o.intAt(0));
  f.uniformPtr = spreadOf(ptrV).uniformPtr;

  const agpu::CasPlan plan = agpu::planCas(f);
  if (!plan.usable())
    return casDecision(plan);

  auto ptrTy =
      ptrV ? dyn_cast<RankedTensorType>(ptrV.getType()) : RankedTensorType();
  const int64_t regs = ptrTy ? registerCount(ptrTy) : 1;
  const Ready ready =
      readyForCounted(o, 1, 3, regs, "an operand has no register names");
  if (!ready.ok())
    return ready.why;
  const Operand &cmp = ready.ops[1];
  const Operand &val = ready.ops[2];

  agpu::ValueNames names;
  for (int64_t r = 0; r < regs; ++r) {
    am::Expr *addr = addressAt(o.operands[0], r);
    if (!addr)
      return declined("tt.atomic_cas", "pointer has no recorded offset");

    const agpu::CasNames nm = agpu::CasNames{}.suffixed(
        std::to_string(o.results[0]) + body_.scope + "_" + std::to_string(r));

    // expected is in-out: Metal writes what it found, so it needs a
    // mutable local.
    cur_->push_back(
        mc.declStmt(agpu::mslTypeOf(*elemP), nm.expected, mc.var(cmp.at(r))));
    cur_->push_back(
        mc.declStmt(agpu::mslTypeOf(*elemP), nm.desired, mc.var(val.at(r))));

    if (plan.strategy == agpu::CasStrategy::Packed16) {
      if (!declarePacked16Word(addr, nm.ptr, nm.isHigh))
        return declined("tt.atomic_cas", "pointer has no recorded offset");
    } else {
      const am::Type wordPtr = am::deviceAtomicPtr(am::Scalar::U32);
      cur_->push_back(
          mc.declStmt(wordPtr, nm.ptr, mc.cast(wordPtr, mc.addrOf(addr))));
    }

    am::Str bound;
    const agpu::Decision d = agpu::emitCas(mc, *cur_, plan, nm, *elemP, &bound);
    if (!d.ok())
      return d;

    names.push_back(bound);
  }

  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitAtomicRmwOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  agpu::RmwOp rop;
  if (!rmwOpFor((triton::RMWOp)o.intAt(0), rop))
    return declined("tt.atomic_rmw", "unhandled read-modify-write operation");

  const agpu::ElemType *elemP = elemOf(o.results[0]);
  if (!elemP)
    return declined("tt.atomic_rmw", "result type was never recorded");

  agpu::AtomicFacts f;
  f.op = rop;
  f.elem = elemP->kind == agpu::ElemType::Kind::Float ? agpu::ElemClass::Float
                                                      : agpu::ElemClass::Int;
  f.bits = elemP->bits;
  // elem/bits can't tell f16 from bf16; the packed path narrows
  // differently for each.
  f.packedElem = *elemP;

  const Value ptrV = mlirValueOf(o.operands[0]);
  const agpu::AddressSpread spread = spreadOf(ptrV);
  f.laneFree = spread.laneFree;
  f.warpFree = spread.warpFree;
  f.uniformPtr = spread.uniformPtr;

  auto ptrTy =
      ptrV ? dyn_cast<RankedTensorType>(ptrV.getType()) : RankedTensorType();
  if (ptrTy)
    f.regFree = freeBitsOf(gpu::toLinearLayout(ptrTy), ptrTy.getContext(),
                           lldim::Register);

  const agpu::AtomicPlan plan =
      agpu::planAtomic(f, memOrderOf((triton::MemSemantic)o.intAt(1)));
  if (!plan.usable())
    return plan.decision(f);
  agpu_.helpers.require(plan);

  const int64_t regs = ptrTy ? registerCount(ptrTy) : 1;
  const Ready ready =
      readyForCounted(o, 1, 2, regs, "the value has no register names");
  if (!ready.ok())
    return ready.why;
  const Operand &val = ready.ops[1];

  am::SmallVec<am::Str, 8> ptrs, values, highs;
  // Metal has no float CAS, so the emulated paths take device
  // atomic_uint* and do the float arithmetic inside. plan.word applies to
  // the native path only.
  const bool emulated = plan.strategy != agpu::AtomicStrategy::Native;
  const am::Scalar word =
      agpu::scalarOfWord(emulated ? agpu::AtomicWord::U32 : plan.word);
  const am::Type wordPtr = am::deviceAtomicPtr(word);
  for (int64_t r = 0; r < regs; ++r) {
    if (plan.replicas.isReplica((int)r)) {
      ptrs.push_back({});
      values.push_back(val.at(r));
      highs.push_back({});
      continue;
    }
    am::Expr *addr = addressAt(o.operands[0], r);
    if (!addr)
      return declined("tt.atomic_rmw", "pointer has no recorded offset");
    const am::Str pn = nameFor('a', o.results[0], r);

    if (plan.strategy == agpu::AtomicStrategy::Packed16) {
      const am::Str hi = pn + "_hi";
      if (!declarePacked16Word(addr, pn, hi))
        return declined("tt.atomic_rmw", "pointer has no recorded offset");
      highs.push_back(hi);
    } else {
      cur_->push_back(
          mc.declStmt(wordPtr, pn, mc.cast(wordPtr, mc.addrOf(addr))));
      highs.push_back({});
    }
    ptrs.push_back(pn);
    values.push_back(val.at(r));
  }

  // Mask is per register.
  agpu::AtomicNames nm;
  nm.result = "old" + std::to_string(o.results[0]) + "_";
  if (plan.election.crossesWarp()) {
    nm.scratch = liveBuffer(agpu::atomicScratchKey(), *elemP);
    if (nm.scratch.empty())
      return declined("tt.atomic_rmw",
                      "a device function cannot declare the broadcast slot");
  }
  const am::SmallVec<am::Str, 8> outs =
      agpu::emitAtomicTensor(mc, *cur_, plan, ptrs, values, nm, highs,
                             [this, &o](int64_t r) { return maskAt(o, 2, r); });
  if (outs.size() != (std::size_t)regs)
    return declined("tt.atomic_rmw", "the emitter refused the plan");

  agpu::ValueNames names;
  for (int64_t r = 0; r < regs; ++r)
    names.push_back(outs[(std::size_t)r]);
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerAtomicHandlers() {
  table_.add("atomic",
             agpu::forOps({"tt.atomic_rmw"}, [this](const agpu::OpView &o) {
               return emitAtomicRmwOp(o);
             }));

  table_.add("cas",
             agpu::forOps({"tt.atomic_cas"}, [this](const agpu::OpView &o) {
               return emitAtomicCasOp(o);
             }));

  table_.add("poll",
             agpu::forOps({"tt.atomic_poll"}, [this](const agpu::OpView &o) {
               return emitAtomicPollOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
