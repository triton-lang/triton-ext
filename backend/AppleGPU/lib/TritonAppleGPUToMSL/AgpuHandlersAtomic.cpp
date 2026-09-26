// Atomic handlers: read-modify-write, compare-and-swap, poll, load, store.
#include "AgpuEmitter.h"
#include "AgpuEnums.h"

#include "agpu/emit/EmitAtomicAccess.h"
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

// A byte or half is reached through its containing word: mask the pointer to
// the word and shift by the offset the address had within it.
bool AgpuEmitter::declareSubWord(am::Expr *addr, agpu::SubWord sub,
                                 const am::Str &wordName,
                                 const am::Str &shiftName) {
  if (!addr)
    return false;
  am::Context &c = agpu_.context();

  const am::Type sizeT = am::Type::scalar(am::Scalar::U64);
  const am::Type wordPtr = am::Type::named(am::builtin::atomic::Uint)
                               .pointerTo(am::AddrSpace::Device);

  am::Expr *asInt = c.cast(sizeT, c.addrOf(addr));
  const int64_t within = sub == agpu::SubWord::Byte ? 3 : 2;

  cur_->push_back(c.declStmt(
      am::Type::scalar(am::Scalar::U32), shiftName,
      c.binary(am::BinOp::Shl,
               c.cast(am::Type::scalar(am::Scalar::U32),
                      c.binary(am::BinOp::And, asInt, c.lit(within))),
               c.lit(3))));
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
  f.hasTimeout = poll.getTimeout() ? true : false;

  const agpu::PollPlan plan = agpu::planPoll(f);
  if (!plan.usable)
    return pollDecision(plan);

  const Value ptrV = mlirValueOf(o.operands[0]);
  auto ptrTy =
      ptrV ? dyn_cast<RankedTensorType>(ptrV.getType()) : RankedTensorType();
  const int64_t regs = ptrTy ? registerCount(ptrTy) : 1;

  const Ready ready =
      readyForCounted(o, 0, 2, regs, "an operand has no register name");
  if (!ready.ok())
    return ready.why;
  const Operand &expected = ready.ops[1];

  agpu::ReplicaMap replicas;
  if (ptrTy)
    replicas.regFree = freeBitsOf(gpu::toLinearLayout(ptrTy),
                                  ptrTy.getContext(), lldim::Register);

  // A poll only loads, so replicated threads can read their own element
  // instead of waiting on an elected one. Only a uniform pointer, whose
  // answer has to cross warps, still elects.
  agpu::AddressSpread spread = spreadOf(ptrV);
  spread.laneFree = 0;
  spread.warpFree = 0;
  const agpu::ThreadElection election = agpu::electFor(spread);

  agpu::PollNames nm;
  const std::string tag = std::to_string(o.results[0]) + body_.scope;
  nm.ptr = "flagp" + tag;
  nm.expected = "want" + tag;
  nm.result = "ready" + tag;
  nm.flag = "seen" + tag;

  const am::Type wantTy = agpu::mslTypeOf(*want);
  am::SmallVec<am::Str, 8> ptrs, expecteds, highs;
  for (int64_t r = 0; r < regs; ++r) {
    if (replicas.isReplica((int)r)) {
      ptrs.push_back({});
      expecteds.push_back({});
      highs.push_back({});
      continue;
    }
    am::Expr *addr = addressAt(o.operands[0], r);
    if (!addr)
      return declined("tt.atomic_poll", "pointer has no recorded offset");

    const am::Str pn = nm.ptr + "_" + std::to_string(r);
    am::Str isHigh;
    if (plan.load == agpu::PollLoad::PackedHalf) {
      isHigh = "hi" + tag + "_" + std::to_string(r);
      if (!declarePacked16Word(addr, pn, isHigh))
        return declined("tt.atomic_poll", "pointer has no recorded offset");
    } else {
      const am::Type wordPtr = agpu::pollPtrType(plan);
      cur_->push_back(
          mc.declStmt(wordPtr, pn, mc.cast(wordPtr, mc.addrOf(addr))));
    }

    const am::Str en = nm.expected + "_" + std::to_string(r);
    cur_->push_back(mc.declStmt(wantTy, en, mc.var(expected.at(r))));

    ptrs.push_back(pn);
    expecteds.push_back(en);
    highs.push_back(isHigh);
  }

  const am::SmallVec<am::Str, 8> outs = agpu::emitPollTensor(
      mc, *cur_, plan, nm, ptrs, expecteds, replicas, election, highs,
      [&](int64_t) { return agpu::electionExpr(mc, election, nm); });
  if (outs.size() != (std::size_t)regs)
    return declined("tt.atomic_poll", "the emitter refused the plan");

  if (!ptrTy) {
    body_.sym.bindScalar(o.results[0], outs[0]);
    return agpu::Decision::emitted();
  }

  agpu::ValueNames names;
  for (int64_t r = 0; r < regs; ++r)
    names.push_back(outs[(std::size_t)r]);
  body_.sym.bindRegs(o.results[0], std::move(names));
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

agpu::Decision AgpuEmitter::emitAtomicAccessOp(const agpu::OpView &o,
                                               agpu::AtomicAccess kind) {
  am::Context &mc = agpu_.context();
  const bool isLoad = kind == agpu::AtomicAccess::Load;
  const char *what = isLoad ? "tt.atomic_load" : "tt.atomic_store";

  const agpu::ValueId elemOwner = isLoad ? o.results[0] : o.operands[1];
  const agpu::ElemType *elemP = elemOf(elemOwner);
  if (!elemP)
    return declined(what, "the value type was never recorded");

  agpu::AtomicAccessFacts f;
  f.kind = kind;
  f.elem = *elemP;

  const agpu::AtomicAccessPlan plan =
      agpu::planAtomicAccess(f, memOrderOf((triton::MemSemantic)o.intAt(0)));
  if (!plan.usable)
    return agpu::atomicAccessDecision(plan);

  const Value ptrV = mlirValueOf(o.operands[0]);
  auto ptrTy =
      ptrV ? dyn_cast<RankedTensorType>(ptrV.getType()) : RankedTensorType();
  const int64_t regs = ptrTy ? registerCount(ptrTy) : 1;

  const std::size_t maskIndex = isLoad ? 1 : 2;
  Ready ready;
  if (!isLoad) {
    ready = readyForCounted(o, 1, 2, regs, "the value has no register names");
    if (!ready.ok())
      return ready.why;
  }

  agpu::AtomicAccessNames nm;
  const std::string tag =
      std::to_string(isLoad ? o.results[0] : o.operands[0]) + body_.scope;
  nm.result = (isLoad ? "atl" : "ats") + tag;

  const am::Type ptrTypeOfPlan = agpu::atomicAccessPtrType(plan);
  agpu::ValueNames names;

  agpu::emitAtomicAccessFenceBefore(mc, *cur_, plan);

  for (int64_t r = 0; r < regs; ++r) {
    am::Expr *addr = addressAt(o.operands[0], r);
    if (!addr)
      return declined(what, "pointer has no recorded offset");

    const am::Str pn = nm.result + "_p" + std::to_string(r);
    agpu::AtomicAccessNames rn = nm;
    rn.result = nm.result + "_" + std::to_string(r);
    rn.shift = nm.result + "_sh" + std::to_string(r);

    if (plan.sub != agpu::SubWord::None) {
      if (!declareSubWord(addr, plan.sub, pn, rn.shift))
        return declined(what, "pointer has no recorded offset");
    } else {
      cur_->push_back(mc.declStmt(ptrTypeOfPlan, pn,
                                  mc.cast(ptrTypeOfPlan, mc.addrOf(addr))));
    }

    am::Expr *guard = maskAt(o, maskIndex, r);
    if (isLoad) {
      cur_->push_back(
          mc.declStmt(agpu::mslTypeOf(*elemP), rn.result, mc.lit(0)));
      am::Block body;
      body.push_back(mc.assign(mc.var(rn.result),
                               agpu::atomicLoadValue(mc, plan, pn, rn)));
      mc.guardedInto(*cur_, guard, std::move(body));
      names.push_back(rn.result);
      continue;
    }

    am::Block body;
    agpu::emitAtomicStoreValue(mc, body, plan, pn, ready.ops[1].at(r), rn);
    mc.guardedInto(*cur_, guard, std::move(body));
  }

  agpu::emitAtomicAccessFenceAfter(mc, *cur_, plan);

  if (!isLoad)
    return agpu::Decision::emitted();

  if (!ptrTy) {
    body_.sym.bindScalar(o.results[0], names[0]);
    return agpu::Decision::emitted();
  }
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerAtomicHandlers() {
  table_.add("atomic",
             agpu::forOps({"tt.atomic_rmw"}, [this](const agpu::OpView &o) {
               return emitAtomicRmwOp(o);
             }));

  table_.add("atomic_access",
             agpu::forOps({"tt.atomic_load"}, [this](const agpu::OpView &o) {
               return emitAtomicAccessOp(o, agpu::AtomicAccess::Load);
             }));

  table_.add("atomic_access_store",
             agpu::forOps({"tt.atomic_store"}, [this](const agpu::OpView &o) {
               return emitAtomicAccessOp(o, agpu::AtomicAccess::Store);
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
