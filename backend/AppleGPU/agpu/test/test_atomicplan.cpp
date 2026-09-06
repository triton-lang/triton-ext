// Atomic lowering decisions: strategy, builtin name, replicas, thread
// election.
#include "agpu/plan/AtomicPlan.h"
#include "harness.h"

#include <set>

using namespace agpu;

namespace {

AtomicFacts intAtomic(RmwOp op, unsigned bits = 32) {
  AtomicFacts f;
  f.op = op;
  f.elem = ElemClass::Int;
  f.bits = bits;
  return f;
}

AtomicFacts floatAtomic(RmwOp op, unsigned bits = 32) {
  AtomicFacts f;
  f.op = op;
  f.elem = ElemClass::Float;
  f.bits = bits;
  return f;
}

const RmwOp kAllOps[] = {RmwOp::Add,  RmwOp::FAdd, RmwOp::Max, RmwOp::Min,
                         RmwOp::UMax, RmwOp::UMin, RmwOp::And, RmwOp::Or,
                         RmwOp::Xor,  RmwOp::Xchg};

} // namespace

int main() {
  // ── the strategy and its reason are one row ────────────────────────────

  CASE("a declined atomic's reason comes from the rule that declined it");
  {
    const ElemClass elems[] = {ElemClass::Int, ElemClass::Float};
    const unsigned widths[] = {8, 16, 32, 64};
    const RmwOp ops[] = {RmwOp::Add, RmwOp::Max, RmwOp::Min, RmwOp::Xchg,
                         RmwOp::And, RmwOp::Or,  RmwOp::Xor};

    for (ElemClass e : elems)
      for (unsigned w : widths)
        for (RmwOp op : ops) {
          AtomicFacts f;
          f.elem = e;
          f.bits = w;
          f.op = op;

          const AtomicRule &r = atomicRuleFor(f);
          CHECK(r.strategy == strategyFor(f));

          if (r.strategy == AtomicStrategy::Unsupported) {
            CHECK_EQ(std::string(declineReason(f)), std::string(r.because));
            CHECK(std::string(declineReason(f)) !=
                  std::string("unsupported atomic"));
          }
        }
  }

  CASE("every rule that can decline carries a reason");
  {
    for (const AtomicRule &r : kAtomicRules)
      if (r.strategy == AtomicStrategy::Unsupported)
        CHECK(r.because != nullptr && *r.because != '\0');
  }

  // ── strategy ───────────────────────────────────────────────────────────

  CASE("integer atomics are native at 32 bits");
  {
    for (RmwOp op : kAllOps)
      CHECK(strategyFor(intAtomic(op)) == AtomicStrategy::Native);
  }

  CASE("64-bit integer atomics are unsupported");
  {
    // Metal has no 64-bit device atomic.
    CHECK(strategyFor(intAtomic(RmwOp::Add, 64)) ==
          AtomicStrategy::Unsupported);
  }

  CASE("float add is native, everything else in float is a CAS loop");
  {
    CHECK(strategyFor(floatAtomic(RmwOp::Add)) == AtomicStrategy::Native);
    CHECK(strategyFor(floatAtomic(RmwOp::FAdd)) == AtomicStrategy::Native);
    CHECK(strategyFor(floatAtomic(RmwOp::Max)) == AtomicStrategy::FloatCas);
    CHECK(strategyFor(floatAtomic(RmwOp::Min)) == AtomicStrategy::FloatCas);
    CHECK(strategyFor(floatAtomic(RmwOp::Xchg)) == AtomicStrategy::FloatCas);
  }

  CASE("a 16-bit float shares its atomic word with its neighbour");
  {
    // No 16-bit atomic exists; the CAS is over the containing 32-bit word.
    CHECK(strategyFor(floatAtomic(RmwOp::Add, 16)) == AtomicStrategy::Packed16);
    CHECK(strategyFor(floatAtomic(RmwOp::Max, 16)) == AtomicStrategy::Packed16);
  }

  CASE("an unsupported float width declines");
  {
    CHECK(strategyFor(floatAtomic(RmwOp::Add, 64)) ==
          AtomicStrategy::Unsupported);
    CHECK(strategyFor(floatAtomic(RmwOp::Max, 8)) ==
          AtomicStrategy::Unsupported);
  }

  // ── the op table ───────────────────────────────────────────────────────

  CASE("every op the strategy admits is renderable");
  {
    for (RmwOp op : kAllOps) {
      CHECK(isRenderable(intAtomic(op)));
      CHECK(builtinFor(op) != nullptr);
    }
  }

  CASE("unsigned min and max name the signed builtin on an unsigned word");
  {
    // Metal has no atomic_fetch_umax: the distinction is carried by the word
    // type.
    CHECK_EQ(std::string(builtinFor(RmwOp::UMax)),
             std::string(builtinFor(RmwOp::Max)));
    CHECK(wordFor(intAtomic(RmwOp::UMax)) == AtomicWord::U32);
    CHECK(wordFor(intAtomic(RmwOp::Max)) == AtomicWord::I32);
  }

  CASE("every float strategy operates on a 32-bit word");
  {
    CHECK(wordFor(floatAtomic(RmwOp::Add)) == AtomicWord::F32);
    CHECK(wordFor(floatAtomic(RmwOp::Max, 16)) == AtomicWord::F32);
  }

  // ── fences ─────────────────────────────────────────────────────────────

  CASE("relaxed needs no fence");
  {
    FencePlan f = fencesFor(MemOrder::Relaxed);
    CHECK(!f.before);
    CHECK(!f.after);
  }

  CASE("release fences before, acquire after");
  {
    // Metal device atomics are relaxed-only, so order is carried by fences.
    CHECK(fencesFor(MemOrder::Release).before);
    CHECK(!fencesFor(MemOrder::Release).after);
    CHECK(!fencesFor(MemOrder::Acquire).before);
    CHECK(fencesFor(MemOrder::Acquire).after);
  }

  CASE("the plan puts acquire-release fences on both sides");
  {
    FencePlan f = fencesFor(MemOrder::AcquireRelease);
    CHECK(f.before);
    CHECK(f.after);
  }

  CASE("only the native path carries fences");
  {
    AtomicPlan native = planAtomic(intAtomic(RmwOp::Add), MemOrder::Acquire);
    CHECK(native.fences.after);

    AtomicPlan cas = planAtomic(floatAtomic(RmwOp::Max), MemOrder::Acquire);
    CHECK(cas.strategy == AtomicStrategy::FloatCas);
    CHECK(!cas.fences.after);
    CHECK(!cas.fences.before);
  }

  // ── replicas ───────────────────────────────────────────────────────────

  CASE("with no free register bits every register is its own atomic");
  {
    ReplicaMap m{0};
    for (int r = 0; r < 8; ++r) {
      CHECK(!m.isReplica(r));
      CHECK_EQ(m.canonicalOf(r), r);
    }
  }

  CASE("registers differing only in free bits share one atomic");
  {
    // Free bit 1: registers 0/2 are one location, 1/3 another.
    ReplicaMap m{0b10};
    CHECK(!m.isReplica(0));
    CHECK(!m.isReplica(1));
    CHECK(m.isReplica(2));
    CHECK(m.isReplica(3));
    CHECK_EQ(m.canonicalOf(2), 0);
    CHECK_EQ(m.canonicalOf(3), 1);
  }

  CASE("canonical registers are exactly the non-replicas");
  {
    for (unsigned free : {0u, 0b1u, 0b10u, 0b11u, 0b101u}) {
      ReplicaMap m{free};
      std::set<int> canonical;
      for (int r = 0; r < 16; ++r) {
        const int c = m.canonicalOf(r);
        CHECK(!m.isReplica(c));
        CHECK_EQ(m.canonicalOf(c), c);
        canonical.insert(c);
      }
      CHECK_EQ((int)canonical.size(), 16 >> __builtin_popcount(free));
    }
  }

  // ── thread election ────────────────────────────────────────────────────

  CASE("an address that every lane bit moves needs no election");
  {
    AtomicFacts f = intAtomic(RmwOp::Add);
    ThreadElection e = electFor(f);
    CHECK(!e.any());
  }

  CASE("a lane bit that does not move the address elects one lane");
  {
    AtomicFacts f = intAtomic(RmwOp::Add);
    f.laneFree = 0b11;
    ThreadElection e = electFor(f);
    CHECK(e.needsLaneTest);
    CHECK_EQ(e.laneMask, 0b11u);
    CHECK(!e.needsWarpTest);
  }

  CASE("lane and warp elections compose");
  {
    AtomicFacts f = intAtomic(RmwOp::Add);
    f.laneFree = 0b1;
    f.warpFree = 0b10;
    ThreadElection e = electFor(f);
    CHECK(e.needsLaneTest);
    CHECK(e.needsWarpTest);
    CHECK_EQ(e.warpMask, 0b10u);
  }

  CASE("a uniform pointer elects thread 0 and ignores the layout masks");
  {
    AtomicFacts f = intAtomic(RmwOp::Add);
    f.uniformPtr = true;
    f.laneFree = 0b11111;
    ThreadElection e = electFor(f);
    CHECK(e.firstThreadOnly);
    CHECK(!e.needsLaneTest);
  }

  CASE("only a group-wide election needs its result published through memory");
  {
    AtomicFacts uniform = intAtomic(RmwOp::Add);
    uniform.uniformPtr = true;
    CHECK(electFor(uniform).crossesWarp());

    AtomicFacts lanes = intAtomic(RmwOp::Add);
    lanes.laneFree = 0b11111;
    CHECK(electFor(lanes).any());
    CHECK(!electFor(lanes).crossesWarp());

    CHECK(!electFor(intAtomic(RmwOp::Add)).crossesWarp());
  }

  // ── the whole plan ─────────────────────────────────────────────────────

  CASE("an unsupported atomic carries no builtin to misuse");
  {
    AtomicPlan p = planAtomic(intAtomic(RmwOp::Add, 64), MemOrder::Relaxed);
    CHECK(!p.usable());
    CHECK(p.builtin == nullptr);
  }

  CASE("a usable plan always names something to emit");
  {
    for (RmwOp op : kAllOps)
      for (unsigned bits : {16u, 32u})
        for (ElemClass ec : {ElemClass::Int, ElemClass::Float}) {
          AtomicFacts f;
          f.op = op;
          f.elem = ec;
          f.bits = bits;
          AtomicPlan p = planAtomic(f, MemOrder::Relaxed);
          if (!p.usable())
            continue;
          if (p.strategy == AtomicStrategy::Native)
            CHECK(p.builtin != nullptr);
        }
  }

  // ── the other two families ─────────────────────────────────────────────

  CASE("width classification is one answer for all three families");
  {
    CHECK(accessFor(16) == WordAccess::Packed16);
    CHECK(accessFor(32) == WordAccess::Direct);
    CHECK(accessFor(64) == WordAccess::Wide64);
    CHECK(accessFor(8) == WordAccess::Unsupported);
    CHECK(accessFor(0) == WordAccess::Unsupported);
  }

  CASE("Wide64 is the one access a CAS cannot use and a poll can");
  {
    // A poll only loads, so `Wide64` is readable; no 64-bit compare-exchange
    // exists, so `CasPlan` leaves its strategy unset on that same row.
    CHECK(accessFor(64) == WordAccess::Wide64);
    for (unsigned bits = 0; bits <= 128; ++bits) {
      const WordAccess a = accessFor(bits);
      const bool readable = a != WordAccess::Unsupported;
      const bool writable =
          a == WordAccess::Direct || a == WordAccess::Packed16;
      if (writable)
        CHECK(readable);
      if (readable && !writable)
        CHECK(a == WordAccess::Wide64);
    }
  }

  // ── declining with a reason ────────────────────────────────────────────

  CASE("an unsupported atomic declines, naming the width it cannot do");
  {
    AtomicFacts f = intAtomic(RmwOp::Add, 64);
    Decision d = planAtomic(f, MemOrder::Relaxed).decision(f);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK(!d.keepLooking());
    CHECK_EQ(d.why(), std::string("no 64-bit device atomic"));
  }

  CASE("a float width with no atomic declines distinctly");
  {
    AtomicFacts f = floatAtomic(RmwOp::Add, 64);
    Decision d = planAtomic(f, MemOrder::Relaxed).decision(f);
    CHECK(d.isDecline());
    CHECK_EQ(d.why(), std::string("no float atomic at this width"));
  }

  CASE("a usable atomic plan reports no decline");
  {
    AtomicFacts f = intAtomic(RmwOp::Add);
    CHECK(planAtomic(f, MemOrder::Relaxed).decision(f).ok());
  }

  return ::agpu_test::report("AtomicPlan");
}
