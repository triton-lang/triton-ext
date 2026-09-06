// AtomicPlan.h - how one atomic is lowered and who executes it.
#ifndef AGPU_ATOMIC_PLAN_H
#define AGPU_ATOMIC_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/msl/Builtins.h"
#include "agpu/plan/Elementwise.h"

#include <cstdint>

namespace agpu {

// The RMW operations TTIR can ask for.
enum class RmwOp { Add, FAdd, Max, Min, UMax, UMin, And, Or, Xor, Xchg };

// How the value is stored, which decides which atomics apply.
enum class ElemClass { Int, Float };

// How an atomic reaches the hardware.
enum class AtomicStrategy {
  Native,   // a Metal atomic_fetch_* on the value's own type
  FloatCas, // 32-bit float: a CAS loop over the integer word
  Packed16, // 16-bit float: CAS over the containing 32-bit word, half chosen
  Unsupported,
};

// Not the value's own type: unsigned max is done in unsigned and every float
// strategy goes through a 32-bit word.
enum class AtomicWord { I32, U32, I64, F32 };

enum class EmuRmw : int { Add = 0, Max = 1, Min = 2, Xchg = 3 };

enum class MemOrder { Relaxed, Acquire, Release, AcquireRelease };

enum class WordAccess {
  Direct,   // the value IS the atomic word
  Packed16, // it shares a 32-bit word with its neighbour; a half is selected
  Wide64,   // a 64-bit word
  Unsupported,
};

// What the IR says about one atomic.
struct AtomicFacts {
  RmwOp op = RmwOp::Add;
  ElemClass elem = ElemClass::Int;
  unsigned bits = 32;

  // f16 and bf16 are both "16-bit float" but round differently.
  ElemType packedElem = f16();

  // Bits of each index that do not move the address, so registers differing
  // only in them address the same location.
  unsigned laneFree = 0;
  unsigned warpFree = 0;
  unsigned regFree = 0;

  bool uniformPtr = false; // a scalar pointer: one thread does the work
};

// `strategyFor` and the decline reason read the same row.
struct AtomicRule {
  AtomicStrategy strategy;
  bool (*applies)(const AtomicFacts &);
  // Read only when the strategy is Unsupported.
  const char *because;
};

// Checked in order, first match wins.
inline constexpr AtomicRule kAtomicRules[] = {
    {AtomicStrategy::Unsupported,
     [](const AtomicFacts &f) {
       return f.elem == ElemClass::Int && f.bits == 64;
     },
     "no 64-bit device atomic"},

    {AtomicStrategy::Native,
     [](const AtomicFacts &f) { return f.elem == ElemClass::Int; }, ""},

    // Metal offers a native float atomic only for addition.
    {AtomicStrategy::Native,
     [](const AtomicFacts &f) {
       return f.bits == 32 && (f.op == RmwOp::Add || f.op == RmwOp::FAdd);
     },
     ""},

    // Everything else in float goes to a CAS loop, chosen by width.
    {AtomicStrategy::FloatCas,
     [](const AtomicFacts &f) { return f.bits == 32; }, ""},
    {AtomicStrategy::Packed16,
     [](const AtomicFacts &f) { return f.bits == 16; }, ""},

    {AtomicStrategy::Unsupported, [](const AtomicFacts &) { return true; },
     "no float atomic at this width"},
};

inline const AtomicRule &atomicRuleFor(const AtomicFacts &f) {
  for (const AtomicRule &r : kAtomicRules)
    if (r.applies(f))
      return r;
  return kAtomicRules[sizeof(kAtomicRules) / sizeof(kAtomicRules[0]) - 1];
}

inline AtomicStrategy strategyFor(const AtomicFacts &f) {
  return atomicRuleFor(f).strategy;
}

inline AtomicWord wordFor(const AtomicFacts &f) {
  if (f.elem == ElemClass::Float)
    return AtomicWord::F32;
  if (f.op == RmwOp::UMax || f.op == RmwOp::UMin)
    return AtomicWord::U32;
  return f.bits == 64 ? AtomicWord::I64 : AtomicWord::I32;
}

struct RmwBuiltin {
  RmwOp op;
  const char *fn;
};

inline constexpr RmwBuiltin kRmwBuiltins[] = {
    {RmwOp::Add, msl::builtin::atomic::FetchAdd},
    {RmwOp::FAdd, msl::builtin::atomic::FetchAdd},
    {RmwOp::Max, msl::builtin::atomic::FetchMax},
    {RmwOp::UMax, msl::builtin::atomic::FetchMax},
    {RmwOp::Min, msl::builtin::atomic::FetchMin},
    {RmwOp::UMin, msl::builtin::atomic::FetchMin},
    {RmwOp::And, msl::builtin::atomic::FetchAnd},
    {RmwOp::Or, msl::builtin::atomic::FetchOr},
    {RmwOp::Xor, msl::builtin::atomic::FetchXor},
    {RmwOp::Xchg, msl::builtin::atomic::Exchange},
};

inline const char *builtinFor(RmwOp op) {
  for (const RmwBuiltin &b : kRmwBuiltins)
    if (b.op == op)
      return b.fn;
  return nullptr;
}

inline bool isRenderable(const AtomicFacts &f) {
  return strategyFor(f) != AtomicStrategy::Unsupported &&
         (strategyFor(f) != AtomicStrategy::Native || builtinFor(f.op));
}

inline constexpr int emuRmwCode(EmuRmw op) { return static_cast<int>(op); }

inline EmuRmw emuRmwFor(RmwOp op) {
  switch (op) {
  case RmwOp::Max:
  case RmwOp::UMax:
    return EmuRmw::Max;
  case RmwOp::Min:
  case RmwOp::UMin:
    return EmuRmw::Min;
  case RmwOp::Xchg:
    return EmuRmw::Xchg;
  default:
    return EmuRmw::Add;
  }
}

// ── memory ordering ───────────────────────────────────────────────────────

// Metal device atomics are relaxed-only; acquire/release/acq_rel are not valid
// MSL memory orders. The requested order becomes device-scope fences around
// the relaxed operation.
struct FencePlan {
  bool before = false; // release: prior writes must be visible first
  bool after = false;  // acquire: later reads must not be hoisted above
};

inline FencePlan fencesFor(MemOrder order) {
  switch (order) {
  case MemOrder::Relaxed:
    return {false, false};
  case MemOrder::Release:
    return {true, false};
  case MemOrder::Acquire:
    return {false, true};
  case MemOrder::AcquireRelease:
    return {true, true};
  }
  return {};
}

// ── the other two families ────────────────────────────────────────────────

// How a value reaches its atomic word, independent of the operation.
inline WordAccess accessFor(unsigned bits) {
  switch (bits) {
  case 16:
    return WordAccess::Packed16;
  case 32:
    return WordAccess::Direct;
  case 64:
    return WordAccess::Wide64;
  }
  return WordAccess::Unsupported;
}

// ── who executes it ───────────────────────────────────────────────────────

// Registers differing only in bits that do not move the address are replicas
// of one atomic. `canonicalOf` names the register that owns the location; a
// replica binds to its result.
struct ReplicaMap {
  unsigned regFree = 0;

  int canonicalOf(int reg) const { return reg & ~(int)regFree; }
  bool isReplica(int reg) const {
    return regFree != 0 && (reg & (int)regFree) != 0;
  }
};

// Split out of AtomicFacts because a plain store asks the same question.
struct AddressSpread {
  // Bits of each index that do not move the address, so threads differing
  // only in them address the same location.
  unsigned laneFree = 0;
  unsigned warpFree = 0;
  bool uniformPtr = false; // a scalar pointer: one thread does the work
};

// Exactly one thread runs: the one whose free bits are zero.
struct ThreadElection {
  bool needsLaneTest = false;
  unsigned laneMask = 0;
  bool needsWarpTest = false;
  unsigned warpMask = 0;
  bool firstThreadOnly = false; // a uniform pointer: thread 0 of the group

  bool any() const { return needsLaneTest || needsWarpTest || firstThreadOnly; }

  // An excluded thread holds no result, so a use of one has to read the
  // winner's. Thread 0 of the group is reachable only through threadgroup
  // memory; a lane or warp election stays inside the warp, where a shuffle
  // carries it.
  bool crossesWarp() const { return firstThreadOnly; }
};

inline ThreadElection electFor(const AddressSpread &s) {
  ThreadElection e;
  if (s.uniformPtr) {
    e.firstThreadOnly = true;
    return e;
  }
  e.needsLaneTest = s.laneFree != 0;
  e.laneMask = s.laneFree;
  e.needsWarpTest = s.warpFree != 0;
  e.warpMask = s.warpFree;
  return e;
}

inline ThreadElection electFor(const AtomicFacts &f) {
  return electFor(AddressSpread{f.laneFree, f.warpFree, f.uniformPtr});
}

// Read from the row that decided the strategy. The operation check is
// separate: an accepted shape can still name an operation with no builtin.
inline const char *declineReason(const AtomicFacts &f) {
  const AtomicRule &r = atomicRuleFor(f);
  if (r.strategy == AtomicStrategy::Unsupported)
    return r.because;
  if (!builtinFor(f.op))
    return "no builtin for this operation";
  return "unsupported atomic";
}

struct AtomicPlan {
  AtomicStrategy strategy = AtomicStrategy::Unsupported;
  AtomicWord word = AtomicWord::I32;
  const char *builtin = nullptr;
  FencePlan fences;
  ReplicaMap replicas;
  ThreadElection election;

  // Unused on the native path.
  EmuRmw emuOp = EmuRmw::Add;

  // Must be named at the call: `__agpu_atomic_rmw_packed16` uses `T` in its
  // return type and body only, so nothing deduces it from the arguments.
  ElemType packedElem = f16();

  // The packed path's CAS loop runs over a 32-bit word while
  // `__agpu_atomic_rmw_packed16<T>` returns the 16-bit element.
  bool resultIsElement() const { return strategy == AtomicStrategy::Packed16; }

  bool usable() const { return strategy != AtomicStrategy::Unsupported; }

  Decision decision(const AtomicFacts &f) const {
    if (usable())
      return Decision::emitted();
    return Decision::declined("emitAtomicRMW", declineReason(f));
  }
};

inline AtomicPlan planAtomic(const AtomicFacts &f, MemOrder order) {
  AtomicPlan p;
  p.strategy = strategyFor(f);
  if (!p.usable())
    return p;

  p.word = wordFor(f);
  p.builtin = builtinFor(f.op);
  if (p.strategy == AtomicStrategy::Native && !p.builtin) {
    p.strategy = AtomicStrategy::Unsupported;
    return p;
  }

  p.packedElem = f.packedElem;

  // The emulated paths go through prelude helpers with their own ordering.
  if (p.strategy == AtomicStrategy::Native)
    p.fences = fencesFor(order);

  p.replicas = ReplicaMap{f.regFree};
  p.election = electFor(f);
  p.emuOp = emuRmwFor(f.op);
  return p;
}

} // namespace agpu

#endif // AGPU_ATOMIC_PLAN_H
