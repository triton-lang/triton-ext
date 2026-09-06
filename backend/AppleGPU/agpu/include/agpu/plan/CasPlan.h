// CasPlan.h - compare-and-exchange and who performs it.
//
// Metal has no 16-bit compare-exchange, so a 16-bit CAS is a 32-bit CAS on the
// containing word with the other half preserved and must retry. Float goes
// through the bit pattern.
#ifndef AGPU_CAS_PLAN_H
#define AGPU_CAS_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/plan/AtomicPlan.h"
#include "agpu/plan/Elementwise.h"

#include <cstdint>

namespace agpu {

// What the IR says about one compare-exchange.
struct CasFacts {
  ElemClass elem = ElemClass::Int;
  unsigned bits = 32;

  // A scalar pointer: every thread would target the same location.
  bool uniformPtr = false;

  MemOrder order = MemOrder::Relaxed;
};

// How the exchange is performed.
enum class CasStrategy {
  Word32,   // a native 32-bit integer compare-exchange
  Packed16, // a 32-bit CAS on the containing word, preserving the other half
  Unsupported,
};

struct CasPlan {
  CasStrategy strategy = CasStrategy::Unsupported;

  // The value reaches the exchange word by reinterpreting its bits. Also the
  // only correct comparison for a float: by value, -0.0 and 0.0 are one
  // expected value and NaN matches nothing.
  bool viaBits = false;

  // A uniform pointer targets one location, so one thread performs it.
  bool electOne = false;

  // Metal device atomics are relaxed-only: acquire is a fence after, release
  // a fence before.
  FencePlan fences;

  bool usable() const { return strategy != CasStrategy::Unsupported; }

  // Both usable strategies retry: the packed form's exchange fails when the
  // other half changed and `compare_exchange_weak` may fail spuriously.
  bool retries() const { return usable(); }
};

inline CasPlan planCas(const CasFacts &f) {
  CasPlan p;
  p.electOne = f.uniformPtr;
  p.fences = fencesFor(f.order);
  p.viaBits = f.elem == ElemClass::Float;

  switch (accessFor(f.bits)) {
  case WordAccess::Packed16:
    p.strategy = CasStrategy::Packed16;
    break;
  case WordAccess::Direct:
    p.strategy = CasStrategy::Word32;
    break;
  case WordAccess::Wide64:
  case WordAccess::Unsupported:
    break;
  }
  return p;
}

inline Decision casDecision(const CasPlan &p) {
  if (p.usable())
    return Decision::emitted();
  return Decision::declined("emitAtomicCAS",
                            "no compare-exchange at this width");
}

} // namespace agpu

#endif // AGPU_CAS_PLAN_H
