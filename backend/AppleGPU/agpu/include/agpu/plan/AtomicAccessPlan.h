// AtomicAccessPlan.h - an atomic that reads or writes a location without
// deriving the new value from the old one.
//
// A load is idempotent, so every thread holding a location can issue its own;
// a store writes a value that does not depend on what was there, so threads
// sharing a location agree. Neither needs the election or the broadcast a
// read-modify-write does.
#ifndef AGPU_ATOMIC_ACCESS_PLAN_H
#define AGPU_ATOMIC_ACCESS_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/plan/AtomicPlan.h"
#include "agpu/plan/ElemType.h"

namespace agpu {

enum class AtomicAccess { Load, Store };

// A value narrower than the atomic word is reached through the word that
// contains it: mask the pointer to the word, then select the part.
enum class SubWord { None, Byte, Half };

struct AtomicAccessFacts {
  AtomicAccess kind = AtomicAccess::Load;
  ElemType elem;
};

struct AtomicAccessPlan {
  AtomicAccess kind = AtomicAccess::Load;
  ElemType elem;
  msl::Scalar word = msl::Scalar::U32;
  SubWord sub = SubWord::None;
  // Metal has no 64-bit atomic load or store. An aligned 64-bit access is
  // single-copy on Apple GPUs, so it goes through a volatile pointer.
  bool wide = false;
  FencePlan fences;
  bool usable = false;
};

inline SubWord subWordFor(unsigned bits) {
  switch (bits) {
  case 1:
  case 8:
    return SubWord::Byte;
  case 16:
    return SubWord::Half;
  }
  return SubWord::None;
}

inline AtomicAccessPlan planAtomicAccess(const AtomicAccessFacts &f,
                                         MemOrder order) {
  AtomicAccessPlan p;
  p.kind = f.kind;
  p.elem = f.elem;
  p.fences = fencesFor(order);
  p.sub = subWordFor(f.elem.bits);

  switch (f.elem.bits) {
  case 1:
  case 8:
  case 16:
  case 32:
    p.word = msl::Scalar::U32;
    break;
  case 64:
    p.word = msl::Scalar::U64;
    p.wide = true;
    break;
  default:
    return p;
  }
  p.usable = true;
  return p;
}

inline Decision atomicAccessDecision(const AtomicAccessPlan &p) {
  if (p.usable)
    return Decision::emitted();
  return Decision::declined(p.kind == AtomicAccess::Load ? "emitAtomicLoad"
                                                         : "emitAtomicStore",
                            "no atomic access at this width");
}

} // namespace agpu

#endif // AGPU_ATOMIC_ACCESS_PLAN_H
