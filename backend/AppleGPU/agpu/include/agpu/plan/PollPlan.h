// PollPlan.h - waiting for another threadgroup to publish a value.
//
// A cross-threadgroup handoff is a store, a device barrier and a load. The
// load side is a poll: spin on a flag until the producer's value appears.
//
// One thread spins. The load must be atomic and
// the pointer volatile or the compiler hoists it. The rest of the
// threadgroup waits on a hard barrier.
#ifndef AGPU_POLL_PLAN_H
#define AGPU_POLL_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/plan/AtomicPlan.h"
#include "agpu/plan/Elementwise.h"

#include <cstdint>

namespace agpu {

// What the IR says about one poll.
struct PollFacts {
  unsigned bits = 32;   // width of the flag
  bool acquire = false; // the poll carries acquire semantics
  // A timeout poll tests once and reports; a plain one spins.
  bool hasTimeout = false;
};

// Metal's atomic loads are 32-bit only. AtomicWord is
// `atomic_load_explicit` on the flag's own word; PackedHalf reads a 16-bit
// flag out of its containing 32-bit word; VolatileWide derefs a volatile
// device pointer, an aligned 64-bit load being single-copy on Apple GPUs.
enum class PollLoad { AtomicWord, PackedHalf, VolatileWide };

// How the poll is performed.
struct PollPlan {
  // The word the flag lives in.
  msl::Scalar word = msl::Scalar::U32;
  PollLoad load = PollLoad::AtomicWord;

  // Whether the loop spins or the test runs once.
  bool spins = true;

  // Acquire ordering on the barrier, when the IR asked for it.
  bool acquire = false;

  bool usable = false;
};

inline PollPlan planPoll(const PollFacts &f) {
  PollPlan p;
  p.acquire = f.acquire;
  p.spins = !f.hasTimeout;

  // A poll only loads, so unlike the CAS path it can read a 64-bit word.
  switch (accessFor(f.bits)) {
  case WordAccess::Packed16:
    p.word = msl::Scalar::U32;
    p.load = PollLoad::PackedHalf;
    break;
  case WordAccess::Direct:
    p.word = msl::Scalar::U32;
    p.load = PollLoad::AtomicWord;
    break;
  case WordAccess::Wide64:
    p.word = msl::Scalar::U64;
    p.load = PollLoad::VolatileWide;
    break;
  case WordAccess::Unsupported:
    return p;
  }
  p.usable = true;
  return p;
}

inline Decision pollDecision(const PollPlan &p) {
  if (p.usable)
    return Decision::emitted();
  return Decision::declined("emitAtomicPoll", "no flag load at this width");
}

} // namespace agpu

#endif // AGPU_POLL_PLAN_H
