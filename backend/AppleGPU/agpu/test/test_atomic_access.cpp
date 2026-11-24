// An atomic load or store: no election, and sub-word access through the
// containing word.
#include "agpu/emit/EmitAtomicAccess.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

ElemType intOf(unsigned bits) {
  ElemType e;
  e.kind = ElemType::Kind::Int;
  e.bits = bits;
  return e;
}

ElemType floatOf(unsigned bits) {
  ElemType e;
  e.kind = ElemType::Kind::Float;
  e.bits = bits;
  return e;
}

AtomicAccessFacts accessOf(ElemType e, AtomicAccess k = AtomicAccess::Load) {
  AtomicAccessFacts f;
  f.kind = k;
  f.elem = e;
  return f;
}

} // namespace

int main() {
  AtomicAccessNames nm;

  CASE("a 32-bit access is the word itself");
  {
    AtomicAccessPlan p =
        planAtomicAccess(accessOf(intOf(32)), MemOrder::Relaxed);
    CHECK(p.usable);
    CHECK(p.sub == SubWord::None);
    CHECK(!p.wide);
  }

  CASE("a byte and a half go through the containing word");
  {
    for (unsigned bits : {1u, 8u}) {
      AtomicAccessPlan p =
          planAtomicAccess(accessOf(intOf(bits)), MemOrder::Relaxed);
      CHECK(p.usable);
      CHECK(p.sub == SubWord::Byte);
      CHECK(p.word == msl::Scalar::U32);
    }
    AtomicAccessPlan h =
        planAtomicAccess(accessOf(intOf(16)), MemOrder::Relaxed);
    CHECK(h.sub == SubWord::Half);
    CHECK(h.word == msl::Scalar::U32);
  }

  CASE("a 64-bit access has no atomic, so it derefs a volatile pointer");
  {
    AtomicAccessPlan p =
        planAtomicAccess(accessOf(intOf(64)), MemOrder::Relaxed);
    CHECK(p.usable);
    CHECK(p.wide);
    CHECK(p.word == msl::Scalar::U64);

    msl::Context c;
    msl::Block body;
    body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::U64), "v",
                              atomicLoadValue(c, p, "p", nm)));
    const std::string out = render(body);
    CHECK_LACKS(out, "atomic_load_explicit");
    CHECK_HAS(out, "*p");
  }

  CASE("a width with no atomic access declines");
  {
    for (unsigned bits : {24u, 128u}) {
      AtomicAccessPlan p =
          planAtomicAccess(accessOf(intOf(bits)), MemOrder::Relaxed);
      CHECK(!p.usable);
      CHECK(atomicAccessDecision(p).isDecline());
    }
  }

  CASE("the load is relaxed, because the fences carry the ordering");
  {
    msl::Context c;
    msl::Block body;
    AtomicAccessPlan p =
        planAtomicAccess(accessOf(intOf(32)), MemOrder::Acquire);
    body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::I32), "v",
                              atomicLoadValue(c, p, "p", nm)));
    const std::string out = render(body);
    CHECK_HAS(out, "memory_order_relaxed");
    CHECK_LACKS(out, "memory_order_seq_cst");
  }

  CASE("a load fences before, acquire also after");
  {
    msl::Context c;
    msl::Block body;
    const AtomicAccessPlan relaxed =
        planAtomicAccess(accessOf(intOf(32)), MemOrder::Relaxed);
    emitAtomicAccessFenceBefore(c, body, relaxed);
    emitAtomicAccessFenceAfter(c, body, relaxed);
    CHECK_EQ(countOf(render(body), "atomic_thread_fence"), 1);

    msl::Block acq;
    const AtomicAccessPlan acquire =
        planAtomicAccess(accessOf(intOf(32)), MemOrder::Acquire);
    emitAtomicAccessFenceBefore(c, acq, acquire);
    emitAtomicAccessFenceAfter(c, acq, acquire);
    CHECK_EQ(countOf(render(acq), "atomic_thread_fence"), 2);
    CHECK_LACKS(render(acq), "threadgroup_barrier");
  }

  CASE("a store fences only before, and only on release");
  {
    msl::Context c;
    msl::Block body;
    const AtomicAccessPlan relaxed = planAtomicAccess(
        accessOf(intOf(32), AtomicAccess::Store), MemOrder::Relaxed);
    emitAtomicAccessFenceBefore(c, body, relaxed);
    emitAtomicAccessFenceAfter(c, body, relaxed);
    CHECK(body.empty());

    const AtomicAccessPlan release = planAtomicAccess(
        accessOf(intOf(32), AtomicAccess::Store), MemOrder::Release);
    emitAtomicAccessFenceBefore(c, body, release);
    emitAtomicAccessFenceAfter(c, body, release);
    CHECK_EQ(countOf(render(body), "atomic_thread_fence"), 1);
  }

  CASE("a sub-word load shifts and masks its part out");
  {
    msl::Context c;
    msl::Block body;
    AtomicAccessPlan p =
        planAtomicAccess(accessOf(intOf(8)), MemOrder::Relaxed);
    body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::I8), "v",
                              atomicLoadValue(c, p, "p", nm)));
    const std::string out = render(body);
    CHECK_HAS(out, ">> sh");
    CHECK_HAS(out, "& 255");
  }

  CASE("a float is reinterpreted, never converted");
  {
    msl::Context c;
    msl::Block body;
    AtomicAccessPlan p =
        planAtomicAccess(accessOf(floatOf(32)), MemOrder::Relaxed);
    body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::F32), "v",
                              atomicLoadValue(c, p, "p", nm)));
    CHECK_HAS(render(body), "as_type");
  }

  CASE("a whole-word store writes the word");
  {
    msl::Context c;
    msl::Block body;
    AtomicAccessPlan p = planAtomicAccess(
        accessOf(intOf(32), AtomicAccess::Store), MemOrder::Relaxed);
    emitAtomicStoreValue(c, body, p, "p", "v", nm);
    const std::string out = render(body);
    CHECK_HAS(out, "atomic_store_explicit");
    CHECK_LACKS(out, "while (");
  }

  CASE("a sub-word store keeps its neighbours, so it retries on contention");
  {
    msl::Context c;
    msl::Block body;
    AtomicAccessPlan p = planAtomicAccess(
        accessOf(intOf(8), AtomicAccess::Store), MemOrder::Relaxed);
    emitAtomicStoreValue(c, body, p, "p", "v", nm);
    const std::string out = render(body);
    CHECK_HAS(out, "atomic_compare_exchange_weak_explicit");
    CHECK_HAS(out, "while (");
    CHECK_HAS(out, "~(255 << sh)");
  }

  CASE("a 64-bit store assigns through the volatile pointer");
  {
    msl::Context c;
    msl::Block body;
    AtomicAccessPlan p = planAtomicAccess(
        accessOf(intOf(64), AtomicAccess::Store), MemOrder::Relaxed);
    emitAtomicStoreValue(c, body, p, "p", "v", nm);
    const std::string out = render(body);
    CHECK_HAS(out, "*p = v");
    CHECK_LACKS(out, "atomic_store_explicit");
  }

  return ::agpu_test::report("AtomicAccess");
}
