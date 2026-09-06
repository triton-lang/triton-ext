// Compare-and-exchange: which word and who performs it.
#include "agpu/emit/EmitCas.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

CasFacts casOf(ElemClass e, unsigned bits, bool uniform = false,
               MemOrder o = MemOrder::Relaxed) {
  CasFacts f;
  f.elem = e;
  f.bits = bits;
  f.uniformPtr = uniform;
  f.order = o;
  return f;
}

} // namespace

int main() {
  CasNames nm;

  // ── which word ─────────────────────────────────────────────────────────

  CASE("a 32-bit integer exchange is native");
  {
    CasPlan p = planCas(casOf(ElemClass::Int, 32));
    CHECK(p.strategy == CasStrategy::Word32);
    // compare_exchange_WEAK may fail spuriously, so one attempt is not enough.
    CHECK(p.retries());
  }

  CASE("a float exchange goes through the bit pattern");
  {
    // Metal has no float compare-exchange.
    CasPlan p = planCas(casOf(ElemClass::Float, 32));
    CHECK(p.strategy == CasStrategy::Word32);
    CHECK(p.viaBits);

    msl::Context c;
    msl::Block body;
    CHECK(emitCas(c, body, p, nm, f32()).ok());
    const std::string out = render(body);
    CHECK(out.find("as_type<uint>(cmp)") != std::string::npos);
    CHECK(out.find("as_type<uint>(val)") != std::string::npos);
    CHECK(out.find("as_type<float>(old_w)") != std::string::npos);
  }

  CASE("a 16-bit exchange is a 32-bit one on the containing word");
  {
    CasPlan p = planCas(casOf(ElemClass::Int, 16));
    CHECK(p.strategy == CasStrategy::Packed16);
    CHECK(p.retries());
  }

  CASE("a width with no compare-exchange declines");
  {
    for (unsigned bits : {8u, 64u}) {
      CasPlan p = planCas(casOf(ElemClass::Int, bits));
      CHECK(!p.usable());
      Decision d = casDecision(p);
      CHECK(d.isDecline());
      CHECK(!d.isBug());
    }
  }

  // ── the retry ──────────────────────────────────────────────────────────

  CASE("a packed exchange retries when the other half moved");
  {
    // A 32-bit exchange on a shared word fails whenever the other half
    // changed.
    msl::Context c;
    msl::Block body;
    emitCas(c, body, planCas(casOf(ElemClass::Int, 16)), nm, i32());
    const std::string out = render(body);
    CHECK(out.find("while (true)") != std::string::npos);
    CHECK_EQ(countOf(out, "break;"), 2);
    CHECK(out.find("old_h != cmp_h") != std::string::npos);

    // The result is declared outside the loop, or it dies at the brace.
    CHECK(out.find("ushort old_h;") != std::string::npos);
    CHECK(out.find("ushort old_h;") < out.find("while (true)"));
  }

  CASE("a packed float exchange goes through the bit pattern");
  {
    CasPlan p = planCas(casOf(ElemClass::Float, 16));
    CHECK(p.strategy == CasStrategy::Packed16);
    CHECK(p.viaBits);

    msl::Context c;
    msl::Block body;
    CHECK(emitCas(c, body, p, nm, f16()).ok());
    const std::string out = render(body);
    CHECK(out.find("as_type<ushort>(cmp)") != std::string::npos);
    CHECK(out.find("as_type<ushort>(val)") != std::string::npos);
    CHECK(out.find("as_type<half>(") != std::string::npos);
  }

  CASE("a packed integer exchange needs no reinterpret");
  {
    CHECK(!planCas(casOf(ElemClass::Int, 16)).viaBits);

    msl::Context c;
    msl::Block body;
    emitCas(c, body, planCas(casOf(ElemClass::Int, 16)), nm,
            ElemType{ElemType::Kind::Int, 16, false});
    CHECK(render(body).find("as_type") == std::string::npos);
  }

  CASE("a native exchange retries only a spurious failure");
  {
    // The loop goes round only while the value found is still the one asked
    // for: that separates a spurious failure from a genuine one.
    msl::Context c;
    msl::Block body;
    emitCas(c, body, planCas(casOf(ElemClass::Int, 32)), nm, i32());
    const std::string out = render(body);
    CHECK(out.find("while (" + nm.result + "_w == " + nm.expected + "_w &&") !=
          std::string::npos);
    CHECK_EQ(countOf(out, "atomic_compare_exchange_weak_explicit"), 1);
  }

  CASE("the expected value is passed by address and is the result");
  {
    // Metal writes what it actually found into the expected variable on
    // failure. Re-loading after a failed exchange could see a third value.
    msl::Context c;
    msl::Block body;
    emitCas(c, body, planCas(casOf(ElemClass::Int, 32)), nm, i32());
    const std::string out = render(body);
    CHECK(out.find("&old_w") != std::string::npos);
    CHECK(out.find("int old = (int)old_w;") != std::string::npos);
  }

  // ── who performs it ────────────────────────────────────────────────────

  CASE("a uniform pointer elects one thread and broadcasts");
  {
    CasPlan p = planCas(casOf(ElemClass::Int, 32, /*uniform=*/true));
    CHECK(p.electOne);

    msl::Context c;
    msl::Block body;
    emitCas(c, body, p, nm, i32());
    const std::string out = render(body);
    CHECK(out.find("if (tid.x == 0)") != std::string::npos);
    CHECK(out.find("threadgroup int casb;") != std::string::npos);
    CHECK(out.find("casb = old;") != std::string::npos);
    CHECK(out.find("int old_b = casb;") != std::string::npos);
    CHECK(out.find("casb = cmp;") != std::string::npos);
    CHECK(out.find("casb = cmp;") < out.find("if (tid.x == 0)"));
    // Without a barrier between the seed and the election, a lagging warp's
    // seed can land after the electing thread's answer.
    CHECK(out.find("casb = cmp;") < out.find("threadgroup_barrier"));
    CHECK(out.find("threadgroup_barrier") < out.find("if (tid.x == 0)"));
    CHECK(out.find("if (tid.x == 0)") <
          out.find("atomic_compare_exchange_weak_explicit"));
  }

  CASE("the broadcast barriers are hard and bracket the read");
  {
    // Four barriers: one after the seed, two bracketing the read and the
    // caller's own. The slot is reused every execution, so in a spin loop the
    // electing thread would overwrite it while others read.
    msl::Context c;
    msl::Block body;
    body.push_back(c.barrier());
    emitCas(c, body, planCas(casOf(ElemClass::Int, 32, true)), nm, i32());
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 4);
    const std::size_t read = out.find(nm.result + "_b = " + nm.shared);
    CHECK(read != std::string::npos);
    CHECK(out.rfind("threadgroup_barrier", read) != std::string::npos);
    CHECK(out.find("threadgroup_barrier", read) != std::string::npos);
  }

  CASE("a per-thread pointer performs its own exchange");
  {
    msl::Context c;
    msl::Block body;
    CasPlan p = planCas(casOf(ElemClass::Int, 32));
    CHECK(!p.electOne);
    emitCas(c, body, p, nm, i32());
    const std::string out = render(body);
    CHECK(out.find("if (tid.x == 0)") == std::string::npos);
    CHECK(out.find("threadgroup") == std::string::npos);
  }

  // ── ordering ───────────────────────────────────────────────────────────

  CASE("release ordering fences before the exchange");
  {
    // Metal device atomics are relaxed-only, so the requested order is
    // carried by fences.
    for (MemOrder o : {MemOrder::Release, MemOrder::AcquireRelease}) {
      CasPlan p = planCas(casOf(ElemClass::Int, 32, false, o));
      CHECK(p.fences.before);

      msl::Context c;
      msl::Block body;
      emitCas(c, body, p, nm, i32());
      const std::string out = render(body);
      CHECK(out.find("atomic_thread_fence") != std::string::npos);
      CHECK(out.find("atomic_thread_fence") <
            out.find("atomic_compare_exchange_weak_explicit"));
    }
  }

  CASE("relaxed and acquire need no leading fence");
  {
    CHECK(!planCas(casOf(ElemClass::Int, 32, false, MemOrder::Relaxed))
               .fences.before);
    CHECK(!planCas(casOf(ElemClass::Int, 32, false, MemOrder::Acquire))
               .fences.before);
  }

  CASE("acquire fences after the exchange");
  {
    CHECK(planCas(casOf(ElemClass::Int, 32, false, MemOrder::Acquire))
              .fences.after);
    CHECK(!planCas(casOf(ElemClass::Int, 32, false, MemOrder::Release))
               .fences.after);
    const CasPlan both =
        planCas(casOf(ElemClass::Int, 32, false, MemOrder::AcquireRelease));
    CHECK(both.fences.before);
    CHECK(both.fences.after);

    msl::Context c;
    msl::Block body;
    emitCas(c, body,
            planCas(casOf(ElemClass::Int, 32, false, MemOrder::Acquire)), nm,
            i32());
    const std::string out = render(body);
    CHECK(out.find("atomic_compare_exchange_weak_explicit") <
          out.find("atomic_thread_fence"));
  }

  CASE("an elected acquire fences after the broadcast");
  {
    // Every lane reads the broadcast answer, so a fence inside the election
    // would order only the electing thread.
    msl::Context c;
    msl::Block body;
    emitCas(c, body,
            planCas(casOf(ElemClass::Int, 32, true, MemOrder::Acquire)), nm,
            i32());
    const std::string out = render(body);
    CHECK(out.find("threadgroup_barrier") < out.find("atomic_thread_fence"));
  }

  CASE("the ordering is a fence and the exchange itself is relaxed");
  {
    msl::Context c;
    msl::Block body;
    emitCas(c, body,
            planCas(casOf(ElemClass::Int, 32, false, MemOrder::AcquireRelease)),
            nm, i32());
    const std::string out = render(body);
    CHECK(out.find("memory_order_seq_cst") <
          out.find("atomic_compare_exchange_weak_explicit"));
    CHECK_EQ(countOf(out, "memory_order_relaxed"), 2);
  }

  CASE("a declined exchange emits nothing");
  {
    msl::Context c;
    msl::Block body;
    Decision d =
        emitCas(c, body, planCas(casOf(ElemClass::Int, 64)), nm, i32());
    CHECK(d.isDecline());
    CHECK(body.empty());
  }

  return ::agpu_test::report("Cas");
}
