// The spin-wait: waiting for another threadgroup to publish a value.
#include "agpu/emit/EmitPoll.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

PollFacts pollOf(unsigned bits, bool timeout = false, bool acquire = false) {
  PollFacts f;
  f.bits = bits;
  f.hasTimeout = timeout;
  f.acquire = acquire;
  return f;
}

} // namespace

int main() {
  PollNames nm;

  CASE("a 16-bit flag is read out of its containing word");
  {
    PollPlan p = planPoll(pollOf(16));
    CHECK(p.usable);
    CHECK(p.load == PollLoad::PackedHalf);
    CHECK(p.word == msl::Scalar::U32);
  }

  CASE("a 32-bit flag is read at its own width");
  {
    CHECK(planPoll(pollOf(32)).word == msl::Scalar::U32);
    CHECK(planPoll(pollOf(32)).load == PollLoad::AtomicWord);
  }

  CASE("a 64-bit flag reads through a volatile deref");
  {
    PollPlan p = planPoll(pollOf(64));
    CHECK(p.usable);
    CHECK(p.load == PollLoad::VolatileWide);
    CHECK(p.word == msl::Scalar::U64);

    msl::Context c;
    msl::Block body;
    CHECK(emitPoll(c, body, p, nm).ok());
    const std::string out = render(body);
    CHECK_LACKS(out, "atomic_load_explicit");
    CHECK_HAS(out, "*flagp");
    CHECK(out.find("while (") < out.find("*flagp"));
  }

  CASE("the 64-bit flag pointer is volatile, in device space");
  {
    std::ostringstream os;
    msl::Printer pr(os);
    pr.printType(pollPtrType(planPoll(pollOf(64))));
    CHECK_EQ(os.str(), "volatile device ulong *");
  }

  CASE("a width with no atomic load declines");
  {
    for (unsigned bits : {8u, 24u, 128u}) {
      PollPlan p = planPoll(pollOf(bits));
      CHECK(!p.usable);
      Decision d = pollDecision(p);
      CHECK(d.isDecline());
      CHECK(!d.isBug());
    }
  }

  CASE("one thread polls");
  {
    msl::Context c;
    msl::Block body;
    CHECK(emitPoll(c, body, planPoll(pollOf(32)), nm).ok());
    const std::string out = render(body);
    CHECK_HAS(out, "if (tid.x == 0)");
    CHECK(out.find("if (tid.x == 0)") < out.find("while ("));
  }

  CASE("the load is re-issued on every iteration");
  {
    msl::Context c;
    msl::Block body;
    CHECK(emitPoll(c, body, planPoll(pollOf(32)), nm).ok());
    const std::string out = render(body);
    const std::size_t loop = out.find("while (");
    CHECK(loop != std::string::npos);
    const std::size_t load = out.find("atomic_load_explicit");
    CHECK(load > loop);
    CHECK(out.find("while (") != std::string::npos);
  }

  CASE("the load is relaxed, because the barrier carries the ordering");
  {
    msl::Context c;
    msl::Block body;
    emitPoll(c, body, planPoll(pollOf(32)), nm);
    const std::string out = render(body);
    CHECK_HAS(out, "memory_order_relaxed");
    CHECK_LACKS(out, "memory_order_seq_cst");
  }

  CASE("the barrier after the poll is hard");
  {
    msl::Context c;
    msl::Block body;
    body.push_back(c.barrier());
    emitPoll(c, body, planPoll(pollOf(32)), nm);
    CHECK_EQ(countOf(render(body), "threadgroup_barrier"), 2);
  }

  CASE("an acquire poll barriers at device scope");
  {
    msl::Context c;
    msl::Block body;
    emitPoll(c, body, planPoll(pollOf(32, /*timeout=*/false, /*acquire=*/true)),
             nm);
    CHECK_HAS(render(body), "mem_device");
  }

  CASE("a timeout poll tests once and emits no loop");
  {
    msl::Context c;
    msl::Block body;
    PollPlan p = planPoll(pollOf(32, /*timeout=*/true));
    CHECK(!p.spins);
    CHECK(emitPoll(c, body, p, nm).ok());
    const std::string out = render(body);
    CHECK(out.find("while (") == std::string::npos);
    CHECK_HAS(out, "ready = seen");
  }

  CASE("a timeout poll publishes its answer, since only one thread tested");
  {
    msl::Context c;
    msl::Block body;
    emitPoll(c, body, planPoll(pollOf(32, /*timeout=*/true)), nm);
    const std::string out = render(body);
    CHECK_HAS(out, "threadgroup bool seen;");
    CHECK(out.find("threadgroup bool seen;") < out.find("if (tid.x == 0)"));
    CHECK_HAS(out, "seen = false;");
    CHECK(out.find("seen = false;") < out.find("threadgroup_barrier"));
    CHECK(out.find("threadgroup_barrier") < out.find("if (tid.x == 0)"));
  }

  CASE("a spinning poll returns true, because it only returns when it arrived");
  {
    msl::Context c;
    msl::Block body;
    emitPoll(c, body, planPoll(pollOf(32)), nm);
    CHECK_HAS(render(body), "bool ready = true;");
  }

  CASE("a 16-bit flag selects its half at runtime");
  {
    msl::Context c;
    msl::Block body;
    emitPoll(c, body, planPoll(pollOf(16)), nm, "hi");
    const std::string out = render(body);
    CHECK_HAS(out, "hi ?");
    CHECK_HAS(out, ">> 16");
    CHECK_HAS(out, "& 65535");
  }

  CASE("a declined poll emits nothing");
  {
    msl::Context c;
    msl::Block body;
    Decision d = emitPoll(c, body, planPoll(pollOf(8)), nm);
    CHECK(d.isDecline());
    CHECK(body.empty());
  }

  return ::agpu_test::report("Poll");
}
