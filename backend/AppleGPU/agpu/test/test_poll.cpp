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

ThreadElection uniform() {
  ThreadElection e;
  e.firstThreadOnly = true;
  return e;
}

// One flag the whole group waits on: the shape a scalar poll takes.
msl::SmallVec<msl::Str, 8> emitOne(msl::Context &c, msl::Block &body,
                                   const PollPlan &p, const PollNames &nm,
                                   const msl::Str &isHigh = {},
                                   ThreadElection e = uniform()) {
  if (!p.usable)
    return {};
  return emitPollTensor(c, body, p, nm, {nm.ptr}, {nm.expected}, ReplicaMap{},
                        e, {isHigh},
                        [&](int64_t) { return electionExpr(c, e, nm); });
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
    CHECK(!emitOne(c, body, p, nm).empty());
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
    CHECK(!emitOne(c, body, planPoll(pollOf(32)), nm).empty());
    const std::string out = render(body);
    CHECK_HAS(out, "if (tid.x == 0)");
    CHECK(out.find("if (tid.x == 0)") < out.find("while ("));
  }

  CASE("the load is re-issued on every iteration");
  {
    msl::Context c;
    msl::Block body;
    CHECK(!emitOne(c, body, planPoll(pollOf(32)), nm).empty());
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
    emitOne(c, body, planPoll(pollOf(32)), nm);
    const std::string out = render(body);
    CHECK_HAS(out, "memory_order_relaxed");
    CHECK_LACKS(out, "memory_order_seq_cst");
  }

  CASE("the barrier after the poll is hard");
  {
    msl::Context c;
    msl::Block body;
    body.push_back(c.barrier());
    emitOne(c, body, planPoll(pollOf(32)), nm);
    CHECK_EQ(countOf(render(body), "threadgroup_barrier"), 2);
  }

  CASE("an acquire poll barriers at device scope");
  {
    msl::Context c;
    msl::Block body;
    emitOne(c, body, planPoll(pollOf(32, /*timeout=*/false, /*acquire=*/true)),
            nm);
    CHECK_HAS(render(body), "mem_device");
  }

  CASE("a timeout poll is planned whatever the budget, since one load is the "
       "floor the op asks for");
  {
    PollPlan p = planPoll(pollOf(32, /*timeout=*/true));
    CHECK(p.usable);
    CHECK(!p.spins);
  }

  CASE("a timeout poll tests once and emits no loop");
  {
    msl::Context c;
    msl::Block body;
    PollPlan p = planPoll(pollOf(32, /*timeout=*/true));
    CHECK(!p.spins);
    CHECK(!emitOne(c, body, p, nm).empty());
    const std::string out = render(body);
    CHECK(out.find("while (") == std::string::npos);
    CHECK_HAS(out, "ready0 = seen0");
  }

  CASE("a timeout poll publishes its answer, since only one thread tested");
  {
    msl::Context c;
    msl::Block body;
    emitOne(c, body, planPoll(pollOf(32, /*timeout=*/true)), nm);
    const std::string out = render(body);
    CHECK_HAS(out, "threadgroup bool seen0;");
    CHECK(out.find("threadgroup bool seen0;") < out.find("if (tid.x == 0)"));
    CHECK_HAS(out, "seen0 = false;");
    CHECK(out.find("seen0 = false;") < out.find("threadgroup_barrier"));
    CHECK(out.find("threadgroup_barrier") < out.find("if (tid.x == 0)"));
  }

  CASE("a spinning poll returns true, because it only returns when it arrived");
  {
    msl::Context c;
    msl::Block body;
    emitOne(c, body, planPoll(pollOf(32)), nm);
    CHECK_HAS(render(body), "bool ready0 = true;");
  }

  CASE("a 16-bit flag selects its half at runtime");
  {
    msl::Context c;
    msl::Block body;
    emitOne(c, body, planPoll(pollOf(16)), nm, "hi");
    const std::string out = render(body);
    CHECK_HAS(out, "hi ?");
    CHECK_HAS(out, ">> 16");
    CHECK_HAS(out, "& 65535");
  }

  CASE("a declined poll emits nothing");
  {
    msl::Context c;
    msl::Block body;
    PollPlan p = planPoll(pollOf(8));
    CHECK(emitOne(c, body, p, nm).empty());
    CHECK(pollDecision(p).isDecline());
    CHECK(body.empty());
  }

  CASE("a poll no thread shares reads on every thread, with no shared slot");
  {
    msl::Context c;
    msl::Block body;
    const msl::SmallVec<msl::Str, 8> outs =
        emitOne(c, body, planPoll(pollOf(32, /*timeout=*/true)), nm, {},
                ThreadElection{});
    CHECK_EQ(outs.size(), 1u);
    const std::string out = render(body);
    CHECK_LACKS(out, "if (tid.x == 0)");
    CHECK_LACKS(out, "threadgroup bool");
    CHECK_HAS(out, "bool seen0;");
  }

  CASE("every element of a tensor poll gets its own slot and answer");
  {
    msl::Context c;
    msl::Block body;
    PollNames tn = nm;
    const msl::SmallVec<msl::Str, 8> outs = emitPollTensor(
        c, body, planPoll(pollOf(32, /*timeout=*/true)), tn, {"p0", "p1"},
        {"w0", "w1"}, ReplicaMap{}, ThreadElection{});
    CHECK_EQ(outs.size(), 2u);
    CHECK(outs[0] != outs[1]);
    const std::string out = render(body);
    CHECK_HAS(out, "p0");
    CHECK_HAS(out, "p1");
    CHECK_EQ(countOf(out, "atomic_load_explicit"), 2);
  }

  CASE("a tensor poll barriers once, not once per element");
  {
    msl::Context c;
    msl::Block body;
    PollNames tn = nm;
    ThreadElection e = uniform();
    emitPollTensor(c, body, planPoll(pollOf(32, /*timeout=*/true)), tn,
                   {"p0", "p1", "p2"}, {"w0", "w1", "w2"}, ReplicaMap{}, e, {},
                   [&](int64_t) { return electionExpr(c, e, tn); });
    CHECK_EQ(countOf(render(body), "threadgroup_barrier"), 2);
  }

  CASE("a replicated register takes the answer of the one that owns it");
  {
    msl::Context c;
    msl::Block body;
    PollNames tn = nm;
    ReplicaMap replicas;
    replicas.regFree = 1;
    const msl::SmallVec<msl::Str, 8> outs =
        emitPollTensor(c, body, planPoll(pollOf(32, /*timeout=*/true)), tn,
                       {"p0", "p1"}, {"w0", "w1"}, replicas, ThreadElection{});
    CHECK_EQ(outs.size(), 2u);
    CHECK_EQ(outs[0], outs[1]);
    CHECK_EQ(countOf(render(body), "atomic_load_explicit"), 1);
  }

  return ::agpu_test::report("Poll");
}
