// Atomic emission: plan -> election -> fences -> operation.
#include "agpu/emit/EmitAtomic.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

AtomicFacts intAdd() {
  AtomicFacts f;
  f.op = RmwOp::Add;
  f.elem = ElemClass::Int;
  f.bits = 32;
  return f;
}

} // namespace

int main() {
  AtomicNames nm;

  CASE("a native atomic names its builtin and a relaxed order");
  {
    msl::Context c;
    msl::Block body;
    emitAtomic(c, body, planAtomic(intAdd(), MemOrder::Relaxed), "p", "v", nm);
    const std::string out = render(body);
    CHECK(out.find("atomic_fetch_add_explicit(p, v, memory_order_relaxed)") !=
          std::string::npos);
  }

  CASE("an unsupported atomic emits nothing at all");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.bits = 64;
    emitAtomic(c, body, planAtomic(f, MemOrder::Relaxed), "p", "v", nm);
    CHECK_EQ(body.size(), 0u);
  }

  CASE("a relaxed atomic emits no fence");
  {
    msl::Context c;
    msl::Block body;
    emitAtomic(c, body, planAtomic(intAdd(), MemOrder::Relaxed), "p", "v", nm);
    CHECK_EQ(countOf(render(body), "fence"), 0);
  }

  CASE("a release barrier precedes the operation");
  {
    msl::Context c;
    msl::Block body;
    emitAtomic(c, body, planAtomic(intAdd(), MemOrder::Release), "p", "v", nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 1);
    CHECK(out.find("threadgroup_barrier") < out.find("atomic_fetch_add"));
  }

  CASE("an acquire barrier follows the operation");
  {
    msl::Context c;
    msl::Block body;
    emitAtomic(c, body, planAtomic(intAdd(), MemOrder::Acquire), "p", "v", nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 1);
    CHECK(out.find("atomic_fetch_add") < out.find("threadgroup_barrier"));
  }

  CASE("the emitted code has acquire-release barriers on both sides");
  {
    msl::Context c;
    msl::Block body;
    emitAtomic(c, body, planAtomic(intAdd(), MemOrder::AcquireRelease), "p",
               "v", nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 2);
    const std::size_t op = out.find("atomic_fetch_add");
    CHECK(out.find("threadgroup_barrier") < op);
    CHECK(out.find("threadgroup_barrier", op) != std::string::npos);
  }

  CASE("the barrier carries device scope and no thread fence is emitted");
  {
    msl::Context c;
    msl::Block body;
    emitAtomic(c, body, planAtomic(intAdd(), MemOrder::Acquire), "p", "v", nm);
    const std::string out = render(body);
    CHECK(out.find("mem_device") != std::string::npos);
    CHECK(out.find("atomic_thread_fence") == std::string::npos);
  }

  CASE("both barriers sit outside the election guard");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.laneFree = 0b11;
    emitAtomic(c, body, planAtomic(f, MemOrder::AcquireRelease), "p", "v", nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 2);
    CHECK(out.find("threadgroup_barrier") < out.find("if ("));
    CHECK(out.find("atomic_fetch_add") < out.rfind("threadgroup_barrier"));
  }

  CASE("no free bits means no guard");
  {
    msl::Context c;
    msl::Block body;
    emitAtomic(c, body, planAtomic(intAdd(), MemOrder::Relaxed), "p", "v", nm);
    CHECK_EQ(countOf(render(body), "if ("), 0);
  }

  CASE("a free lane bit elects the lane whose bits are zero");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.laneFree = 0b11;
    emitAtomic(c, body, planAtomic(f, MemOrder::Relaxed), "p", "v", nm);
    const std::string out = render(body);
    CHECK(out.find("if ((lane & 3) == 0)") != std::string::npos);
  }

  CASE("lane and warp elections are conjoined");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.laneFree = 0b1;
    f.warpFree = 0b10;
    emitAtomic(c, body, planAtomic(f, MemOrder::Relaxed), "p", "v", nm);
    const std::string out = render(body);
    CHECK(out.find("lane & 1") != std::string::npos);
    CHECK(out.find("warp & 2") != std::string::npos);
    CHECK(out.find("&&") != std::string::npos);
  }

  CASE("a uniform pointer elects thread 0, by its x component");
  {
    // `tid` is the uint3 the ABI declares, so `tid == 0` is a bool3, which
    // Metal refuses as a condition.
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.uniformPtr = true;
    emitAtomic(c, body, planAtomic(f, MemOrder::Relaxed), "p", "v", nm);
    const std::string out = render(body);
    CHECK(out.find("if (tid.x == 0)") != std::string::npos);
    CHECK(out.find("if (tid == 0)") == std::string::npos);
  }

  CASE("the fence sits inside the election, with the operation");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.laneFree = 0b1;
    emitAtomic(c, body, planAtomic(f, MemOrder::Acquire), "p", "v", nm);
    const std::string out = render(body);
    CHECK(out.find("if (") < out.find("atomic_thread_fence"));
  }

  CASE("every register issues its own atomic when none are replicas");
  {
    msl::Context c;
    msl::Block body;
    msl::SmallVec<msl::Str, 8> ptrs{"p0", "p1", "p2", "p3"};
    msl::SmallVec<msl::Str, 8> vals{"v0", "v1", "v2", "v3"};
    auto res = emitAtomicTensor(
        c, body, planAtomic(intAdd(), MemOrder::Relaxed), ptrs, vals, nm);
    CHECK_EQ(countOf(render(body), "atomic_fetch_add"), 4);
    CHECK_EQ(res.size(), 4u);
  }

  CASE("a masked tensor guards each register with its own mask");
  {
    // Register 0's predicate says nothing about register 3's; results stay
    // declared outside the guards so consumers read names in scope.
    msl::Context c;
    msl::Block body;
    msl::SmallVec<msl::Str, 8> ptrs{"p0", "p1", "p2", "p3"};
    msl::SmallVec<msl::Str, 8> vals{"v0", "v1", "v2", "v3"};
    auto res = emitAtomicTensor(
        c, body, planAtomic(intAdd(), MemOrder::Relaxed), ptrs, vals, nm, {},
        [&c](int64_t r) { return c.var("m" + std::to_string(r)); });
    CHECK_EQ(res.size(), 4u);
    const std::string out = render(body);
    for (int r = 0; r < 4; ++r)
      CHECK(out.find("if (m" + std::to_string(r) + ")") != std::string::npos);
    CHECK(out.find(res[0]) < out.find("if (m0)"));
  }

  CASE("a replica binds to its canonical, so the op issues once");
  {
    // Registers 2 and 3 address the same locations as 0 and 1, so issuing
    // all four would apply the operation twice.
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.regFree = 0b10;
    msl::SmallVec<msl::Str, 8> ptrs{"p0", "p1", "p2", "p3"};
    msl::SmallVec<msl::Str, 8> vals{"v0", "v1", "v2", "v3"};
    auto res = emitAtomicTensor(c, body, planAtomic(f, MemOrder::Relaxed), ptrs,
                                vals, nm);
    CHECK_EQ(countOf(render(body), "atomic_fetch_add"), 2);
    CHECK_EQ(res[2], res[0]);
    CHECK_EQ(res[3], res[1]);
    CHECK(res[0] != res[1]);
  }

  CASE("a packed-16 tensor selects the half per register");
  {
    // Which half a value occupies is a property of its own address: element 1
    // of an fp16 tensor is the high half of element 0's word.
    msl::Context c;
    msl::Block body;
    AtomicFacts f;
    f.op = RmwOp::Max;
    f.elem = ElemClass::Float;
    f.bits = 16;
    const AtomicPlan p = planAtomic(f, MemOrder::Relaxed);
    CHECK(p.strategy == AtomicStrategy::Packed16);

    msl::SmallVec<msl::Str, 8> ptrs{"w0", "w1"};
    msl::SmallVec<msl::Str, 8> vals{"v0", "v1"};
    msl::SmallVec<msl::Str, 8> highs{"hi0", "hi1"};
    auto res = emitAtomicTensor(c, body, p, ptrs, vals, nm, highs);
    CHECK_EQ(res.size(), 2u);

    const std::string out = render(body);
    CHECK(out.find("hi0") != std::string::npos);
    CHECK(out.find("hi1") != std::string::npos);
  }

  CASE("the packed-16 helper is called at a named type");
  {
    // `T` is the helper's return type and appears in no parameter, so
    // nothing deduces it. f16 and bf16 are both 16-bit floats, so `elem` and
    // `bits` alone cannot tell them apart.
    const auto emitAt = [&](ElemType e) {
      msl::Context c;
      msl::Block body;
      AtomicFacts f;
      f.op = RmwOp::Add;
      f.elem = ElemClass::Float;
      f.bits = 16;
      f.packedElem = e;
      const AtomicPlan p = planAtomic(f, MemOrder::Relaxed);
      emitAtomicTensor(c, body, p, {"w0"}, {"v0"}, nm, {"hi0"});
      return render(body);
    };

    CHECK(emitAt(f16()).find("__agpu_atomic_rmw_packed16<half>") !=
          std::string::npos);
    CHECK(emitAt(bf16()).find("__agpu_atomic_rmw_packed16<bfloat>") !=
          std::string::npos);
  }

  CASE("every register still gets a result name to read");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.regFree = 0b11;
    msl::SmallVec<msl::Str, 8> ptrs{"p0", "p1", "p2", "p3"};
    msl::SmallVec<msl::Str, 8> vals{"v0", "v1", "v2", "v3"};
    auto res = emitAtomicTensor(c, body, planAtomic(f, MemOrder::Relaxed), ptrs,
                                vals, nm);
    CHECK_EQ(countOf(render(body), "atomic_fetch_add"), 1);
    for (const msl::Str &n : res)
      CHECK(!n.empty());
  }

  CASE("a float max goes through the CAS helper, unfenced");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f;
    f.op = RmwOp::Max;
    f.elem = ElemClass::Float;
    f.bits = 32;
    emitAtomic(c, body, planAtomic(f, MemOrder::AcquireRelease), "p", "v", nm);
    const std::string out = render(body);
    CHECK(out.find("__agpu_atomic_rmw_f32") != std::string::npos);
    CHECK(out.find("atomic_fetch") == std::string::npos);
    CHECK_EQ(countOf(out, "atomic_thread_fence"), 0);
  }

  CASE("a 16-bit float goes through the packed helper");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f;
    f.op = RmwOp::Add;
    f.elem = ElemClass::Float;
    f.bits = 16;
    emitAtomic(c, body, planAtomic(f, MemOrder::Relaxed), "p", "v", nm);
    CHECK(render(body).find("__agpu_atomic_rmw_packed16") != std::string::npos);
  }

  CASE("no election spells the thread id without a component");
  {
    // `tid` is a uint3, so it may appear only as `tid.<component>`.
    msl::Context c;
    auto electionText = [&](AtomicFacts f) {
      msl::Block body;
      emitAtomic(c, body, planAtomic(f, MemOrder::Relaxed), "p", "v", nm);
      return render(body);
    };

    AtomicFacts uniform = intAdd();
    uniform.uniformPtr = true;
    AtomicFacts lanes = intAdd();
    lanes.laneFree = 0b11;
    AtomicFacts warps = intAdd();
    warps.warpFree = 0b1;

    for (const AtomicFacts &f : {uniform, lanes, warps}) {
      const std::string out = electionText(f);
      for (std::size_t i = out.find("tid"); i != std::string::npos;
           i = out.find("tid", i + 1))
        CHECK(i + 3 < out.size() && out[i + 3] == '.');
    }
  }

  CASE("a group-wide election publishes its result to the threads it excluded");
  {
    // Left at its initialiser, an excluded thread reads zero. A loop over the
    // result then exits early and strands the winner on the next barrier.
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.uniformPtr = true;
    AtomicNames bn = nm;
    bn.scratch = "ascr";
    emitAtomic(c, body, planAtomic(f, MemOrder::Relaxed), "p", "v", bn);
    const std::string out = render(body);
    CHECK(out.find("ascr[0] = old") != std::string::npos);
    CHECK(out.find("old = ascr[0]") != std::string::npos);

    // Every barrier stands outside the election: a threadgroup_barrier under
    // divergent control flow is undefined in Metal. Two bracket the publish,
    // and a third closes the read so the pool cannot reuse the slot under it.
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 3);
    const std::size_t publish = out.find("ascr[0] = old");
    CHECK(publish != std::string::npos);
    CHECK(out.rfind("threadgroup_barrier", publish) <
          out.rfind("if (", publish));
  }

  CASE("an election inside one warp asks for no threadgroup memory");
  {
    msl::Context c;
    msl::Block body;
    AtomicFacts f = intAdd();
    f.laneFree = 0b11;
    AtomicNames bn = nm;
    bn.scratch = "ascr";
    emitAtomic(c, body, planAtomic(f, MemOrder::Relaxed), "p", "v", bn);
    const std::string out = render(body);
    CHECK(out.find("ascr") == std::string::npos);
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 0);
  }

  return ::agpu_test::report("EmitAtomic");
}
