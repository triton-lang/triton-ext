// The whole function: signature, ABI, prologue, size policy.
#include "agpu/emit/EmitKernel.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;

namespace {

std::string render(msl::Stmt *s) {
  std::ostringstream os;
  msl::Printer p(os);
  msl::Block b{s};
  p.printBlock(b);
  return os.str();
}

KernelArg ptr(const char *n) { return KernelArg{n, f32(), true}; }
KernelArg scalar(const char *n, ElemType e = i32()) {
  return KernelArg{n, e, false};
}

BodyFn trivialBody(int n) {
  return [n](msl::Context &c, bool) {
    msl::Block b;
    for (int i = 0; i < n; ++i)
      b.push_back(
          c.declStmt(msl::Context::i32(), "t" + std::to_string(i), c.lit(i)));
    return b;
  };
}

} // namespace

int main() {
  CASE("pointers take buffer bindings in order");
  {
    KernelAbi abi = planKernelAbi({ptr("a"), ptr("b"), ptr("c")}, 4);
    CHECK_EQ(abi.placements[0].index, 0);
    CHECK_EQ(abi.placements[1].index, 1);
    CHECK_EQ(abi.placements[2].index, 2);
    CHECK_EQ(abi.bufferCount, 3);
    CHECK(!abi.hasArgBuffer);
  }

  CASE("scalars pack together into a single argument buffer");
  {
    KernelAbi abi =
        planKernelAbi({ptr("a"), scalar("m"), scalar("n"), scalar("k")}, 4);
    CHECK_EQ(abi.bufferCount, 2); // the pointer, plus the argument buffer
    CHECK(abi.hasArgBuffer);
    for (int i = 1; i <= 3; ++i)
      CHECK(abi.placements[i].slot == ArgSlot::ArgBuffer);
  }

  CASE("scalars are packed at their natural alignment");
  {
    // A misaligned constant read is undefined on Apple silicon.
    KernelAbi abi =
        planKernelAbi({scalar("b", ElemType{ElemType::Kind::Bool, 1, false}),
                       scalar("w", ElemType{ElemType::Kind::Int, 32, false}),
                       scalar("h", ElemType{ElemType::Kind::Int, 16, false})},
                      4);
    CHECK_EQ(abi.placements[0].offset, 0); // bool, 1 byte
    CHECK_EQ(abi.placements[1].offset, 4); // int aligns to 4
    CHECK_EQ(abi.placements[2].offset, 8); // short follows
    CHECK_EQ(abi.argBufferBytes, 10);
  }

  CASE("too many bindings declines with a reason");
  {
    std::vector<KernelArg> args;
    for (int i = 0; i < 40; ++i)
      args.push_back(ptr("p"));
    KernelAbi abi = planKernelAbi(args, 4);
    CHECK(!abi.usable());
    Decision d = abiDecision(abi);
    CHECK(d.isDecline());
    CHECK_EQ(d.why(), std::string("more buffer bindings than Metal allows"));
  }

  CASE("exactly 31 bindings is allowed");
  {
    std::vector<KernelArg> args;
    for (int i = 0; i < 31; ++i)
      args.push_back(ptr("p"));
    CHECK(planKernelAbi(args, 4).usable());
  }

  CASE("the kernel declares its buffers and the three builtins");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "add";
    f.args = {ptr("x"), ptr("y"), scalar("n")};
    f.numWarps = 4;
    KernelResult r = emitKernel(c, f, trivialBody(1));
    CHECK(r.ok());
    const std::string out = render(r.fn);
    CHECK(out.find("kernel void add(") != std::string::npos);
    CHECK(out.find("[[buffer(0)]]") != std::string::npos);
    CHECK(out.find("[[buffer(1)]]") != std::string::npos);
    CHECK(out.find("[[buffer(2)]]") != std::string::npos); // the arg buffer
    CHECK(out.find("[[threadgroup_position_in_grid]]") != std::string::npos);
    CHECK(out.find("[[thread_position_in_threadgroup]]") != std::string::npos);
    CHECK(out.find("[[threadgroups_per_grid]]") != std::string::npos);
  }

  CASE("a coherent argument is declared coherent and only that one");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "acc";
    KernelArg shared = ptr("s");
    shared.coherent = true;
    f.args = {shared, ptr("y")};
    KernelResult r = emitKernel(c, f, trivialBody(1));
    CHECK(r.ok());
    const std::string out = render(r.fn);
    CHECK(out.find("device coherent(device) float * s") != std::string::npos);
    CHECK(out.find("device float * y") != std::string::npos);
    CHECK_EQ(countOf(out, "coherent"), 1);
  }

  CASE("a binding is a plain number the host can compare");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {ptr("x"), ptr("y"), scalar("n")};
    KernelResult r = emitKernel(c, f, trivialBody(1));

    const KernelAbi abi = planKernelAbi(f.args, f.numWarps);
    int checked = 0;
    for (const msl::Function::Param &p : r.fn->params) {
      if (p.attribute.kind() != msl::Attribute::Kind::Buffer)
        continue;
      bool found = p.attribute == msl::Attribute::buffer(abi.bufferCount - 1);
      for (const ArgPlacement &pl : abi.placements)
        found = found || (pl.slot == ArgSlot::Buffer &&
                          p.attribute == msl::Attribute::buffer(pl.index));
      CHECK(found);
      ++checked;
    }
    CHECK_EQ(checked, 3); // two pointers and the argument buffer
  }

  CASE("the launch qualifier carries its raw value");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {ptr("x")};
    f.numWarps = 8;
    f.poolBytes = 40000;
    KernelResult r = emitKernel(c, f, trivialBody(1));
    CHECK(r.fn->qualifier.kind() ==
          msl::Attribute::Kind::MaxTotalThreadsPerThreadgroup);
    CHECK_EQ(r.fn->qualifier.value(), 256);
  }

  CASE("a kernel with no scalars declares no argument buffer");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "copy";
    f.args = {ptr("x"), ptr("y")};
    KernelResult r = emitKernel(c, f, trivialBody(1));
    const std::string out = render(r.fn);
    CHECK(out.find("[[buffer(2)]]") == std::string::npos);
    CHECK(out.find("constant") == std::string::npos);
  }

  CASE("scalars are unpacked from the argument buffer at their offsets");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {scalar("m"), scalar("n")};
    KernelResult r = emitKernel(c, f, trivialBody(1));
    const std::string out = render(r.fn);
    CHECK(out.find("int m = *(constant int *)args;") != std::string::npos ||
          out.find("int m = *(constant int *)(args + 0)") != std::string::npos);
    CHECK(out.find("args + 4") != std::string::npos);
  }

  CASE("lane and warp are derived once, from the flat thread index");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {ptr("x")};
    KernelResult r = emitKernel(c, f, trivialBody(1));
    const std::string out = render(r.fn);
    CHECK(out.find("int lane = tid.x & 31;") != std::string::npos);
    CHECK(out.find("int warp = tid.x / 32;") != std::string::npos);
    CHECK_EQ(countOf(out, "int lane ="), 1);
  }

  CASE("the prologue precedes the body");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {scalar("n")};
    KernelResult r = emitKernel(c, f, trivialBody(2));
    const std::string out = render(r.fn);
    CHECK(out.find("int n =") < out.find("int lane ="));
    CHECK(out.find("int lane =") < out.find("int t0 ="));
  }

  CASE("the threadgroup size is pinned for wide launches and lost residency");
  {
    // Without pinning, a register-hungry kernel compiles to a pipeline capped
    // below its own launch and dispatch rejects it as OutOfResources.
    CHECK(shouldPinThreadgroupSize(16000, 65536, 512));  // wide: pin
    CHECK(shouldPinThreadgroupSize(40000, 65536, 128));  // over half: pin
    CHECK(!shouldPinThreadgroupSize(16000, 65536, 384)); // admitted: do not
  }

  CASE("a large pool pins the launch size");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {ptr("x")};
    f.numWarps = 8;
    f.poolBytes = 40000;
    KernelResult r = emitKernel(c, f, trivialBody(1));
    CHECK(render(r.fn).find("max_total_threads_per_threadgroup(256)") !=
          std::string::npos);
  }

  CASE("a small pool leaves it unpinned, buying the second threadgroup");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {ptr("x")};
    f.poolBytes = 4096;
    KernelResult r = emitKernel(c, f, trivialBody(1));
    CHECK(render(r.fn).find("max_total_threads") == std::string::npos);
  }

  CASE("a small kernel is emitted once and not rolled");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {ptr("x")};
    KernelResult r = emitKernel(c, f, trivialBody(4));
    CHECK(!r.reemitted);
    CHECK(!r.shrink.rollKSteps);
    CHECK(r.size.decls >= 4);
  }

  CASE("an oversized kernel is re-emitted in its rolled form");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "big";
    f.args = {ptr("x")};

    bool sawRoll = false;
    BodyFn body = [&](msl::Context &cc, bool rollK) {
      sawRoll = sawRoll || rollK;
      msl::Block b;
      const int n = rollK ? 8 : 12000;
      for (int i = 0; i < n; ++i)
        b.push_back(cc.declStmt(msl::Type::matrix("simdgroup_float8x8"),
                                "f" + std::to_string(i), nullptr));
      return b;
    };

    KernelResult r = emitKernel(c, f, body);
    CHECK(sawRoll);
    CHECK(r.reemitted);
    CHECK(r.size.decls < 12000);
  }

  CASE("a re-emit that does not help is discarded");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "big";
    f.args = {ptr("x")};
    BodyFn body = [](msl::Context &cc, bool) {
      msl::Block b;
      for (int i = 0; i < 12000; ++i)
        b.push_back(cc.declStmt(msl::Type::matrix("simdgroup_float8x8"),
                                "f" + std::to_string(i), nullptr));
      return b;
    };
    KernelResult r = emitKernel(c, f, body);
    CHECK(!r.reemitted);
    CHECK_EQ(r.size.decls, 12002);
  }

  CASE("guards are fused after the body is final");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {ptr("x")};
    BodyFn body = [](msl::Context &cc, bool) {
      msl::Block b;
      for (int i = 0; i < 6; ++i)
        b.push_back(cc.ifStmt(
            cc.var("p"),
            msl::Block{cc.assign(cc.var("x" + std::to_string(i)), cc.lit(1))}));
      return b;
    };
    KernelResult r = emitKernel(c, f, body);
    CHECK(r.shrink.fuseGuards);
    CHECK_EQ(countOf(render(r.fn), "if (p)"), 1);
  }

  CASE("stores separated by other work are sunk, then fused");
  {
    // Fusion merges only adjacent guards, so the sink moves these six stores
    // together first. A non-literal index yields an invalid CoordSet and the
    // sink stops.
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {ptr("out")};
    BodyFn body = [](msl::Context &cc, bool) {
      msl::Block b;
      for (int i = 0; i < 6; ++i) {
        b.push_back(cc.ifStmt(
            cc.var("p"),
            msl::Block{cc.assign(cc.subscript(cc.var("out"), cc.lit(i * 8)),
                                 cc.lit(1))}));
        b.push_back(cc.declStmt(msl::Context::i32(), "t" + std::to_string(i),
                                cc.lit(i)));
      }
      return b;
    };
    KernelResult r = emitKernel(c, f, body);
    CHECK(r.shrink.fuseGuards);
    CHECK_EQ(countOf(render(r.fn), "if (p)"), 1);
  }

  CASE("a kernel reads as a kernel");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "vecadd";
    f.args = {ptr("out"), ptr("a"), ptr("b"), scalar("n")};
    f.numWarps = 4;
    KernelResult r = emitKernel(c, f, trivialBody(1));
    const std::string out = render(r.fn);
    CHECK(out.find("kernel void vecadd(") != std::string::npos);
    CHECK(out.find("device float * out") != std::string::npos);
    CHECK(out.find("constant uchar * args") != std::string::npos);
    CHECK(out.find("uint3 tgid") != std::string::npos);
    CHECK(r.decision.ok());
  }

  return ::agpu_test::report("EmitKernel");
}
