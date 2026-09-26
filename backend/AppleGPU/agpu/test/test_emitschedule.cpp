// Statement order: pure declarations placed right before their first reader.
#include "agpu/emit/EmitSchedule.h"
#include "agpu/msl/Context.h"
#include "harness.h"
#include "render.h"

using namespace agpu;
using agpu_test::render;

namespace {

bool before(const std::string &text, const char *a, const char *b) {
  const std::size_t pa = text.find(a), pb = text.find(b);
  return pa != std::string::npos && pb != std::string::npos && pa < pb;
}

} // namespace

int main() {
  const msl::Type f32 = msl::Context::f32();

  CASE("an op-major chain runs one register at a time");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(f32, "a0",
                   c.binary(msl::BinOp::Add, c.var("x0"), c.litF(1))),
        c.declStmt(f32, "a1",
                   c.binary(msl::BinOp::Add, c.var("x1"), c.litF(1))),
        c.declStmt(f32, "b0",
                   c.binary(msl::BinOp::Mul, c.var("a0"), c.litF(2))),
        c.declStmt(f32, "b1",
                   c.binary(msl::BinOp::Mul, c.var("a1"), c.litF(2))),
        c.assign(c.var("o0"), c.var("b0")),
        c.assign(c.var("o1"), c.var("b1")),
    };
    sinkToFirstReader(body);
    const std::string t = render(body);
    CHECK(before(t, "float a0", "float b0"));
    CHECK(before(t, "float b0", "o0 = b0"));
    CHECK(before(t, "o0 = b0", "float a1"));
    CHECK(before(t, "float a1", "float b1"));
    CHECK(before(t, "float b1", "o1 = b1"));
  }

  CASE("a declaration does not pass a write to what it reads");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(f32, "a", c.binary(msl::BinOp::Add, c.var("x"), c.litF(1))),
        c.assign(c.var("x"), c.litF(5)),
        c.assign(c.var("y"), c.var("a")),
    };
    sinkToFirstReader(body);
    CHECK(before(render(body), "float a", "x = 5"));
  }

  CASE("a declaration does not pass a write to its own name");
  {
    msl::Context c;
    msl::Block arm{c.assign(c.var("v"), c.litF(1))};
    msl::Block body{
        c.declStmt(f32, "v", c.litF(0)),
        c.declStmt(f32, "u", c.var("x")),
        c.ifStmt(c.var("p"), std::move(arm)),
        c.assign(c.var("y"), c.binary(msl::BinOp::Add, c.var("v"), c.var("u"))),
    };
    sinkToFirstReader(body);
    const std::string t = render(body);
    CHECK(before(t, "float v", "if (p)"));
    CHECK(before(t, "if (p)", "float u"));
  }

  CASE("a masked load's register sinks to the branch that fills it");
  {
    msl::Context c;
    msl::Block fill{
        c.assign(c.var("l0"), c.subscript(c.var("buf"), c.var("f")))};
    msl::Block body{
        c.declStmt(f32, "l0", c.var("k")),
        c.declStmt(msl::Context::i32(), "e", c.var("i")),
        c.declStmt(msl::Context::i32(), "f", c.var("e")),
        c.ifStmt(c.var("p"), std::move(fill)),
        c.assign(c.var("o"),
                 c.binary(msl::BinOp::Add, c.var("l0"), c.var("f"))),
    };
    sinkToFirstReader(body);
    const std::string t = render(body);
    CHECK(before(t, "int e", "float l0"));
    CHECK(before(t, "float l0", "if (p)"));
  }

  CASE("a call returning a second value through a reference stays put");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(f32, "s",
                   c.call("metal::precise::sincos", {c.var("x"), c.var("cv")})),
        c.declStmt(f32, "t", c.binary(msl::BinOp::Mul, c.var("cv"), c.litF(2))),
        c.assign(c.var("o"), c.var("t")),
        c.assign(c.var("p"), c.var("s")),
    };
    sinkToFirstReader(body);
    CHECK(before(render(body), "float s", "float t"));
  }

  CASE("a read of memory stays ahead of the store after it");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(f32, "l", c.subscript(c.var("buf"), c.var("i"))),
        c.assign(c.subscript(c.var("buf"), c.var("i")), c.litF(3)),
        c.assign(c.var("o"), c.var("l")),
    };
    sinkToFirstReader(body);
    CHECK(before(render(body), "float l", "buf[i] = 3"));
  }

  CASE("a read of a threadgroup scalar stays ahead of the barrier after it");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(f32.inAddrSpace(msl::AddrSpace::Threadgroup), "s"),
        c.declStmt(f32, "b", c.var("s")),
        c.hardBarrier(),
        c.assign(c.var("o"), c.var("b")),
    };
    sinkToFirstReader(body);
    CHECK(before(render(body), "float b", "threadgroup_barrier"));
  }

  return ::agpu_test::report("EmitSchedule");
}
