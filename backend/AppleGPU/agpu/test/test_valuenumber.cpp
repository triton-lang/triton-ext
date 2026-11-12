// One declaration per value: repeats dropped, readers renamed.
#include "agpu/msl/Context.h"
#include "agpu/msl/ValueNumber.h"
#include "harness.h"
#include "render.h"

using namespace agpu;
using agpu_test::render;

namespace {

bool has(const std::string &text, const char *s) {
  return text.find(s) != std::string::npos;
}

msl::Expr *plus(msl::Context &c, msl::Expr *a, msl::Expr *b) {
  return c.binary(msl::BinOp::Add, a, b);
}

} // namespace

int main() {
  const msl::Type i32 = msl::Context::i32();
  const msl::Type f32 = msl::Context::f32();

  CASE("a repeated value is dropped and its reader renamed");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(i32, "a", plus(c, c.var("x"), c.var("y"))),
        c.declStmt(i32, "b", plus(c, c.var("x"), c.var("y"))),
        c.assign(c.var("o"), c.var("b")),
    };
    msl::reuseEqualValues(body);
    const std::string t = render(body);
    CHECK(!has(t, "int b"));
    CHECK(has(t, "o = a"));
  }

  CASE("twin chains through accumulators meet");
  {
    msl::Context c;
    msl::Block body;
    for (const char *r : {"0", "1"}) {
      const std::string acc = std::string("r") + r, t = std::string("t") + r,
                        u = std::string("u") + r;
      body.push_back(c.declStmt(i32, acc, c.var("p")));
      body.push_back(c.declStmt(i32, t, plus(c, c.var(acc), c.var("q"))));
      body.push_back(c.assign(c.var(acc), c.var(t)));
      body.push_back(
          c.declStmt(i32, u, c.binary(msl::BinOp::Mul, c.var(acc), c.lit(2))));
      body.push_back(c.assign(c.var(std::string("o") + r), c.var(u)));
    }
    msl::reuseEqualValues(body);
    const std::string t = render(body);
    CHECK(has(t, "int u0 = t0 * 2"));
    CHECK(!has(t, "int u1"));
    CHECK(has(t, "o1 = u0"));
  }

  CASE("a value one arm assigns does not stand in the other arm");
  {
    msl::Context c;
    msl::Block thenArm{c.assign(c.var("x"), c.var("a"))};
    msl::Block elseArm{
        c.declStmt(i32, "u", plus(c, c.var("x"), c.lit(1))),
        c.assign(c.var("o"), c.var("u")),
    };
    msl::Block body{
        c.declStmt(i32, "a", c.var("y")),
        c.declStmt(i32, "x", c.var("z")),
        c.declStmt(i32, "h", plus(c, c.var("a"), c.lit(1))),
        c.ifElse(c.var("p"), std::move(thenArm), std::move(elseArm)),
        c.assign(c.var("w"), c.var("h")),
    };
    msl::reuseEqualValues(body);
    const std::string t = render(body);
    CHECK(has(t, "int u = x + 1"));
    CHECK(has(t, "o = u"));
  }

  CASE("a loop's own writes have no value inside it");
  {
    msl::Context c;
    msl::Block loop{
        c.declStmt(i32, "u", plus(c, c.var("x"), c.lit(1))),
        c.assign(c.var("x"), c.var("u")),
    };
    msl::Block body{
        c.declStmt(i32, "a", c.var("y")),
        c.declStmt(i32, "x", c.var("a")),
        c.declStmt(i32, "h", plus(c, c.var("a"), c.lit(1))),
        c.whileStmt(c.var("p"), std::move(loop)),
        c.assign(c.var("w"), c.var("h")),
    };
    msl::reuseEqualValues(body);
    CHECK(has(render(body), "int u = x + 1"));
  }

  CASE("a read of memory is not reused across a store");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(f32, "l1", c.subscript(c.var("buf"), c.var("i"))),
        c.assign(c.subscript(c.var("buf"), c.var("i")), c.litF(3)),
        c.declStmt(f32, "l2", c.subscript(c.var("buf"), c.var("i"))),
        c.assign(c.var("o"), plus(c, c.var("l1"), c.var("l2"))),
    };
    msl::reuseEqualValues(body);
    CHECK(has(render(body), "float l2"));
  }

  CASE("a threadgroup scalar is read anew");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(i32.inAddrSpace(msl::AddrSpace::Threadgroup), "s"),
        c.declStmt(i32, "b1", c.var("s")),
        c.hardBarrier(),
        c.declStmt(i32, "b2", c.var("s")),
        c.assign(c.var("o"), plus(c, c.var("b1"), c.var("b2"))),
    };
    msl::reuseEqualValues(body);
    CHECK(has(render(body), "int b2 = s"));
  }

  CASE("a lane call is reused only in its own block");
  {
    msl::Context c;
    const auto shuffle = [&] {
      return c.call("simd_shuffle_xor", {c.var("v"), c.lit(1)});
    };
    msl::Block arm{
        c.declStmt(i32, "b", shuffle()),
        c.assign(c.var("o"), c.var("b")),
    };
    msl::Block body{
        c.declStmt(i32, "v", c.var("x")),     c.declStmt(i32, "a", shuffle()),
        c.declStmt(i32, "d", shuffle()),      c.assign(c.var("w"), c.var("d")),
        c.ifStmt(c.var("p"), std::move(arm)),
    };
    msl::reuseEqualValues(body);
    const std::string t = render(body);
    CHECK(!has(t, "int d"));
    CHECK(has(t, "w = a"));
    CHECK(has(t, "int b = simd_shuffle_xor"));
  }

  CASE("an assignment's target keeps its own name");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(i32, "h", c.var("a")),
        c.declStmt(i32, "v", c.var("h")),
        c.assign(c.var("v"), plus(c, c.var("v"), c.lit(1))),
        c.assign(c.var("o"), c.var("v")),
    };
    msl::reuseEqualValues(body);
    const std::string t = render(body);
    CHECK(has(t, "v = h + 1"));
    CHECK(!has(t, "h = h + 1"));
  }

  CASE("bits reinterpreted back to their own type are the source");
  {
    msl::Context c;
    msl::Block body{
        c.declStmt(i32, "i", c.var("x")),
        c.declStmt(f32, "f", c.bitcast(f32, c.var("i"))),
        c.declStmt(i32, "j", c.bitcast(i32, c.var("f"))),
        c.assign(c.var("o"), c.var("j")),
    };
    msl::reuseEqualValues(body);
    const std::string t = render(body);
    CHECK(!has(t, "int j"));
    CHECK(has(t, "o = i"));
  }

  return ::agpu_test::report("ValueNumber");
}
