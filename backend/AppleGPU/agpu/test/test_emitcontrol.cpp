// Structured control flow. MSL has no block arguments, so a region result
// becomes a variable declared before the construct and assigned on every path
// out.
#include "agpu/emit/EmitControl.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

CarriedValue val(std::initializer_list<const char *> regs, ElemType e = i32()) {
  CarriedValue v;
  for (const char *r : regs)
    v.regs.push_back(r);
  v.elem = e;
  return v;
}

Carried one(std::initializer_list<const char *> regs, ElemType e = i32()) {
  Carried c;
  c.push_back(val(regs, e));
  return c;
}

} // namespace

int main() {
  CASE("results are declared ahead of the construct");
  {
    // A variable declared inside a region is out of scope after it.
    msl::Context c;
    msl::Block body;
    Decision d = emitIf(c, body, "p", one({"r0"}), msl::Block{}, one({"t0"}),
                        false, msl::Block{}, {});
    CHECK(d.ok());
    const std::string out = render(body);
    CHECK_HAS(out, "int r0;");
    CHECK(out.find("int r0;") < out.find("if ("));
  }

  CASE("each arm assigns the results it yields");
  {
    msl::Context c;
    msl::Block body;
    emitIf(c, body, "p", one({"r0"}), msl::Block{}, one({"t0"}),
           /*hasElse=*/true, msl::Block{}, one({"e0"}));
    const std::string out = render(body);
    CHECK_HAS(out, "r0 = t0;");
    CHECK_HAS(out, "r0 = e0;");
    CHECK_HAS(out, "else");
  }

  CASE("an if with no else leaves the results at their declared value");
  {
    msl::Context c;
    msl::Block body;
    emitIf(c, body, "p", one({"r0"}), msl::Block{}, one({"t0"}), false,
           msl::Block{}, {});
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "r0 ="), 1);
    CHECK_LACKS(out, "else");
  }

  CASE("a multi-register result assigns every register");
  {
    msl::Context c;
    msl::Block body;
    emitIf(c, body, "p", one({"r0", "r1"}), msl::Block{}, one({"t0", "t1"}),
           false, msl::Block{}, {});
    const std::string out = render(body);
    CHECK_HAS(out, "r0 = t0;");
    CHECK_HAS(out, "r1 = t1;");
  }

  CASE("a yield whose arity disagrees is a failure");
  {
    // The IR is malformed.
    msl::Context c;
    msl::Block body;
    Decision d = emitIf(c, body, "p", one({"r0", "r1"}), msl::Block{},
                        one({"t0"}), false, msl::Block{}, {});
    CHECK(d.isBug());
    CHECK(!d.isDecline());
  }

  CASE("an if with no results emits no declarations");
  {
    msl::Context c;
    msl::Block body;
    Decision d =
        emitIf(c, body, "p", {}, msl::Block{}, {}, false, msl::Block{}, {});
    CHECK(d.ok());
    CHECK_EQ(countOf(render(body), "int "), 0);
  }

  CASE("carried values are seeded before the loop and updated by the yield");
  {
    // One set of variables serves init, body and result; separate result
    // variables would need a copy per iteration.
    msl::Context c;
    msl::Block body;
    LoopBounds b = constBounds(c, "i", 0, 8);
    Decision d = emitFor(c, body, b, one({"acc"}), one({"init"}), msl::Block{},
                         one({"next"}));
    CHECK(d.ok());
    const std::string out = render(body);
    CHECK_HAS(out, "int acc = init;");
    CHECK(out.find("int acc = init;") < out.find("for ("));
    CHECK_HAS(out, "acc = next;");
  }

  CASE("the loop spells its bounds and step");
  {
    msl::Context c;
    msl::Block body;
    emitFor(c, body, constBounds(c, "k", 0, 64, 8), {}, {}, msl::Block{}, {});
    const std::string out = render(body);
    CHECK_HAS(out, "int k = 0");
    CHECK_HAS(out, "k < 64");
    CHECK_HAS(out, "k += 8");
  }

  CASE("the bounds are expressions, so a runtime trip count is expressible");
  {
    msl::Context c;
    msl::Block body;
    LoopBounds b;
    b.iv = "k";
    b.lo = c.lit(0);
    b.hi = c.var("K");
    b.step = c.var("BLOCK_K");
    emitFor(c, body, b, {}, {}, msl::Block{}, {});
    const std::string out = render(body);
    CHECK_HAS(out, "k < K");
    CHECK_HAS(out, "k += BLOCK_K");
  }

  CASE("a wide induction variable selects the trip-count form");
  {
    // AGX computes a 64-bit induction variable in the Gauss-sum closed form at
    // i65 intermediate width and gets it wrong. Count iterations in a narrow
    // counter and derive the value.
    msl::Context c;
    msl::Block body;
    LoopBounds b = constBounds(c, "i", 0, 8, 1, /*wideIv=*/true);
    emitFor(c, body, b, {}, {}, msl::Block{}, {});
    const std::string out = render(body);

    CHECK_HAS(out, "int i_tc = 0");
    CHECK_HAS(out, "i_tc += 1");
    CHECK_HAS(out, "long i = i_tc;");
    // The exit test is inside the body: the derived value does not exist until
    // the body computes it.
    CHECK_HAS(out, "for (int i_tc = 0; ; i_tc += 1)");
    CHECK_HAS(out, "if (!(i < 8)) break;");
    CHECK(out.find("long i =") < out.find("break;"));
  }

  CASE("a narrow induction variable keeps the plain loop");
  {
    // The trip-count form costs a multiply per iteration.
    msl::Context c;
    msl::Block body;
    emitFor(c, body, constBounds(c, "i", 0, 8), {}, {}, msl::Block{}, {});
    const std::string out = render(body);
    CHECK_HAS(out, "for (int i = 0; i < 8; i += 1)");
    CHECK_LACKS(out, "_tc");
  }

  CASE("a mismatched carried arity fails");
  {
    msl::Context c;
    msl::Block body;
    Decision d = emitFor(c, body, LoopBounds{}, one({"acc"}), {}, msl::Block{},
                         one({"next"}));
    CHECK(d.isBug());
  }

  CASE("the condition is tested inside the loop, with an early break");
  {
    // scf.while splits condition and body across two regions; MSL has no such
    // split, so the before region becomes the loop head.
    msl::Context c;
    msl::Block body;
    Decision d =
        emitWhile(c, body, one({"acc"}), one({"init"}), msl::Block{}, "cond",
                  one({"res"}), one({"fwd"}), msl::Block{}, one({"next"}));
    CHECK(d.ok());
    const std::string out = render(body);
    CHECK_HAS(out, "while (true)");
    CHECK_HAS(out, "if (!cond)");
    CHECK_HAS(out, "break;");
  }

  CASE("the exit forwards the results before breaking");
  {
    // After the break nothing runs.
    msl::Context c;
    msl::Block body;
    emitWhile(c, body, one({"acc"}), one({"init"}), msl::Block{}, "cond",
              one({"res"}), one({"fwd"}), msl::Block{}, one({"next"}));
    const std::string out = render(body);
    const std::size_t fwd = out.find("res = fwd;");
    const std::size_t brk = out.find("break;");
    CHECK(fwd != std::string::npos);
    CHECK(brk != std::string::npos);
    CHECK(fwd < brk);
  }

  CASE("the carried values update at the end of the body");
  {
    msl::Context c;
    msl::Block body;
    emitWhile(c, body, one({"acc"}), one({"init"}), msl::Block{}, "cond",
              one({"res"}), one({"fwd"}), msl::Block{}, one({"next"}));
    const std::string out = render(body);
    CHECK_HAS(out, "int acc = init;");
    CHECK_HAS(out, "acc = next;");
    CHECK(out.find("break;") < out.find("acc = next;"));
  }

  CASE("results and carried values are separate variables");
  {
    // A while's results come from the condition region's arguments.
    msl::Context c;
    msl::Block body;
    emitWhile(c, body, one({"acc"}), one({"init"}), msl::Block{}, "cond",
              one({"res"}), one({"fwd"}), msl::Block{}, one({"next"}));
    const std::string out = render(body);
    CHECK_HAS(out, "int res;");
    CHECK_HAS(out, "int acc = init;");
  }

  CASE("a carried value keeps its element type");
  {
    msl::Context c;
    msl::Block body;
    emitFor(c, body, LoopBounds{}, one({"acc"}, f32()), one({"init"}, f32()),
            msl::Block{}, one({"next"}, f32()));
    CHECK_HAS(render(body), "float acc = init;");
  }

  return ::agpu_test::report("EmitControl");
}
