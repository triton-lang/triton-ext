// Elementwise emission: the per-element operations.
#include "agpu/emit/EmitElementwise.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

namespace {

ElemType u32() { return ElemType{ElemType::Kind::Int, 32, true}; }

} // namespace

int main() {
  CASE("an unsigned operation promotes its operands and declares to match");
  {
    // A declaration disagreeing with the promotion compiles and gives the
    // wrong quotient over half the input range.
    msl::Context c;
    const std::string s =
        render(emitEw(c, EwOp::DivU, i32(), "d", c.var("a"), c.var("b")));
    CHECK_EQ(s, std::string("uint d = (uint)a / (uint)b;\n"));
  }

  CASE("a signed operation promotes nothing");
  {
    msl::Context c;
    CHECK_EQ(render(emitEw(c, EwOp::DivS, i32(), "d", c.var("a"), c.var("b"))),
             std::string("int d = a / b;\n"));
    CHECK_EQ(render(emitEw(c, EwOp::Add, f32(), "d", c.var("a"), c.var("b"))),
             std::string("float d = a + b;\n"));
  }

  CASE("an already-unsigned operand is not cast again");
  {
    msl::Context c;
    CHECK_EQ(render(emitEw(c, EwOp::DivU, u32(), "d", c.var("a"), c.var("b"))),
             std::string("uint d = a / b;\n"));
  }

  CASE("an emitted comparison yields bool whatever it consumed");
  {
    msl::Context c;
    CHECK_EQ(
        render(emitEw(c, EwOp::CmpLtU, i32(), "d", c.var("a"), c.var("b"))),
        std::string("bool d = (uint)a < (uint)b;\n"));
  }

  CASE("greater-than exists in both signednesses");
  {
    msl::Context c;
    CHECK_EQ(
        render(emitEw(c, EwOp::CmpGtS, i32(), "d", c.var("a"), c.var("b"))),
        std::string("bool d = a > b;\n"));
    CHECK_EQ(
        render(emitEw(c, EwOp::CmpGeU, i32(), "d", c.var("a"), c.var("b"))),
        std::string("bool d = (uint)a >= (uint)b;\n"));
  }

  CASE("float division is its own distinct operation");
  {
    msl::Context c;
    CHECK_EQ(render(emitEw(c, EwOp::DivF, f32(), "d", c.var("a"), c.var("b"))),
             std::string("float d = a / b;\n"));
    CHECK(checkEw(EwOp::DivS, f32()).isDecline());
    CHECK(checkEw(EwOp::DivF, i32()).isDecline());
  }

  CASE("float remainder declines to an operator and names its family");
  {
    Decision d = checkEw(EwOp::RemF, f32());
    CHECK(d.isDecline());
    CHECK(checkMath2(MathFn2::Fmod, f32()).ok());
  }

  CASE("an ordered predicate on an ordered operator needs no guard");
  {
    msl::Context c;
    for (auto [pred, text] : {std::pair{FCmp::OLt, "a < b"},
                              {FCmp::OLe, "a <= b"},
                              {FCmp::OGt, "a > b"},
                              {FCmp::OGe, "a >= b"},
                              {FCmp::OEq, "a == b"}}) {
      CHECK(planFCmp(pred).guard == NanGuard::None);
      CHECK_EQ(render(fcmpExpr(c, pred, "a", "b")), std::string(text));
    }
  }

  CASE("ONE is the one ordered predicate that needs a mask");
  {
    // `!=` is MSL's only unordered operator: true when either operand is NaN,
    // so an ordered not-equal has to take those cases back out.
    msl::Context c;
    CHECK(planFCmp(FCmp::ONe).guard == NanGuard::MaskOut);
    const std::string s = render(fcmpExpr(c, FCmp::ONe, "a", "b"));
    CHECK_EQ(s, std::string("!(metal::isnan(a) || metal::isnan(b)) && a != b"));
  }

  CASE("UNE needs no guard, because its operator is already unordered");
  {
    msl::Context c;
    CHECK(planFCmp(FCmp::UNe).guard == NanGuard::None);
    CHECK_EQ(render(fcmpExpr(c, FCmp::UNe, "a", "b")), std::string("a != b"));
  }

  CASE("an unordered predicate ORs the NaN cases back in");
  {
    msl::Context c;
    for (FCmp pred : {FCmp::ULt, FCmp::ULe, FCmp::UGt, FCmp::UGe, FCmp::UEq}) {
      CHECK(planFCmp(pred).guard == NanGuard::OrIn);
      const std::string s = render(fcmpExpr(c, pred, "a", "b"));
      CHECK(s.find("metal::isnan(a) || metal::isnan(b) ||") !=
            std::string::npos);
    }
    CHECK_EQ(render(fcmpExpr(c, FCmp::ULt, "a", "b")),
             std::string("metal::isnan(a) || metal::isnan(b) || a < b"));
  }

  CASE("ord and uno test for NaN");
  {
    msl::Context c;
    CHECK(planFCmp(FCmp::Uno).kind == FCmpPlan::Kind::NanTest);
    CHECK_EQ(render(fcmpExpr(c, FCmp::Uno, "a", "b")),
             std::string("metal::isnan(a) || metal::isnan(b)"));
    CHECK_EQ(render(fcmpExpr(c, FCmp::Ord, "a", "b")),
             std::string("!(metal::isnan(a) || metal::isnan(b))"));
  }

  CASE("the constant predicates emit a bare constant");
  {
    msl::Context c;
    CHECK(planFCmp(FCmp::True).kind == FCmpPlan::Kind::Constant);
    CHECK_EQ(render(fcmpExpr(c, FCmp::True, "a", "b")), std::string("true"));
    CHECK_EQ(render(fcmpExpr(c, FCmp::False, "a", "b")), std::string("false"));
  }

  CASE("every one of the sixteen predicates is answered");
  {
    msl::Context c;
    const FCmp all[] = {FCmp::False, FCmp::OEq, FCmp::OGt, FCmp::OGe,
                        FCmp::OLt,   FCmp::OLe, FCmp::ONe, FCmp::Ord,
                        FCmp::UEq,   FCmp::UGt, FCmp::UGe, FCmp::ULt,
                        FCmp::ULe,   FCmp::UNe, FCmp::Uno, FCmp::True};
    CHECK_EQ((int)(sizeof(all) / sizeof(all[0])), 16);
    for (FCmp pred : all)
      CHECK(!render(fcmpExpr(c, pred, "a", "b")).empty());
  }

  CASE("an ordered and an unordered form of one relation differ");
  {
    msl::Context c;
    CHECK(render(fcmpExpr(c, FCmp::OLt, "a", "b")) !=
          render(fcmpExpr(c, FCmp::ULt, "a", "b")));
    CHECK(render(fcmpExpr(c, FCmp::ONe, "a", "b")) !=
          render(fcmpExpr(c, FCmp::UNe, "a", "b")));
  }

  CASE("min and max are plain calls when NaN need not propagate");
  {
    msl::Context c;
    CHECK_EQ(render(minMaxExpr(c, MathFn2::Min, f32(), "a", "b", false)),
             std::string("metal::min(a, b)"));
    CHECK_EQ(render(minMaxExpr(c, MathFn2::Max, f32(), "a", "b", false)),
             std::string("metal::max(a, b)"));
  }

  CASE("propagating a NaN produces one of the right type by adding");
  {
    // metal::min is IEEE minNum: it returns the other operand when one is NaN.
    // `a + b` produces a NaN of the right width with no per-type constant.
    msl::Context c;
    CHECK_EQ(render(minMaxExpr(c, MathFn2::Min, f32(), "a", "b", true)),
             std::string("(metal::isnan(a) || metal::isnan(b)) ? a + b : "
                         "metal::min(a, b)"));
  }

  CASE("an integer min never gets the NaN guard, however it is asked");
  {
    msl::Context c;
    CHECK(!minMaxPropagatesNan(MathFn2::Min, i32(), true));
    CHECK_EQ(render(minMaxExpr(c, MathFn2::Min, i32(), "a", "b", true)),
             std::string("metal::min(a, b)"));
  }

  CASE("a predicate always returns bool");
  {
    msl::Context c;
    CHECK(mathResultType(MathFn::Isnan, f32()) == i1());
    CHECK(mathResultType(MathFn::Exp, f32()) == f32());
    CHECK_EQ(render(emitMath(c, MathFn::Isnan, f32(), "d", c.var("v"))),
             std::string("bool d = metal::isnan(v);\n"));
    CHECK_EQ(render(emitMath(c, MathFn::Exp, f32(), "d", c.var("v"))),
             std::string("float d = metal::precise::exp(v);\n"));
  }

  CASE("the binary and ternary math families have spellings");
  {
    msl::Context c;
    CHECK_EQ(render(mathExpr(c, MathFn2::Pow, c.var("a"), c.var("b"))),
             std::string("metal::precise::pow(a, b)"));
    CHECK_EQ(render(mathExpr(c, MathFn2::Fmod, c.var("a"), c.var("b"))),
             std::string("metal::precise::fmod(a, b)"));
    CHECK_EQ(
        render(mathExpr(c, MathFn3::Fma, c.var("a"), c.var("b"), c.var("d"))),
        std::string("metal::fma(a, b, d)"));
    CHECK_EQ(render(mathExpr(c, MathFn3::Clamp, c.var("v"), c.var("lo"),
                             c.var("hi"))),
             std::string("metal::clamp(v, lo, hi)"));
  }

  CASE("a float-only function declines on an integer and mulhi the reverse");
  {
    CHECK(checkMath2(MathFn2::Pow, i32()).isDecline());
    CHECK(checkMath2(MathFn2::Min, i32()).ok());
    CHECK(checkMath2(MathFn2::Mulhi, i32()).ok());
    CHECK(checkMath2(MathFn2::Mulhi, f32()).isDecline());
    CHECK(checkMath3(MathFn3::Fma, i32()).isDecline());
  }

  CASE("select renders as a ternary expression");
  {
    // Both arms are already in registers; a branch would only add divergence.
    msl::Context c;
    CHECK_EQ(
        render(emitSelect(c, f32(), "d", c.var("p"), c.var("t"), c.var("f"))),
        std::string("float d = p ? t : f;\n"));
  }

  CASE("a bf16 operation casts both operands and declares f32");
  {
    // `bfloat d = a * b` leaves the evaluation width unspecified, and AGX2 and
    // AGX3 read it differently. Widening the declaration alone changes
    // nothing: the operands are what carry the ambiguity.
    msl::Context c;
    CHECK_EQ(render(emitEw(c, EwOp::Mul, bf16(), "d", c.var("a"), c.var("b"))),
             std::string("float d = (float)a * (float)b;\n"));
    CHECK_EQ(render(emitEw(c, EwOp::Add, f16(), "d", c.var("a"), c.var("b"))),
             std::string("float d = (float)a + (float)b;\n"));
  }

  CASE("a bf16 comparison widens its operands but still declares bool");
  {
    msl::Context c;
    CHECK_EQ(
        render(emitEw(c, EwOp::CmpLtS, bf16(), "d", c.var("a"), c.var("b"))),
        std::string("bool d = (float)a < (float)b;\n"));
  }

  CASE("an fp8 operation is left alone: it is spelled as a byte");
  {
    msl::Context c;
    CHECK_EQ(render(emitEw(c, EwOp::Mul, e4m3(), "d", c.var("a"), c.var("b"))),
             std::string("uchar d = a * b;\n"));
  }

  return ::agpu_test::report("EmitElementwise");
}
