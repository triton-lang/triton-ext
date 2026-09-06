// Elementwise operations and the type they produce.
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/MathFn.h"
#include "harness.h"

using namespace agpu;

namespace {

ElemType intOf(unsigned bits, bool uns = false) {
  return {ElemType::Kind::Int, bits, uns};
}

const EwOp kAllOps[] = {EwOp::Add,    EwOp::Sub,    EwOp::Mul,   EwOp::DivS,
                        EwOp::DivU,   EwOp::RemS,   EwOp::RemU,  EwOp::And,
                        EwOp::Or,     EwOp::Xor,    EwOp::Shl,   EwOp::ShrS,
                        EwOp::ShrU,   EwOp::CmpEq,  EwOp::CmpNe, EwOp::CmpLtS,
                        EwOp::CmpLtU, EwOp::CmpLeS, EwOp::CmpLeU};

} // namespace

int main() {
  // ── the signedness invariant ───────────────────────────────────────────

  CASE("an unsigned operation promotes both the operands and the result");
  {
    // MSL has one `/`: which division you get is decided by the type it is
    // applied to.
    EwTypes t = typesFor(EwOp::DivU, intOf(32));
    CHECK(t.operand.isUnsigned);
    CHECK(t.result.isUnsigned);
    CHECK_EQ(t.operand.bits, 32u);
    CHECK_EQ(t.result.bits, 32u);
  }

  CASE("a signed operation promotes neither");
  {
    EwTypes t = typesFor(EwOp::DivS, intOf(32));
    CHECK(!t.operand.isUnsigned);
    CHECK(!t.result.isUnsigned);
  }

  CASE("the operand and result signedness agree for every arithmetic op");
  {
    for (EwOp op : kAllOps) {
      if (isComparison(op))
        continue;
      EwTypes t = typesFor(op, intOf(32));
      CHECK_EQ(t.operand.isUnsigned, t.result.isUnsigned);
      CHECK_EQ(t.operand.isUnsigned, needsUnsigned(op));
    }
  }

  CASE("unsigned shift-right is a promotion and signed is not");
  {
    CHECK(needsUnsigned(EwOp::ShrU));
    CHECK(!needsUnsigned(EwOp::ShrS));
    CHECK(typesFor(EwOp::ShrU, intOf(32)).operand.isUnsigned);
    CHECK(!typesFor(EwOp::ShrS, intOf(32)).operand.isUnsigned);
  }

  CASE("both shifts spell the same operator");
  {
    msl::BinOp s, u;
    CHECK(spellingOf(EwOp::ShrS, s));
    CHECK(spellingOf(EwOp::ShrU, u));
    CHECK(s == u);
    CHECK(typesFor(EwOp::ShrS, intOf(32)).operand.isUnsigned !=
          typesFor(EwOp::ShrU, intOf(32)).operand.isUnsigned);
  }

  // ── comparisons ────────────────────────────────────────────────────────

  CASE("a planned comparison yields bool whatever it consumed");
  {
    for (EwOp op : kAllOps) {
      if (!isComparison(op))
        continue;
      CHECK(typesFor(op, intOf(32)).result.kind == ElemType::Kind::Bool);
      CHECK(typesFor(op, f32()).result.kind == ElemType::Kind::Bool);
    }
  }

  CASE("an unsigned comparison reads unsigned but yields plain bool");
  {
    EwTypes t = typesFor(EwOp::CmpLtU, intOf(32));
    CHECK(t.operand.isUnsigned);
    CHECK(t.result.kind == ElemType::Kind::Bool);
    CHECK(!t.result.isUnsigned);
  }

  // ── i1 ─────────────────────────────────────────────────────────────────

  CASE("i1 arithmetic stays i1");
  {
    EwTypes t = typesFor(EwOp::Add, i1());
    CHECK(t.result.kind == ElemType::Kind::Bool);
    CHECK(t.operand.kind == ElemType::Kind::Bool);
  }

  CASE("an unsigned op on i1 does not promote it");
  {
    EwTypes t = typesFor(EwOp::DivU, i1());
    CHECK(t.result.kind == ElemType::Kind::Bool);
    CHECK(!t.result.isUnsigned);
  }

  // ── MSL spellings ──────────────────────────────────────────────────────

  CASE("integer widths map to their MSL names, signed and unsigned");
  {
    using S = msl::Scalar;
    CHECK(mslTypeOf(intOf(32)).scalarKind() == S::I32);
    CHECK(mslTypeOf(intOf(32, true)).scalarKind() == S::U32);
    CHECK(mslTypeOf(intOf(8)).scalarKind() == S::I8);
    CHECK(mslTypeOf(intOf(8, true)).scalarKind() == S::U8);
    CHECK(mslTypeOf(intOf(64, true)).scalarKind() == S::U64);
    CHECK(mslTypeOf(i1()).scalarKind() == S::Bool);
    CHECK(mslTypeOf(f32()).scalarKind() == S::F32);
  }

  CASE("a divui declares uint, end to end");
  {
    EwTypes t = typesFor(EwOp::DivU, intOf(32));
    CHECK(mslTypeOf(t.result).scalarKind() == msl::Scalar::U32);
    CHECK(mslTypeOf(t.operand).scalarKind() == msl::Scalar::U32);
    EwTypes s = typesFor(EwOp::DivS, intOf(32));
    CHECK(mslTypeOf(s.result).scalarKind() == msl::Scalar::I32);
  }

  CASE("every op in the table has a spelling");
  {
    for (EwOp op : kAllOps) {
      msl::BinOp b;
      CHECK(spellingOf(op, b));
    }
  }

  // ── declining ──────────────────────────────────────────────────────────

  CASE("a bitwise op on a float declines with a reason");
  {
    Decision d = checkEw(EwOp::And, f32());
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK_EQ(d.why(), std::string("bitwise op on a float"));
  }

  CASE("an integer remainder on a float declines");
  {
    CHECK(checkEw(EwOp::RemS, f32()).isDecline());
    CHECK(checkEw(EwOp::RemU, f32()).isDecline());
    CHECK(checkEw(EwOp::RemS, intOf(32)).ok());
  }

  CASE("arithmetic and comparison apply to both classes");
  {
    for (EwOp op : {EwOp::Add, EwOp::Sub, EwOp::Mul, EwOp::CmpEq}) {
      CHECK(checkEw(op, intOf(32)).ok());
      CHECK(checkEw(op, f32()).ok());
    }
  }

  // ── math functions ─────────────────────────────────────────────────────

  CASE("math functions name their MSL spelling");
  {
    CHECK_EQ(std::string(mathNameOf(MathFn::Rsqrt)),
             std::string("metal::rsqrt"));
    CHECK_EQ(std::string(mathNameOf(MathFn::Exp2)), std::string("metal::exp2"));

    CHECK_EQ(std::string(mathNameOf(MathFn::Tanh)),
             std::string("metal::precise::tanh"));

    CHECK_EQ(std::string(mathNameOf(MathFn::Exp)),
             std::string("metal::precise::exp"));
  }

  CASE("every MathFn has a spelling");
  {
    for (int i = 0; i < (int)MathFn::Count; ++i) {
      const MathFn fn = (MathFn)i;
      const char *name = mathNameOf(fn);
      CHECK(name != nullptr);
      if (!name)
        continue;
      CHECK(name[0] != '\0');
      CHECK(checkMath(fn, f32()).ok());
    }
  }

  CASE("every MathFn2 and MathFn3 has a spelling too");
  {
    for (int i = 0; i < (int)MathFn2::Count; ++i) {
      const char *name = mathNameOf((MathFn2)i);
      CHECK(name != nullptr);
      if (name)
        CHECK(name[0] != '\0');
    }
    for (int i = 0; i < (int)MathFn3::Count; ++i) {
      const char *name = mathNameOf((MathFn3)i);
      CHECK(name != nullptr);
      if (name)
        CHECK(name[0] != '\0');
    }
  }

  CASE("f64 narrows to f32 and something can ask before it happens");
  {
    // Metal has no double, so f64 narrows to f32.
    CHECK(narrowsSilently(f64()));
    CHECK(mslTypeOf(f64()) == mslTypeOf(f32()));
    CHECK(narrowedTo(f64()) == f32());
    CHECK(narrowedTo(f16()) == f16());

    for (ElemType e : {f32(), f16(), bf16(), e4m3(), e5m2(), i32(), i1()})
      CHECK(!narrowsSilently(e));

    CHECK(!narrowsSilently(ElemType{ElemType::Kind::Int, 64, false}));
    CHECK(mslTypeOf(ElemType{ElemType::Kind::Int, 64, false}) ==
          msl::Type::scalar(msl::Scalar::I64));
  }

  CASE("abs covers integers, which is why math.absi needs no entry");
  {
    CHECK(checkMath(MathFn::Abs, i32()).ok());
    CHECK(checkMath(MathFn::Abs, f32()).ok());
    CHECK_EQ(std::string(mathNameOf(MathFn::Abs)), std::string("metal::abs"));

    // metal::abs is overloaded both ways; given a bfloat the compiler cannot
    // choose, so bf16/f16/f32 re-spell to fabs while int keeps abs.
    for (ElemType f : {f32(), f16(), bf16()})
      CHECK_EQ(std::string(mathNameOf(MathFn::Abs, f)),
               std::string("metal::fabs"));
    CHECK_EQ(std::string(mathNameOf(MathFn::Abs, i32())),
             std::string("metal::abs"));
    CHECK_EQ(std::string(mathNameOf(MathFn::Floor, f32())),
             std::string(mathNameOf(MathFn::Floor)));

    CHECK(mathResultNarrows(MathFn::Abs, f16()));
    CHECK(mathResultNarrows(MathFn::Abs, bf16()));
    CHECK(!mathResultNarrows(MathFn::Abs, f32()));
    CHECK(!mathResultNarrows(MathFn::Abs, i32()));
    CHECK(!mathResultNarrows(MathFn::Isnan, f16()));

    CHECK(checkMath(MathFn::Floor, i32()).isDecline());
    CHECK(checkMath(MathFn::Round, i32()).isDecline());
    CHECK(checkMath(MathFn::Tan, i32()).isDecline());
  }

  CASE("round and roundeven are genuinely different functions");
  {
    // round takes a half away from zero, roundeven takes it to even:
    // round(+2.5) is 3 and rint(+2.5) is 2.
    CHECK_EQ(std::string(mathNameOf(MathFn::Round)),
             std::string("metal::round"));
    CHECK_EQ(std::string(mathNameOf(MathFn::RoundEven)),
             std::string("metal::rint"));
    CHECK(std::string(mathNameOf(MathFn::Round)) !=
          std::string(mathNameOf(MathFn::RoundEven)));
    CHECK_EQ(std::string(mathNameOf(MathFn::Trunc)),
             std::string("metal::trunc"));
  }

  CASE("every transcendental is precise and only the measured ones are not");
  {
    for (MathFn fn :
         {MathFn::Exp, MathFn::Exp10, MathFn::Log, MathFn::Log2, MathFn::Log10,
          MathFn::Sin, MathFn::Cos, MathFn::Tanh, MathFn::Tan, MathFn::Asin,
          MathFn::Acos, MathFn::Atan, MathFn::Sinh, MathFn::Cosh, MathFn::Sqrt})
      CHECK(std::string(mathNameOf(fn)).find("precise::") != std::string::npos);

    for (MathFn fn : {MathFn::Round, MathFn::RoundEven, MathFn::Trunc})
      CHECK(std::string(mathNameOf(fn)).find("precise::") == std::string::npos);

    for (MathFn fn : {MathFn::Exp2, MathFn::Rsqrt, MathFn::Abs, MathFn::Floor,
                      MathFn::Ceil})
      CHECK(std::string(mathNameOf(fn)).find("precise::") == std::string::npos);

    CHECK(std::string(mathNameOf(MathFn2::Pow)).find("precise::") !=
          std::string::npos);
    CHECK(std::string(mathNameOf(MathFn2::Atan2)).find("precise::") !=
          std::string::npos);
    CHECK(std::string(mathNameOf(MathFn2::Min)).find("precise::") ==
          std::string::npos);
  }

  CASE("a float-only function declines on an integer");
  {
    Decision d = checkMath(MathFn::Exp, intOf(32));
    CHECK(d.isDecline());
    CHECK_EQ(d.why(), std::string("float-only function on a non-float"));
    CHECK(checkMath(MathFn::Exp, f32()).ok());
  }

  CASE("abs applies to both classes");
  {
    CHECK(checkMath(MathFn::Abs, f32()).ok());
    CHECK(checkMath(MathFn::Abs, intOf(32)).ok());
  }

  // ── evaluation width ──────────────────────────────────────────────────

  CASE("the 16-bit floats evaluate at f32");
  {
    CHECK(widensToF32(bf16()));
    CHECK(widensToF32(f16()));
    CHECK(evalWidthFor(bf16()) == f32());
    CHECK(evalWidthFor(f16()) == f32());
  }

  CASE("fp8 does not widen: spelled as a byte, an f32 vector would truncate");
  {
    for (ElemType e : {e4m3(), e5m2(), e4b8(), e5b16()}) {
      CHECK(!widensToF32(e));
      CHECK(evalWidthFor(e) == e);
    }
  }

  CASE("nothing wider or non-float widens");
  {
    CHECK(!widensToF32(f32()));
    CHECK(!widensToF32(intOf(16)));
    CHECK(!widensToF32(i1()));
    CHECK(evalWidthFor(f32()) == f32());
    CHECK(evalWidthFor(intOf(16)) == intOf(16));
  }

  CASE("a widened comparison still yields bool");
  {
    const EwTypes t = typesFor(EwOp::CmpLtS, bf16());
    CHECK(evalWidthFor(t.operand) == f32());
    CHECK(evalWidthFor(t.result) == i1());
  }

  return ::agpu_test::report("Elementwise");
}
