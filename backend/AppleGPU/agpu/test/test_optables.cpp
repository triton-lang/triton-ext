// The bridge's op tables: which agpu fact each MLIR op name carries.
#include "AgpuOpTables.h"
#include "agpu/emit/EmitElementwise.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <set>
#include <sstream>

using namespace agpu;
using agpu_test::render;
using namespace mlir::triton::applegpu::bridge;

namespace {

// `mathFnFor` fills its out-parameter only on a hit.
MathFn mathFnOf(const char *op) {
  MathFn got = MathFn::Exp;
  CHECK(mathFnFor(op, got));
  return got;
}

std::vector<std::string_view> allNames() {
  std::vector<std::string_view> out;
  for (std::string_view n : ewOpNames())
    out.push_back(n);
  for (std::string_view n : mathOpNames())
    out.push_back(n);
  for (std::string_view n : math2OpNames())
    out.push_back(n);
  for (std::string_view n : castOpNames())
    out.push_back(n);
  return out;
}

} // namespace

int main() {
  CASE("no op name appears in two tables");
  {
    // Dispatch runs handlers in registration order: first table wins.
    std::set<std::string_view> seen;
    for (std::string_view n : allNames()) {
      CHECK(seen.count(n) == 0);
      seen.insert(n);
    }
  }

  CASE("remf is fmod");
  {
    EwOp ew;
    CHECK(!ewOpFor("arith.remf", ew));
    const Math2Name *m = math2For("arith.remf");
    CHECK(m != nullptr);
    CHECK(m->fn == MathFn2::Fmod);
    CHECK(checkEw(EwOp::RemF, f32()).isDecline());
    CHECK(checkMath2(MathFn2::Fmod, f32()).ok());
  }

  CASE("the signed and unsigned integer ops are not confused");
  {
    EwOp ew;
    CHECK(ewOpFor("arith.divsi", ew) && ew == EwOp::DivS);
    CHECK(ewOpFor("arith.divui", ew) && ew == EwOp::DivU);
    CHECK(ewOpFor("arith.remsi", ew) && ew == EwOp::RemS);
    CHECK(ewOpFor("arith.remui", ew) && ew == EwOp::RemU);
    CHECK(ewOpFor("arith.shrsi", ew) && ew == EwOp::ShrS);
    CHECK(ewOpFor("arith.shrui", ew) && ew == EwOp::ShrU);
    CHECK(!needsUnsigned(EwOp::DivS));
    CHECK(needsUnsigned(EwOp::DivU));
  }

  CASE("every operator row has a spelling its handler can emit");
  {
    for (const EwName &e : kEwNames) {
      msl::BinOp bo;
      CHECK(spellingOf(e.ew, bo));
    }
  }

  CASE("every math row has a spelling and none is the Count sentinel");
  {
    for (const MathName &m : kMathNames) {
      CHECK(m.fn != MathFn::Count);
      CHECK(mathNameOf(m.fn) != nullptr);
    }
  }

  CASE("round and roundeven are different functions");
  {
    // Half away from zero versus half to even.
    const MathFn a = mathFnOf("math.round");
    const MathFn b = mathFnOf("math.roundeven");
    CHECK(a == MathFn::Round);
    CHECK(b == MathFn::RoundEven);
    CHECK(std::string(mathNameOf(a)) != std::string(mathNameOf(b)));
  }

  CASE("absf and absi share one function, which is not float-only");
  {
    const MathFn a = mathFnOf("math.absf");
    const MathFn b = mathFnOf("math.absi");
    CHECK(a == b);
    CHECK(checkMath(a, i32()).ok());
    CHECK(checkMath(MathFn::Exp, i32()).isDecline());
  }

  CASE("exp and sqrt both take the precise spelling");
  {
    const MathFn e = mathFnOf("math.exp");
    const MathFn s = mathFnOf("math.sqrt");
    CHECK_EQ(std::string(mathNameOf(e)), std::string("metal::precise::exp"));
    // metal::sqrt(12.25) answers 3.5000002, one ulp above the exact root.
    CHECK_EQ(std::string(mathNameOf(s)), std::string("metal::precise::sqrt"));
  }

  CASE("tt.precise_sqrt maps to metal::precise::sqrt");
  {
    CHECK(mathFnOf("tt.precise_sqrt") == mathFnOf("math.sqrt"));
  }

  CASE("minimumf propagates NaN and minnumf does not");
  {
    // `metal::min` returns the other operand on a NaN: IEEE minNum.
    const Math2Name *mi = math2For("arith.minimumf");
    const Math2Name *mn = math2For("arith.minnumf");
    CHECK(mi && mn);
    CHECK(mi->fn == mn->fn);
    CHECK(mi->propagateNan);
    CHECK(!mn->propagateNan);

    msl::Context c;
    const std::string prop =
        render(minMaxExpr(c, mi->fn, f32(), "a", "b", mi->propagateNan));
    const std::string plain =
        render(minMaxExpr(c, mn->fn, f32(), "a", "b", mn->propagateNan));
    CHECK(prop != plain);
    CHECK(prop.find("isnan") != std::string::npos);
    CHECK_EQ(plain, std::string("metal::min(a, b)"));
  }

  CASE("maximumf and minimumf are not the same call");
  {
    const Math2Name *mx = math2For("arith.maximumf");
    const Math2Name *mi = math2For("arith.minimumf");
    CHECK(mx && mi && mx->fn != mi->fn);
    CHECK_EQ(std::string(mathNameOf(mx->fn)), std::string("metal::max"));
    CHECK_EQ(std::string(mathNameOf(mi->fn)), std::string("metal::min"));
  }

  CASE("only the ui rows read their operands as unsigned");
  {
    // MLIR integers are signless: the op says how to read them.
    CHECK(math2For("arith.minui")->readsUnsigned);
    CHECK(math2For("arith.maxui")->readsUnsigned);
    CHECK(!math2For("arith.minsi")->readsUnsigned);
    CHECK(!math2For("arith.maxsi")->readsUnsigned);
    CHECK(!math2For("arith.minimumf")->readsUnsigned);

    // `tt.mulhiui` is the high half of an unsigned product.
    CHECK(math2For("tt.mulhiui")->readsUnsigned);
  }

  CASE("a ui row's name and its flag agree");
  {
    for (const Math2Name &m : kMath2Names) {
      const std::string_view op = m.op;
      if (op.size() > 2 && op.substr(op.size() - 2) == "ui")
        CHECK(m.readsUnsigned);
    }
  }

  CASE("every math2 row applies to some type its handler will see");
  {
    for (const Math2Name &m : kMath2Names) {
      CHECK(mathNameOf(m.fn) != nullptr);
      CHECK(checkMath2(m.fn, f32()).ok() || checkMath2(m.fn, i32()).ok());
    }
  }

  CASE("mulhi is an integer op and the float-only ones refuse integers");
  {
    CHECK(math2For("tt.mulhiui")->fn == MathFn2::Mulhi);
    CHECK(checkMath2(MathFn2::Mulhi, f32()).isDecline());
    CHECK(checkMath2(MathFn2::Pow, i32()).isDecline());
    CHECK(checkMath2(MathFn2::Atan2, i32()).isDecline());
  }

  CASE("only the bit-preserving ops reinterpret");
  {
    // `as_type<uint>(1.0f)` is 0x3f800000 and `(uint)1.0f` is 1.
    CHECK(castFor("arith.bitcast")->reinterpret);
    CHECK(castFor("tt.bitcast")->reinterpret);
    CHECK(castFor("tt.int_to_ptr")->reinterpret);
    CHECK(castFor("tt.ptr_to_int")->reinterpret);
    CHECK(!castFor("arith.sitofp")->reinterpret);
    CHECK(!castFor("arith.fptosi")->reinterpret);
    CHECK(!castFor("arith.extf")->reinterpret);
  }

  CASE("extui zero-extends where extsi sign-extends");
  {
    CHECK(castFor("arith.extui")->readsUnsigned);
    CHECK(!castFor("arith.extsi")->readsUnsigned);
    CHECK(castFor("arith.uitofp")->readsUnsigned);
    CHECK(!castFor("arith.sitofp")->readsUnsigned);
  }

  CASE("fptoui declares an unsigned result and fptosi does not");
  {
    CHECK(castFor("arith.fptoui")->writesUnsigned);
    CHECK(!castFor("arith.fptosi")->writesUnsigned);
    CHECK(!castFor("arith.extui")->writesUnsigned);
  }

  CASE("a pointer spells as a pointer");
  {
    ElemType ptr;
    ptr.kind = ElemType::Kind::Pointer;
    ptr.bits = 64;
    ptr.pointee = msl::Scalar::I32;

    const msl::Type t = mslTypeOf(ptr);
    CHECK(t.form() == msl::Type::Form::Pointer);
    CHECK(t.scalarKind() == msl::Scalar::I32);
    CHECK(!(t == mslTypeOf(ElemType{ElemType::Kind::Int, 64, true})));

    // The pointee is part of the type.
    ElemType other = ptr;
    other.pointee = msl::Scalar::F32;
    CHECK(!(ptr == other));
    CHECK(i32() == i32());
  }

  CASE("every cast row is a conversion planConvert can answer");
  {
    CHECK(planConvert(f16(), f32(), Rounding::Default).usable());
    CHECK(planConvert(f32(), f16(), Rounding::Default).usable());
    CHECK(planConvert(i32(), f32(), Rounding::Default).usable());
    CHECK(planConvert(f32(), i32(), Rounding::Default).usable());
    CHECK(planConvert(f32(), f32(), Rounding::Default).kind ==
          ConvertKind::None);
  }

  CASE("rtz asks for a helper and the default rounding does not");
  {
    // MSL offers no way to request a rounding mode, so RTZ is a bit operation
    // on the mantissa.
    CHECK(planConvert(f32(), f16(), Rounding::RTZ).needsHelper());
    CHECK(planConvert(f32(), f16(), Rounding::RTNE).needsHelper());
    CHECK(!planConvert(f32(), f16(), Rounding::Default).needsHelper());
  }

  return ::agpu_test::report("OpTables");
}
