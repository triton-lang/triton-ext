// ub.poison: MSL has no undef, so "anything" has to become something.
#include "agpu/emit/EmitPoison.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

int main() {
  CASE("a poisoned pointer renders as nullptr");
  {
    msl::Context c;
    const msl::Type ptr =
        msl::Type::scalar(msl::Scalar::F32).pointerTo(msl::AddrSpace::Device);
    CHECK_EQ(render(poisonPointer(c, ptr)), std::string("nullptr"));
  }

  CASE("a poisoned float keeps a float-typed zero");
  {
    msl::Context c;
    const std::string h = render(poisonDecl(c, "p", f16()));
    CHECK(h.find("half p =") != std::string::npos);
    CHECK(h.find("(half)") == std::string::npos);

    const std::string f = render(poisonDecl(c, "q", f32()));
    CHECK(f.find("float q =") != std::string::npos);
  }

  CASE("a poisoned integer is an integer zero");
  {
    msl::Context c;
    const std::string s = render(poisonDecl(c, "p", i32()));
    CHECK(s.find("int p = 0") != std::string::npos);
  }

  CASE("zero keeps a leaked poison in range, whatever it is scaled by");
  {
    msl::Context c;
    msl::Expr *idx = poisonValue(c, i32());
    msl::Expr *addr = c.binary(msl::BinOp::Mul, idx, c.lit(4096));
    CHECK_EQ(render(addr), std::string("0"));
  }

  CASE("two nulls of different pointee types are the same value");
  {
    msl::Context c;
    msl::Literal *a = c.litNull(
        msl::Type::scalar(msl::Scalar::F32).pointerTo(msl::AddrSpace::Device));
    msl::Literal *b = c.litNull(
        msl::Type::scalar(msl::Scalar::I32).pointerTo(msl::AddrSpace::Device));
    CHECK(a->sameValueAs(*b));
  }

  CASE("a null pointer declaration carries its pointee type");
  {
    msl::Context c;
    const msl::Type ptr =
        msl::Type::scalar(msl::Scalar::I32).pointerTo(msl::AddrSpace::Device);
    const std::string s = render(poisonPointerDecl(c, "p", ptr));
    CHECK(s.find("nullptr") != std::string::npos);
    CHECK(s.find("int") != std::string::npos);
  }

  return ::agpu_test::report("EmitPoison");
}
