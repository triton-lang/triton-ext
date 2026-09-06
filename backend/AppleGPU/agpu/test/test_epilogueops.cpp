// EpilogueOps tests.
#include "agpu/plan/EpilogueOps.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("the table covers the reference's five ops plus the four min/max");
  {
    CHECK_EQ(epilogueBinaryCount(), 9u);
    CHECK(isEpilogueBinary("arith.addf"));
    CHECK(isEpilogueBinary("arith.subf"));
    CHECK(isEpilogueBinary("arith.mulf"));
    CHECK(isEpilogueBinary("arith.divf"));
    CHECK(isEpilogueBinary("tt.precise_divf"));
    CHECK(isEpilogueBinary("arith.maxnumf"));
    CHECK(isEpilogueBinary("arith.minnumf"));
    CHECK(isEpilogueBinary("arith.maximumf"));
    CHECK(isEpilogueBinary("arith.minimumf"));
    EpilogueBinOp bin{};
    CHECK(epilogueBinOpOf("arith.maximumf", bin) &&
          bin == EpilogueBinOp::MaxPropagate);
    CHECK(epilogueBinOpOf("arith.maxnumf", bin) && bin == EpilogueBinOp::Max);
  }

  CASE("the invariant: recognised implies renderable");
  {
    for (const EpilogueBinary &e : kEpilogueBinary) {
      CHECK(isEpilogueBinary(e.op));
      EpilogueBinOp out{};
      CHECK(epilogueBinOpOf(e.op, out));
      CHECK(static_cast<int>(out) == static_cast<int>(e.bin));
    }
  }

  CASE("... and not recognised implies not rendered");
  {
    EpilogueBinOp out{};
    CHECK(!isEpilogueBinary("arith.addi"));
    CHECK(!epilogueBinOpOf("arith.addi", out));
    CHECK(!isEpilogueBinary("arith.shli"));
    CHECK(!isEpilogueBinary("arith.shrsi"));
    CHECK(!isEpilogueBinary("arith.shrui"));
  }

  CASE("unknown ops decline and leave the out parameter alone");
  {
    EpilogueBinOp out = EpilogueBinOp::Mul;
    CHECK(!epilogueBinOpOf("some.unknown.op", out));
    CHECK(out == EpilogueBinOp::Mul);
    CHECK(!isEpilogueBinary(""));
  }

  CASE("both divf spellings map to the same operator");
  {
    EpilogueBinOp a{}, b{};
    CHECK(epilogueBinOpOf("arith.divf", a));
    CHECK(epilogueBinOpOf("tt.precise_divf", b));
    CHECK(a == b);
    CHECK(a == EpilogueBinOp::Div);
  }

  CASE("lookup is a pure function of the name");
  {
    const EpilogueBinary *x = epilogueBinaryFor("arith.mulf");
    const EpilogueBinary *y = epilogueBinaryFor("arith.mulf");
    CHECK(x == y);
    CHECK(x != nullptr);
    CHECK(epilogueBinaryFor("nope") == nullptr);
  }

  return ::agpu_test::report("EpilogueOps");
}
