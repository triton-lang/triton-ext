// How wide a load or store can go.
#include "agpu/plan/AccessWidth.h"
#include "fixtures.h"
#include "harness.h"

using namespace agpu;

namespace {

using agpu_test::contiguousBases;

PtrDims ptr(int64_t contig, int64_t align) {
  return PtrDims(2, PtrInfo{contig, align});
}

AccessPlan planAccess(const RegBases &bases, const PtrDims &ptr, VecElem elem) {
  return agpu::planAccess(bases, /*runtime=*/{}, ptr, elem);
}

} // namespace

int main() {
  CASE("registers stepping by powers of two along one dim are a run");
  {
    RegRun r = longestRegRun(contiguousBases(3));
    CHECK_EQ(r.length, 8);
    CHECK_EQ(r.dim, 1);
  }

  CASE("a single register is a run of one");
  {
    RegRun r = longestRegRun(contiguousBases(0));
    CHECK_EQ(r.length, 1);
    CHECK_EQ(r.dim, -1);
  }

  CASE("a run that turns a corner stops there");
  {
    RegBases b = contiguousBases(2);
    b.push_back({4, 0});
    RegRun r = longestRegRun(b);
    CHECK_EQ(r.length, 4);
    CHECK_EQ(r.dim, 1);
  }

  CASE("a bit moving along two dimensions ends the run");
  {
    RegBases b = contiguousBases(1);
    b.push_back({2, 2});
    CHECK_EQ(longestRegRun(b).length, 2);
  }

  CASE("a bit with the wrong stride ends the run");
  {
    RegBases b = contiguousBases(1);
    b.push_back({0, 4});
    CHECK_EQ(longestRegRun(b).length, 2);
  }

  CASE("a register that moves nothing ends the run");
  {
    RegBases b = contiguousBases(1);
    b.push_back({0, 0});
    CHECK_EQ(longestRegRun(b).length, 2);
  }

  CASE("a lane basis inside the run shrinks it to the boundary");
  {
    RegRun r = longestRegRun(contiguousBases(3), /*runtime=*/{0, 2});
    CHECK_EQ(r.length, 2);
    CHECK_EQ(r.dim, 1);
  }

  CASE("a lane basis past the run leaves it whole");
  {
    RegRun r = longestRegRun(contiguousBases(3), {0, 8 | 16});
    CHECK_EQ(r.length, 8);
  }

  CASE("a lane basis on another dimension does not shrink the run");
  {
    RegRun r = longestRegRun(contiguousBases(3), {3, 0});
    CHECK_EQ(r.length, 8);
  }

  CASE("a higher register bit with low bits set shrinks the run");
  {
    RegBases b = contiguousBases(2);
    b.push_back({0, 6});
    CHECK_EQ(longestRegRun(b).length, 2);

    RegBases odd = contiguousBases(2);
    odd.push_back({0, 3});
    CHECK_EQ(longestRegRun(odd).length, 1);
  }

  CASE("the width never exceeds Metal's widest portable vector");
  {
    AccessPlan p =
        planAccess(contiguousBases(3), ptr(16, 16), VecElem::Packable);
    CHECK_EQ(p.width, kMaxAccessWidth);
  }

  CASE("contiguity is a hard limit");
  {
    AccessPlan p =
        planAccess(contiguousBases(2), ptr(2, 16), VecElem::Packable);
    CHECK_EQ(p.width, 2);
  }

  CASE("contiguity below the element width kills vectorisation entirely");
  {
    AccessPlan p =
        planAccess(contiguousBases(3), ptr(1, 16), VecElem::Packable);
    CHECK_EQ(p.width, 1);
    CHECK(!p.vectorised());
  }

  CASE("an underaligned access goes wide through a packed type");
  {
    AccessPlan p = planAccess(contiguousBases(2), ptr(4, 2), VecElem::Packable);
    CHECK_EQ(p.width, 4);
    CHECK(p.packed);
  }

  CASE("an integer access packs too, keeping its full width");
  {
    AccessPlan p = planAccess(contiguousBases(2), ptr(4, 2), VecElem::Packable);
    CHECK_EQ(p.width, 4);
    CHECK(p.packed);
  }

  CASE("a properly aligned access needs no packed type");
  {
    AccessPlan p = planAccess(contiguousBases(2), ptr(4, 4), VecElem::Packable);
    CHECK_EQ(p.width, 4);
    CHECK(!p.packed);
  }

  CASE("8- and 64-bit elements have no vector access");
  {
    CHECK(vecElemOf(8) == VecElem::Packable);
    CHECK(vecElemOf(64) == VecElem::Unsupported);
    AccessPlan p = planAccess(contiguousBases(3), ptr(8, 8), vecElemOf(64));
    CHECK_EQ(p.width, 1);
  }

  CASE("16- and 32-bit floats are packable, integers are not");
  {
    CHECK(vecElemOf(32) == VecElem::Packable);
    CHECK(vecElemOf(16) == VecElem::Packable);
  }

  CASE("the chosen width never exceeds any of its three limits");
  {
    for (int bits = 0; bits <= 4; ++bits)
      for (int64_t contig : {1, 2, 4, 8, 16})
        for (int64_t align : {1, 2, 4, 8, 16}) {
          const RegBases b = contiguousBases(bits);
          AccessPlan p = planAccess(b, ptr(contig, align), VecElem::Packable);
          CHECK(p.width <= longestRegRun(b).length);
          CHECK(p.width <= contig);
          if (!p.packed)
            CHECK(p.width <= align);
          CHECK(p.width >= 1);
          CHECK((p.width & (p.width - 1)) == 0);
        }
  }

  CASE("a vectorised access names the dimension it walks");
  {
    AccessPlan p =
        planAccess(contiguousBases(2, /*dim=*/0), ptr(4, 4), VecElem::Packable);
    CHECK_EQ(p.width, 4);
    CHECK_EQ(p.dim, 0);
    AccessPlan s = planAccess(contiguousBases(0), ptr(4, 4), VecElem::Packable);
    CHECK_EQ(s.dim, -1);
  }

  CASE("a scalar access declines with a reason and is not a failure");
  {
    AccessPlan p = planAccess(contiguousBases(3), ptr(1, 1), VecElem::Packable);
    Decision d = widthDecision(p, VecElem::Packable);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK_EQ(d.why(), std::string("registers are not contiguous"));
  }

  CASE("an unsupported element width declines distinctly");
  {
    AccessPlan p =
        planAccess(contiguousBases(3), ptr(8, 8), VecElem::Unsupported);
    Decision d = widthDecision(p, VecElem::Unsupported);
    CHECK(d.isDecline());
    CHECK_EQ(d.why(), std::string("element width has no vector type"));
  }

  CASE("a vectorised access reports no decline");
  {
    AccessPlan p = planAccess(contiguousBases(2), ptr(4, 4), VecElem::Packable);
    CHECK(widthDecision(p, VecElem::Packable).ok());
  }

  return ::agpu_test::report("AccessWidth");
}
