#include "agpu/emit/EmitReshape.h"
#include "harness.h"

using namespace agpu;

namespace {

msl::SmallVec<msl::Str, 8> namesOf(int n, const char *prefix = "v") {
  msl::SmallVec<msl::Str, 8> out;
  for (int i = 0; i < n; ++i)
    out.push_back(msl::Str(prefix) + std::to_string(i));
  return out;
}

} // namespace

int main() {
  CASE("join reads the operand off the trailing coordinate");
  {
    InterleaveFacts f;
    f.src = {{0}, {1}, {2}};
    f.dst = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}};

    const InterleavePlan p = planJoinFrom(f);
    CHECK(p.usable);
    const auto out = interleaveNames(p, namesOf(3, "a"), namesOf(3, "b"));
    CHECK_EQ((int)out.size(), 6);
    CHECK_EQ(out[0], std::string("a0"));
    CHECK_EQ(out[1], std::string("b0"));
    CHECK_EQ(out[2], std::string("a1"));
    CHECK_EQ(out[3], std::string("b1"));
  }

  CASE("join follows the layout's own register numbering");
  {
    InterleaveFacts f;
    f.src = {{0}, {1}, {2}};
    f.dst = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}};

    const auto out =
        interleaveNames(planJoinFrom(f), namesOf(3, "a"), namesOf(3, "b"));
    CHECK_EQ((int)out.size(), 6);
    CHECK_EQ(out[0], std::string("a0"));
    CHECK_EQ(out[1], std::string("a1"));
    CHECK_EQ(out[2], std::string("a2"));
    CHECK_EQ(out[3], std::string("b0"));
    CHECK_EQ(out[4], std::string("b1"));
    CHECK_EQ(out[5], std::string("b2"));
  }

  CASE("split is join's inverse");
  {
    const auto lhs = namesOf(4, "a");
    const auto rhs = namesOf(4, "b");

    InterleaveFacts jf;
    for (int32_t i = 0; i < 4; ++i)
      jf.src.push_back({i});
    for (int32_t i = 0; i < 4; ++i) {
      jf.dst.push_back({i, 0});
      jf.dst.push_back({i, 1});
    }
    const auto joined = interleaveNames(planJoinFrom(jf), lhs, rhs);
    CHECK_EQ((int)joined.size(), 8);

    InterleaveFacts sf;
    sf.src = jf.dst;
    for (int32_t i = 0; i < 4; ++i)
      sf.dst.push_back({i});

    const auto left = interleaveNames(planSplitFrom(sf, 0), joined, {});
    const auto right = interleaveNames(planSplitFrom(sf, 1), joined, {});
    CHECK_EQ((int)left.size(), 4);
    for (int i = 0; i < 4; ++i) {
      CHECK_EQ(left[(std::size_t)i], lhs[(std::size_t)i]);
      CHECK_EQ(right[(std::size_t)i], rhs[(std::size_t)i]);
    }
  }

  CASE("a result register with no source coordinate is not usable");
  {
    InterleaveFacts f;
    f.src = {{0}, {1}};
    f.dst = {{0, 0}, {5, 1}};
    CHECK(!planJoinFrom(f).usable);
    CHECK(!interleaveDecision(planJoinFrom(f)).ok());

    InterleaveFacts bad;
    bad.src = {{0}};
    bad.dst = {{0, 2}};
    CHECK(!planJoinFrom(bad).usable);

    InterleaveFacts empty;
    CHECK(!planJoinFrom(empty).usable);
    CHECK(!planSplitFrom(empty, 0).usable);
    CHECK(!planSplitFrom(empty, 2).usable);
  }

  CASE("two fp4 elements come from one byte, low nibble first");
  {
    InterleaveFacts f;
    f.src = {{0}, {1}};
    f.dst = {{0}, {1}, {2}, {3}};

    const Fp4UnpackPlan p = planFp4Unpack(f, 0);
    CHECK(p.usable);
    CHECK_EQ((int)p.from.size(), 4);
    CHECK_EQ(p.from[0].reg, 0);
    CHECK(!p.from[0].high);
    CHECK_EQ(p.from[1].reg, 0);
    CHECK(p.from[1].high);
    CHECK_EQ(p.from[2].reg, 1);
    CHECK(!p.from[2].high);
    CHECK_EQ(p.from[3].reg, 1);
    CHECK(p.from[3].high);
  }

  CASE("the nibble follows the layout's own numbering");
  {
    InterleaveFacts f;
    f.src = {{0}, {1}};
    f.dst = {{0}, {2}, {1}, {3}};

    const Fp4UnpackPlan p = planFp4Unpack(f, 0);
    CHECK(p.usable);
    CHECK_EQ(p.from[0].reg, 0);
    CHECK(!p.from[0].high);
    CHECK_EQ(p.from[1].reg, 1);
    CHECK(!p.from[1].high);
    CHECK_EQ(p.from[2].reg, 0);
    CHECK(p.from[2].high);
    CHECK_EQ(p.from[3].reg, 1);
    CHECK(p.from[3].high);
  }

  CASE("the packed axis is the one the op names");
  {
    InterleaveFacts f;
    f.src = {{0, 0}, {1, 0}};
    f.dst = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    const Fp4UnpackPlan p = planFp4Unpack(f, 1);
    CHECK(p.usable);
    CHECK_EQ(p.from[0].reg, 0);
    CHECK(!p.from[0].high);
    CHECK_EQ(p.from[1].reg, 0);
    CHECK(p.from[1].high);
    CHECK_EQ(p.from[2].reg, 1);
    CHECK(!p.from[2].high);
  }

  CASE("an fp4 result with no source byte is not usable");
  {
    InterleaveFacts f;
    f.src = {{0}};
    f.dst = {{0}, {9}};
    CHECK(!planFp4Unpack(f, 0).usable);

    InterleaveFacts f2;
    f2.src = {{0}};
    f2.dst = {{0}};
    CHECK(!planFp4Unpack(f2, 3).usable);
    CHECK(!planFp4Unpack(f2, -1).usable);

    InterleaveFacts empty;
    CHECK(!planFp4Unpack(empty, 0).usable);
  }

  return ::agpu_test::report("Reshape");
}
