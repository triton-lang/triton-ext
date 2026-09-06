#include "agpu/emit/EmitShuffle.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

std::vector<int32_t> identityLanes() {
  std::vector<int32_t> m(kWarpSize);
  for (int32_t l = 0; l < (int32_t)kWarpSize; ++l)
    m[l] = l;
  return m;
}

std::vector<int32_t> xorLanes(int32_t mask) {
  std::vector<int32_t> m(kWarpSize);
  for (int32_t l = 0; l < (int32_t)kWarpSize; ++l)
    m[l] = l ^ mask;
  return m;
}

msl::SmallVec<msl::Str, 8> names(const char *p, int n) {
  msl::SmallVec<msl::Str, 8> v;
  for (int i = 0; i < n; ++i)
    v.push_back(msl::Str(p) + std::to_string(i));
  return v;
}

} // namespace

int main() {
  ShuffleNames nm;
  const ElemType elem = f32();

  CASE("an XOR permutation is affine, with an identity basis and an offset");
  {
    // Affine: `lane ^ 5` maps 0 to 5.
    std::vector<int32_t> basis;
    int32_t offset = 0;
    CHECK(permIsLinear(xorLanes(0b101), basis, offset));
    CHECK_EQ(offset, 0b101);
    CHECK_EQ(basis.size(), 5u);
    if (basis.size() == 5u)
      for (int b = 0; b < 5; ++b)
        CHECK_EQ(basis[b], 1 << b);
  }

  CASE("the identity permutation has no offset");
  {
    std::vector<int32_t> basis;
    int32_t offset = 0;
    CHECK(permIsLinear(identityLanes(), basis, offset));
    CHECK_EQ(offset, 0);
    for (std::size_t b = 0; b < basis.size(); ++b)
      CHECK_EQ(basis[b], 1 << b);
  }

  CASE("a permutation that moves lane 0 can still be affine");
  {
    std::vector<int32_t> basis;
    int32_t offset = 0;
    CHECK(permIsLinear(xorLanes(3), basis, offset));
    CHECK_EQ(offset, 3);
  }

  CASE("linearity is verified exhaustively over every lane");
  {
    // A permutation can agree with an affine map on every basis vector and
    // disagree elsewhere.
    std::vector<int32_t> m = identityLanes();
    std::swap(m[3], m[5]);
    std::vector<int32_t> basis;
    int32_t offset = 0;
    CHECK(!permIsLinear(m, basis, offset));
    CHECK(basis.empty());
    CHECK_EQ(offset, 0);
  }

  CASE("registers sharing one permutation compute the lane once");
  {
    msl::Context c;
    msl::Block body;
    ShufflePlan p = planShuffle({0, 1}, {xorLanes(4), xorLanes(4)});
    CHECK(p.uniformLanePerm);
    CHECK(p.linearLanePerm);
    emitShuffle(c, body, p, names("v", 2), names("d", 2), elem, nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "int sl ="), 1);
    CHECK_EQ(countOf(out, "simd_shuffle"), 2);
  }

  CASE("differing permutations cannot be spelled at all");
  {
    // The lane index is built once, so two registers wanting different
    // permutations have no single form.
    ShufflePlan p = planShuffle({0, 1}, {xorLanes(4), xorLanes(8)});
    CHECK(!p.uniformLanePerm);
    CHECK(!p.usable());

    msl::Context c;
    msl::Block body;
    CHECK(emitShuffle(c, body, p, names("v", 2), names("d", 2), elem, nm)
              .empty());
    CHECK_EQ(body.size(), 0u);
  }

  CASE("a non-linear permutation falls back to a table");
  {
    msl::Context c;
    msl::Block body;
    std::vector<int32_t> m = identityLanes();
    std::swap(m[3], m[5]);
    ShufflePlan p = planShuffle({0}, {m});
    CHECK(p.usable());
    CHECK(!p.linearLanePerm);
    emitShuffle(c, body, p, names("v", 1), names("d", 1), elem, nm);
    CHECK(render(body).find("lanetab[lane]") != std::string::npos);
  }

  CASE("a conversion where every lane keeps its own is a rename");
  {
    // Nothing emitted: the result names are the source names.
    msl::Context c;
    msl::Block body;
    ShufflePlan p = planShuffle({0, 1}, {identityLanes(), identityLanes()});
    CHECK(p.isRebind());
    CHECK_EQ(p.shuffleCount(), 0);
    auto out = emitShuffle(c, body, p, names("v", 2), names("d", 2), elem, nm);
    CHECK_EQ(body.size(), 0u);
    CHECK_EQ(out[0], std::string("v0"));
    CHECK_EQ(out[1], std::string("v1"));
  }

  CASE("a register that needs no shuffle is left alone even when others do");
  {
    msl::Context c;
    msl::Block body;
    ShufflePlan p = planShuffle({0, 1}, {identityLanes(), xorLanes(2)});
    CHECK(!p.isRebind());
    CHECK_EQ(p.shuffleCount(), 1);
    auto out = emitShuffle(c, body, p, names("v", 2), names("d", 2), elem, nm);
    CHECK_EQ(countOf(render(body), "simd_shuffle"), 1);
    CHECK_EQ(out[0], std::string("v0")); // unchanged
    CHECK_EQ(out[1], std::string("d1"));
  }

  CASE("each step reads the source register it names");
  {
    // A conversion permutes registers as well as lanes, so a step's srcReg
    // is not its own index.
    msl::Context c;
    msl::Block body;
    ShufflePlan p = planShuffle({1, 0}, {xorLanes(1), xorLanes(1)});
    emitShuffle(c, body, p, names("v", 2), names("d", 2), elem, nm);
    const std::string out = render(body);
    CHECK(out.find("float d0 = simd_shuffle(v1") != std::string::npos);
    CHECK(out.find("float d1 = simd_shuffle(v0") != std::string::npos);
  }

  CASE("a lane outside the warp is not a shuffle at all");
  {
    // Elements from another warp need the pool.
    std::vector<int32_t> m = identityLanes();
    m[0] = 40;
    ShufflePlan p = planShuffle({0}, {m});
    CHECK(!p.usable());
    Decision d = shuffleDecision(p);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK_EQ(d.why(), std::string("elements cross a warp boundary"));
  }

  CASE("a usable shuffle plan reports no decline");
  {
    CHECK(shuffleDecision(planShuffle({0}, {xorLanes(1)})).ok());
  }

  CASE("mismatched register and lane-map counts produce nothing");
  {
    CHECK(!planShuffle({0, 1}, {xorLanes(1)}).usable());
  }

  CASE("a shuffled conversion emits no barrier and no pool traffic");
  {
    msl::Context c;
    msl::Block body;
    ShufflePlan p = planShuffle(
        {0, 1, 2, 3}, {xorLanes(1), xorLanes(1), xorLanes(1), xorLanes(1)});
    emitShuffle(c, body, p, names("v", 4), names("d", 4), elem, nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "barrier"), 0);
    CHECK_EQ(countOf(out, "["), 0);
    CHECK_EQ(countOf(out, "simd_shuffle"), 4);
  }

  CASE("a layout change whose element is already in the lane is a rebind");
  {
    std::vector<int64_t> ident;
    for (int64_t l = 0; l < (int64_t)kWarpSize; ++l)
      ident.push_back(l);

    const ShufflePlan p = planShuffleFromElems({ident}, {ident});
    CHECK(p.usable());
    CHECK(p.isRebind());
    CHECK_EQ(p.shuffleCount(), (int64_t)0);
  }

  CASE("a lane permutation stays in the warp");
  {
    // The destination wants what lane `l ^ 1` holds: one shuffle.
    std::vector<int64_t> src, dst;
    for (int64_t l = 0; l < (int64_t)kWarpSize; ++l) {
      src.push_back(l);
      dst.push_back(l ^ 1);
    }

    const ShufflePlan p = planShuffleFromElems({src}, {dst});
    CHECK(p.usable());
    CHECK(!p.isRebind());
    CHECK(p.uniformLanePerm);
    CHECK(p.linearLanePerm);
    CHECK_EQ(p.shuffleCount(), (int64_t)1);
  }

  CASE("an element in another warp cannot be shuffled to");
  {
    std::vector<int64_t> src, dst;
    for (int64_t l = 0; l < (int64_t)kWarpSize; ++l) {
      src.push_back(l);
      dst.push_back(l + (int64_t)kWarpSize);
    }

    const ShufflePlan p = planShuffleFromElems({src}, {dst});
    CHECK(!p.usable());
    CHECK(shuffleDecision(p).isDecline());
  }

  CASE("one destination register must read one source register");
  {
    // `simd_shuffle` shuffles one variable across the warp.
    std::vector<int64_t> r0, r1, mixed;
    for (int64_t l = 0; l < (int64_t)kWarpSize; ++l) {
      r0.push_back(l);
      r1.push_back(l + (int64_t)kWarpSize);
      mixed.push_back(l < 16 ? l : l + (int64_t)kWarpSize);
    }

    CHECK(!planShuffleFromElems({r0, r1}, {mixed}).usable());
    CHECK(planShuffleFromElems({r0, r1}, {r0}).usable());
    CHECK(planShuffleFromElems({r0, r1}, {r1}).usable());
  }

  CASE("an empty or ragged map is not a shuffle");
  {
    CHECK(!planShuffleFromElems({}, {}).usable());
    CHECK(!planShuffleFromElems({{0, 1}}, {{0, 1}}).usable());
  }

  return ::agpu_test::report("Shuffle");
}
