// Type conversion: what rounds and what has no MSL type.
#include "agpu/emit/EmitConvert.h"
#include "agpu/emit/PrintModule.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

int main() {
  // ── the ordinary case ──────────────────────────────────────────────────

  CASE("a same-type conversion emits nothing at all");
  {
    ConvertPlan p = planConvert(f32(), f32(), Rounding::Default);
    CHECK(p.kind == ConvertKind::None);

    msl::Context c;
    msl::Block body;
    auto out = emitConvert(c, body, p, {"v0", "v1"}, {"d0", "d1"}, f32());
    CHECK_EQ(body.size(), 0u);
    CHECK_EQ(out[0], std::string("v0"));
  }

  CASE("a widening conversion is a plain cast");
  {
    ConvertPlan p = planConvert(f16(), f32(), Rounding::Default);
    CHECK(p.kind == ConvertKind::Cast);
    CHECK(!p.needsHelper());

    msl::Context c;
    msl::Block body;
    emitConvert(c, body, p, {"v0"}, {"d0"}, f32());
    CHECK(render(body).find("(float)v0") != std::string::npos);
  }

  CASE("an int-to-float conversion is a cast");
  {
    CHECK(planConvert(i32(), f32(), Rounding::Default).kind ==
          ConvertKind::Cast);
    CHECK(planConvert(f32(), i32(), Rounding::Default).kind ==
          ConvertKind::Cast);
  }

  // ── rounding ───────────────────────────────────────────────────────────

  CASE("an unrequested rounding mode uses the cast");
  {
    CHECK(planConvert(f32(), f16(), Rounding::Default).kind ==
          ConvertKind::Cast);
  }

  CASE("nearest-even narrowing needs a helper too, despite the cast");
  {
    // The cast is nearest-even under Metal's fast-math float model, but that
    // model assumes no NaN/Inf and may flush subnormals.
    ConvertPlan p = planConvert(f32(), f16(), Rounding::RTNE);
    CHECK(p.kind == ConvertKind::NarrowRtne);
    CHECK(p.needsHelper());
    CHECK_EQ(convertHelperName(p), std::string("__agpu_rtne_half"));

    CHECK_EQ(convertHelperName(planConvert(f32(), bf16(), Rounding::RTNE)),
             std::string("__agpu_rtne_bfloat"));
  }

  CASE("round-toward-zero narrowing needs a helper");
  {
    // MSL offers no way to ask for a rounding mode, so RTZ is a bit operation
    // on the mantissa.
    ConvertPlan p = planConvert(f32(), f16(), Rounding::RTZ);
    CHECK(p.kind == ConvertKind::NarrowRtz);
    CHECK(p.needsHelper());

    msl::Context c;
    msl::Block body;
    emitConvert(c, body, p, {"v0"}, {"d0"}, f16());
    CHECK(render(body).find("__agpu_rtz_half(v0)") != std::string::npos);
  }

  CASE("rounding only matters when the conversion narrows");
  {
    CHECK(planConvert(f16(), f32(), Rounding::RTZ).kind == ConvertKind::Cast);
    CHECK(!narrowsFloat(f16(), f32()));
    CHECK(narrowsFloat(f32(), f16()));
  }

  CASE("bfloat is its own distinct type");
  {
    // bf16 and half are both 16-bit floats with different mantissas.
    CHECK(!(bf16() == f16()));
    CHECK(mslTypeOf(bf16()).scalarKind() == msl::Scalar::BF16);
    CHECK(mslTypeOf(f16()).scalarKind() == msl::Scalar::F16);

    CHECK_EQ(convertHelperName(planConvert(f32(), bf16(), Rounding::RTZ)),
             std::string("__agpu_rtz_bfloat"));
    CHECK_EQ(convertHelperName(planConvert(f32(), f16(), Rounding::RTZ)),
             std::string("__agpu_rtz_half"));
  }

  CASE("signedness and float variant are separate states");
  {
    CHECK(fp8KindOf(e5m2()) == Fp8Kind::E5M2);
    CHECK(fp8KindOf(e4m3()) == Fp8Kind::E4M3);
    ElemType u32{ElemType::Kind::Int, 32, true};
    CHECK(fp8KindOf(u32) == Fp8Kind::None);
    CHECK(mslTypeOf(u32).scalarKind() == msl::Scalar::U32);
  }

  // ── fp8 ────────────────────────────────────────────────────────────────

  CASE("fp8 has no MSL type, so every conversion is a pack or unpack");
  {
    ConvertPlan pack = planConvert(f32(), e4m3(), Rounding::Default);
    CHECK(pack.kind == ConvertKind::Fp8Pack);
    CHECK(pack.fp8 == Fp8Kind::E4M3);

    ConvertPlan unpack = planConvert(e4m3(), f32(), Rounding::Default);
    CHECK(unpack.kind == ConvertKind::Fp8Unpack);
  }

  CASE("the two fp8 encodings are distinct");
  {
    // e4m3 has 4 exponent bits with bias 7, e5m2 has 5 with bias 15. Using
    // one encoding's helper for the other scales every value by a power of two.
    CHECK(fp8KindOf(e4m3()) == Fp8Kind::E4M3);
    CHECK(fp8KindOf(e5m2()) == Fp8Kind::E5M2);
    CHECK(fp8KindOf(f32()) == Fp8Kind::None);

    msl::Context c;
    msl::Block a, b;
    emitConvert(c, a, planConvert(f32(), e4m3(), Rounding::Default), {"v"},
                {"d"}, e4m3());
    emitConvert(c, b, planConvert(f32(), e5m2(), Rounding::Default), {"v"},
                {"d"}, e5m2());
    CHECK(render(a).find("__agpu_f32_to_e4m3") != std::string::npos);
    CHECK(render(b).find("__agpu_f32_to_e5m2") != std::string::npos);
  }

  CASE("the bias variants are their own distinct encodings");
  {
    // e4b8 has e4m3's mantissa split and a bias of 8; e5b16 has e5m2's and a
    // bias of 16.
    CHECK(fp8KindOf(e4b8()) == Fp8Kind::E4B8);
    CHECK(fp8KindOf(e5b16()) == Fp8Kind::E5B16);
    CHECK(!(e4b8() == e4m3()));
    CHECK(!(e5b16() == e5m2()));

    msl::Context c;
    msl::Block a, b;
    emitConvert(c, a, planConvert(f32(), e4b8(), Rounding::Default), {"v"},
                {"d"}, e4b8());
    emitConvert(c, b, planConvert(e5b16(), f32(), Rounding::Default), {"v"},
                {"d"}, f32());
    CHECK(render(a).find("__agpu_f32_to_e4b8") != std::string::npos);
    CHECK(render(b).find("__agpu_e5b16_to_f32") != std::string::npos);
  }

  CASE("each fp8 encoding pulls in its own helper and no other");
  {
    HelperSet h;
    h.require(planConvert(f32(), e4b8(), Rounding::Default));
    CHECK(h.has(Helper::Fp8PackE4B8));
    CHECK(!h.has(Helper::Fp8PackE4M3));
    CHECK(!h.has(Helper::Fp8UnpackE4B8));

    std::ostringstream os;
    printPrelude(os, h);
    const std::string out = os.str();
    CHECK(out.find("__agpu_f32_to_e4b8") != std::string::npos);
    CHECK(out.find("e4m3") == std::string::npos);

    HelperSet h2;
    h2.require(planConvert(e5b16(), f32(), Rounding::Default));
    CHECK(h2.has(Helper::Fp8UnpackE5B16));
    CHECK(!h2.has(Helper::Fp8UnpackE5M2));
  }

  CASE("the fp4 table names every value it can be asked for");
  {
    // e2m1 is four bits: there is no inf and no NaN and 0x7 is 6.0 rather
    // than an infinity.
    HelperSet h;
    h.add(Helper::Fp4Unpack);
    std::ostringstream os;
    printPrelude(os, h);
    const std::string out = os.str();
    CHECK(out.find("__agpu_e2m1_to_f32") != std::string::npos);
    CHECK(out.find("6.0f") != std::string::npos);
    CHECK(out.find("-6.0f") != std::string::npos);
    CHECK(out.find("0.5f") != std::string::npos);
    CHECK(out.find("nib & 0xfu") != std::string::npos);

    HelperSet other;
    other.require(planConvert(f32(), e4m3(), Rounding::Default));
    CHECK(!other.has(Helper::Fp4Unpack));
  }

  CASE("the bias variants reach narrow floats the same way");
  {
    const ConvertPlan pack = planConvert(f16(), e5b16(), Rounding::Default);
    CHECK(pack.kind == ConvertKind::Fp8Pack);
    CHECK(pack.widensOperand);
    CHECK(planConvert(e4b8(), bf16(), Rounding::Default).kind ==
          ConvertKind::Fp8Unpack);
  }

  CASE("fp8 to fp8 declines, with no intermediate guessed at");
  {
    ConvertPlan p = planConvert(e4m3(), e5m2(), Rounding::Default);
    CHECK(!p.usable());
    Decision d = convertDecision(p);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
  }

  CASE("fp8 reaches a narrow float through f32, exactly");
  {
    // The helpers convert to and from f32 only. f16 and bf16 are subsets of
    // f32, so the widen and narrow around them add no rounding.
    const ConvertPlan pack = planConvert(f16(), e4m3(), Rounding::Default);
    CHECK(pack.usable());
    CHECK(pack.kind == ConvertKind::Fp8Pack);
    CHECK(pack.widensOperand);

    const ConvertPlan unpack = planConvert(e4m3(), f16(), Rounding::Default);
    CHECK(unpack.usable());
    CHECK(unpack.kind == ConvertKind::Fp8Unpack);

    CHECK(!planConvert(f32(), e4m3(), Rounding::Default).widensOperand);
  }

  CASE("fp8 to a non-float still declines");
  {
    CHECK(!planConvert(e4m3(), i32(), Rounding::Default).usable());
    CHECK(!planConvert(i32(), e4m3(), Rounding::Default).usable());
  }

  // ── the prelude carries exactly what is used ───────────────────────────

  CASE("a conversion asks for its own helper and no other");
  {
    HelperSet h;
    h.require(planConvert(f32(), e4m3(), Rounding::Default));
    CHECK(h.has(Helper::Fp8PackE4M3));
    CHECK(!h.has(Helper::Fp8PackE5M2));
    CHECK(!h.has(Helper::Fp8UnpackE4M3));
    CHECK(!h.has(Helper::RtzHalf));

    std::ostringstream os;
    printPrelude(os, h);
    const std::string out = os.str();
    CHECK(out.find("__agpu_f32_to_e4m3") != std::string::npos);
    CHECK(out.find("e5m2") == std::string::npos);
  }

  CASE("half and bfloat ask for different narrowing helpers");
  {
    HelperSet hh;
    hh.require(planConvert(f32(), f16(), Rounding::RTZ));
    CHECK(hh.has(Helper::RtzHalf));
    CHECK(!hh.has(Helper::RtzBfloat));

    HelperSet hb;
    hb.require(planConvert(f32(), bf16(), Rounding::RTZ));
    CHECK(hb.has(Helper::RtzBfloat));
    CHECK(!hb.has(Helper::RtzHalf));

    std::ostringstream os;
    printPrelude(os, hb);
    const std::string out = os.str();
    CHECK(out.find("__agpu_rtz_bfloat") != std::string::npos);
    CHECK(out.find("__agpu_rtz_half") == std::string::npos);
  }

  CASE("a cast asks for nothing");
  {
    HelperSet h;
    h.require(planConvert(f16(), f32(), Rounding::Default));
    h.require(planConvert(f32(), f16(), Rounding::Default));
    h.require(planConvert(i32(), f32(), Rounding::Default));
    CHECK(!h.any());
  }

  CASE("the helper a plan names is the one the prelude defines");
  {
    HelperSet h;
    ConvertPlan p = planConvert(f32(), f16(), Rounding::RTZ);
    h.require(p);
    std::ostringstream os;
    printPrelude(os, h);
    CHECK(os.str().find(convertHelperName(p)) != std::string::npos);
  }

  return ::agpu_test::report("Convert");
}
