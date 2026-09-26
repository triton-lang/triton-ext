// The helper definitions a kernel needs before it can link.
#include "agpu/emit/EmitConvert.h"
#include "agpu/emit/PrintModule.h"
#include "harness.h"

#include <sstream>

using namespace agpu;

namespace {

std::string prelude(const HelperSet &h) {
  std::ostringstream os;
  printPrelude(os, h);
  return os.str();
}

AtomicFacts atomicOf(RmwOp op, ElemClass elem, unsigned bits) {
  AtomicFacts f;
  f.op = op;
  f.elem = elem;
  f.bits = bits;
  return f;
}

} // namespace

int main() {
  CASE("every module opens with the includes, helpers or not");
  {
    HelperSet h;
    CHECK(!h.any());
    const std::string out = prelude(h);
    CHECK_HAS(out, "#include <metal_stdlib>");
    CHECK_HAS(out, "#include <metal_simdgroup_matrix>");
    CHECK_HAS(out, "using namespace metal;");
  }

  CASE("a kernel needing no helpers gets the header and nothing else");
  {
    HelperSet h;
    std::ostringstream os;
    printModuleHeader(os);
    CHECK_EQ(prelude(h), os.str());
  }

  CASE("a second kernel in one file does not repeat the header");
  {
    HelperSet h;
    std::ostringstream os;
    printPrelude(os, h, /*header=*/false);
    CHECK_EQ(os.str(), std::string(""));
  }

  CASE("a native atomic asks for no helper");
  {
    HelperSet h;
    h.require(planAtomic(atomicOf(RmwOp::Add, ElemClass::Int, 32),
                         MemOrder::Relaxed));
    CHECK(!h.any());
  }

  CASE("a float CAS atomic asks for its helper");
  {
    HelperSet h;
    h.require(planAtomic(atomicOf(RmwOp::Max, ElemClass::Float, 32),
                         MemOrder::Relaxed));
    CHECK(h.has(Helper::AtomicRmwF32));
    CHECK(!h.has(Helper::AtomicRmwPacked16));
    CHECK_HAS(prelude(h), "__agpu_atomic_rmw_f32");
  }

  CASE("a 16-bit float atomic asks for the packed helper");
  {
    HelperSet h;
    h.require(planAtomic(atomicOf(RmwOp::Add, ElemClass::Float, 16),
                         MemOrder::Relaxed));
    CHECK(h.has(Helper::AtomicRmwPacked16));
    CHECK(!h.has(Helper::AtomicRmwF32));
    CHECK_HAS(prelude(h), "__agpu_atomic_rmw_packed16");
  }

  CASE("the RMW selectors in the helper body come from the enum");
  {
    HelperSet h;
    h.add(Helper::AtomicRmwF32);
    h.add(Helper::AtomicRmwPacked16);
    const std::string out = prelude(h);

    CHECK(out.find("op == " + std::to_string(emuRmwCode(EmuRmw::Add)) +
                   " ? old + v") != std::string::npos);
    CHECK(out.find("op == " + std::to_string(emuRmwCode(EmuRmw::Max)) +
                   " ? metal::max(old, v)") != std::string::npos);
    CHECK(out.find("op == " + std::to_string(emuRmwCode(EmuRmw::Min)) +
                   " ? metal::min(old, v)") != std::string::npos);

    CHECK(out.find("op == " + std::to_string(emuRmwCode(EmuRmw::Max)) +
                   " ? metal::max(cur, v)") != std::string::npos);
  }

  CASE("the packed atomic narrows through a dedicated helper");
  {
    HelperSet h;
    h.require(planAtomic(atomicOf(RmwOp::Add, ElemClass::Float, 16),
                         MemOrder::Relaxed));
    const std::string out = prelude(h);
    CHECK_HAS(out, "__agpu_narrow16<T>(next)");
    CHECK(out.find("as_type<ushort>(T(next))") == std::string::npos);

    CHECK(h.has(Helper::RtneIntHalf));
    CHECK(h.has(Helper::RtneIntBfloat));
    CHECK_HAS(out, "inline ushort __agpu_rtne_int_half");
    CHECK_HAS(out, "inline ushort __agpu_rtne_int_bfloat");
    CHECK(out.find("__agpu_rtne_int_half") <
          out.find("__agpu_atomic_rmw_packed16"));
  }

  CASE("an unsupported atomic asks for nothing");
  {
    HelperSet h;
    h.require(planAtomic(atomicOf(RmwOp::Add, ElemClass::Int, 64),
                         MemOrder::Relaxed));
    CHECK(!h.any());
  }

  CASE("erf uses its own tighter fit");
  {
    HelperSet h;
    h.add(Helper::Erf);
    const std::string out = prelude(h);
    CHECK_HAS(out, "0.3275911f");
    CHECK_LACKS(out, "1.26551223f");
    CHECK_HAS(out, "metal::sign(x)");
  }

  CASE("erf is a helper because Metal has none");
  {
    HelperSet h;
    h.require(MathFn::Erf);
    CHECK(h.has(Helper::Erf));
    CHECK_HAS(prelude(h), "__agpu_erf");

    HelperSet other;
    other.require(MathFn::Exp);
    CHECK(!other.any());
  }

  CASE("fp8 packs round to nearest-even");
  {
    HelperSet h;
    h.add(Helper::Fp8PackE4M3);
    const std::string out = prelude(h);
    CHECK_HAS(out, "rem > 0x80000u || rem == 0x80000u && m & 1u");
  }

  CASE("fp8 encodes subnormals");
  {
    HelperSet h;
    h.add(Helper::Fp8PackE4M3);
    h.add(Helper::Fp8UnpackE4M3);
    const std::string out = prelude(h);
    CHECK_HAS(out, "if (ex < -6)");
    CHECK(out.find("while ((m & 0x8u) == 0u)") != std::string::npos);
  }

  CASE("e4m3 saturates below its NaN slot, e5m2 to its infinity");
  {
    HelperSet h;
    h.add(Helper::Fp8PackE4M3);
    h.add(Helper::Fp8PackE5M2);
    const std::string out = prelude(h);
    CHECK_HAS(out, "ex >= 16 || ex == 15 && mant > 0x600000u");
    CHECK_HAS(out, "sgn | 0x7eu");
    CHECK_HAS(out, "sgn | 0x7cu");
  }

  CASE("round-toward-zero saturates to the largest finite half");
  {
    HelperSet h;
    h.add(Helper::RtzHalf);
    const std::string out = prelude(h);
    CHECK_HAS(out, "sgn | 0x7bffu");
    CHECK_HAS(out, "(mant ? 0x200u : 0u)");
  }

  CASE("round-to-nearest-even saturates to infinity, unlike toward-zero");
  {
    HelperSet rtne, rtz;
    rtne.add(Helper::RtneIntHalf);
    rtz.add(Helper::RtzHalf);
    CHECK_HAS(prelude(rtne), "sgn | 0x7c00u");
    CHECK_HAS(prelude(rtz), "sgn | 0x7bffu");
  }

  CASE("bfloat's nearest-even rounds before truncating");
  {
    HelperSet ne, z;
    ne.add(Helper::RtneIntBfloat);
    z.add(Helper::RtzBfloat);
    const std::string neSrc = prelude(ne);
    CHECK_HAS(neSrc, "u + 0x7fffu + lsb");
    // The NaN arm quiets the payload, through a select.
    CHECK_HAS(neSrc, "0x40u : 0u");
    CHECK(neSrc.find("if (") == std::string::npos);
    CHECK_LACKS(prelude(z), "0x7fffu");
  }

  CASE("asking twice emits one definition");
  {
    HelperSet h;
    for (int i = 0; i < 3; ++i)
      h.require(planAtomic(atomicOf(RmwOp::Max, ElemClass::Float, 32),
                           MemOrder::Relaxed));
    const std::string out = prelude(h);
    std::size_t n = 0;
    for (std::size_t i = out.find("inline float __agpu_atomic_rmw_f32");
         i != std::string::npos;
         i = out.find("inline float __agpu_atomic_rmw_f32", i + 1))
      ++n;
    CHECK_EQ(n, 1u);
  }

  CASE("several helpers emit in a fixed order");
  {
    HelperSet h;
    h.add(Helper::AtomicRmwF32);
    h.add(Helper::AtomicRmwPacked16);
    h.add(Helper::Erf);
    CHECK_EQ(prelude(h), prelude(h));
    const std::string out = prelude(h);
    CHECK(out.find("__agpu_atomic_rmw_f32") < out.find("__agpu_erf"));
    CHECK(out.find("__agpu_erf") < out.find("__agpu_atomic_rmw_packed16"));

    HelperSet reversed;
    reversed.add(Helper::Erf);
    reversed.add(Helper::AtomicRmwPacked16);
    reversed.add(Helper::AtomicRmwF32);
    CHECK_EQ(prelude(reversed), out);
  }

  CASE("every helper that can be named has a body that defines it");
  {
    for (unsigned i = 0; i < unsigned(Helper::Count); ++i) {
      const Helper which = Helper(i);
      const char *name = helperName(which);
      const std::string src = helperSource(which);
      CHECK(name != nullptr && *name != '\0');
      CHECK(!src.empty());
      CHECK_HAS(src, name);

      HelperSet only;
      only.add(which);
      CHECK_HAS(prelude(only), name);
    }
  }

  CASE("every conversion needing a helper names one the prelude defines");
  {
    const ElemType narrow[] = {f16(), bf16(), e4m3(), e5m2()};
    const Rounding modes[] = {Rounding::Default, Rounding::RTNE, Rounding::RTZ};
    for (ElemType to : narrow)
      for (Rounding r : modes) {
        for (ConvertPlan p :
             {planConvert(f32(), to, r), planConvert(to, f32(), r)}) {
          if (!p.usable() || !p.needsHelper())
            continue;
          HelperSet h;
          h.require(p);
          CHECK(h.any());
          const std::string name = convertHelperName(p);
          CHECK(!name.empty());
          CHECK_HAS(prelude(h), name);
        }
      }
  }

  CASE("erf's table entry names the prelude helper");
  {
    CHECK_EQ(std::string(mathNameOf(MathFn::Erf)),
             std::string(msl::builtin::helper::Erf));
    CHECK(checkMath(MathFn::Erf, f32()).ok());
    CHECK(checkMath(MathFn::Erf, i32()).isDecline());
  }

  CASE("cbrt lowers to the prelude, because Metal has none in either space");
  {
    CHECK_EQ(std::string(mathNameOf(MathFn::Cbrt)),
             std::string(msl::builtin::helper::Cbrt));
    CHECK(checkMath(MathFn::Cbrt, f32()).ok());
    CHECK(checkMath(MathFn::Cbrt, i32()).isDecline());
  }

  CASE("every helper-backed math function has its body emitted");
  {
    for (const MathSpelling &s : kMathSpellings) {
      const bool spelledAsHelper = std::string(s.name).rfind("__agpu_", 0) == 0;
      Helper h;
      CHECK_EQ(spelledAsHelper, mathHelper(s.fn, h));
      if (!spelledAsHelper)
        continue;
      CHECK_EQ(std::string(s.name), std::string(helperName(h)));
      const std::string body = helperSource(h);
      CHECK_HAS(body, helperName(h));
    }
  }

  CASE("requiring a helper-backed function pulls its body in");
  {
    HelperSet hs;
    CHECK(!hs.has(Helper::Cbrt));
    hs.require(MathFn::Cbrt);
    CHECK(hs.has(Helper::Cbrt));
    HelperSet plain;
    plain.require(MathFn::Sqrt);
    CHECK(!plain.has(Helper::Cbrt));
    CHECK(!plain.has(Helper::Erf));
  }

  CASE("an f64 argument is narrowed from its two words, never loaded as one");
  {
    const std::string body = helperSource(Helper::NarrowF64);
    CHECK_HAS(body, "uint hi");
    CHECK_HAS(body, "uint lo");
    CHECK_LACKS(body, "double");
    // 1023 - 127 is the whole of the conversion: re-bias, then round.
    CHECK_HAS(body, "1023");
    CHECK_HAS(body, "127");
  }

  CASE("an f32 fma pulls in the soft path it falls back to");
  {
    HelperSet hs;
    hs.require(MathFn3::Fma, f32());
    CHECK(hs.has(Helper::Fma));
    CHECK(hs.has(Helper::SoftFma));

    HelperSet other;
    other.require(MathFn3::Clamp, f32());
    CHECK(!other.has(Helper::Fma));
    other.require(MathFn3::Fma, f16());
    CHECK(!other.has(Helper::Fma));
  }

  CASE("the soft path is defined before the fma that calls it");
  {
    CHECK((unsigned)Helper::SoftFma < (unsigned)Helper::Fma);
  }

  CASE("a subnormal operand or result leaves the hardware fma");
  {
    const std::string body = helperSource(Helper::Fma);
    CHECK_HAS(body, "metal::fma");
    CHECK_HAS(body, helperName(Helper::SoftFma));
    // Operands and result each get their own test: a flushed operand can
    // leave a result that looks perfectly normal.
    std::size_t tests = 0;
    for (std::size_t at = body.find("- 1u < 0x7fffffu");
         at != std::string::npos; at = body.find("- 1u < 0x7fffffu", at + 1))
      ++tests;
    CHECK_EQ(tests, 3u);
    CHECK_HAS(body, "r == 0.0f");
    CHECK_HAS(body, "__builtin_expect(bad, 0)");
  }

  CASE("a zero operand keeps the hardware fma, which is exact there");
  {
    const std::string body = helperSource(Helper::Fma);
    CHECK_LACKS(body, "0x7f800000u) == 0u");
    CHECK_HAS(body, "< 174u");
    CHECK_HAS(body, "mc >> 23 < 24u");
  }

  CASE("the soft path runs the hardware fma only where nothing is subnormal");
  {
    const std::string body = helperSource(Helper::SoftFma);
    CHECK_HAS(body, "metal::fma(fa, fb, g)");
    CHECK_HAS(body, "0x3f800000u");
    CHECK_LACKS(body, "ulong");
  }

  return ::agpu_test::report("Prelude");
}
