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
    CHECK(out.find("#include <metal_stdlib>") != std::string::npos);
    CHECK(out.find("#include <metal_simdgroup_matrix>") != std::string::npos);
    CHECK(out.find("using namespace metal;") != std::string::npos);
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
    CHECK(prelude(h).find("__agpu_atomic_rmw_f32") != std::string::npos);
  }

  CASE("a 16-bit float atomic asks for the packed helper");
  {
    HelperSet h;
    h.require(planAtomic(atomicOf(RmwOp::Add, ElemClass::Float, 16),
                         MemOrder::Relaxed));
    CHECK(h.has(Helper::AtomicRmwPacked16));
    CHECK(!h.has(Helper::AtomicRmwF32));
    CHECK(prelude(h).find("__agpu_atomic_rmw_packed16") != std::string::npos);
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
    CHECK_EQ(emuRmwLadder("old"), [&] {
      std::string s = emuRmwLadder("cur");
      for (std::size_t i = s.find("cur"); i != std::string::npos;
           i = s.find("cur", i + 3))
        s.replace(i, 3, "old");
      return s;
    }());
  }

  CASE("the packed atomic narrows through a dedicated helper");
  {
    HelperSet h;
    h.require(planAtomic(atomicOf(RmwOp::Add, ElemClass::Float, 16),
                         MemOrder::Relaxed));
    const std::string out = prelude(h);
    CHECK(out.find("__agpu_narrow16<T>(next)") != std::string::npos);
    CHECK(out.find("as_type<ushort>(T(next))") == std::string::npos);

    CHECK(h.has(Helper::RtneIntHalf));
    CHECK(h.has(Helper::RtneIntBfloat));
    CHECK(out.find("inline ushort __agpu_rtne_int_half") != std::string::npos);
    CHECK(out.find("inline ushort __agpu_rtne_int_bfloat") !=
          std::string::npos);
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
    CHECK(out.find("0.3275911f") != std::string::npos);
    CHECK(out.find("1.26551223f") == std::string::npos);
    CHECK(out.find("metal::sign(x)") != std::string::npos);
  }

  CASE("erf is a helper because Metal has none");
  {
    HelperSet h;
    h.require(MathFn::Erf);
    CHECK(h.has(Helper::Erf));
    CHECK(prelude(h).find("__agpu_erf") != std::string::npos);

    HelperSet other;
    other.require(MathFn::Exp);
    CHECK(!other.any());
  }

  CASE("fp8 packs round to nearest-even");
  {
    HelperSet h;
    h.add(Helper::Fp8PackE4M3);
    const std::string out = prelude(h);
    CHECK(out.find("rem > 0x80000u || rem == 0x80000u && m & 1u") !=
          std::string::npos);
  }

  CASE("fp8 encodes subnormals");
  {
    HelperSet h;
    h.add(Helper::Fp8PackE4M3);
    h.add(Helper::Fp8UnpackE4M3);
    const std::string out = prelude(h);
    CHECK(out.find("if (ex < -6)") != std::string::npos);
    CHECK(out.find("while ((m & 0x8u) == 0u)") != std::string::npos);
  }

  CASE("e4m3 saturates below its NaN slot, e5m2 to its infinity");
  {
    HelperSet h;
    h.add(Helper::Fp8PackE4M3);
    h.add(Helper::Fp8PackE5M2);
    const std::string out = prelude(h);
    CHECK(out.find("ex >= 16 || ex == 15 && mant > 0x600000u") !=
          std::string::npos);
    CHECK(out.find("sgn | 0x7eu") != std::string::npos);
    CHECK(out.find("sgn | 0x7cu") != std::string::npos);
  }

  CASE("round-toward-zero saturates to the largest finite half");
  {
    HelperSet h;
    h.add(Helper::RtzHalf);
    const std::string out = prelude(h);
    CHECK(out.find("sgn | 0x7bffu") != std::string::npos);
    CHECK(out.find("(mant ? 0x200u : 0u)") != std::string::npos);
  }

  CASE("round-to-nearest-even saturates to infinity, unlike toward-zero");
  {
    HelperSet rtne, rtz;
    rtne.add(Helper::RtneIntHalf);
    rtz.add(Helper::RtzHalf);
    CHECK(prelude(rtne).find("sgn | 0x7c00u") != std::string::npos);
    CHECK(prelude(rtz).find("sgn | 0x7bffu") != std::string::npos);
  }

  CASE("bfloat's nearest-even rounds before truncating");
  {
    HelperSet ne, z;
    ne.add(Helper::RtneIntBfloat);
    z.add(Helper::RtzBfloat);
    const std::string neSrc = prelude(ne);
    CHECK(neSrc.find("u + 0x7fffu + lsb") != std::string::npos);
    CHECK(neSrc.find("(mant ? 0x40u : 0u)") != std::string::npos);
    CHECK(prelude(z).find("0x7fffu") == std::string::npos);
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
      CHECK(src.find(name) != std::string::npos);

      HelperSet only;
      only.add(which);
      CHECK(prelude(only).find(name) != std::string::npos);
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
          CHECK(prelude(h).find(name) != std::string::npos);
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
      CHECK(body.find(helperName(h)) != std::string::npos);
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

  return ::agpu_test::report("Prelude");
}
