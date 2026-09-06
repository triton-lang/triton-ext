// The builtin name table.
#include "agpu/emit/EmitAtomic.h"
#include "agpu/emit/EmitBand.h"
#include "agpu/emit/EmitDirect.h"
#include "agpu/emit/EmitGridBuiltins.h"
#include "agpu/emit/EmitKernel.h"
#include "agpu/emit/EmitPanel.h"
#include "agpu/emit/EmitReduce.h"
#include "agpu/emit/EmitScan.h"
#include "agpu/emit/EmitShuffle.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/AtomicPlan.h"
#include "agpu/plan/CanonicalFragment.h"
#include "agpu/plan/DeviceFn.h"
#include "agpu/plan/Elementwise.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace agpu;
using agpu_test::has;
using agpu_test::render;
namespace bi = agpu::msl::builtin;

namespace {

std::string renderExpr(msl::Context &c, msl::Expr *e) {
  return render(c.exprStmt(e));
}

std::vector<const char *> allNames() {
  return {
      bi::math::Abs,
      bi::math::Floor,
      bi::math::Ceil,
      bi::accuracy::Sqrt.fast,
      bi::accuracy::Rsqrt.fast,
      bi::accuracy::Exp2.fast,
      bi::accuracy::Sin.precise,
      bi::accuracy::Cos.precise,
      bi::accuracy::Tanh.precise,
      bi::accuracy::Exp.precise,
      bi::accuracy::Log.precise,
      bi::accuracy::Log2.precise,
      bi::accuracy::Log10.precise,
      bi::accuracy::Exp10.precise,
      bi::accuracy::Pow.precise,
      bi::accuracy::Atan2.precise,
      bi::simd::Shuffle,
      bi::simd::ShuffleUp,
      bi::simd::ShuffleDown,
      bi::simd::ShuffleXor,
      bi::sg::Load,
      bi::sg::Store,
      bi::sg::MultiplyAccumulate,
      bi::atomic::Load,
      bi::atomic::Store,
      bi::atomic::Exchange,
      bi::atomic::ThreadFence,
      bi::atomic::FetchAdd,
      bi::atomic::FetchAnd,
      bi::atomic::FetchMax,
      bi::atomic::FetchMin,
      bi::atomic::FetchOr,
      bi::atomic::FetchXor,
      bi::atomic::CompareExchangeWeak,
      bi::order::Relaxed,
      bi::order::SeqCst,
      bi::memflags::Device,
      bi::memflags::Threadgroup,
      bi::comp::X,
      bi::comp::Y,
      bi::comp::Z,
  };
}

} // namespace

int main() {
  // ── the table itself ────────────────────────────────────────────────────

  CASE("every owned name is non-empty and spelled once");
  {
    std::set<std::string> seen;
    for (const char *n : allNames()) {
      CHECK(n != nullptr);
      CHECK(std::string(n) != std::string());
      CHECK(seen.insert(std::string(n)).second);
    }
  }

  CASE("a metal:: name carries its namespace and nothing else");
  {
    for (const char *n : {bi::math::Round, bi::math::Trunc, bi::math::Abs})
      CHECK(has(n, "metal::") && !has(n, "precise::"));
    for (const bi::AccuracyPair s :
         {bi::accuracy::Sin, bi::accuracy::Exp, bi::accuracy::Sqrt,
          bi::accuracy::Fmod, bi::accuracy::Rsqrt}) {
      CHECK(has(s.fast, "metal::") && !has(s.fast, "precise::"));
      CHECK(has(s.precise, "metal::precise::"));
    }
  }

  CASE("a builtin free function is unqualified");
  {
    // simd_*, simdgroup_*, atomic_* and the memory orders are global in MSL.
    for (const char *n : {bi::simd::Shuffle, bi::sg::Load, bi::atomic::FetchAdd,
                          bi::order::Relaxed})
      CHECK(!has(n, "metal::"));
  }

  CASE("a component name answers by axis and refuses a fourth");
  {
    CHECK_EQ(std::string(bi::comp::of(0)), std::string(bi::comp::X));
    CHECK_EQ(std::string(bi::comp::of(1)), std::string(bi::comp::Y));
    CHECK_EQ(std::string(bi::comp::of(2)), std::string(bi::comp::Z));
    CHECK(bi::comp::of(3) == nullptr);
    CHECK(bi::comp::of(-1) == nullptr);
  }

  // ── the plan tables name the same things ────────────────────────────────

  CASE("the RMW table names builtins from the atomic table");
  {
    const std::set<std::string> owned = {
        bi::atomic::FetchAdd, bi::atomic::FetchMax, bi::atomic::FetchMin,
        bi::atomic::FetchAnd, bi::atomic::FetchOr,  bi::atomic::FetchXor,
        bi::atomic::Exchange};
    for (const RmwBuiltin &b : kRmwBuiltins) {
      CHECK(b.fn != nullptr);
      CHECK(owned.count(std::string(b.fn)) == 1);
    }
  }

  CASE("every math name is owned, by the builtins or by the prelude");
  {
    // A name is owned if it comes from `metal::` (a builtin) or from the
    // prelude's `__agpu_` (a helper Metal has no counterpart for).
    auto owned = [](const char *n) {
      const std::string s(n);
      return s.rfind("metal::", 0) == 0 || s.rfind("__agpu_", 0) == 0;
    };
    for (const MathSpelling &s : kMathSpellings) {
      CHECK(s.name != nullptr);
      CHECK(owned(s.name));
    }
    for (const MathSpelling2 &s : kMathSpellings2) {
      CHECK(s.name != nullptr);
      CHECK(owned(s.name));
    }
    for (MathFn3 fn : {MathFn3::Fma, MathFn3::Clamp})
      CHECK(owned(mathNameOf(fn)));

    CHECK(mathNameOf(MathFn::Exp) == bi::accuracy::Exp.precise);
    CHECK(mathNameOf(MathFn2::Fmod) == bi::accuracy::Fmod.precise);
    CHECK(mathNameOf(MathFn::Erf) == bi::helper::Erf);
    CHECK(mathNameOf(MathFn2::Min) == bi::math::Min);
  }

  CASE("the transcendentals are precise and the default set is measured");
  {
    for (const MathFn fn :
         {MathFn::Sin, MathFn::Cos, MathFn::Tanh, MathFn::Exp, MathFn::Exp10,
          MathFn::Log, MathFn::Log2, MathFn::Log10, MathFn::Sqrt})
      CHECK(has(mathNameOf(fn), "precise::"));

    for (const MathFn2 fn : {MathFn2::Fmod, MathFn2::Pow, MathFn2::Atan2})
      CHECK(has(mathNameOf(fn), "precise::"));

    for (const MathFn fn : {MathFn::Exp2, MathFn::Rsqrt, MathFn::Abs,
                            MathFn::Floor, MathFn::Ceil})
      CHECK(!has(mathNameOf(fn), "precise::"));
  }

  CASE("the fragment type name is built from the table's affixes");
  {
    const std::string t = kSimdgroup8x8.mslType("half");
    const std::string d = std::to_string(kSgFragDim);
    CHECK_EQ(t, std::string(bi::sg::TypePrefix) + "half" + d + "x" + d);
    CHECK_EQ(kSimdgroup8x8.dim(), kSgFragDim);
  }

  // ── what the emitters actually write ────────────────────────────────────

  CASE("the shuffles the emitters write come from the table");
  {
    msl::Context c;
    ScanPlan fwd, rev;
    rev.reverse = true;
    CHECK(has(renderExpr(c, scanShuffle(c, fwd, "v", 1, f32())),
              bi::simd::ShuffleUp));
    CHECK(has(renderExpr(c, scanShuffle(c, rev, "v", 1, f32())),
              bi::simd::ShuffleDown));
    CHECK(has(renderExpr(c, shuffleXor(c, "v", 2)), bi::simd::ShuffleXor));
  }

  CASE("a type the hardware cannot shuffle travels as one that it can");
  {
    // `simd_shuffle` and friends are constrained to 32-bit scalars and
    // vectors of them; anything else is a compile error at the call.
    msl::Context c;

    const std::string wide =
        renderExpr(c, shuffleXor(c, "v", 2, ElemType{ElemType::Kind::Int, 64}));
    CHECK(has(wide, "uint2"));
    CHECK(has(wide, bi::simd::ShuffleXor));

    CHECK(has(renderExpr(c, shuffleXor(c, "v", 2, f16())), "ushort"));

    CHECK(!has(renderExpr(c, shuffleXor(c, "v", 2, f32())), "as_type"));
  }

  CASE("the fence names its flag and its order from the table");
  {
    msl::Context c;
    const std::string out = render(deviceFence(c));
    CHECK(has(out, bi::atomic::ThreadFence));
    CHECK(has(out, bi::memflags::Device));
    CHECK(has(out, bi::order::SeqCst));
  }

  CASE("a simdgroup load and store name the table's intrinsics");
  {
    msl::Context c;
    CHECK(has(render(sgLoad(c, "f", "buf", c.lit(0), 8)), bi::sg::Load));
    CHECK(has(render(sgStore(c, "f", "buf", c.lit(0), 8)), bi::sg::Store));
    CHECK(has(render(sgMma(c, "acc", "a", "b")), bi::sg::MultiplyAccumulate));
  }

  // ── the minted names have one owner too ─────────────────────────────────

  CASE("a grid query reads the names the kernel signature declares");
  {
    KernelNames kn;
    GridNames gn;
    CHECK_EQ(gn.of(GridQuery::ProgramId), kn.threadgroupPos);
    CHECK_EQ(gn.of(GridQuery::NumPrograms), kn.gridSize);
  }

  CASE("renaming the grid parameters renames what the query reads");
  {
    msl::Context c;
    GridNames gn;
    gn.threadgroupPos = "__tgpos";
    gn.gridSize = "__numtg";
    CHECK(has(render(emitGridQuery(c, GridQuery::ProgramId, 0, "p", gn)),
              "__tgpos.x"));
    CHECK(has(render(emitGridQuery(c, GridQuery::NumPrograms, 1, "n", gn)),
              "__numtg.y"));
  }

  CASE("the thread names cannot disagree, because there is one declaration");
  {
    KernelNames kn;
    CHECK_EQ(static_cast<const ThreadNames &>(AtomicNames{}).laneId, kn.laneId);
    CHECK_EQ(static_cast<const ThreadNames &>(ReduceNames{}).warpId, kn.warpId);
    CHECK_EQ(static_cast<const ThreadNames &>(ScanNames{}).laneId, kn.laneId);
    ReduceNames rn;
    static_cast<ThreadNames &>(rn).laneId = "L";
    CHECK_EQ(rn.laneId, std::string("L"));
    CHECK(rn.laneId != kn.laneId);

    CHECK_EQ(static_cast<const ThreadNames &>(BandNames{}).laneId, kn.laneId);
    CHECK_EQ(static_cast<const ThreadNames &>(DirectNames{}).warpId, kn.warpId);
  }

  CASE("a device function spells the thread context the way a kernel does");
  {
    const KernelNames kn;
    const DeviceFnNames dn;
    CHECK_EQ(static_cast<const ThreadNames &>(dn).threadId, kn.threadId);
    CHECK_EQ(static_cast<const ThreadNames &>(dn).laneId, kn.laneId);
    CHECK_EQ(static_cast<const ThreadNames &>(dn).warpId, kn.warpId);
    CHECK_EQ(dn.threadgroupPos, kn.threadgroupPos);
    CHECK_EQ(dn.gridSize, kn.gridSize);
    CHECK_EQ(dn.pool, kn.pool);
  }

  CASE("names two emitters must agree on are inherited verbatim");
  {
    // EmitDirect calls into EmitPanel, so poolC, acc and kVar must agree at
    // runtime. A differing default reads an undeclared variable.
    DirectNames d;
    PanelNames p;
    CHECK_EQ(d.poolC, p.poolC);
    CHECK_EQ(d.acc, p.acc);
    CHECK_EQ(d.kVar, p.kVar);
    CHECK_EQ(d.accElem, p.accElem);
    CHECK_EQ(d.opElem, p.opElem);

    static_cast<MmaNames &>(d).poolC = "POOL";
    CHECK_EQ(d.poolC, std::string("POOL"));
    CHECK(d.poolC != p.poolC);
  }

  CASE("the reduce and scan scratch is one type, resolved per operand");
  {
    CHECK(ReduceNames{}.scratch.empty());
    CHECK(ScanNames{}.scratch.empty());
  }

  CASE("no two names in one scope mean different things");
  {
    ScanNames s;
    BandNames b;
    ReduceNames r;
    CHECK(s.carry != b.buffer);
    CHECK(s.acc != b.flat);
    CHECK(s.peer != b.flat);
    CHECK(r.acc != b.buffer);
    CHECK(r.peer != b.buffer);
  }

  return ::agpu_test::report("Builtins");
}
