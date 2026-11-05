// The small op families: histogram, grid builtins and gather.
#include "agpu/emit/EmitGather.h"
#include "agpu/emit/EmitGridBuiltins.h"
#include "agpu/emit/EmitHistogram.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

msl::SmallVec<msl::Str, 8> regs(std::initializer_list<const char *> names) {
  msl::SmallVec<msl::Str, 8> v;
  for (const char *n : names)
    v.push_back(n);
  return v;
}

} // namespace

int main() {
  CASE("zeroing splits the strides evenly across the thread count");
  {
    msl::Context c;
    msl::Block body;
    HistogramNames nm;
    HistogramPlan p = planHistogram(/*bins=*/256, /*numWarps=*/4, 0, 0);
    CHECK_EQ(p.threads, 128);
    CHECK_EQ(p.zeroSteps(), 2);
    emitHistogramZero(c, body, p, nm);
    const std::string out = render(body);
    CHECK_HAS(out, "int zi = tid.x");
    CHECK_HAS(out, "zi < 256");
    CHECK_HAS(out, "zi += 128");
    CHECK(out.find("atomic_store_explicit(&bins[zi], 0") != std::string::npos);
  }

  CASE("fewer bins than threads still zeroes each exactly once");
  {
    HistogramPlan p = planHistogram(/*bins=*/32, /*numWarps=*/4, 0, 0);
    CHECK_EQ(p.zeroSteps(), 1);
  }

  CASE("the zeroing is barriered before any counting");
  {
    msl::Context c;
    msl::Block body;
    HistogramNames nm;
    emitHistogramZero(c, body, planHistogram(64, 1, 0, 0), nm);
    CHECK_EQ(countOf(render(body), "threadgroup_barrier"), 1);
  }

  CASE("counting increments a bin per source register");
  {
    msl::Context c;
    msl::Block body;
    HistogramNames nm;
    emitHistogramCount(c, body, planHistogram(64, 1, 0, 0), regs({"v0", "v1"}),
                       nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "atomic_fetch_add"), 2);
    CHECK_HAS(out, "&bins[v0]");
  }

  CASE("a source value outside the bins is not counted");
  {
    msl::Context c;
    msl::Block body;
    HistogramNames nm;
    emitHistogramCount(c, body, planHistogram(64, 1, 0, 0), regs({"v0"}), nm);
    CHECK_HAS(render(body), "v0 < 64");
  }

  CASE("a masked histogram counts only what the mask admits");
  {
    msl::Context c;
    msl::Block body;
    HistogramNames nm;
    emitHistogramCount(c, body, planHistogram(64, 1, 0, 0), regs({"v0"}), nm,
                       regs({"m0"}));
    CHECK_HAS(render(body), "m0");
  }

  CASE("a free lane bit elects one owner per element");
  {
    msl::Context c;
    msl::Block body;
    HistogramNames nm;
    HistogramPlan p = planHistogram(64, 1, /*laneFree=*/0b11, 0);
    CHECK(p.election.needsLaneTest);
    emitHistogramCount(c, body, p, regs({"v0"}), nm);
    CHECK(render(body).find("if ((lane & 3) == 0)") != std::string::npos);
  }

  CASE("no free bits means every thread counts its own element");
  {
    msl::Context c;
    msl::Block body;
    HistogramNames nm;
    emitHistogramCount(c, body, planHistogram(64, 1, 0, 0), regs({"v0"}), nm);
    const std::string out = render(body);
    CHECK_LACKS(out, "lane");
    CHECK_LACKS(out, "warp");
  }

  CASE("a program id reads its axis component and casts to int");
  {
    msl::Context c;
    msl::Stmt *s = emitGridQuery(c, GridQuery::ProgramId, 0, "pid");
    const std::string out = render(s);
    CHECK_HAS(out, "int pid = (int)tgid.x;");
  }

  CASE("each axis names its own component");
  {
    msl::Context c;
    CHECK(
        render(emitGridQuery(c, GridQuery::ProgramId, 1, "p")).find("tgid.y") !=
        std::string::npos);
    CHECK(
        render(emitGridQuery(c, GridQuery::ProgramId, 2, "p")).find("tgid.z") !=
        std::string::npos);
  }

  CASE("num_programs reads the count builtin");
  {
    msl::Context c;
    CHECK(render(emitGridQuery(c, GridQuery::NumPrograms, 0, "n"))
              .find("tgcount.x") != std::string::npos);
  }

  CASE("a fourth axis has no component and is refused");
  {
    msl::Context c;
    CHECK(axisComponent(3) == nullptr);
    CHECK(emitGridQuery(c, GridQuery::ProgramId, 3, "p") == nullptr);
  }

  CASE("gather reads at a runtime index");
  {
    msl::Context c;
    msl::Block body;
    msl::SmallVec<msl::Expr *, 8> offs{c.var("i0"), c.var("i1")};
    emitGather(c, body, "buf", offs, regs({"g0", "g1"}), f32());
    const std::string out = render(body);
    CHECK_HAS(out, "float g0 = buf[i0];");
    CHECK_HAS(out, "float g1 = buf[i1];");
  }

  CASE("gather declares each result with the element type");
  {
    msl::Context c;
    msl::Block body;
    msl::SmallVec<msl::Expr *, 8> offs{c.var("i0")};
    emitGather(c, body, "buf", offs, regs({"g0"}),
               ElemType{ElemType::Kind::Int, 32, false});
    CHECK_HAS(render(body), "int g0 = buf[i0];");
  }

  CASE("gather reads a compound offset");
  {
    msl::Context c;
    msl::Block body;
    msl::SmallVec<msl::Expr *, 8> offs{c.binary(
        msl::BinOp::Add, c.binary(msl::BinOp::Mul, c.var("i0"), c.lit(16)),
        c.var("col"))};
    emitGather(c, body, "buf", offs, regs({"g0"}), f32());
    CHECK(render(body).find("float g0 = buf[i0 * 16 + col];") !=
          std::string::npos);
  }

  return ::agpu_test::report("EmitMisc");
}
