// The entry point: one header, one object, one call per kernel.
#include "agpu/Emitter.h"
#include "fixtures.h"
#include "harness.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;

namespace {

DotFacts gemm(int64_t M, int64_t N, int64_t K) {
  DotFacts f;
  f.M = M;
  f.N = N;
  f.K = K;
  f.aElemBytes = 2;
  f.bElemBytes = 2;
  f.numWarps = 4;
  return f;
}

DotInputs dotInputs() {
  DotInputs in;
  in.a = {"pA", 64};
  in.b = {"pB", 64};
  return in;
}

} // namespace

int main() {
  CASE("a caller emits a whole module with one object and one print");
  {
    Emitter e;
    e.addKernel(
        [] {
          KernelFacts f;
          f.name = "k";
          f.args = {{"out", f32(), true}};
          f.numWarps = 4;
          return f;
        }(),
        [&](msl::Context &c, bool) {
          msl::Block b;
          b.push_back(
              c.assign(c.subscript(c.var("out"), c.var("lane")), c.litF(1.0)));
          return b;
        });

    std::ostringstream os;
    ModuleResult r = e.print(os);
    CHECK(r.ok());

    const std::string out = os.str();
    CHECK(out.find("#include <metal_stdlib>") != std::string::npos);
    CHECK(out.find("kernel void k") != std::string::npos);
    CHECK(out.find("#include <metal_stdlib>") < out.find("kernel void k"));
  }

  CASE("a helper named by any kernel reaches the prelude before all of them");
  {
    Emitter e;
    for (int i = 0; i < 3; ++i) {
      KernelFacts f;
      f.name = "k" + std::to_string(i);
      f.args = {{"out", f32(), true}};
      f.numWarps = 1;
      e.addKernel(f, [](msl::Context &, bool) { return msl::Block{}; });
    }
    e.helpers.add(Helper::Erf);

    std::ostringstream os;
    CHECK(e.print(os).ok());
    const std::string out = os.str();
    CHECK(out.find("__agpu_erf") != std::string::npos);
    CHECK(out.find("__agpu_erf") < out.find("kernel void k0"));
  }

  CASE("a module needing no helper emits none");
  {
    Emitter e;
    KernelFacts f;
    f.name = "k";
    f.args = {{"out", f32(), true}};
    f.numWarps = 1;
    e.addKernel(f, [](msl::Context &, bool) { return msl::Block{}; });

    std::ostringstream os;
    CHECK(e.print(os).ok());
    CHECK(os.str().find("__agpu_") == std::string::npos);
  }

  CASE("a dot is emitted from its facts and records what it needs");
  {
    Emitter e;
    msl::Block body;
    Decision d = e.dot(body, gemm(64, 64, 64), dotInputs());
    CHECK(d.ok());
    CHECK(!body.empty());

    const FunctionPool p = e.pool.plan();
    CHECK(p.scratch > Bytes(0));
    CHECK_EQ(std::string(p.driver), std::string("dot"));
  }

  CASE("a caller can ask what a shape costs before emitting it");
  {
    Emitter e;
    const Plan p = e.planFor(gemm(64, 64, 64));
    CHECK(p.kind == Plan::Kind::Direct);

    const Plan panel = e.planFor(gemm(512, 512, 64));
    CHECK(panel.kind == Plan::Kind::Panel);

    CHECK(e.pool.plan().scratch == Bytes(0));
  }

  CASE("an unsupported dot declines and emits nothing");
  {
    Emitter e;
    msl::Block body;
    Decision d = e.dot(body, gemm(64, 64, 12), dotInputs());
    CHECK(d.isDecline());
    CHECK(body.empty());
    CHECK(e.pool.plan().scratch == Bytes(0));
  }

  CASE("a decline is recorded for later reporting");
  {
    Emitter e;
    msl::Block body;
    e.dot(body, gemm(64, 64, 12), dotInputs());
    CHECK_EQ(e.declines.size(), 1u);
    CHECK(e.declines.declined("shape is not fragment-aligned"));
  }

  CASE("an autotune sweep over one shape reads as one problem");
  {
    // N configurations of one kernel are one site.
    Emitter e;
    for (int warps : {1, 2, 4, 8}) {
      DotFacts f = gemm(64, 64, 12);
      f.numWarps = warps;
      e.site = DeclineSite{"kernel.py:31", "warps=" + std::to_string(warps)};
      msl::Block body;
      e.dot(body, f, dotInputs());
    }

    const auto sum = e.declines.summary();
    CHECK_EQ((int)sum.size(), 1);
    CHECK_EQ(sum[0].occurrences, 4u);
    CHECK_EQ(sum[0].distinctSites(), 1u);
  }

  CASE("a module that emits everything declines nothing");
  {
    Emitter e;
    msl::Block body;
    CHECK(e.dot(body, gemm(64, 64, 64), dotInputs()).ok());
    CHECK(e.declines.empty());
  }

  CASE("a module past the hardware limit is refused before it is printed");
  {
    // Past that limit a declaration compiles and links, then takes down
    // MTLCompilerService.
    Emitter e;
    KernelFacts f;
    f.name = "k";
    f.args = {{"out", f32(), true}};
    f.numWarps = 1;
    e.addKernel(f, [](msl::Context &, bool) { return msl::Block{}; });
    e.pool.scratch("huge", Bytes(kTGResidentBudgetBytes + 1));

    std::ostringstream os;
    ModuleResult r = e.print(os);
    CHECK(!r.ok());
    CHECK(r.decision.isDecline());
    CHECK(os.str().empty());
  }

  CASE("a module that only costs occupancy is printed");
  {
    Emitter e;
    KernelFacts f;
    f.name = "k";
    f.args = {{"out", f32(), true}};
    f.numWarps = 1;
    e.addKernel(f, [](msl::Context &, bool) { return msl::Block{}; });
    e.pool.scratch("dot", Bytes(kTGResidentBudgetBytes));

    std::ostringstream os;
    CHECK(e.print(os).ok());
    CHECK(os.str().find("kernel void k") != std::string::npos);
  }

  CASE("a device function is declared before the kernel that calls it");
  {
    // MSL has no forward reference, so the order is a language requirement.
    Emitter e;
    DeviceFnFacts df;
    df.name = "helper";
    df.params = {DeviceValue{f32(), false, 1}};
    df.results = {DeviceValue{f32(), false, 1}};

    msl::Block dbody;
    dbody.push_back(e.context().returnStmt(e.context().var("x")));
    e.addDeviceFn(df, {"x"}, std::move(dbody));

    KernelFacts kf;
    kf.name = "entry";
    kf.args = {{"out", f32(), true}};
    kf.numWarps = 1;
    e.addKernel(kf, [](msl::Context &, bool) { return msl::Block{}; });

    std::ostringstream os;
    ModuleResult r = e.print(os);
    CHECK(r.ok());
    const std::string out = os.str();
    CHECK_EQ(countOf(out, "float helper("), 2); // prototype and definition
    CHECK(out.find("float helper(float, uint3") != std::string::npos);
    CHECK(out.find("float helper(float, uint3") <
          out.find("kernel void entry"));
  }

  CASE("the decline summary is gated and goes to its own stream");
  {
    Emitter e;
    e.declines.record(Decision::declined("emitDot", "ragged k"),
                      DeclineSite{"a.py:1", "w4"});

    std::ostringstream diag;
    e.printDeclineSummary(diag);
    CHECK(diag.str().empty()); // gate is off by default

    e.gates.set(Gate::LogReject);
    std::ostringstream on;
    e.printDeclineSummary(on);
    CHECK(on.str().find("distinct rejects: 1") != std::string::npos);
    CHECK(on.str().find("ragged k") != std::string::npos);
  }

  CASE("a vestigial op emits nothing and says nothing");
  {
    // A previous pass already consumed the information these ops carry.
    Emitter e;
    e.gates.set(Gate::LogReject);
    e.site = DeclineSite{"kernel.py:41", "w4"};

    CHECK(e.vestigial("scf.yield").ok());
    CHECK(e.vestigial("llvm.intr.assume").ok());
    CHECK(e.declines.empty());

    std::ostringstream os;
    e.printDeclineSummary(os);
    CHECK(os.str().empty());
  }

  CASE("an op outside the table keeps the caller looking");
  {
    Emitter e;
    const Decision d = e.vestigial("tt.dot");
    CHECK(d.keepLooking());
    CHECK(!d.ok());
    CHECK(e.declines.empty());
  }

  CASE("an autotune sweep over one declining shape is one problem");
  {
    Emitter e;
    e.gates.set(Gate::LogReject);
    for (const char *cfg : {"w1", "w2", "w4"}) {
      e.site = DeclineSite{"kernel.py:41", cfg};
      e.declines.record(Decision::declined("emitScan", "gapped lane ladder"),
                        e.site);
    }
    const std::vector<DeclineTally> rows = e.declines.summary();
    CHECK_EQ(rows.size(), (std::size_t)1);
    CHECK_EQ(rows[0].occurrences, (std::size_t)3);
    CHECK_EQ(rows[0].distinctSites(), (std::size_t)1);
    CHECK_EQ(rows[0].distinctConfigs(), (std::size_t)3);
  }

  CASE("f64 is spelled float and the user is told");
  {
    Emitter e;
    e.gates.set(Gate::LogReject);
    e.site = DeclineSite{"model.py:7", "w4"};

    e.noteIfNarrowed(f64());
    CHECK(mslTypeOf(f64()) == mslTypeOf(f32()));

    std::ostringstream os;
    e.printDeclineSummary(os);
    const std::string out = os.str();
    CHECK(out.find("f64") != std::string::npos);
    CHECK(out.find("MSL-PLAN-SITE") != std::string::npos);
    CHECK(out.find("distinct rejects: 0") != std::string::npos);
  }

  CASE("a type that loses nothing says nothing");
  {
    Emitter e;
    for (ElemType t : {f32(), f16(), bf16(), i32()})
      e.noteIfNarrowed(t);
    CHECK(e.declines.empty());
  }

  CASE("a rebinding renames registers and emits nothing");
  {
    Emitter e;
    Rebind r;
    r.from = {2, 0, 3, 1};
    std::vector<msl::Str> out;
    CHECK(e.rebindTo(r, {"a", "b", "c", "d"}, out).ok());
    CHECK_EQ(out, (std::vector<msl::Str>{"c", "a", "d", "b"}));
    CHECK(e.declines.empty());
  }

  CASE("layouts that disagree decline and the user hears about it");
  {
    // A result register with no source needs a data movement.
    Emitter e;
    e.gates.set(Gate::LogReject);
    e.site = DeclineSite{"model.py:12", "w4"};
    Rebind r;
    r.from = {0, -1};
    std::vector<msl::Str> out;
    CHECK(e.rebindTo(r, {"a"}, out).isDecline());

    std::ostringstream os;
    e.printDeclineSummary(os);
    CHECK(os.str().find("layouts disagree") != std::string::npos);
    CHECK(os.str().find("model.py:12") != std::string::npos);
  }

  CASE("a source list shorter than the plan asserts as a caller bug");
  {
    Emitter e;
    Rebind r;
    r.from = {0, 1};
    std::vector<msl::Str> out;
    const Decision d = e.rebindTo(r, {"a"}, out);
    CHECK(d.isBug());
    CHECK(!d.isDecline());
  }

  CASE("a clean module prints no summary even with the gate on");
  {
    Emitter e;
    e.gates.set(Gate::LogReject);
    std::ostringstream os;
    e.printDeclineSummary(os);
    CHECK(os.str().empty());
  }

  return ::agpu_test::report("Emitter");
}
