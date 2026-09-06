// Device functions: the signature, its prototype and the calls to it.
#include "agpu/emit/EmitDeviceFn.h"
#include "agpu/emit/PrintModule.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

static PrintPlan prints;
static AssertPlan asserts;

namespace {

DeviceValue scalar(ElemType e) { return DeviceValue{e, false, 1}; }
DeviceValue pointer(ElemType e) { return DeviceValue{e, true, 1}; }
DeviceValue tensor(ElemType e, int64_t regs) {
  return DeviceValue{e, false, regs};
}

DeviceFnFacts fnOf(const char *name, std::vector<DeviceValue> params,
                   std::vector<DeviceValue> results) {
  DeviceFnFacts f;
  f.name = name;
  f.params = std::move(params);
  f.results = std::move(results);
  return f;
}

} // namespace

int main() {
  // ── the return shape ───────────────────────────────────────────────────

  CASE("no results is void and returns nothing");
  {
    DeviceFnFacts f = fnOf("store_it", {pointer(f32())}, {});
    DeviceFnAbi abi = planDeviceFn(f);
    CHECK(abi.ret == RetShape::Void);
    CHECK(abi.retFields.empty());
    CHECK(!abi.returnsStruct());

    msl::Context c;
    CHECK_EQ(render(emitDeviceReturn(c, abi, {})), std::string("return;\n"));
  }

  CASE("one scalar result comes back directly");
  {
    DeviceFnFacts f = fnOf("sum", {scalar(f32())}, {scalar(f32())});
    DeviceFnAbi abi = planDeviceFn(f);
    CHECK(abi.ret == RetShape::Scalar);
    CHECK(!abi.returnsStruct());

    msl::Context c;
    CHECK(mslTypeOf(f32()) == deviceRetType(f, abi));
    CHECK_EQ(render(emitDeviceReturn(c, abi, {"v"})),
             std::string("return v;\n"));
    CHECK(emitRetStruct(c, f, abi) == nullptr);
  }

  CASE("one tensor result becomes a struct, one field per register");
  {
    DeviceFnFacts f = fnOf("load_tile", {pointer(f32())}, {tensor(f32(), 4)});
    DeviceFnAbi abi = planDeviceFn(f);
    CHECK(abi.ret == RetShape::Struct);
    CHECK_EQ((int)abi.retFields.size(), 4);

    msl::Context c;
    const std::string s = render(emitRetStruct(c, f, abi));
    CHECK(s.find("struct load_tile_ret {") != std::string::npos);
    CHECK_EQ(countOf(s, "float f"), 4);
    CHECK(s.find("float f0;") != std::string::npos);
    CHECK(s.find("float f3;") != std::string::npos);

    CHECK_EQ(render(emitDeviceReturn(c, abi, {"r0", "r1", "r2", "r3"})),
             std::string("return {r0, r1, r2, r3};\n"));
  }

  CASE("two results become a struct even when both are scalars");
  {
    DeviceFnFacts f = fnOf("divmod", {scalar(i32()), scalar(i32())},
                           {scalar(i32()), scalar(i32())});
    DeviceFnAbi abi = planDeviceFn(f);
    CHECK(abi.ret == RetShape::Struct);
    CHECK_EQ((int)abi.retFields.size(), 2);

    msl::Context c;
    CHECK(deviceRetType(f, abi) == msl::Type::named("divmod_ret"));
  }

  CASE("mixed results contribute their own register counts");
  {
    DeviceFnFacts f = fnOf("both", {}, {tensor(f16(), 2), scalar(i32())});
    DeviceFnAbi abi = planDeviceFn(f);
    CHECK_EQ((int)abi.retFields.size(), 3);
    CHECK(abi.retFields[0] == f16());
    CHECK(abi.retFields[1] == f16());
    CHECK(abi.retFields[2] == i32());

    msl::Context c;
    const std::string s = render(emitRetStruct(c, f, abi));
    CHECK(s.find("half f0;") != std::string::npos);
    CHECK(s.find("half f1;") != std::string::npos);
    CHECK(s.find("int f2;") != std::string::npos);
  }

  // ── the implicit arguments ─────────────────────────────────────────────

  CASE("thread context arrives without attributes");
  {
    // [[thread_position_in_threadgroup]] on a non-kernel function does not
    // compile, so the same three values are bare uint3s the caller fills in.
    DeviceFnFacts f = fnOf("g", {scalar(f32())}, {});
    DeviceFnAbi abi = planDeviceFn(f);

    msl::Context c;
    msl::Function *fn = emitDeviceFn(c, f, abi, {"x"}, {});
    const std::string s = render(fn);
    const DeviceFnNames dnm;
    CHECK(s.find("uint3 " + dnm.threadgroupPos) != std::string::npos);
    CHECK(s.find("uint3 " + dnm.threadId) != std::string::npos);
    CHECK(s.find("uint3 " + dnm.gridSize) != std::string::npos);
    CHECK(s.find("[[") == std::string::npos);
    CHECK(s.find("kernel ") == std::string::npos);
  }

  CASE("a module with a pool threads it down as a fourth parameter");
  {
    // MSL forbids declaring threadgroup memory outside a kernel, so the pool
    // is declared once in the kernel and passed as a pointer.
    DeviceFnFacts f = fnOf("g", {}, {});
    f.moduleNeedsPool = true;
    DeviceFnAbi abi = planDeviceFn(f);
    CHECK(abi.needsPool());
    CHECK_EQ((int)abi.implicit.size(), 4);

    msl::Context c;
    const std::string s = render(emitDeviceFn(c, f, abi, {}, {}));
    CHECK(s.find("threadgroup char * " + DeviceFnNames{}.pool) !=
          std::string::npos);

    DeviceFnFacts none = fnOf("h", {}, {});
    DeviceFnAbi noPool = planDeviceFn(none);
    CHECK(!noPool.needsPool());
    CHECK_EQ((int)noPool.implicit.size(), 3);
    msl::Context c2;
    CHECK(render(emitDeviceFn(c2, none, noPool, {}, {}))
              .find(DeviceFnNames{}.pool) == std::string::npos);
  }

  CASE("a module that asserts threads the buffer down to every device fn");
  {
    // The assert can sit in a callee, so the kernel takes the buffer and
    // forwards it.
    DeviceFnFacts f = fnOf("g", {}, {});
    f.moduleAsserts = true;
    DeviceFnAbi abi = planDeviceFn(f);
    CHECK(abi.needsAsserts());
    CHECK_EQ((int)abi.implicit.size(), 4);

    msl::Context c;
    const std::string s = render(emitDeviceFn(c, f, abi, {}, {}));
    CHECK(s.find("device atomic_uint * " + DeviceFnNames{}.assertBuffer) !=
          std::string::npos);

    DeviceFnFacts none = fnOf("h", {}, {});
    DeviceFnAbi noAssert = planDeviceFn(none);
    CHECK(!noAssert.needsAsserts());
    msl::Context c2;
    CHECK(render(emitDeviceFn(c2, none, noAssert, {}, {}))
              .find(DeviceFnNames{}.assertBuffer) == std::string::npos);
  }

  // ── the prototype ──────────────────────────────────────────────────────

  CASE("a prototype is just the signature and a semicolon");
  {
    DeviceFnFacts f =
        fnOf("g", {scalar(f32()), pointer(i32())}, {scalar(f32())});
    DeviceFnAbi abi = planDeviceFn(f);

    msl::Context c;
    const std::string proto = render(emitDeviceProto(c, f, abi));
    CHECK(proto.find(");") != std::string::npos);
    CHECK(proto.find("{") == std::string::npos);
    CHECK(proto.find("float g(") != std::string::npos);

    msl::Function *empty = emitDeviceFn(c, f, abi, {"a", "b"}, {});
    const std::string s = render(empty);
    CHECK(s.find("{") != std::string::npos);
  }

  CASE("the prototype and the definition declare the same parameter types");
  {
    DeviceFnFacts f = fnOf("g", {scalar(f32()), pointer(i32()), scalar(i1())},
                           {scalar(i32())});
    f.moduleNeedsPool = true;
    DeviceFnAbi abi = planDeviceFn(f);

    msl::Context c;
    msl::Function *proto = emitDeviceProto(c, f, abi);
    msl::Function *def = emitDeviceFn(c, f, abi, {"a", "b", "flag"}, {});

    CHECK_EQ(proto->params.size(), def->params.size());
    for (std::size_t i = 0; i < proto->params.size(); ++i)
      CHECK(proto->params[i].type == def->params[i].type);
    CHECK(proto->returnType == def->returnType);

    CHECK_EQ(def->params[0].name, std::string("a"));
    CHECK(proto->params[0].name.empty());
    CHECK_EQ(proto->params[3].name, DeviceFnNames{}.threadgroupPos);
  }

  // ── the call ───────────────────────────────────────────────────────────

  CASE("a call passes the implicit arguments in the ABI's order");
  {
    DeviceFnFacts f = fnOf("g", {scalar(f32())}, {});
    f.moduleNeedsPool = true;
    DeviceFnAbi abi = planDeviceFn(f);

    CallerContext caller{"tgid", "tid", "tgcount", "pool", ""};
    msl::Context c;
    msl::Block body;
    emitDeviceCall(c, body, f, abi, {"x"}, caller, {});
    CHECK_EQ(render(body), std::string("g(x, tgid, tid, tgcount, pool);\n"));
  }

  CASE("the call's argument order matches the parameter order, position for "
       "position");
  {
    DeviceFnFacts f = fnOf("g", {scalar(f32()), scalar(i32())}, {});
    f.moduleNeedsPool = true;
    DeviceFnAbi abi = planDeviceFn(f);

    msl::Context c;
    msl::Function *def = emitDeviceFn(c, f, abi, {"p0", "p1"}, {});
    CallerContext caller{"TG", "TI", "NT", "PL", ""};
    msl::Block body;
    emitDeviceCall(c, body, f, abi, {"A0", "A1"}, caller, {});
    const std::string call = render(body);

    const std::vector<std::string> expect = {"A0", "A1", "TG",
                                             "TI", "NT", "PL"};
    CHECK_EQ((int)def->params.size(), (int)expect.size());
    std::size_t at = 0;
    for (const std::string &e : expect) {
      const std::size_t found = call.find(e, at);
      CHECK(found != std::string::npos);
      at = found;
    }
  }

  CASE("a struct result is destructured through the same field names it was "
       "packed with");
  {
    DeviceFnFacts f = fnOf("load_tile", {pointer(f32())}, {tensor(f32(), 2)});
    DeviceFnAbi abi = planDeviceFn(f);

    msl::Context c;
    CallerContext caller{"tgid", "tid", "tgcount", "", ""};
    msl::Block body;
    emitDeviceCall(c, body, f, abi, {"p"}, caller, {"v0", "v1"}, "t");
    const std::string s = render(body);
    CHECK(s.find("load_tile_ret t = load_tile(p, tgid, tid, tgcount);") !=
          std::string::npos);
    CHECK(s.find("float v0 = t.f0;") != std::string::npos);
    CHECK(s.find("float v1 = t.f1;") != std::string::npos);

    const std::string decl = render(emitRetStruct(c, f, abi));
    CHECK(decl.find("float f0;") != std::string::npos);
    CHECK(decl.find("float f1;") != std::string::npos);
  }

  CASE("a void call emits as a bare statement");
  {
    DeviceFnFacts f = fnOf("g", {}, {});
    DeviceFnAbi abi = planDeviceFn(f);
    msl::Context c;
    msl::Block body;
    emitDeviceCall(c, body, f, abi, {}, CallerContext{"a", "b", "c", "", ""},
                   {});
    CHECK_EQ(render(body), std::string("g(a, b, c);\n"));
  }

  CASE("a scalar call declares the result directly, with no struct in sight");
  {
    DeviceFnFacts f = fnOf("g", {scalar(f32())}, {scalar(f32())});
    DeviceFnAbi abi = planDeviceFn(f);
    msl::Context c;
    msl::Block body;
    emitDeviceCall(c, body, f, abi, {"x"}, CallerContext{"a", "b", "c", "", ""},
                   {"r"});
    const std::string s = render(body);
    CHECK_EQ(s, std::string("float r = g(x, a, b, c);\n"));
    CHECK(s.find("_ret") == std::string::npos);
  }

  // ── the module driver ──────────────────────────────────────────────────

  CASE("a module emits in the order the language requires");
  {
    ModuleFacts m;
    DeviceFnUnit u;
    u.facts = fnOf("callee", {scalar(f32())}, {tensor(f32(), 2)});
    u.paramNames = {"x"};
    m.deviceFns.push_back(std::move(u));

    KernelUnit k;
    k.facts.name = "entry";
    k.facts.args = {{"out", f32(), true}};
    k.facts.numWarps = 1;
    k.buildBody = [](msl::Context &, bool) { return msl::Block{}; };
    m.kernels.push_back(std::move(k));
    m.functionPools = {planFunctionPool({{"reduce", Bytes(256)}}),
                       planFunctionPool({{"dot", Bytes(1024)}})};

    msl::Context c;
    HelperSet helpers;
    std::ostringstream os;
    ModuleResult r = emitModule(c, m, helpers, prints, asserts, os);
    CHECK(r.ok());

    const std::string out = os.str();
    const std::size_t inc = out.find("#include <metal_stdlib>");
    const std::size_t st = out.find("struct callee_ret");
    const std::size_t proto = out.find("callee_ret callee(float, uint3");
    const std::size_t def = out.find("callee_ret callee(float x");
    const std::size_t kern = out.find("kernel void entry");
    CHECK(inc != std::string::npos);
    CHECK(st != std::string::npos);
    CHECK(proto != std::string::npos);
    CHECK(def != std::string::npos);
    CHECK(kern != std::string::npos);
    CHECK(inc < st);
    CHECK(st < proto);
    CHECK(proto < def);
    CHECK(def < kern);
  }

  CASE("a module's pool is the largest any function needs, in every kernel");
  {
    ModuleFacts m;
    DeviceFnUnit u;
    u.facts = fnOf("callee", {}, {});
    u.facts.moduleNeedsPool = true;
    m.deviceFns.push_back(std::move(u));

    KernelUnit k;
    k.facts.name = "entry";
    k.facts.args = {{"out", f32(), true}};
    k.facts.numWarps = 1;
    k.facts.poolBytes = 64;
    k.buildBody = [](msl::Context &, bool) { return msl::Block{}; };
    m.kernels.push_back(std::move(k));
    m.functionPools = {planFunctionPool({{"reduce", Bytes(64)}}),
                       planFunctionPool({{"dot", Bytes(4096)}})};

    msl::Context c;
    HelperSet helpers;
    std::ostringstream os;
    ModuleResult r = emitModule(c, m, helpers, prints, asserts, os);
    CHECK(r.ok());
    CHECK_EQ(r.poolBytes, 4096);

    const std::string out = os.str();
    CHECK(out.find("threadgroup char pool[4096]") != std::string::npos);
    CHECK(out.find("pool[64]") == std::string::npos);
    CHECK(out.find("threadgroup char * " + DeviceFnNames{}.pool) !=
          std::string::npos);
  }

  CASE("an over-limit module is refused before a character is emitted");
  {
    ModuleFacts m;
    KernelUnit k;
    k.facts.name = "entry";
    k.facts.args = {{"out", f32(), true}};
    k.facts.numWarps = 1;
    k.buildBody = [](msl::Context &, bool) { return msl::Block{}; };
    m.kernels.push_back(std::move(k));
    m.functionPools = {
        planFunctionPool({{"dot", Bytes(kTGResidentBudgetBytes + 1)}})};

    msl::Context c;
    HelperSet helpers;
    std::ostringstream os;
    ModuleResult r = emitModule(c, m, helpers, prints, asserts, os);
    CHECK(!r.ok());
    CHECK(r.decision.isDecline());
    CHECK(os.str().empty());
  }

  CASE("a module that only costs occupancy is emitted");
  {
    ModuleFacts m;
    KernelUnit k;
    k.facts.name = "entry";
    k.facts.args = {{"out", f32(), true}};
    k.facts.numWarps = 1;
    k.buildBody = [](msl::Context &, bool) { return msl::Block{}; };
    m.kernels.push_back(std::move(k));
    m.functionPools = {
        planFunctionPool({{"dot", Bytes(kTGResidentBudgetBytes)}})};

    msl::Context c;
    HelperSet helpers;
    std::ostringstream os;
    ModuleResult r = emitModule(c, m, helpers, prints, asserts, os);
    CHECK(r.ok());
    CHECK_EQ(r.poolBytes, kTGResidentBudgetBytes);
    CHECK_EQ(std::string(r.pool.driver), std::string("dot"));
  }

  CASE("a module with no pool passes no pool pointer anywhere");
  {
    ModuleFacts m;
    DeviceFnUnit u;
    u.facts = fnOf("callee", {}, {});
    m.deviceFns.push_back(std::move(u));

    KernelUnit k;
    k.facts.name = "entry";
    k.facts.args = {{"out", f32(), true}};
    k.facts.numWarps = 1;
    k.buildBody = [](msl::Context &, bool) { return msl::Block{}; };
    m.kernels.push_back(std::move(k));

    msl::Context c;
    HelperSet helpers;
    std::ostringstream os;
    ModuleResult r = emitModule(c, m, helpers, prints, asserts, os);
    CHECK(r.ok());
    CHECK_EQ(r.poolBytes, 0);
    const std::string out = os.str();
    CHECK(out.find("threadgroup char * " + DeviceFnNames{}.pool) ==
          std::string::npos);
    CHECK(out.find("threadgroup char pool") == std::string::npos);
  }

  CASE("a module of kernels alone emits no prototypes and no structs");
  {
    ModuleFacts m;
    KernelUnit k;
    k.facts.name = "entry";
    k.facts.args = {{"out", f32(), true}};
    k.facts.numWarps = 1;
    k.buildBody = [](msl::Context &, bool) { return msl::Block{}; };
    m.kernels.push_back(std::move(k));

    msl::Context c;
    HelperSet helpers;
    std::ostringstream os;
    ModuleResult r = emitModule(c, m, helpers, prints, asserts, os);
    CHECK(r.ok());
    CHECK(r.protos.empty());
    CHECK(r.retStructs.empty());
    CHECK(os.str().find("struct ") == std::string::npos);
  }

  return ::agpu_test::report("DeviceFn");
}
