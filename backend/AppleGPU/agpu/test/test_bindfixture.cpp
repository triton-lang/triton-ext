// Driving bind/ with a fixture: a whole kernel, no IR.
#include "agpu/Emitter.h"
#include "agpu/bind/Dispatch.h"
#include "agpu/bind/SymbolTable.h"
#include "agpu/emit/EmitElementwise.h"
#include "agpu/emit/EmitRange.h"
#include "agpu/msl/Printer.h"
#include "harness.h"

#include <sstream>

using namespace agpu;

namespace {

struct FixtureOp {
  enum class Kind { Range, Binary, Load, Store } kind;
  ValueId result = 0;
  std::vector<ValueId> operands;
  EwOp binOp = EwOp::Add;
  msl::Str buffer; // Load/Store: which pointer
  int64_t rangeStart = 0;
};

bool runFixture(msl::Context &c, msl::Block &body, SymbolTable &sym,
                const std::vector<FixtureOp> &ops, int regs,
                const LayoutBasis &lb) {
  int tmp = 0;
  auto fresh = [&] { return "v" + std::to_string(tmp++); };

  for (const FixtureOp &op : ops) {
    switch (op.kind) {
    case FixtureOp::Kind::Range: {
      ValueNames names;
      for (int r = 0; r < regs; ++r) {
        const msl::Str n = fresh();
        body.push_back(
            c.declStmt(mslTypeOf(i32()), n,
                       rangeElem(c, lb, r, op.rangeStart, "lane", "warp")));
        names.push_back(n);
      }
      sym.bindRegs(op.result, std::move(names));
      break;
    }
    case FixtureOp::Kind::Load: {
      const msl::Str *idx = nullptr;
      if (op.operands.empty() || !(idx = sym.regAt(op.operands[0], 0)))
        return false;
      ValueNames names;
      for (int r = 0; r < regs; ++r) {
        const msl::Str *ir = sym.regAt(op.operands[0], (std::size_t)r);
        if (!ir)
          return false;
        const msl::Str n = fresh();
        body.push_back(c.declStmt(mslTypeOf(f32()), n,
                                  c.subscript(c.var(op.buffer), c.var(*ir))));
        names.push_back(n);
      }
      sym.bindRegs(op.result, std::move(names));
      break;
    }
    case FixtureOp::Kind::Binary: {
      if (op.operands.size() != 2)
        return false;
      ValueNames names;
      for (int r = 0; r < regs; ++r) {
        const msl::Str *a = sym.regAt(op.operands[0], (std::size_t)r);
        const msl::Str *b = sym.regAt(op.operands[1], (std::size_t)r);
        if (!a || !b)
          return false;
        const msl::Str n = fresh();
        body.push_back(emitEw(c, op.binOp, f32(), n, c.var(*a), c.var(*b)));
        names.push_back(n);
      }
      sym.bindRegs(op.result, std::move(names));
      break;
    }
    case FixtureOp::Kind::Store: {
      if (op.operands.size() != 2)
        return false;
      for (int r = 0; r < regs; ++r) {
        const msl::Str *idx = sym.regAt(op.operands[0], (std::size_t)r);
        const msl::Str *val = sym.regAt(op.operands[1], (std::size_t)r);
        if (!idx || !val)
          return false;
        body.push_back(
            c.assign(c.subscript(c.var(op.buffer), c.var(*idx)), c.var(*val)));
      }
      sym.bindDataless(op.result);
      break;
    }
    }
  }
  return true;
}

OpView op(std::string_view name, std::vector<ValueId> operands = {},
          std::vector<ValueId> results = {}) {
  OpView o;
  o.name = name;
  o.operands = std::move(operands);
  o.results = std::move(results);
  return o;
}

LayoutBasis laneMajor() {
  return LayoutBasis{/*reg=*/{32}, /*lane=*/{1, 2, 4, 8, 16}, /*warp=*/{},
                     /*block=*/{}};
}

} // namespace

int main() {
  CASE("a vecadd threads through the table and comes out as MSL");
  {
    msl::Context c;
    msl::Block body;
    SymbolTable sym;

    const std::vector<FixtureOp> ops = {
        {FixtureOp::Kind::Range, 1, {}, EwOp::Add, "", 0},
        {FixtureOp::Kind::Load, 2, {1}, EwOp::Add, "a", 0},
        {FixtureOp::Kind::Load, 3, {1}, EwOp::Add, "b", 0},
        {FixtureOp::Kind::Binary, 4, {2, 3}, EwOp::Add, "", 0},
        {FixtureOp::Kind::Store, 5, {1, 4}, EwOp::Add, "out", 0},
    };

    CHECK(runFixture(c, body, sym, ops, 2, laneMajor()));

    std::ostringstream os;
    msl::Printer p(os);
    p.printBlock(body);
    const std::string msl = os.str();

    CHECK(msl.find("lane & 31") != std::string::npos);
    CHECK(msl.find("a[") != std::string::npos);
    CHECK(msl.find("b[") != std::string::npos);
    CHECK(msl.find("out[") != std::string::npos);
    CHECK(msl.find(" + ") != std::string::npos);

    for (ValueId v : {1u, 2u, 3u, 4u})
      CHECK(sym.regCount(v) == 2);
    CHECK(sym.isBound(5));
    CHECK(sym.isDataless(5));
  }

  CASE("reading a value the walk has not defined fails and says so");
  {
    msl::Context c;
    msl::Block body;
    SymbolTable sym;

    const std::vector<FixtureOp> ops = {
        {FixtureOp::Kind::Binary, 4, {2, 3}, EwOp::Add, "", 0},
    };
    CHECK(!runFixture(c, body, sym, ops, 2, laneMajor()));
  }

  CASE("a dataless value used as an operand also fails");
  {
    msl::Context c;
    msl::Block body;
    SymbolTable sym;
    sym.bindDataless(2);
    sym.bindRegs(3, {"x0", "x1"});

    const std::vector<FixtureOp> ops = {
        {FixtureOp::Kind::Binary, 4, {2, 3}, EwOp::Add, "", 0},
    };
    CHECK(!runFixture(c, body, sym, ops, 2, laneMajor()));
    CHECK(sym.isBound(2));
    CHECK(sym.isDataless(2));
  }

  CASE("a splat operand broadcasts across a tensor's registers");
  {
    msl::Context c;
    msl::Block body;
    SymbolTable sym;
    sym.bindScalar(1, "scale");
    sym.bindRegs(2, {"a0", "a1", "a2", "a3"});

    const std::vector<FixtureOp> ops = {
        {FixtureOp::Kind::Binary, 3, {2, 1}, EwOp::Mul, "", 0},
    };
    CHECK(runFixture(c, body, sym, ops, 4, laneMajor()));
    CHECK_EQ(sym.regCount(3), (std::size_t)4);

    std::ostringstream os;
    msl::Printer p(os);
    p.printBlock(body);
    const std::string msl = os.str();
    CHECK(msl.find("a0 * scale") != std::string::npos);
    CHECK(msl.find("a3 * scale") != std::string::npos);
  }

  CASE("the same fixture drives a whole kernel through Emitter");
  {
    Emitter e;
    KernelFacts f;
    f.name = "vecadd";
    f.args = {{"out", f32(), true}, {"a", f32(), true}, {"b", f32(), true}};

    const std::vector<FixtureOp> ops = {
        {FixtureOp::Kind::Range, 1, {}, EwOp::Add, "", 0},
        {FixtureOp::Kind::Load, 2, {1}, EwOp::Add, "a", 0},
        {FixtureOp::Kind::Load, 3, {1}, EwOp::Add, "b", 0},
        {FixtureOp::Kind::Binary, 4, {2, 3}, EwOp::Add, "", 0},
        {FixtureOp::Kind::Store, 5, {1, 4}, EwOp::Add, "out", 0},
    };

    bool built = false;
    e.addKernel(f, [&](msl::Context &c, bool) {
      msl::Block body;
      SymbolTable sym;
      built = runFixture(c, body, sym, ops, 2, laneMajor());
      return body;
    });

    std::ostringstream os;
    const ModuleResult r = e.print(os);
    CHECK(built);
    CHECK(r.ok());

    const std::string msl = os.str();
    CHECK(msl.find("kernel void vecadd") != std::string::npos);
    CHECK(msl.find("#include <metal_stdlib>") != std::string::npos);
    CHECK(msl.find("out[") != std::string::npos);
  }

  CASE("the fixture runs through DispatchTable, the way the bridge does");
  {
    msl::Context c;
    msl::Block body;
    SymbolTable sym;
    const LayoutBasis lb = laneMajor();
    const int regs = 2;
    int tmp = 0;
    auto fresh = [&] { return "v" + std::to_string(tmp++); };

    DispatchTable t;

    t.add("range", forOps({"tt.make_range"}, [&](const OpView &o) {
            ValueNames ns;
            for (int r = 0; r < regs; ++r) {
              const msl::Str n = fresh();
              body.push_back(
                  c.declStmt(mslTypeOf(i32()), n,
                             rangeElem(c, lb, r, o.intAt(0), "lane", "warp")));
              ns.push_back(n);
            }
            sym.bindRegs(o.results[0], std::move(ns));
            return Decision::emitted();
          }));

    t.add("load", forOps({"tt.load"}, [&](const OpView &o) {
            ValueNames ns;
            for (int r = 0; r < regs; ++r) {
              const msl::Str *idx = sym.regAt(o.operands[0], (std::size_t)r);
              if (!idx)
                return Decision::failed();
              const msl::Str n = fresh();
              body.push_back(c.declStmt(
                  mslTypeOf(f32()), n,
                  c.subscript(c.var(msl::Str(o.text)), c.var(*idx))));
              ns.push_back(n);
            }
            sym.bindRegs(o.results[0], std::move(ns));
            return Decision::emitted();
          }));

    t.add("binop", forOps({"arith.addf"}, [&](const OpView &o) {
            ValueNames ns;
            for (int r = 0; r < regs; ++r) {
              const msl::Str *a = sym.regAt(o.operands[0], (std::size_t)r);
              const msl::Str *b = sym.regAt(o.operands[1], (std::size_t)r);
              if (!a || !b)
                return Decision::failed();
              const msl::Str n = fresh();
              body.push_back(
                  emitEw(c, EwOp::Add, f32(), n, c.var(*a), c.var(*b)));
              ns.push_back(n);
            }
            sym.bindRegs(o.results[0], std::move(ns));
            return Decision::emitted();
          }));

    t.add("store", forOps({"tt.store"}, [&](const OpView &o) {
            for (int r = 0; r < regs; ++r) {
              const msl::Str *idx = sym.regAt(o.operands[0], (std::size_t)r);
              const msl::Str *val = sym.regAt(o.operands[1], (std::size_t)r);
              if (!idx || !val)
                return Decision::failed();
              body.push_back(
                  c.assign(c.subscript(c.var(msl::Str(o.text)), c.var(*idx)),
                           c.var(*val)));
            }
            sym.bindDataless(o.results[0]);
            return Decision::emitted();
          }));

    std::vector<OpView> program;
    {
      OpView o = op("tt.make_range", {}, {1});
      o.ints = {0};
      program.push_back(o);
    }
    {
      OpView o = op("tt.load", {1}, {2});
      o.text = "a";
      program.push_back(o);
    }
    {
      OpView o = op("tt.load", {1}, {3});
      o.text = "b";
      program.push_back(o);
    }
    program.push_back(op("arith.addf", {2, 3}, {4}));
    {
      OpView o = op("tt.store", {1, 4}, {5});
      o.text = "out";
      program.push_back(o);
    }

    for (const OpView &o : program) {
      std::string who;
      const Decision d = t.runNamed(o, who);
      CHECK(d.ok());
      CHECK(!who.empty());
    }

    std::ostringstream os;
    msl::Printer p(os);
    p.printBlock(body);
    const std::string msl = os.str();
    CHECK(msl.find("a[") != std::string::npos);
    CHECK(msl.find("b[") != std::string::npos);
    CHECK(msl.find(" + ") != std::string::npos);
    CHECK(msl.find("out[") != std::string::npos);
    CHECK(sym.isDataless(5));
  }

  CASE("an op the table does not know declines, naming itself");
  {
    DispatchTable t;
    t.add("range", forOps({"tt.make_range"},
                          [](const OpView &) { return Decision::emitted(); }));

    const Decision d = t.run(op("tt.histogram", {1}, {2}));
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK_EQ(d.where(), msl::Str("tt.histogram"));
  }

  return ::agpu_test::report("BindFixture");
}
