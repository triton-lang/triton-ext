// Not a unit test: a generator whose output is fed to the Metal compiler by
// metal_compiles.sh.
#include "agpu/Emitter.h"
#include "agpu/bind/SymbolTable.h"
#include "agpu/emit/EmitElementwise.h"
#include "agpu/emit/EmitKernel.h"
#include "agpu/emit/EmitMove.h"
#include "agpu/emit/LayoutExpr.h"
#include "agpu/msl/Printer.h"

#include <iostream>
#include <limits>
#include <sstream>
#include <string>

using namespace agpu;

namespace {

void printEmittedKernel(std::ostream &os) {
  msl::Context c;
  KernelFacts f;
  f.name = "agpu_probe_kernel";
  f.args = {{"out", f32(), true},
            {"a", f32(), true},
            {"b", f32(), true},
            {"n", i32(), false},
            {"alpha", f32(), false}};
  f.numWarps = 4;

  KernelResult r = emitKernel(c, f, [&](msl::Context &cc) {
    msl::Block b;

    MoveSite site;
    site.elem = [&cc](int64_t r) {
      return cc.subscript(cc.var("a"), cc.lit(r));
    };
    site.guard = [&cc](int64_t r) {
      return cc.binary(msl::BinOp::Lt, cc.lit(r), cc.var("n"));
    };
    for (int i = 0; i < 4; ++i)
      site.values.push_back("va" + std::to_string(i));

    MoveFacts mf;
    mf.regCount = 4;
    mf.hasMask = true;
    emitMove(cc, b, mf, site, f32());
    return b;
  });

  msl::Block blk{r.fn};
  msl::Printer p(os);
  p.printBlock(blk);
}

} // namespace

void printBindVecadd(std::ostream &os) {
  Emitter e;
  KernelFacts f;
  f.name = "agpu_probe_bind_vecadd";
  f.args = {{"out", f32(), true}, {"a", f32(), true}, {"b", f32(), true}};

  const LayoutBasis lb{/*reg=*/{32}, /*lane=*/{1, 2, 4, 8, 16},
                       /*warp=*/{}, /*block=*/{}};

  e.addKernel(f, [&](msl::Context &c) {
    msl::Block body;
    SymbolTable sym;
    int t = 0;
    auto fresh = [&] { return "v" + std::to_string(t++); };

    ValueNames idx;
    for (int r = 0; r < 2; ++r) {
      const msl::Str n = fresh();
      body.push_back(
          c.declStmt(mslTypeOf(i32()), n, coordExpr(c, lb, r, "lane", "warp")));
      idx.push_back(n);
    }
    sym.bindRegs(1, idx);

    ValueId next = 2;
    for (const char *buf : {"a", "b"}) {
      ValueNames ns;
      for (int r = 0; r < 2; ++r) {
        const msl::Str n = fresh();
        body.push_back(c.declStmt(
            mslTypeOf(f32()), n,
            c.subscript(c.var(buf), c.var(*sym.regAt(1, (std::size_t)r)))));
        ns.push_back(n);
      }
      sym.bindRegs(next++, ns);
    }

    ValueNames sum;
    for (int r = 0; r < 2; ++r) {
      const msl::Str n = fresh();
      body.push_back(c.declStmt(mslTypeOf(f32()), n,
                                ewExpr(c, EwOp::Add, f32(),
                                       c.var(*sym.regAt(2, (std::size_t)r)),
                                       c.var(*sym.regAt(3, (std::size_t)r)))));
      sum.push_back(n);
    }
    sym.bindRegs(4, sum);

    for (int r = 0; r < 2; ++r)
      body.push_back(c.assign(
          c.subscript(c.var("out"), c.var(*sym.regAt(1, (std::size_t)r))),
          c.var(*sym.regAt(4, (std::size_t)r))));
    return body;
  });

  std::ostringstream own;
  e.print(own);
  const std::string text = own.str();
  const std::size_t at = text.find("kernel void");
  if (at != std::string::npos)
    os << text.substr(at);
}

void printPlannedModule(std::ostream &os) {
  Emitter e;

  KernelFacts f;
  f.name = "agpu_probe_module";
  f.args = {{"out", f32(), true}, {"n", i32(), false}};
  f.numWarps = 4;

  e.addKernel(f, [&](msl::Context &c) {
    msl::Block b;
    b.push_back(
        c.assign(c.subscript(c.var("out"), c.var("lane")), c.litF(0.5)));
    return b;
  });

  std::ostringstream own;
  e.print(own);

  const std::string text = own.str();
  const std::size_t kernelAt = text.find("kernel void");
  if (kernelAt != std::string::npos)
    os << text.substr(kernelAt);
}

int main() {
  printModuleHeader(std::cout);

  printEmittedKernel(std::cout);
  std::cout << "\n";

  printPlannedModule(std::cout);
  std::cout << "\n";

  printBindVecadd(std::cout);
  return 0;
}
