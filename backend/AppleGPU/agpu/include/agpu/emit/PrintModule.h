// PrintModule.h - an emitted module as MSL text, in the order MSL requires.
//
// Order: includes/helpers, return structs, every device function's prototype
// (MSL has no forward reference), device function bodies, then kernels.
#ifndef AGPU_PRINT_MODULE_H
#define AGPU_PRINT_MODULE_H

#include "agpu/emit/EmitModule.h"
#include "agpu/emit/Prelude.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/AssertPlan.h"

#include <ostream>

namespace agpu {

inline void printModuleHeader(std::ostream &os) {
  os << "#include <metal_stdlib>\n"
     << "#include <metal_simdgroup_matrix>\n"
     << "using namespace metal;\n\n";
}

// `header` is on by default; a caller emitting several kernels into one file
// passes false after the first.
inline void printPrelude(std::ostream &os, const HelperSet &h,
                         bool header = true) {
  if (header)
    printModuleHeader(os);

  for (unsigned i = 0; i < unsigned(Helper::Count); ++i) {
    const Helper which = Helper(i);
    if (h.has(which))
      os << helperSource(which) << "\n\n";
  }
}

inline void printModuleBody(std::ostream &os, const ModuleResult &r) {
  msl::Printer p(os);
  for (msl::Stmt *s : r.retStructs) {
    msl::Block b{s};
    p.printBlock(b);
    os << "\n";
  }

  if (!r.protos.empty()) {
    msl::Block b;
    for (msl::Function *fn : r.protos)
      b.push_back(fn);
    p.printBlock(b);
    os << "\n";
  }

  for (msl::Function *fn : r.deviceFns) {
    msl::Block b{fn};
    p.printBlock(b);
  }

  for (const KernelResult &kr : r.kernels) {
    msl::Block b{kr.fn};
    p.printBlock(b);
  }
}

// A helper may be added while a body is built, so `helpers` must be the set as
// it stood after `emitModule` returned.
inline void printModule(std::ostream &os, const ModuleResult &r,
                        const HelperSet &helpers) {
  printPrelude(os, helpers);
  printModuleBody(os, r);
}

// As above, plus the layout the host needs to read the debug buffers back.
inline void printModule(std::ostream &os, const ModuleResult &r,
                        const HelperSet &helpers, const PrintPlan &prints,
                        const AssertPlan &asserts) {
  printPrelude(os, helpers);
  // As comments: this is a .metal file.
  if (prints.prints())
    os << "/*\n" << printLayoutText(prints) << "*/\n\n";
  if (asserts.asserts())
    os << "/*\n" << assertLayoutText(asserts) << "*/\n\n";
  printModuleBody(os, r);
}

inline ModuleResult emitModule(msl::Context &c, ModuleFacts &m,
                               HelperSet &helpers, PrintPlan &prints,
                               AssertPlan &asserts, std::ostream &os,
                               const DeviceFnNames &dnm = {}) {
  ModuleResult r = emitModule(c, m, dnm);
  if (r.ok())
    printModule(os, r, helpers, prints, asserts);
  return r;
}

} // namespace agpu

#endif // AGPU_PRINT_MODULE_H
