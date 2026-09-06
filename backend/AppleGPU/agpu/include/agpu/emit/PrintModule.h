// PrintModule.h - an emitted module as MSL text, in the order MSL requires.
#ifndef AGPU_PRINT_MODULE_H
#define AGPU_PRINT_MODULE_H

#include "agpu/emit/EmitModule.h"
#include "agpu/msl/Printer.h"

#include <ostream>

namespace agpu {

inline void printModuleHeader(std::ostream &os) {
  os << "#include <metal_stdlib>\n"
     << "using namespace metal;\n\n";
}

inline void printModuleBody(std::ostream &os, const ModuleResult &r) {
  msl::Printer p(os);
  for (const KernelResult &kr : r.kernels) {
    msl::Block b{kr.fn};
    p.printBlock(b);
  }
}

inline void printModule(std::ostream &os, const ModuleResult &r) {
  printModuleHeader(os);
  printModuleBody(os, r);
}

inline ModuleResult emitModule(msl::Context &c, ModuleFacts &m,
                               std::ostream &os) {
  ModuleResult r = emitModule(c, m);
  if (r.ok())
    printModule(os, r);
  return r;
}

} // namespace agpu

#endif // AGPU_PRINT_MODULE_H
