// EmitModule.h - every function of a translation unit, as AST. PrintModule.h
// turns the result into text.
#ifndef AGPU_EMIT_MODULE_H
#define AGPU_EMIT_MODULE_H

#include "agpu/emit/EmitKernel.h"

#include <vector>

namespace agpu {

// One kernel, ready to emit.
struct KernelUnit {
  KernelFacts facts;
  BodyFn buildBody;
  KernelNames names;
};

struct ModuleFacts {
  std::vector<KernelUnit> kernels;
};

struct ModuleResult {
  std::vector<KernelResult> kernels;

  Decision decision = Decision::failed();

  bool ok() const { return decision.ok(); }
};

inline ModuleResult emitModule(msl::Context &c, ModuleFacts &m) {
  ModuleResult r;

  for (KernelUnit &k : m.kernels) {
    KernelResult kr = emitKernel(c, k.facts, k.buildBody, k.names);
    if (!kr.ok()) {
      r.decision = kr.decision;
      return r;
    }
    r.kernels.push_back(kr);
  }

  r.decision = Decision::emitted();
  return r;
}

} // namespace agpu

#endif // AGPU_EMIT_MODULE_H
