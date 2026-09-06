// EmitModule.h - every function of a translation unit, as AST. PrintModule.h
// turns the result into text.
//
// MSL forbids declaring threadgroup memory outside a kernel, so each kernel
// declares one pool buffer, sized to the larger of what its own body used
// and what was requested before the bodies were built (device functions,
// direct callers).
#ifndef AGPU_EMIT_MODULE_H
#define AGPU_EMIT_MODULE_H

#include "agpu/emit/EmitDeviceFn.h"
#include "agpu/emit/EmitKernel.h"
#include "agpu/emit/Prelude.h"
#include "agpu/plan/PoolPlan.h"

#include <vector>

namespace agpu {

// One device function, ready to emit.
struct DeviceFnUnit {
  DeviceFnFacts facts;
  std::vector<msl::Str> paramNames;
  msl::Block body;
};

// One kernel, ready to emit.
struct KernelUnit {
  KernelFacts facts;
  BodyFn buildBody;
  KernelNames names;
};

struct ModuleFacts {
  std::vector<DeviceFnUnit> deviceFns;
  std::vector<KernelUnit> kernels;

  // An over-budget module names the operation responsible.
  std::vector<FunctionPool> functionPools;
};

struct ModuleResult {
  std::vector<msl::Function *> protos;
  std::vector<msl::Function *> deviceFns;
  std::vector<msl::Stmt *> retStructs;
  std::vector<KernelResult> kernels;

  FunctionPool pool;
  int64_t poolBytes = 0;
  Decision decision = Decision::failed();

  bool ok() const { return decision.ok(); }
};

inline ModuleResult emitModule(msl::Context &c, ModuleFacts &m,
                               const DeviceFnNames &dnm = {}) {
  ModuleResult r;
  r.pool = planModulePool(m.functionPools);
  // Scratch only: live buffers declare themselves at their own sites, so
  // `total()` would reserve their bytes twice.
  r.poolBytes = r.pool.scratch.count();

  // A threadgroup declaration past the hardware limit compiles and links, then
  // takes down MTLCompilerService at pipeline-state creation. Judged again
  // after the bodies are built.
  r.decision = poolDecision(r.pool);
  if (!r.decision.ok())
    return r;

  std::vector<DeviceFnAbi> abis;
  abis.reserve(m.deviceFns.size());
  for (DeviceFnUnit &u : m.deviceFns) {
    DeviceFnAbi abi = planDeviceFn(u.facts);
    r.decision = deviceFnDecision(abi);
    if (!r.decision.ok())
      return r;
    abis.push_back(std::move(abi));
  }

  // Bodies first: building one may add to the helper set, which must be
  // printed before anything that calls a helper.
  for (std::size_t i = 0; i < m.deviceFns.size(); ++i)
    r.deviceFns.push_back(emitDeviceFn(c, m.deviceFns[i].facts, abis[i],
                                       m.deviceFns[i].paramNames,
                                       std::move(m.deviceFns[i].body), dnm));

  for (KernelUnit &k : m.kernels) {
    k.facts.poolBytes = std::max(k.facts.poolBytes, r.poolBytes);
    KernelResult kr = emitKernel(c, k.facts, k.buildBody, k.names);
    if (!kr.ok()) {
      r.decision = kr.decision;
      return r;
    }
    if (Bytes(kr.poolBytes) > r.pool.scratch) {
      r.pool.scratch = Bytes(kr.poolBytes);
      r.pool.driver = "kernel body";
    }
    r.kernels.push_back(kr);
  }
  r.poolBytes = r.pool.scratch.count();
  r.decision = poolDecision(r.pool);
  if (!r.decision.ok())
    return r;

  for (std::size_t i = 0; i < m.deviceFns.size(); ++i)
    if (msl::Stmt *s = emitRetStruct(c, m.deviceFns[i].facts, abis[i], dnm))
      r.retStructs.push_back(s);

  for (std::size_t i = 0; i < m.deviceFns.size(); ++i)
    r.protos.push_back(emitDeviceProto(c, m.deviceFns[i].facts, abis[i], dnm));

  r.decision = Decision::emitted();
  return r;
}

} // namespace agpu

#endif // AGPU_EMIT_MODULE_H
