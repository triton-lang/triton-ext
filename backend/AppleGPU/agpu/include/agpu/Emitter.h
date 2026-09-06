// Emitter.h - the entry point: assembles core/, plan/, emit/ and msl/.
#ifndef AGPU_EMITTER_H
#define AGPU_EMITTER_H

#include "agpu/emit/PrintModule.h"
#include "agpu/plan/Vestigial.h"

#include <ostream>
#include <string_view>
#include <vector>

namespace agpu {

// A whole translation unit under construction.
class Emitter {
public:
  DeclineLog declines;

  // Where the next decline happened, if the caller knows.
  DeclineSite site;

  msl::Context &context() { return ctx_; }

  // Ops handled by emitting nothing (`scf.yield`, `llvm.intr.assume`). Returns
  // `notMine()` for an op outside the table.
  Decision vestigial(std::string_view op) {
    const Decision d = vestigialDecision(op);
    declines.record(d, site);
    return d;
  }

  // f64 becomes float: Metal has no double. The kernel still runs, so the
  // loss is recorded.
  void noteIfNarrowed(ElemType e) {
    if (narrowsSilently(e))
      declines.note(
          Decision::declined("type", "f64 has no Metal type; computing in f32"),
          site);
  }

  void addKernel(KernelFacts f, BodyFn build, KernelNames names = {}) {
    KernelUnit k;
    k.facts = std::move(f);
    k.buildBody = std::move(build);
    k.names = std::move(names);
    kernels_.push_back(std::move(k));
  }

  // In the order MSL requires: includes, then kernels.
  ModuleResult print(std::ostream &os) {
    ModuleFacts m;
    m.kernels = std::move(kernels_);
    return emitModule(ctx_, m, os);
  }

private:
  msl::Context ctx_;
  std::vector<KernelUnit> kernels_;
};

} // namespace agpu

#endif // AGPU_EMITTER_H
