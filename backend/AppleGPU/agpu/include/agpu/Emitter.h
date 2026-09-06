// Emitter.h - the entry point: assembles core/, plan/, emit/ and msl/.
#ifndef AGPU_EMITTER_H
#define AGPU_EMITTER_H

#include "agpu/core/Gates.h"
#include "agpu/emit/EmitDot.h"
#include "agpu/emit/EmitRebind.h"
#include "agpu/emit/PrintModule.h"
#include "agpu/plan/Vestigial.h"

#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace agpu {

// A whole translation unit under construction.
class Emitter {
public:
  explicit Emitter(Bytes budget = Bytes(kTGResidentBudgetBytes))
      : budget_(budget) {}

  HelperSet helpers;

  // Every print site, in walk order. A caller that re-walks a body must clear
  // this first or sites get double-numbered.
  PrintPlan prints;

  // Every assert site, on the same terms as `prints`.
  AssertPlan asserts;

  DeclineLog declines;

  // Where the next decline happened, if the caller knows.
  DeclineSite site;

  // Diagnostics only: no gate changes emitted code or the cache key.
  GateSet gates;

  msl::Context &context() { return ctx_; }

  // Lets a caller ask what a shape will cost before emitting it.
  Plan planFor(const DotFacts &f) const { return planDot(f, budget_); }

  Decision dot(msl::Block &body, const DotFacts &f, const DotInputs &in) {
    const Plan p = planFor(f);
    // cNeed exists only once the plan is chosen, and the reservation must
    // precede print()'s pool.plan().
    if (p.kind != Plan::Kind::Unsupported)
      pool.scratch("dot", p.pool.cNeed);
    const Decision d = emitDot(ctx_, body, p, in);
    declines.record(d, site);
    return d;
  }

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

  // splat, expand_dims, broadcast, split and trans are pure renamings, so
  // nothing is emitted. Declined when a result register has no source.
  Decision rebindTo(const Rebind &r, const std::vector<msl::Str> &src,
                    std::vector<msl::Str> &out) {
    const Decision d = rebindDecision(r);
    declines.record(d, site);
    if (!d.ok())
      return d;
    out = aliasRebind(r, src);
    // Empty names after a complete plan means a short source list: caller
    // error.
    if (!allNamed(out))
      return Decision::failed();
    return d;
  }

  // What the walk has asked the threadgroup pool for.
  PoolRequests pool;

  // ── the units ──────────────────────────────────────────────────────────

  void addKernel(KernelFacts f, BodyFn build, KernelNames names = {}) {
    KernelUnit k;
    k.facts = std::move(f);
    k.buildBody = std::move(build);
    k.names = std::move(names);
    kernels_.push_back(std::move(k));
  }

  void addDeviceFn(DeviceFnFacts f, std::vector<msl::Str> paramNames,
                   msl::Block body) {
    DeviceFnUnit u;
    u.facts = std::move(f);
    u.paramNames = std::move(paramNames);
    u.body = std::move(body);
    deviceFns_.push_back(std::move(u));
  }

  // In the order MSL requires: includes, helpers, return structs, prototypes,
  // device bodies, kernels. Refuses the module if the pool is over budget.
  ModuleResult print(std::ostream &os) {
    ModuleFacts m;
    m.deviceFns = std::move(deviceFns_);
    m.kernels = std::move(kernels_);
    m.functionPools = {pool.plan()};
    return emitModule(ctx_, m, helpers, prints, asserts, os);
  }

  // Kept off `print`'s stream so the summary does not land in the .metal file.
  // Prints nothing when nothing declined.
  void printDeclineSummary(std::ostream &os) const {
    if (!gates.on(Gate::LogReject))
      return;
    declines.printSummary(os);
  }

private:
  msl::Context ctx_;
  Bytes budget_;
  std::vector<DeviceFnUnit> deviceFns_;
  std::vector<KernelUnit> kernels_;
};

} // namespace agpu

#endif // AGPU_EMITTER_H
