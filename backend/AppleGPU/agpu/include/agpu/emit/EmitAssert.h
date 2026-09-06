// EmitAssert.h - one `tt.assert` as `if (!cond) { record(); return; }`.
#ifndef AGPU_EMIT_ASSERT_H
#define AGPU_EMIT_ASSERT_H

#include "agpu/core/Names.h"
#include "agpu/emit/KernelAbi.h"
#include "agpu/emit/Prelude.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/AssertPlan.h"

#include <vector>

namespace agpu {

// Fires when any register the thread holds is false: `!r0 || !r1 || ...`.
inline msl::Expr *assertFiresWhen(msl::Context &c,
                                  const std::vector<msl::Str> &regs) {
  std::vector<msl::Expr *> failed;
  failed.reserve(regs.size());
  for (const msl::Str &r : regs)
    failed.push_back(c.unary(msl::UnOp::LNot, c.var(r)));
  return c.chain(msl::BinOp::LOr, failed);
}

// Record before the return, or the failure goes unreported.
inline msl::Block assertFailureBody(msl::Context &c, const AssertSite &site,
                                    const KernelNames &nm) {
  msl::Block body;
  body.push_back(c.exprStmt(
      c.call(helperName(Helper::AssertRecord),
             {c.var(nm.assertBuffer), c.lit(site.site),
              c.member(c.var(nm.threadgroupPos), msl::builtin::comp::X),
              c.member(c.var(nm.threadId), msl::builtin::comp::X)})));

  // `Continue` emits no return: a thread that skips a barrier hangs the kernel.
  if (site.halt == AssertHalt::Return)
    body.push_back(c.returnStmt());
  return body;
}

// `regs` are the condition's registers for this thread.
inline void emitAssert(msl::Context &c, msl::Block &body,
                       const AssertSite &site,
                       const std::vector<msl::Str> &regs,
                       const KernelNames &nm) {
  c.guardedInto(body, assertFiresWhen(c, regs), assertFailureBody(c, site, nm));
}

} // namespace agpu

#endif // AGPU_EMIT_ASSERT_H
