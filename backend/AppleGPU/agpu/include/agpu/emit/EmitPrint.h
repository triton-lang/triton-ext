// EmitPrint.h - one `tt.print`, emitted as appends to the print buffer.
//
// Metal has no printf in a compute kernel, so the host does the formatting.
// Buffer layout lives in `plan/PrintPlan.h`.
#ifndef AGPU_EMIT_PRINT_H
#define AGPU_EMIT_PRINT_H

#include "agpu/core/Names.h"
#include "agpu/emit/KernelAbi.h"
#include "agpu/emit/Prelude.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/PrintPlan.h"

#include <functional>

namespace agpu {

// A float travels as its bits. Narrower floats widen to f32 before the
// bitcast so the host sees the fields where it expects them.
inline msl::Expr *printWordOf(msl::Context &c, msl::Expr *value, ElemType e) {
  if (printTypeOf(e) != PrintType::Float)
    return c.cast(msl::Type::scalar(msl::Scalar::U32), value);
  msl::Expr *wide =
      e.bits == 32 ? value : c.cast(msl::Type::scalar(msl::Scalar::F32), value);
  return c.bitcast(msl::Type::scalar(msl::Scalar::U32), wide);
}

struct PrintIdentity {
  msl::Expr *pid = nullptr;
  msl::Expr *tid = nullptr;
};

inline PrintIdentity printIdentity(msl::Context &c, const KernelNames &nm) {
  PrintIdentity id;
  id.pid = c.member(c.var(nm.threadgroupPos), msl::builtin::comp::X);
  id.tid = c.member(c.var(nm.threadId), msl::builtin::comp::X);
  return id;
}

// Which element of the print one value is, given operand and register index.
using PrintIndexFn =
    std::function<msl::Expr *(std::size_t op, std::size_t reg)>;

inline msl::Stmt *emitPrintAppend(msl::Context &c, const KernelNames &nm,
                                  const PrintIdentity &id, std::int32_t site,
                                  msl::Expr *index, PrintType type,
                                  msl::Expr *word, std::int32_t operand) {
  return c.exprStmt(
      c.call(helperName(Helper::PrintAppend),
             {c.var(nm.printBuffer), c.lit(site), id.pid, id.tid, index,
              c.lit(static_cast<std::int64_t>(type)), word, c.lit(operand)}));
}

// One append per value, in operand order then register order. A print with no
// operands still emits one append.
inline void emitPrint(msl::Context &c, msl::Block &body, const PrintSite &site,
                      const PrintIndexFn &indexOf, const KernelNames &nm) {
  const PrintIdentity id = printIdentity(c, nm);

  if (site.operands.empty()) {
    body.push_back(emitPrintAppend(c, nm, id, site.site, c.lit(0),
                                   PrintType::UInt, c.lit(0), 0));
    return;
  }

  for (std::size_t o = 0; o < site.operands.size(); ++o) {
    const PrintOperand &op = site.operands[o];
    for (std::size_t r = 0; r < op.regs.size(); ++r)
      body.push_back(emitPrintAppend(
          c, nm, id, site.site, indexOf(o, r), op.type(),
          printWordOf(c, c.var(op.regs[r]), op.elem), (std::int32_t)o));
  }
}

} // namespace agpu

#endif // AGPU_EMIT_PRINT_H
