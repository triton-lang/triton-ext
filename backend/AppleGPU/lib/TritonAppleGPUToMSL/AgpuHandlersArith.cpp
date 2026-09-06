// Elementwise and comparison handlers.
#include "AgpuEmitter.h"
#include "AgpuOpTables.h"

#include "agpu/emit/EmitElementwise.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

agpu::Decision AgpuEmitter::emitCompareOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  agpu::EwOp ew;
  if (!cmpOpFor(o.name, o.intAt(0), ew))
    return declined(o.name, "unhandled comparison predicate");

  const Ready ready = readyFor(o, 2);
  if (!ready.ok())
    return ready.why;
  const agpu::ElemType *operandP = elemOf(o.operands[0]);
  if (!operandP)
    return declined(o.name, "operand type was never recorded");

  const Operand &a = ready[0];
  const Operand &b = ready[1];

  const agpu::EwTypes t = agpu::typesFor(ew, *operandP);
  return emitPerRegister(o, ready.regs, t.result, 'c', [&](int64_t r) {
    RegValue v;
    v.value = agpu::ewExpr(mc, ew, *operandP, mc.var(a.at(r)), mc.var(b.at(r)));
    return v;
  });
}

agpu::Decision AgpuEmitter::emitElementwiseOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  agpu::EwOp ew;
  if (!ewOpFor(o.name, ew))
    return agpu::Decision::notMine();

  const Ready ready = readyFor(o, 2);
  if (!ready.ok())
    return ready.why;
  const Operand &a = ready[0];
  const Operand &b = ready[1];

  const agpu::ElemType operand =
      elemOf(o.operands[0]) ? *elemOf(o.operands[0]) : ready.elem;
  const agpu::EwTypes t = agpu::typesFor(ew, operand);

  return emitPerRegister(o, ready.regs, t.result, 'e', [&](int64_t r) {
    RegValue v;
    v.value = agpu::ewExpr(mc, ew, operand, mc.var(a.at(r)), mc.var(b.at(r)));
    return v;
  });
}

void AgpuEmitter::registerArithHandlers() {
  table_.add("elementwise",
             agpu::forOps(ewOpNames(), [this](const agpu::OpView &o) {
               return emitElementwiseOp(o);
             }));

  table_.add("compare",
             agpu::forOps(cmpOpNames(), [this](const agpu::OpView &o) {
               return emitCompareOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
