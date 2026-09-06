// Handlers that introduce a value: grid query, splat, constant, range.
#include "AgpuEmitter.h"
#include "AgpuOpTables.h"

#include "agpu/emit/EmitGridBuiltins.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

agpu::Decision AgpuEmitter::emitConstantOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const auto kv = constantFor_.find(o.results.empty() ? 0 : o.results[0]);
  if (o.results.size() != 1 || kv == constantFor_.end() || kv->second.empty())
    return declined("arith.constant", "constant of an unrepresentable type");

  const std::vector<ConstantValue> &vals = kv->second;
  const agpu::ElemType *elemP = elemOf(o.results[0]);
  if (!elemP)
    return declined(o.name, "result type was never recorded");
  const agpu::ElemType elem = *elemP;
  const am::Type ty = agpu::mslTypeOf(elem);

  auto literal = [&](const ConstantValue &v) {
    return v.isFloat ? mc.litF(v.f, ty) : mc.lit(v.i, ty);
  };

  if (vals.size() == 1) {
    const am::Str n = nameFor('k', o.results[0], 0);
    cur_->push_back(mc.declStmt(ty, n, literal(vals[0])));
    body_.sym.bindScalar(o.results[0], n);
    return agpu::Decision::emitted();
  }

  // Pick each literal by the element the register holds: the attribute is
  // in row-major element order, the registers are in the layout's order.
  const Value res = mlirValueOf(o.results[0]);
  if (!res)
    return declined(o.name, "result value was never recorded");
  const auto resTy = cast<RankedTensorType>(res.getType());
  return emitPerRegister(o, registerCount(resTy), elem, 'k', [&](int64_t r) {
    RegValue v;
    const std::optional<int64_t> flat = flatElemAt(resTy, (int)r);
    if (flat && *flat < (int64_t)vals.size())
      v.value = literal(vals[(std::size_t)*flat]);
    return v;
  });
}

agpu::Decision AgpuEmitter::emitSplatOp(const agpu::OpView &o) {
  const Ready ready = readyForCounted(o, 0, 1, 1, "the source was never bound");
  if (!ready.ok())
    return ready.why;
  body_.sym.bindScalar(o.results[0], ready.ops[0].at(0));

  // A pointer register is a name and an offset, so a splat carries
  // both.
  for (int64_t r = 0, n = registersHeldBy(o.results[0]); r < n; ++r)
    inheritOffset(o.operands[0], 0, o.results[0], r);

  inheritBasePointer(o.operands[0], o.results[0]);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitGridQueryOp(const agpu::OpView &o) {
  const auto q = o.name == kGetProgramId ? agpu::GridQuery::ProgramId
                                         : agpu::GridQuery::NumPrograms;
  const am::Str n = nameFor('g', o.results[0], 0);
  am::Stmt *s = agpu::emitGridQuery(agpu_.context(), q, (int)o.intAt(0), n);
  if (!s)
    return declined(o.name, "grid axis above 2");
  cur_->push_back(s);
  body_.sym.bindScalar(o.results[0], n);
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerValueHandlers() {
  table_.add("grid", agpu::forOps({kGetProgramId, "tt.get_num_programs"},
                                  [this](const agpu::OpView &o) {
                                    return emitGridQueryOp(o);
                                  }));

  table_.add("splat", agpu::forOps({"tt.splat"}, [this](const agpu::OpView &o) {
               return emitSplatOp(o);
             }));

  table_.add("constant",
             agpu::forOps({"arith.constant"}, [this](const agpu::OpView &o) {
               return emitConstantOp(o);
             }));
}

agpu::Decision AgpuEmitter::emitMakeRangeOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const Value res = mlirValueOf(o.results[0]);
  if (!res)
    return declined(o.name, "result value was never recorded");
  const int64_t regs = registerCount(cast<RankedTensorType>(res.getType()));
  agpu::ValueNames names;
  for (int64_t r = 0; r < regs; ++r) {
    am::Expr *coord = coordOf(res, (int)r);
    if (!coord)
      return declined("tt.make_range", "layout rank not handled yet");
    const am::Str n = nameFor('r', o.results[0], r);
    am::Expr *val = o.intAt(0) == 0
                        ? coord
                        : mc.binary(am::BinOp::Add, mc.lit(o.intAt(0)), coord);
    cur_->push_back(mc.declStmt(agpu::mslTypeOf(agpu::i32()), n, val));
    names.push_back(n);
  }
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerRangeHandler() {
  table_.add("range",
             agpu::forOps({"tt.make_range"}, [this](const agpu::OpView &o) {
               return emitMakeRangeOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
