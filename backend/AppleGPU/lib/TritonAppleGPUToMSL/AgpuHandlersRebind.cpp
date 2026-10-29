// Shape op handlers: expand_dims, broadcast. Both re-describe which register
// holds which coordinate, so neither emits anything.
#include "AgpuEmitter.h"
#include "AgpuOpTables.h"

#include "agpu/plan/RebindPlan.h"

namespace mlir::triton::applegpu::bridge {

agpu::Decision AgpuEmitter::emitRebindOp(const agpu::OpView &o) {
  if (o.operands.size() != 1 || o.results.size() != 1)
    return declined(o.name, "unexpected operand or result count");

  const Value src = mlirValueOf(o.operands[0]);
  if (!src)
    return declined(o.name, "operand value was never recorded");
  const Value res = mlirValueOf(o.results[0]);
  if (!res)
    return declined(o.name, "result value was never recorded");
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  auto resTy = dyn_cast<RankedTensorType>(res.getType());
  if (!srcTy || !resTy)
    return declined(o.name, "an operand is not a ranked tensor");

  // plan/RebindPlan.h decides which source register feeds each result
  // register; this layer supplies the coordinate sets.
  const int64_t srcRegs = registerCount(srcTy);
  std::vector<agpu::RegCoord> srcCoords;
  srcCoords.reserve((std::size_t)srcRegs);
  for (int64_t r = 0; r < srcRegs; ++r) {
    const std::optional<int64_t> e = flatElemAt(srcTy, (int)r);
    srcCoords.push_back(e ? agpu::RegCoord{(int32_t)*e}
                          : agpu::RegCoord{-1 - (int32_t)r});
  }

  const int64_t resRegs = registerCount(resTy);
  std::vector<agpu::RegCoord> resCoords;
  resCoords.reserve((std::size_t)resRegs);
  for (int64_t r = 0; r < resRegs; ++r) {
    const std::optional<int64_t> want = elemThroughRebind(srcTy, resTy, (int)r);
    if (!want)
      return declined(o.name, "cannot map a result register to a "
                              "source element");
    resCoords.push_back({(int32_t)*want});
  }

  const agpu::Rebind plan =
      agpu::rebind(resCoords, agpu::indexByCoord(srcCoords),
                   [](const agpu::RegCoord &rc, agpu::RegCoord &want) {
                     want = rc;
                     return true;
                   });

  if (!plan.complete())
    return declined(o.name, "a result register reads another thread's lane");

  const Ready ready =
      readyForCounted(o, 0, 1, srcRegs, "a source register was never bound");
  if (!ready.ok())
    return ready.why;

  agpu::ValueNames names;
  for (int64_t r = 0; r < resRegs; ++r) {
    const int from = plan.from[(std::size_t)r];
    names.push_back(ready.ops[0].at(from));

    // A pointer register is a name and an offset.
    inheritOffset(o.operands[0], (int64_t)from, o.results[0], r);
  }
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerRebindHandler() {
  table_.add("rebind", agpu::forOps({kExpandDims, kBroadcast},
                                    [this](const agpu::OpView &o) {
                                      return emitRebindOp(o);
                                    }));
}

} // namespace mlir::triton::applegpu::bridge
