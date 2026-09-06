// Device access handlers: addptr, load and store. Pointers are never
// materialised; an access is base[off].
#include "AgpuEmitter.h"
#include "AgpuOpTables.h"

#include "agpu/emit/EmitMove.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

// Sets `bases` and `runtime`; leaves them empty for a value with no layout.
std::vector<agpu::LayoutBasis> AgpuEmitter::layoutDimsOf(Value v) {
  const auto rt =
      v ? dyn_cast<RankedTensorType>(v.getType()) : RankedTensorType();
  if (!rt)
    return {};
  const agpu::CoordSource cs = coordSourceOf(rt);
  if ((int)cs.dims.size() != rt.getRank())
    return {};
  return cs.dims;
}

PtrOffset AgpuEmitter::offsetSum(agpu::ValueId basePtr, int64_t reg,
                                 const am::Str &added) {
  const auto prior = body_.offsetOf.find({basePtr, reg});
  if (prior == body_.offsetOf.end())
    return PtrOffset{added, am::Context::i32(), false};

  const am::Str name = "off" + std::to_string(basePtr) + "_" +
                       std::to_string(reg) + "_" +
                       std::to_string(body_.tempSeq++);
  cur_->push_back(agpu_.context().declStmt(
      am::Context::i32(), name,
      agpu_.context().binary(am::BinOp::Add,
                             agpu_.context().var(prior->second.name),
                             agpu_.context().var(added))));
  return PtrOffset{name, am::Context::i32(), true};
}

agpu::Decision AgpuEmitter::emitLoad(const agpu::OpView &o,
                                     std::size_t maskIndex) {
  const Ready ready = readyFor(o, 1);
  if (!ready.ok())
    return ready.why;

  const std::size_t otherIndex = maskIndex + 1;
  const bool hasOther = o.operands.size() > otherIndex;
  Operand other(body_.sym, hasOther ? o.operands[otherIndex] : 0,
                hasOther ? ready.regs : 0);
  if (hasOther && !other.ok())
    return declined(o.name, "an `other` register has no name");

  // Check every register up front: declining halfway would leave bound names
  // undefined.
  for (int64_t r = 0; r < ready.regs; ++r)
    if (!addressAt(o.operands[0], r))
      return declined(o.name, "cannot build register " + std::to_string(r));

  agpu::MoveFacts f;
  f.regCount = ready.regs;
  f.hasMask = o.operands.size() > maskIndex;
  f.hasOther = hasOther;

  agpu::MoveSite site;
  site.elem = [this, ptr = o.operands[0]](int64_t r) {
    return addressAt(ptr, r);
  };
  if (f.hasMask)
    site.guard = [this, &o, maskIndex](int64_t r) {
      return maskAt(o, maskIndex, r);
    };
  if (hasOther)
    site.other = [this, &other](int64_t r) {
      return agpu_.context().var(other.at(r));
    };
  agpu::ValueNames names;
  for (int64_t r = 0; r < ready.regs; ++r) {
    names.push_back(nameFor('l', o.results[0], r));
    site.values.push_back(names.back());
  }

  agpu::emitMove(agpu_.context(), *cur_, f, site, ready.elem);
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitStore(const agpu::OpView &o,
                                      std::size_t maskIndex) {
  const agpu::ValueId ptr = o.operands[0];
  // Count registers from the pointer's layout: registers can share a base
  // name via broadcast and counting names would under-store.
  const Value ptrV = mlirValueOf(ptr);
  const int64_t regs = registersHeldBy(ptr);
  const Ready ready =
      readyForCounted(o, 1, 2, regs, "stored value has no register names");
  if (!ready.ok())
    return ready.why;
  const Operand &val = ready.ops[1];

  // A layout not distributed over some lane/warp bit gives several threads the
  // same address, and electing one writer is not in this stage.
  if (addressesAreRedundant(ptrV))
    return declined(o.name, "several threads address one element");

  for (int64_t r = 0; r < regs; ++r)
    if (!addressAt(ptr, r))
      return declined(o.name, "pointer has no recorded offset");

  const agpu::ElemType *ve = elemOf(o.operands[1]);
  agpu::MoveFacts f;
  f.regCount = regs;
  f.isStore = true;
  f.hasMask = o.operands.size() > maskIndex;

  agpu::MoveSite site;
  site.elem = [this, ptr](int64_t r) { return addressAt(ptr, r); };
  if (f.hasMask)
    site.guard = [this, &o, maskIndex](int64_t r) {
      return maskAt(o, maskIndex, r);
    };
  for (int64_t r = 0; r < regs; ++r)
    site.values.push_back(val.at(r));

  agpu::emitMove(agpu_.context(), *cur_, f, site, ve ? *ve : agpu::f32());

  if (!o.results.empty())
    body_.sym.bindDataless(o.results[0]);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitAddPtrOp(const agpu::OpView &o) {
  if (o.operands.size() != 2 || o.results.size() != 1)
    return declined("tt.addptr", "unexpected operand or result count");

  const Ready ready = readyFor(o, 2);
  if (!ready.ok())
    return ready.why;

  agpu::ValueNames names;
  for (int64_t r = 0; r < ready.regs; ++r) {
    names.push_back(ready.ops[0].at(r));

    // Offset accumulates along the chain: a 2-D address is
    // addptr(addptr(p, row), col).
    body_.offsetOf[std::pair<agpu::ValueId, int64_t>{o.results[0], r}] =
        offsetSum(o.operands[0], r, ready.ops[1].at(r));
  }
  body_.sym.bindRegs(o.results[0], std::move(names));

  return agpu::Decision::emitted();
}

void AgpuEmitter::registerAddPtrHandler() {
  // addptr: base + offset, per register.
  table_.add("addptr",
             agpu::forOps({"tt.addptr"}, [this](const agpu::OpView &o) {
               return emitAddPtrOp(o);
             }));
}

agpu::Decision AgpuEmitter::emitMemoryOp(const agpu::OpView &o) {
  const bool isLoad = o.name == kLoad;
  const std::size_t maskAt = isLoad ? 1u : 2u;

  if (!isLoad && o.operands.size() > maskAt + 1)
    return declined(o.name, "store with an unexpected extra operand");

  return isLoad ? emitLoad(o, maskAt) : emitStore(o, maskAt);
}

void AgpuEmitter::registerMemoryHandler() {
  // load/store dispatch here; mask position differs (tt.load %p, %m vs
  // tt.store %p, %v, %m).
  table_.add("memory",
             agpu::forOps({kLoad, "tt.store"}, [this](const agpu::OpView &o) {
               return emitMemoryOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
