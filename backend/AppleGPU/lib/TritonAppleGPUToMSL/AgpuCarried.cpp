// Carried values: the registers an scf region hands across its boundary, and
// the walk that re-enters a region with them bound.
#include "AgpuEmitter.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;
int64_t AgpuEmitter::registersHeldByType(Type t) const {
  auto rt = dyn_cast<RankedTensorType>(t);
  return rt ? registerCount(rt) : 1;
}

agpu::CarriedValue AgpuEmitter::carriedFresh(Value v) {
  agpu::CarriedValue cv;
  cv.regs = freshNames(v, registersHeldByType(v.getType()));
  // Uses the held type: a carried pointer needs the address's own type,
  // which `elemTypeOf`'s pointee does not give.
  if (const std::optional<agpu::ElemType> e = heldTypeOf(v.getType()))
    cv.elem = *e;
  bindCarried(v, cv);

  if (cv.elem.isPointer())
    markBasePointer(idOf(v));
  return cv;
}

agpu::Decision AgpuEmitter::carriedFrom(Value v, const agpu::CarriedValue &like,
                                        agpu::CarriedValue &out,
                                        std::string_view where,
                                        std::string_view why) {
  const Operand from(body_.sym, idOf(v), (int64_t)like.regs.size());
  if (!from.ok())
    return declined(where, std::string(why));
  out.elem = like.elem;

  // A carried variable holds an address. Elsewhere `addptr` binds the base
  // name and keeps the offset alongside, so an access is `base[off]`, but a
  // yield has nowhere to put the offset. This is the one place a pointer
  // becomes a value.
  const bool isPtr = like.elem.isPointer();
  for (std::size_t r = 0; r < like.regs.size(); ++r) {
    const am::Str narrowed = inIrType(idOf(v), (int64_t)r);
    const am::Str name = narrowed.empty() ? from.at((int64_t)r) : narrowed;
    const auto off = isPtr ? body_.offsetOf.find({idOf(v), (int64_t)r})
                           : body_.offsetOf.end();
    if (off == body_.offsetOf.end()) {
      out.regs.push_back(name);
      continue;
    }
    const am::Str addr = "pa" + std::to_string(idOf(v)) + "_" +
                         std::to_string(r) + "_" +
                         std::to_string(body_.tempSeq++);
    cur_->push_back(agpu_.context().declStmt(
        agpu::mslTypeOf(like.elem), addr,
        agpu_.context().binary(am::BinOp::Add, agpu_.context().var(name),
                               agpu_.context().var(off->second.name))));
    out.regs.push_back(addr);
  }
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::carriedOperands(Operation *term,
                                            const agpu::Carried &like,
                                            agpu::Carried &out,
                                            std::string_view where) {
  if (!term || term->getNumOperands() != like.size())
    return declined(where, "the yield does not match its carried values");
  for (std::size_t i = 0; i < like.size(); ++i) {
    agpu::CarriedValue cv;
    if (const agpu::Decision d =
            carriedFrom(term->getOperand(i), like[i], cv, where,
                        "a yielded value has no register names");
        !d.ok())
      return d;
    out.push_back(cv);
  }
  return agpu::Decision::emitted();
}

void AgpuEmitter::bindCarried(Value v, const agpu::CarriedValue &cv) {
  body_.sym.bindRegs(idOf(v), cv.regs);
  valueFor_[idOf(v)] = v;
  elemFor_[idOf(v)] = cv.elem;
}

agpu::Decision AgpuEmitter::walkRegion(Region &region, am::Block &into) {
  return walkRegion(region, into, [] { return agpu::Decision::emitted(); });
}

agpu::Decision
AgpuEmitter::walkRegion(Region &region, am::Block &into,
                        const llvm::function_ref<agpu::Decision()> &atEnd) {
  if (!walkBlock(region.front(), into).ok())
    return agpu::Decision::failed();
  const CurBlock here(*this, into);
  return atEnd();
}

agpu::Decision AgpuEmitter::carriedFor(Value v, agpu::Carried &out,
                                       const agpu::ValueNames &names) {
  const std::optional<agpu::ElemType> e = heldTypeOf(v.getType());
  if (!e)
    return declined("scf.for", "a carried value has no element type");
  agpu::CarriedValue cv;
  cv.elem = *e;
  for (const am::Str &n : names)
    cv.regs.push_back(n);
  out.push_back(std::move(cv));
  return agpu::Decision::emitted();
}

} // namespace mlir::triton::applegpu::bridge
