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
  if (const std::optional<agpu::ElemType> e = heldTypeFor(v))
    cv.elem = *e;
  bindCarried(v, cv);

  if (cv.elem.isPointer())
    markBasePointer(idOf(v));
  return cv;
}

agpu::Result<agpu::CarriedValue>
AgpuEmitter::carriedFrom(Value v, const agpu::CarriedValue &like,
                         std::string_view where, std::string_view why) {
  using R = agpu::Result<agpu::CarriedValue>;
  const Operand from(body_.sym, idOf(v), (int64_t)like.regs.size());
  if (!from.ok())
    return R::no(declined(where, std::string(why)));
  agpu::CarriedValue out;
  out.elem = like.elem;

  // A carried variable holds an address. Elsewhere `addptr` binds the base
  // name and keeps the offset alongside, so an access is `base[off]`, but a
  // yield has nowhere to put the offset. This is the one place a pointer
  // becomes a value.
  const bool isPtr = like.elem.isPointer();
  for (std::size_t r = 0; r < like.regs.size(); ++r) {
    const am::Str name = inIrType(idOf(v), from.at((int64_t)r));
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
  return R::of(std::move(out));
}

agpu::Result<agpu::Carried>
AgpuEmitter::carriedOperands(Operation *term, const agpu::Carried &like,
                             std::string_view where) {
  using R = agpu::Result<agpu::Carried>;
  if (!term || term->getNumOperands() != like.size())
    return R::no(
        declined(where, "the yield does not match its carried values"));
  agpu::Carried out;
  for (std::size_t i = 0; i < like.size(); ++i) {
    const agpu::Result<agpu::CarriedValue> cv =
        carriedFrom(term->getOperand(i), like[i], where,
                    "a yielded value has no register names");
    if (!cv.ok())
      return R::no(cv.why);
    out.push_back(cv.value);
  }
  return R::of(std::move(out));
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
  if (const agpu::Decision d = walkBlock(region.front(), into); !d.ok())
    return d;
  const CurBlock here(*this, into);
  return atEnd();
}

agpu::Result<agpu::CarriedValue>
AgpuEmitter::carriedFor(Value v, const agpu::ValueNames &names) {
  using R = agpu::Result<agpu::CarriedValue>;
  const std::optional<agpu::ElemType> e = heldTypeFor(v);
  if (!e)
    return R::no(declined("scf.for", "a carried value has no element type"));
  agpu::CarriedValue cv;
  cv.elem = *e;
  for (const am::Str &n : names)
    cv.regs.push_back(n);
  return R::of(std::move(cv));
}

} // namespace mlir::triton::applegpu::bridge
