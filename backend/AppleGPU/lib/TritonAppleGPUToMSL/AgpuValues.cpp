// What the walk knows about a value: identity, coherence, pointer
// bookkeeping and the layout queries the handlers ask.
#include "AgpuEmitter.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

agpu::LaunchFacts launchFactsOf(Operation *scope) {
  agpu::LaunchFacts f;
  scope->walk([&](Operation *op) {
    if (auto p = dyn_cast<triton::AtomicPollOp>(op)) {
      if (!p.getTimeout())
        f.blockingPoll = true;
    } else if (isa<triton::GetNumProgramsOp>(op)) {
      f.readsGridExtent = true;
    } else if (isa<triton::AtomicRMWOp>(op)) {
      if (op->getParentOfType<scf::WhileOp>() ||
          op->getParentOfType<scf::ForOp>())
        f.atomicInLoop = true;
    }
  });
  return f;
}

static BlockArgument traceToKernelArg(Value v) {
  while (v) {
    if (BlockArgument arg = dyn_cast<BlockArgument>(v))
      return arg;
    Operation *def = v.getDefiningOp();
    if (auto ap = dyn_cast_or_null<triton::AddPtrOp>(def))
      v = ap.getPtr();
    else if (auto sp = dyn_cast_or_null<triton::SplatOp>(def))
      v = sp.getSrc();
    else if (auto bc = dyn_cast_or_null<triton::BitcastOp>(def))
      v = bc.getSrc();
    else
      return BlockArgument();
  }
  return BlockArgument();
}

static int loopDepthOf(Operation *op) {
  int depth = 0;
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (isa<LoopLikeOpInterface>(p))
      ++depth;
  return depth;
}

static bool ordersDeviceMemory(Operation *op) {
  if (auto b = dyn_cast<triton::gpu::BarrierOp>(op)) {
    const uint32_t bits = (uint32_t)b.getAddrSpace();
    return bits & ((uint32_t)triton::gpu::AddrSpace::GlobalRead |
                   (uint32_t)triton::gpu::AddrSpace::GlobalWrite);
  }
  return isa<mlir::gpu::BarrierOp>(op);
}

agpu::CoherenceFacts AgpuEmitter::coherenceFactsOf(triton::FuncOp func) {
  agpu::CoherenceFacts f;
  Block &entry = func.getBody().front();

  auto record = [&](Operation *op, Value ptr, agpu::AccessKind kind) {
    const BlockArgument base = traceToKernelArg(ptr);
    if (!base || base.getOwner() != &entry)
      return;
    agpu::BufferAccess a;
    a.buffer = (int)base.getArgNumber();
    a.kind = kind;
    a.loopDepth = loopDepthOf(op);
    a.isTensor = isa<RankedTensorType>(ptr.getType());
    f.accesses.push_back(a);
  };

  func.walk([&](Operation *op) {
    if (auto st = dyn_cast<triton::StoreOp>(op))
      record(st, st.getPtr(), agpu::AccessKind::Store);
    else if (auto ld = dyn_cast<triton::LoadOp>(op))
      record(ld, ld.getPtr(), agpu::AccessKind::Load);
    if (ordersDeviceMemory(op))
      f.hasDeviceBarrier = true;
  });

  return f;
}

bool AgpuEmitter::coherentBuffer(Value ptr) const {
  const BlockArgument arg = traceToKernelArg(ptr);
  return arg && coherentArgs_.count((int)arg.getArgNumber()) > 0;
}

void AgpuEmitter::markBasePointer(agpu::ValueId v) { body_.basePtrs.insert(v); }

void AgpuEmitter::inheritBasePointer(agpu::ValueId from, agpu::ValueId to) {
  if (body_.basePtrs.count(from))
    body_.basePtrs.insert(to);
}

std::vector<std::pair<am::Str, am::Type>>
AgpuEmitter::storageOf(agpu::ValueId v) const {
  std::vector<std::pair<am::Str, am::Type>> out;
  const auto add = [&out](const am::Str &n, am::Type t) {
    for (const auto &e : out)
      if (e.first == n)
        return;
    out.emplace_back(n, t);
  };

  const agpu::ElemType *elem = elemOf(v);
  const am::Type ty = elem ? agpu::mslTypeOf(*elem) : am::Context::i32();

  if (elem && elem->isPointer())
    for (int64_t r = 0, n = (int64_t)body_.sym.regCount(v); r < n; ++r) {
      const auto off = body_.offsetOf.find({v, r});
      if (off != body_.offsetOf.end() && off->second.owned)
        add(off->second.name, off->second.type);
    }

  for (const am::Str &n : body_.sym.ownedNamesOf(v))
    add(n, ty);
  return out;
}

void AgpuEmitter::inheritOffset(agpu::ValueId from, int64_t fromReg,
                                agpu::ValueId to, int64_t toReg) {
  const auto off = body_.offsetOf.find({from, fromReg});
  if (off == body_.offsetOf.end())
    return;
  PtrOffset copy = off->second;
  copy.owned = false;
  body_.offsetOf[{to, toReg}] = std::move(copy);
}

am::Expr *AgpuEmitter::addressAt(agpu::ValueId ptr, int64_t reg) {
  am::Context &mc = agpu_.context();
  const am::Str *base = body_.sym.regAt(ptr, (std::size_t)reg);
  if (!base)
    return nullptr;
  const auto off = body_.offsetOf.find({ptr, reg});
  if (off != body_.offsetOf.end())
    return mc.subscript(mc.var(*base), mc.var(off->second.name));

  if (body_.basePtrs.count(ptr))
    return mc.subscript(mc.var(*base), mc.lit(0));
  return nullptr;
}

bool AgpuEmitter::scaledRegisterDeltas(RankedTensorType rt, int64_t regs,
                                       llvm::ArrayRef<int64_t> scales,
                                       std::vector<int64_t> &deltas) {
  std::vector<int64_t> c0;
  for (int64_t r = 0; r < regs; ++r) {
    const std::optional<std::vector<int64_t>> c = registerCoordAt(rt, (int)r);
    if (!c || c->size() != scales.size())
      return false;
    if (r == 0) {
      c0 = *c;
      deltas.push_back(0);
      continue;
    }
    int64_t d = 0;
    for (std::size_t dim = 0; dim < c->size(); ++dim)
      d += scales[dim] * ((*c)[dim] - c0[dim]);
    deltas.push_back(d);
  }
  return true;
}

agpu::AddressSpread AgpuEmitter::spreadOf(Value ptr) {
  auto ptrTy =
      ptr ? dyn_cast<RankedTensorType>(ptr.getType()) : RankedTensorType();
  if (!ptrTy)
    return agpu::AddressSpread{0, 0, true};
  return spreadOf(ptrTy);
}

agpu::AddressSpread AgpuEmitter::spreadOf(RankedTensorType ty) {
  const LinearLayout ll = gpu::toLinearLayout(ty);
  MLIRContext *ctx = ty.getContext();
  return agpu::AddressSpread{freeBitsOf(ll, ctx, lldim::Lane),
                             freeBitsOf(ll, ctx, lldim::Warp), false};
}

am::Expr *AgpuEmitter::maskAt(const agpu::OpView &o, std::size_t maskIndex,
                              int64_t reg) {
  if (o.operands.size() <= maskIndex)
    return nullptr;
  const am::Str *m = body_.sym.regAt(o.operands[maskIndex], (std::size_t)reg);
  return m ? agpu_.context().var(*m) : nullptr;
}

agpu::CoordSource AgpuEmitter::coordSourceOf(RankedTensorType ty) {
  agpu::CoordSource cs;
  cs.hoist = &body_.hoist;
  const LinearLayout ll = gpu::toLinearLayout(ty);
  for (int d = 0; d < ty.getRank(); ++d)
    if (const std::optional<StringAttr> dim = outDimAt(ll, d))
      cs.dims.push_back(layoutSourceOf(ll, ty.getContext(), *dim).basis());
  return cs;
}

am::Expr *AgpuEmitter::coordOf(Value v, int reg, int axis) {
  auto rt = dyn_cast<RankedTensorType>(v.getType());
  if (!rt)
    return nullptr;
  const LinearLayout ll = gpu::toLinearLayout(rt);
  const std::optional<StringAttr> dim = outDimAt(ll, axis);
  if (!dim)
    return nullptr;
  const agpu::LayoutBasis lb =
      layoutSourceOf(ll, rt.getContext(), *dim).basis();
  return body_.hoist.coord(agpu_.context(), lb, reg);
}

} // namespace mlir::triton::applegpu::bridge
