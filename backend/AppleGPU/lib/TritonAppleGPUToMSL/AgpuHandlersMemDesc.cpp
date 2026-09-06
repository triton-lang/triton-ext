// Threadgroup buffer handlers: local_alloc, local_load, memdesc_subslice/index.
#include "AgpuEmitter.h"

#include "agpu/emit/EmitMemDesc.h"
#include "agpu/emit/EmitPoison.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

// Strides follow the shared encoding's order: fastest-varying dimension gets
// stride 1. Declines a real swizzle (vec/perPhase/maxPhase past 1), which
// strides can't express.
static std::optional<agpu::TileView> tileViewOfMemDesc(gpu::MemDescType mt) {
  auto shared = dyn_cast<gpu::SwizzledSharedEncodingAttr>(mt.getEncoding());
  if (!shared)
    return std::nullopt;
  if (shared.getVec() != 1 || shared.getPerPhase() != 1 ||
      shared.getMaxPhase() != 1)
    return std::nullopt;
  const auto order = shared.getOrder();
  const int rank = mt.getRank();
  if ((int)order.size() != rank)
    return std::nullopt;

  agpu::TileView::Coord extent(mt.getShape().begin(), mt.getShape().end());
  agpu::TileView::Coord strides(rank, 1);
  int64_t acc = 1;
  for (int i = 0; i < rank; ++i) {
    strides[order[i]] = acc;
    acc *= extent[order[i]];
  }
  return agpu::TileView(std::move(extent), std::move(strides));
}

agpu::Decision AgpuEmitter::emitLocalAlloc(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const Value res = mlirValueOf(o.results[0]);
  auto alloc =
      res ? res.getDefiningOp<gpu::LocalAllocOp>() : gpu::LocalAllocOp{};
  if (!alloc)
    return declined("ttg.local_alloc", "the op was never recorded");
  auto mt = cast<gpu::MemDescType>(res.getType());
  const std::optional<agpu::TileView> view = tileViewOfMemDesc(mt);
  if (!view)
    return declined("ttg.local_alloc", "the shared encoding is not unswizzled");
  const std::optional<agpu::ElemType> elem = elemTypeOf(mt.getElementType());
  if (!elem)
    return declined("ttg.local_alloc", "the element has no representation");

  const agpu::MemDesc md{"md" + std::to_string(o.results[0]), *view};
  cur_->push_back(agpu::memDescDecl(mc, md, agpu::mslTypeOf(*elem)));

  if (const Value src = alloc.getSrc()) {
    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTy)
      return declined("ttg.local_alloc", "the source is not a tensor");
    cur_->push_back(mc.hardBarrier());
    if (const agpu::Decision d =
            stageWholeTensor(idOf(src), srcTy, md.buffer, md.view, *elem,
                             "ttg.local_alloc", "a source");
        !d.ok())
      return d;
    cur_->push_back(mc.hardBarrier());
  }

  body_.memDescOf[o.results[0]] = md;
  body_.sym.bindDataless(o.results[0]);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitLocalLoad(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const auto it = body_.memDescOf.find(o.operands[0]);
  if (it == body_.memDescOf.end())
    return declined("ttg.local_load", "the handle was never bound to a buffer");
  const agpu::MemDesc &md = it->second;

  const Value res = mlirValueOf(o.results[0]);
  auto resTy =
      res ? dyn_cast<RankedTensorType>(res.getType()) : RankedTensorType();
  if (!resTy)
    return declined("ttg.local_load", "the result is not a tensor");
  const agpu::ElemType *elem = elemOf(o.results[0]);
  if (!elem)
    return declined("ttg.local_load", "result type was never recorded");

  am::SmallVec<agpu::StageAction, 8> actions;
  // The result has no names yet; `local_load` mints its own below.
  const am::SmallVec<am::Str, 8> unused =
      stagedNamesOf(o.results[0], registerCount(resTy));
  if (const agpu::Decision d =
          planTileActions(o.results[0], resTy, wholeWindowsOf(resTy), md.view,
                          elem->bits, actions, unused, "ttg.local_load");
      !d.ok())
    return d;
  int64_t covered = 0;
  for (const agpu::StageAction &a : actions)
    covered += a.width;
  const int64_t regs = registerCount(resTy);
  if (covered < regs)
    return declined("ttg.local_load", "a result register never lands");

  am::SmallVec<am::Str, 8> dst;
  agpu::ValueNames names;
  for (int64_t r = 0; r < regs; ++r) {
    const am::Str n = nameFor('t', o.results[0], r);
    cur_->push_back(agpu::poisonDecl(mc, n, *elem));
    dst.push_back(n);
    names.push_back(n);
  }
  agpu::emitReadback(mc, *cur_, md.view, md.buffer, actions, dst, {},
                     coordSourceOf(resTy), *elem, *elem);
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitMemDescViewOp(const agpu::OpView &o) {
  if (o.operands.empty() || o.results.size() != 1)
    return declined(o.name, "unexpected operand or result count");
  const auto it = body_.memDescOf.find(o.operands[0]);
  if (it == body_.memDescOf.end())
    return declined(o.name, "the handle was never bound to a "
                            "buffer");
  const agpu::MemDesc parent = it->second;
  const Value res = mlirValueOf(o.results[0]);

  if (auto sub = res ? res.getDefiningOp<gpu::MemDescSubsliceOp>()
                     : gpu::MemDescSubsliceOp{}) {
    auto mt = cast<gpu::MemDescType>(res.getType());
    const auto offs = sub.getOffsets();
    if ((int)offs.size() != parent.view.rank())
      return declined("ttg.memdesc_subslice", "offset rank mismatch");
    agpu::TileView::Coord at(offs.begin(), offs.end());
    agpu::TileView::Coord ext(mt.getShape().begin(), mt.getShape().end());
    body_.memDescOf[o.results[0]] = parent.subslice(at, ext);
  } else if (auto ix = res ? res.getDefiningOp<gpu::MemDescIndexOp>()
                           : gpu::MemDescIndexOp{}) {
    // The index must be a compile-time constant.
    if (o.operands.size() < 2)
      return declined("ttg.memdesc_index", "expected a handle and an index");
    const auto k = constantFor_.find(o.operands[1]);
    if (k == constantFor_.end() || k->second.empty() || !k->second[0].known ||
        k->second[0].isFloat)
      return declined("ttg.memdesc_index",
                      "a runtime buffer index is not addressable");
    body_.memDescOf[o.results[0]] = parent.index(k->second[0].i);
  } else {
    return declined(o.name, "the op was never recorded");
  }
  body_.sym.bindDataless(o.results[0]);
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerMemDescHandler() {
  table_.add("localAlloc",
             agpu::forOps({"ttg.local_alloc"}, [this](const agpu::OpView &o) {
               if (o.results.size() != 1)
                 return declined("ttg.local_alloc", "expected one result");
               return emitLocalAlloc(o);
             }));

  table_.add("localLoad",
             agpu::forOps({"ttg.local_load"}, [this](const agpu::OpView &o) {
               if (o.operands.size() != 1 || o.results.size() != 1)
                 return declined("ttg.local_load",
                                 "unexpected operand or result count");
               return emitLocalLoad(o);
             }));

  table_.add("memdescView",
             agpu::forOps({"ttg.memdesc_subslice", "ttg.memdesc_index"},
                          [this](const agpu::OpView &o) {
                            return emitMemDescViewOp(o);
                          }));
}

} // namespace mlir::triton::applegpu::bridge
