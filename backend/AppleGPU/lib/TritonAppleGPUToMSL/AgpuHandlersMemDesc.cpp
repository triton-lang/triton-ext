// Threadgroup buffer handlers: local_alloc, local_load, memdesc_subslice/index.
#include "AgpuDeviceTile.h"
#include "AgpuEmitter.h"

#include "agpu/emit/EmitMemDesc.h"
#include "agpu/emit/EmitPoison.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

// Strides follow the shared encoding's order: fastest-varying dimension gets
// stride 1.
static std::optional<agpu::TileView> stridedView(gpu::MemDescType mt,
                                                 ArrayRef<unsigned> order) {
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

static std::optional<agpu::TileView> paddedView(gpu::PaddedSharedEncodingAttr p,
                                                gpu::MemDescType mt) {
  const SmallVector<unsigned> order = p.getOrder();
  std::optional<agpu::TileView> v = stridedView(mt, order);
  if (!v)
    return std::nullopt;

  const ArrayRef<unsigned> intervals = p.getIntervals();
  const ArrayRef<unsigned> pads = p.getPaddings();
  if (intervals.size() != pads.size())
    return std::nullopt;

  agpu::Padding pad;
  for (std::size_t i = 0; i < intervals.size(); ++i)
    pad.rules.push_back({(int64_t)intervals[i], (int64_t)pads[i]});
  v->setPadding(std::move(pad));
  return v;
}

// Whether `v` reproduces the encoding's own offset map, checked over the whole
// range: an encoding whose addressing this does not model would otherwise
// silently address the wrong element.
static bool viewMatchesLayout(const agpu::TileView &v, gpu::MemDescType mt) {
  MLIRContext *ctx = mt.getContext();
  const LinearLayout ll = gpu::toLinearLayout(mt);
  const auto kOffset = StringAttr::get(ctx, "offset");
  if (!ll.hasInDim(kOffset))
    return false;

  const int rank = mt.getRank();
  const int32_t offsets = ll.getInDimSize(kOffset);
  for (int32_t off = 0; off < offsets; ++off) {
    // apply() requires every in-dim the layout declares, in its own order.
    SmallVector<std::pair<StringAttr, int32_t>> in;
    for (StringAttr d : ll.getInDimNames())
      in.push_back({d, d == kOffset ? off : 0});
    const SmallVector<std::pair<StringAttr, int32_t>> out = ll.apply(in);
    if ((int)out.size() != rank)
      return false;
    agpu::TileView::Coord at(rank, 0);
    for (int d = 0; d < rank; ++d)
      at[d] = out[d].second;
    if (v.offsetOf(at) != off)
      return false;
  }
  return true;
}

// A strided tile, accepted only when it reproduces the encoding's offsets.
static std::optional<agpu::TileView> probedView(gpu::MemDescType mt,
                                                ArrayRef<unsigned> order) {
  std::optional<agpu::TileView> v = stridedView(mt, order);
  if (!v || !viewMatchesLayout(*v, mt))
    return std::nullopt;
  return v;
}

// An encoding whose offsets are a strided tile permuted by an XOR, which is
// what a bank-spreading shared layout amounts to however it spells itself.
// The parameters are searched because the encoding states them in its own
// terms, and each candidate is checked against the layout.
static std::optional<agpu::TileView>
probedSwizzledView(gpu::MemDescType mt, ArrayRef<unsigned> ord) {
  std::optional<agpu::TileView> base = stridedView(mt, ord);
  if (!base || mt.getRank() < 2)
    return std::nullopt;

  const int64_t rowWidth = base->extentAt((int)ord[0]);
  const int64_t slowExtent = base->extentAt((int)ord[1]);
  // A byte-width swizzle repeats within a row, so the span it reaches over is
  // searched alongside its parameters.
  // A panel layout stores each span-wide slice whole, so its rows are `span`
  // apart and the next panel a whole panel away.
  agpu::TileView::Coord panelStride = base->stride();
  for (int64_t span = rowWidth; span >= 2; span >>= 1) {
    panelStride[ord[0]] = 1;
    panelStride[ord[1]] = span;
    const agpu::TileView panels(base->extent(), panelStride);

    for (int64_t vec = 1; vec <= span; vec <<= 1)
      for (int64_t perPhase = 1; perPhase <= slowExtent; perPhase <<= 1)
        for (int64_t maxPhase = 2; maxPhase <= span; maxPhase <<= 1) {
          agpu::Swizzle sw;
          sw.vec = vec;
          sw.perPhase = perPhase;
          sw.maxPhase = maxPhase;
          sw.groupDim = ord[0];
          sw.phaseDim = ord[1];
          sw.groupExtent = span;

          agpu::TileView inline_ = *base;
          inline_.setSwizzle(sw);
          if (viewMatchesLayout(inline_, mt))
            return inline_;

          sw.tileStride = slowExtent * span;
          agpu::TileView panelled = panels;
          panelled.setSwizzle(sw);
          if (viewMatchesLayout(panelled, mt))
            return panelled;
        }
  }
  return std::nullopt;
}

static std::optional<agpu::TileView> tileViewOfMemDesc(gpu::MemDescType mt) {
  // A multi-buffered allocation: the encoding lays out one buffer, the
  // trailing dimensions, and the leading ones index whole buffers placed one
  // after another. Padding counted over the whole allocation would make the
  // buffers differ, so a padded one is left out.
  const int lead =
      (int)mt.getRank() - (int)gpu::getCGALayout(mt.getEncoding()).getRank();
  if (lead > 0) {
    const auto one = gpu::MemDescType::get(
        mt.getShape().drop_front(lead), mt.getElementType(), mt.getEncoding(),
        mt.getMemorySpace(), mt.getMutableMemory());
    const std::optional<agpu::TileView> slice = tileViewOfMemDesc(one);
    if (!slice || slice->padding().pads() || slice->shifted() ||
        slice->cosizeElems() != slice->sizeElems())
      return std::nullopt;
    agpu::TileView::Coord extent(mt.getShape().begin(), mt.getShape().end());
    agpu::TileView::Coord stride(lead);
    int64_t acc = slice->sizeElems();
    for (int d = lead; d-- > 0;) {
      stride[d] = acc;
      acc *= extent[d];
    }
    stride.insert(stride.end(), slice->stride().begin(), slice->stride().end());
    agpu::Swizzle sw = slice->swizzle();
    sw.groupDim += lead;
    sw.phaseDim += lead;
    return agpu::TileView(std::move(extent), std::move(stride), sw,
                          slice->origin());
  }

  if (auto p = dyn_cast<gpu::PaddedSharedEncodingAttr>(mt.getEncoding()))
    return paddedView(p, mt);
  if (auto lin = dyn_cast<gpu::SharedLinearEncodingAttr>(mt.getEncoding()))
    return probedView(mt, lin.getOrder());
  if (auto nv = dyn_cast<gpu::NVMMASharedEncodingAttr>(mt.getEncoding())) {
    const int rank = mt.getRank();
    if (rank < 2)
      return std::nullopt;
    SmallVector<unsigned> order;
    for (int d = rank; d-- > 0;)
      order.push_back((unsigned)d);
    if (nv.getTransposed())
      std::swap(order[0], order[1]);
    if (std::optional<agpu::TileView> v = probedView(mt, order))
      return v;
    return probedSwizzledView(mt, order);
  }

  auto shared = dyn_cast<gpu::SwizzledSharedEncodingAttr>(mt.getEncoding());
  if (!shared)
    return std::nullopt;
  const auto order = shared.getOrder();
  const int rank = mt.getRank();
  std::optional<agpu::TileView> base = stridedView(mt, order);
  if (!base)
    return std::nullopt;
  agpu::TileView::Coord extent(mt.getShape().begin(), mt.getShape().end());

  agpu::Swizzle sw;
  sw.vec = shared.getVec();
  sw.perPhase = shared.getPerPhase();
  sw.maxPhase = shared.getMaxPhase();
  // A rank-1 tile is one row, whose phase is always zero.
  sw.groupDim = order[0];
  sw.groupExtent = extent[(std::size_t)order[0]];
  if (rank > 1) {
    sw.phaseDim = order[1];
  } else {
    sw.phaseDim = order[0];
    sw.maxPhase = 1;
  }

  base->setSwizzle(sw);
  return base;
}

// The view `md` addresses, from the IR alone: the same slices and windows
// `emitMemDescViewOp` binds, without the buffer's name.
static std::optional<agpu::TileView> staticViewOf(Value md) {
  if (md.getDefiningOp<gpu::LocalAllocOp>())
    return tileViewOfMemDesc(cast<gpu::MemDescType>(md.getType()));
  if (auto ix = md.getDefiningOp<gpu::MemDescIndexOp>()) {
    const std::optional<agpu::TileView> parent = staticViewOf(ix.getSrc());
    if (!parent || parent->rank() < 2 || !parent->slicesAt(0))
      return std::nullopt;
    APInt k;
    if (matchPattern(ix.getIndex(), m_ConstantInt(&k)))
      return parent->slice(k.getSExtValue());
    if (parent->padding().pads())
      return std::nullopt;
    return parent->slice(0);
  }
  if (auto sub = md.getDefiningOp<gpu::MemDescSubsliceOp>()) {
    const std::optional<agpu::TileView> parent = staticViewOf(sub.getSrc());
    const auto offs = sub.getOffsets();
    if (!parent || (int)offs.size() != parent->rank())
      return std::nullopt;
    auto mt = cast<gpu::MemDescType>(md.getType());
    return agpu::MemDesc{{}, *parent}
        .subslice(
            agpu::TileView::Coord(offs.begin(), offs.end()),
            agpu::TileView::Coord(mt.getShape().begin(), mt.getShape().end()))
        .view;
  }
  return std::nullopt;
}

// Whether `op`, or an op nested in it, may write shared memory.
static bool mayWriteShared(Operation *op) {
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  return !effects || llvm::any_of(*effects, [](const auto &e) {
    return isa<MemoryEffects::Write>(e.getEffect()) &&
           (isa<gpu::SharedMemory>(e.getResource()) ||
            isa<SideEffects::DefaultResource>(e.getResource()));
  });
}

Value sharedTileOf(Operation *dot, Value operand) {
  auto load = throughLayoutChange(operand).getDefiningOp<gpu::LocalLoadOp>();
  if (!load || load->getBlock() != dot->getBlock())
    return {};
  for (Operation *op = load->getNextNode(); op && op != dot;
       op = op->getNextNode())
    if (mayWriteShared(op))
      return {};

  auto ty = cast<RankedTensorType>(operand.getType());
  auto mt = cast<gpu::MemDescType>(load.getSrc().getType());
  if (mt.getElementType() != ty.getElementType() ||
      !isa<FloatType>(ty.getElementType()))
    return {};
  const std::optional<agpu::TileView> v = staticViewOf(load.getSrc());
  if (!v || v->rank() != 2 || v->swizzle().permutes() || v->padding().pads() ||
      v->shifted())
    return {};
  // Fragments read element pairs, so every row must start on an even element.
  const int64_t pitch = v->strideAt(0);
  if (v->extent() !=
          agpu::TileView::Coord(ty.getShape().begin(), ty.getShape().end()) ||
      v->strideAt(1) != 1 || pitch < v->extentAt(1) || pitch % 2 != 0 ||
      v->origin() % 2 != 0)
    return {};
  return load.getSrc();
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
    return declined("ttg.local_alloc",
                    "the shared encoding is not addressable");
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
  if (const agpu::Decision d =
          planTileActions(resTy, wholeWindowsOf(resTy), md.view, elem->bits,
                          actions, "ttg.local_load");
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
    if (o.operands.size() < 2)
      return declined("ttg.memdesc_index", "expected a handle and an index");
    if (!parent.view.slicesAt(0))
      return declined("ttg.memdesc_index",
                      "the index is a dimension the swizzle permutes");
    const auto k = constantFor_.find(o.operands[1]);
    if (k != constantFor_.end() && !k->second.empty() && k->second[0].known &&
        !k->second[0].isFloat) {
      body_.memDescOf[o.results[0]] = parent.index(k->second[0].i);
    } else {
      // A runtime index moves the base by whole slices, each the first one
      // shifted; padding counted over the whole buffer breaks that.
      if (parent.view.padding().pads())
        return declined("ttg.memdesc_index",
                        "a runtime index into a padded buffer");
      const am::Str *i = body_.sym.scalarName(o.operands[1]);
      if (!i)
        return declined("ttg.memdesc_index", "the index has no emitted name");
      const std::optional<agpu::ElemType> elem =
          elemTypeOf(cast<gpu::MemDescType>(res.getType()).getElementType());
      if (!elem)
        return declined("ttg.memdesc_index",
                        "the element has no representation");
      const am::Str name = "md" + std::to_string(o.results[0]);
      cur_->push_back(agpu_.context().declStmt(
          agpu::mslTypeOf(*elem).pointerTo(am::AddrSpace::Threadgroup), name,
          agpu_.context().binary(
              am::BinOp::Add, agpu_.context().var(parent.buffer),
              agpu_.context().binary(
                  am::BinOp::Mul, agpu_.context().var(*i),
                  agpu_.context().lit(parent.view.strideAt(0))))));
      body_.memDescOf[o.results[0]] = agpu::MemDesc{name, parent.view.slice(0)};
    }
  } else {
    return declined(o.name, "the op was never recorded");
  }
  body_.sym.bindDataless(o.results[0]);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitLocalStore(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const auto it = body_.memDescOf.find(o.operands[1]);
  if (it == body_.memDescOf.end())
    return declined("ttg.local_store",
                    "the handle was never bound to a buffer");
  const agpu::MemDesc &md = it->second;

  const Value src = mlirValueOf(o.operands[0]);
  if (!src)
    return declined("ttg.local_store", "the source was never recorded");
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  if (!srcTy)
    return declined("ttg.local_store", "the source is not a tensor");
  const std::optional<agpu::ElemType> elem = elemTypeOf(srcTy.getElementType());
  if (!elem)
    return declined("ttg.local_store", "the element has no representation");

  cur_->push_back(mc.hardBarrier());
  if (const agpu::Decision d =
          stageWholeTensor(idOf(src), srcTy, md.buffer, md.view, *elem,
                           "ttg.local_store", "a source");
      !d.ok())
    return d;
  cur_->push_back(mc.hardBarrier());
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

  table_.add("localStore",
             agpu::forOps({"ttg.local_store"}, [this](const agpu::OpView &o) {
               if (o.operands.size() != 2)
                 return declined("ttg.local_store", "unexpected operand count");
               return emitLocalStore(o);
             }));

  table_.add("memdescView",
             agpu::forOps({"ttg.memdesc_subslice", "ttg.memdesc_index"},
                          [this](const agpu::OpView &o) {
                            return emitMemDescViewOp(o);
                          }));
}

} // namespace mlir::triton::applegpu::bridge
