// Metal declares threadgroup memory once per kernel at a fixed size, so every
// op's scratch is a region of one buffer, carved while the body is built and
// declared afterward at the extent the body addressed. A buffer outliving one
// op's carve declares itself and goes through the plan's live channel.
#include "AgpuDeviceTile.h"
#include "AgpuDotChain.h"
#include "AgpuEmitter.h"

#include "agpu/core/Names.h"
#include "agpu/emit/EmitBand.h"
#include "agpu/emit/EmitHistogram.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

void PoolLedger::carve(const PoolNeed &need) {
  current_.clear();
  int64_t at = 0;
  for (const PoolNeed::Region &r : need.regions) {
    const int64_t offset = r.atBase ? 0 : at;
    std::size_t found = regions_.size();
    bool nameTaken = false;
    for (std::size_t i = 0; i < regions_.size(); ++i) {
      if (regions_[i].name != r.name)
        continue;
      nameTaken = true;
      if (regions_[i].offset == offset && regions_[i].elem == r.elem) {
        found = i;
        break;
      }
    }
    if (found == regions_.size()) {
      PoolRegion nr{
          r.name,
          r.elem,
          offset,
          r.alignedBytes(),
          false,
          nameTaken ? r.name + "_" + std::to_string(regions_.size()) : r.name};
      regions_.push_back(std::move(nr));
    } else if (r.alignedBytes() > regions_[found].bytes) {
      regions_[found].bytes = r.alignedBytes();
    }
    current_[r.name] = found;
    if (!r.atBase)
      at += r.alignedBytes();
  }
}

am::Str PoolLedger::use(const am::Str &name) {
  const auto it = current_.find(name);
  if (it == current_.end() || it->second >= regions_.size())
    return {};
  PoolRegion &r = regions_[it->second];
  r.used = true;
  return r.decl;
}

am::Str PoolLedger::peek(const am::Str &name) const {
  const auto it = current_.find(name);
  if (it == current_.end() || it->second >= regions_.size())
    return {};
  return regions_[it->second].decl;
}

int64_t PoolLedger::usedBytes() const {
  int64_t n = 0;
  for (const PoolRegion &r : regions_)
    if (r.used)
      n = std::max(n, r.offset + r.bytes);
  return n;
}

// The scratch a layout round trip needs: one band. Returns the plan, so the
// carve and the convert handler share one capacity. The live bytes are the
// local_allocs scanPool recorded before any body builds, so both callers see
// the same capacity.
agpu::BandPlan AgpuEmitter::bandPlanFor(RankedTensorType ty,
                                        const agpu::ElemType &elem) const {
  const llvm::ArrayRef<int64_t> shape = ty.getShape();
  const int64_t rowElems = shape.empty() ? 0 : shape.back();
  return agpu::planBand(
      tileElemCount(ty), agpu::byteWidthOf(elem),
      agpu::Capacity(agpu::Bytes(agpu::kTGResidentBudgetBytes),
                     agpu_.pool.plan().live),
      rowElems);
}

// The axis permutation an op applies, empty for none. Only tt.trans has one.
static llvm::ArrayRef<int32_t> transposeOrderOf(Operation *op) {
  if (auto tr = dyn_cast<triton::TransOp>(op))
    return tr.getOrder();
  return {};
}

PoolNeed AgpuEmitter::poolNeedOf(Operation *op) {
  PoolNeed need;

  // A dot stages its operands into the pool and reads its accumulators back
  // through it. Both are live at once, so they are one request.
  if (auto dot = dyn_cast<triton::DotOp>(op)) {
    const DotShape shape = dotShapeOf(dot);
    if (!shape.aTy)
      return need;
    const agpu::DotFacts f = dotFactsOf(shape);
    // Through the emitter's own planFor, so the carve and the handler plan
    // against one budget.
    const agpu::Plan plan = agpu_.planFor(f);
    const agpu::MmaNames mnm;

    // The regions hold the operands' own elements, or f32 for the lifted
    // integer dot.
    const agpu::ElemType aElem =
        plan.intThroughFloat ? agpu::f32() : shape.aElem;
    const agpu::ElemType bElem =
        plan.intThroughFloat ? agpu::f32() : shape.bElem;

    // A panelled dot addresses one panel of each operand.
    if (plan.kind == agpu::Plan::Kind::Panel) {
      const agpu::Panel &p = plan.panel().panel;
      if (!plan.facts.aDirect)
        need.add(mnm.poolA, aElem, p.aBytes.count());
      need.add(mnm.poolB, bElem, p.bBytes.count());
      const agpu::Plan::CPoolRegion panelC = plan.cPoolRegion();
      if (panelC.bytes > 0)
        need.add(mnm.poolC, plan.cPoolElem(), panelC.bytes,
                 panelC.overlaysOperands);
      return need;
    }

    // Otherwise the plan's numbers: staged operands with bank pad and fragment
    // alignment, and C's reservation, a whole tile when it fits and a band when
    // it does not.
    if (plan.stage.a > agpu::Bytes(0))
      need.add(mnm.poolA, aElem, plan.stage.a.count());
    if (plan.stage.b > agpu::Bytes(0))
      need.add(mnm.poolB, bElem, plan.stage.b.count());
    // Size and placement both come from the plan: a fused C overlays the
    // operands at the pool's base.
    const agpu::Plan::CPoolRegion cRegion = plan.cPoolRegion();
    if (cRegion.bytes > 0)
      need.add(mnm.poolC, plan.cPoolElem(), cRegion.bytes,
               cRegion.overlaysOperands);
    if (plan.edgeScratch > agpu::Bytes(0))
      need.add(mnm.poolE, agpu::f32(), plan.edgeScratch.count());
    return need;
  }

  // A histogram's bins live in threadgroup memory for the whole op: one
  // atomic_uint per bin, bin count is the result's extent.
  if (auto hist = dyn_cast<triton::HistogramOp>(op)) {
    if (auto resTy = dyn_cast<RankedTensorType>(hist.getResult().getType())) {
      const agpu::HistogramNames hnm;
      need.add(hnm.bins, agpu::i32(),
               tileElemCount(resTy) * agpu::byteWidthOf(agpu::i32()));
    }
    return need;
  }

  // A gather reads the source tile at a runtime index, so the whole source
  // crosses the pool.
  if (auto ga = dyn_cast<triton::GatherOp>(op)) {
    auto srcTy = dyn_cast<RankedTensorType>(ga.getSrc().getType());
    const std::optional<agpu::ElemType> e =
        srcTy ? elemTypeOf(srcTy.getElementType()) : std::nullopt;
    if (srcTy && e) {
      const agpu::BandNames bnm;
      need.add(bnm.buffer, *e, tileElemCount(srcTy) * agpu::byteWidthOf(*e));
    }
    return need;
  }

  // A cross-warp reduction or scan publishes through the pool, sized by the
  // same plan the handler emits from.
  if (auto red = dyn_cast<triton::ReduceOp>(op)) {
    auto srcTy = red.getSrcs().empty()
                     ? RankedTensorType()
                     : dyn_cast<RankedTensorType>(red.getSrcs()[0].getType());
    agpu::ReductionPlan plan;
    if (srcTy && reductionPlanOf(red, srcTy, plan).ok() && plan.crossWarp())
      for (int k = 0; k < (int)red.getSrcs().size(); ++k) {
        const agpu::ElemType e = plan.elemAt(k);
        need.add(agpu::reduceScratchKey(k), e,
                 plan.scratch.slotsPerOperand * agpu::byteWidthOf(e));
      }
    return need;
  }
  if (auto scan = dyn_cast<triton::ScanOp>(op)) {
    auto srcTy = scan.getSrcs().empty()
                     ? RankedTensorType()
                     : dyn_cast<RankedTensorType>(scan.getSrcs()[0].getType());
    agpu::ScanFacts facts;
    if (srcTy && scanPlanOf(scan, srcTy, facts).ok()) {
      const agpu::ScanPlan plan = agpu::planScan(facts);
      if (plan.usable && plan.crossWarp)
        for (int k = 0; k < (int)scan.getSrcs().size(); ++k) {
          const agpu::ElemType e = plan.elemAt(k);
          need.add(agpu::scanScratchKey(k), e,
                   plan.scratch.slotsPerOperand * agpu::byteWidthOf(e));
        }
    }
    return need;
  }

  // Whether the round trip happens depends on facts only emission has, so this
  // carves for the worst case; an unused carve never reaches the declaration.
  if (isa<gpu::ConvertLayoutOp, triton::TransOp, triton::ReshapeOp>(op)) {
    // Weaker than emission's test, which also requires the source already
    // bound. A convert skipped here but not elided at emission declines on
    // that same unbound read before touching the pool, so a new path reaching
    // threadgroup memory must decline on an unbound source too.
    if (isa<gpu::ConvertLayoutOp>(op) && usedOnlyByDot(op->getResult(0)))
      return need;

    // The same query emitRedistribute asks, so reservation and addressing
    // agree.
    if (auto srcTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType()))
      if (auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType()))
        if (shuffleFor(srcTy, resTy, transposeOrderOf(op)).usable())
          return need;

    const agpu::BandNames bnm;
    auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    // movedTypeOf: a pointer tensor moves its i32 offsets through the pool.
    const std::optional<agpu::ElemType> e =
        movedTypeOf(op->getResult(0).getType());
    if (ty && e)
      need.add(bnm.buffer, *e, bandPlanFor(ty, *e).bytes().count());
  }
  return need;
}

void AgpuEmitter::scanPool(triton::FuncOp func) {
  // Recorded in the pre-pass because the scalar's definition is emitted before
  // the walk reaches the dot.
  func.walk([&](triton::DotOp dot) {
    const DotShape shape = dotShapeOf(dot);
    if (!agpu_.planFor(dotFactsOf(shape)).storesCDirect())
      return;
    for (const WindowBounds::Clamp &cl : shape.cClamps) {
      if (clampPoison_.count(cl.start))
        continue;
      const auto it = clampOf_.find(cl.start);
      if (it != clampOf_.end() && it->second != cl.to) {
        clampOf_.erase(it);
        clampPoison_.insert(cl.start);
        continue;
      }
      clampOf_[cl.start] = cl.to;
    }
  });

  // A ttg.local_alloc coexists with every operation's pool use, so its bytes
  // go through the plan's live channel.
  func.walk([&](gpu::LocalAllocOp a) {
    auto mt = cast<gpu::MemDescType>(a.getResult().getType());
    const std::optional<agpu::ElemType> e = elemTypeOf(mt.getElementType());
    int64_t elems = 1;
    for (int64_t d : mt.getShape())
      elems *= d;
    agpu_.pool.live(agpu::Bytes(elems * (e ? agpu::byteWidthOf(*e) : 1)));
  });

  func.walk([&](triton::AtomicRMWOp rmw) {
    const std::optional<agpu::ElemType> e =
        elemTypeOf(rmw.getResult().getType());
    if (e && agpu::electFor(spreadOf(rmw.getPtr())).crossesWarp())
      agpu_.pool.live(agpu::Bytes(agpu::byteWidthOf(*e)));
  });
}

// `threadgroup T *name = (threadgroup T *)(pool + off);` per used region.
am::Block AgpuEmitter::poolDecls() {
  am::Context &mc = agpu_.context();
  am::Block out;
  for (const PoolRegion &r : body_.pool.regions()) {
    if (!r.used)
      continue;
    const am::Type ptr =
        agpu::mslTypeOf(r.elem).pointerTo(am::AddrSpace::Threadgroup);
    am::Expr *base = mc.var(agpu::KernelNames{}.pool);
    if (r.offset)
      base = mc.binary(am::BinOp::Add, base, mc.lit(r.offset));
    out.push_back(mc.declStmt(ptr, r.decl, mc.cast(ptr, base)));
  }
  return out;
}

am::Str AgpuEmitter::liveBuffer(const am::Str &name, agpu::ElemType elem) {
  if (!body_.declaresThreadgroup)
    return {};
  const am::Type ty =
      agpu::mslTypeOf(elem).inAddrSpace(am::AddrSpace::Threadgroup);
  bool nameTaken = false;
  for (const LiveBuffer &b : body_.liveBuffers) {
    if (b.name != name)
      continue;
    if (b.elem == ty)
      return b.decl;
    nameTaken = true;
  }
  const am::Str decl =
      nameTaken ? name + "_" + std::to_string(body_.liveBuffers.size()) : name;
  body_.liveDecls.push_back(agpu_.context().arrayDecl(ty, decl, 1));
  body_.liveBuffers.push_back({name, ty, decl});
  return decl;
}

} // namespace mlir::triton::applegpu::bridge
