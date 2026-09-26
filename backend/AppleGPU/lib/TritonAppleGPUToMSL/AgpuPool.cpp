// Metal declares threadgroup memory once per kernel at a fixed size, so every
// op's scratch is a region of one buffer, carved while the body is built and
// declared afterward at the extent the body addressed. A buffer outliving one
// op's carve declares itself and goes through the plan's live channel.
#include "AgpuDeviceTile.h"
#include "AgpuDotChain.h"
#include "AgpuEmitter.h"

#include "agpu/core/Names.h"
#include "agpu/cost/Occupancy.h"
#include "agpu/emit/EmitBand.h"
#include "agpu/emit/EmitHistogram.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

void PoolLedger::carve(const PoolNeed &need, int64_t floor) {
  current_.clear();
  int64_t at = floor;
  for (const PoolNeed::Region &r : need.regions) {
    const int64_t offset = r.atBase ? floor : at;
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

PoolNeed AgpuEmitter::dotPoolNeed(Operation *dot, const DotShape &shape,
                                  const agpu::DotFacts &f) {
  PoolNeed need;
  // Through the emitter's own planFor, so the carve and the handler plan
  // against one budget.
  const agpu::Plan plan = agpu_.planFor(f);
  const agpu::MmaNames mnm;

  // The regions hold the operands' own elements, or f32 for the lifted
  // integer dot.
  const agpu::ElemType aElem = plan.intThroughFloat ? agpu::f32() : shape.aElem;
  const agpu::ElemType bElem = plan.intThroughFloat ? agpu::f32() : shape.bElem;

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
  if (plan.stage.a > agpu::Bytes(0) && !body_.residentBuf.count({dot, 0}))
    need.add(mnm.poolA, aElem, plan.stage.a.count());
  if (plan.stage.b > agpu::Bytes(0) && !body_.residentBuf.count({dot, 1}))
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

PoolNeed AgpuEmitter::poolNeedOf(Operation *op) {
  PoolNeed need;

  // A dot stages its operands into the pool and reads its accumulators back
  // through it. Both are live at once, so they are one request.
  if (auto dot = dyn_cast<triton::DotOp>(op)) {
    const DotShape shape = dotShapeOf(dot);
    if (!shape.aTy)
      return need;
    return dotPoolNeed(dot, shape, dotFactsOf(shape));
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
    if (!srcTy)
      return need;
    const agpu::Result<agpu::ReductionPlan> planned =
        reductionPlanOf(red, srcTy);
    const agpu::ReductionPlan &plan = planned.value;
    if (planned.ok() && plan.crossWarp())
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
    const agpu::Result<agpu::ScanFacts> facts =
        srcTy ? scanPlanOf(scan, srcTy)
              : agpu::Result<agpu::ScanFacts>::no(agpu::Decision::notMine());
    if (facts.ok()) {
      const agpu::ScanPlan plan = agpu::planScan(facts.value);
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
    if (reshapeKeepsRegisters(op))
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

// Clamping a window's start moves the whole tile only when nothing else
// carries the row it came from: `start` must be `x * c` with the multiply
// the only use of `x`. Otherwise part of the tile keeps the unclamped row
// and the overlap it stores twice differs.
static bool clampShiftsWholeTile(Value start) {
  auto mul = start ? start.getDefiningOp<arith::MulIOp>() : arith::MulIOp{};
  if (!mul)
    return false;
  for (Value x : {mul.getLhs(), mul.getRhs()})
    if (!matchPattern(x, m_Constant()) && !x.hasOneUse())
      return false;
  return true;
}

void AgpuEmitter::scanPool(triton::FuncOp func) {
  // The scan plans the unrolled build; a rolled rebuild replans its own dots.
  rollK_ = false;
  // Recorded in the pre-pass because the scalar's definition is emitted before
  // the walk reaches the dot.
  func.walk([&](triton::DotOp dot) {
    const DotShape shape = dotShapeOf(dot);
    if (!agpu_.planFor(dotFactsOf(shape)).storesCDirect())
      return;
    for (const WindowBounds::Clamp &cl : shape.cClamps) {
      if (clampPoison_.count(cl.start))
        continue;
      if (!clampShiftsWholeTile(cl.start)) {
        clampOf_.erase(cl.start);
        clampPoison_.insert(cl.start);
        continue;
      }
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

  planResidentOperands(func);
}

void AgpuEmitter::planResidentOperands(triton::FuncOp func) {
  if (!func.isPublic())
    return;
  directInvariantA_.clear();
  planSeedGrants(func);
  std::vector<ResidentOperand> found;
  int buffers = 0;
  func.walk([&](triton::DotOp dot) {
    auto loop = dot->getParentOfType<scf::ForOp>();
    if (!loop)
      return;
    const DotShape shape = dotShapeOf(dot);
    if (!shape.aTy)
      return;
    // The planner may turn a device-direct A back into a staged one.
    const agpu::Plan plan = agpu_.planFor(dotFactsOf(shape));
    const agpu::DotFacts &f = plan.facts;
    if (plan.kind == agpu::Plan::Kind::Panel)
      return;
    const bool pad = plan.padStagedOperands();
    for (int which = 0; which < 2; ++which) {
      if (which == 0 && f.aDirect)
        continue;
      const agpu::Bytes stage = which == 0 ? plan.stage.a : plan.stage.b;
      const Value source = throughLayoutChange(dot->getOperand(which));
      if (stage <= agpu::Bytes(0) || !isa<RankedTensorType>(source.getType()) ||
          !loop.isDefinedOutsideOfLoop(source))
        continue;

      ResidentOperand r;
      r.dot = dot;
      r.which = which;
      r.source = source;
      r.loop = loop;
      r.elem = stagedElemOf(plan, which == 0 ? shape.aElem : shape.bElem);
      r.view =
          which == 0 ? agpu::stagedAView(f, pad) : agpu::stagedBView(f, pad);
      r.bytes = stage.count();
      r.buffer = -1;
      for (const ResidentOperand &o : found)
        if (o.source == r.source && o.loop == r.loop && o.elem == r.elem &&
            o.view == r.view && o.bytes == r.bytes) {
          r.buffer = o.buffer;
          break;
        }
      if (r.buffer < 0)
        r.buffer = buffers++;
      found.push_back(r);
    }
  });

  // Loops that run one after another reuse the same slots; a nested loop
  // stacks its slots above its ancestors'.
  std::map<Operation *, std::vector<int>> perLoopBuffers;
  for (ResidentOperand r : found) {
    std::vector<int> &v = perLoopBuffers[r.loop.getOperation()];
    if (!llvm::is_contained(v, r.buffer))
      v.push_back(r.buffer);
  }
  struct Slot {
    int index;
    agpu::ElemType elem;
    int64_t bytes;
  };
  std::vector<Slot> slots;
  for (ResidentOperand &r : found) {
    Operation *loopOp = r.loop.getOperation();
    int index = 0;
    for (Operation *p = loopOp->getParentOp(); p; p = p->getParentOp())
      if (const auto it = perLoopBuffers.find(p); it != perLoopBuffers.end())
        index += (int)it->second.size();
    const std::vector<int> &own = perLoopBuffers[loopOp];
    index += (int)(llvm::find(own, r.buffer) - own.begin());
    int id = 0;
    while (id < (int)slots.size() &&
           !(slots[id].index == index && slots[id].elem == r.elem))
      ++id;
    if (id == (int)slots.size())
      slots.push_back({index, r.elem, 0});
    slots[id].bytes = std::max(slots[id].bytes, r.bytes);
    r.buffer = id;
  }
  for (ResidentOperand &r : found)
    r.bytes = slots[r.buffer].bytes;
  buffers = (int)slots.size();
  const std::vector<ResidentOperand> candidates = found;

  // The pool's peak with `residents` held and the dots in `inPlace` reading
  // their A in place, plus the residents' own bytes.
  const auto peakWhere = [&](const std::vector<ResidentOperand> &residents,
                             const std::set<Operation *> &inPlace) {
    for (const ResidentOperand &r : residents)
      body_.residentBuf[{r.dot, r.which}] = "";
    int64_t peak = 0;
    func.walk([&](Operation *op) {
      if (op == func.getOperation())
        return;
      auto dot = dyn_cast<triton::DotOp>(op);
      if (dot && inPlace.count(op)) {
        DotShape shape = dotShapeOf(dot);
        shape.aRestagedEachTrip = true;
        peak =
            std::max(peak, dotPoolNeed(op, shape, dotFactsOf(shape)).bytes());
        return;
      }
      peak = std::max(peak, poolNeedOf(op).bytes());
    });
    body_.residentBuf.clear();
    std::map<int, int64_t> bytes;
    for (const ResidentOperand &r : residents)
      bytes[r.buffer] = r.bytes;
    for (const auto &[b, n] : bytes)
      peak += n;
    return peak;
  };
  const auto peakWith = [&](const std::vector<ResidentOperand> &cands,
                            int keep) {
    std::vector<ResidentOperand> kept;
    for (const ResidentOperand &r : cands)
      if (r.buffer < keep)
        kept.push_back(r);
    return peakWhere(kept, {});
  };

  const int64_t live = agpu_.pool.plan().live.count();
  const int64_t threads = agpu::threadsFor(numWarps());
  const auto fit = [&](std::vector<ResidentOperand> cands) {
    std::map<int, int> ids;
    for (ResidentOperand &r : cands)
      r.buffer = ids.try_emplace(r.buffer, (int)ids.size()).first->second;
    int keep = (int)ids.size();
    while (keep > 0 &&
           peakWith(cands, keep) > agpu::kTGResidentBudgetBytes - live)
      --keep;
    llvm::erase_if(cands,
                   [&](const ResidentOperand &r) { return r.buffer >= keep; });
    return cands;
  };

  // One invariant operand per loop does not spill; only more is worth a
  // resident threadgroup.
  std::map<Operation *, std::set<int>> perLoop;
  for (ResidentOperand r : found)
    perLoop[r.loop.getOperation()].insert(r.buffer);
  std::vector<ResidentOperand> chosen = fit(found);
  if (!agpu::cost::gainsResidency(peakWith(found, 0) + live,
                                  peakWith(chosen, buffers) + live, threads)) {
    llvm::erase_if(found, [&](ResidentOperand r) {
      return perLoop[r.loop.getOperation()].size() < 2;
    });
    chosen = fit(found);
  }

  const auto readableInPlace = [&](const ResidentOperand &r) {
    return r.which == 0 &&
           (bool)dotShapeOf(cast<triton::DotOp>(r.dot)).aDevice.base;
  };

  // A resident A that could be read in place gives its buffer up where that
  // buys a resident threadgroup. The buffer goes only with all its readers.
  std::set<int> triedBuffers;
  for (const ResidentOperand &r : std::vector<ResidentOperand>(chosen)) {
    if (!triedBuffers.insert(r.buffer).second)
      continue;
    std::set<Operation *> readers;
    bool movable = true;
    for (const ResidentOperand &c : chosen)
      if (c.buffer == r.buffer) {
        movable = movable && readableInPlace(c);
        readers.insert(c.dot);
      }
    if (!movable)
      continue;
    std::vector<ResidentOperand> without = chosen;
    llvm::erase_if(without, [&](const ResidentOperand &c) {
      return c.buffer == r.buffer;
    });
    if (agpu::cost::gainsResidency(peakWhere(chosen, {}) + live,
                                   peakWhere(without, readers) + live,
                                   threads)) {
      chosen = std::move(without);
      directInvariantA_.insert(readers.begin(), readers.end());
    }
  }

  // An A turned away here is restaged every trip. Read it in place instead,
  // but only where the pool bytes that frees buy a resident threadgroup:
  // otherwise staged fragments are the faster read. Every dot restaging the
  // same A switches together; one left staging keeps the peak.
  llvm::DenseSet<Value> triedSources;
  for (const ResidentOperand &r : candidates) {
    if (r.which != 0 || !triedSources.insert(r.source).second)
      continue;
    std::set<Operation *> readers;
    for (const ResidentOperand &c : candidates)
      if (c.source == r.source && readableInPlace(c) &&
          !directInvariantA_.count(c.dot) &&
          llvm::none_of(chosen, [&](const ResidentOperand &k) {
            return k.dot == c.dot && k.which == 0;
          }))
        readers.insert(c.dot);
    if (!readers.empty() &&
        agpu::cost::gainsResidency(peakWhere(chosen, {}) + live,
                                   peakWhere(chosen, readers) + live, threads))
      directInvariantA_.insert(readers.begin(), readers.end());
  }

  std::set<int> reserved;
  for (const ResidentOperand &r : chosen) {
    if (reserved.insert(r.buffer).second)
      agpu_.pool.live(agpu::Bytes(r.bytes));
    residents_.push_back(r);
  }
}

// Before residency, so resident operands are planned against the pool the
// seeds leave. Priced as a rolled build prices it, where no seed is free: a
// grant is kept only where it is needed for the certain residency all of them
// together reach.
void AgpuEmitter::planSeedGrants(triton::FuncOp func) {
  aSeedGranted_.clear();
  struct Grant {
    Operation *dot;
    int64_t need;
  };
  std::vector<std::pair<Operation *, int64_t>> needs;
  std::vector<Grant> offers;
  func.walk([&](Operation *op) {
    if (op == func.getOperation())
      return;
    auto dot = dyn_cast<triton::DotOp>(op);
    if (!dot) {
      needs.push_back({op, poolNeedOf(op).bytes()});
      return;
    }
    const DotShape shape = dotShapeOf(dot);
    agpu::DotFacts f = dotFactsOf(shape);
    f.rollK = true;
    const int64_t staged = dotPoolNeed(dot, shape, f).bytes();
    needs.push_back({op, staged});
    f.aSeedGranted = true;
    if (!agpu_.planFor(f).facts.aFromRegs)
      return;
    const int64_t seeded = dotPoolNeed(dot, shape, f).bytes();
    if (seeded < staged)
      offers.push_back({op, seeded});
  });
  if (offers.empty())
    return;

  const int64_t live = agpu_.pool.plan().live.count();
  const int64_t threads = agpu::threadsFor(numWarps());
  std::set<Operation *> granted;
  const auto residency = [&] {
    int64_t peak = 0;
    for (const auto &[op, need] : needs) {
      int64_t n = need;
      for (const Grant &g : offers)
        if (g.dot == op && granted.count(op))
          n = g.need;
      peak = std::max(peak, n);
    }
    return agpu::cost::certainResidency(peak + live, threads);
  };
  const int64_t without = residency();
  for (const Grant &g : offers)
    granted.insert(g.dot);
  const int64_t target = residency();
  if (target <= without)
    return;
  for (const Grant &g : offers) {
    granted.erase(g.dot);
    if (residency() < target)
      granted.insert(g.dot);
  }
  aSeedGranted_ = std::move(granted);
}

void AgpuEmitter::stageResidentOperands(scf::ForOp loop) {
  if (!body_.declaresThreadgroup)
    return;
  am::Context &mc = agpu_.context();
  std::map<int, am::Str> staged;
  std::set<int> failed, written;
  am::Block writes;
  {
    const CurBlock into(*this, writes);
    for (const ResidentOperand &r : residents_) {
      if (r.loop != loop || failed.count(r.buffer))
        continue;
      if (!staged.count(r.buffer)) {
        auto ty = cast<RankedTensorType>(r.source.getType());
        const agpu::ValueId id = idOf(r.source);
        const am::Str name = "res" + std::to_string(r.buffer);
        const auto held = body_.residentHeld.find(r.buffer);
        if (held != body_.residentHeld.end() && held->second.source == id &&
            held->second.elem == r.elem && held->second.view == r.view &&
            held->second.block == loop->getBlock()) {
          staged[r.buffer] = name;
          body_.residentBuf[{r.dot, r.which}] = name;
          continue;
        }
        body_.residentHeld.erase(r.buffer);
        if (!body_.sym.regAt(id, 0) ||
            !stageWholeTensor(id, ty, name, r.view, r.elem, "tt.dot",
                              "a resident")
                 .ok()) {
          failed.insert(r.buffer);
          continue;
        }
        if (body_.residentDeclared.insert(r.buffer).second)
          body_.liveDecls.push_back(mc.arrayDecl(
              agpu::mslTypeOf(r.elem).inAddrSpace(am::AddrSpace::Threadgroup),
              name, r.bytes / agpu::byteWidthOf(r.elem)));
        staged[r.buffer] = name;
        written.insert(r.buffer);
        body_.residentHeld[r.buffer] = {id, r.elem, r.view, loop->getBlock()};
      }
      body_.residentBuf[{r.dot, r.which}] = staged[r.buffer];
    }
  }
  if (written.empty())
    return;
  // Before: a loop re-entered from an outer one overwrites what the last
  // entry's dots may still be reading. After: the first dot reads it.
  cur_->push_back(mc.barrier());
  for (am::Stmt *s : writes)
    cur_->push_back(s);
  cur_->push_back(mc.barrier());
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
