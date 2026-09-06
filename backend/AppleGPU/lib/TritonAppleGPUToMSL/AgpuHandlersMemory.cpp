// Device access handlers: addptr, load, store, barrier. Pointers are never
// materialised; an access is base[off].
#include "AgpuEmitter.h"
#include "AgpuOpTables.h"

#include "agpu/emit/EmitElection.h"
#include "agpu/emit/EmitMove.h"
#include "agpu/plan/BarrierPlan.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

agpu::PtrDims AgpuEmitter::ptrDimsOf(Value ptr, const agpu::ElemType &elem) {
  agpu::PtrDims out;
  const auto rt =
      ptr ? dyn_cast<RankedTensorType>(ptr.getType()) : RankedTensorType();
  if (!rt)
    return out;
  AxisInfo *ai = axisInfo().getAxisInfo(ptr);
  if (!ai)
    return out;
  const bool bytes = isTensorOfPointers(rt);
  for (int d = 0; d < rt.getRank(); ++d)
    out.push_back(agpu::ptrInfoFrom(
        agpu::AxisReport{ai->getContiguity(d), ai->getDivisibility(d), bytes},
        elem));
  return out;
}

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

namespace {

Value throughExpand(Value v) {
  while (auto ed = v.getDefiningOp<ExpandDimsOp>())
    v = ed.getSrc();
  return v;
}

bool splatIntOf(Value v, int64_t &out) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  auto dense =
      cst ? dyn_cast<DenseElementsAttr>(cst.getValue()) : DenseElementsAttr();
  if (!dense || !dense.isSplat())
    return false;
  auto i = dyn_cast<IntegerAttr>(dense.getSplatValue<Attribute>());
  if (!i)
    return false;
  out = i.getInt();
  return true;
}

} // namespace

// A mask is a bound only when it reads `iota < constant` on one axis of the
// laid-out tensor: the compared index is then the coordinate itself, so the
// layout decides the mask and no lane can disagree.
agpu::MaskBound AgpuEmitter::maskBoundOf(Value mask, Value laidOut) {
  agpu::MaskBound b;
  const auto lt = laidOut ? dyn_cast<RankedTensorType>(laidOut.getType())
                          : RankedTensorType();
  auto cmp = mask ? mask.getDefiningOp<arith::CmpIOp>() : arith::CmpIOp();
  if (!lt || !cmp)
    return b;

  Value idx, limit;
  if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
    idx = cmp.getLhs();
    limit = cmp.getRhs();
  } else if (cmp.getPredicate() == arith::CmpIPredicate::sgt) {
    idx = cmp.getRhs();
    limit = cmp.getLhs();
  } else {
    return b;
  }
  if (!splatIntOf(limit, b.limit))
    return b;

  Value src = throughExpand(idx);
  auto mr = src.getDefiningOp<MakeRangeOp>();
  if (!mr || mr.getStart() != 0)
    return b;

  const auto mt = dyn_cast<RankedTensorType>(mask.getType());
  if (!mt || mt.getShape() != lt.getShape())
    return b;

  // The range is the one axis the mask varies along, so it is the only one
  // whose extent it can be; an ambiguous shape is left unrecognised.
  int dim = -1;
  for (int d = 0; d < lt.getRank(); ++d) {
    if (lt.getDimSize(d) != (int64_t)mr.getEnd())
      continue;
    if (dim >= 0)
      return b;
    dim = d;
  }
  if (dim < 0)
    return b;

  const std::vector<agpu::LayoutBasis> dims = layoutDimsOf(laidOut);
  if ((int)dims.size() != lt.getRank())
    return b;

  b.known = true;
  b.dim = dim;
  b.dimSize = lt.getDimSize(dim);
  b.basis = dims[(std::size_t)dim];
  return b;
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
  f.elemBits = ready.elem.bits;
  f.hasMask = o.operands.size() > maskIndex;
  f.hasOther = hasOther;
  f.coherent = coherentBuffer(mlirValueOf(o.operands[0]));
  f.ptr = ptrDimsOf(mlirValueOf(o.operands[0]), ready.elem);

  const std::vector<agpu::LayoutBasis> lDims =
      layoutDimsOf(mlirValueOf(o.results[0]));
  f.bases = agpu::regBasesOf(lDims);
  f.runtime = agpu::runtimeSpanOf(lDims);
  if (f.hasMask)
    f.bound = maskBoundOf(mlirValueOf(o.operands[maskIndex]),
                          mlirValueOf(o.results[0]));

  agpu::MoveSite site;
  // When the pointer's registers are one affine family, materialise one base
  // and subscript it by literal deltas, so each register costs no address of
  // its own.
  am::Str derivedBase;
  std::vector<int64_t> deltas;
  [&] {
    const auto it = body_.affine.find(o.operands[0]);
    if (it == body_.affine.end() || ready.regs < 2 || f.coherent)
      return;
    const Value pv = mlirValueOf(o.operands[0]);
    auto rt =
        pv ? dyn_cast<RankedTensorType>(pv.getType()) : RankedTensorType();
    if (!rt || (int)it->second.scales.size() != rt.getRank())
      return;
    // Deltas are arithmetic differences of lane-0 coordinates; the layout
    // composes over GF(2), so check the two agree.
    if (!affineRegisterDeltas(rt, (int)ready.regs))
      return;
    if (!scaledRegisterDeltas(rt, ready.regs, it->second.scales, deltas))
      return;
    // uniformNameOf confirms one base for every register; a tile with two bases
    // would read one buffer's deltas off another.
    const am::Str *base = body_.sym.uniformNameOf(o.operands[0]);
    const auto off = body_.offsetOf.find({o.operands[0], 0});
    if (!base || off == body_.offsetOf.end())
      return;
    derivedBase =
        derivedDevicePointer(*base, agpu_.context().var(off->second.name),
                             ready.elem, "pl" + std::to_string(o.results[0]));
  }();
  if (!derivedBase.empty())
    site.elem = [this, derivedBase, deltas](int64_t r) -> am::Expr * {
      return agpu_.context().subscript(
          agpu_.context().var(derivedBase),
          agpu_.context().lit(deltas[(std::size_t)r]));
    };
  else
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

  const agpu::MovePlan p = agpu::planMove(f);
  agpu::emitMove(agpu_.context(), *cur_, f, p, site, ready.elem);
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
  // same address, so elect one writer per store.
  am::Expr *const elected = agpu::electionExpr(
      agpu_.context(), agpu::electFor(spreadOf(ptrV)), agpu::ThreadNames{});

  for (int64_t r = 0; r < regs; ++r)
    if (!addressAt(ptr, r))
      return declined(o.name, "pointer has no recorded offset");

  const agpu::ElemType *ve = elemOf(o.operands[1]);
  agpu::MoveFacts f;
  f.regCount = regs;
  f.isStore = true;
  f.elemBits = ve ? ve->bits : 0; // unknown element: every access scalar
  f.hasMask = o.operands.size() > maskIndex || elected != nullptr;
  f.coherent = coherentBuffer(ptrV);
  if (ve)
    f.ptr = ptrDimsOf(ptrV, *ve);

  const std::vector<agpu::LayoutBasis> sDims = layoutDimsOf(ptrV);
  f.bases = agpu::regBasesOf(sDims);
  f.runtime = agpu::runtimeSpanOf(sDims);
  f.guardHasRuntimeTerm = elected != nullptr;
  if (o.operands.size() > maskIndex)
    f.bound = maskBoundOf(mlirValueOf(o.operands[maskIndex]), ptrV);

  agpu::MoveSite site;
  site.elem = [this, ptr](int64_t r) { return addressAt(ptr, r); };
  if (f.hasMask)
    site.guard = [this, &o, maskIndex, elected](int64_t r) {
      return agpu_.context().allOf(elected, maskAt(o, maskIndex, r));
    };
  for (int64_t r = 0; r < regs; ++r) {
    const am::Str narrowed = inIrType(o.operands[1], r);
    site.values.push_back(narrowed.empty() ? val.at(r) : narrowed);
  }

  const agpu::MovePlan p = agpu::planMove(f);
  agpu::emitMove(agpu_.context(), *cur_, f, p, site, ve ? *ve : agpu::f32());

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

  // The pointer's affine family follows Add's rule over base and offset.
  {
    const auto famOf = [&](std::size_t i) {
      const auto it = body_.affine.find(o.operands[i]);
      return it != body_.affine.end() ? it->second : agpu::AffineFamily{};
    };
    const auto uniformOp = [&](std::size_t i) {
      const am::Str *first = body_.sym.regAt(o.operands[i], 0);
      if (!first)
        return false;
      for (int64_t r = 1; r < ready.regs; ++r) {
        const am::Str *n = body_.sym.regAt(o.operands[i], (std::size_t)r);
        if (!n || *n != *first)
          return false;
      }
      // A base whose offsets differ per register is not uniform even if
      // its name repeats. Only addptr needs this: offsetOf is populated for
      // pointer values alone, and the elementwise fold's operands are always
      // arith results.
      if (i == 0)
        for (int64_t r = 0; r < ready.regs; ++r)
          if (body_.offsetOf.count({o.operands[0], r}))
            return famOf(0).ok();
      return true;
    };
    const Value res = mlirValueOf(o.results[0]);
    auto rt =
        res ? dyn_cast<RankedTensorType>(res.getType()) : RankedTensorType();
    if (rt) {
      const agpu::AffineFamily fam =
          agpu::foldFamily(agpu::EwOp::Add, famOf(0), uniformOp(0), famOf(1),
                           uniformOp(1), nullptr, nullptr, (int)rt.getRank());
      if (fam.ok())
        body_.affine[o.results[0]] = fam;
    }
  }
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

agpu::Decision AgpuEmitter::emitBarrierOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const uint32_t spaces = o.name == kBarrier
                              ? ((uint32_t)agpu::BarrierSpace::GlobalRead |
                                 (uint32_t)agpu::BarrierSpace::GlobalWrite)
                              : (uint32_t)o.intAt(0);
  const agpu::BarrierPlan p = agpu::planBarrier(spaces);
  cur_->push_back(mc.barrier(p.scope));
  if (p.needsDeviceFence)
    cur_->push_back(agpu::deviceFence(mc));
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerBarrierHandler() {
  // threadgroup_barrier orders memory only within the threadgroup even with
  // mem_device set, so a device-ordering barrier needs a fence behind it.
  table_.add("barrier", agpu::forOps({"ttg.barrier", kBarrier},
                                     [this](const agpu::OpView &o) {
                                       return emitBarrierOp(o);
                                     }));
}

} // namespace mlir::triton::applegpu::bridge
