// The direct C drain: turning DrainStepFact (Values) into DrainStep (names),
// and the device pointers the drain stores through.
#include "AgpuDotChain.h"
#include "AgpuEmitter.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

static am::Expr *windowRowStartExpr(am::Context &c, const am::Str &name,
                                    int64_t mod) {
  am::Expr *e = c.var(name);
  return mod ? c.binary(am::BinOp::Rem, e, c.lit(mod)) : e;
}

agpu::Decision AgpuEmitter::baseOffsetExpr(const DeviceTile &t,
                                           const char *which, am::Expr *&out) {
  am::Context &mc = agpu_.context();
  out = nullptr;
  // A scalar addptr binds to its base's name and keeps the offset aside.
  if (const auto off = body_.offsetOf.find({idOf(t.base), 0});
      off != body_.offsetOf.end())
    out = mc.var(off->second.name);
  if (!t.baseOffset)
    return agpu::Decision::emitted();
  const am::Str *n = body_.sym.scalarName(idOf(t.baseOffset));
  if (!n)
    return declined("tt.dot",
                    std::string(which) + "'s window base offset has no name");
  out = out ? mc.binary(am::BinOp::Add, out, mc.var(*n)) : mc.var(*n);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::strideOf(const DeviceTile &t, const char *which,
                                     agpu::Stride &out) {
  if (!t.rowStride) {
    out = agpu::Stride(t.rowStrideK);
    return agpu::Decision::emitted();
  }
  const am::Str *n = body_.sym.scalarName(idOf(t.rowStride));
  if (!n)
    return declined("tt.dot",
                    std::string(which) + "'s window row stride has no name");
  out = agpu::Stride::runtime(*n);
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::readADirect(const DotOperands &ops,
                                        agpu::DotInputs &in) {
  const am::Str *base = body_.sym.scalarName(idOf(ops.shape.aDevice.base));
  if (!base)
    return declined("tt.dot", "A is device-resident but its base has no name");
  const agpu::Decision strideOk =
      strideOf(ops.shape.aDevice, "A", in.a.leadingDim);
  if (!strideOk.ok())
    return strideOk;

  in.a.buffer = *base;
  in.a.space = am::AddrSpace::Device;

  // Window starting past the tensor corner: point one fresh pointer at it, so
  // the offset does not enter every fragment load.
  am::Expr *baseOff = nullptr;
  if (const agpu::Decision d = baseOffsetExpr(ops.shape.aDevice, "A", baseOff);
      !d.ok())
    return d;

  if (baseOff || ops.shape.aDevice.rowStart || ops.shape.aDevice.colStart) {
    am::Expr *off = baseOff;
    if (ops.shape.aDevice.rowStart) {
      const am::Str *rs =
          body_.sym.scalarName(idOf(ops.shape.aDevice.rowStart));
      if (!rs)
        return declined("tt.dot", "A's window row offset has no name");
      am::Expr *row = in.a.leadingDim.scale(
          agpu_.context(), windowRowStartExpr(agpu_.context(), *rs,
                                              ops.shape.aDevice.rowStartMod));
      off = off ? agpu_.context().binary(am::BinOp::Add, off, row) : row;
    }
    if (ops.shape.aDevice.colStart) {
      const am::Str *cs =
          body_.sym.scalarName(idOf(ops.shape.aDevice.colStart));
      if (!cs)
        return declined("tt.dot", "A's window column offset has no name");
      am::Expr *col = agpu_.context().var(*cs);
      off = off ? agpu_.context().binary(am::BinOp::Add, off, col) : col;
    }
    in.a.buffer = derivedDevicePointer(*base, off, ops.shape.aElem,
                                       "pAdev" + std::to_string(body_.dotSeq));
    in.a.space = am::AddrSpace::Device;
  }
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::resolveDrainSteps(const DotOperands &ops,
                                              agpu::DotInputs &in) {
  // A splat constant carries no type of its own, and an f32 literal beside a
  // narrowed accumulator makes `metal::max` ambiguous.
  const agpu::ElemType splatElem =
      elemTypeOf(ops.shape.cDevice.base.getType()).value_or(agpu::f32());

  // Resolve one folded operand to names the drain can spell.
  const auto resolveOperand = [&](const auto &sf,
                                  agpu::DrainOperand &od) -> agpu::Decision {
    using K = std::decay_t<decltype(sf.kind)>;
    switch (sf.kind) {
    default:
      break;
    case K::Splat: {
      if (!sf.splat) {
        od.kind = agpu::DrainOperand::Kind::Splat;
        od.splat = agpu_.context().litF(sf.splatConst, mslTypeOf(splatElem));
        break;
      }
      const am::Str *n = body_.sym.scalarName(idOf(sf.splat));
      if (!n)
        return declined("tt.dot",
                        "a folded operand's scalar has no emitted name");
      od.kind = agpu::DrainOperand::Kind::Splat;
      od.splat = agpu_.context().var(*n);
      break;
    }
    case K::Window: {
      const am::Str *base = body_.sym.scalarName(idOf(sf.addend.window.base));
      if (!base)
        return declined("tt.dot",
                        "a folded operand's base pointer has no name");
      od.base = *base;

      // A memo declared f32 over a half operand rounds differently than the
      // un-memoised arm, so read the pointee's element once, here.
      const std::optional<agpu::ElemType> oe =
          elemTypeOf(sf.addend.window.base.getType());
      if (!oe)
        return declined("tt.dot", "a folded operand's element is unreadable");
      od.elem = *oe;

      am::Expr *opBaseOff = nullptr;
      if (const agpu::Decision d =
              baseOffsetExpr(sf.addend.window, "a folded operand", opBaseOff);
          !d.ok())
        return d;
      od.baseOffset = opBaseOff;
      switch (sf.addend.form) {
      case DrainAddend::Form::Row:
        od.kind = agpu::DrainOperand::Kind::Row;
        break;
      case DrainAddend::Form::Col:
        od.kind = agpu::DrainOperand::Kind::Col;
        break;
      default:
        if (const agpu::Decision d =
                strideOf(sf.addend.window, "a folded operand", od.leadingDim);
            !d.ok())
          return d;
        od.kind = agpu::DrainOperand::Kind::Tile;
        break;
      }
      break;
    }
    }
    return agpu::Decision::emitted();
  };

  // A folded unary may need a prelude helper (erf has no Metal spelling).
  // The op's own handler never runs once the drain absorbs it.
  const auto requireHelperFor = [&](Operation *op) {
    const auto name = op->getName().getStringRef();
    agpu::MathFn fn;
    if (agpu::epilogueUnaryFnOf({name.data(), name.size()}, fn))
      agpu_.helpers.require(fn);
  };

  for (const DrainStepFact &sf : ops.shape.cSteps) {
    agpu::DrainStep step;
    step.op = sf.op->getName().getStringRef().str();
    step.roundBefore = sf.roundBefore;
    requireHelperFor(sf.op);
    if (sf.kind == DrainStepFact::Operand::AccChain) {
      step.operand.kind = agpu::DrainOperand::Kind::AccChain;
      step.branchBase = sf.branchBase;
      for (const DrainBranchLinkFact &lf : sf.branch) {
        agpu::DrainBranchLink link;
        link.op = lf.op->getName().getStringRef().str();
        requireHelperFor(lf.op);
        if (const agpu::Decision d = resolveOperand(lf, link.operand); !d.ok())
          return d;
        step.branch.push_back(std::move(link));
      }
    } else if (const agpu::Decision d = resolveOperand(sf, step.operand);
               !d.ok()) {
      return d;
    }
    in.cSteps.push_back(step);
  }
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::resolveDirectCStore(const DotOperands &ops,
                                                const agpu::Plan &plan,
                                                agpu::DotInputs &in) {
  const am::Str *base = body_.sym.scalarName(idOf(ops.shape.cDevice.base));
  if (!base)
    return declined("tt.dot", "C's device window has no base name");
  if (const agpu::Decision d =
          strideOf(ops.shape.cDevice, "C", in.cStore.leadingDim);
      !d.ok())
    return d;
  in.cStore.base = *base;
  if (const std::optional<agpu::ElemType> ce =
          elemTypeOf(ops.shape.cDevice.base.getType()))
    in.cStore.elem = *ce;

  // Not a derived pointer: a fused drain stores after the loop this dot sits
  // in, outside any block declared here.
  if (const agpu::Decision d =
          baseOffsetExpr(ops.shape.cDevice, "C", in.cStore.baseOffset);
      !d.ok())
    return d;
  if (ops.shape.cDevice.rowStart) {
    const am::Str *rs = body_.sym.scalarName(idOf(ops.shape.cDevice.rowStart));
    if (!rs)
      return declined("tt.dot", "C's device window row offset has no name");
    in.cStore.rowStart =
        windowRowStartExpr(agpu_.context(), *rs, ops.shape.cDevice.rowStartMod);
  }
  if (ops.shape.cDevice.colStart) {
    const am::Str *cs = body_.sym.scalarName(idOf(ops.shape.cDevice.colStart));
    if (!cs)
      return declined("tt.dot", "C's device window column offset has no name");
    in.cStore.colStart = agpu_.context().var(*cs);
  }

  const auto boundExpr = [&](const AxisBound &b, am::Expr *&out) -> bool {
    if (!b.present)
      return true;
    if (!b.limit) {
      out = agpu_.context().lit(b.constant);
      return true;
    }
    const am::Str *n = body_.sym.scalarName(idOf(b.limit));
    if (n)
      out = agpu_.context().var(*n);
    return n != nullptr;
  };
  // A start scalar that took the licensed clamp needs no bound: the whole
  // window is already inside.
  const auto clamped = [&](Value start) {
    if (!start || clampPoison_.count(start) || !body_.clampApplied.count(start))
      return false;
    for (const WindowBounds::Clamp &cl : ops.shape.cClamps)
      if (cl.start == start)
        return true;
    return false;
  };
  const AxisBound cRow =
      clamped(ops.shape.cDevice.rowStart) ? AxisBound{} : ops.shape.cRowBound;
  const AxisBound cCol =
      clamped(ops.shape.cDevice.colStart) ? AxisBound{} : ops.shape.cColBound;
  if (!boundExpr(cRow, in.cStore.rowBound) ||
      !boundExpr(cCol, in.cStore.colBound))
    return declined("tt.dot", "the store's mask bound has no emitted name");
  in.cStore.tileRows = plan.facts.M;
  in.cStore.tileCols = plan.facts.N;
  if (plan.edgeScratchFits())
    in.cStore.edgeScratch = in.direct.poolE;

  if (ops.shape.cUniformGuard) {
    const am::Str *g = body_.sym.scalarName(idOf(ops.shape.cUniformGuard));
    if (!g)
      return declined("tt.dot", "the store's uniform mask has no emitted name");
    in.cStore.uniformGuard = agpu_.context().var(*g);
  }

  return resolveDrainSteps(ops, in);
}

} // namespace mlir::triton::applegpu::bridge
