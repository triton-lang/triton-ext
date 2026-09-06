// What a tt.dot is: its operands, its shape and the facts the planner reads.
#include "AgpuDotChain.h"
#include "AgpuEmitter.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;
am::Str AgpuEmitter::derivedDevicePointer(const am::Str &base, am::Expr *offset,
                                          const agpu::ElemType &elem,
                                          am::Str name) {
  am::Context &mc = agpu_.context();
  const am::Type ptr = agpu::mslTypeOf(elem).pointerTo(am::AddrSpace::Device);
  cur_->push_back(
      mc.declStmt(ptr, name, mc.binary(am::BinOp::Add, mc.var(base), offset)));
  return name;
}

DotOperands AgpuEmitter::dotOperandsOf(const agpu::OpView &o) {
  DotOperands d;
  if (o.operands.size() < 3 || o.results.size() != 1) {
    d.why = declined("tt.dot", "expected 3 operands and 1 result, got " +
                                   std::to_string(o.operands.size()) + " and " +
                                   std::to_string(o.results.size()));
    return d;
  }

  const agpu::ValueId ids[3] = {o.operands[0], o.operands[1], o.results[0]};
  RankedTensorType *tys[3] = {&d.shape.aTy, &d.shape.bTy, &d.shape.cTy};
  agpu::ElemType *elems[3] = {&d.shape.aElem, &d.shape.bElem, &d.shape.cElem};

  for (int i = 0; i < 3; ++i) {
    const Value v = mlirValueOf(ids[i]);
    const agpu::ElemType *e = elemOf(ids[i]);
    if (!v || !e) {
      d.why = declined("tt.dot", "an operand's type was never recorded");
      return d;
    }
    *tys[i] = dyn_cast<RankedTensorType>(v.getType());
    *elems[i] = *e;
  }

  if (!d.shape.aTy || !d.shape.bTy || !d.shape.cTy ||
      d.shape.cTy.getRank() < 2 || d.shape.cTy.getRank() > 3 ||
      d.shape.aTy.getRank() != d.shape.cTy.getRank() ||
      d.shape.bTy.getRank() != d.shape.cTy.getRank()) {
    d.why = declined("tt.dot", "operands are not 2-D or batched 3-D");
    return d;
  }

  d.shape.aDevice = deviceTileOf(mlirValueOf(o.operands[0]));

  const agpu::ValueId ops[2] = {o.operands[0], o.operands[1]};
  agpu::ValueId *stage[2] = {&d.aStage, &d.bStage};
  RankedTensorType *stageTy[2] = {&d.aStageTy, &d.bStageTy};
  for (int i = 0; i < 2; ++i) {
    *stage[i] = ops[i];
    *stageTy[i] = i == 0 ? d.shape.aTy : d.shape.bTy;

    const Value through = throughLayoutChange(mlirValueOf(ops[i]));
    auto ty = dyn_cast<RankedTensorType>(through.getType());
    if (!ty)
      continue;
    const agpu::ValueId id = idOf(through);
    if (body_.sym.regAt(id, 0)) {
      *stage[i] = id;
      *stageTy[i] = ty;
    }
  }

  fillAccumulatorCarry(d.shape, mlirValueOf(o.operands[2]),
                       mlirValueOf(o.results[0]));

  // Where the result lands: a single-use convert_layout or a fused loop's
  // scf.yield may absorb the readback instead.
  d.cOut = o.results[0];
  d.cOutTy = d.shape.cTy;

  const bool fused =
      d.shape.cCarried &&
      agpu_.planFor(dotFactsOf(d.shape)).accumulatorsOutlivePass();

  // A fused dot's fragments start at zero, so its incoming C is the loop init.
  const Value fusedInit = fused ? fusedInitOf(d.shape.cCarried) : Value();
  const bool addsC =
      fused ? !zeroSplat(fusedInit) : !isZeroTensor(o.operands[2]);

  const auto landingTy = [&](Value v) -> RankedTensorType {
    auto ty = v ? dyn_cast<RankedTensorType>(v.getType()) : RankedTensorType();
    // `cIn` is indexed by register in C's layout, so a readback that adds
    // needs an interchangeable layout.
    if (ty && (int)coordSourceOf(ty).dims.size() == ty.getRank() &&
        registerCoordAt(ty, 0) &&
        (!addsC || layoutsInterchangeable(d.shape.cTy, ty)))
      return ty;
    return {};
  };
  if (fused) {
    const Value afterLoop = convertAfter(d.shape.cCarried);
    Value landing = afterLoop ? afterLoop : d.shape.cCarried;
    if (landing && !landingTy(landing))
      landing = d.shape.cCarried;
    if (const RankedTensorType ty = landingTy(landing)) {
      d.cOut = idOf(landing);
      d.cOutTy = ty;
    }
  } else if (const Value landing = readbackLandingOf(d.shape)) {
    d.cOut = idOf(landing);
    d.cOutTy = cast<RankedTensorType>(landing.getType());
  }

  if (addsC) {
    const agpu::ValueId incoming = fused ? idOf(fusedInit) : o.operands[2];
    const Operand acc(body_.sym, incoming, registerCount(d.shape.cTy));
    if (!acc.ok()) {
      d.why =
          declined("tt.dot", "the incoming accumulator has no register names");
      return d;
    }
    for (int64_t r = 0; r < acc.registers(); ++r)
      d.cIn.push_back(acc.at(r));
  }
  return d;
}

bool AgpuEmitter::isZeroTensor(agpu::ValueId v) const {
  const auto it = constantFor_.find(v);
  if (it == constantFor_.end() || it->second.empty())
    return false;
  for (const ConstantValue &k : it->second) {
    if (!k.known)
      return false;
    if (k.isFloat ? k.f != 0.0 : k.i != 0)
      return false;
  }
  return true;
}

DotShape AgpuEmitter::dotShapeOf(triton::DotOp dot) const {
  DotShape d;
  d.aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
  d.bTy = dyn_cast<RankedTensorType>(dot.getB().getType());
  d.cTy = dyn_cast<RankedTensorType>(dot.getC().getType());
  if (!d.aTy || !d.bTy || !d.cTy)
    return {};

  const std::optional<agpu::ElemType> aE = elemTypeOf(dot.getA().getType());
  const std::optional<agpu::ElemType> bE = elemTypeOf(dot.getB().getType());
  const std::optional<agpu::ElemType> cE = elemTypeOf(dot.getC().getType());
  if (!aE || !bE || !cE)
    return {};
  d.aElem = *aE;
  d.bElem = *bE;
  d.cElem = *cE;
  d.aDevice = deviceTileOf(dot.getA());
  fillAccumulatorCarry(d, dot.getC(), dot.getResult());
  return d;
}

Value AgpuEmitter::readbackLandingOf(const DotShape &shape) {
  if (!shape.cResult || !shape.cTy)
    return {};
  const Value landing = convertAfter(shape.cResult);
  auto ty = landing ? dyn_cast<RankedTensorType>(landing.getType())
                    : RankedTensorType();
  if (!ty || (int)coordSourceOf(ty).dims.size() != ty.getRank() ||
      !registerCoordAt(ty, 0))
    return {};
  // `cIn` is indexed by register in C's layout, so a readback that adds needs
  // an interchangeable layout.
  const bool addsC = !isZeroTensor(idOf(shape.cInput));
  if (addsC && !layoutsInterchangeable(shape.cTy, ty))
    return {};
  return landing;
}

RankedTensorType AgpuEmitter::renameLandingTypeOf(const DotShape &shape) {
  const Value landing = readbackLandingOf(shape);
  if (!landing)
    return shape.cTy;
  const auto landTy = cast<RankedTensorType>(landing.getType());
  if (layoutsInterchangeable(shape.cTy, landTy))
    return landTy;
  return shuffleFor(shape.cTy, landTy).usable() ? shape.cTy : landTy;
}

agpu::DotFacts AgpuEmitter::dotFactsOf(const DotShape &shape) {
  agpu::DotFacts f;
  // A shape whose types were unreadable leaves M, N and K at zero, so the
  // facts read as unusable and no null type is dereferenced.
  if (!shape.aTy || !shape.bTy || !shape.cTy)
    return f;

  const int rank = shape.cTy.getRank();
  f.rank = rank;
  f.Bd = rank > 2 ? shape.cTy.getShape()[0] : 1;
  f.M = shape.cTy.getShape()[rank - 2];
  f.N = shape.cTy.getShape()[rank - 1];
  f.K = shape.aTy.getShape()[rank - 1];
  f.numWarps = numWarps();
  f.aElemBytes = agpu::byteWidthOf(shape.aElem);
  f.bElemBytes = agpu::byteWidthOf(shape.bElem);
  f.intAcc = shape.cElem.kind == agpu::ElemType::Kind::Int;
  f.carriedAcc = shape.accumulatorIsCarried();

  f.fusedAcc = shape.accumulatorOutlivesLoop();
  f.cInitNonzero = f.fusedAcc && !zeroSplat(fusedInitOf(shape.cCarried));
  f.aInPlace = f.bInPlace = false;
  f.aDirect = (bool)shape.aDevice.base;
  if (!f.fusedAcc) {
    const RankedTensorType renameTy = renameLandingTypeOf(shape);
    f.cDims = coordSourceOf(renameTy).dims;
    f.cRegs = registerCount(renameTy);
  }

  // `cElem` is the fp32 accumulator. The tensor behind the store window may
  // be narrower and the pointee's width decides whether a direct
  // simdgroup_store type-checks.
  const std::optional<agpu::ElemType> cStored =
      shape.cDevice.base ? elemTypeOf(shape.cDevice.base.getType())
                         : std::nullopt;
  if (cStored)
    f.cElemBytes = agpu::byteWidthOf(*cStored);

  // Spellability is checked here because the direct drain has no fallback
  // arm: a name missing at the drain declines the kernel. Checked at the
  // dot's own site (the K loop when C is carried): names computed after the
  // loop are not bound yet when the dot inside it emits.
  Operation *const drainSite =
      shape.cCarried ? shape.cCarried.getDefiningOp() : shape.cStore;
  const auto spelled = [&](Value v) {
    return !v || (drainSite && willHaveScalarName(v, drainSite));
  };
  const auto boundSpelled = [&](const AxisBound &b) {
    return !b.present || !b.limit || spelled(b.limit);
  };
  const auto splatSpelled = [&](Value splat) {
    return !splat || spelled(splat);
  };
  const auto windowSpelled = [&](const DrainAddend &a) {
    return spelled(a.window.base) && spelled(a.window.baseOffset) &&
           (a.form == DrainAddend::Form::Row ||
            a.form == DrainAddend::Form::Col || spelled(a.window.rowStride));
  };
  const auto stepSpelled = [&](const DrainStepFact &sf) -> bool {
    switch (sf.kind) {
    case DrainStepFact::Operand::None:
      return true;
    case DrainStepFact::Operand::Splat:
      return splatSpelled(sf.splat);
    case DrainStepFact::Operand::Window:
      return windowSpelled(sf.addend);
    case DrainStepFact::Operand::AccChain:
      return llvm::all_of(sf.branch, [&](const DrainBranchLinkFact &l) {
        switch (l.kind) {
        case DrainBranchLinkFact::Operand::None:
          return true;
        case DrainBranchLinkFact::Operand::Splat:
          return splatSpelled(l.splat);
        case DrainBranchLinkFact::Operand::Window:
          return windowSpelled(l.addend);
        }
        return false;
      });
    }
    return false;
  };
  const bool cDrainSpelled =
      (!shape.cDevice.base ||
       (spelled(shape.cDevice.base) && spelled(shape.cDevice.rowStride) &&
        spelled(shape.cDevice.baseOffset) && spelled(shape.cDevice.rowStart) &&
        spelled(shape.cDevice.colStart))) &&
      boundSpelled(shape.cRowBound) && boundSpelled(shape.cColBound) &&
      spelled(shape.cUniformGuard) && llvm::all_of(shape.cSteps, stepSpelled);

  // `spelled` reads bindings, which only accumulate, so this answer moves as
  // emission proceeds. The first ask is scanPool's, before any binding
  // exists, and its answer is pinned here so the clamps the scan records and
  // the drain the dot takes cannot disagree. Pinning the scan's answer is
  // sound because bindings only accumulate: a scan-time true never turns
  // false at the dot.
  const bool direct =
      shape.cStore && cStored && cDrainSpelled && !f.raggedM() && !f.raggedN();
  if (!shape.cStore) {
    f.cDirect = false;
    return f;
  }
  f.cDirect = cDirectOf_.try_emplace(shape.cStore, direct).first->second;

  return f;
}

} // namespace mlir::triton::applegpu::bridge
