// Elementwise, comparison, select, negate, math and cast handlers.
#include "AgpuEmitter.h"
#include "AgpuEnums.h"
#include "AgpuOpTables.h"

#include "agpu/emit/EmitConvert.h"
#include "agpu/emit/EmitElementwise.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

am::Str AgpuEmitter::castTo(const agpu::ElemType &to, const am::Str &src) {
  am::Context &mc = agpu_.context();
  const am::Str name = "p" + src + "_" + std::to_string(body_.tempSeq++);
  cur_->push_back(mc.declStmt(agpu::mslTypeOf(to), name,
                              mc.cast(agpu::mslTypeOf(to), mc.var(src))));
  return name;
}

am::Str AgpuEmitter::inIrType(agpu::ValueId v, int64_t r) {
  const am::Str *name = body_.sym.regAt(v, r);
  if (!name)
    return am::Str();
  const auto it = declaredFor_.find(v);
  if (it == declaredFor_.end())
    return *name;
  const agpu::ElemType *ir = elemOf(v);
  if (!ir || it->second == *ir)
    return *name;
  return castTo(*ir, *name);
}

agpu::Decision AgpuEmitter::emitReinterpretCast(const agpu::OpView &o,
                                                const Ready &ready,
                                                const Operand &a,
                                                agpu::ElemType from,
                                                agpu::ElemType to) {
  am::Context &mc = agpu_.context();
  // Held type: with the pointee type, ptr<i1>->ptr<i8> reads as a width
  // change.
  const Value fromV = mlirValueOf(o.operands[0]);
  const Value toV = mlirValueOf(o.results[0]);
  if (fromV)
    if (const std::optional<agpu::ElemType> h = heldTypeOf(fromV.getType()))
      from = *h;
  if (toV)
    if (const std::optional<agpu::ElemType> h = heldTypeOf(toV.getType()))
      to = *h;

  const am::Type toTy = agpu::mslTypeOf(to);
  const bool fromPtr = from.isPointer();
  const bool toPtr = to.isPointer();

  bool sameStride = fromPtr && toPtr;
  if (sameStride && fromV && toV) {
    const std::optional<agpu::ElemType> fp = elemTypeOf(fromV.getType());
    const std::optional<agpu::ElemType> tp = elemTypeOf(toV.getType());
    sameStride = fp && tp && agpu::byteWidthOf(*fp) == agpu::byteWidthOf(*tp);
  }

  // Rename when the MSL spellings agree or both sides are pointers of
  // one element width. A pointer's address is not in its register name,
  // so offsets are inherited too.
  if (sameStride || agpu::mslTypeOf(from) == toTy) {
    agpu::ValueNames names;
    for (int64_t r = 0; r < ready.regs; ++r)
      names.push_back(a.at(r));
    body_.sym.bindRegs(o.results[0], std::move(names));

    if (toPtr) {
      for (int64_t r = 0; r < ready.regs; ++r)
        inheritOffset(o.operands[0], r, o.results[0], r);
      inheritBasePointer(o.operands[0], o.results[0]);
    }
    return agpu::Decision::emitted();
  }

  // Metal has no as_type between an integer and a pointer. Both are 64
  // bits, so a value cast loses nothing.
  if (fromPtr || toPtr) {
    if (toPtr)
      markBasePointer(o.results[0]);
    return emitPerRegister(o, ready.regs, to, 'b', [&](int64_t r) {
      RegValue v;
      am::Expr *addr = fromPtr ? addressAt(o.operands[0], r) : nullptr;
      v.value = mc.cast(toTy, addr ? mc.addrOf(addr) : mc.var(a.at(r)));
      return v;
    });
  }

  if (from.bits != to.bits)
    return declined(o.name, "reinterpret between different widths");
  return emitPerRegister(o, ready.regs, to, 'b', [&](int64_t r) {
    RegValue v;
    const am::Str src = inIrType(o.operands[0], r);
    v.value = mc.bitcast(toTy, mc.var(src.empty() ? a.at(r) : src));
    return v;
  });
}

agpu::Decision
AgpuEmitter::emitConvertCast(const agpu::OpView &o, const Ready &ready,
                             const Operand &a, const agpu::ElemType &from,
                             const agpu::ElemType &to,
                             const agpu::ElemType &fromDeclared) {
  am::Context &mc = agpu_.context();
  agpu::Rounding rounding = agpu::Rounding::Default;
  if (o.name == kFpToFp && !o.ints.empty())
    rounding = roundingFor(o.intAt(0));

  const agpu::ConvertPlan p = agpu::planConvert(from, to, rounding);
  const agpu::Decision d = agpu::convertDecision(p);
  if (!d.ok())
    return d;

  if (p.kind == agpu::ConvertKind::None) {
    agpu::ValueNames names;
    for (int64_t r = 0; r < ready.regs; ++r)
      names.push_back(a.at(r));
    body_.sym.bindRegs(o.results[0], std::move(names));
    return agpu::Decision::emitted();
  }

  agpu_.helpers.require(p);

  return emitPerRegister(o, ready.regs, to, 'x', [&](int64_t r) {
    RegValue v;
    // Read at the promoted type first so extui zero-extends.
    am::Expr *src = mc.var(a.at(r));
    if (!(from == fromDeclared))
      src = mc.cast(agpu::mslTypeOf(from), src);
    v.value = agpu::convertExpr(mc, p, src, to);
    return v;
  });
}

agpu::Decision AgpuEmitter::emitCastOp(const agpu::OpView &o) {
  const CastName *cn = castFor(o.name);
  if (!cn)
    return agpu::Decision::notMine();

  const Ready ready = readyFor(o, 1);
  if (!ready.ok())
    return ready.why;
  const agpu::ElemType *fromP = elemOf(o.operands[0]);
  if (!fromP)
    return declined(o.name, "operand type was never recorded");

  const Operand &a = ready[0];

  // extui reads its source as unsigned; fptoui declares its result
  // unsigned.
  agpu::ElemType from = *fromP;
  if (cn->readsUnsigned)
    from.isUnsigned = true;
  agpu::ElemType to = ready.elem;
  if (cn->writesUnsigned)
    to.isUnsigned = true;

  if (cn->reinterpret)
    return emitReinterpretCast(o, ready, a, from, to);

  return emitConvertCast(o, ready, a, from, to, declaredOf(o.operands[0]));
}

agpu::Decision AgpuEmitter::emitMath3Op(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const Math3Name *m = math3For(o.name);
  if (!m)
    return declined(o.name, "no three-operand math row");
  const Ready ready = readyFor(o, 3);
  if (!ready.ok())
    return ready.why;
  const agpu::Decision d = agpu::checkMath3(m->fn, ready.elem);
  if (!d.ok())
    return d;

  const Operand &v0 = ready[0];
  const Operand &v1 = ready[1];
  const Operand &v2 = ready[2];

  const bool propagateNan = o.intAt(0, 0) != 0;

  return emitPerRegister(o, ready.regs, ready.elem, 'm', [&](int64_t r) {
    RegValue v;
    v.value = agpu::clampExpr(mc, m->fn, ready.elem, v0.at(r), mc.var(v1.at(r)),
                              mc.var(v2.at(r)), propagateNan);
    return v;
  });
}

agpu::Decision AgpuEmitter::emitMath2Op(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const Math2Name *m = math2For(o.name);
  if (!m)
    return agpu::Decision::notMine();

  const Ready ready = readyFor(o, 2);
  if (!ready.ok())
    return ready.why;
  const agpu::ElemType *operandP = elemOf(o.operands[0]);
  if (!operandP)
    return declined(o.name, "operand type was never recorded");

  // MLIR integers are signless; `arith.maxui` is what says to read this
  // one as unsigned.
  agpu::ElemType operand = *operandP;
  if (m->readsUnsigned)
    operand.isUnsigned = true;

  const agpu::Decision d = agpu::checkMath2(m->fn, operand);
  if (!d.ok())
    return d;

  const Operand &a = ready[0];
  const Operand &b = ready[1];

  const bool promote = !(operand == *operandP);
  return emitPerRegister(o, ready.regs, operand, 'm', [&](int64_t r) {
    RegValue v;
    // MSL picks the overload from argument types, so the promotion must reach
    // the operands as well as the result declaration.
    const am::Str an = promote ? castTo(operand, a.at(r)) : a.at(r);
    const am::Str bn = promote ? castTo(operand, b.at(r)) : b.at(r);
    if (m->fn == agpu::MathFn2::Min || m->fn == agpu::MathFn2::Max) {
      v.value = agpu::minMaxExpr(mc, m->fn, operand, an, bn, m->propagateNan);
      return v;
    }
    v.value = agpu::mathExpr(mc, m->fn, mc.var(an), mc.var(bn));
    return v;
  });
}

agpu::Decision AgpuEmitter::emitMath1Op(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  agpu::MathFn fn;
  if (!mathFnFor(o.name, fn))
    return agpu::Decision::notMine();

  const Ready ready = readyFor(o, 1);
  if (!ready.ok())
    return ready.why;
  const agpu::ElemType *operandP = elemOf(o.operands[0]);
  if (!operandP)
    return declined(o.name, "operand type was never recorded");
  const agpu::Decision d = agpu::checkMath(fn, *operandP);
  if (!d.ok())
    return d;
  // erf and cbrt have no Metal spelling; their name is a prelude helper.
  agpu_.helpers.require(fn);

  const Operand &a = ready[0];

  const agpu::ElemType result = agpu::mathResultType(fn, *operandP);
  return emitPerRegister(o, ready.regs, result, 'm', [&](int64_t r) {
    RegValue v;
    v.value = agpu::mathExpr(mc, fn, *operandP, mc.var(a.at(r)));
    return v;
  });
}

agpu::Decision AgpuEmitter::emitNegateOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const Ready ready = readyFor(o, 1);
  if (!ready.ok())
    return ready.why;
  const Operand &a = ready[0];

  return emitPerRegister(o, ready.regs, ready.elem, 'n', [&](int64_t r) {
    RegValue v;
    v.value = mc.unary(am::UnOp::Neg, mc.var(a.at(r)));
    return v;
  });
}

agpu::Decision AgpuEmitter::emitSelectOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  const Ready ready = readyFor(o, 3);
  if (!ready.ok())
    return ready.why;
  const Operand &c = ready[0];
  const Operand &t = ready[1];
  const Operand &f = ready[2];

  // Declared in the type the operands are held in. For a pointer,
  // ready.elem is the pointee type.
  const Value res = mlirValueOf(o.results[0]);
  const std::optional<agpu::ElemType> held =
      res ? heldTypeOf(res.getType()) : std::optional<agpu::ElemType>();
  const agpu::ElemType elem = held ? *held : ready.elem;

  return emitPerRegister(o, ready.regs, elem, 's', [&](int64_t r) {
    RegValue v;
    const am::Str tn = inIrType(o.operands[1], r);
    const am::Str fn = inIrType(o.operands[2], r);
    v.value = mc.ternary(mc.var(c.at(r)), mc.var(tn.empty() ? t.at(r) : tn),
                         mc.var(fn.empty() ? f.at(r) : fn));
    return v;
  });
}

agpu::Decision AgpuEmitter::emitCompareFOp(const agpu::OpView &o) {
  agpu::FCmp pred;
  if (!fcmpPredFor(o.intAt(0), pred))
    return declined(o.name, "unhandled float comparison predicate");

  const Ready ready = readyFor(o, 2);
  if (!ready.ok())
    return ready.why;
  const Operand &a = ready[0];
  const Operand &b = ready[1];

  return emitPerRegister(o, ready.regs, agpu::i1(), 'c', [&](int64_t r) {
    RegValue v;
    v.value = agpu::fcmpExpr(agpu_.context(), pred, a.at(r), b.at(r));
    return v;
  });
}

agpu::Decision AgpuEmitter::emitCompareOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  agpu::EwOp ew;
  if (!cmpOpFor(o.name, o.intAt(0), ew))
    return declined(o.name, "unhandled comparison predicate");

  const Ready ready = readyFor(o, 2);
  if (!ready.ok())
    return ready.why;
  const agpu::ElemType *operandP = elemOf(o.operands[0]);
  if (!operandP)
    return declined(o.name, "operand type was never recorded");

  const Operand &a = ready[0];
  const Operand &b = ready[1];

  const agpu::EwTypes t = agpu::typesFor(ew, *operandP);
  return emitPerRegister(o, ready.regs, t.result, 'c', [&](int64_t r) {
    RegValue v;
    v.value = agpu::ewExpr(mc, ew, *operandP, mc.var(a.at(r)), mc.var(b.at(r)));
    return v;
  });
}

agpu::Decision AgpuEmitter::emitElementwiseOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  agpu::EwOp ew;
  if (!ewOpFor(o.name, ew))
    return agpu::Decision::notMine();

  const Ready ready = readyFor(o, 2);
  if (!ready.ok())
    return ready.why;
  const Operand &a = ready[0];
  const Operand &b = ready[1];

  const agpu::ElemType operand =
      elemOf(o.operands[0]) ? *elemOf(o.operands[0]) : ready.elem;
  agpu::EwTypes t = agpu::typesFor(ew, operand);
  t.result = agpu::evalWidthFor(t.result);

  // Where the value is affine in a coordinate, emit register 0 and the
  // rest as literal deltas from it.
  {
    const auto uniform = [&](const Operand &x) {
      for (int64_t r = 1; r < ready.regs; ++r)
        if (x.at(r) != x.at(0))
          return false;
      return true;
    };
    const auto famOf = [&](std::size_t i) {
      const auto it = body_.affine.find(o.operands[i]);
      return it != body_.affine.end() ? it->second : agpu::AffineFamily{};
    };
    const auto constSplat = [&](std::size_t i, int64_t &out) {
      Value v = mlirValueOf(o.operands[i]);
      while (v)
        if (auto sp = v.getDefiningOp<SplatOp>())
          v = sp.getSrc();
        else
          break;
      APInt c;
      if (v && matchPattern(v, m_ConstantInt(&c))) {
        out = c.getSExtValue();
        return true;
      }
      DenseIntElementsAttr d;
      if (v && matchPattern(v, m_Constant(&d)) && d.isSplat()) {
        out = d.getSplatValue<APInt>().getSExtValue();
        return true;
      }
      return false;
    };

    const Value res = mlirValueOf(o.results[0]);
    auto rt =
        res ? dyn_cast<RankedTensorType>(res.getType()) : RankedTensorType();
    // Deltas are differences of registerCoordAt at lane 0; only valid
    // where register bases and runtime bases are disjoint.
    const bool foldable = rt && affineRegisterDeltas(rt, (int)ready.regs);
    agpu::AffineFamily fam;
    std::vector<int64_t> deltas;
    if ((ew == agpu::EwOp::RemS || ew == agpu::EwOp::RemU) && foldable) {
      [&] {
        Operation *op = res.getDefiningOp();
        if (!op)
          return;
        auto contig = op->getAttrOfType<DenseElementsAttr>("tt.contiguity");
        if (!contig || !contig.isSplat() || !uniform(b))
          return;
        std::vector<std::vector<int64_t>> coords;
        for (int64_t r = 0; r < ready.regs; ++r) {
          const std::optional<std::vector<int64_t>> c =
              registerCoordAt(rt, (int)r);
          if (!c)
            return;
          coords.push_back(*c);
        }
        const agpu::RemFold fold = agpu::planRemFold(
            coords,
            std::vector<int64_t>(rt.getShape().begin(), rt.getShape().end()),
            contig.getSplatValue<APInt>().getSExtValue());
        if (!fold.ok)
          return;
        fam = agpu::uniformFamily(rt.getRank());
        fam.scales[(std::size_t)fold.axis] = 1;
        deltas = fold.deltas;
      }();
    } else if (foldable) {
      int64_t ac = 0, bc = 0;
      const bool aC = constSplat(0, ac), bC = constSplat(1, bc);
      fam = agpu::foldFamily(ew, famOf(0), uniform(a), famOf(1), uniform(b),
                             aC ? &ac : nullptr, bC ? &bc : nullptr,
                             (int)rt.getRank());
      if (fam.ok() &&
          !scaledRegisterDeltas(rt, ready.regs, fam.scales, deltas)) {
        deltas.clear();
        fam = agpu::AffineFamily{};
      }
    }
    const bool anyDelta = [&] {
      for (int64_t d : deltas)
        if (d)
          return true;
      return false;
    }();
    if (fam.ok() && ready.regs > 1 && anyDelta) {
      body_.affine[o.results[0]] = fam;
      return emitPerRegister(o, ready.regs, t.result, 'e', [&](int64_t r) {
        RegValue v;
        if (r == 0) {
          v.value =
              agpu::ewExpr(mc, ew, operand, mc.var(a.at(0)), mc.var(b.at(0)));
          return v;
        }
        v.value =
            mc.binary(am::BinOp::Add, mc.var(nameFor('e', o.results[0], 0)),
                      mc.lit(deltas[(std::size_t)r]));
        return v;
      });
    }
    if (fam.ok())
      body_.affine[o.results[0]] = fam;
  }

  return emitPerRegister(o, ready.regs, t.result, 'e', [&](int64_t r) {
    RegValue v;
    v.value = agpu::ewExpr(mc, ew, operand, mc.var(a.at(r)), mc.var(b.at(r)));
    if (ready.regs == 1) {
      const Value res = mlirValueOf(o.results[0]);
      if (res && !clampPoison_.count(res))
        if (const auto it = clampOf_.find(res); it != clampOf_.end()) {
          v.value =
              mc.call(am::builtin::math::Min, {v.value, mc.lit(it->second)});
          body_.clampApplied.insert(res);
        }
    }
    return v;
  });
}

void AgpuEmitter::registerArithHandlers() {
  table_.add("elementwise",
             agpu::forOps(ewOpNames(), [this](const agpu::OpView &o) {
               return emitElementwiseOp(o);
             }));

  table_.add("compare",
             agpu::forOps(cmpOpNames(), [this](const agpu::OpView &o) {
               return emitCompareOp(o);
             }));

  // Half the predicates need an explicit isnan() term; fcmpExpr adds it.
  // Operands are passed as names because the guarded forms use each twice.
  table_.add("comparef",
             agpu::forOps({"arith.cmpf"}, [this](const agpu::OpView &o) {
               return emitCompareFOp(o);
             }));

  // The condition may be a scalar i1 selecting between two tensors, so its
  // registers are asked for at the result's count.
  table_.add("select",
             agpu::forOps({"arith.select"}, [this](const agpu::OpView &o) {
               return emitSelectOp(o);
             }));

  table_.add("negate",
             agpu::forOps({"arith.negf"}, [this](const agpu::OpView &o) {
               return emitNegateOp(o);
             }));

  table_.add("math1",
             agpu::forOps(mathOpNames(), [this](const agpu::OpView &o) {
               return emitMath1Op(o);
             }));

  // metal::min returns the other operand on a NaN (IEEE minNum), which is not
  // what arith.minimumf means; the table's propagateNan column selects the
  // guarded form.
  table_.add("math2",
             agpu::forOps(math2OpNames(), [this](const agpu::OpView &o) {
               return emitMath2Op(o);
             }));

  table_.add("math3",
             agpu::forOps(math3OpNames(), [this](const agpu::OpView &o) {
               return emitMath3Op(o);
             }));

  // planConvert picks static_cast vs rounding helper vs fp8 pack from the two
  // element types. Reinterpreting ops are handled below instead.
  table_.add("cast", agpu::forOps(castOpNames(), [this](const agpu::OpView &o) {
               return emitCastOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
