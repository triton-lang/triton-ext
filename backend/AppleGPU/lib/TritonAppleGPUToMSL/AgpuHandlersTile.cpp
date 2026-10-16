// Histogram and gather: the two ops that stage a whole tile through the pool.
#include "AgpuEmitter.h"

#include "agpu/emit/EmitBand.h"
#include "agpu/emit/EmitGather.h"
#include "agpu/emit/EmitHistogram.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

agpu::Decision AgpuEmitter::emitGatherOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  if (o.operands.size() != 2 || o.results.size() != 1)
    return declined("tt.gather", "unexpected operand or result count");
  const Value res = mlirValueOf(o.results[0]);
  const Value srcV = mlirValueOf(o.operands[0]);
  auto srcTy =
      srcV ? dyn_cast<RankedTensorType>(srcV.getType()) : RankedTensorType();
  auto resTy =
      res ? dyn_cast<RankedTensorType>(res.getType()) : RankedTensorType();
  if (!srcTy || !resTy)
    return declined("tt.gather", "operand or result is not a tensor");
  const int axis = (int)o.intAt(0);
  if (axis < 0 || axis >= srcTy.getRank())
    return declined("tt.gather", "the axis is out of range");

  const agpu::ElemType *elemP = elemOf(o.results[0]);
  if (!elemP)
    return declined("tt.gather", "result type was never recorded");

  const agpu::BandNames bnm;
  const am::Str buf = body_.pool.use(bnm.buffer);
  if (buf.empty())
    return declined("tt.gather", "the scratch region was never carved");

  const agpu::TileView srcView = rowMajorViewOf(srcTy);
  const agpu::CoordSource srcCoords = coordSourceOf(srcTy);
  if ((int)srcCoords.dims.size() != srcTy.getRank())
    return declined("tt.gather", "the source has no per-thread "
                                 "coordinates");
  const Ready src = readyForCounted(o, 0, 1, registerCount(srcTy),
                                    "a source register has no name");
  if (!src.ok())
    return src.why;

  cur_->push_back(mc.hardBarrier());
  for (int64_t r = 0; r < src.regs; ++r)
    cur_->push_back(mc.assign(
        mc.subscript(mc.var(buf),
                     agpu::offsetExprOf(mc, srcView, (int)r, srcCoords)),
        mc.var(src.ops[0].at(r))));
  cur_->push_back(mc.hardBarrier());

  const agpu::CoordSource resCoords = coordSourceOf(resTy);
  if ((int)resCoords.dims.size() != resTy.getRank())
    return declined("tt.gather", "the result has no per-thread "
                                 "coordinates");
  const llvm::ArrayRef<int64_t> srcShape = srcTy.getShape();

  const Ready idx = readyForCounted(o, 1, 2, registerCount(resTy),
                                    "an index register has no name");
  if (!idx.ok())
    return idx.why;

  am::SmallVec<am::Expr *, 8> offsets;
  for (int64_t r = 0; r < idx.regs; ++r) {
    const am::Str &ix = idx.ops[1].at(r);
    std::vector<am::Expr *> coord;
    for (int d = 0; d < (int)srcShape.size(); ++d)
      coord.push_back(d == axis ? agpu::gatherIndexExpr(mc, ix, srcShape[axis])
                                : resCoords.of(mc, (int)r, d));

    offsets.push_back(srcView.linearize<am::Expr *>(
        coord,
        [&](am::Expr *t, int64_t s) {
          return mc.binary(am::BinOp::Mul, t, mc.lit(s));
        },
        [&](am::Expr *a, am::Expr *b) {
          return mc.binary(am::BinOp::Add, a, b);
        },
        [&](int64_t k) { return mc.lit(k); }));
  }

  am::SmallVec<am::Str, 8> outNames;
  agpu::ValueNames bound;
  for (int64_t r = 0, n = registerCount(resTy); r < n; ++r) {
    const am::Str name = nameFor('g', o.results[0], r);
    outNames.push_back(name);
    bound.push_back(name);
  }
  agpu::emitGather(mc, *cur_, buf, offsets, outNames, *elemP);
  body_.sym.bindRegs(o.results[0], std::move(bound));
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitHistogramOp(const agpu::OpView &o) {
  am::Context &mc = agpu_.context();
  if (o.operands.empty() || o.results.size() != 1)
    return declined("tt.histogram", "unexpected operand or result count");
  const Value res = mlirValueOf(o.results[0]);
  const Value src = mlirValueOf(o.operands[0]);
  auto resTy =
      res ? dyn_cast<RankedTensorType>(res.getType()) : RankedTensorType();
  auto srcTy =
      src ? dyn_cast<RankedTensorType>(src.getType()) : RankedTensorType();
  if (!resTy || !srcTy)
    return declined("tt.histogram", "operand or result is not a tensor");

  // Threads differing only in bits that do not move the source index hold
  // the same element; counting per thread would multiply every bin.
  const agpu::AddressSpread spread = spreadOf(src);
  const agpu::HistogramPlan plan = agpu::planHistogram(
      tileElemCount(resTy), numWarps(), spread.laneFree, spread.warpFree);

  agpu::HistogramNames hnm;
  const am::Str region = body_.pool.use(hnm.bins);
  if (region.empty())
    return declined("tt.histogram", "the bin region was never carved");

  const am::Type binPtr =
      am::atomicPtr(am::Scalar::U32, am::AddrSpace::Threadgroup);
  hnm.bins = region + "_a";
  cur_->push_back(
      mc.declStmt(binPtr, hnm.bins, mc.cast(binPtr, mc.var(region))));

  agpu::emitHistogramZero(mc, *cur_, plan, hnm);

  const Ready source = readyForCounted(o, 0, 1, registerCount(srcTy),
                                       "a source register has no name");
  if (!source.ok())
    return source.why;

  am::SmallVec<am::Str, 8> srcRegs;
  for (int64_t r = 0; r < source.regs; ++r)
    srcRegs.push_back(source.ops[0].at(r));

  am::SmallVec<am::Str, 8> maskRegs;
  if (o.operands.size() > 1)
    for (std::size_t r = 0; r < body_.sym.regCount(o.operands[1]); ++r) {
      const am::Str *m = body_.sym.regAt(o.operands[1], r);
      if (!m)
        return declined("tt.histogram", "a mask register has no name");
      maskRegs.push_back(*m);
    }

  agpu::emitHistogramCount(mc, *cur_, plan, srcRegs, hnm, maskRegs);

  const agpu::ElemType *elemP = elemOf(o.results[0]);
  if (!elemP)
    return declined("tt.histogram", "result type was never recorded");
  const agpu::CoordSource resCoords = coordSourceOf(resTy);
  if ((int)resCoords.dims.size() != resTy.getRank())
    return declined("tt.histogram", "the result has no per-thread "
                                    "coordinates");
  const agpu::TileView resView = rowMajorViewOf(resTy);

  agpu::ValueNames names;
  for (int64_t r = 0, n = registerCount(resTy); r < n; ++r) {
    const am::Str name = nameFor('h', o.results[0], r);
    am::Expr *idx = agpu::offsetExprOf(mc, resView, (int)r, resCoords);
    cur_->push_back(mc.declStmt(
        agpu::mslTypeOf(*elemP), name,
        mc.cast(agpu::mslTypeOf(*elemP),
                mc.call(am::builtin::atomic::Load,
                        {mc.addrOf(mc.subscript(mc.var(hnm.bins), idx)),
                         mc.var(am::builtin::order::Relaxed)}))));
    names.push_back(name);
  }
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerTileHandlers() {
  table_.add("histogram",
             agpu::forOps({"tt.histogram"}, [this](const agpu::OpView &o) {
               return emitHistogramOp(o);
             }));

  table_.add("gather",
             agpu::forOps({"tt.gather"}, [this](const agpu::OpView &o) {
               return emitGatherOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
