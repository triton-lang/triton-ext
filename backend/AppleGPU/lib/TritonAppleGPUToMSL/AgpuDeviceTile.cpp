#include "AgpuDeviceTile.h"

#include "agpu/msl/Containers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::applegpu::bridge {

// `convert_layout` redistributes elements among threads without changing
// which element an index names.
Value throughLayoutChange(Value v) {
  while (auto cvt = v.getDefiningOp<gpu::ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

namespace {

struct PtrAndOffset {
  Value ptr, offset;

  explicit operator bool() const { return (bool)ptr; }
};

// The value behind any stack of broadcasts and layout changes.
Value throughBroadcast(Value v) {
  while (true) {
    if (auto bc = v.getDefiningOp<BroadcastOp>()) {
      v = bc.getSrc();
      continue;
    }
    if (auto cvt = v.getDefiningOp<gpu::ConvertLayoutOp>()) {
      v = cvt.getSrc();
      continue;
    }
    return v;
  }
}

// `expand_dims(make_range)` along `axis`, possibly shifted by a splat scalar,
// which `start` receives. `mod` non-null admits the safety modulo
// `(start + iota) % bound`, accepted only when `tt.contiguity` guarantees the
// run does not wrap; the bound lands in `*mod`.
bool windowIota(Value v, int axis, int64_t extent, Value &start,
                int64_t *mod = nullptr);

// The scalar every element of a splat holds, or null.
Value splatScalar(Value v) {
  auto sp = throughBroadcast(v).getDefiningOp<SplatOp>();
  return sp ? sp.getSrc() : Value{};
}

PtrAndOffset addPtrParts(Value v) {
  if (auto ap = throughBroadcast(v).getDefiningOp<AddPtrOp>())
    return {ap.getPtr(), ap.getOffset()};
  return {};
}

bool denseIntSplatOf(Value v, int64_t &out);

bool windowIota(Value v, int axis, int64_t extent, Value &start, int64_t *mod) {
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    Value s = splatScalar(add.getLhs());
    Value rest = add.getRhs();
    if (!s) {
      s = splatScalar(add.getRhs());
      rest = add.getLhs();
    }
    if (!s || !windowIota(throughBroadcast(rest), axis, extent, start, mod))
      return false;
    if (start || (mod && *mod))
      return false;
    start = s;
    return true;
  }

  auto ed = v.getDefiningOp<ExpandDimsOp>();
  if (!ed || (int)ed.getAxis() == axis)
    return false;
  Value src = throughBroadcast(ed.getSrc());

  if (auto rem = src.getDefiningOp<arith::RemSIOp>(); rem && mod) {
    int64_t bound = 0;
    if (!denseIntSplatOf(rem.getRhs(), bound))
      return false;
    auto contig = rem->getAttrOfType<DenseElementsAttr>("tt.contiguity");
    if (!contig || !contig.isSplat() ||
        contig.getSplatValue<APInt>().getSExtValue() < extent)
      return false;
    *mod = bound;
    src = throughBroadcast(rem.getLhs());
  }

  if (auto mr = src.getDefiningOp<MakeRangeOp>())
    return mr.getStart() == 0 && mr.getEnd() == (uint32_t)extent;

  auto add = src.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  start = splatScalar(add.getLhs());
  Value rest = add.getRhs();
  if (!start) {
    start = splatScalar(add.getRhs());
    rest = add.getLhs();
  }
  auto mr = rest ? throughBroadcast(rest).getDefiningOp<MakeRangeOp>()
                 : MakeRangeOp();
  return start && mr && mr.getStart() == 0 && mr.getEnd() == (uint32_t)extent;
}

} // namespace

bool usedOnlyByDot(Value v) {
  if (v.use_empty())
    return false;
  for (Operation *user : v.getUsers()) {
    auto dot = dyn_cast<DotOp>(user);
    // The C operand is read as registers, so its layout matters.
    if (!dot || user->getOperand(2) == v)
      return false;
  }
  return true;
}

namespace {

bool denseIntSplatOf(Value v, int64_t &out) {
  auto cst = throughBroadcast(v).getDefiningOp<arith::ConstantOp>();
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

// `muli(rowIota, stride)` in either order, stride a splat scalar or dense
// constant. Fills the row half of `t` on success.
bool rowHalfOf(Value v, int64_t rows, DeviceTile &t) {
  auto mul = throughBroadcast(v).getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  for (int side = 0; side < 2; ++side) {
    const Value sv = side ? mul.getLhs() : mul.getRhs();
    const Value iota = side ? mul.getRhs() : mul.getLhs();
    Value stride = splatScalar(sv);
    int64_t strideK = 0;
    if (!stride && !denseIntSplatOf(sv, strideK))
      continue;
    Value rowStart;
    int64_t mod = 0;
    if (!windowIota(throughBroadcast(iota), 0, rows, rowStart, &mod))
      continue;
    if (mod && !rowStart)
      mod = 0;
    t.rowStride = stride;
    t.rowStrideK = stride ? 0 : strideK;
    t.rowStart = rowStart;
    t.rowStartMod = mod;
    return true;
  }
  return false;
}

} // namespace

namespace {

// Every addend of an offset sum, in source order. `addptr(addptr(p, X), Y)`
// and `addptr(p, addi(X, Y))` are the same address, so both flatten to one
// list.
void offsetTerms(Value v, agpu::msl::SmallVec<Value, 4> &out) {
  v = throughBroadcast(v);
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    offsetTerms(add.getLhs(), out);
    offsetTerms(add.getRhs(), out);
    return;
  }
  out.push_back(v);
}

// The base pointer under any stack of addptrs, with every offset it passed
// collected. Null when the innermost pointer is not a splat scalar.
Value windowBaseAndTerms(Value ptrTensor,
                         agpu::msl::SmallVec<Value, 4> &terms) {
  Value v = ptrTensor;
  while (const PtrAndOffset p = addPtrParts(v)) {
    offsetTerms(p.offset, terms);
    v = p.ptr;
  }
  return splatScalar(v);
}

} // namespace

DeviceTile deviceWindowOf(Value ptrTensor) {
  auto tileTy = dyn_cast<RankedTensorType>(ptrTensor.getType());
  if (!tileTy || tileTy.getRank() != 2)
    return {};
  const int64_t rows = tileTy.getShape()[0], cols = tileTy.getShape()[1];

  agpu::msl::SmallVec<Value, 4> terms;
  DeviceTile t;
  t.base = windowBaseAndTerms(ptrTensor, terms);
  if (!t.base)
    return {};

  // Exactly one row half and one column half. A term that is neither is
  // uniform and shifts the base.
  bool haveRow = false, haveCol = false;
  Value uniform;
  for (Value term : terms) {
    if (!haveCol && windowIota(term, 1, cols, t.colStart)) {
      haveCol = true;
      continue;
    }
    if (!haveRow && rowHalfOf(term, rows, t)) {
      haveRow = true;
      continue;
    }
    if (uniform)
      return {};
    uniform = splatScalar(term);
    if (!uniform)
      return {};
  }
  if (!haveRow || !haveCol)
    return {};
  t.baseOffset = uniform;
  return t;
}

Value splatScalarOf(Value v) { return splatScalar(throughLayoutChange(v)); }

bool splatConstantOf(Value v, double &out) {
  auto cst = throughBroadcast(throughLayoutChange(v))
                 .getDefiningOp<arith::ConstantOp>();
  auto dense =
      cst ? dyn_cast<DenseElementsAttr>(cst.getValue()) : DenseElementsAttr();
  if (!dense || !dense.isSplat())
    return false;
  if (auto f = dyn_cast<FloatAttr>(dense.getSplatValue<Attribute>())) {
    out = f.getValueAsDouble();
    return true;
  }
  return false;
}

namespace {

// Whether `v` is provably a multiple of `blk`: a constant, a multiply with a
// constant-multiple side, or a sum of two such.
bool multipleOf(Value v, int64_t blk) {
  if (blk <= 0)
    return false;
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return c.getSExtValue() % blk == 0;
  if (auto add = v.getDefiningOp<arith::AddIOp>())
    return multipleOf(add.getLhs(), blk) && multipleOf(add.getRhs(), blk);
  auto mul = v.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  for (Value side : {mul.getRhs(), mul.getLhs()})
    if (matchPattern(side, m_ConstantInt(&c)) && c.getSExtValue() != 0 &&
        c.getSExtValue() % blk == 0)
      return true;
  return false;
}

// Whether a bound can never cut through the tile: limit and start both
// multiples of the extent make it all-in or all-out and the exact grid never
// launches an all-out tile.
bool neverRagged(const AxisBound &b, Value start, int64_t extent) {
  return !b.limit && extent > 0 && b.constant % extent == 0 &&
         multipleOf(start, extent);
}

// One conjunct of a mask: `idx < limit` (or `limit > idx`), with `idx` the
// window's `start + iota` along one axis. A different start means the bound
// guards other coordinates.
bool boundConjunct(Value term, const DeviceTile &w, int64_t rows, int64_t cols,
                   WindowBounds &out) {
  auto cmp = throughBroadcast(term).getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return false;
  Value idx, limit;
  if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
    idx = cmp.getLhs();
    limit = cmp.getRhs();
  } else if (cmp.getPredicate() == arith::CmpIPredicate::sgt) {
    idx = cmp.getRhs();
    limit = cmp.getLhs();
  } else {
    return false;
  }

  AxisBound b;
  b.present = true;
  b.limit = splatScalar(limit);
  if (!b.limit && !denseIntSplatOf(limit, b.constant))
    return false;

  Value start;
  if (windowIota(throughBroadcast(idx), 0, rows, start)) {
    // A row start wrapped by the licensed modulo reads from `rm % M` while
    // the mask compares the un-wrapped `rm`.
    if (start != w.rowStart || w.rowStartMod || out.row.present)
      return false;
    if (neverRagged(b, start, rows))
      return true;
    // Clamp candidate: dropped at the drain only once the scalar's definition
    // is shown to take the min.
    if (!b.limit && start && b.constant >= rows)
      out.clamps.push_back({start, b.constant - rows});
    out.row = b;
    return true;
  }
  start = Value();
  if (windowIota(throughBroadcast(idx), 1, cols, start)) {
    if (start != w.colStart || out.col.present)
      return false;
    if (neverRagged(b, start, cols))
      return true;
    if (!b.limit && start && b.constant >= cols)
      out.clamps.push_back({start, b.constant - cols});
    out.col = b;
    return true;
  }
  return false;
}

} // namespace

WindowBounds windowBoundsOf(Value mask, const DeviceTile &window, int64_t rows,
                            int64_t cols) {
  WindowBounds out;
  if (!mask)
    return out;

  // The conjunction, flattened. Every term must be understood; an unknown one
  // fails the whole thing.
  llvm::SmallVector<Value, 4> terms{mask};
  while (!terms.empty()) {
    Value t = terms.pop_back_val();
    if (auto andi = throughBroadcast(t).getDefiningOp<arith::AndIOp>()) {
      terms.push_back(andi.getLhs());
      terms.push_back(andi.getRhs());
      continue;
    }
    if (boundConjunct(t, window, rows, cols, out))
      continue;
    // A conjunct that is one i1 splatted guards every element the same way.
    Value u = splatScalar(t);
    if (!u || out.uniform)
      return WindowBounds{};
    out.uniform = u;
  }
  out.ok = true;
  return out;
}

DrainAddend drainAddendOf(Value v, const DeviceTile &at) {
  auto load = throughLayoutChange(v).getDefiningOp<LoadOp>();
  if (!load)
    return {};

  // A full window at the store's coordinates. The row modulo is part of the
  // coordinate and must match too.
  if (DeviceTile w = deviceWindowOf(load.getPtr()); w.base) {
    if (w.rowStart != at.rowStart || w.colStart != at.colStart ||
        w.rowStartMod != at.rowStartMod)
      return {};
    return {DrainAddend::Form::Tile, load, w};
  }

  // A broadcast row or column: `broadcast(addptr(splat(base), iota))` with
  // one axis and no stride.
  Value ptr = load.getPtr();
  auto ptrTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!ptrTy || ptrTy.getRank() != 2)
    return {};
  const PtrAndOffset parts = addPtrParts(ptr);
  Value base = parts.ptr ? splatScalar(parts.ptr) : Value{};
  if (!base)
    return {};

  Value colStart;
  if (windowIota(throughBroadcast(parts.offset), 1, ptrTy.getShape()[1],
                 colStart)) {
    if (colStart != at.colStart)
      return {};
    DeviceTile row;
    row.base = base;
    row.colStart = colStart;
    return {DrainAddend::Form::Row, load, row};
  }

  Value rowStart;
  int64_t mod = 0;
  if (windowIota(throughBroadcast(parts.offset), 0, ptrTy.getShape()[0],
                 rowStart, &mod)) {
    if (rowStart != at.rowStart || mod != at.rowStartMod)
      return {};
    DeviceTile col;
    col.base = base;
    col.rowStart = rowStart;
    col.rowStartMod = mod;
    return {DrainAddend::Form::Col, load, col};
  }
  return {};
}

DeviceTile deviceTileOf(Value operand) {
  auto load = throughLayoutChange(operand).getDefiningOp<LoadOp>();
  // `simdgroup_load` has no mask and a masked load reads something other
  // than the tile.
  if (!load || load.getMask() || load.getOther())
    return {};
  return deviceWindowOf(load.getPtr());
}

} // namespace mlir::triton::applegpu::bridge
