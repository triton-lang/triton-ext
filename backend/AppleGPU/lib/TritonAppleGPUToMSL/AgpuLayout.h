// AgpuLayout - a LinearLayout's bases, as integers.
#ifndef AGPU_BRIDGE_LAYOUT_H
#define AGPU_BRIDGE_LAYOUT_H

#include "Dialect/TritonAppleGPU/IR/LinearLayoutDims.h"
#include "agpu/bind/LayoutBind.h"
#include "agpu/core/Units.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

#include <optional>
#include <vector>

namespace mlir::triton::applegpu::bridge {

namespace lldim = mlir::triton::applegpu::lldim;

// Consumers treat a missing dimension and an all-zero-basis one alike.
inline agpu::BasisRow basisRow(const LinearLayout &ll, MLIRContext *ctx,
                               llvm::StringRef inDim, StringAttr outDim) {
  agpu::BasisRow row;
  const auto dim = StringAttr::get(ctx, inDim);
  if (!ll.hasInDim(dim))
    return row;
  for (int b = 0, n = ll.getInDimSizeLog2(dim); b < n; ++b)
    row.push_back(ll.getBasis(dim, b, outDim));
  return row;
}

inline agpu::LayoutSource layoutSourceOf(const LinearLayout &ll,
                                         MLIRContext *ctx, StringAttr outDim) {
  agpu::LayoutSource s;
  s.reg = basisRow(ll, ctx, lldim::Register, outDim);
  s.lane = basisRow(ll, ctx, lldim::Lane, outDim);
  s.warp = basisRow(ll, ctx, lldim::Warp, outDim);
  s.block = basisRow(ll, ctx, lldim::Block, outDim);
  return s;
}

inline agpu::LayoutSource layoutSourceOf(RankedTensorType rt,
                                         StringAttr outDim) {
  return layoutSourceOf(gpu::toLinearLayout(rt), rt.getContext(), outDim);
}

inline std::optional<StringAttr> outDimAt(const LinearLayout &ll, int axis) {
  int i = 0;
  for (StringAttr n : ll.getOutDimNames())
    if (i++ == axis)
      return n;
  return std::nullopt;
}

// LinearLayout::apply requires a dimension the layout does not have be
// skipped.
inline std::optional<SmallVector<int32_t>>
applyAt(const LinearLayout &ll, MLIRContext *ctx, ArrayRef<int64_t> shape,
        int32_t reg, int32_t lane, int32_t warp, int32_t block) {
  SmallVector<std::pair<StringAttr, int32_t>> ins;
  ins.push_back({StringAttr::get(ctx, lldim::Register), reg});
  const std::pair<llvm::StringRef, int32_t> rest[] = {
      {lldim::Lane, lane}, {lldim::Warp, warp}, {lldim::Block, block}};
  for (const auto &d : rest) {
    const auto dim = StringAttr::get(ctx, d.first);
    if (ll.hasInDim(dim))
      ins.push_back({dim, d.second});
  }

  const auto outs = ll.apply(ins);
  if (outs.size() != shape.size())
    return std::nullopt;

  SmallVector<int32_t> coord;
  for (std::size_t d = 0; d < outs.size(); ++d) {
    if (outs[d].second < 0 || outs[d].second >= shape[d])
      return std::nullopt;
    coord.push_back(outs[d].second);
  }
  return coord;
}

// Row-major, matching DenseElementsAttr's storage order.
template <class Coord>
inline int64_t flatIndex(ArrayRef<int64_t> shape, const Coord &coord) {
  int64_t flat = 0;
  for (std::size_t d = 0; d < shape.size(); ++d)
    flat = flat * shape[d] + (int64_t)coord[d];
  return flat;
}

// Where register `reg` sits, for lane 0: one coordinate per axis.
inline std::optional<std::vector<int64_t>> registerCoordAt(RankedTensorType rt,
                                                           int reg) {
  const std::optional<SmallVector<int32_t>> c = applyAt(
      gpu::toLinearLayout(rt), rt.getContext(), rt.getShape(), reg, 0, 0, 0);
  if (!c)
    return std::nullopt;
  return std::vector<int64_t>(c->begin(), c->end());
}

// Register `reg`'s lane-0 position as one flat, row-major index.
inline std::optional<int64_t> flatElemAt(RankedTensorType rt, int reg) {
  const std::optional<std::vector<int64_t>> coord = registerCoordAt(rt, reg);
  if (!coord)
    return std::nullopt;

  return flatIndex(rt.getShape(), *coord);
}

// Bitmask of this dimension's bases that move nothing: a set bit is an index
// bit the layout ignores, so the value is replicated across it.
inline unsigned freeBitsOf(const LinearLayout &ll, MLIRContext *ctx,
                           llvm::StringRef inDim) {
  const auto dim = StringAttr::get(ctx, inDim);
  if (!ll.hasInDim(dim))
    return 0;
  unsigned free = 0;
  for (int b = 0, n = ll.getInDimSizeLog2(dim); b < n; ++b) {
    bool moves = false;
    for (StringAttr out : ll.getOutDimNames())
      if (ll.getBasis(dim, b, out) != 0) {
        moves = true;
        break;
      }
    if (!moves)
      free |= (1u << b);
  }
  return free;
}

// How many registers each thread holds for this layout.
inline int64_t registerCount(const LinearLayout &ll, MLIRContext *ctx) {
  const auto kReg = StringAttr::get(ctx, lldim::Register);
  return ll.hasInDim(kReg) ? ll.getInDimSize(kReg) : 1;
}

inline int64_t registerCount(RankedTensorType rt) {
  return registerCount(gpu::toLinearLayout(rt), rt.getContext());
}

// The source element feeding a result register, across a shape change that
// moves no data (expand_dims, broadcast). Works in coordinates, since an
// index means different things under two layouts.
inline std::optional<int64_t>
elemThroughRebind(RankedTensorType srcTy, RankedTensorType resTy, int resReg) {
  const ArrayRef<int64_t> srcShape = srcTy.getShape();
  const ArrayRef<int64_t> resShape = resTy.getShape();

  const std::optional<SmallVector<int32_t>> got =
      applyAt(gpu::toLinearLayout(resTy), resTy.getContext(), resShape, resReg,
              0, 0, 0);
  if (!got)
    return std::nullopt;
  const SmallVector<int32_t> &coord = *got;

  SmallVector<int32_t> srcCoord;
  if (resShape.size() == srcShape.size()) {
    for (std::size_t d = 0; d < coord.size(); ++d)
      srcCoord.push_back(srcShape[d] == 1 ? 0 : coord[d]);
  } else if (resShape.size() == srcShape.size() + 1) {
    std::size_t s = 0;
    for (std::size_t d = 0; d < coord.size(); ++d) {
      if (s < srcShape.size() && resShape[d] == srcShape[s]) {
        srcCoord.push_back(coord[d]);
        ++s;
        continue;
      }
      if (resShape[d] != 1)
        return std::nullopt;
    }
    if (s != srcShape.size())
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  for (std::size_t d = 0; d < srcCoord.size(); ++d)
    if (srcCoord[d] < 0 || srcCoord[d] >= srcShape[d])
      return std::nullopt;
  return flatIndex(srcShape, srcCoord);
}

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_LAYOUT_H
