// AgpuLayout - a LinearLayout's bases, as integers.
#ifndef AGPU_BRIDGE_LAYOUT_H
#define AGPU_BRIDGE_LAYOUT_H

#include "AgpuShape.h"

#include "Dialect/TritonAppleGPU/IR/LinearLayoutDims.h"
#include "agpu/bind/LayoutBind.h"
#include "agpu/core/Units.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

#include <optional>
#include <set>
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

// `group` is the coordinate on every axis but the scanned one; sorting by it
// makes each scan a contiguous run.
struct ScanRegisterKey {
  std::vector<int64_t> group;
  int64_t position = 0;
  int reg = 0;
};

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

// Whether a difference of two registers' lane-0 coordinates holds in every
// lane, on every axis. True only where register bases and lane/warp/block
// bases are disjoint.
inline bool registerDeltasAreAffine(RankedTensorType rt, int regs) {
  const LinearLayout ll = gpu::toLinearLayout(rt);
  for (int axis = 0; axis < (int)rt.getRank(); ++axis) {
    const std::optional<StringAttr> dim = outDimAt(ll, axis);
    if (!dim)
      return false;
    const agpu::LayoutBasis lb =
        layoutSourceOf(ll, rt.getContext(), *dim).basis();
    for (int r = 0; r < regs; ++r)
      if (!lb.registerDeltasAreAffine(r))
        return false;
  }
  return true;
}

// The element register `reg` holds for lane 0, as a flat row-major index
// into the tensor's shape. Meaningful only when it does not depend on lane.
inline std::optional<int64_t> flatElemAt(RankedTensorType rt, int reg) {
  const std::optional<std::vector<int64_t>> coord = registerCoordAt(rt, reg);
  if (!coord)
    return std::nullopt;

  return flatIndex(rt.getShape(), *coord);
}

// Whether two layouts put the same element in the same register of the same
// thread, so a value under one can be renamed to the other.
inline bool layoutsInterchangeable(RankedTensorType a, RankedTensorType b) {
  if (a.getShape() != b.getShape())
    return false;
  return gpu::toLinearLayout(a) == gpu::toLinearLayout(b);
}

// The source element feeding a result register, across a shape change that
// moves no data (expand_dims, broadcast, convert_layout). Works in
// coordinates, since an index means different things under two layouts.
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

// The source element feeding a result register of a transpose. `order[d]`
// names which source axis becomes result axis d, the `tt.trans` convention.
inline std::optional<int64_t> elemThroughTranspose(RankedTensorType srcTy,
                                                   RankedTensorType resTy,
                                                   ArrayRef<int32_t> order,
                                                   int resReg) {
  const std::optional<std::vector<int64_t>> coord =
      registerCoordAt(resTy, resReg);
  if (!coord)
    return std::nullopt;

  const ArrayRef<int64_t> srcShape = srcTy.getShape();
  if (order.size() != coord->size() || srcShape.size() != coord->size())
    return std::nullopt;

  std::vector<int64_t> srcCoord(srcShape.size(), 0);
  for (std::size_t d = 0; d < coord->size(); ++d) {
    const int32_t s = order[d];
    if (s < 0 || (std::size_t)s >= srcShape.size())
      return std::nullopt;
    if ((*coord)[d] < 0 || (*coord)[d] >= srcShape[s])
      return std::nullopt;
    srcCoord[(std::size_t)s] = (*coord)[d];
  }

  return flatIndex(srcShape, srcCoord);
}

// A reshape moves no data: element k of the source is element k of the
// result, row-major. `allow_reorder` is ignored; taking that permission
// would make source and result disagree about which element is which.
inline std::optional<int64_t>
elemThroughReshape(RankedTensorType srcTy, RankedTensorType resTy, int resReg) {
  int64_t srcCount = 1, resCount = 1;
  for (int64_t d : srcTy.getShape())
    srcCount *= d;
  for (int64_t d : resTy.getShape())
    resCount *= d;
  if (srcCount != resCount)
    return std::nullopt;
  return flatElemAt(resTy, resReg);
}

// Bits of one input dimension whose value does not move the address (zero
// basis on every output dim): threads/registers differing only in such a bit
// are replicas of one location. Matters for atomics, where a replica
// performing one is a wrong answer.
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

// Which elements one warp of a layout holds, as a set of flat indices.
std::set<int64_t> elemsOfWarp(RankedTensorType rt, int32_t warp);

// Whether every warp holds the same elements under both layouts, so a
// shuffle can move them without crossing a warp boundary.
bool warpsAgree(RankedTensorType srcTy, RankedTensorType resTy,
                llvm::ArrayRef<int32_t> order);

// Per register, the flat element each lane holds. Costs registers x 32
// layout applications, so callers that ask repeatedly go through
// AgpuEmitter::elemsPerLaneOf, which memoises it.
std::vector<std::vector<int64_t>> elemsPerLane(RankedTensorType rt);

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_LAYOUT_H
