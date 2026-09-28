// The layout queries that cost real work: each applies the layout once per
// register, lane or warp, so they compile once here and not in every
// translation unit that asks.
#include "AgpuLayout.h"
#include "AgpuEmitter.h"

namespace mlir::triton::applegpu::bridge {

// The memo lives beside what it memoises: elemsPerLane costs registers x 32
// layout applications and the shuffle proof asks for the same type twice.
const std::vector<std::vector<int64_t>> &
AgpuEmitter::elemsPerLaneOf(RankedTensorType rt) {
  const auto it = elemsPerLane_.find(rt.getAsOpaquePointer());
  if (it != elemsPerLane_.end())
    return it->second;
  return elemsPerLane_.emplace(rt.getAsOpaquePointer(), elemsPerLane(rt))
      .first->second;
}

// Which elements one warp of a layout holds, as a set of flat indices.
std::set<int64_t> elemsOfWarp(RankedTensorType rt, int32_t warp) {
  const LinearLayout ll = gpu::toLinearLayout(rt);
  MLIRContext *ctx = rt.getContext();
  const auto kLane = StringAttr::get(ctx, lldim::Lane);
  if (!ll.hasInDim(kLane))
    return {};

  const ArrayRef<int64_t> shape = rt.getShape();
  const int32_t lanes = ll.getInDimSize(kLane);
  std::set<int64_t> out;
  for (int64_t r = 0, n = registerCount(ll, ctx); r < n; ++r)
    for (int32_t l = 0; l < lanes; ++l) {
      const std::optional<SmallVector<int32_t>> coord =
          applyAt(ll, ctx, shape, (int32_t)r, l, warp, 0);
      if (!coord)
        return {};
      out.insert(flatIndex(shape, *coord));
    }
  return out;
}

// Whether two layouts put the same elements in the same warp, which is what
// a `simd_shuffle` needs since it only moves values within a warp. `order`
// renumbers the source into the result's element numbering.
bool warpsAgree(RankedTensorType srcTy, RankedTensorType resTy,
                llvm::ArrayRef<int32_t> order) {
  const LinearLayout srcLL = gpu::toLinearLayout(srcTy);
  MLIRContext *ctx = srcTy.getContext();
  const auto kWarp = StringAttr::get(ctx, lldim::Warp);
  const int32_t warps = srcLL.hasInDim(kWarp) ? srcLL.getInDimSize(kWarp) : 1;

  const std::optional<std::vector<int64_t>> renumber =
      order.empty() ? std::optional<std::vector<int64_t>>()
                    : transposedElemMap(srcTy, resTy, order);
  if (!order.empty() && !renumber)
    return false;

  for (int32_t w = 0; w < warps; ++w) {
    const std::set<int64_t> from = elemsOfWarp(srcTy, w);
    const std::set<int64_t> to = elemsOfWarp(resTy, w);
    if (from.empty() || to.empty() || from.size() != to.size())
      return false;

    std::set<int64_t> moved;
    for (int64_t e : from) {
      if (!renumber) {
        moved.insert(e);
        continue;
      }
      if (e < 0 || e >= (int64_t)renumber->size())
        return false;
      moved.insert((*renumber)[(std::size_t)e]);
    }
    if (moved != to)
      return false;
  }
  return true;
}

// Which element every (register, lane) of a layout holds, indexed
// [register][lane]. Empty when the layout has no full warp dimension.
std::vector<std::vector<int64_t>> elemsPerLane(RankedTensorType rt) {
  const LinearLayout ll = gpu::toLinearLayout(rt);
  MLIRContext *ctx = rt.getContext();
  const auto kLane = StringAttr::get(ctx, lldim::Lane);
  if (!ll.hasInDim(kLane) || ll.getInDimSize(kLane) != (int32_t)agpu::kWarpSize)
    return {};

  const ArrayRef<int64_t> shape = rt.getShape();
  const int64_t regs = registerCount(ll, ctx);
  std::vector<std::vector<int64_t>> out;
  for (int64_t r = 0; r < regs; ++r) {
    std::vector<int64_t> perLane;
    for (int32_t l = 0; l < (int32_t)agpu::kWarpSize; ++l) {
      const std::optional<SmallVector<int32_t>> coord =
          applyAt(ll, ctx, shape, (int32_t)r, l, 0, 0);
      if (!coord)
        return {};
      perLane.push_back(flatIndex(shape, *coord));
    }
    out.push_back(std::move(perLane));
  }
  return out;
}

} // namespace mlir::triton::applegpu::bridge
