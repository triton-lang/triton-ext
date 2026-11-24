// AgpuShape - a tensor's shape as counts and views: how many elements there
// are and in what order they linearize.
#ifndef AGPU_BRIDGE_SHAPE_H
#define AGPU_BRIDGE_SHAPE_H

#include "agpu/core/CoordGuard.h"
#include "agpu/core/TileView.h"

#include "mlir/IR/BuiltinTypes.h"

#include <optional>
#include <vector>

namespace mlir::triton::applegpu::bridge {

// Every axis of the tensor, full: the windows a whole-tensor staging or
// readback is planned against, batch axis included.
inline std::vector<agpu::CoordWindow> wholeWindowsOf(RankedTensorType ty) {
  std::vector<agpu::CoordWindow> w;
  for (int d = 0; d < ty.getRank(); ++d)
    w.push_back(agpu::CoordWindow{d, 0, ty.getShape()[d]});
  return w;
}

// How many elements a tensor holds, or 1 for a scalar.
inline int64_t tileElemCount(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  if (!rt)
    return 1;
  int64_t n = 1;
  for (int64_t d : rt.getShape())
    n *= d;
  return n;
}

// A tensor's shape as a row-major view: element (i, j, ...) at the offset a
// C array would put it.
inline agpu::TileView rowMajorViewOf(RankedTensorType t) {
  return agpu::TileView::rowMajor({t.getShape().begin(), t.getShape().end()});
}

// Each source element's flat index in the result's numbering, for a
// transpose. `order[d]` is the source axis result axis `d` comes from, the
// `tt.trans` convention.
inline std::optional<std::vector<int64_t>>
transposedElemMap(RankedTensorType srcTy, RankedTensorType resTy,
                  llvm::ArrayRef<int32_t> order) {
  const llvm::ArrayRef<int64_t> srcShape = srcTy.getShape();
  const llvm::ArrayRef<int64_t> resShape = resTy.getShape();
  if (order.size() != srcShape.size() || resShape.size() != srcShape.size())
    return std::nullopt;

  const int64_t n = tileElemCount(srcTy);
  std::vector<int64_t> out((std::size_t)n, 0);
  std::vector<int64_t> coord(srcShape.size(), 0);
  for (int64_t flat = 0; flat < n; ++flat) {
    int64_t rest = flat;
    for (std::size_t d = srcShape.size(); d-- > 0;) {
      coord[d] = rest % srcShape[d];
      rest /= srcShape[d];
    }
    int64_t moved = 0;
    for (std::size_t d = 0; d < resShape.size(); ++d) {
      const std::size_t from = (std::size_t)order[d];
      if (from >= coord.size() || coord[from] >= resShape[d])
        return std::nullopt;
      moved = moved * resShape[d] + coord[from];
    }
    out[(std::size_t)flat] = moved;
  }
  return out;
}

// One view of a tensor, read with its axes permuted. `order[d]` names which
// source axis becomes axis d of the reading.
inline agpu::TileView permutedView(const agpu::TileView &v,
                                   llvm::ArrayRef<int32_t> order) {
  agpu::TileView::Coord extent, stride;
  for (int32_t s : order) {
    extent.push_back(v.extent()[(std::size_t)s]);
    stride.push_back(v.stride()[(std::size_t)s]);
  }
  return agpu::TileView(std::move(extent), std::move(stride));
}

// Whether `res` is `src` with its axes permuted by `order`.
inline bool isPermutationOf(llvm::ArrayRef<int64_t> src,
                            llvm::ArrayRef<int64_t> res,
                            llvm::ArrayRef<int32_t> order) {
  if (order.size() != src.size() || res.size() != src.size())
    return false;
  std::vector<bool> seen(src.size(), false);
  for (std::size_t d = 0; d < order.size(); ++d) {
    const int32_t s = order[d];
    if (s < 0 || (std::size_t)s >= src.size() || seen[(std::size_t)s])
      return false;
    seen[(std::size_t)s] = true;
    if (res[d] != src[(std::size_t)s])
      return false;
  }
  return true;
}

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_SHAPE_H
