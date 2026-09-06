// MemDesc.h - a handle onto part of a threadgroup buffer.
#ifndef AGPU_MEM_DESC_H
#define AGPU_MEM_DESC_H

#include "agpu/core/Containers.h"
#include "agpu/core/TileView.h"

namespace agpu {

struct MemDesc {
  msl::Str buffer;
  TileView view;

  // `memdesc_index`: slice `i` of a multi-buffered allocation.
  MemDesc index(int64_t i) const { return {buffer, view.slice(i)}; }

  // `memdesc_subslice`: a sub-rectangle at `offsets`.
  MemDesc subslice(const TileView::Coord &offsets,
                   const TileView::Coord &extent) const {
    return {buffer, view.subview(offsets, extent)};
  }

  // Subslice with no explicit extent: runs to the end of each dimension.
  MemDesc subslice(const TileView::Coord &offsets) const {
    TileView::Coord ext(offsets.size());
    for (std::size_t d = 0; d < offsets.size(); ++d)
      ext[d] = view.extentAt((int)d) - offsets[d];
    return subslice(offsets, ext);
  }

  int64_t offsetOf(const TileView::Coord &c) const { return view.offsetOf(c); }
  int64_t cosizeElems() const { return view.cosizeElems(); }
};

inline MemDesc allocMemDesc(msl::Str buffer, TileView::Coord extent) {
  return {std::move(buffer), TileView::rowMajor(std::move(extent))};
}

// `count` slices of `extent`, contiguous, modelled as one rank+1 view.
inline MemDesc allocMultiBuffered(msl::Str buffer, int64_t count,
                                  TileView::Coord extent) {
  TileView::Coord full;
  full.push_back(count);
  for (int64_t d : extent)
    full.push_back(d);
  return {std::move(buffer), TileView::rowMajor(std::move(full))};
}

} // namespace agpu

#endif // AGPU_MEM_DESC_H
