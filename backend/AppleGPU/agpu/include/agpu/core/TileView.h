// TileView - a rectangular tile resident somewhere addressable.
//
// Whoever reserves space for a tile calls cosizeElems() on the same object the
// emitter addresses through.
//
// No permutation here: bank spread comes from a padded stride instead, see
// core/Padding.h.
#ifndef AGPU_TILE_VIEW_H
#define AGPU_TILE_VIEW_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace agpu {

// Extents and strides are in elements, innermost dimension last. Strides are
// explicit: a padded row or a transposed operand is the same type with
// different numbers.
class TileView {
public:
  using Coord = std::vector<int64_t>;

  TileView() = default;

  TileView(Coord extent, Coord stride, int64_t origin = 0)
      : extent_(std::move(extent)), stride_(std::move(stride)),
        origin_(origin) {
    assert(extent_.size() == stride_.size());
  }

  static TileView rowMajor(std::initializer_list<int64_t> extent) {
    return rowMajor(Coord(extent));
  }
  static TileView rowMajor(Coord extent) {
    Coord stride(extent.size(), 1);
    int64_t s = 1;
    for (std::size_t d = extent.size(); d-- > 0;) {
      stride[d] = s;
      s *= extent[d];
    }
    return TileView(std::move(extent), std::move(stride));
  }

  // `pad` extra elements appended to each row's stride only (extent unchanged).
  static TileView rowMajorPadded(Coord extent, int64_t pad) {
    assert(!extent.empty());
    TileView v = rowMajor(extent);
    if (pad == 0)
      return v;
    const std::size_t last = extent.size() - 1;
    int64_t s = v.extent_[last] + pad;
    v.stride_[last] = 1;
    for (std::size_t d = last; d-- > 0;) {
      v.stride_[d] = s;
      s *= v.extent_[d];
    }
    return v;
  }

  int rank() const { return static_cast<int>(extent_.size()); }
  const Coord &extent() const { return extent_; }
  const Coord &stride() const { return stride_; }
  int64_t origin() const { return origin_; }
  int64_t extentAt(int d) const { return extent_[d]; }
  int64_t strideAt(int d) const { return stride_[d]; }

  // Templated over the term type so `offsetOf` (integers) and `offsetExprOf`
  // in Emit.h (AST nodes) share one loop.
  template <typename T, typename ScaleFn, typename AddFn, typename UnitFn>
  T linearize(const std::vector<T> &coord, ScaleFn scale, AddFn add,
              UnitFn unit) const {
    T off = unit(origin_);
    for (std::size_t d = 0; d < coord.size(); ++d)
      off = add(off, scale(coord[d], stride_[d]));
    return off;
  }

  int64_t offsetOf(const Coord &coord) const {
    assert(coord.size() == extent_.size());
    return linearize<int64_t>(
        coord, [](int64_t v, int64_t s) { return v * s; },
        [](int64_t a, int64_t b) { return a + b; },
        [](int64_t v) { return v; });
  }
  int64_t offsetOf(std::initializer_list<int64_t> coord) const {
    return offsetOf(Coord(coord));
  }

  // Keeps this view's strides; the origin absorbs the offset.
  TileView subview(const Coord &at, const Coord &ext) const {
    assert(at.size() == extent_.size() && ext.size() == extent_.size());
    return TileView(ext, stride_, offsetOf(at));
  }
  TileView subview(std::initializer_list<int64_t> at,
                   std::initializer_list<int64_t> ext) const {
    return subview(Coord(at), Coord(ext));
  }

  // The buffer holds only the window, but the caller's coordinates are still
  // the tensor's and the origin absorbs the subtraction. The origin goes
  // negative, so `cosizeElems()` on the result is not meaningful.
  TileView originAt(const Coord &at) const {
    assert(at.size() == extent_.size());
    int64_t off = 0;
    for (std::size_t d = 0; d < at.size(); ++d)
      off += at[d] * stride_[d];
    TileView v = *this;
    v.origin_ = origin_ - off;
    return v;
  }
  TileView originAt(std::initializer_list<int64_t> at) const {
    return originAt(Coord(at));
  }

  // Drop a dimension by fixing its coordinate; the origin carries it.
  TileView slice(int64_t at, int dim = 0) const {
    assert(rank() > 1 && dim < rank());
    Coord at3(extent_.size(), 0);
    at3[dim] = at;
    const int64_t off = offsetOf(at3);
    Coord e, s;
    for (int d = 0; d < rank(); ++d) {
      if (d == dim)
        continue;
      e.push_back(extent_[d]);
      s.push_back(stride_[d]);
    }
    return TileView(std::move(e), std::move(s), off);
  }

  // Sizing query: a pool reservation must be at least this large.
  int64_t cosizeElems() const {
    if (extent_.empty())
      return 0;
    int64_t last = origin_;
    for (std::size_t d = 0; d < extent_.size(); ++d) {
      if (extent_[d] <= 0)
        return 0;
      last += (extent_[d] - 1) * stride_[d];
    }
    return last + 1;
  }

  int64_t sizeElems() const {
    if (extent_.empty())
      return 0;
    int64_t n = 1;
    for (int64_t e : extent_)
      n *= e;
    return n;
  }

  bool isDense() const { return sizeElems() == cosizeElems() - origin_; }

  bool innermostContiguous() const {
    return !stride_.empty() && stride_.back() == 1;
  }

  bool operator==(const TileView &o) const {
    return extent_ == o.extent_ && stride_ == o.stride_ && origin_ == o.origin_;
  }
  bool operator!=(const TileView &o) const { return !(*this == o); }

private:
  Coord extent_;
  Coord stride_;
  int64_t origin_ = 0;
};

} // namespace agpu

#endif // AGPU_TILE_VIEW_H
