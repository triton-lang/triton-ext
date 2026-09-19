// TileView - a rectangular tile resident somewhere addressable.
//
// Whoever reserves space for a tile calls cosizeElems() on the same object the
// emitter addresses through.
//
#ifndef AGPU_TILE_VIEW_H
#define AGPU_TILE_VIEW_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace agpu {

// An XOR permutation of the offset inside a row. `vec` elements stay
// contiguous, `perPhase` consecutive rows share a phase, and `maxPhase` phases
// cycle. All three at 1 is the identity.
struct Swizzle {
  int64_t vec = 1;
  int64_t perPhase = 1;
  int64_t maxPhase = 1;
  int phaseDim = 0;
  int groupDim = 1;
  // How wide the permutation reaches. A row wider than this holds several
  // independent tiles of it, which is how a byte-width swizzle repeats. Kept
  // by a window so its phases match the parent's.
  int64_t groupExtent = 0;
  // Distance to the next tile. Zero leaves the tiles inline, one row after
  // another; a byte-width encoding instead stores each as a whole panel, so
  // the next one starts a rows-worth away.
  int64_t tileStride = 0;

  bool permutes() const { return maxPhase > 1; }
  bool identity() const { return vec == 1 && perPhase == 1 && maxPhase == 1; }

  // The span belongs to the encoding, so a window onto a swizzled parent keeps
  // the parent's and does not shrink to its own width.
  int64_t spanOver(int64_t viewExtent) const {
    return groupExtent > 0 ? groupExtent : viewExtent;
  }

  // The XOR must land inside the span, so the phase cannot exceed the number of
  // groups the span holds; `maxPhase` is the nominal cycle, which a narrow span
  // cannot use in full.
  int64_t effectiveMaxPhase(int64_t span) const {
    const int64_t groups = vec > 0 ? span / vec : 0;
    return groups > 0 ? std::min(maxPhase, groups) : 1;
  }
};

// Extra elements spliced into the linear offset: every `interval` elements
// gain `pad` more. Several rules compose, each measured on the unpadded
// offset, which keeps the map monotonic and so order-preserving.
struct Padding {
  struct Rule {
    int64_t interval = 0;
    int64_t pad = 0;
  };
  std::vector<Rule> rules;

  bool pads() const { return !rules.empty(); }

  int64_t extraBefore(int64_t offset) const {
    int64_t extra = 0;
    for (const Rule &r : rules)
      if (r.interval > 0)
        extra += (offset / r.interval) * r.pad;
    return extra;
  }
};

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

  TileView(Coord extent, Coord stride, Swizzle sw, int64_t origin = 0)
      : extent_(std::move(extent)), stride_(std::move(stride)), swizzle_(sw),
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

  const Swizzle &swizzle() const { return swizzle_; }
  void setSwizzle(Swizzle sw) { swizzle_ = sw; }

  const Padding &padding() const { return padding_; }
  void setPadding(Padding p) { padding_ = std::move(p); }

  const Coord &shift() const { return shift_; }
  bool shifted() const { return !shift_.empty(); }

  // A window keeps the offset as a coordinate, because the permutation is
  // defined on the parent's coordinates and an added origin would miss it.
  TileView window(const Coord &at, const Coord &ext) const {
    assert(at.size() == extent_.size() && ext.size() == extent_.size());
    TileView v(ext, stride_, swizzle_, origin_);
    v.shift_ = at;
    for (std::size_t d = 0; d < shift_.size(); ++d)
      v.shift_[d] += shift_[d];
    return v;
  }

  int rank() const { return static_cast<int>(extent_.size()); }
  const Coord &extent() const { return extent_; }
  const Coord &stride() const { return stride_; }
  int64_t origin() const { return origin_; }
  int64_t extentAt(int d) const { return extent_[d]; }
  int64_t strideAt(int d) const { return stride_[d]; }

  // Templated over the term type so `offsetOf` (integers) and `offsetExprOf`
  // in Emit.h (AST nodes) share one loop. The swizzle folds in here so both
  // spellings permute alike; a reader and a writer that disagreed would
  // silently exchange the wrong element.
  template <typename T, typename ScaleFn, typename AddFn, typename UnitFn,
            typename SwizzleFn>
  T linearize(const std::vector<T> &coord, ScaleFn scale, AddFn add,
              UnitFn unit, SwizzleFn swizzleGroup) const {
    std::vector<T> at = coord;
    if (!shift_.empty())
      for (std::size_t d = 0; d < at.size() && d < shift_.size(); ++d)
        at[d] = add(at[d], unit(shift_[d]));

    T off = unit(origin_);
    for (std::size_t d = 0; d < at.size(); ++d) {
      const int dim = static_cast<int>(d);
      if (swizzle_.permutes() && dim == swizzle_.groupDim) {
        off = add(off, scale(swizzleGroup(at[d], at), stride_[d]));
        continue;
      }
      off = add(off, scale(at[d], stride_[d]));
    }
    return off;
  }

  template <typename T, typename ScaleFn, typename AddFn, typename UnitFn>
  T linearize(const std::vector<T> &coord, ScaleFn scale, AddFn add,
              UnitFn unit) const {
    assert(!swizzle_.permutes() &&
           "a swizzled view needs the swizzle-aware linearize");
    return linearize<T>(coord, scale, add, unit,
                        [](const T &g, const std::vector<T> &) { return g; });
  }

  int64_t offsetOf(const Coord &coord) const {
    assert(coord.size() == extent_.size());
    const Swizzle &sw = swizzle_;
    const int64_t off = linearize<int64_t>(
        coord, [](int64_t v, int64_t s) { return v * s; },
        [](int64_t a, int64_t b) { return a + b; }, [](int64_t v) { return v; },
        [&sw, this](int64_t g, const Coord &all) {
          const int64_t width = sw.spanOver(extent_[(std::size_t)sw.groupDim]);
          const int64_t mp = sw.effectiveMaxPhase(width);
          const int64_t phase =
              (all[(std::size_t)sw.phaseDim] / sw.perPhase) % mp;
          const int64_t tile = g / width, within = g % width;
          const int64_t swizzled =
              ((within / sw.vec) ^ phase) * sw.vec + within % sw.vec;
          const int64_t step = sw.tileStride > 0 ? sw.tileStride : width;
          return tile * step + swizzled;
        });
    return off + padding_.extraBefore(off);
  }
  int64_t offsetOf(std::initializer_list<int64_t> coord) const {
    return offsetOf(Coord(coord));
  }

  // Keeps this view's strides; the origin absorbs the offset. An additive
  // origin cannot carry a permutation, so a swizzled view takes `window`.
  TileView subview(const Coord &at, const Coord &ext) const {
    assert(at.size() == extent_.size() && ext.size() == extent_.size());
    assert(!swizzle_.permutes());
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
    assert(!swizzle_.permutes());
    TileView v = *this;
    v.origin_ = 2 * origin_ - offsetOf(at);
    return v;
  }
  TileView originAt(std::initializer_list<int64_t> at) const {
    return originAt(Coord(at));
  }

  // Drop a dimension by fixing its coordinate; the origin carries it.
  TileView slice(int64_t at, int dim = 0) const {
    assert(rank() > 1 && dim < rank());
    assert(!swizzle_.permutes());
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

  // Sizing query: a pool reservation must be at least this large. A swizzle
  // moves the highest offset off the last coordinate, so the corner no longer
  // bounds it and the row span does.
  int64_t cosizeElems() const {
    if (extent_.empty())
      return 0;
    Coord last(extent_.size());
    for (std::size_t d = 0; d < extent_.size(); ++d) {
      if (extent_[d] <= 0)
        return 0;
      last[d] = extent_[d] - 1;
    }
    if (!swizzle_.permutes())
      return offsetOf(last) + 1;

    const std::size_t g = (std::size_t)swizzle_.groupDim;
    int64_t span = 0;
    for (std::size_t d = 0; d < extent_.size(); ++d)
      if (d != g)
        span += last[d] * stride_[d];
    const int64_t width = swizzle_.spanOver(extent_[g]);
    const int64_t tiles = width > 0 ? (extent_[g] + width - 1) / width : 1;
    const int64_t step =
        swizzle_.tileStride > 0 ? swizzle_.tileStride : width * stride_[g];
    const int64_t raw =
        span + (tiles - 1) * step + width * stride_[g] + origin_;
    return raw + padding_.extraBefore(raw);
  }

  int64_t sizeElems() const {
    if (extent_.empty())
      return 0;
    int64_t n = 1;
    for (int64_t e : extent_)
      n *= e;
    return n;
  }

  bool operator==(const TileView &o) const {
    return extent_ == o.extent_ && stride_ == o.stride_ &&
           origin_ == o.origin_ && swizzle_.vec == o.swizzle_.vec &&
           swizzle_.perPhase == o.swizzle_.perPhase &&
           swizzle_.maxPhase == o.swizzle_.maxPhase &&
           swizzle_.phaseDim == o.swizzle_.phaseDim &&
           swizzle_.groupDim == o.swizzle_.groupDim;
  }
  bool operator!=(const TileView &o) const { return !(*this == o); }

private:
  Coord extent_;
  Coord stride_;
  Coord shift_;
  Swizzle swizzle_;
  Padding padding_;
  int64_t origin_ = 0;
};

} // namespace agpu

#endif // AGPU_TILE_VIEW_H
