// BandPlan.h - staging a tile that does not fit the pool, in bands.
#ifndef AGPU_BAND_PLAN_H
#define AGPU_BAND_PLAN_H

#include "agpu/core/TileView.h"
#include "agpu/core/Units.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace agpu {

// What is left of the threadgroup budget for this operation.
class Capacity {
public:
  Capacity() = default;
  Capacity(Bytes budget, Bytes live) : budget_(budget), live_(live) {}

  // Saturating: an over-committed pool yields zero.
  Bytes available() const { return maxBytes(budget_ - live_, Bytes(0)); }

  Bytes budget() const { return budget_; }
  Bytes live() const { return live_; }

private:
  Bytes budget_, live_;
};

// How a tile is staged through the pool.
enum class BandKind {
  Whole, // the tile fits; one scatter and one gather
  Flat,  // the tile is banded as a flat element run
};

// Read by both the reservation and the emission: `bytes()` is what the pool
// must hold, `elems()` how many elements a band covers.
class BandPlan {
public:
  BandKind kind() const { return kind_; }
  bool banded() const { return kind_ != BandKind::Whole; }

  int64_t elems() const { return elems_; }

  Bytes bytes() const { return Bytes(elems_ * elemBytes_); }

  int64_t bandCount() const {
    if (elems_ <= 0)
      return 1;
    return (total_ + elems_ - 1) / elems_;
  }

  // The band containing flat element `i`, as a half-open element range.
  struct Band {
    int64_t lo = 0, hi = 0;
    int64_t size() const { return hi - lo; }
  };
  Band bandAt(int64_t b) const {
    const int64_t lo = b * elems_;
    assert(lo < total_ && "band index past the tile");
    return Band{lo, std::min(lo + elems_, total_)};
  }

  friend BandPlan planBand(int64_t totalElems, int64_t elemBytes, Capacity cap,
                           int64_t rowElems);

private:
  BandKind kind_ = BandKind::Whole;
  int64_t elems_ = 0;
  int64_t total_ = 0;
  int64_t elemBytes_ = 1;
};

// Even split into `nBands`. A `rowElems` above zero bands on row edges.
inline BandPlan planBand(int64_t totalElems, int64_t elemBytes, Capacity cap,
                         int64_t rowElems = 0) {
  BandPlan p;
  p.total_ = totalElems;
  p.elemBytes_ = std::max<int64_t>(elemBytes, 1);

  if (totalElems <= 0 || elemBytes <= 0) {
    p.elems_ = std::max<int64_t>(totalElems, 0);
    return p;
  }
  if (Bytes(totalElems * elemBytes) <= cap.available()) {
    p.elems_ = totalElems;
    return p;
  }

  int64_t perBand = cap.available().count() / elemBytes;
  if (perBand < 1)
    perBand = 1;
  const int64_t nBands = (totalElems + perBand - 1) / perBand;
  p.kind_ = BandKind::Flat;
  p.elems_ = (totalElems + nBands - 1) / nBands;

  if (rowElems > 0 && rowElems <= perBand && totalElems % rowElems == 0) {
    p.elems_ = (p.elems_ / rowElems) * rowElems;
    if (p.elems_ < rowElems)
      p.elems_ = rowElems;
  }
  return p;
}

} // namespace agpu

#endif // AGPU_BAND_PLAN_H
