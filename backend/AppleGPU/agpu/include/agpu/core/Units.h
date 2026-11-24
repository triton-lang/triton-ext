// Units.h - byte counts as a distinct type, so pool arithmetic cannot mix
// bytes of threadgroup memory with element counts of a tile.
#ifndef AGPU_UNITS_H
#define AGPU_UNITS_H

#include <algorithm>
#include <cstdint>

namespace agpu {

class Bytes {
public:
  Bytes() = default;
  explicit Bytes(int64_t n) : n_(n) {}
  int64_t count() const { return n_; }

  Bytes operator+(Bytes o) const { return Bytes(n_ + o.n_); }
  Bytes operator-(Bytes o) const { return Bytes(n_ - o.n_); }
  Bytes &operator+=(Bytes o) {
    n_ += o.n_;
    return *this;
  }
  bool operator==(Bytes o) const { return n_ == o.n_; }
  bool operator!=(Bytes o) const { return n_ != o.n_; }
  bool operator<(Bytes o) const { return n_ < o.n_; }
  bool operator<=(Bytes o) const { return n_ <= o.n_; }
  bool operator>(Bytes o) const { return n_ > o.n_; }
  bool operator>=(Bytes o) const { return n_ >= o.n_; }

private:
  int64_t n_ = 0;
};

inline Bytes minBytes(Bytes a, Bytes b) { return std::min(a, b); }
inline Bytes maxBytes(Bytes a, Bytes b) { return std::max(a, b); }

// ── the hardware budget ───────────────────────────────────────────────────

// Metal's maxThreadgroupMemoryLength: the most one threadgroup may declare.
inline constexpr int64_t kTGResidentBudgetBytes = 32768;

// Alignment of a threadgroup pool's base address (Metal's widest vector
// access).
inline constexpr int64_t kTGPoolAlignBytes = 16;

inline constexpr int64_t kWarpSize = 32;

inline constexpr int64_t threadsFor(int64_t numWarps) {
  return numWarps * kWarpSize;
}

// Side of the simdgroup MMA fragment. Metal offers exactly one shape.
inline constexpr int64_t kSgFragDim = 8;

// Rounds up: an extent of 60 needs eight fragments. There is no smaller MMA,
// and the surplus is discarded on the way out.
inline constexpr int64_t fragsFor(int64_t extent) {
  return (extent + kSgFragDim - 1) / kSgFragDim;
}

inline constexpr int64_t fragAlignedExtent(int64_t extent) {
  return fragsFor(extent) * kSgFragDim;
}

// Accumulators are fp32 regardless of operand type.
inline constexpr int64_t kAccBytes = 4;

// `planTileActions` takes this as its bit-width argument.
inline constexpr int64_t kAccBits = kAccBytes * 8;

} // namespace agpu

#endif // AGPU_UNITS_H
