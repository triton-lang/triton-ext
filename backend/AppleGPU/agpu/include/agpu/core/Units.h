// Units.h - byte and element counts as distinct types, so pool arithmetic
// cannot mix bytes of threadgroup memory with elements of a tile.
#ifndef AGPU_UNITS_H
#define AGPU_UNITS_H

#include <algorithm>
#include <cstdint>

namespace agpu {

class Elems {
public:
  Elems() = default;
  explicit Elems(int64_t n) : n_(n) {}
  int64_t count() const { return n_; }
  bool operator==(Elems o) const { return n_ == o.n_; }
  bool operator<(Elems o) const { return n_ < o.n_; }

private:
  int64_t n_ = 0;
};

class Bytes {
public:
  Bytes() = default;
  explicit Bytes(int64_t n) : n_(n) {}
  int64_t count() const { return n_; }

  Elems inElems(int64_t elemBytes) const {
    return Elems(elemBytes > 0 ? n_ / elemBytes : 0);
  }

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

inline constexpr int64_t kTGResidentBudgetBytes = 32768;

// Twice the per-threadgroup cap: what a core hands out across concurrently
// resident threadgroups.
inline constexpr int64_t kTGCoreBudgetBytes = 65536;

// Threadgroups that stay resident when a pool of `bytes` is declared. A step
// function: 16 KB gives four, 22 KB gives two.
inline constexpr int64_t tgResidency(int64_t bytes) {
  return bytes > 0 ? kTGCoreBudgetBytes / bytes : kTGCoreBudgetBytes;
}

inline constexpr int64_t tgPoolForResidency(int64_t n) {
  return n > 0 ? kTGCoreBudgetBytes / n : kTGCoreBudgetBytes;
}

// Resident threadgroups past which threadgroup memory stops binding occupancy.
// Registers and warp slots bind first above this line.
inline constexpr int64_t kTGResidencyFloor = 6;

// Widest launch Metal admits regardless of register appetite. A register-hungry
// kernel can compile to a pipeline capped at 384 threads and a wider launch is
// then rejected at dispatch as OutOfResources.
inline constexpr int64_t kAlwaysAdmittedThreads = 384;

// Alignment of a threadgroup pool's base address (Metal's widest vector
// access). `kPadBytes` is a whole multiple of it.
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
