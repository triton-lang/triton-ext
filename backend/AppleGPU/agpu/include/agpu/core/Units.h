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

// ── the hardware budget ───────────────────────────────────────────────────

// What a core hands out across concurrently resident threadgroups.
inline constexpr int64_t kTGCoreBudgetBytes = 65536;

// Widest launch Metal admits regardless of register appetite. A register-hungry
// kernel can compile to a pipeline capped at 384 threads and a wider launch is

inline constexpr int64_t kWarpSize = 32;

inline constexpr int64_t threadsFor(int64_t numWarps) {
  return numWarps * kWarpSize;
}

} // namespace agpu

#endif // AGPU_UNITS_H
