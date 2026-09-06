// CoordSet - the addresses an index expression can reach: a base plus an xor
// free-mask (GF(2) lattice). `valid=false` means unknown, i.e. may alias.
#ifndef AGPU_COORD_SET_H
#define AGPU_COORD_SET_H

#include <cstdint>

namespace agpu {

struct CoordSet {
  int32_t base = 0;     // the bits that do not vary
  int32_t freeMask = 0; // the bits that do
  bool valid = false;   // whether this describes the expression at all

  bool contains(int32_t addr) const {
    return valid && (addr & ~freeMask) == (base & ~freeMask);
  }

  int64_t size() const {
    if (!valid)
      return 0;
    int64_t n = 1;
    for (int b = 0; b < 32; ++b)
      if (freeMask & (1 << b))
        n *= 2;
    return n;
  }
};

inline CoordSet unknownCoords() { return CoordSet{}; }

inline CoordSet exactCoord(int32_t v) { return CoordSet{v, 0, true}; }

// False whenever either set is invalid.
inline bool provablyDisjoint(const CoordSet &a, const CoordSet &b) {
  if (!a.valid || !b.valid)
    return false;
  const int32_t free = a.freeMask | b.freeMask;
  return (a.base & ~free) != (b.base & ~free);
}

// Not the negation of provablyDisjoint: two overlapping sets are not the same
// address.
inline bool provablySame(const CoordSet &a, const CoordSet &b) {
  return a.valid && b.valid && a.freeMask == 0 && b.freeMask == 0 &&
         a.base == b.base;
}

// An offset overlapping the free mask invalidates.
inline CoordSet offsetBy(const CoordSet &s, int32_t k) {
  if (!s.valid || (k & s.freeMask))
    return unknownCoords();
  return CoordSet{s.base ^ k, s.freeMask, true};
}

// Over-approximated to the smallest lattice element containing both.
inline CoordSet unionOf(const CoordSet &a, const CoordSet &b) {
  if (!a.valid || !b.valid)
    return unknownCoords();
  const int32_t differing = a.base ^ b.base;
  return CoordSet{a.base & ~differing, a.freeMask | b.freeMask | differing,
                  true};
}

} // namespace agpu

#endif // AGPU_COORD_SET_H
