// Padding - bank-conflict avoidance for tiles that simdgroup ops read.
//
// A tile whose row stride is a whole number of 256-bit lines puts every row
// at the same bank offset, so a column access serialises. Rows are
// lengthened by half a line. simdgroup_load/store take a base pointer plus
// leading dimension, so a per-row permutation cannot be expressed to them; a
// padded stride is just a different leading dimension.
#ifndef AGPU_PADDING_H
#define AGPU_PADDING_H

#include <cstdint>

namespace agpu {

inline constexpr int64_t kPadLineBytes = 32;
inline constexpr int64_t kPadBytes = 16;

// Elements appended to a staged row of `cols` elements of `elemBytes` each.
inline int64_t padElemsFor(int64_t cols, int64_t elemBytes) {
  if (cols <= 0 || elemBytes <= 0)
    return 0;
  if ((cols * elemBytes) % kPadLineBytes != 0)
    return 0;
  return kPadBytes / elemBytes;
}

} // namespace agpu

#endif // AGPU_PADDING_H
