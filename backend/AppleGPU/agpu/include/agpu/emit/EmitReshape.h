// EmitReshape.h - the two-source rebindings, which emit nothing.
//
// `join`, `split` and `cat` rebind registers without moving data, so this
// returns a name list and appends no statement.
#ifndef AGPU_EMIT_RESHAPE_H
#define AGPU_EMIT_RESHAPE_H

#include "agpu/msl/Containers.h"
#include "agpu/plan/ReshapePlan.h"

namespace agpu {

// `join` / `split`: two register lists interleaved and its inverse.
inline msl::SmallVec<msl::Str, 8>
interleaveNames(const InterleavePlan &p, const msl::SmallVec<msl::Str, 8> &lhs,
                const msl::SmallVec<msl::Str, 8> &rhs) {
  msl::SmallVec<msl::Str, 8> out;
  if (!p.usable)
    return out;
  for (const auto &pick : p.from) {
    const msl::SmallVec<msl::Str, 8> &from = pick.first == 0 ? lhs : rhs;
    if (pick.second < 0 || pick.second >= (int)from.size())
      return {};
    out.push_back(from[(std::size_t)pick.second]);
  }
  return out;
}

} // namespace agpu

#endif // AGPU_EMIT_RESHAPE_H
