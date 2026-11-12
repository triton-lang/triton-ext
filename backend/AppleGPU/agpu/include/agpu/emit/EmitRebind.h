// Emitting a rebinding, which usually means emitting nothing.
//
// RebindPlan says which source register feeds each result register. Alias:
// result names are the source names reordered. Copy: new names declared and
// assigned, for callers that need their own names.
#ifndef AGPU_EMIT_REBIND_H
#define AGPU_EMIT_REBIND_H

#include "agpu/msl/Context.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/RebindPlan.h"

#include <vector>

namespace agpu {

// The result's register names, taken from the source's. Emits nothing.
inline std::vector<msl::Str> aliasRebind(const Rebind &r,
                                         const std::vector<msl::Str> &src) {
  std::vector<msl::Str> out;
  out.reserve(r.from.size());
  for (int f : r.from)
    out.push_back(f >= 0 && f < (int)src.size() ? src[(std::size_t)f]
                                                : msl::Str{});
  return out;
}

inline Decision copyRebind(msl::Context &c, msl::Block &body, const Rebind &r,
                           const std::vector<msl::Str> &src,
                           const std::vector<msl::Str> &dst, ElemType elem) {
  if (Decision d = rebindDecision(r); !d.ok())
    return d;
  if (dst.size() != r.from.size())
    return Decision::declined("rebind",
                              "destination name count does not match");
  for (std::size_t n = 0; n < r.from.size(); ++n) {
    const int f = r.from[n];
    if (f < 0 || f >= (int)src.size())
      return Decision::declined("rebind", "source register out of range");
    body.push_back(
        c.declStmt(mslTypeOf(elem), dst[n], c.var(src[(std::size_t)f])));
  }
  return Decision::emitted();
}

// A join, whose result draws from two sources. A register claimed by neither
// leaves an empty name; `rebindDecision` reports it.
inline std::vector<msl::Str>
aliasJoin(const std::vector<Rebind> &rs,
          const std::vector<std::vector<msl::Str>> &srcs) {
  std::vector<msl::Str> out;
  if (rs.empty())
    return out;
  out.assign(rs[0].from.size(), msl::Str{});
  for (std::size_t s = 0; s < rs.size() && s < srcs.size(); ++s)
    for (std::size_t n = 0; n < rs[s].from.size() && n < out.size(); ++n) {
      const int f = rs[s].from[n];
      if (f >= 0 && f < (int)srcs[s].size())
        out[n] = srcs[s][(std::size_t)f];
    }
  return out;
}

inline bool allNamed(const std::vector<msl::Str> &names) {
  for (const msl::Str &n : names)
    if (n.empty())
      return false;
  return !names.empty();
}

} // namespace agpu

#endif // AGPU_EMIT_REBIND_H
