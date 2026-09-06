// SymbolTable - what names hold the value an IR value stands for. A value is
// an opaque `ValueId`.
//
// Unbound is not dataless; they are separate queries. A read must not create
// an entry. Use `alias` to bind from another binding: `bindRegs(dst,
// namesOf(src))` passes a reference into the map and the insert frees the
// storage mid-copy.
#ifndef AGPU_SYMBOL_TABLE_H
#define AGPU_SYMBOL_TABLE_H

#include "agpu/core/ValueId.h"
#include "agpu/msl/Containers.h"

#include <map>
#include <vector>

namespace agpu {

class SymbolTable {
public:
  // ── binding ────────────────────────────────────────────────────────────

  // A scalar: one register.
  void bindScalar(ValueId v, msl::Str name) {
    ValueNames regs{std::move(name)};
    claim(v, regs);
    map_[v] = std::move(regs);
  }

  // A tensor: one name per register.
  void bindRegs(ValueId v, ValueNames regs) {
    claim(v, regs);
    map_[v] = std::move(regs);
  }

  // A value carrying no materialised register. Distinct from unbound.
  void bindDataless(ValueId v) { map_[v] = ValueNames{}; }

  // Give `dst` the same names as `src`, copying before the insert.
  bool alias(ValueId dst, ValueId src) {
    const auto it = map_.find(src);
    if (it == map_.end())
      return false;
    ValueNames copy = it->second;
    claim(dst, copy);
    map_[dst] = std::move(copy);
    return true;
  }

  // ── reading ────────────────────────────────────────────────────────────

  bool isBound(ValueId v) const { return map_.count(v) != 0; }

  bool isDataless(ValueId v) const {
    const auto it = map_.find(v);
    return it != map_.end() && it->second.empty();
  }

  // Null when the walk has not bound this value, which happens for a caller
  // asking ahead (a fused GEMM asking for a dot's C before the walk gets
  // there).
  const ValueNames *namesOf(ValueId v) const {
    const auto it = map_.find(v);
    return it == map_.end() ? nullptr : &it->second;
  }

  std::size_t regCount(ValueId v) const {
    const ValueNames *n = namesOf(v);
    return n ? n->size() : 0;
  }

  // Register `r`, with a splat read as broadcasting. Null for unbound,
  // dataless, or an index past a genuine tensor.
  const msl::Str *regAt(ValueId v, std::size_t r) const {
    const ValueNames *n = namesOf(v);
    if (!n || n->empty())
      return nullptr;
    if (n->size() == 1)
      return &(*n)[0];
    return r < n->size() ? &(*n)[r] : nullptr;
  }

  const msl::Str *scalarName(ValueId v) const {
    const ValueNames *n = namesOf(v);
    return n && n->size() == 1 ? &(*n)[0] : nullptr;
  }

  // The one name every register carries, or null when they differ or the value
  // is unbound/dataless. A pointer tensor over one buffer binds this way: each
  // register holds the base's name, with per-element offsets beside it.
  const msl::Str *uniformNameOf(ValueId v) const {
    const ValueNames *n = namesOf(v);
    if (!n || n->empty())
      return nullptr;
    for (const msl::Str &s : *n)
      if (s != (*n)[0])
        return nullptr;
    return &(*n)[0];
  }

  // The names this value introduced (a splat, rename, or addptr binds a name
  // an earlier value owns). First binder wins.
  // Declaring a borrowed name again redefines a kernel parameter or shadows
  // the variable it points at.
  ValueNames ownedNamesOf(ValueId v) const {
    ValueNames out;
    const ValueNames *n = namesOf(v);
    if (!n)
      return out;
    for (const msl::Str &s : *n) {
      const auto it = owner_.find(s);
      if (it == owner_.end() || it->second != v)
        continue;
      // A tensor may hold one name in several registers and a declaration is
      // per name.
      bool seen = false;
      for (const msl::Str &k : out)
        seen = seen || k == s;
      if (!seen)
        out.push_back(s);
    }
    return out;
  }

  bool ownsAnyName(ValueId v) const { return !ownedNamesOf(v).empty(); }

  ValueId ownerOf(const msl::Str &name) const {
    const auto it = owner_.find(name);
    return it == owner_.end() ? kNoValue : it->second;
  }

  std::size_t size() const { return map_.size(); }
  void clear() {
    map_.clear();
    owner_.clear();
  }

private:
  void claim(ValueId v, const ValueNames &regs) {
    for (const msl::Str &s : regs)
      owner_.emplace(s, v);
  }

  std::map<ValueId, ValueNames> map_;
  std::map<msl::Str, ValueId> owner_;
};

} // namespace agpu

#endif // AGPU_SYMBOL_TABLE_H
