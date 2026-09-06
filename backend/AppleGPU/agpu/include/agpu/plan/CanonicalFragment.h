// CanonicalFragment - the MMA fragment shape and how it is spelled.
#ifndef AGPU_CANONICAL_FRAGMENT_H
#define AGPU_CANONICAL_FRAGMENT_H

#include "agpu/core/Units.h"
#include "agpu/msl/Ast.h"
#include "agpu/msl/Builtins.h"

#include <cstdint>
#include <string>

namespace agpu {

// The forms addressable without staging. Anything else takes the fallback.
enum class FragmentKind {
  None = 0,     // not recognised: use the pool round trip
  Simdgroup8x8, // Metal's one MMA fragment shape
};

class CanonicalFragment {
public:
  CanonicalFragment() = default;
  explicit CanonicalFragment(FragmentKind k) : kind_(k) {}

  FragmentKind kind() const { return kind_; }
  explicit operator bool() const { return kind_ != FragmentKind::None; }

  int64_t dim() const {
    switch (kind_) {
    case FragmentKind::Simdgroup8x8:
      return kSgFragDim;
    case FragmentKind::None:
      return 0;
    }
    return 0;
  }

  // Metal's 8x8 fragment is 64 elements over 32 lanes.
  int64_t elemsPerLane() const { return lanes() ? dim() * dim() / lanes() : 0; }

  int64_t lanes() const {
    return kind_ == FragmentKind::Simdgroup8x8 ? kWarpSize : 0;
  }

  // The one place `simdgroup_*8x8` is spelled.
  std::string mslType(const std::string &elem) const {
    if (kind_ != FragmentKind::Simdgroup8x8)
      return {};
    const std::string d = std::to_string(dim());
    return msl::builtin::sg::TypePrefix + elem + d + "x" + d;
  }

  // A matrix type: the size analysis counts fragments a function declares,
  // and a named type would also match other opaque types.
  msl::Type mslTypeNode(const std::string &elem) const {
    return msl::Type::matrix(mslType(elem));
  }

  // Not `make_filled_simdgroup_matrix`: its element type and dimensions are
  // not deducible from its argument.
  std::string zeroCtor(const std::string &elem) const { return mslType(elem); }

  bool operator==(const CanonicalFragment &o) const { return kind_ == o.kind_; }

private:
  FragmentKind kind_ = FragmentKind::None;
};

inline const CanonicalFragment kSimdgroup8x8{FragmentKind::Simdgroup8x8};

} // namespace agpu

#endif // AGPU_CANONICAL_FRAGMENT_H
