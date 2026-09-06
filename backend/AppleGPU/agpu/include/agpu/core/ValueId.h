// ValueId - a value in the source IR. Nothing here interprets the number; the
// caller may map it from an `mlir::Value`, a counter, or a test literal.
#ifndef AGPU_VALUE_ID_H
#define AGPU_VALUE_ID_H

#include "agpu/core/Containers.h"

#include <cstdint>
#include <vector>

namespace agpu {

using ValueId = std::int32_t;
using BlockId = std::int32_t;

inline constexpr ValueId kNoValue = -1;

// The names holding one value, one per register. Empty is a real answer: a
// memdesc or an assert result carries no materialised register.
using ValueNames = msl::SmallVec<msl::Str, 8>;

} // namespace agpu

#endif // AGPU_VALUE_ID_H
