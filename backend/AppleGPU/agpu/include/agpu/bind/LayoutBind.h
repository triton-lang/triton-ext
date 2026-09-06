// LayoutBind - a layout's bases, as the IR hands them over.
//
// Everything this backend wants from a `LinearLayout` is
// `getBasis(inDim, bit, outDim)`. What arrives here is `LayoutBasis`, which
// `emit/LayoutExpr.h` turns into an expression.
#ifndef AGPU_LAYOUT_BIND_H
#define AGPU_LAYOUT_BIND_H

#include "agpu/emit/LayoutExpr.h"
#include "agpu/emit/primitives/CoordHoist.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace agpu {

// One input dimension's bases along one output dimension, as the IR gives
// them: index is the bit, value is what that bit contributes.
struct LayoutSource {
  BasisRow reg, lane, warp, block;

  LayoutBasis basis() const { return LayoutBasis{reg, lane, warp, block}; }
};

} // namespace agpu

#endif // AGPU_LAYOUT_BIND_H
