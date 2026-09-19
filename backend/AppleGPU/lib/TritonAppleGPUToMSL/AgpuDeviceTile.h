// AgpuDeviceTile - recognising a tile already laid out in device memory, so a
// dot can read its fragments in place and skip the staging pass through
// threadgroup memory. Anything unproven is declined.
#ifndef AGPU_BRIDGE_DEVICE_TILE_H
#define AGPU_BRIDGE_DEVICE_TILE_H

#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::applegpu::bridge {

// A tile the MMA can read where it lies: a base pointer and a row stride.
struct DeviceTile {
  Value base;      // the tensor's base pointer, a scalar !tt.ptr
  Value rowStride; // elements between consecutive rows, an integer scalar

  // The stride when it is a compile-time number.
  // `rowStride` null with this nonzero is that case; both empty is a window
  // with no row half at all (`DrainAddend::broadcastRow`).
  int64_t rowStrideK = 0;

  // A modulus wrapped around the row start (e.g. `rm % M`), admitted only
  // under the op's own `tt.contiguity` (see `windowIota`). Zero: no modulo.
  // The emitter spells the start as `rowStart % rowStartMod`.
  int64_t rowStartMod = 0;

  // An offset uniform across the whole tile, added to the base (e.g. a
  // batched kernel's `z * batchStride`). Null when there is none.
  Value baseOffset;

  // The window's first row and column, integer scalars, or null for zero.
  // Fragment loads read from `base + baseOffset + rowStart * rowStride +
  // colStart`.
  Value rowStart;
  Value colStart;
};

// The device tile behind a dot operand: a row-major window matching
// `simdgroup_load(f, p, stride)`'s addressing. Null `base` for anything else.
DeviceTile deviceTileOf(Value operand);

// The same proof on a tensor of pointers: what a `tt.store` writes through.
DeviceTile deviceWindowOf(Value ptrTensor);

// One axis of a proven mask: `coord < limit`, the limit a runtime scalar or
// a compile-time number. `present` false means the axis was never masked,
// distinct from a bound of zero, which is an empty window.
struct AxisBound {
  Value limit;
  int64_t constant = 0;
  bool present = false;

  bool operator==(const AxisBound &o) const {
    return present == o.present && limit == o.limit && constant == o.constant;
  }
};

// A mask reduced to per-axis bounds on the window's own coordinates. `ok`
// requires every conjunct be a bound on one of the window's axes, with the
// compared index provably `start + iota` for that axis's own start.
struct WindowBounds {
  AxisBound row, col;

  // A ragged axis can instead be clamped at its origin: `min(start, limit -
  // extent)` shifts the edge tile onto the last full window so the mask is
  // provably true. Overlap columns are stored twice with identical values.
  struct Clamp {
    Value start;
    int64_t to = 0;
  };
  llvm::SmallVector<Clamp, 2> clamps;

  // A conjunct uniform across the tile. It bounds no axis: the drain spells
  // it as one guard around the whole drain.
  Value uniform;

  bool ok = false;
};

// The bounds behind `mask`, for a `rows x cols` tile addressed through
// `window`.
WindowBounds windowBoundsOf(Value mask, const DeviceTile &window, int64_t rows,
                            int64_t cols);

// The device window behind a drain-folded operand: a `tt.load` reading a full
// window at `at`'s own coordinates, or a broadcast row or column. A mask on
// the load does not decline here; the caller must prove it against the same
// bounds as the store's.
struct DrainAddend {
  enum class Form { None, Tile, Row, Col };
  Form form = Form::None;
  triton::LoadOp load;
  DeviceTile window;

  bool ok() const { return form != Form::None; }
};

DrainAddend drainAddendOf(Value v, const DeviceTile &at);

// The scalar every element of `v` holds, looking through layout changes and
// broadcasts: a `tt.splat`'s source. Null when `v` is not a splat.
Value splatScalarOf(Value v);

// The number every element of `v` holds, for a dense splat constant.
bool splatConstantOf(Value v, double &out);

// The same value with any `convert_layout` stripped: a convert changes neither
// where an element lives in device memory nor which element an index names.
Value throughLayoutChange(Value v);

// Whether every use of `v` is a `tt.dot` operand, which never needs its layout
// established. Dead-code analysis cannot remove the unused convert afterwards,
// since its scatter writes a threadgroup buffer.
bool usedOnlyByDot(Value v);

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_DEVICE_TILE_H
