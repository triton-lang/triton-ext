// AgpuPool - the threadgroup pool: what one op carves, and the body's ledger
// of what was carved and what was addressed. Implementation in AgpuPool.cpp.
#ifndef AGPU_BRIDGE_POOL_H
#define AGPU_BRIDGE_POOL_H

#include "agpu/msl/Ast.h"
#include "agpu/plan/ElemType.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir::triton::applegpu::bridge {

// A typed region carved out of the one threadgroup buffer.
//
// Only a kernel may declare threadgroup memory in MSL and `emitKernel`
// declares one, so every typed buffer is a pointer into it at an offset.
struct PoolRegion {
  agpu::msl::Str name;
  agpu::ElemType elem;
  int64_t offset = 0;
  int64_t bytes = 0;
  // Whether anything addressed it. Carving reserves the offset; only a
  // handler that emits through the region declares the pointer.
  bool used = false;
  // The pointer this region is declared and addressed as. Usually `name`,
  // but two ops sharing a name may disagree about offset or element type,
  // and then the later one gets its own suffixed declaration.
  agpu::msl::Str decl;
};

// The pool regions one operation carves, named and sized, in carve order.
// Carving costs nothing: only regions the handler addressed reach the
// declaration.
struct PoolNeed {
  struct Region {
    agpu::msl::Str name;
    agpu::ElemType elem;
    int64_t bytes = 0;
    // Sits at the pool's base, overlaying the sequential layout. Only legal
    // when the two never hold live data at once.
    bool atBase = false;

    // A typed pointer must be aligned to its element or the hardware faults.
    int64_t alignedBytes() const {
      const int64_t a = agpu::byteWidthOf(elem);
      return ((bytes + a - 1) / a) * a;
    }
  };

  agpu::msl::SmallVec<Region, 4> regions;

  void add(agpu::msl::Str name, agpu::ElemType elem, int64_t bytes,
           bool atBase = false) {
    regions.push_back({std::move(name), elem, bytes, atBase});
  }
  bool empty() const { return regions.empty(); }
};

// The body's pool bookkeeping: which regions were carved, at what offsets,
// and which of them anything actually addressed.
class PoolLedger {
public:
  PoolLedger() = default;

  // Lay out one op's regions in the threadgroup buffer, from zero: regions
  // of different ops do not coexist. A layout matching an existing region
  // shares its declaration; a conflicting one gets its own suffixed
  // declaration at its own offset.
  void carve(const PoolNeed &need);

  // Separate from carve because a handler may decide not to use the pool after
  // carving, e.g. a convert_layout that turns out to be a rename. Empty means
  // the region was never carved for this op, so callers decline on it.
  agpu::msl::Str use(const agpu::msl::Str &name);

  // The declaration `use` would answer, without marking the region used.
  agpu::msl::Str peek(const agpu::msl::Str &name) const;

  // Every region carved this body, in the order first asked for. Order
  // decides the offsets.
  const std::vector<PoolRegion> &regions() const { return regions_; }

  // The extent of the regions the body addressed; an unused carve costs
  // nothing.
  int64_t usedBytes() const;

private:
  std::vector<PoolRegion> regions_;

  // The current op's name -> region resolution, rebuilt by each carve.
  std::map<agpu::msl::Str, std::size_t> current_;
};

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_POOL_H
