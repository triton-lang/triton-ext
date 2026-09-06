// AccessPlan.h - lowering one load or store.
#ifndef AGPU_ACCESS_PLAN_H
#define AGPU_ACCESS_PLAN_H

#include "agpu/core/CoordGuard.h"
#include "agpu/core/Decline.h"
#include "agpu/plan/AccessWidth.h"
#include "agpu/plan/LayoutBasis.h"

#include <cstdint>
#include <vector>

namespace agpu {

inline constexpr int64_t kMaskFastPathMinRegs = 4;

// A mask read as `coordinate < limit`, with the layout that builds it
struct MaskBound {
  bool known = false;
  int dim = 0;
  int64_t limit = 0;
  int64_t dimSize = 0;
  LayoutBasis basis;
};

// What the IR says about one access.
struct MoveFacts {
  int64_t regCount = 1;
  unsigned elemBits = 32;
  bool hasMask = false;
  bool hasOther = false; // a value for lanes the mask excludes
  bool isStore = false;

  // Bypass the cache. Decided by `CoherencePlan`.
  bool coherent = false;

  PtrDims ptr; // what the axis analysis knows, per dimension
  RegBases bases;
  RuntimeSpan runtime; // lane/warp/block bases per dimension, or'd

  // A further runtime term ANDed with the mask, which elision must not drop
  bool guardHasRuntimeTerm = false;

  MaskBound bound;

  const char *where() const { return isStore ? "emitStore" : "emitLoad"; }
};

struct MaskGuards {
  std::vector<CoordGuard::Kind> kinds;

  bool empty() const { return kinds.empty(); }
  bool deadAt(int64_t r) const {
    return !kinds.empty() && kinds[(std::size_t)r] == CoordGuard::Kind::Dead;
  }
  bool testedAt(int64_t r) const {
    return kinds.empty() || kinds[(std::size_t)r] == CoordGuard::Kind::Needed;
  }
};

inline MaskGuards planMaskGuards(const MoveFacts &f) {
  MaskGuards g;
  if (!f.hasMask || !f.bound.known || f.guardHasRuntimeTerm)
    return g;
  const CoordWindow window{f.bound.dim, 0, f.bound.limit};
  g.kinds.reserve((std::size_t)f.regCount);
  for (int64_t r = 0; r < f.regCount; ++r)
    g.kinds.push_back(
        planGuard({f.bound.basis.rangeOf((int)r, f.bound.dim, f.bound.dimSize)},
                  {window})
            .kind());
  return g;
}

// A run exists only if the register count divides by the width; a trailing
// partial run declines the whole vectorisation.
struct RunPlan {
  int64_t width = 1;
  bool packed = false;
  int64_t runs = 0; // number of vector accesses

  bool vectorised() const { return width > 1; }
};

inline RunPlan planRuns(const MoveFacts &f, const AccessPlan &w) {
  RunPlan r;
  r.width = 1;
  r.packed = false;
  if (w.vectorised() && f.regCount % w.width == 0) {
    r.width = w.width;
    r.packed = w.packed;
  }
  r.runs = f.regCount / r.width;
  return r;
}

// A run straddling the bound keeps its guards
inline bool runIsUnguarded(const MaskGuards &g, int64_t base, int64_t width) {
  if (g.empty())
    return false;
  for (int64_t i = 0; i < width; ++i)
    if (g.testedAt(base + i) || g.deadAt(base + i))
      return false;
  return true;
}

inline bool runIsDead(const MaskGuards &g, int64_t base, int64_t width) {
  if (g.empty())
    return false;
  for (int64_t i = 0; i < width; ++i)
    if (!g.deadAt(base + i))
      return false;
  return true;
}

// Individually-guarded loads issue serially. Peeling gives the all-true case
// one unconditional batch, with the else arm keeping masked semantics. A mask
// the layout already decides needs no peel.
inline bool peelsFastPath(const MoveFacts &f, const MaskGuards &g) {
  if (!g.empty())
    return false;
  return f.hasMask && f.regCount >= kMaskFastPathMinRegs;
}

// A masked load must leave something defined in every register; the mask is a
// runtime value. An unmasked load initialises nothing.
enum class MaskedInit {
  None,  // a store, or an unmasked load: nothing to initialise
  Other, // the IR supplied a value
  Zero,  // it did not, so the typed zero
};

inline MaskedInit initFor(const MoveFacts &f) {
  if (f.isStore || !f.hasMask)
    return MaskedInit::None;
  return f.hasOther ? MaskedInit::Other : MaskedInit::Zero;
}

struct MovePlan {
  RunPlan runs;
  bool peel = false;
  MaskedInit init = MaskedInit::None;
  VecElem elem = VecElem::Unsupported;

  bool coherent = false;

  MaskGuards guards;

  // Read by `moveDecision` for the reason the access did not vectorise.
  AccessPlan access;

  int64_t width() const { return runs.width; }
  bool vectorised() const { return runs.vectorised(); }
};

inline MovePlan planMove(const MoveFacts &f) {
  MovePlan p;
  p.elem = vecElemOf(f.elemBits);
  p.access = planAccess(f.bases, f.runtime, f.ptr, p.elem);
  p.runs = planRuns(f, p.access);
  p.guards = planMaskGuards(f);
  p.peel = peelsFastPath(f, p.guards);
  p.init = initFor(f);
  p.coherent = f.coherent;
  return p;
}

inline Decision moveDecision(const MovePlan &p, const MoveFacts &f) {
  if (p.vectorised())
    return Decision::emitted();
  const Decision w = widthDecision(p.access, p.elem);
  if (w.ok())
    return Decision::declined(f.where(),
                              "registers do not form an aligned run");
  return Decision::declined(f.where(), w.why());
}

} // namespace agpu

#endif // AGPU_ACCESS_PLAN_H
