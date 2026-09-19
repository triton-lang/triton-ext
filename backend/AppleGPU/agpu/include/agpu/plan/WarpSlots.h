// WarpSlots - which (mi, ni) fragment a warp emits and in what K form.
#ifndef AGPU_WARP_SLOTS_H
#define AGPU_WARP_SLOTS_H

#include "agpu/core/Units.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

namespace agpu {

// One fragment a warp owns: its position in the (mT x nT) fragment grid and
// which accumulator it belongs to.
struct Slot {
  int64_t mi = 0;
  int64_t ni = 0;
  int acc = 0; // index into the warp's accumulator list

  bool operator==(const Slot &o) const {
    return mi == o.mi && ni == o.ni && acc == o.acc;
  }
};

// Whether the K steps are unrolled at compile time or rolled into an
// emitted loop.
class KStep {
public:
  static KStep unrolled(int64_t ki) { return KStep(ki, false); }
  static KStep rolled() { return KStep(0, true); }

  struct Offset {
    bool fromLoopVar = false; // scale the loop counter, else use `constant`
    int64_t scale = 1;        // multiplier on the loop counter
    int64_t constant = 0;     // the whole offset when not from the counter
  };

  Offset kOffset(int64_t stride) const {
    if (rolled_)
      return Offset{true, stride, 0};
    return Offset{false, 1, ki_ * kSgFragDim * stride};
  }

private:
  KStep(int64_t ki, bool rolled) : ki_(ki), rolled_(rolled) {}
  int64_t ki_ = 0;
  bool rolled_ = false;
};

// Where fragment `f` sits in an (mT x nT) grid, row-major. Emitters call
// these; they do not spell `f / nT` themselves.
inline int64_t fragRowOf(int64_t f, int64_t nT) { return f / nT; }
inline int64_t fragColOf(int64_t f, int64_t nT) { return f % nT; }

// Warps clamped to the number of fragments: fewer fragments than warps
// leaves the extras idle. Floored at 1 so `fragsPerWarpFor` can divide
// without its own zero test.
inline int64_t effectiveWarps(int64_t numWarps, int64_t nFrag) {
  return std::max<int64_t>(1, std::min(numWarps, nFrag));
}

// ── a fragment coordinate that may depend on the warp id ──────────────────

// One axis of a slot position: `warpScale * ((warpId / warpDiv) % warpMod) +
// constant`, where `warpDiv == 1, warpMod == 0` is the plain affine form and
// `warpScale == 0` a compile-time constant.
//
// div/mod serve the two-axis warp cover: a warp at (warpId / gN, warpId % gN)
// of a (gM x gN) warp grid owns a block of the fragment grid.
struct SlotCoord {
  int64_t constant = 0;
  int64_t warpScale = 0; // 0 => a compile-time constant
  int64_t warpDiv = 1;   // divides warpId first
  int64_t warpMod = 0;   // then wraps it; 0 = no wrap

  static SlotCoord fixed(int64_t v) { return SlotCoord{v, 0, 1, 0}; }
  static SlotCoord affine(int64_t scale, int64_t c) {
    return SlotCoord{c, scale, 1, 0};
  }
  static SlotCoord blocked(int64_t scale, int64_t c, int64_t div, int64_t mod) {
    return SlotCoord{c, scale, div, mod};
  }

  bool isConst() const { return warpScale == 0; }

  // Not used by emission. Tests use it to state what the emitted expression
  // must mean and to check a plan covers the grid once.
  int64_t at(int64_t w) const {
    int64_t t = warpDiv > 1 ? w / warpDiv : w;
    if (warpMod > 0)
      t %= warpMod;
    return warpScale * t + constant;
  }

  bool operator==(const SlotCoord &o) const {
    return constant == o.constant && warpScale == o.warpScale &&
           warpDiv == o.warpDiv && warpMod == o.warpMod;
  }
  bool operator<(const SlotCoord &o) const {
    if (warpScale != o.warpScale)
      return warpScale < o.warpScale;
    if (warpDiv != o.warpDiv)
      return warpDiv < o.warpDiv;
    if (warpMod != o.warpMod)
      return warpMod < o.warpMod;
    return constant < o.constant;
  }
};

// A slot whose position may be warp-dependent. `Slot` is the resolved form
// used by the panel path, where every warp is emitted separately.
struct WarpSlot {
  SlotCoord mi, ni;
  int acc = 0;

  bool operator==(const WarpSlot &o) const {
    return mi == o.mi && ni == o.ni && acc == o.acc;
  }
};

// ── how the warp loop is spelled ──────────────────────────────────────────

// Whether one warpId-parameterised block covers every warp, or each warp
// needs its own guarded copy.
enum class WarpForm {
  Parameterised, // one block, coordinates affine in warpId
  PerWarp,       // numWarps copies under `if (warpId == w)`
};

// Which fragments warp w owns and how their coordinates are spelled, once a
// form has been chosen.
struct WarpProgram {
  WarpForm form = WarpForm::PerWarp;
  const char *because = "";

  // Parameterised forms only. Each warp owns a contiguous (miCount x niCount)
  // block of the fragment grid; the warps tile it as a (mT/miCount) by
  // (nT/niCount) warp grid, row-major in warpId.
  int64_t miCount = 0;
  int64_t niCount = 0;

  // Slots for warp `w`. A Parameterised program ignores `w`: one call
  // produces the block serving every warp.
  std::vector<WarpSlot> slots(int64_t w, int64_t mT, int64_t nT,
                              int64_t numWarps) const;

  // Blocks the emitter produces: 1 when parameterised, numWarps otherwise.
  int64_t blockCount(int64_t numWarps) const {
    if (form == WarpForm::Parameterised)
      return 1;
    return numWarps;
  }

  // The warp this block is restricted to, or nullopt when it serves every
  // warp.
  std::optional<int64_t> guardWarp(int64_t block) const {
    if (form == WarpForm::Parameterised)
      return std::nullopt;
    return block;
  }
};

struct WarpCover {
  int64_t mi = 0, ni = 0;
  bool set() const { return mi > 0 && ni > 0; }
  bool operator==(const WarpCover &o) const { return mi == o.mi && ni == o.ni; }
};

// The grid a warp program covers: fragment counts and warp count.
struct WarpGrid {
  int64_t mT = 0, nT = 0, numWarps = 1;
  bool aDirect = false; // A read from device memory: warps want M bands

  WarpCover cover;

  // The banded emitter selects each band's slots by row at compile time, so a
  // form whose row coordinate is affine in the warp id cannot be filtered.
  bool bandedC = false;

  // Parameterised forms store C right after the MMA with no barrier, so an
  // overlapping C would overwrite operands other warps are still reading.
  bool disjointC = true;

  // Warps the kernel launches, against `numWarps`, what it was planned for.
  // Zero means the same. A dot smaller than the launch leaves warps idle.
  int64_t hwWarps = 0;

  // Whether idle hardware warps exist and must be fenced off every block the
  // program does not already guard. A parameterised block is spelled in
  // `warpId` and reached by every warp, so an unassigned warp would load
  // past the staged tile and store past the tensor.
  bool guardsIdleWarps() const { return hwWarps > numWarps; }

  int64_t nFrag() const { return mT * nT; }
  bool warpsDivideFrags() const {
    return numWarps > 0 && nFrag() % numWarps == 0;
  }
};

inline std::vector<WarpSlot>
WarpProgram::slots(int64_t w, int64_t mT, int64_t nT, int64_t numWarps) const {
  std::vector<WarpSlot> out;
  // A single warp's id is zero, so its coordinates fold to constants here
  // and one-warp kernels never mention `warp`.
  const auto fold = [numWarps](SlotCoord s) {
    return numWarps == 1 ? SlotCoord::fixed(s.at(0)) : s;
  };
  switch (form) {
  case WarpForm::Parameterised: {
    // Warp w sits at (w / gN % gM, w % gN) of the warp grid and owns
    // fragment rows {row + r * gM} by columns {col + c * gN}: warps
    // interleave at fragment granularity, which is how `apple_mma` lays C
    // out, so a register's fragment is its own warp's. An axis the warps do
    // not divide folds to a constant.
    //
    // The moduli are identities at runtime (w < gM * gN) and exist for the
    // compiler: a raw `warp` term has no provable range, so the epilogue's
    // bounds guards survive to the AIR as branches.
    const int64_t gM = mT / miCount, gN = nT / niCount;
    for (int64_t r = 0; r < miCount; ++r) {
      const SlotCoord mi = gM == 1
                               ? SlotCoord::fixed(r)
                               : fold(SlotCoord::blocked(1, r * gM, gN, gM));
      for (int64_t col = 0; col < niCount; ++col) {
        const SlotCoord ni = gN == 1
                                 ? SlotCoord::fixed(col)
                                 : fold(SlotCoord::blocked(1, col * gN, 1, gN));
        out.push_back({mi, ni, (int)out.size()});
      }
    }
    return out;
  }

  case WarpForm::PerWarp:
    // Warp w owns every fragment congruent to w.
    for (int64_t f = w; f < mT * nT; f += numWarps)
      out.push_back({SlotCoord::fixed(fragRowOf(f, nT)),
                     SlotCoord::fixed(fragColOf(f, nT)), (int)out.size()});
    return out;
  }
  return out;
}

// ── which cover of the fragment grid the warps take ───────────────────────
//
// Every exact cover assigns each warp a (miCount x niCount) block, loading
// miCount A fragments and niCount B fragments per K slice. Covers are
// compared lexicographically on:
//
//   1. Device traffic per threadgroup. A device-read A is fetched by every
//      warp owning its row-block, so a warp grid with gN columns reads the
//      whole A slice gN times. Pool re-reads do not count: those are on-chip
//      broadcasts.
//
//   2. Operand fragments per warp, miCount + niCount, for a fixed product.
//      Minimised at the squarest factorization.
inline std::vector<WarpCover> exactCovers(const WarpGrid &g) {
  std::vector<WarpCover> out;
  if (!g.disjointC || g.numWarps <= 0 || !g.warpsDivideFrags())
    return out;
  const int64_t fpw = g.nFrag() / g.numWarps;
  for (int64_t mi = 1; mi <= fpw; ++mi) {
    if (fpw % mi)
      continue;
    const int64_t ni = fpw / mi;
    if (g.mT % mi || g.nT % ni)
      continue;
    if ((g.mT / mi) * (g.nT / ni) != g.numWarps)
      continue;
    // A banded C is drained by compile-time row, so only covers whose row
    // coordinate is constant (miCount == mT, one warp-grid row) qualify.
    if (g.bandedC && mi != g.mT)
      continue;
    out.push_back({mi, ni});
  }
  return out;
}

inline int64_t coverDeviceTraffic(const WarpGrid &g, const WarpCover &c) {
  return g.aDirect ? g.numWarps * c.mi : 0;
}

inline WarpProgram planWarpProgram(const WarpGrid &g) {
  WarpProgram p;
  p.because = "no usable exact cover, so each warp gets its own guarded block";
  const std::vector<WarpCover> covers = exactCovers(g);
  if (covers.empty())
    return p;

  WarpCover best;
  for (const WarpCover &c : covers) {
    if (g.cover.set() && c == g.cover) {
      best = c;
      p.because = "the cover the plan chose";
      break;
    }
    // The ascending-mi scan breaks full ties toward the wider block.
    if (best.set() &&
        (coverDeviceTraffic(g, c) > coverDeviceTraffic(g, best) ||
         (coverDeviceTraffic(g, c) == coverDeviceTraffic(g, best) &&
          c.mi + c.ni >= best.mi + best.ni)))
      continue;
    best = c;
    p.because = "the cover reading device A once, then the squarest";
  }
  p.form = WarpForm::Parameterised;
  p.miCount = best.mi;
  p.niCount = best.ni;
  return p;
}

} // namespace agpu

#endif // AGPU_WARP_SLOTS_H
