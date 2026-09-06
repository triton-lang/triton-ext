// PanelSchedule.h - the panel-dot iteration order, as data.
#ifndef AGPU_PANEL_SCHEDULE_H
#define AGPU_PANEL_SCHEDULE_H

#include "agpu/core/CoordGuard.h"
#include "agpu/core/TileView.h"
#include "agpu/core/Units.h"
#include "agpu/plan/DotPlan.h"
#include "agpu/plan/WarpSlots.h"

#include <cstdint>
#include <vector>

namespace agpu {

// Half-open extent on one axis.
struct Range {
  int64_t lo = 0, hi = 0;
  int64_t size() const { return hi - lo; }
  bool operator==(const Range &o) const { return lo == o.lo && hi == o.hi; }
};

// One (batch, M, N, K) tile of the panel walk.
struct PanelTile {
  int64_t batch = 0;
  Range m, n, k;

  // The product has a leading batch axis, so every register coordinate and
  // every window carries one more dimension, even at Bd == 1.
  bool batched = false;

  // Fragment counts within this tile, rounded up: a ragged final tile still
  // computes its edge fragment, there is no smaller MMA.
  int64_t mFrags() const { return fragsFor(m.size()); }
  int64_t nFrags() const { return fragsFor(n.size()); }
  int64_t kSteps() const { return fragsFor(k.size()); }

  // The K panel that completes the contraction: only then is C read back.
  bool finalK = false;

  bool stageB = true;

  // Element widths, for the bank-conflict decision.
  int64_t aElemBytes = 0;
  int64_t bElemBytes = 0;

  // The pitch the panel chose (`Panel::stagePad`).
  bool stagePad = true;

  // A is read straight from device memory: no StageA phase, no barrier for
  // it, no pool region.
  bool aDirect = false;

  WarpCover cover;
  bool renameC = false;

  // Pool views for this tile, in fragment-aligned extents: the MMA reads and
  // writes whole 8x8 fragments however ragged the shape is. `m` and `n`
  // still carry the true extent and the readback guard is built from them.
  // Shape comes from `stagedTileView`, the same formula `panelCost` reads.
  TileView aView() const {
    return stagedTileView(m.size(), k.size(), aElemBytes, stagePad);
  }
  TileView bView() const {
    return stagedTileView(k.size(), fragAlignedExtent(n.size()), bElemBytes,
                          stagePad);
  }
  TileView cView() const {
    // C is fp32 accumulators, whatever the operands are.
    return stagedTileView(m.size(), fragAlignedExtent(n.size()), kAccBytes,
                          stagePad);
  }

  // The same views in the product's coordinates: they carry the tile's
  // position as an origin. The MMA keeps the plain views above, its fragment
  // indices are tile-local. A batched frame has the batch axis at stride 0,
  // since the pool holds one slice.
  TileView aStagedView() const {
    return productFrame(aView().originAt({m.lo, k.lo}));
  }
  TileView bStagedView() const {
    return productFrame(bView().originAt({k.lo, n.lo}));
  }
  TileView cStagedView() const {
    return productFrame(cView().originAt({m.lo, n.lo}));
  }

  // Whether this tile's edge falls mid-fragment on either axis. The readback
  // needs a guard when it does.
  bool raggedM() const { return m.size() % kSgFragDim != 0; }
  bool raggedN() const { return n.size() % kSgFragDim != 0; }
  bool ragged() const { return raggedM() || raggedN(); }

  // The window of C this tile may write back: its real elements, in
  // tile-local coordinates.
  std::vector<CoordWindow> readbackWindows() const {
    return windowsFor({0, m.size()}, {0, n.size()});
  }

  // The window of each operand this tile covers, in the product's frame.
  // A is (m, k), B is (k, n), C is (m, n), each behind the batch axis when
  // the product has one.
  std::vector<CoordWindow> aWindows() const { return windowsFor(m, k); }
  std::vector<CoordWindow> bWindows() const { return windowsFor(k, n); }
  std::vector<CoordWindow> cWindows() const { return windowsFor(m, n); }

private:
  std::vector<CoordWindow> windowsFor(const Range &r0, const Range &r1) const {
    std::vector<CoordWindow> w;
    if (batched)
      w.push_back(batchWindow(0, batch));
    const int d = batched ? 1 : 0;
    w.push_back(CoordWindow{d, r0.lo, r0.hi});
    w.push_back(CoordWindow{d + 1, r1.lo, r1.hi});
    return w;
  }

  TileView productFrame(const TileView &v) const {
    if (!batched)
      return v;
    return TileView({1, v.extentAt(0), v.extentAt(1)},
                    {0, v.strideAt(0), v.strideAt(1)}, v.origin());
  }
};

// The whole walk, flattened.
struct PanelSchedule {
  std::vector<PanelTile> tiles;

  int64_t size() const { return (int64_t)tiles.size(); }
  bool empty() const { return tiles.empty(); }

  // Tiles that read C back. One per (batch, M, N) position.
  int64_t readbackCount() const {
    int64_t n = 0;
    for (const PanelTile &t : tiles)
      if (t.finalK)
        ++n;
    return n;
  }
};

// `m0` is the inner loop relative to `n0` and `k0` so consecutive tiles share
// a B panel, which is what lets `stageB` skip a restage where no drain has
// intervened. Reordering the nest silently disables that and still emits
// correct output.
template <class Fn>
inline void forEachPanelTile(const DotFacts &f, const Panel &p, Fn &&fn) {
  if (p.mp <= 0 || p.np <= 0 || p.kp <= 0)
    return;

  int64_t resB = -1, resN = -1, resK = -1;
  for (int64_t bi = 0; bi < f.Bd; ++bi)
    for (int64_t n0 = 0; n0 < f.N; n0 += p.np)
      for (int64_t m0 = 0; m0 < f.M; m0 += p.mp)
        for (int64_t k0 = 0; k0 < f.K; k0 += p.kp) {
          PanelTile t;
          t.batch = bi;
          t.batched = f.batched();
          t.m = {m0, std::min(m0 + p.mp, f.M)};
          t.n = {n0, std::min(n0 + p.np, f.N)};
          t.k = {k0, std::min(k0 + p.kp, f.K)};
          t.finalK = (k0 + p.kp) >= f.K;
          t.aElemBytes = f.aElemBytes;
          t.bElemBytes = f.bElemBytes;
          t.stagePad = p.stagePad;
          t.aDirect = f.aDirect;
          if (p.renameWarps.set()) {
            t.renameC = true;
            t.cover = {t.mFrags() / p.renameWarps.mi,
                       t.nFrags() / p.renameWarps.ni};
          }
          t.stageB = !(bi == resB && n0 == resN && k0 == resK);
          if (t.stageB) {
            resB = bi;
            resN = n0;
            resK = k0;
          }
          fn(t);
          // C shares the pool's bytes with the operands, which is sound
          // only because every tile stages what it reads after the
          // previous tile's readback. A drain therefore forfeits the
          // resident B: its bytes now hold C.
          if (t.finalK)
            resB = resN = resK = -1;
        }
}

// Materialises every tile. A large panelled dot reaches millions of them, so
// prefer `forEachPanelTile` where the whole schedule is not needed at once.
inline PanelSchedule planPanelSchedule(const DotFacts &f, const Panel &p) {
  PanelSchedule s;
  forEachPanelTile(f, p, [&](const PanelTile &t) { s.tiles.push_back(t); });
  return s;
}

// ── what happens inside one tile ──────────────────────────────────────────

// The phases of a tile, in the order the emitter runs them.
enum class PanelPhase {
  StageA,   // scatter A registers into the pool
  StageB,   // ... then B, after a barrier
  Mma,      // ... then the MMA grid, after a barrier
  Drain,    // ... on the final K panel, store the accumulators into the pool
  Readback, // ... and gather C, one barrier after the drain
  Rename,
};

// Every phase reads what the previous one wrote, so every transition is a
// barrier, except a rename, which reads only its own warp's fragments.
inline bool needsBarrierBefore(PanelPhase ph) {
  return ph != PanelPhase::Rename;
}

// The phases a tile runs, in order. A device-resident A has no StageA phase
// and no barrier for it.
inline std::vector<PanelPhase> phasesOf(const PanelTile &t) {
  std::vector<PanelPhase> ph;
  if (!t.aDirect)
    ph.push_back(PanelPhase::StageA);
  if (t.stageB)
    ph.push_back(PanelPhase::StageB);
  ph.push_back(PanelPhase::Mma);
  if (t.finalK && t.renameC) {
    ph.push_back(PanelPhase::Rename);
  } else if (t.finalK) {
    ph.push_back(PanelPhase::Drain);
    ph.push_back(PanelPhase::Readback);
  }
  return ph;
}

// The warp grid a tile's MMA phase runs on, from the tile's fragment counts.
// `aDirect` stays false even for a device-resident A: the grid's flag picks
// the row-affine warp form, which a panel tile's MMA does not implement.
inline WarpGrid panelWarpGrid(const PanelTile &t, int64_t numWarps,
                              int64_t hwWarps = 0) {
  WarpGrid g;
  g.mT = t.mFrags();
  g.nT = t.nFrags();
  g.numWarps = numWarps;
  g.hwWarps = hwWarps;
  g.cover = t.cover;
  return g;
}

inline ReadbackPlan panelTileReadback(const DotFacts &f, const PanelTile &t) {
  if (!t.cover.set())
    return {};
  WarpProgram wp;
  wp.form = WarpForm::Parameterised;
  wp.miCount = t.cover.mi;
  wp.niCount = t.cover.ni;
  const int64_t warps = warpsFor(f);
  ReadbackWindow w;
  w.rowLo = t.m.lo;
  w.rowHi = t.m.lo + t.mFrags() * kSgFragDim;
  w.colLo = t.n.lo;
  w.colHi = t.n.lo + t.nFrags() * kSgFragDim;
  w.batch = t.batched ? t.batch : -1;
  return planReadback(f.cDims, wp.slots(0, t.mFrags(), t.nFrags(), warps),
                      f.cRegs, warps, w);
}

// ── invariants worth asserting ────────────────────────────────────────────

// Every (batch, m, n) position must read C back exactly once.
inline bool readbackIsExactlyOncePerPosition(const DotFacts &f,
                                             const PanelSchedule &s,
                                             const Panel &p) {
  const int64_t panelsM = (f.M + p.mp - 1) / p.mp;
  const int64_t panelsN = (f.N + p.np - 1) / p.np;
  return s.readbackCount() == f.Bd * panelsM * panelsN;
}

// Every element of the output is covered by exactly one (batch, m, n).
inline bool tilesCoverOutput(const DotFacts &f, const PanelSchedule &s) {
  std::vector<int> hit((std::size_t)(f.Bd * f.M * f.N), 0);
  for (const PanelTile &t : s.tiles) {
    if (!t.finalK)
      continue;
    for (int64_t i = t.m.lo; i < t.m.hi; ++i)
      for (int64_t j = t.n.lo; j < t.n.hi; ++j)
        ++hit[(std::size_t)((t.batch * f.M + i) * f.N + j)];
  }
  for (int h : hit)
    if (h != 1)
      return false;
  return true;
}

// Every K panel is visited exactly once per output position.
inline bool contractionIsComplete(const DotFacts &f, const PanelSchedule &s,
                                  const Panel &p) {
  const int64_t panelsK = (f.K + p.kp - 1) / p.kp;
  int64_t perPosition = 0;
  for (const PanelTile &t : s.tiles)
    if (t.batch == 0 && t.m.lo == 0 && t.n.lo == 0)
      ++perPosition;
  return perPosition == panelsK;
}

} // namespace agpu

#endif // AGPU_PANEL_SCHEDULE_H
