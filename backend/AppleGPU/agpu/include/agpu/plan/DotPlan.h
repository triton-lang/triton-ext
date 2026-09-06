// DotPlan.h - choosing how a dot is lowered and what it costs.
#ifndef AGPU_DOT_PLAN_H
#define AGPU_DOT_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/core/Padding.h"
#include "agpu/core/TileView.h"
#include "agpu/core/Units.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/ReadbackPlan.h"
#include "agpu/plan/WarpSlots.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <variant>

namespace agpu {

// ── what the IR tells us ──────────────────────────────────────────────────

// Facts about one dot, derived from TTGIR once and never recomputed.
struct DotFacts {
  int rank = 2;
  int64_t Bd = 1, M = 0, N = 0, K = 0;
  int64_t aElemBytes = 0, bElemBytes = 0;
  // What C is stored as. Accumulation is always kAccBytes wide.
  int64_t cElemBytes = 4;
  int64_t numWarps = 1;

  bool intAcc = false;   // integer accumulator: scalar path
  bool aInPlace = false; // A already resident in a threadgroup buffer
  bool bInPlace = false;
  bool aDirect = false;      // A readable straight from device memory
  bool carriedAcc = false;   // C is loop-carried: this dot runs per iteration
  bool fusedAcc = false;     // C lives in registers across a K loop
  bool cInitNonzero = false; // the fused loop's init is not zeros
  bool cDirect = false;      // C stores straight to device
  bool cFallback = false;    // ... but keeps a pool arm for ragged tiles
  bool cRename = false;      // C's consumers read the fragment's own lanes

  std::vector<LayoutBasis> cDims;
  int64_t cRegs = 0;

  // Fragment grid, rounded up: there is no MMA smaller than 8x8. The readback
  // guards against storing a partial fragment past the edge.
  int64_t mT() const { return fragsFor(M); }
  int64_t nT() const { return fragsFor(N); }
  int64_t kT() const { return fragsFor(K); }
  int64_t nFrag() const { return mT() * nT(); }

  // Facts-space form of `cDrainSkipsPool`, for fit tests that run before a
  // strategy is chosen. The two must agree.
  bool cCostsPoolNothing() const { return (cDirect && !cFallback) || cRename; }

  bool raggedM() const { return M % kSgFragDim != 0; }
  bool raggedN() const { return N % kSgFragDim != 0; }
  bool ragged() const { return raggedM() || raggedN(); }

  // A leading batch axis exists, even of extent 1.
  bool batched() const { return rank > 2; }

  // K may not be ragged: a partial K fragment would sum operand elements that
  // do not exist. Nothing zero-pads it yet.
  bool usable() const {
    return M != 0 && N != 0 && K != 0 && K % kSgFragDim == 0;
  }
};

// ── operand staging cost ──────────────────────────────────────────────────

// Bytes A and B occupy in the pool. An unstaged operand costs nothing; a
// staged one carries row padding so consecutive rows land in different banks.
struct StageBytes {
  Bytes a, b;

  Bytes ab() const { return a + b; }
};

// Rows rounded up to whole fragments, columns padded per core/Padding.h. The
// pool reservation, the staging scatter and simdgroup_load's leading dimension
// all derive from this view. `cols` is the extent the caller means to address.
inline TileView stagedTileView(int64_t rows, int64_t cols, int64_t elemBytes,
                               bool pad = true) {
  return TileView::rowMajorPadded({fragAlignedExtent(rows), cols},
                                  pad ? padElemsFor(cols, elemBytes) : 0);
}

inline Bytes stagedTileBytes(int64_t rows, int64_t cols, int64_t elemBytes,
                             bool pad = true) {
  return Bytes(stagedTileView(rows, cols, elemBytes, pad).cosizeElems() *
               elemBytes);
}

// The whole staged operand, batch axis included: Bd slices laid back to back,
// batch stride the slice cosize. Only the scalar dot stages a batched operand
// whole; the MMA strategies go to the panel walk, one slice at a time.
inline TileView stagedOperandView(const DotFacts &f, int64_t rows, int64_t cols,
                                  int64_t elemBytes, bool pad = true) {
  const TileView slice = stagedTileView(rows, cols, elemBytes, pad);
  if (!f.batched())
    return slice;
  return TileView({f.Bd, slice.extentAt(0), slice.extentAt(1)},
                  {slice.cosizeElems(), slice.strideAt(0), slice.strideAt(1)});
}

inline Bytes stagedOperandBytes(const DotFacts &f, int64_t rows, int64_t cols,
                                int64_t elemBytes, bool pad = true) {
  return Bytes(stagedOperandView(f, rows, cols, elemBytes, pad).cosizeElems() *
               elemBytes);
}

inline StageBytes planStageBytes(const DotFacts &f, bool pad = true) {
  StageBytes s;
  if (!f.aInPlace && !f.aDirect)
    s.a = stagedOperandBytes(f, f.M, f.K, f.aElemBytes, pad);
  if (!f.bInPlace)
    s.b = stagedOperandBytes(f, f.K, fragAlignedExtent(f.N), f.bElemBytes, pad);
  return s;
}

// ── panel geometry ────────────────────────────────────────────────────────

// A panel: the sub-tile of the operands that fits the pool at once. When A
// and B together exceed the budget, the dot walks panels instead. C overlays
// the operands: registers carry it across the K panels and it reaches the pool
// only at the drain.
struct Panel {
  int64_t mp = 0, np = 0, kp = 0;
  Bytes aBytes, bBytes, cBytes;

  // The pitch every tile of this walk stages at.
  bool stagePad = true;

  WarpCover renameWarps;

  Bytes total() const { return maxBytes(aBytes + bBytes, cBytes); }
};

// Bytes come from `stagedTileBytes`, the same formula PanelSchedule.h's tile
// views address through. A device-resident A (`aStaged == false`) costs
// nothing: its fragments are read where they lie.
inline Panel panelCost(int64_t m, int64_t n, int64_t k, int64_t elemBytes,
                       int64_t accBytes, bool aStaged = true, bool pad = true) {
  Panel p;
  p.mp = m;
  p.np = n;
  p.kp = k;
  p.stagePad = pad;
  p.aBytes = aStaged ? stagedTileBytes(m, k, elemBytes, pad) : Bytes(0);
  p.bBytes = stagedTileBytes(k, fragAlignedExtent(n), elemBytes, pad);
  p.cBytes = stagedTileBytes(m, fragAlignedExtent(n), accBytes, pad);
  return p;
}

// Ordered so `stageB`'s consecutive-tile skip survives the M walk: fewest
// tiles, then the largest panel, then the wider N.
inline Panel planPanelAt(int64_t M, int64_t N, int64_t K, int64_t elemBytes,
                         int64_t accBytes, Bytes budget, bool aStaged,
                         bool pad) {
  const auto cost = [&](int64_t m, int64_t n, int64_t k) {
    return panelCost(m, n, k, elemBytes, accBytes, aStaged, pad);
  };
  int64_t mp = M, np = N;
  if (cost(mp, np, K).cBytes > budget) {
    int64_t bestTiles = 0, bestArea = 0;
    bool found = false;
    for (int64_t m = fragAlignedExtent(M); m >= kSgFragDim; m -= kSgFragDim)
      for (int64_t n = fragAlignedExtent(N); n >= kSgFragDim; n -= kSgFragDim) {
        if (cost(m, n, K).cBytes > budget)
          continue;
        const int64_t tiles = ((M + m - 1) / m) * ((N + n - 1) / n);
        const int64_t area = m * n;
        if (found && (tiles > bestTiles ||
                      (tiles == bestTiles &&
                       (area < bestArea || (area == bestArea && n <= np)))))
          continue;
        found = true;
        bestTiles = tiles;
        bestArea = area;
        mp = m;
        np = n;
      }
    if (!found) {
      mp = kSgFragDim;
      np = kSgFragDim;
    }
  }
  // Fragment steps: halving K=24 reaches 6, which rounds to zero fragments
  // and computes zero.
  int64_t kp = K - K % kSgFragDim;
  while (kp > kSgFragDim && cost(mp, np, kp).total() > budget)
    kp -= kSgFragDim;
  while (cost(mp, np, kp).total() > budget) {
    if (mp >= np && mp > kSgFragDim)
      mp -= kSgFragDim;
    else if (np > kSgFragDim)
      np -= kSgFragDim;
    else
      break;
  }
  return cost(mp, np, kp);
}

// How many tiles the walk emits for this panel.
inline int64_t panelTiles(int64_t M, int64_t N, int64_t K, const Panel &p) {
  return ((M + p.mp - 1) / p.mp) * ((N + p.np - 1) / p.np) *
         ((K + p.kp - 1) / p.kp);
}

// Whether a fused dot's staged operands carry the bank pad. Dropped when only
// the plain pitch fits the overlay, or when the pad costs a residency step
// below `kTGResidencyFloor`. Where neither pitch fits, answers padded, which
// is what the strategy was sized against.
inline bool fusedPadWorthCarrying(Bytes padded, Bytes plain, Bytes budget) {
  if (padded > budget)
    return !(plain <= budget);
  if (plain > budget)
    return true;
  const int64_t padRes = tgResidency(padded.count());
  return padRes >= tgResidency(plain.count()) || padRes >= kTGResidencyFloor;
}

// The pad, unless it costs whole tiles: an extra tile is a full restage plus
// barriers.
inline Panel planPanel(int64_t M, int64_t N, int64_t K, int64_t elemBytes,
                       int64_t accBytes, Bytes budget, bool aStaged = true) {
  const Panel padded =
      planPanelAt(M, N, K, elemBytes, accBytes, budget, aStaged, true);
  const Panel plain =
      planPanelAt(M, N, K, elemBytes, accBytes, budget, aStaged, false);
  return panelTiles(M, N, K, plain) < panelTiles(M, N, K, padded) ? plain
                                                                  : padded;
}

// ── strategy ──────────────────────────────────────────────────────────────

// ── the integer lift ──────────────────────────────────────────────────────

// The K past which an i8 dot's f32 accumulation stops being exact: an 8-bit
// product fits 16 bits, f32 represents integers up to 2^24, so a sum of K
// products is exact while K * 2^16 <= 2^24.
inline constexpr int64_t kIntLiftMaxK = (1 << 24) / (1 << 16);

// `simdgroup_matrix` has no integer element, so an integer dot normally
// lowers to a per-thread loop. An i8 dot whose K fits the bound above lifts
// instead: operands staged as f32, multiplied in f32, readback back to i32.
//
// Exclusions:
//   - fused accumulator: the sum crosses a runtime trip count, so the bound
//     cannot be proven. A carried accumulator is fine, each trip is one K.
//   - operands wider than a byte overflow the bound almost immediately.
//   - in-place or device-read operands are typed i8 and `simdgroup_load`
//     cannot convert.
inline bool liftsToFloatMma(const DotFacts &f) {
  if (!f.intAcc || f.fusedAcc)
    return false;
  if (f.aElemBytes != 1 || f.bElemBytes != 1)
    return false;
  if (f.aInPlace || f.bInPlace)
    return false;
  return f.K > 0 && f.K <= kIntLiftMaxK;
}

// Per-strategy payloads. A field lives in exactly one of these.

// Integer accumulator: a per-thread K loop, no MMA.
struct ScalarParams {
  bool intAcc = false;

  // What the K loop accumulates in and what both operands widen to before
  // the multiply. An integer dot accumulates at 32 bits.
  ElemType acc = i32();

  // Dropped when only the plain pitch fits, since the K loop then reads
  // plain elements directly.
  bool stagePad = true;
};

struct PanelParams {
  Panel panel;
  int64_t panelsM = 0, panelsN = 0, panelsK = 0;
  int64_t tiles() const { return panelsM * panelsN * panelsK; }
};

struct DirectParams {
  int64_t fragsPerWarp = 0;
  bool disjointC = false; // C does not overlap A/B in the pool
  int64_t bandRows = 0;   // rows of C read back at a time

  // Dropped when the pad alone flips a whole-tile C into a banded readback.
  // Every staged-tile consumer reads this flag.
  bool stagePad = true;
};

struct FusedParams {
  int64_t fragsPerWarp = 0;
  bool cDirect = false;

  // See `fusedPadWorthCarrying`.
  bool stagePad = true;
};

// Zero where the operands leave no room: the drain falls back to guarded
// scalar stores.
inline Bytes edgeScratchFor(const DotFacts &f, Bytes staged, Bytes budget) {
  const Bytes want(f.numWarps * kSgFragDim * kSgFragDim * kAccBytes);
  return staged + want <= budget ? want : Bytes(0);
}

// Shared by `Plan::storesCDirect` (what the emitter writes) and `CReserve`
// (what the pool reserves). The two must not disagree.
inline bool cDrainSkipsPool(const FusedParams &fp, const DotFacts &f) {
  return fp.cDirect && f.cCostsPoolNothing();
}

using StrategyParams =
    std::variant<ScalarParams, PanelParams, DirectParams, FusedParams>;

// What the pool must hold. `cNeed` is the total including C.
struct PoolPlan {
  Bytes stagedAB;
  Bytes cNeed;

  // Every arm of `CReserve` folds `stagedAB` into `cNeed` already.
  Bytes reserved() const { return cNeed; }
  // C's own reservation beyond the operands, saturating.
  Bytes cReserve() const { return maxBytes(cNeed - stagedAB, Bytes(0)); }
  int64_t residency() const { return tgResidency(reserved().count()); }
};

// A readback that emits a threadgroup C pointer needs something to name: a
// pointer into a zero-byte region is not legal to declare. A floor on zero
// only; a provably-direct store needs nothing.
inline constexpr int64_t kMinPoolPtrBytes = kSgFragDim * kSgFragDim * kAccBytes;

// What the budget admits, precomputed once for the strategy rules below. The
// staged path can band C; a fused drain crosses C whole.
struct DotFit {
  bool operandsAndBand = false; // staged A+B plus one band of C, either pitch
  bool wholeC = false;          // one whole C tile, at the fused pitch
};

struct Plan {
  enum class Kind { Unsupported, Scalar, Panel, Direct, Fused };

  Kind kind = Kind::Unsupported;
  DotFacts facts;
  StageBytes stage;
  PoolPlan pool;
  StrategyParams params;
  ReadbackPlan readback;
  WarpCover cover;

  Bytes edgeScratch;

  // Set when `liftsToFloatMma` fired: `facts` then carry the lifted shape,
  // 4-byte staged operands and no direct reads. Anything downstream spelling
  // an operand element asks this.
  bool intThroughFloat = false;

  // What the budget admitted, kept so nothing has to re-derive it.
  DotFit fit;

  // `std::get` aborts on the wrong alternative in this exceptions-disabled
  // build, so these use `get_if` and return a static default instead.
  const ScalarParams &scalar() const { return payload<ScalarParams>(); }
  const PanelParams &panel() const { return payload<PanelParams>(); }
  const DirectParams &direct() const { return payload<DirectParams>(); }
  const FusedParams &fused() const { return payload<FusedParams>(); }

  // Accumulators belong to an enclosing loop: declared before this dot,
  // stored after it, only multiplied into here.
  bool accumulatorsOutlivePass() const { return kind == Kind::Fused; }

  bool readsBackByRename() const {
    return kind != Kind::Scalar && kind != Kind::Unsupported && facts.cRename &&
           !storesCDirect();
  }

  // The pitch C is staged at, from whichever payload carries it.
  bool padStagedC() const {
    if (const DirectParams *d = std::get_if<DirectParams>(&params))
      return d->stagePad;
    if (const FusedParams *fp = std::get_if<FusedParams>(&params))
      return fp->stagePad;
    if (const ScalarParams *sp = std::get_if<ScalarParams>(&params))
      return sp->stagePad;
    return true;
  }

  // The C tile as staged: fragment-aligned, at this plan's pitch.
  TileView cStagedView() const {
    return stagedTileView(facts.M, fragAlignedExtent(facts.N), kAccBytes,
                          padStagedC());
  }

  // Rows of C the pool holds at once. Only a Direct plan bands; everything
  // else crosses whole.
  int64_t cBandRows() const {
    const DirectParams *d = std::get_if<DirectParams>(&params);
    return d ? d->bandRows : cStagedView().extentAt(0);
  }

  // Fused drain stores C straight to the device tensor: no pool region, no
  // readback, no post-loop barriers.
  bool storesCDirect() const {
    const FusedParams *fp = std::get_if<FusedParams>(&params);
    return fp && cDrainSkipsPool(*fp, facts);
  }

  bool edgeScratchFits() const { return edgeScratch.count() > 0; }

  bool cThroughPool() const {
    return kind != Kind::Scalar && !storesCDirect() && !facts.cRename;
  }

  // C's region of the threadgroup pool: how many bytes and whether they
  // overlay the staged operands at the pool base or follow them.
  //
  // A fused dot's C overlays: the drain runs once, after the loop's last MMA
  // read of A and B, behind a barrier. Every other strategy's C coexists with
  // the operands and sits after them.
  struct CPoolRegion {
    int64_t bytes = 0;
    bool overlaysOperands = false;
  };

  // The C staging region holds accumulator fragments, always fp32:
  // `simdgroup_store` deduces its pointer type from the fragment and a
  // `threadgroup half *` is a deduction conflict.
  ElemType cPoolElem() const { return f32(); }

  CPoolRegion cPoolRegion() const {
    // A drain that never reaches the pool has no C region: nothing indexes it
    // and nothing names it either.
    if (!cThroughPool())
      return {0, false};
    if (kind == Kind::Fused)
      return {stagedTileBytes(facts.M, fragAlignedExtent(facts.N), kAccBytes,
                              padStagedC())
                  .count(),
              true};
    if (kind == Kind::Panel)
      return {std::max(panel().panel.cBytes.count(), kMinPoolPtrBytes), true};
    return {std::max(pool.cReserve().count(), kMinPoolPtrBytes), false};
  }

private:
  template <class T> const T &payload() const {
    if (const T *p = std::get_if<T>(&params))
      return *p;
    static const T empty{};
    return empty;
  }
};

// What C costs. One overload per strategy payload.
struct CReserve {
  const DotFacts &f;
  Bytes stagedAB;
  Bytes budget;

  // Whole fragments: a 60-row C occupies 64 rows.
  Bytes cFull(bool pad = true) const {
    return stagedTileBytes(f.M, fragAlignedExtent(f.N), kAccBytes, pad);
  }

  // One band of C: 8 rows, what simdgroup_store writes at a time.
  Bytes cBand(bool pad = true) const {
    return Bytes(kSgFragDim *
                 stagedTileView(f.M, fragAlignedExtent(f.N), kAccBytes, pad)
                     .strideAt(0) *
                 kAccBytes);
  }

  // A scalar dot keeps its running sum in a register.
  Bytes operator()(const ScalarParams &) const { return stagedAB; }

  // A panel reserves exactly the panel, which already includes its C.
  Bytes operator()(const PanelParams &pp) const { return pp.panel.total(); }

  // A fused dot storing straight to device with no ragged arm never
  // materialises C. Otherwise C overlays the staged operands, so the
  // reservation is a max.
  //
  // No clamp to the budget: `kKindRules` already refuses to fuse a C too
  // large to cross the pool whole and clamping would reserve less than the
  // drain addresses.
  Bytes operator()(const FusedParams &fp) const {
    if (cDrainSkipsPool(fp, f))
      return stagedAB + edgeScratchFor(f, stagedAB, budget);
    if (f.cRename)
      return stagedAB;
    return maxBytes(stagedAB, cFull(fp.stagePad));
  }

  // Direct: C beside the operands if the whole tile fits, else as many whole
  // bands as the remainder holds, floored at one. `selectKind` already
  // guaranteed one band fits or the shape would have panelled.
  Bytes operator()(const DirectParams &dp) const {
    if (f.cRename)
      return stagedAB;
    const bool pad = dp.stagePad;
    if (stagedAB + cFull(pad) <= budget)
      return stagedAB + cFull(pad);
    const Bytes left = maxBytes(budget - stagedAB, Bytes(0));
    const Bytes bands(left.count() - left.count() % cBand(pad).count());
    return stagedAB + maxBytes(cBand(pad), minBytes(cFull(pad), bands));
  }
};

// The C reservation ladder. Kind is not a parameter: `params` carries it.
inline PoolPlan planPool(const DotFacts &f, const StageBytes &sb,
                         const StrategyParams &params, Bytes budget) {
  PoolPlan p;
  p.stagedAB = sb.ab();
  p.cNeed = std::visit(CReserve{f, p.stagedAB, budget}, params);

  // A zero reservation still has to be nameable: see kMinPoolPtrBytes.
  if (p.cNeed == Bytes(0))
    p.cNeed = Bytes(kMinPoolPtrBytes);
  return p;
}

// Rows of C that fit what is left after the operands, in whole fragments.
// `rowElems` is the staged C row stride, padding included.
inline int64_t bandRowsFor(int64_t rowElems, Bytes cBudget, int64_t accBytes) {
  if (rowElems <= 0 || accBytes <= 0)
    return 0;
  const int64_t rows = cBudget.count() / (rowElems * accBytes);
  return std::max<int64_t>(kSgFragDim, rows - rows % kSgFragDim);
}

// Fields a strategy can only fill once the pool is fixed: the band depends on
// what C was left, which depends on the strategy.
struct PoolDependent {
  const DotFacts &f;
  const PoolPlan &pool;

  void operator()(ScalarParams &) const {}
  void operator()(PanelParams &) const {}
  void operator()(FusedParams &) const {}

  void operator()(DirectParams &dp) const {
    dp.disjointC = pool.cReserve() > Bytes(0);
    // Asked as bytes: the whole tile's last row is short of the pad, so the
    // row division would come out one row shy.
    const Bytes full =
        stagedTileBytes(f.M, fragAlignedExtent(f.N), kAccBytes, dp.stagePad);
    dp.bandRows = f.cRename || pool.cReserve() >= full
                      ? fragAlignedExtent(f.M)
                      : bandRowsFor(stagedTileView(f.M, fragAlignedExtent(f.N),
                                                   kAccBytes, dp.stagePad)
                                        .strideAt(0),
                                    pool.cReserve(), kAccBytes);
  }
};

// Warps that actually own fragments and how many each owns.
inline int64_t warpsFor(const DotFacts &f) {
  return effectiveWarps(f.numWarps, f.nFrag());
}
inline int64_t fragsPerWarpFor(const DotFacts &f) {
  // effectiveWarps floors at 1, so the divide is always safe.
  const int64_t nw = warpsFor(f);
  return (f.nFrag() + nw - 1) / nw;
}

inline WarpGrid warpGridFor(const DotFacts &f, bool bandedC) {
  WarpGrid g;
  g.mT = f.mT();
  g.nT = f.nT();
  g.numWarps = warpsFor(f);
  g.hwWarps = f.numWarps;
  g.aDirect = f.aDirect;
  g.bandedC = bandedC;
  return g;
}

struct CoverChoice {
  WarpCover cover;
  ReadbackPlan readback;
  int64_t penalty = 0;
};

inline WarpCover panelRenameWarps(const DotFacts &f, const Panel &pn) {
  if (f.cDims.size() < 2 || f.cRegs <= 0 || f.intAcc)
    return {};
  const LayoutBasis &rowB = f.cDims[f.cDims.size() - 2];
  const LayoutBasis &colB = f.cDims.back();
  if (rowB.warp.size() != colB.warp.size())
    return {};
  int64_t wm = 1, wn = 1;
  for (std::size_t b = 0; b < rowB.warp.size(); ++b) {
    const bool onRow = rowB.warp[b] != 0, onCol = colB.warp[b] != 0;
    if (onRow == onCol)
      return {};
    (onRow ? wm : wn) *= 2;
  }
  if (wm * wn != f.numWarps || warpsFor(f) != f.numWarps)
    return {};
  const int64_t rowSpan = kSgFragDim * wm, colSpan = kSgFragDim * wn;
  if (f.M % rowSpan || f.N % colSpan || pn.mp % rowSpan || pn.np % colSpan)
    return {};
  const int64_t mT = std::min(pn.mp, f.M) / kSgFragDim;
  const int64_t nT = std::min(pn.np, f.N) / kSgFragDim;
  WarpProgram wp;
  wp.form = WarpForm::Parameterised;
  wp.miCount = mT / wm;
  wp.niCount = nT / wn;
  ReadbackWindow w;
  w.rowHi = mT * kSgFragDim;
  w.colHi = nT * kSgFragDim;
  w.batch = f.batched() ? 0 : -1;
  const ReadbackPlan rb = planReadback(f.cDims, wp.slots(0, mT, nT, f.numWarps),
                                       f.cRegs, f.numWarps, w);
  return rb.rename() ? WarpCover{wm, wn} : WarpCover{};
}

// The cover C's layout wants, taken only while its extra operand loads stay
// within the round trip a rename removes. An idle warp forbids it: its lanes
// hold duplicate coordinates and would store unassigned registers.
inline CoverChoice planCover(const DotFacts &f) {
  CoverChoice c;
  if (f.cDims.empty() || f.cRegs <= 0 || f.numWarps <= 0)
    return c;
  const WarpGrid g = warpGridFor(f, false);
  if (g.guardsIdleWarps())
    return c;
  const WarpProgram base = planWarpProgram(g);
  if (base.form != WarpForm::Parameterised)
    return c;
  const int64_t baseFrags = base.miCount + base.niCount;
  const int64_t baseDev = coverDeviceTraffic(g, {base.miCount, base.niCount});
  const int64_t roundTrip = 2 * (g.nFrag() / g.numWarps);

  int64_t bestPenalty = -1;
  for (const WarpCover &cand : exactCovers(g)) {
    if (coverDeviceTraffic(g, cand) != baseDev)
      continue;
    const int64_t penalty = (cand.mi + cand.ni - baseFrags) * f.kT();
    if (penalty > roundTrip || (bestPenalty >= 0 && penalty >= bestPenalty))
      continue;
    WarpProgram wp;
    wp.form = WarpForm::Parameterised;
    wp.miCount = cand.mi;
    wp.niCount = cand.ni;
    ReadbackPlan rb = planReadback(f.cDims, wp.slots(0, g.mT, g.nT, g.numWarps),
                                   f.cRegs, g.numWarps);
    if (!rb.rename())
      continue;
    bestPenalty = penalty;
    c.penalty = penalty;
    c.cover = cand;
    c.readback = std::move(rb);
  }
  return c;
}

// Which lowering a dot takes. Rows are checked in priority order.
struct KindRule {
  Plan::Kind kind;
  bool (*applies)(const DotFacts &, const DotFit &);
  const char *because;
};

inline constexpr KindRule kKindRules[] = {
    {Plan::Kind::Scalar,
     [](const DotFacts &f, const DotFit &) { return f.intAcc; },
     "an integer accumulator has no MMA path"},
    {Plan::Kind::Panel,
     [](const DotFacts &, const DotFit &fit) { return !fit.operandsAndBand; },
     "A and B together overflow the pool, or leave no room for C"},
    // A batched MMA dot walks the panel schedule even when one slice fits
    // whole: the panel walk is the only place a batch loop exists.
    // `readbackFor` resolves register names at compile time, so a runtime
    // slice counter cannot select the registers a slice's result lands in.
    {Plan::Kind::Panel,
     [](const DotFacts &f, const DotFit &) { return f.batched(); },
     "a batched dot walks one slice of the product at a time"},
    // The fused drain never bands, so a C the pool cannot hold whole falls
    // through to Direct, which pays a per-iteration round trip but bands.
    {Plan::Kind::Fused,
     [](const DotFacts &f, const DotFit &fit) {
       return f.fusedAcc && fit.wholeC;
     },
     "C stays in registers across the K loop"},
    {Plan::Kind::Direct, [](const DotFacts &, const DotFit &) { return true; },
     "a single dot whose operands fit"},
};

inline Plan::Kind selectKind(const DotFacts &f, const DotFit &fit) {
  for (const KindRule &r : kKindRules)
    if (r.applies(f, fit))
      return r.kind;
  return Plan::Kind::Unsupported;
}

// Why a plan is what it is, for diagnostics.
inline const char *reasonFor(Plan::Kind k) {
  for (const KindRule &r : kKindRules)
    if (r.kind == k)
      return r.because;
  return "shape is not fragment-aligned";
}

// The rule that fired, unlike `reasonFor`: two rows select Panel, so
// `reasonFor(Panel)` returns whichever comes first. Re-runs the predicates,
// which are pure functions of facts the plan already carries.
inline const char *ruleFiredFor(const DotFacts &f, const DotFit &fit) {
  for (const KindRule &r : kKindRules)
    if (r.applies(f, fit))
      return r.because;
  return reasonFor(Plan::Kind::Unsupported);
}

inline Decision dotDecision(const Plan &p) {
  if (p.kind != Plan::Kind::Unsupported)
    return Decision::emitted();
  return Decision::declined("emitDot", reasonFor(p.kind));
}

// Staging is decided before anything asks whether it fits, so a shape cannot
// be judged to fit unpadded and then overflow once staged.
inline Plan planDot(const DotFacts &facts, Bytes budget) {
  Plan p;
  DotFacts f = facts;

  if (!f.usable()) {
    p.facts = f;
    p.kind = Plan::Kind::Unsupported;
    return p;
  }

  // -1. The integer lift, before anything reads the operand widths, because
  //     it changes them: a lifted dot stages f32. The direct arms close
  //     because their buffers are typed i8 where fragments need float.
  if (liftsToFloatMma(f)) {
    p.intThroughFloat = true;
    f.intAcc = false;
    f.aElemBytes = 4;
    f.bElemBytes = 4;
    f.aDirect = false;
    f.cDirect = false;
    f.cFallback = false;
  }

  // The device-window proofs behind `aDirect` and `cDirect` describe a 2-D
  // window with no batch axis, so honouring either would address slice 0 for
  // every slice.
  if (f.batched())
    f.aDirect = f.cDirect = false;

  // The direct drain stores fragments only, so an addend has nowhere to land.
  if (f.cInitNonzero) {
    f.cDirect = false;
    f.cFallback = false;
  }

  // Staged fragments measured faster than device-resident ones, so A stages
  // whenever the whole dot still fits. Not for a loop-carried dot, which would
  // pay a scatter and a barrier every trip.
  if (f.aDirect && !f.carriedAcc) {
    DotFacts staged = f;
    staged.aDirect = false;
    const bool wholePadded =
        planStageBytes(staged).ab() +
            stagedTileBytes(f.M, fragAlignedExtent(f.N), kAccBytes) <=
        budget;
    const bool wholePlain =
        planStageBytes(staged, false).ab() +
            stagedTileBytes(f.M, fragAlignedExtent(f.N), kAccBytes, false) <=
        budget;
    if (wholePadded || wholePlain)
      f.aDirect = false;
  }

  CoverChoice chosen;
  if (!f.batched() && !f.intAcc)
    chosen = planCover(f);
  if (f.fusedAcc && chosen.penalty != 0)
    chosen = {};
  const bool mayRename = chosen.readback.rename();
  p.facts = f;

  // 1. What the operands cost, padding included.
  p.stage = planStageBytes(f);

  // 2. Which strategy. "Fits" means the operands plus at least one band of C,
  //    the smallest unit the banded readback can store, unless C never
  //    reaches the pool (`cDirect`). Asked at both pitches: the Direct arm
  //    can emit unpadded, so a shape only the plain pitch admits must not
  //    panel.
  const auto bandBytes = [&](bool pad) {
    return Bytes(
        f.cCostsPoolNothing() || mayRename
            ? 0
            : kSgFragDim *
                  stagedTileView(f.M, fragAlignedExtent(f.N), kAccBytes, pad)
                      .strideAt(0) *
                  kAccBytes);
  };
  const bool fitsPadded = p.stage.ab() + bandBytes(true) <= budget;
  const bool fitsPlain =
      planStageBytes(f, false).ab() + bandBytes(false) <= budget;
  DotFit fit;
  fit.operandsAndBand = fitsPadded || fitsPlain;
  // The whole tile is the bar because the fused drain has no banded arm and
  // the overlay's max at one pitch, because C shares the pool's bytes with
  // operands staged at that same pitch. Asked at both pitches.
  const auto wholeCBytes = [&](bool pad) {
    const Bytes cWhole(
        f.cCostsPoolNothing() || mayRename
            ? 0
            : stagedTileBytes(f.M, fragAlignedExtent(f.N), kAccBytes, pad)
                  .count());
    return maxBytes(planStageBytes(f, pad).ab(), cWhole);
  };
  const auto wholeCFits = [&](bool pad) { return wholeCBytes(pad) <= budget; };
  fit.wholeC = wholeCFits(true) || wholeCFits(false);
  p.fit = fit;
  p.kind = selectKind(f, fit);

  if ((p.kind == Plan::Kind::Direct || p.kind == Plan::Kind::Fused) &&
      mayRename) {
    p.cover = chosen.cover;
    p.readback = std::move(chosen.readback);
    f.cRename = p.facts.cRename = true;
  }

  // 3. The strategy's own parameters.
  switch (p.kind) {
  case Plan::Kind::Scalar: {
    ScalarParams sp;
    sp.intAcc = f.intAcc;
    // The pad, unless only the plain pitch fits.
    if (p.stage.ab() > budget) {
      const StageBytes plain = planStageBytes(f, /*pad=*/false);
      if (plain.ab() <= budget) {
        sp.stagePad = false;
        p.stage = plain;
      }
    }
    p.params = sp;
    break;
  }

  case Plan::Kind::Panel: {
    // Panels address the pool directly: the panel is the staging.
    p.stage = StageBytes{};
    PanelParams pp;
    pp.panel = planPanel(f.M, f.N, f.K, f.aElemBytes, kAccBytes, budget,
                         /*aStaged=*/!f.aDirect);
    pp.panelsM = (f.M + pp.panel.mp - 1) / pp.panel.mp;
    pp.panelsN = (f.N + pp.panel.np - 1) / pp.panel.np;
    pp.panelsK = (f.K + pp.panel.kp - 1) / pp.panel.kp;
    pp.panel.renameWarps = panelRenameWarps(f, pp.panel);
    if (pp.panel.renameWarps.set())
      f.cRename = p.facts.cRename = true;
    p.params = pp;
    break;
  }

  case Plan::Kind::Fused: {
    FusedParams fp;
    fp.fragsPerWarp = fragsPerWarpFor(f);
    fp.cDirect = f.cDirect;
    if (!fusedPadWorthCarrying(wholeCBytes(true), wholeCBytes(false), budget)) {
      fp.stagePad = false;
      p.stage = planStageBytes(f, false);
    }
    p.params = fp;
    break;
  }

  case Plan::Kind::Direct: {
    DirectParams dp;
    dp.fragsPerWarp = fragsPerWarpFor(f);

    // Keep the pad if the padded pitch fits whole. Drop it if unpadded fits
    // whole where padded bands, or if the padded pitch does not fit at all,
    // which would put the reservation over the budget it was admitted under.
    // A shape that bands at both pitches keeps the pad.
    const Bytes cPadded(
        stagedTileBytes(f.M, fragAlignedExtent(f.N), kAccBytes, true));
    if (p.stage.ab() + cPadded > budget) {
      const StageBytes plain = planStageBytes(f, /*pad=*/false);
      const Bytes cPlain(
          stagedTileBytes(f.M, fragAlignedExtent(f.N), kAccBytes, false));
      if (plain.ab() + cPlain <= budget || !fitsPadded) {
        dp.stagePad = false;
        p.stage = plain;
      }
    }
    p.params = dp;
    break;
  }

  case Plan::Kind::Unsupported:
    break;
  }

  // 4. What the pool must hold.
  p.pool = planPool(f, p.stage, p.params, budget);

  // 5. Fields that need the pool fixed first. Only Direct has any.
  std::visit(PoolDependent{f, p.pool}, p.params);
  if (p.storesCDirect())
    p.edgeScratch = edgeScratchFor(f, p.stage.ab(), budget);
  return p;
}

// The whole table's verdict, for MSL_DOT_PLAN_DEBUG. Takes facts and fit,
// since the chosen kind alone cannot say which of two Panel rows fired.
inline std::string dotPlanReport(const DotFacts &f, const DotFit &fit) {
  std::string out;
  // Element widths belong in the shape: two dots of one shape can differ only
  // by operand width and reach opposite strategies.
  out += std::to_string(f.M) + "x" + std::to_string(f.N) + "x" +
         std::to_string(f.K);
  out +=
      " a" + std::to_string(f.aElemBytes) + "b" + std::to_string(f.bElemBytes);
  // C's width only when it was read; otherwise the field holds its f32
  // default, which printed bare would read as a measurement.
  out += "c" + (f.cDirect ? std::to_string(f.cElemBytes) : std::string("?"));
  out += " -> ";
  out += ruleFiredFor(f, fit);

  out += " [fusedAcc=" + std::string(f.fusedAcc ? "y" : "n");
  out += " wholeC=" + std::string(fit.wholeC ? "y" : "n");
  out += " opsAndBand=" + std::string(fit.operandsAndBand ? "y" : "n");
  out += " batched=" + std::string(f.batched() ? "y" : "n");
  out += " intAcc=" + std::string(f.intAcc ? "y" : "n");
  out += " carriedAcc=" + std::string(f.carriedAcc ? "y" : "n");
  out += " cDirect=" + std::string(f.cDirect ? "y" : "n") + "]";
  return out;
}

} // namespace agpu

#endif // AGPU_DOT_PLAN_H
