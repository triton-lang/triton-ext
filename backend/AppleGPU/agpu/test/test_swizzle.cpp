// A swizzled view permutes offsets inside a row. The permutation must be a
// bijection: two logical elements sharing an offset would alias in threadgroup
// memory and one write would destroy the other.
#include "agpu/core/TileView.h"
#include "harness.h"

#include <algorithm>
#include <array>
#include <set>
#include <vector>

using agpu::Swizzle;
using agpu::TileView;
using Coord = TileView::Coord;

namespace {

TileView swizzled(int64_t rows, int64_t cols, int64_t vec, int64_t perPhase,
                  int64_t maxPhase) {
  Swizzle sw;
  sw.vec = vec;
  sw.perPhase = perPhase;
  sw.maxPhase = maxPhase;
  sw.phaseDim = 0;
  sw.groupDim = 1;
  sw.groupExtent = cols;
  return TileView({rows, cols}, {cols, 1}, sw);
}

// A swizzle reaching over `span` elements, repeated across a wider row.
TileView spanned(int64_t rows, int64_t cols, int64_t span, int64_t vec,
                 int64_t perPhase, int64_t maxPhase) {
  Swizzle sw;
  sw.vec = vec;
  sw.perPhase = perPhase;
  sw.maxPhase = maxPhase;
  sw.phaseDim = 0;
  sw.groupDim = 1;
  sw.groupExtent = span;
  return TileView({rows, cols}, {cols, 1}, sw);
}

// Every element of the tile lands on its own offset, inside the tile's bounds.
void checkBijective(const TileView &v) {
  std::set<int64_t> seen;
  const int64_t rows = v.extentAt(0), cols = v.extentAt(1);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t off = v.offsetOf({r, c});
      CHECK(off >= 0 && off < rows * cols);
      CHECK(seen.insert(off).second);
    }
  CHECK_EQ((int64_t)seen.size(), rows * cols);
}

} // namespace

int main() {
  CASE("an identity swizzle addresses like row-major");
  {
    TileView plain = TileView::rowMajor({8, 16});
    TileView sw = swizzled(8, 16, 1, 1, 1);
    for (int64_t r = 0; r < 8; ++r)
      for (int64_t c = 0; c < 16; ++c)
        CHECK_EQ(sw.offsetOf({r, c}), plain.offsetOf({r, c}));
  }

  CASE("maxPhase 1 permutes nothing whatever vec says");
  {
    TileView plain = TileView::rowMajor({8, 32});
    TileView sw = swizzled(8, 32, 8, 1, 1);
    for (int64_t r = 0; r < 8; ++r)
      for (int64_t c = 0; c < 32; ++c)
        CHECK_EQ(sw.offsetOf({r, c}), plain.offsetOf({r, c}));
  }

  CASE("row 0 is never permuted: its phase is 0");
  {
    TileView sw = swizzled(8, 32, 4, 2, 4);
    for (int64_t c = 0; c < 32; ++c)
      CHECK_EQ(sw.offsetOf({0, c}), c);
  }

  CASE("the swizzles the gluon suite exercises are bijections");
  {
    checkBijective(swizzled(16, 64, 4, 2, 4));
    checkBijective(swizzled(16, 64, 2, 2, 4));
    checkBijective(swizzled(16, 64, 8, 1, 8));
    checkBijective(swizzled(16, 64, 16, 1, 16));
    checkBijective(swizzled(8, 32, 8, 2, 4));
  }

  CASE("the reservation covers every permuted offset");
  {
    // A 16x32 tile permuted by vec=8,maxPhase=8: the corner is no longer the
    // highest offset, so sizing from it would under-reserve and the tail would
    // write past the buffer.
    for (auto [rows, cols, vec, pp, mp] :
         std::vector<std::array<int64_t, 5>>{{16, 32, 8, 1, 8},
                                             {16, 32, 4, 2, 4},
                                             {16, 32, 16, 1, 16},
                                             {16, 64, 4, 2, 4},
                                             {8, 32, 8, 2, 4}}) {
      TileView v = swizzled(rows, cols, vec, pp, mp);
      int64_t hi = 0;
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
          hi = std::max(hi, v.offsetOf({r, c}));
      CHECK(v.cosizeElems() > hi);
    }
  }

  CASE("a row narrower than one vec group is not permuted");
  {
    // 64x1 with vec=4: a single column holds no whole group, so every phase
    // must collapse to zero or the offset leaves the row.
    for (int64_t vec : {2, 4, 8}) {
      TileView v = swizzled(64, 1, vec, 1, 4);
      for (int64_t r = 0; r < 64; ++r)
        CHECK_EQ(v.offsetOf({r, 0}), r);
      checkBijective(v);
    }
  }

  CASE("perPhase rows share one phase");
  {
    TileView sw = swizzled(8, 32, 4, 2, 4);
    for (int64_t c = 0; c < 32; ++c) {
      CHECK_EQ(sw.offsetOf({2, c}) - 2 * 32, sw.offsetOf({3, c}) - 3 * 32);
      CHECK_EQ(sw.offsetOf({4, c}) - 4 * 32, sw.offsetOf({5, c}) - 5 * 32);
    }
  }

  CASE("elements inside one vec group stay contiguous");
  {
    TileView sw = swizzled(16, 64, 4, 2, 4);
    for (int64_t r = 0; r < 16; ++r)
      for (int64_t g = 0; g < 64 / 4; ++g) {
        const int64_t base = sw.offsetOf({r, g * 4});
        for (int64_t k = 1; k < 4; ++k)
          CHECK_EQ(sw.offsetOf({r, g * 4 + k}), base + k);
      }
  }

  CASE("a phase actually moves a group off the diagonal");
  {
    TileView sw = swizzled(8, 32, 4, 1, 4);
    // Row 1 has phase 1, so group 0 lands where group 1 would sit.
    CHECK_EQ(sw.offsetOf({1, 0}), 32 + 4);
    CHECK_EQ(sw.offsetOf({1, 4}), 32 + 0);
  }

  CASE("a swizzle narrower than the row repeats across it");
  {
    // A 64-byte swizzle over f16 spans 32 elements, so a 128-wide row holds
    // four independent tiles of it.
    TileView v = spanned(8, 128, 32, 8, 1, 4);
    checkBijective(v);
    for (int64_t r = 0; r < 8; ++r)
      for (int64_t t = 0; t < 4; ++t)
        for (int64_t c = 0; c < 32; ++c)
          CHECK_EQ(v.offsetOf({r, t * 32 + c}), v.offsetOf({r, c}) + t * 32);
  }

  CASE("a full-width span is the plain swizzle");
  {
    TileView wide = spanned(8, 32, 32, 4, 2, 4);
    TileView plain = swizzled(8, 32, 4, 2, 4);
    for (int64_t r = 0; r < 8; ++r)
      for (int64_t c = 0; c < 32; ++c)
        CHECK_EQ(wide.offsetOf({r, c}), plain.offsetOf({r, c}));
  }

  CASE("a panelled swizzle stores each span-wide slice whole");
  {
    // The NVMMA shape: 128x128 in 32-wide panels, each panel contiguous, with
    // the XOR inside it. Sampled against the offsets the encoding reports.
    Swizzle sw;
    sw.vec = 8;
    sw.perPhase = 2;
    sw.maxPhase = 4;
    sw.phaseDim = 0;
    sw.groupDim = 1;
    sw.groupExtent = 32;
    sw.tileStride = 128 * 32;
    TileView v({128, 128}, {32, 1}, sw);
    CHECK_EQ(v.offsetOf({0, 0}), 0);
    CHECK_EQ(v.offsetOf({0, 32}), 4096);
    CHECK_EQ(v.offsetOf({1, 0}), 32);
    CHECK_EQ(v.offsetOf({2, 0}), 72);
    CHECK_EQ(v.offsetOf({2, 8}), 64);
    CHECK_EQ(v.cosizeElems(), 128 * 128);

    std::set<int64_t> seen;
    for (int64_t r = 0; r < 128; ++r)
      for (int64_t c = 0; c < 128; ++c) {
        const int64_t off = v.offsetOf({r, c});
        CHECK(off >= 0 && off < 128 * 128);
        CHECK(seen.insert(off).second);
      }
  }

  CASE("a window addresses like its swizzled parent");
  {
    TileView parent = swizzled(32, 64, 8, 1, 8);
    const Coord at{8, 16}, ext{8, 16};
    TileView w = parent.window(at, ext);
    for (int64_t r = 0; r < ext[0]; ++r)
      for (int64_t c = 0; c < ext[1]; ++c)
        CHECK_EQ(w.offsetOf({r, c}), parent.offsetOf({r + at[0], c + at[1]}));
  }

  CASE("windows compose");
  {
    TileView parent = swizzled(32, 64, 4, 2, 4);
    TileView a = parent.window({8, 8}, {16, 32});
    TileView b = a.window({4, 8}, {8, 16});
    for (int64_t r = 0; r < 8; ++r)
      for (int64_t c = 0; c < 16; ++c)
        CHECK_EQ(b.offsetOf({r, c}), parent.offsetOf({r + 12, c + 16}));
  }

  return ::agpu_test::report("Swizzle");
}
