// Cases are the shapes EmitMSLDot.cpp produces.
//
// An F prefix on a case names the bug report it was written for, in a tracker
// outside this tree. The sentence after it stands on its own; the tag only
// ties the case back to its report.
#include "agpu/core/TileView.h"
#include "harness.h"

using agpu::TileView;
using Coord = TileView::Coord;

int main() {
  CASE("rowMajor strides and corners");
  {
    TileView v = TileView::rowMajor({64, 32});
    CHECK_EQ(v.stride(), Coord({32, 1}));
    CHECK_EQ(v.offsetOf({0, 0}), 0);
    CHECK_EQ(v.offsetOf({0, 31}), 31);
    CHECK_EQ(v.offsetOf({1, 0}), 32);
    CHECK_EQ(v.offsetOf({63, 31}), 64 * 32 - 1);
    CHECK_EQ(v.cosizeElems(), 64 * 32);
    CHECK_EQ(v.sizeElems(), 64 * 32);
    CHECK(v.isDense());
    CHECK(v.innermostContiguous());
  }

  CASE("fragment (mi,ni) in a row-major N-strided C tile");
  {
    const int64_t M = 64, N = 128, kFrag = 8;
    TileView c = TileView::rowMajor({M, N});
    for (int64_t mi = 0; mi < M / kFrag; ++mi)
      for (int64_t ni = 0; ni < N / kFrag; ++ni)
        CHECK_EQ(c.offsetOf({mi * kFrag, ni * kFrag}),
                 mi * kFrag * N + ni * kFrag);
  }

  CASE("F1: band-relative readback, (mi*8 - r0)*N + ni*8");
  {
    const int64_t M = 64, N = 128, kFrag = 8, r0 = 16, bandRows = 16;
    TileView cFull = TileView::rowMajor({M, N});
    TileView band = cFull.subview({r0, 0}, {bandRows, N});
    for (int64_t mi = r0 / kFrag; mi < (r0 + bandRows) / kFrag; ++mi)
      for (int64_t ni = 0; ni < N / kFrag; ++ni) {
        int64_t got =
            band.offsetOf({mi * kFrag - r0, ni * kFrag}) - band.origin();
        CHECK_EQ(got, (mi * kFrag - r0) * N + ni * kFrag);
      }
  }

  CASE("F1: panel accumulator, mi*8*npCur + ni*8");
  {
    const int64_t mpCur = 32, npCur = 64, kFrag = 8;
    TileView pc = TileView::rowMajor({mpCur, npCur});
    for (int64_t mi = 0; mi < mpCur / kFrag; ++mi)
      for (int64_t ni = 0; ni < npCur / kFrag; ++ni)
        CHECK_EQ(pc.offsetOf({mi * kFrag, ni * kFrag}),
                 mi * kFrag * npCur + ni * kFrag);
  }

  CASE("F1: A fragment, mi*8*kpCur + ki*8");
  {
    const int64_t mpCur = 32, kpCur = 32, kFrag = 8;
    TileView pa = TileView::rowMajor({mpCur, kpCur});
    for (int64_t mi = 0; mi < mpCur / kFrag; ++mi)
      for (int64_t ki = 0; ki < kpCur / kFrag; ++ki)
        CHECK_EQ(pa.offsetOf({mi * kFrag, ki * kFrag}),
                 mi * kFrag * kpCur + ki * kFrag);
  }

  CASE("panel origin absorbs m0/k0");
  {
    const int64_t M = 128, K = 64, mp = 32, kp = 32;
    TileView aFull = TileView::rowMajor({M, K});
    const int64_t m0 = 64, k0 = 32;
    TileView panel = aFull.subview({m0, k0}, {mp, kp});
    CHECK_EQ(panel.origin(), m0 * K + k0);
    CHECK_EQ(panel.offsetOf({0, 0}), m0 * K + k0);
    CHECK_EQ(panel.offsetOf({1, 0}), (m0 + 1) * K + k0);
    CHECK_EQ(panel.stride(), Coord({K, 1}));
    CHECK(!panel.isDense());
  }

  CASE("rank-3 batch slice removes the rank test");
  {
    const int64_t Bd = 4, M = 32, N = 64;
    TileView c3 = TileView::rowMajor({Bd, M, N});
    CHECK_EQ(c3.stride(), Coord({M * N, N, 1}));
    for (int64_t bi = 0; bi < Bd; ++bi) {
      TileView c2 = c3.slice(bi);
      CHECK_EQ(c2.rank(), 2);
      CHECK_EQ(c2.origin(), bi * M * N);
      CHECK_EQ(c2.stride(), Coord({N, 1}));
      CHECK_EQ(c2.offsetOf({3, 5}), bi * M * N + 3 * N + 5);
    }
  }

  CASE("padded rows: stride grows, extent does not");
  {
    const int64_t rows = 16, cols = 32, pad = 4;
    TileView p = TileView::rowMajorPadded({rows, cols}, pad);
    CHECK_EQ(p.extent(), Coord({rows, cols}));
    CHECK_EQ(p.stride(), Coord({cols + pad, 1}));
    CHECK_EQ(p.offsetOf({1, 0}), cols + pad);
    CHECK_EQ(p.sizeElems(), rows * cols);
    CHECK_EQ(p.cosizeElems(), (rows - 1) * (cols + pad) + cols);
    CHECK(!p.isDense());
    CHECK(p.innermostContiguous());
  }

  CASE("ragged final panel is a smaller subview, same strides");
  {
    const int64_t M = 100, N = 128, mp = 32;
    TileView cFull = TileView::rowMajor({M, N});
    const int64_t m0 = 96;
    const int64_t rows = std::min<int64_t>(mp, M - m0);
    CHECK_EQ(rows, 4);
    TileView tail = cFull.subview({m0, 0}, {rows, N});
    CHECK_EQ(tail.extentAt(0), 4);
    CHECK_EQ(tail.offsetOf({0, 0}), m0 * N);
    CHECK_EQ(tail.offsetOf({3, N - 1}), (m0 + 3) * N + N - 1);
    CHECK_EQ(tail.cosizeElems(), (m0 + 3) * N + N);
  }

  CASE("cosizeElems is the reservation, from the same object");
  {
    TileView pool = TileView::rowMajorPadded({16, 32}, 4);
    const int64_t reserved = pool.cosizeElems();
    for (int64_t i = 0; i < pool.extentAt(0); ++i)
      for (int64_t j = 0; j < pool.extentAt(1); ++j)
        CHECK(pool.offsetOf({i, j}) < reserved);
  }

  CASE("exact pool capacity boundary");
  {
    const int64_t budgetElems = 8192;
    TileView t = TileView::rowMajor({64, 128});
    CHECK_EQ(t.sizeElems(), budgetElems);
    CHECK_EQ(t.cosizeElems(), budgetElems);
  }

  CASE("originAt maps a larger tensor's coordinates into the window's buffer");
  {
    TileView tile = TileView::rowMajor({24, 32}).originAt({24, 32});
    CHECK_EQ(tile.offsetOf({24, 32}), 0);
    CHECK_EQ(tile.offsetOf({24, 33}), 1);
    CHECK_EQ(tile.offsetOf({25, 32}), 32);
    CHECK_EQ(tile.strideAt(0), 32);
    CHECK_EQ(tile.strideAt(1), 1);
  }

  CASE("transposed B is represented as a stride swap");
  {
    const int64_t K = 32, N = 64;
    TileView bRow = TileView::rowMajor({K, N});
    TileView bCol({K, N}, {1, K});
    CHECK_EQ(bRow.offsetOf({2, 3}), 2 * N + 3);
    CHECK_EQ(bCol.offsetOf({2, 3}), 2 + 3 * K);
    CHECK(!bCol.innermostContiguous());
    CHECK_EQ(bCol.cosizeElems(), K * N);
  }

  return ::agpu_test::report("TileView");
}
