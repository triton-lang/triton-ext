// A padded shared encoding splices extra elements into the linear offset. The
// map must stay injective and monotonic: two elements sharing an offset would
// alias in threadgroup memory, and a reservation sized from the unpadded
// corner would be short.
#include "agpu/core/TileView.h"
#include "harness.h"

#include <algorithm>
#include <set>
#include <vector>

using agpu::Padding;
using agpu::TileView;

namespace {

TileView padded(int64_t rows, int64_t cols, std::vector<Padding::Rule> rules) {
  TileView v = TileView::rowMajor({rows, cols});
  Padding p;
  p.rules = std::move(rules);
  v.setPadding(std::move(p));
  return v;
}

const std::vector<std::vector<Padding::Rule>> &encodings() {
  static const std::vector<std::vector<Padding::Rule>> all{
      {{16, 4}}, {{32, 8}}, {{64, 4}, {128, 8}}};
  return all;
}

} // namespace

int main() {
  CASE("no rules leaves the offset alone");
  {
    TileView plain = TileView::rowMajor({8, 16});
    TileView p = padded(8, 16, {});
    for (int64_t r = 0; r < 8; ++r)
      for (int64_t c = 0; c < 16; ++c)
        CHECK_EQ(p.offsetOf({r, c}), plain.offsetOf({r, c}));
  }

  CASE("one rule inserts pad elements every interval");
  {
    TileView v = padded(4, 16, {{16, 4}});
    CHECK_EQ(v.offsetOf({0, 0}), 0);
    CHECK_EQ(v.offsetOf({0, 15}), 15);
    CHECK_EQ(v.offsetOf({1, 0}), 16 + 4);
    CHECK_EQ(v.offsetOf({1, 15}), 31 + 4);
    CHECK_EQ(v.offsetOf({2, 0}), 32 + 8);
  }

  CASE("rules compose, each measured on the unpadded offset");
  {
    // At offset 128, [64:+4] contributes 2*4 and [128:+8] contributes 1*8.
    // Measuring the second rule on the already-padded offset would
    // double-count.
    TileView v = padded(8, 64, {{64, 4}, {128, 8}});
    CHECK_EQ(v.offsetOf({0, 0}), 0);
    CHECK_EQ(v.offsetOf({1, 0}), 64 + 4);
    CHECK_EQ(v.offsetOf({2, 0}), 128 + 8 + 8);
    CHECK_EQ(v.offsetOf({3, 0}), 192 + 12 + 8);
  }

  CASE("the map is injective and increasing");
  {
    for (const std::vector<Padding::Rule> &rules : encodings()) {
      TileView v = padded(16, 32, rules);
      std::set<int64_t> seen;
      int64_t prev = -1;
      for (int64_t r = 0; r < 16; ++r)
        for (int64_t c = 0; c < 32; ++c) {
          const int64_t off = v.offsetOf({r, c});
          CHECK(seen.insert(off).second);
          CHECK(off > prev);
          prev = off;
        }
    }
  }

  CASE("the reservation covers the padded corner");
  {
    for (const std::vector<Padding::Rule> &rules : encodings()) {
      TileView v = padded(16, 32, rules);
      int64_t hi = 0;
      for (int64_t r = 0; r < 16; ++r)
        for (int64_t c = 0; c < 32; ++c)
          hi = std::max(hi, v.offsetOf({r, c}));
      CHECK(v.cosizeElems() > hi);
    }
  }

  return ::agpu_test::report("SharedPadding");
}
