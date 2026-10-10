// Banding a tile that does not fit the pool.
#include "agpu/plan/BandPlan.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("capacity is the budget minus what is already live");
  {
    Capacity c(Bytes(32768), Bytes(8192));
    CHECK_EQ(c.available().count(), 24576);
  }

  CASE("an over-committed pool's capacity floors at zero");
  {
    Capacity c(Bytes(1024), Bytes(4096));
    CHECK_EQ(c.available().count(), 0);
  }

  CASE("a band is at least one element even with no room");
  {
    BandPlan p = planBand(100, 4, Capacity(Bytes(0), Bytes(0)));
    CHECK_EQ(p.elems(), 1);
    CHECK_EQ(p.bandCount(), 100);
  }

  CASE("a tile that fits is not banded");
  {
    BandPlan p = planBand(1024, 4, Capacity(Bytes(32768), Bytes(0)));
    CHECK(!p.banded());
    CHECK(p.kind() == BandKind::Whole);
    CHECK_EQ(p.elems(), 1024);
    CHECK_EQ(p.bandCount(), 1);
    CHECK_EQ(p.bytes().count(), 4096);
  }

  CASE("a tile that exactly fills the capacity is still whole");
  {
    BandPlan p = planBand(1024, 4, Capacity(Bytes(4096), Bytes(0)));
    CHECK(!p.banded());
    CHECK_EQ(p.bandCount(), 1);
  }

  CASE("one byte over the capacity bands");
  {
    BandPlan p = planBand(1025, 4, Capacity(Bytes(4096), Bytes(0)));
    CHECK(p.banded());
    CHECK_EQ(p.bandCount(), 2);
  }

  CASE("a row width holds every band boundary on a row edge");
  {
    const BandPlan flat = planBand(700, 4, Capacity(Bytes(1024), Bytes(0)));
    CHECK(flat.elems() % 7 != 0);

    const BandPlan rows = planBand(700, 4, Capacity(Bytes(1024), Bytes(0)), 7);
    CHECK_EQ(rows.elems() % 7, 0);
    for (int64_t b = 0; b < rows.bandCount(); ++b) {
      CHECK_EQ(rows.bandAt(b).lo % 7, 0);
      CHECK(rows.bandAt(b).size() <= 256);
    }
    CHECK_EQ(rows.bandAt(rows.bandCount() - 1).hi, 700);
  }

  CASE("a row wider than the capacity keeps the flat split");
  {
    const BandPlan p = planBand(1000, 4, Capacity(Bytes(1024), Bytes(0)), 500);
    CHECK(p.banded());
    CHECK_EQ(p.bandAt(p.bandCount() - 1).hi, 1000);
  }

  CASE("a row width that does not divide the tile keeps the flat split");
  {
    const BandPlan p = planBand(1005, 4, Capacity(Bytes(1024), Bytes(0)), 10);
    CHECK(p.banded());
    CHECK_EQ(p.bandAt(p.bandCount() - 1).hi, 1005);
  }

  CASE("the bytes reserved are the bytes a band occupies");
  {
    const Capacity cap(Bytes(32768), Bytes(20000));
    for (int64_t total : {100, 1000, 4096, 10000, 65536}) {
      BandPlan p = planBand(total, 4, cap);
      CHECK_EQ(p.bytes().count(), p.elems() * 4);
      CHECK(p.bytes() <= cap.available() || p.elems() == 1);
    }
  }

  CASE("a band never exceeds the capacity it was planned against");
  {
    for (int64_t live : {0, 4096, 16384, 30000}) {
      const Capacity cap(Bytes(32768), Bytes(live));
      BandPlan p = planBand(100000, 2, cap);
      CHECK(p.bytes() <= cap.available() || p.elems() == 1);
    }
  }

  CASE("bands cover the tile exactly once");
  {
    for (int64_t total : {7, 64, 1000, 4095, 4096, 4097}) {
      BandPlan p = planBand(total, 4, Capacity(Bytes(2048), Bytes(0)));
      int64_t seen = 0;
      for (int64_t b = 0; b < p.bandCount(); ++b) {
        BandPlan::Band r = p.bandAt(b);
        CHECK_EQ(r.lo, seen);
        CHECK(r.size() > 0);
        seen = r.hi;
      }
      CHECK_EQ(seen, total);
    }
  }

  CASE("bands are split evenly across the capacity");
  {
    BandPlan p = planBand(1000, 4, Capacity(Bytes(2048), Bytes(0)));
    CHECK_EQ(p.bandCount(), 2);
    CHECK_EQ(p.elems(), 500); // not 512 + 488
  }

  CASE("live pool bytes shrink the band");
  {
    const Capacity empty(Bytes(32768), Bytes(0));
    const Capacity busy(Bytes(32768), Bytes(30000));
    CHECK(planBand(65536, 2, busy).elems() < planBand(65536, 2, empty).elems());
  }

  return ::agpu_test::report("BandPlan");
}
