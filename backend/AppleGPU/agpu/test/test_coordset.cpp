// CoordSet: can these two index expressions ever name the same address?
#include "agpu/core/CoordSet.h"
#include "harness.h"

#include <set>
#include <vector>

using namespace agpu;

namespace {

std::set<int32_t> reach(const CoordSet &s) {
  std::set<int32_t> out;
  if (!s.valid)
    return out;
  for (int32_t a = 0; a < 256; ++a)
    if (s.contains(a))
      out.insert(a);
  return out;
}

bool overlaps(const std::set<int32_t> &a, const std::set<int32_t> &b) {
  for (int32_t v : a)
    if (b.count(v))
      return true;
  return false;
}

std::vector<CoordSet> corpus() {
  std::vector<CoordSet> out;
  for (int32_t base = 0; base < 16; ++base)
    for (int32_t free : {0, 1, 3, 5, 7, 8, 12})
      out.push_back(CoordSet{base, free, true});
  return out;
}

} // namespace

int main() {
  CASE("provablyDisjoint never claims disjoint when the sets overlap");
  {
    const std::vector<CoordSet> all = corpus();
    int disjointCalls = 0;
    for (const CoordSet &a : all)
      for (const CoordSet &b : all) {
        if (!provablyDisjoint(a, b))
          continue;
        ++disjointCalls;
        CHECK(!overlaps(reach(a), reach(b)));
      }
    // Not vacuously safe by always saying false.
    CHECK(disjointCalls > 100);
  }

  CASE("contains agrees with the base-and-mask definition");
  {
    const CoordSet s{64, 7, true};
    CHECK_EQ((int)s.size(), 8);
    const std::set<int32_t> r = reach(s);
    CHECK_EQ((int)r.size(), 8);
    for (int32_t a = 64; a < 72; ++a)
      CHECK(s.contains(a));
    CHECK(!s.contains(63));
    CHECK(!s.contains(72));
  }

  CASE("an unknown set may alias anything");
  {
    const CoordSet unknown = unknownCoords();
    CHECK(!provablyDisjoint(unknown, exactCoord(4)));
    CHECK(!provablyDisjoint(exactCoord(4), unknown));
    CHECK(!provablyDisjoint(unknown, unknown));
    CHECK(!provablyDisjoint(unknown, CoordSet{1024, 0, true}));
    CHECK_EQ((int)unknown.size(), 0);
  }

  CASE("two exact addresses are disjoint exactly when they differ");
  {
    CHECK(provablyDisjoint(exactCoord(4), exactCoord(5)));
    CHECK(!provablyDisjoint(exactCoord(4), exactCoord(4)));
  }

  CASE("a free bit defeats disjointness on that bit alone");
  {
    CHECK(provablyDisjoint(CoordSet{4, 0, true}, CoordSet{5, 0, true}));
    CHECK(!provablyDisjoint(CoordSet{4, 1, true}, CoordSet{5, 0, true}));
    CHECK(provablyDisjoint(CoordSet{4, 1, true}, CoordSet{64, 1, true}));
  }

  CASE("same is stronger than not-disjoint");
  {
    const CoordSet a{4, 3, true}, b{4, 1, true};
    CHECK(!provablyDisjoint(a, b));
    CHECK(!provablySame(a, b));
    CHECK(provablySame(exactCoord(4), exactCoord(4)));
    CHECK(!provablySame(exactCoord(4), exactCoord(5)));
    // A set with free bits is several addresses.
    CHECK(!provablySame(CoordSet{4, 1, true}, CoordSet{4, 1, true}));
  }

  CASE("an offset that overlaps the free mask invalidates the set");
  {
    const CoordSet s{4, 1, true};
    CHECK(!offsetBy(s, 1).valid);
    CHECK(!provablyDisjoint(offsetBy(s, 1), exactCoord(99)));

    const CoordSet moved = offsetBy(s, 64);
    CHECK(moved.valid);
    CHECK_EQ((int)reach(moved).size(), 2);
    CHECK(moved.contains(68));
    CHECK(moved.contains(69));
    CHECK(!offsetBy(unknownCoords(), 4).valid);
  }

  CASE("a union contains both, over every pair in the corpus");
  {
    const std::vector<CoordSet> all = corpus();
    for (const CoordSet &a : all)
      for (const CoordSet &b : all) {
        const CoordSet u = unionOf(a, b);
        CHECK(u.valid);
        for (int32_t v : reach(a))
          CHECK(u.contains(v));
        for (int32_t v : reach(b))
          CHECK(u.contains(v));
      }
  }

  CASE("a union with an unknown set is unknown");
  {
    CHECK(!unionOf(exactCoord(4), unknownCoords()).valid);
    CHECK(!unionOf(unknownCoords(), exactCoord(4)).valid);
  }

  return ::agpu_test::report("CoordSet");
}
