// map_elementwise: block arguments count source-major, register names count
// group-major.
#include "agpu/plan/MapPlan.h"
#include "harness.h"

#include <map>
#include <set>
#include <utility>

using namespace agpu;

int main() {
  CASE("the binding is a bijection, over every shape worth trying");
  {
    for (int sources = 1; sources <= 4; ++sources)
      for (int results = 1; results <= 3; ++results)
        for (int pack : {1, 2, 4})
          for (int groups = 1; groups <= 4; ++groups) {
            MapPlan p{{sources, results, groups * pack, pack, false}};
            CHECK(p.usable());
            CHECK(mapDecision(p).ok());
            CHECK_EQ(p.groups(), groups);

            std::map<std::pair<int, int>, std::pair<int, int>> bind;
            std::set<std::pair<int, int>> names;
            for (int g = 0; g < p.groups(); ++g)
              for (int s = 0; s < sources; ++s)
                for (int e = 0; e < pack; ++e) {
                  const std::pair<int, int> slot{g, p.blockArgument(s, e)};
                  const std::pair<int, int> name{s, p.sourceRegister(g, e)};
                  CHECK(bind.emplace(slot, name).second); // injective
                  CHECK(names.insert(name).second);       // and no reuse
                }
            CHECK_EQ((int)bind.size(), groups * p.numBlockArguments());
            CHECK_EQ((int)names.size(), sources * groups * pack);
          }
  }

  CASE("the two orders produce genuinely different maps");
  {
    // With pack 2 and 3 groups the two orders agree at (1,1) only.
    MapPlan p{{2, 1, 6, 2, false}};
    CHECK_EQ(p.blockArgument(1, 1), 3);
    CHECK_EQ(p.sourceRegister(1, 1), 3);
    // Where they part:
    CHECK_EQ(p.blockArgument(1, 0), 2);
    CHECK_EQ(p.sourceRegister(2, 0), 4);
    CHECK_EQ(p.blockArgument(0, 1), 1);
    CHECK_EQ(p.sourceRegister(0, 1), 1);
  }

  CASE("pack 1 makes the two orders coincide, which is why it hides the bug");
  {
    MapPlan p{{3, 2, 5, 1, false}};
    for (int s = 0; s < 3; ++s)
      CHECK_EQ(p.blockArgument(s, 0), s);
    for (int g = 0; g < 5; ++g)
      CHECK_EQ(p.sourceRegister(g, 0), g);
    CHECK_EQ(p.groups(), 5);
  }

  CASE("results interleave the same way");
  {
    MapPlan p{{1, 3, 4, 2, false}};
    CHECK_EQ(p.numResultOperands(), 6);
    std::set<int> seen;
    for (int k = 0; k < 3; ++k)
      for (int e = 0; e < 2; ++e)
        CHECK(seen.insert(p.resultOperand(k, e)).second);
    CHECK_EQ((int)seen.size(), 6);
    CHECK_EQ(p.resultOperand(2, 1), 5);
  }

  CASE("a multi-block region needs one capture per result element");
  {
    MapPlan single{{1, 2, 4, 2, false}};
    CHECK(!single.needsCaptures());
    CHECK_EQ(single.numCaptures(), 0);

    MapPlan multi{{1, 2, 4, 2, true}};
    CHECK(multi.needsCaptures());
    CHECK_EQ(multi.numCaptures(), 4);
    // Per result element: pack 2 and 2 results is 4.
    CHECK_EQ(multi.numCaptures(), multi.numResultOperands());
  }

  CASE("a fractional pack declines");
  {
    Decision d = mapDecision(MapPlan{{1, 1, 5, 2, false}});
    CHECK(d.isDecline());
    CHECK((!MapPlan{{1, 1, 5, 2, false}}.usable()));

    CHECK((mapDecision(MapPlan{{1, 1, 4, 0, false}}).isDecline()));
    CHECK((mapDecision(MapPlan{{0, 1, 4, 2, false}}).isDecline()));
    CHECK((mapDecision(MapPlan{{1, 0, 4, 2, false}}).isDecline()));
    CHECK((mapDecision(MapPlan{{1, 1, 0, 2, false}}).isDecline()));
  }

  CASE("a pack equal to the register count inlines the body once");
  {
    MapPlan p{{2, 1, 8, 8, false}};
    CHECK(mapDecision(p).ok());
    CHECK_EQ(p.groups(), 1);
    CHECK_EQ(p.numBlockArguments(), 16);
  }

  return ::agpu_test::report("MapPlan");
}
