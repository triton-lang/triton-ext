#include "agpu/plan/LaunchPlan.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("a kernel that never waits takes any grid");
  {
    CHECK(residencyFor(LaunchFacts{}) == GridResidency::Independent);
  }

  CASE("a blocking poll waits for a threadgroup it cannot schedule");
  {
    LaunchFacts f;
    f.blockingPoll = true;
    CHECK(residencyFor(f) == GridResidency::CoResident);
  }

  CASE("a mutex takes any grid, though it spins on a device atomic");
  {
    LaunchFacts f;
    f.atomicInLoop = true;
    CHECK(residencyFor(f) == GridResidency::Independent);
  }

  CASE("reading the grid extent alone is not a wait");
  {
    LaunchFacts f;
    f.readsGridExtent = true;
    CHECK(residencyFor(f) == GridResidency::Independent);
  }

  CASE("spinning on a value derived from the grid extent is a wait");
  {
    LaunchFacts f;
    f.atomicInLoop = true;
    f.readsGridExtent = true;
    CHECK(residencyFor(f) == GridResidency::CoResident);
  }

  CASE("the attribute the host reads is spelled once");
  {
    CHECK(std::string(kGridResidencyAttr) == "applegpu.grid_coresident");
  }

  return ::agpu_test::report("LaunchPlan");
}
