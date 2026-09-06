// The table consumes only ops for which emitting nothing is free. Everything
// else keeps the dispatcher looking.
#include "agpu/plan/Vestigial.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("an op that already ran its course emits nothing and loses nothing");
  {
    CHECK(isVestigial("scf.yield"));
    CHECK(vestigialDecision("scf.yield").ok());

    CHECK(vestigialDecision("llvm.intr.assume").ok());
    CHECK(vestigialDecision("llvm.assume").keepLooking());
    CHECK(vestigialDecision("ttg.local_dealloc").ok());
    CHECK(vestigialDecision("scf.condition").ok());
  }

  CASE("an op outside the table keeps the dispatcher looking");
  {
    CHECK(!isVestigial("tt.dot"));
    CHECK(vestigialDecision("tt.dot").keepLooking());
    CHECK(!vestigialDecision("tt.dot").ok());
  }

  CASE("the ops that grew a real lowering are no longer consumed here");
  {
    CHECK(!isVestigial("tt.print"));
    CHECK(vestigialDecision("tt.print").keepLooking());

    CHECK(!isVestigial("tt.assert"));
    CHECK(vestigialDecision("tt.assert").keepLooking());
  }

  CASE("every row is free to drop -- no row declines");
  {
    for (const std::string_view op : kVestigial) {
      const Decision d = vestigialDecision(op);
      CHECK(d.ok());
      CHECK(!d.isDecline());
      CHECK(!d.isBug());
      CHECK(!d.keepLooking());
    }
    CHECK(vestigialCount() > 0);
  }

  return ::agpu_test::report("Vestigial");
}
