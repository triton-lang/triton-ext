#include "agpu/cost/Occupancy.h"
#include "harness.h"

using namespace agpu::cost;

int main() {
  CASE("the register file reproduces every measured pipeline width");
  {
    // Kernels compiled on M1 Pro whose width Metal chose (none pinned): the
    // highest register their disassembly touches, plus one, and
    // maxTotalThreadsPerThreadgroup.
    const struct {
      int64_t regs, maxThreads;
    } measured[] = {
        {66, 768},  {81, 576},  {82, 576},  {83, 576},  {109, 448}, {113, 448},
        {114, 448}, {120, 384}, {121, 384}, {125, 384}, {128, 384},
    };
    for (const auto &m : measured)
      CHECK_EQ(threadsForRegs(m.regs), m.maxThreads);
  }

  CASE("a full-register kernel is the widest launch always admitted");
  {
    CHECK_EQ(kAlwaysAdmittedThreads, 384);
    CHECK_EQ(threadsForRegs(kMaxRegsPerThread), kAlwaysAdmittedThreads);
  }

  CASE("threadgroup memory is a step function of the declared pool");
  {
    CHECK_EQ(tgResidency(15360), 4);
    CHECK_EQ(tgResidency(20480), 3);
    CHECK_EQ(tgResidency(20736), 2);
    CHECK_EQ(tgResidency(7680), 8);
  }

  CASE("registers cap residency below what threadgroup memory allows");
  {
    CHECK_EQ(certainResidency(4096, 64), 6);
    CHECK_EQ(certainResidency(4096, 128), 3);
    CHECK_EQ(certainResidency(4096, 256), 1);
    CHECK_EQ(certainResidency(4096, 512), 1);
    CHECK_EQ(certainResidency(32768, 128), 1);
    CHECK_EQ(certainResidency(20480, 128), 3);
  }

  CASE("the typical kernel stops threadgroup memory binding at six");
  {
    CHECK_EQ(regResidency(RegisterBand{}.typical, 128), 6);
  }

  CASE("a smaller pool gains where some register count in the band gains");
  {
    CHECK(gainsResidency(32768, 20480, 128));
    CHECK(gainsResidency(12288, 9216, 128));
    CHECK(!gainsResidency(8192, 6144, 128));
    CHECK(!gainsResidency(20480, 20480, 128));
  }

  CASE("a larger pool loses only where the band says it does");
  {
    CHECK(!losesResidency(4000, 4336, 128));
    CHECK(losesResidency(20480, 20736, 128));
    CHECK(losesResidency(8192, 9216, 32));
  }

  CASE("a certain gain holds at every register count");
  {
    CHECK(certainlyGainsResidency(32768, 17408, 128));
    CHECK(!certainlyGainsResidency(12288, 9216, 128));
    CHECK(!certainlyGainsResidency(32768, 16384, 256));
  }

  return ::agpu_test::report("Occupancy");
}
