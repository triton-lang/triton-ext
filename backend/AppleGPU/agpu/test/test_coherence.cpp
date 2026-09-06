// Which buffers need coherent accesses.
#include "agpu/msl/Printer.h"
#include "agpu/plan/Coherence.h"
#include "harness.h"

#include <sstream>

using namespace agpu;

namespace {

BufferAccess scalar(int buf, AccessKind kind, int depth) {
  BufferAccess a;
  a.buffer = buf;
  a.kind = kind;
  a.loopDepth = depth;
  return a;
}
BufferAccess tile(int buf, AccessKind kind, int depth) {
  BufferAccess a = scalar(buf, kind, depth);
  a.isTensor = true;
  return a;
}

BufferAccess load(int buf, int depth = 0) {
  return scalar(buf, AccessKind::Load, depth);
}
BufferAccess store(int buf, int depth = 0) {
  return scalar(buf, AccessKind::Store, depth);
}
BufferAccess tensorLoad(int buf, int depth = 0) {
  return tile(buf, AccessKind::Load, depth);
}
BufferAccess tensorStore(int buf, int depth = 0) {
  return tile(buf, AccessKind::Store, depth);
}

} // namespace

int main() {
  CASE("a kernel that reads inputs and writes outputs needs no coherence");
  {
    CoherenceFacts f;
    f.accesses = {load(0), load(1), store(2)};
    CoherencePlan p = planCoherence(f);
    CHECK(!p.any());
    CHECK(!p.needsCoherent(0));
    CHECK(!p.needsCoherent(2));
  }

  CASE("a buffer stored and loaded outside any loop needs nothing");
  {
    CoherenceFacts f;
    f.accesses = {store(0), load(0)};
    CHECK(!planCoherence(f).any());
  }

  CASE("a buffer both stored and loaded in one loop is coherent");
  {
    CoherenceFacts f;
    f.accesses = {store(0, /*loopDepth=*/1), load(0, 1)};
    CoherencePlan p = planCoherence(f);
    CHECK(p.needsCoherent(0));
    CHECK_EQ(p.buffers().size(), 1u);
  }

  CASE("a loop that only stores is not coherent");
  {
    CoherenceFacts f;
    f.accesses = {store(0, 1), store(1, 1)};
    CHECK(!planCoherence(f).any());
  }

  CASE("a loop that only loads is not coherent");
  {
    CoherenceFacts f;
    f.accesses = {load(0, 1), load(1, 1)};
    CHECK(!planCoherence(f).any());
  }

  CASE("a store in one loop and a load in another is not coherent");
  {
    CoherenceFacts f;
    f.accesses = {store(0, 1), load(0, 2)};
    CHECK(!planCoherence(f).any());
  }

  CASE("only the buffer that round-trips becomes coherent");
  {
    CoherenceFacts f;
    f.accesses = {store(0, 1), load(0, 1), load(1, 1), store(2, 1)};
    CoherencePlan p = planCoherence(f);
    CHECK(p.needsCoherent(0));
    CHECK(!p.needsCoherent(1));
    CHECK(!p.needsCoherent(2));
  }

  CASE("a device barrier makes every stored-and-loaded buffer coherent");
  {
    CoherenceFacts f;
    f.accesses = {store(0), load(0)};
    f.hasDeviceBarrier = true;
    CHECK(planCoherence(f).needsCoherent(0));
  }

  CASE("the barrier trigger ignores loop depth");
  {
    CoherenceFacts f;
    f.accesses = {store(3, 0), load(3, 0)};
    f.hasDeviceBarrier = true;
    CHECK(planCoherence(f).needsCoherent(3));
  }

  CASE("a barrier does not make a write-only buffer coherent");
  {
    CoherenceFacts f;
    f.accesses = {store(0), load(1)};
    f.hasDeviceBarrier = true;
    CoherencePlan p = planCoherence(f);
    CHECK(!p.needsCoherent(0));
    CHECK(!p.needsCoherent(1));
  }

  CASE("the two triggers are independent");
  {
    CoherenceFacts f;
    f.accesses = {store(0, 1), load(0, 1), store(1), load(1)};
    f.hasDeviceBarrier = true;
    CoherencePlan p = planCoherence(f);
    CHECK(p.needsCoherent(0));
    CHECK(p.needsCoherent(1));
  }

  CASE("a tile stored and loaded in one loop is not coherent");
  {
    CoherenceFacts f;
    f.accesses = {tensorStore(0, 1), tensorLoad(0, 1)};
    CHECK(!planCoherence(f).any());
  }

  CASE("a device barrier makes a stored-and-loaded tile coherent");
  {
    CoherenceFacts f;
    f.accesses = {tensorStore(0), tensorLoad(0)};
    f.hasDeviceBarrier = true;
    CHECK(planCoherence(f).needsCoherent(0));
  }

  CASE("a buffer qualifying twice is listed once");
  {
    CoherenceFacts f;
    f.accesses = {store(0, 1), load(0, 1)};
    f.hasDeviceBarrier = true;
    CHECK_EQ(planCoherence(f).buffers().size(), 1u);
  }

  CASE("a coherent pointer prints its address-space attribute");
  {
    std::ostringstream os;
    msl::Printer p(os);
    p.printType(msl::Type::scalar(msl::Scalar::F32)
                    .pointerTo(msl::AddrSpace::Device, /*coherent=*/true));
    CHECK_EQ(os.str(), std::string("device coherent(device) float *"));
  }

  CASE("a plain pointer prints none");
  {
    std::ostringstream os;
    msl::Printer p(os);
    p.printType(
        msl::Type::scalar(msl::Scalar::F32).pointerTo(msl::AddrSpace::Device));
    CHECK_EQ(os.str(), std::string("device float *"));
  }

  CASE("coherent is part of the pointer's identity");
  {
    const msl::Type plain =
        msl::Type::scalar(msl::Scalar::F32).pointerTo(msl::AddrSpace::Device);
    const msl::Type coh = msl::Type::scalar(msl::Scalar::F32)
                              .pointerTo(msl::AddrSpace::Device, true);
    CHECK(!(plain == coh));
    CHECK(coh == msl::Type::scalar(msl::Scalar::F32)
                     .pointerTo(msl::AddrSpace::Device, true));
  }

  return ::agpu_test::report("Coherence");
}
