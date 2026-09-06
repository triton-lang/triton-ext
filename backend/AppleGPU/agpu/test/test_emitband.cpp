// The banded threadgroup round-trip.
#include "agpu/emit/EmitBand.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <set>

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

BandReg at(msl::Context &c, int reg, int64_t off) {
  return BandReg{reg, c.lit(off), CoordRange{0, off, off}};
}

BandIO simpleIO(msl::Context &c, int n, int64_t stride, bool permute = false) {
  BandIO io;
  for (int i = 0; i < n; ++i) {
    const int dstSlot = permute ? (n - 1 - i) : i;
    io.src.push_back(at(c, i, i * stride));
    io.srcValues.push_back(c.var("v" + std::to_string(i)));
    io.dst.push_back(at(c, i, dstSlot * stride));
    io.dstNames.push_back("o" + std::to_string(i));
  }
  return io;
}

} // namespace

int main() {
  BandNames nm;

  CASE("a register wholly inside a band needs no test");
  {
    CoordGuard g = bandGuard(CoordRange{0, 10, 20}, BandPlan::Band{0, 64});
    CHECK(g.isUnguarded());
  }

  CASE("a register that cannot reach a band is dead");
  {
    CoordGuard g = bandGuard(CoordRange{0, 100, 120}, BandPlan::Band{0, 64});
    CHECK(g.isDead());
  }

  CASE("a straddling register emits exactly the term it needs");
  {
    // Reaches below the band only: one `>= lo` term, no upper test.
    CoordGuard lower =
        bandGuard(CoordRange{0, 50, 70}, BandPlan::Band{64, 128});
    CHECK(lower.needsTest());
    CHECK_EQ(lower.terms().size(), 1u);
    CHECK(lower.terms()[0].op == GuardTerm::Op::Ge);
    CHECK_EQ(lower.terms()[0].bound, 64);

    // Reaches above only: one `< hi` term.
    CoordGuard upper =
        bandGuard(CoordRange{0, 70, 200}, BandPlan::Band{64, 128});
    CHECK(upper.needsTest());
    CHECK_EQ(upper.terms().size(), 1u);
    CHECK(upper.terms()[0].op == GuardTerm::Op::Lt);

    // Spans the whole band: both.
    CoordGuard both = bandGuard(CoordRange{0, 0, 200}, BandPlan::Band{64, 128});
    CHECK_EQ(both.terms().size(), 2u);
  }

  CASE("a tile that fits emits direct assignments with no guard");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(4, 4, Capacity(Bytes(32768), Bytes(0)));
    CHECK(!p.banded());
    emitBandRoundTrip(c, body, p, simpleIO(c, 4, 1, /*permute=*/true), nm);
    const std::string out = render(body);

    CHECK_EQ(countOf(out, "if ("), 0);
    CHECK_EQ(countOf(out, "int f"), 0);
    CHECK(out.find("sc[0] = v0;") != std::string::npos);
    CHECK(out.find("sc[3] = v3;") != std::string::npos);
    CHECK(out.find("o0 = sc[3];") != std::string::npos);
    CHECK(out.find("o3 = sc[0];") != std::string::npos);
  }

  CASE("a fitting tile still barriers exactly twice");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(4, 4, Capacity(Bytes(32768), Bytes(0)));
    emitBandRoundTrip(c, body, p, simpleIO(c, 4, 1, /*permute=*/true), nm);
    // Before the scatter, between the halves and after the gather. Whatever
    // follows the round trip may write the buffer the gather is still reading.
    CHECK_EQ(countOf(render(body), "threadgroup_barrier"), 3);
  }

  CASE("a banded round trip barriers twice per band");
  {
    msl::Context c;
    msl::Block body;
    // 8 elements of 4 B in a 16 B cap: 2 bands of 4.
    BandPlan p = planBand(8, 4, Capacity(Bytes(16), Bytes(0)));
    CHECK_EQ(p.bandCount(), 2);
    emitBandRoundTrip(c, body, p, simpleIO(c, 8, 1, /*permute=*/true), nm);
    CHECK_EQ(countOf(render(body), "threadgroup_barrier"), 2 * 2 + 1);
  }

  CASE("a register appears only in the band it lands in");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(8, 4, Capacity(Bytes(16), Bytes(0)));
    emitBandRoundTrip(c, body, p, simpleIO(c, 8, 1, /*permute=*/true), nm);
    const std::string out = render(body);
    for (int i = 0; i < 8; ++i)
      CHECK_EQ(countOf(out, "v" + std::to_string(i) + ";"), 1);
  }

  CASE("a fixed-offset register in a band needs no test either");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(8, 4, Capacity(Bytes(16), Bytes(0)));
    emitBandRoundTrip(c, body, p, simpleIO(c, 8, 1, /*permute=*/true), nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if ("), 0);
    CHECK_EQ(countOf(out, "int f"), 0);
  }

  CASE("the second band's index is band-relative");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(8, 4, Capacity(Bytes(16), Bytes(0)));
    emitBandRoundTrip(c, body, p, simpleIO(c, 8, 1, /*permute=*/true), nm);
    const std::string out = render(body);
    // Register 4 sits at flat offset 4, which is slot 0 of band 1.
    CHECK(out.find("sc[0] = v4;") != std::string::npos);
    CHECK(out.find("sc[3] = v7;") != std::string::npos);
  }

  CASE("a spanning register is emitted under a test, once per band");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(8, 4, Capacity(Bytes(16), Bytes(0)));

    // Source and destination read different runtime slots, so this isn't
    // elided as an identity round trip.
    BandIO io;
    io.src.push_back(BandReg{0, c.var("idx"), CoordRange{0, 0, 7}});
    io.srcValues.push_back(c.var("v0"));
    io.dst.push_back(BandReg{0, c.var("jdx"), CoordRange{0, 0, 7}});
    io.dstNames.push_back("o0");

    emitBandRoundTrip(c, body, p, io, nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "int f = idx;"), 2);
    CHECK_EQ(countOf(out, "int f = jdx;"), 2);
    CHECK(out.find("f < 4") != std::string::npos);
    CHECK(out.find("f >= 4") != std::string::npos);
  }

  CASE("the scatter and gather halves agree on which band a register touches");
  {
    const CoordRange r{0, 3, 9};
    CHECK(bandGuard(r, BandPlan::Band{0, 16}).isUnguarded());
    CHECK(bandGuard(r, BandPlan::Band{4, 8}).kind() ==
          CoordGuard::Kind::Needed);
    CHECK(bandGuard(r, BandPlan::Band{16, 20}).isDead());
  }

  CASE("scatter writes the buffer and gather reads it");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(2, 4, Capacity(Bytes(32768), Bytes(0)));
    emitBandRoundTrip(c, body, p, simpleIO(c, 2, 1, /*permute=*/true), nm);
    const std::string out = render(body);
    CHECK(out.find("sc[0] = v0;") != std::string::npos);
    CHECK(out.find("o0 = sc[1];") != std::string::npos);
    CHECK(out.find("sc[0] = v0;") < out.find("o0 = sc[1];"));
  }

  CASE("a dead register emits nothing in either direction");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(8, 4, Capacity(Bytes(16), Bytes(0)));

    BandIO io;
    // Lives only in band 0, so band 1 must emit nothing for it.
    io.src.push_back(at(c, 0, 1));
    io.srcValues.push_back(c.var("v0"));
    io.dst.push_back(at(c, 0, 2));
    io.dstNames.push_back("o0");

    emitBandRoundTrip(c, body, p, io, nm);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "v0"), 1);
    CHECK_EQ(countOf(out, "o0"), 1);

    // Two per band plus the trailing one. Collapsing one band's gather
    // barrier with the next band's scatter barrier would remove the
    // separation between accesses to the same buffer.
    CHECK_EQ(countOf(out, "threadgroup_barrier"), 2 * 2 + 1);
  }

  CASE("expressions compare by structure");
  {
    msl::Context c;
    CHECK(msl::exprsEqual(c.var("x"), c.var("x")));
    CHECK(!msl::exprsEqual(c.var("x"), c.var("y")));
    CHECK(msl::exprsEqual(c.lit(8), c.lit(8)));
    CHECK(!msl::exprsEqual(c.lit(8), c.lit(9)));

    msl::Expr *a = c.binary(msl::BinOp::Add, c.var("i"), c.lit(4));
    msl::Expr *b = c.binary(msl::BinOp::Add, c.var("i"), c.lit(4));
    CHECK(msl::exprsEqual(a, b));

    msl::Expr *sub = c.binary(msl::BinOp::Sub, c.var("i"), c.lit(4));
    CHECK(!msl::exprsEqual(a, sub));

    msl::Expr *swapped = c.binary(msl::BinOp::Add, c.lit(4), c.var("i"));
    CHECK(!msl::exprsEqual(a, swapped));
  }

  CASE("folding makes the identity test fire where text comparison cannot");
  {
    msl::Context c;
    msl::Expr *folded = c.binary(msl::BinOp::Add, c.lit(4), c.lit(4));
    CHECK(msl::exprsEqual(folded, c.lit(8)));
  }

  CASE("a call compares its callee along with its arguments");
  {
    msl::Context c;
    msl::Expr *f = c.call("foo", {c.var("x")});
    msl::Expr *g = c.call("bar", {c.var("x")});
    CHECK(!msl::exprsEqual(f, g));
    CHECK(msl::exprsEqual(f, c.call("foo", {c.var("x")})));
    CHECK(!msl::exprsEqual(f, c.call("foo", {c.var("x"), c.var("y")})));
  }

  CASE("a literal's type is part of its identity");
  {
    msl::Context c;
    CHECK(!msl::exprsEqual(c.lit(7, msl::Context::u32()),
                           c.lit(7, msl::Context::i32())));
    CHECK(msl::exprsEqual(c.lit(7, msl::Context::u32()),
                          c.lit(7, msl::Context::u32())));
  }

  CASE("a null expression equals only another null");
  {
    msl::Context c;
    CHECK(msl::exprsEqual(nullptr, nullptr));
    CHECK(!msl::exprsEqual(nullptr, c.var("x")));
    CHECK(!msl::exprsEqual(c.var("x"), nullptr));
  }

  CASE("a round trip whose registers land where they started is elided");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(4, 4, Capacity(Bytes(32768), Bytes(0)));
    BandIO io = simpleIO(c, 4, 1);

    CHECK(roundTripIsIdentity(io));
    CHECK(emitBandRoundTrip(c, body, p, io, nm) == RoundTrip::Elided);
    CHECK_EQ(body.size(), 0u);
  }

  CASE("a permuting round trip is emitted");
  {
    msl::Context c;
    msl::Block body;
    BandPlan p = planBand(4, 4, Capacity(Bytes(32768), Bytes(0)));

    BandIO io;
    for (int i = 0; i < 4; ++i) {
      io.src.push_back(at(c, i, i));
      io.srcValues.push_back(c.var("v" + std::to_string(i)));
      io.dst.push_back(at(c, i, 3 - i));
      io.dstNames.push_back("o" + std::to_string(i));
    }
    CHECK(!roundTripIsIdentity(io));
    CHECK(emitBandRoundTrip(c, body, p, io, nm) == RoundTrip::Emitted);
    CHECK(body.size() > 0u);
  }

  CASE("the identity test is asked ahead of banding");
  {
    msl::Context c;
    for (Capacity cap : {Capacity(Bytes(32768), Bytes(0)), // fits
                         Capacity(Bytes(16), Bytes(0))}) { // bands
      msl::Block body;
      BandPlan p = planBand(8, 4, cap);
      CHECK(emitBandRoundTrip(c, body, p, simpleIO(c, 8, 1), nm) ==
            RoundTrip::Elided);
      CHECK_EQ(body.size(), 0u);
    }
  }

  CASE("the scatter election guards writes and never a barrier");
  {
    // A barrier inside a guard some threads fail is a hang, so barriers stay
    // outside the election. The gather is unguarded: every thread needs its
    // values back.
    msl::Context c;
    msl::Block body;
    BandIO io = simpleIO(c, 4, 1, /*permute=*/true);
    io.scatterGuard = c.binary(msl::BinOp::Eq, c.var("lane"), c.lit(0));
    BandPlan p = planBand(4, 4, Capacity(Bytes(32768), Bytes(0)));
    CHECK(emitBandRoundTrip(c, body, p, io, nm) == RoundTrip::Emitted);
    const std::string out = render(body);
    CHECK_EQ(countOf(out, "if (lane == 0)"), 1);
    CHECK(out.find("if (lane == 0)") < out.find("sc[3] = v0"));
    CHECK(out.find("threadgroup_barrier") < out.find("if (lane == 0)"));
    CHECK_EQ(countOf(out, "o0 = sc["), 1); // gather unguarded, per name
    CHECK(out.find("o0 = sc[") > out.rfind("if (lane == 0)"));
  }

  return ::agpu_test::report("EmitBand");
}
