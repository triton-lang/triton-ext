#include "agpu/plan/ScanPlan.h"
#include "harness.h"

using namespace agpu;

namespace {

ScanFacts facts(std::vector<AxisBit> lane, std::vector<AxisBit> warp = {},
                int64_t numWarps = 1, int64_t regs = 1) {
  ScanFacts f;
  f.laneBits = std::move(lane);
  f.warpBits = std::move(warp);
  f.numWarps = numWarps;
  f.regCount = regs;
  for (int64_t b = 0, s = 1; (1 << b) < regs; ++b, s *= 2)
    f.regBits.push_back({(int)b, (int32_t)s});
  return f;
}

} // namespace

int main() {
  // ── the contiguity condition ───────────────────────────────────────────

  CASE("contiguous lane bits are accepted, at any offset");
  {
    CHECK(laneBitsContiguous(0b111));
    CHECK(laneBitsContiguous(0b11100));
    CHECK(laneBitsContiguous(0b1));
    CHECK(laneBitsContiguous(0));
  }

  CASE("a gap in the lane bits is declined");
  {
    CHECK(!laneBitsContiguous(0b101));
    CHECK(!laneBitsContiguous(0b1011));

    ScanFacts f = facts({{0, 1}, {2, 4}});
    Decision d = scanDecline(f);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK_EQ(d.why(), std::string("axis lane bits are not contiguous"));
    CHECK(!planScan(f).usable);
  }

  // ── the halving ladder ─────────────────────────────────────────────────

  CASE("an unbroken halving ladder is accepted");
  {
    CHECK(stridesFormLadder(facts({{0, 1}, {1, 2}, {2, 4}})));
    CHECK(stridesFormLadder(facts({{0, 1}, {1, 2}}, {{0, 4}})));
  }

  CASE("a skipped halving step is declined");
  {
    ScanFacts f = facts({{0, 1}, {1, 4}});
    CHECK(!stridesFormLadder(f));
    Decision d = scanDecline(f);
    CHECK(d.isDecline());
    CHECK_EQ(d.why(), std::string("axis strides skip a halving step"));
    CHECK(!planScan(f).usable);
  }

  CASE("duplicate strides do not break the ladder");
  {
    CHECK(stridesFormLadder(facts({{0, 1}, {1, 1}, {2, 2}})));
  }

  CASE("an empty axis is trivially fine");
  {
    ScanFacts f = facts({});
    CHECK(scanDecline(f).ok());
    CHECK(planScan(f).usable);
  }

  // ── the lane ladder ────────────────────────────────────────────────────

  CASE("the ladder is shift-by-powers-of-two, one step per lane bit");
  {
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}, {2, 4}}));
    CHECK_EQ(p.laneSteps.size(), 3u);
    CHECK_EQ(p.laneSteps[0].delta, 1);
    CHECK_EQ(p.laneSteps[1].delta, 2);
    CHECK_EQ(p.laneSteps[2].delta, 4);
  }

  CASE("every lane step is guarded");
  {
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}}));
    for (const ScanStep &s : p.laneSteps)
      CHECK(s.guarded);
  }

  CASE("a full 32-lane scan is five steps");
  {
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}, {2, 4}, {3, 8}, {4, 16}}));
    CHECK_EQ(p.laneSteps.size(), 5u);
    CHECK_EQ(p.laneSteps[4].delta, 16);
  }

  CASE("no lane bits means no shuffle at all");
  {
    ScanPlan p = planScan(facts({}));
    CHECK_EQ(p.laneSteps.size(), 0u);
  }

  // ── windows ────────────────────────────────────────────────────────────

  CASE("the window is twice the reach");
  {
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}}, {}, 1, /*regs=*/4));
    CHECK_EQ(p.reach, 2);
    CHECK_EQ(p.windowRegs, 4);
    CHECK(p.needsLocalPass());
  }

  CASE("the layout says which registers chain");
  {
    ScanFacts f = facts({{0, 1}, {1, 2}}, {}, 1, /*regs=*/16);
    CHECK(planScan(f).usable);
    CHECK(scanDecline(f).ok());

    CHECK_EQ((int)localRegBits(f), 2);
    CHECK_EQ(planScan(f).windowRegs, 4);

    ScanFacts across;
    across.warpBits = {{3, 1}};
    across.numWarps = 16;
    across.regCount = 4;
    CHECK(planScan(across).usable);
    CHECK_EQ(planScan(across).windowRegs, 1);
  }

  CASE("a register bit above the spread is a window");
  {
    ScanFacts f;
    f.laneBits = {{3, 1}, {4, 2}};
    f.warpBits = {{0, 4}, {1, 8}};
    f.regBits = {{0, 16}};
    f.numWarps = 4;
    f.regCount = 2;
    CHECK_EQ(spreadReach(f), 8);
    CHECK_EQ((int)localRegBits(f), 0);
    CHECK_EQ(planScan(f).windowRegs, 1);

    ScanFacts both;
    for (int b = 0; b < 5; ++b)
      both.laneBits.push_back({b, (int32_t)(4 << b)});
    both.warpBits = {{0, 128}, {1, 256}};
    both.regBits = {{0, 1}, {1, 2}, {2, 512}};
    both.numWarps = 4;
    both.regCount = 8;
    CHECK_EQ(spreadReach(both), 256);
    CHECK_EQ((int)localRegBits(both), 2);
    CHECK_EQ(planScan(both).windowRegs, 4);
    CHECK(planScan(both).usable);

    ScanFacts alone;
    alone.regBits = {{0, 1}, {1, 2}};
    alone.regCount = 4;
    CHECK_EQ(spreadReach(alone), 0);
    CHECK_EQ((int)localRegBits(alone), 2);
  }

  CASE("with no ladder, one window covers every register");
  {
    ScanFacts f = facts({}, {}, 1, /*regs=*/64);
    CHECK(planScan(f).usable);
    CHECK_EQ(planScan(f).windowRegs, 64);
  }

  CASE("a register count below the window caps it");
  {
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}}, {}, 1, /*regs=*/2));
    CHECK_EQ(p.windowRegs, 2);
  }

  CASE("one register per window needs no local pass");
  {
    ScanPlan p = planScan(facts({{0, 1}}, {}, 1, /*regs=*/1));
    CHECK_EQ(p.windowRegs, 1);
    CHECK(!p.needsLocalPass());
  }

  // ── cross-warp ─────────────────────────────────────────────────────────

  CASE("a scan within one warp needs no scratch");
  {
    ScanPlan p = planScan(facts({{0, 1}, {1, 2}}, {}, 8));
    CHECK(!p.crossWarp);
    CHECK_EQ(p.scratch.slotsPerOperand, 0);
  }

  CASE("a scan spanning warps reserves scratch for every warp");
  {
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}}, 8));
    CHECK(p.crossWarp);
    CHECK_EQ(p.scratch.slotsPerOperand, 8 * 32);
    CHECK(p.scratch.slotFor(7, 31) < p.scratch.slotsPerOperand);
  }

  CASE("a single warp is never cross-warp, whatever the mask says");
  {
    ScanPlan p = planScan(facts({{0, 1}}, {{0, 2}}, /*numWarps=*/1));
    CHECK(!p.crossWarp);
  }

  // ── reverse ────────────────────────────────────────────────────────────

  CASE("reverse is carried straight through");
  {
    ScanFacts f = facts({{0, 1}});
    f.reverse = true;
    CHECK(planScan(f).reverse);
    CHECK(!planScan(facts({{0, 1}})).reverse);
  }

  CASE("reverse changes all four direction answers");
  {
    ScanFacts f = facts({{0, 1}});
    f.reverse = true;
    const ScanPlan rev = planScan(f);
    const ScanPlan fwd = planScan(facts({{0, 1}}));

    CHECK_EQ(std::string(fwd.shuffleName()),
             std::string(msl::builtin::simd::ShuffleUp));
    CHECK_EQ(std::string(rev.shuffleName()),
             std::string(msl::builtin::simd::ShuffleDown));

    CHECK(fwd.guardOp() == msl::BinOp::Ge);
    CHECK(rev.guardOp() == msl::BinOp::Le);

    CHECK_EQ(fwd.guardBound(1, 32), 1);
    CHECK_EQ(rev.guardBound(1, 32), 0);

    ScanFacts wideF = facts({{0, 1}, {1, 2}, {2, 4}, {3, 8}, {4, 16}});
    CHECK_EQ(planScan(wideF).guardBound(4, 32), 4);
    wideF.reverse = true;
    CHECK_EQ(planScan(wideF).guardBound(4, 32), 27);

    CHECK_EQ(fwd.totalLane(32), 1);
    CHECK_EQ(rev.totalLane(32), 0);
    wideF.reverse = false;
    CHECK_EQ(planScan(wideF).totalLane(32), 31);

    CHECK(fwd.carryOp() == msl::BinOp::Gt);
    CHECK(rev.carryOp() == msl::BinOp::Lt);
  }

  CASE("the ladder computes a real prefix sum, in both directions");
  {
    const int64_t W = 32;
    for (bool reverse : {false, true}) {
      ScanFacts f;
      for (int b = 0, s = 1; b < 5; ++b, s <<= 1)
        f.laneBits.push_back({b, s});
      f.numWarps = 1;
      f.regCount = 1;
      f.reverse = reverse;
      const ScanPlan p = planScan(f);
      CHECK(p.usable);

      std::vector<double> v(W);
      for (int64_t i = 0; i < W; ++i)
        v[(std::size_t)i] = double(i + 1);

      for (const ScanStep &st : p.laneSteps) {
        std::vector<double> peer(W, 0.0);
        for (int64_t i = 0; i < W; ++i) {
          const int64_t src = reverse ? i + st.delta : i - st.delta;
          peer[(std::size_t)i] =
              (src >= 0 && src < W) ? v[(std::size_t)src] : 0.0;
        }
        std::vector<double> next = v;
        for (int64_t i = 0; i < W; ++i) {
          const bool apply =
              !st.guarded || (reverse ? i <= p.guardBound(st.delta, W)
                                      : i >= p.guardBound(st.delta, W));
          if (apply)
            next[(std::size_t)i] = v[(std::size_t)i] + peer[(std::size_t)i];
        }
        v = next;
      }

      for (int64_t i = 0; i < W; ++i) {
        double want = 0;
        if (reverse)
          for (int64_t j = i; j < W; ++j)
            want += double(j + 1);
        else
          for (int64_t j = 0; j <= i; ++j)
            want += double(j + 1);
        CHECK_EQ(v[(std::size_t)i], want);
      }
    }
  }

  CASE("the cross-warp carry is correct for a partial warp mask");
  {
    const int64_t NW = 8;
    for (const std::vector<AxisBit> &warpBits :
         {std::vector<AxisBit>{{0, 2}}, std::vector<AxisBit>{{0, 2}, {1, 4}},
          std::vector<AxisBit>{{0, 2}, {1, 4}, {2, 8}},
          std::vector<AxisBit>{{1, 2}}, std::vector<AxisBit>{{2, 2}},
          std::vector<AxisBit>{{0, 2}, {2, 4}}}) {
      ScanFacts f;
      f.laneBits = {{0, 1}};
      f.warpBits = warpBits;
      f.numWarps = NW;
      f.regCount = 1;
      const ScanPlan p = planScan(f);
      CHECK(p.usable);
      CHECK_EQ(p.warpMask, maskOf(warpBits));
      CHECK(!p.carryWarps(NW).empty());

      std::vector<double> v(NW);
      for (int64_t w = 0; w < NW; ++w)
        v[(std::size_t)w] = double(w + 1);
      const std::vector<double> published = v;

      const std::vector<int64_t> carry = p.carryFoldOrder(NW);
      const unsigned anchorMask = p.anchorMask(NW);
      std::vector<double> got = v;
      for (const int64_t w : carry)
        for (int64_t me = 0; me < NW; ++me)
          if (((unsigned)me & p.warpMask) > (unsigned)w)
            got[(std::size_t)me] += published[(
                std::size_t)(((unsigned)me & anchorMask) + (unsigned)w)];

      for (int64_t me = 0; me < NW; ++me) {
        double want = 0;
        const unsigned anchor = (unsigned)me & anchorMask;
        const unsigned pos = (unsigned)me & p.warpMask;
        for (int64_t o = 0; o < NW; ++o)
          if (((unsigned)o & ~p.warpMask) == 0 && (unsigned)o <= pos)
            want += published[(std::size_t)(anchor + (unsigned)o)];
        CHECK_EQ(got[(std::size_t)me], want);
      }
    }
  }

  CASE("the whole scan is a prefix sum over lanes and registers");
  {
    const int64_t W = 32;
    for (int laneBits : {1, 2, 5})
      for (int64_t regs : {1, 2, 4}) {
        ScanFacts f;
        for (int b = 0, s = 1; b < laneBits; ++b, s <<= 1)
          f.laneBits.push_back({b, s});
        f.numWarps = 1;
        f.regCount = regs;
        for (int64_t b = 0, s = 1; (1 << b) < regs; ++b, s *= 2)
          f.regBits.push_back({(int)b, (int32_t)s});
        const ScanPlan p = planScan(f);
        if (!p.usable)
          continue;
        if (p.windowRegs != regs)
          continue;

        std::vector<std::vector<double>> v((std::size_t)W);
        for (int64_t l = 0; l < W; ++l)
          for (int64_t r = 0; r < regs; ++r)
            v[(std::size_t)l].push_back(double(l * 100 + r + 1));
        std::vector<std::vector<double>> acc = v;

        for (int64_t l = 0; l < W; ++l)
          for (int64_t r = 1; r < regs; ++r)
            if (!p.startsWindow(r))
              acc[(std::size_t)l][(std::size_t)r] +=
                  acc[(std::size_t)l][(std::size_t)(r - 1)];

        std::vector<double> laneScan((std::size_t)W);
        for (int64_t l = 0; l < W; ++l)
          laneScan[(std::size_t)l] =
              acc[(std::size_t)l][(std::size_t)(regs - 1)];

        for (const ScanStep &st : p.laneSteps) {
          std::vector<double> next = laneScan;
          for (int64_t l = 0; l < W; ++l) {
            const int64_t src = l - st.delta;
            const double peer = src >= 0 ? laneScan[(std::size_t)src] : 0.0;
            if (!st.guarded || l >= p.guardBound(st.delta, W))
              next[(std::size_t)l] += peer;
          }
          laneScan = next;
        }

        if (!p.laneSteps.empty()) {
          const int64_t low = p.laneSteps.front().delta;
          for (int64_t l = p.guardBound(low, W); l < W; ++l)
            for (int64_t r = 0; r < regs; ++r)
              acc[(std::size_t)l][(std::size_t)r] +=
                  laneScan[(std::size_t)(l - low)];
        }

        const int64_t axisLanes = int64_t(1) << laneBits;
        for (int64_t l = 0; l < axisLanes; ++l)
          for (int64_t r = 0; r < regs; ++r) {
            double want = 0;
            for (int64_t l2 = 0; l2 <= l; ++l2)
              for (int64_t r2 = 0; r2 < regs; ++r2) {
                if (l2 == l && r2 > r)
                  continue;
                want += v[(std::size_t)l2][(std::size_t)r2];
              }
            CHECK_EQ(acc[(std::size_t)l][(std::size_t)r], want);
          }
      }
  }

  CASE("registers above the spread are segments of one scan and they chain");
  {
    const int64_t W = 32;
    const int64_t NW = 4;
    const int64_t localRegs = 2;
    const int64_t blocks = 2;
    const int64_t regs = localRegs * blocks;
    const int64_t axis = 512;

    ScanFacts f;
    for (int b = 0; b < 5; ++b)
      f.laneBits.push_back({b, (int32_t)(localRegs << b)});
    f.warpBits = {{0, 64}, {1, 128}};
    f.regBits = {{0, 1}, {1, 256}};
    f.numWarps = NW;
    f.regCount = regs;

    const ScanPlan p = planScan(f);
    CHECK(p.usable);
    CHECK(p.crossWarp);
    CHECK_EQ((int)localRegBits(f), 1);
    CHECK_EQ(p.windowRegs, localRegs);
    CHECK_EQ(p.chainedWindows, blocks);

    const auto posOf = [&](int64_t b, int64_t w, int64_t l, int64_t r) {
      return b * (axis / blocks) + w * (W * localRegs) + l * localRegs + r;
    };
    const auto label = [](int64_t pos) { return std::to_string(pos) + ","; };

    std::vector<std::vector<std::vector<std::string>>> acc(
        (std::size_t)NW,
        std::vector<std::vector<std::string>>(
            (std::size_t)W, std::vector<std::string>((std::size_t)regs)));
    for (int64_t w = 0; w < NW; ++w)
      for (int64_t l = 0; l < W; ++l)
        for (int64_t b = 0; b < blocks; ++b)
          for (int64_t r = 0; r < localRegs; ++r)
            acc[(std::size_t)w][(std::size_t)l]
               [(std::size_t)(b * localRegs + r)] = label(posOf(b, w, l, r));

    std::string segCarry;
    for (int64_t base = 0; base < regs; base += p.windowRegs) {
      if ((base / p.windowRegs) % p.chainedWindows == 0)
        segCarry.clear();

      for (int64_t w = 0; w < NW; ++w)
        for (int64_t l = 0; l < W; ++l)
          for (int64_t r = 1; r < p.windowRegs; ++r)
            acc[(std::size_t)w][(std::size_t)l][(std::size_t)(base + r)] =
                acc[(std::size_t)w][(std::size_t)l]
                   [(std::size_t)(base + r - 1)] +
                acc[(std::size_t)w][(std::size_t)l][(std::size_t)(base + r)];

      std::vector<std::vector<std::string>> laneScan(
          (std::size_t)NW, std::vector<std::string>((std::size_t)W));
      for (int64_t w = 0; w < NW; ++w)
        for (int64_t l = 0; l < W; ++l)
          laneScan[(std::size_t)w][(std::size_t)l] =
              acc[(std::size_t)w][(std::size_t)l]
                 [(std::size_t)(base + p.windowRegs - 1)];

      for (const ScanStep &st : p.laneSteps)
        for (int64_t w = 0; w < NW; ++w) {
          std::vector<std::string> next = laneScan[(std::size_t)w];
          for (int64_t l = 0; l < W; ++l) {
            const int64_t src = l - st.delta;
            if (!st.guarded || l >= p.guardBound(st.delta, W))
              next[(std::size_t)l] =
                  laneScan[(std::size_t)w][(std::size_t)src] +
                  next[(std::size_t)l];
          }
          laneScan[(std::size_t)w] = next;
        }

      std::vector<std::string> totals((std::size_t)NW);
      for (int64_t w = 0; w < NW; ++w)
        totals[(std::size_t)w] =
            laneScan[(std::size_t)w][(std::size_t)p.totalLane(W)];

      if (!p.laneSteps.empty()) {
        const int64_t low = p.laneSteps.front().delta;
        for (int64_t w = 0; w < NW; ++w)
          for (int64_t l = p.guardBound(low, W); l < W; ++l)
            for (int64_t r = 0; r < p.windowRegs; ++r)
              acc[(std::size_t)w][(std::size_t)l][(std::size_t)(base + r)] =
                  laneScan[(std::size_t)w][(std::size_t)(l - low)] +
                  acc[(std::size_t)w][(std::size_t)l][(std::size_t)(base + r)];
      }

      const std::vector<int64_t> fold = p.carryFoldOrder(NW);
      for (int64_t me = 0; me < NW; ++me) {
        const unsigned anchor = (unsigned)me & p.anchorMask(NW);
        const unsigned pos = (unsigned)me & p.warpMask;
        std::string carry = totals[(std::size_t)(anchor + fold.front())];
        for (std::size_t i = 1; i < fold.size(); ++i)
          if (pos > (unsigned)fold[i])
            carry += totals[(std::size_t)(anchor + fold[i])];
        if (pos > (unsigned)fold.front())
          for (int64_t l = 0; l < W; ++l)
            for (int64_t r = 0; r < p.windowRegs; ++r)
              acc[(std::size_t)me][(std::size_t)l][(std::size_t)(base + r)] =
                  carry +
                  acc[(std::size_t)me][(std::size_t)l][(std::size_t)(base + r)];
      }

      if (!segCarry.empty())
        for (int64_t w = 0; w < NW; ++w)
          for (int64_t l = 0; l < W; ++l)
            for (int64_t r = 0; r < p.windowRegs; ++r)
              acc[(std::size_t)w][(std::size_t)l][(std::size_t)(base + r)] =
                  segCarry +
                  acc[(std::size_t)w][(std::size_t)l][(std::size_t)(base + r)];

      segCarry = acc[(std::size_t)p.finalWarp()][(std::size_t)p.totalLane(W)]
                    [(std::size_t)(base + p.windowRegs - 1)];
    }

    std::string prefix;
    std::vector<std::string> want((std::size_t)axis);
    for (int64_t pos = 0; pos < axis; ++pos) {
      prefix += label(pos);
      want[(std::size_t)pos] = prefix;
    }
    for (int64_t b = 0; b < blocks; ++b)
      for (int64_t w = 0; w < NW; ++w)
        for (int64_t l = 0; l < W; ++l)
          for (int64_t r = 0; r < localRegs; ++r)
            CHECK_EQ(acc[(std::size_t)w][(std::size_t)l]
                        [(std::size_t)(b * localRegs + r)],
                     want[(std::size_t)posOf(b, w, l, r)]);
  }

  CASE("carryWarps yields only the warps the axis traverses");
  {
    ScanFacts f = facts({{0, 1}}, {{0, 2}}, 8);
    const ScanPlan p = planScan(f);
    CHECK_EQ(p.warpMask, 1u);
    CHECK_EQ((int)p.carryWarps(8).size(), 2);
    CHECK_EQ(p.anchorMask(8), 6u);

    ScanFacts all = facts({{0, 1}}, {{0, 2}, {1, 4}, {2, 8}}, 8);
    CHECK_EQ((int)planScan(all).carryWarps(8).size(), 8);
    CHECK_EQ(planScan(all).anchorMask(8), 0u);
  }

  CASE("every direction answer differs between the two, so none is a stub");
  {
    ScanFacts f = facts({{0, 1}});
    f.reverse = true;
    const ScanPlan rev = planScan(f);
    const ScanPlan fwd = planScan(facts({{0, 1}}));
    CHECK(std::string(fwd.shuffleName()) != std::string(rev.shuffleName()));
    CHECK(fwd.guardOp() != rev.guardOp());
    CHECK(fwd.guardBound(1, 32) != rev.guardBound(1, 32));
    CHECK(fwd.totalLane(32) != rev.totalLane(32));
    CHECK(fwd.carryOp() != rev.carryOp());
  }

  // ── multi-operand agreement ────────────────────────────────────────────

  CASE("operands that disagree on register layout are refused");
  {
    ScanFacts f = facts({{0, 1}});
    f.regsPerOperand = {4, 2};
    CHECK(!operandsShareLayout(f));
    Decision d = scanDecline(f);
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK(!planScan(f).usable);
  }

  CASE("operands that agree are emitted");
  {
    ScanFacts f = facts({{0, 1}});
    f.regsPerOperand = {4, 4};
    CHECK(operandsShareLayout(f));
    CHECK(planScan(f).usable);
  }

  CASE("saying nothing about operands is not a disagreement");
  {
    ScanFacts f = facts({{0, 1}});
    CHECK(f.regsPerOperand.empty());
    CHECK(operandsShareLayout(f));
    CHECK(planScan(f).usable);
  }

  // ── the decline is not a failure ───────────────────────────────────────

  CASE("a declined scan reports a reason and is not a bug");
  {
    for (ScanFacts f : {facts({{0, 1}, {2, 4}}), facts({{0, 1}, {1, 4}})}) {
      Decision d = scanDecline(f);
      CHECK(d.isDecline());
      CHECK(!d.isBug());
      CHECK(!d.keepLooking());
      CHECK(!d.why().empty());
      CHECK_EQ(d.where(), std::string("emitScan"));
    }
  }

  return ::agpu_test::report("ScanPlan");
}
