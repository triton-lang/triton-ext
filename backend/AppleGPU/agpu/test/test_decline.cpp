// The decline channel: "I do not cover this" is a third state, distinct from
// a failure or a miss.
#include "agpu/core/Decline.h"
#include "harness.h"

#include <sstream>

using namespace agpu;

int main() {
  // ── the third state ────────────────────────────────────────────────────

  CASE("a decline is neither a failure nor a miss");
  {
    Decision d = Decision::declined("emitScan", "unsupported-lane-layout");
    CHECK(!d.ok());
    CHECK(d.isDecline());
    CHECK(!d.isBug());
    CHECK(!d.keepLooking());

    Decision f = Decision::failed();
    CHECK(!f.ok());
    CHECK(!f.isDecline());
    CHECK(f.isBug());

    Decision n = Decision::notMine();
    CHECK(!n.ok());
    CHECK(n.keepLooking());
    CHECK(!n.isBug());
  }

  CASE("only a miss makes the caller keep looking");
  {
    CHECK(Decision::notMine().keepLooking());
    CHECK(!Decision::declined("emitScan", "x").keepLooking());
    CHECK(!Decision::failed().keepLooking());
    CHECK(!Decision::emitted().keepLooking());
  }

  CASE("only a failure is a bug");
  {
    // A decline means the input is legal and not covered.
    CHECK(Decision::failed().isBug());
    CHECK(!Decision::declined("emitScan", "x").isBug());
    CHECK(!Decision::notMine().isBug());
  }

  // ── the reason survives ────────────────────────────────────────────────

  CASE("a decline names what declined and why");
  {
    Decision d = Decision::declined("emitScan", "gapped-stride-ladder");
    CHECK_EQ(d.where(), std::string("emitScan"));
    CHECK_EQ(d.why(), std::string("gapped-stride-ladder"));
    CHECK_EQ(d.message(), std::string("emitScan: gapped-stride-ladder"));
  }

  CASE("nothing else carries a message");
  {
    CHECK(Decision::failed().message().empty());
    CHECK(Decision::notMine().message().empty());
    CHECK(Decision::emitted().message().empty());
  }

  // ── the log ────────────────────────────────────────────────────────────

  CASE("a log collects declines and ignores everything else");
  {
    DeclineLog log;
    log.record(Decision::emitted());
    log.record(Decision::notMine());
    log.record(Decision::failed());
    CHECK(log.empty());

    log.record(Decision::declined("emitScan", "unsupported-lane-layout"));
    log.record(Decision::declined("emitAtomicRMW", "int64-unsupported"));
    CHECK_EQ(log.size(), 2u);
  }

  CASE("a run reports every shape it could not cover");
  {
    DeclineLog log;
    log.record(Decision::declined("emitScan", "unsupported-lane-layout"));
    log.record(Decision::declined("emitScan", "gapped-stride-ladder"));
    log.record(Decision::declined("emitAtomicRMW", "int64-unsupported"));

    CHECK(log.declined("gapped-stride-ladder"));
    CHECK(log.declined("int64-unsupported"));
    CHECK(!log.declined("never-happened"));
    CHECK_EQ(log.size(), 3u);
  }

  CASE("the same reason from two sites is recorded twice");
  {
    DeclineLog log;
    log.record(Decision::declined("emitScan", "unsupported-lane-layout"));
    log.record(Decision::declined("emitScan", "unsupported-lane-layout"));
    CHECK_EQ(log.size(), 2u);
  }

  // ── the summary ────────────────────────────────────────────────────────

  CASE("occurrences and distinct sites are counted separately");
  {
    // Autotuning recompiles one kernel under N configurations, so N identical
    // declines collapse to a single site.
    DeclineLog log;
    const Decision d = Decision::declined("emitScan", "gapped ladder");
    for (int cfg = 0; cfg < 4; ++cfg)
      log.record(d,
                 DeclineSite{"kernel.py:31", "warps=" + std::to_string(cfg)});

    const auto sum = log.summary();
    CHECK_EQ((int)sum.size(), 1);
    CHECK_EQ(sum[0].occurrences, 4u);
    CHECK_EQ(sum[0].distinctSites(), 1u);
    CHECK_EQ(sum[0].where, std::string("emitScan"));
    CHECK_EQ(sum[0].why, std::string("gapped ladder"));
  }

  CASE("different sites with one reason are counted apart");
  {
    DeclineLog log;
    const Decision d = Decision::declined("emitScan", "gapped ladder");
    log.record(d, DeclineSite{"a.py:1", "warps=4"});
    log.record(d, DeclineSite{"b.py:9", "warps=4"});
    log.record(d, DeclineSite{"b.py:9", "warps=8"});

    const auto sum = log.summary();
    CHECK_EQ((int)sum.size(), 1);
    CHECK_EQ(sum[0].occurrences, 3u);
    CHECK_EQ(sum[0].distinctSites(), 2u);
  }

  CASE("one site declining for two reasons is two rows");
  {
    DeclineLog log;
    log.record(Decision::declined("emitScan", "gapped ladder"),
               DeclineSite{"a.py:1", "warps=4"});
    log.record(Decision::declined("emitScan", "non-contiguous lane bits"),
               DeclineSite{"a.py:1", "warps=4"});
    log.record(Decision::declined("emitDot", "gapped ladder"),
               DeclineSite{"a.py:1", "warps=4"});

    CHECK_EQ((int)log.summary().size(), 3);
  }

  CASE("a decline with no site still counts as an occurrence");
  {
    DeclineLog log;
    log.record(Decision::declined("emitScan", "gapped ladder"));
    const auto sum = log.summary();
    CHECK_EQ(sum[0].occurrences, 1u);
    CHECK_EQ(sum[0].distinctSites(), 0u);
  }

  CASE("nothing but declines reaches the summary");
  {
    DeclineLog log;
    log.record(Decision::emitted(), DeclineSite{"a.py:1", ""});
    log.record(Decision::notMine(), DeclineSite{"a.py:2", ""});
    log.record(Decision::failed(), DeclineSite{"a.py:3", ""});
    CHECK(log.empty());
    CHECK(log.summary().empty());
  }

  // ── what a report line means ───────────────────────────────────────────

  CASE("a reject and a plan note do not merge into one row");
  {
    // Keying the summary on (where, why) alone would merge a reject with a
    // plan note that happens to share a reason.
    DeclineLog log;
    log.record(Decision::declined("emitDot", "k not a multiple of 8"),
               DeclineSite{"a.py:1", "w4"});
    log.note(Decision::declined("emitDot", "k not a multiple of 8"),
             DeclineSite{"a.py:1", "w4"});

    const std::vector<DeclineTally> rows = log.summary();
    CHECK_EQ(rows.size(), (std::size_t)2);
    CHECK(rows[0].tag != rows[1].tag);
  }

  CASE("a reject is the default, because it is the one that costs something");
  {
    DeclineLog log;
    log.record(Decision::declined("emitScan", "gapped stride"));
    CHECK(log.summary()[0].tag == DeclineTag::Reject);

    DeclineLog notes;
    notes.note(Decision::declined("emitDot", "chose the direct path"));
    CHECK(notes.summary()[0].tag == DeclineTag::Plan);
  }

  CASE("sites and configs are counted independently");
  {
    // One site under three configs is one problem; three sites under one
    // config is three.
    DeclineLog sweep;
    for (const char *cfg : {"w1", "w2", "w4"})
      sweep.record(Decision::declined("emitDot", "shape"),
                   DeclineSite{"a.py:9", cfg});
    CHECK_EQ(sweep.summary()[0].occurrences, (std::size_t)3);
    CHECK_EQ(sweep.summary()[0].distinctSites(), (std::size_t)1);
    CHECK_EQ(sweep.summary()[0].distinctConfigs(), (std::size_t)3);

    DeclineLog spread;
    for (const char *site : {"a.py:1", "a.py:2", "a.py:3"})
      spread.record(Decision::declined("emitDot", "shape"),
                    DeclineSite{site, "w4"});
    CHECK_EQ(spread.summary()[0].distinctSites(), (std::size_t)3);
    CHECK_EQ(spread.summary()[0].distinctConfigs(), (std::size_t)1);
  }

  CASE("an unsupplied site or config is not counted as one named empty");
  {
    DeclineLog log;
    log.record(Decision::declined("emitDot", "shape"));
    // The empty config is written out.
    log.record(Decision::declined("emitDot", "shape"),
               DeclineSite{"a.py:1", ""});
    CHECK_EQ(log.summary()[0].distinctSites(), (std::size_t)1);
    CHECK_EQ(log.summary()[0].distinctConfigs(), (std::size_t)0);
  }

  // ── the teardown summary ───────────────────────────────────────────────

  CASE("the summary leads with rejects and counts only those");
  {
    // A run whose every line is a plan note has no rejects to report.
    DeclineLog log;
    log.record(Decision::declined("emitDot", "ragged k"),
               DeclineSite{"a.py:1", "w4"});
    log.note(Decision::declined("emitDot", "took the panel path"),
             DeclineSite{"a.py:1", "w4"});
    log.note(Decision::declined("emitScan", "one warp, no carry"),
             DeclineSite{"a.py:5", "w4"});

    std::ostringstream os;
    log.printSummary(os);
    const std::string out = os.str();

    CHECK(out.find("distinct rejects: 1") != std::string::npos);
    CHECK(out.find("plan notes: 2") != std::string::npos);
    CHECK(out.find("MSL-REJECT-SITE") < out.find("MSL-PLAN-SITE"));
    CHECK(out.find("sites=1") != std::string::npos);
    CHECK(out.find("configs=1") != std::string::npos);
  }

  CASE("the summary names the sites beyond just counting them");
  {
    DeclineLog log;
    log.record(Decision::declined("tt.print", "not implemented"),
               DeclineSite{"attention.py:41", "w4"});
    std::ostringstream os;
    log.printSummary(os);
    CHECK(os.str().find("attention.py:41") != std::string::npos);
  }

  CASE("many sites are capped and the summary states the cap");
  {
    DeclineLog log;
    for (int i = 0; i < 9; ++i)
      log.record(Decision::declined("tt.print", "not implemented"),
                 DeclineSite{"a.py:" + std::to_string(i), "w4"});
    std::ostringstream os;
    log.printSummary(os);
    const std::string out = os.str();

    CHECK(out.find("sites=9") != std::string::npos);
    CHECK(out.find("a.py:0") != std::string::npos);
    CHECK(out.find("a.py:2") != std::string::npos);
    CHECK(out.find("(+6 more)") != std::string::npos);
    CHECK(out.find("a.py:8") == std::string::npos);
  }

  CASE("a decline with no site named prints no location clause");
  {
    DeclineLog log;
    log.record(Decision::declined("emitDot", "shape"));
    std::ostringstream os;
    log.printSummary(os);
    CHECK(os.str().find("\tat ") == std::string::npos);
    CHECK(os.str().find("sites=0") != std::string::npos);
  }

  CASE("an empty log prints nothing at all");
  {
    DeclineLog log;
    std::ostringstream os;
    log.printSummary(os);
    CHECK(os.str().empty());

    log.record(Decision::emitted(), DeclineSite{"a.py:1", "w4"});
    std::ostringstream os2;
    log.printSummary(os2);
    CHECK(os2.str().empty());
  }

  CASE("a plan-only run reports zero rejects");
  {
    DeclineLog log;
    log.note(Decision::declined("emitDot", "took the direct path"));
    std::ostringstream os;
    log.printSummary(os);
    CHECK(os.str().find("distinct rejects: 0") != std::string::npos);
    CHECK(os.str().find("plan notes: 1") != std::string::npos);
  }

  return ::agpu_test::report("Decline");
}
