#include "agpu/emit/EmitRegion.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::render;

namespace {

RegionNames namesFor() {
  RegionNames nm;
  nm.namesOf = [](ValueId v) { return ValueNames{"v" + std::to_string(v)}; };
  nm.typeOf = [](ValueId) { return msl::Type::scalar(msl::Scalar::I32); };
  return nm;
}

RegionNames namesForTensor(int regs) {
  RegionNames nm = namesFor();
  nm.namesOf = [regs](ValueId v) {
    ValueNames out;
    for (int r = 0; r < regs; ++r)
      out.push_back("v" + std::to_string(v) + "_" + std::to_string(r));
    return out;
  };
  return nm;
}

RegionFacts chain() {
  RegionFacts f;
  f.blocks.resize(2);
  f.blocks[0].term = TermKind::Branch;
  f.blocks[0].edges = {Edge{1, {}}};
  f.blocks[1].term = TermKind::Return;
  return f;
}

} // namespace

int main() {
  CASE("a value read only where it is defined stays inside its case");
  {
    RegionFacts f = chain();
    f.blocks[0].defines = {10};
    f.blocks[0].reads = {10};
    RegionPlan p = planRegion(f);
    CHECK(p.hoisted.empty());
  }

  CASE("a value read in another block must be hoisted");
  {
    // An MSL variable dies at its closing brace.
    RegionFacts f = chain();
    f.blocks[0].defines = {10};
    f.blocks[1].reads = {10};
    RegionPlan p = planRegion(f);
    CHECK_EQ((int)p.hoisted.size(), 1);
    CHECK_EQ(p.hoisted[0], 10);
  }

  CASE("a value read by an earlier block still crosses");
  {
    RegionFacts f;
    f.blocks.resize(2);
    f.blocks[0].reads = {7};
    f.blocks[0].term = TermKind::Branch;
    f.blocks[0].edges = {Edge{1, {}}};
    f.blocks[1].defines = {7};
    f.blocks[1].term = TermKind::Branch;
    f.blocks[1].edges = {Edge{0, {}}};

    RegionPlan p = planRegion(f);
    CHECK(detail::contains(p.hoisted, 7));
  }

  CASE("a value carried by an edge from elsewhere has crossed");
  {
    // An edge argument counts as a read in the source block.
    RegionFacts f;
    f.blocks.resize(3);
    f.blocks[0].defines = {5};
    f.blocks[0].term = TermKind::Branch;
    f.blocks[0].edges = {Edge{1, {}}};
    f.blocks[1].term = TermKind::Branch;
    f.blocks[1].edges = {Edge{2, {5}}}; // block 1 carries block 0's value
    f.blocks[2].params = {9};
    f.blocks[2].term = TermKind::Return;

    RegionPlan p = planRegion(f);
    CHECK(detail::contains(p.hoisted, 5));
  }

  CASE("every block parameter is hoisted, whatever the read pattern");
  {
    RegionFacts f = chain();
    f.blocks[0].edges = {Edge{1, {3}}};
    f.blocks[1].params = {4};
    f.blocks[1].reads = {4};

    RegionPlan p = planRegion(f);
    CHECK(detail::contains(p.hoisted, 4));
  }

  CASE("a value is hoisted once, however many blocks read it");
  {
    RegionFacts f;
    f.blocks.resize(3);
    f.blocks[0].defines = {1};
    f.blocks[0].term = TermKind::Branch;
    f.blocks[0].edges = {Edge{1, {}}};
    f.blocks[1].reads = {1};
    f.blocks[1].term = TermKind::Branch;
    f.blocks[1].edges = {Edge{2, {}}};
    f.blocks[2].reads = {1};
    f.blocks[2].term = TermKind::Return;

    RegionPlan p = planRegion(f);
    CHECK_EQ((int)p.hoisted.size(), 1);
  }

  CASE("an edge must supply exactly the arguments its destination binds");
  {
    RegionFacts f = chain();
    f.blocks[1].params = {4, 5};
    f.blocks[0].edges = {Edge{1, {3}}}; // one argument, two parameters
    CHECK(!regionDecision(f, planRegion(f)).ok());

    f.blocks[0].edges = {Edge{1, {3, 6}}};
    CHECK(regionDecision(f, planRegion(f)).ok());
  }

  CASE("a terminator must have the edges its kind implies");
  {
    RegionFacts f = chain();
    f.blocks[0].term = TermKind::CondBranch; // two edges, one supplied
    CHECK(!regionDecision(f, planRegion(f)).ok());

    RegionFacts g = chain();
    g.blocks[1].term = TermKind::Return;
    g.blocks[1].edges = {Edge{0, {}}}; // a return with an edge
    CHECK(!regionDecision(g, planRegion(g)).ok());
  }

  CASE("an edge to a block that does not exist is refused");
  {
    RegionFacts f = chain();
    f.blocks[0].edges = {Edge{7, {}}};
    CHECK(!regionDecision(f, planRegion(f)).ok());
  }

  CASE("an empty region declines");
  {
    RegionFacts f;
    RegionPlan p = planRegion(f);
    CHECK(!p.usable);
    CHECK(regionDecision(f, p).isDecline());
  }

  CASE("reachability follows the edges from the entry");
  {
    RegionFacts f;
    f.blocks.resize(3);
    f.blocks[0].term = TermKind::Branch;
    f.blocks[0].edges = {Edge{1, {}}};
    f.blocks[1].term = TermKind::Return;
    f.blocks[2].term = TermKind::Return; // nothing branches here

    const std::vector<bool> seen = reachableBlocks(f);
    CHECK(seen[0]);
    CHECK(seen[1]);
    CHECK(!seen[2]);
  }

  CASE("a cycle in the edges still terminates the search");
  {
    RegionFacts f;
    f.blocks.resize(2);
    f.blocks[0].term = TermKind::Branch;
    f.blocks[0].edges = {Edge{1, {}}};
    f.blocks[1].term = TermKind::Branch;
    f.blocks[1].edges = {Edge{0, {}}};
    const std::vector<bool> seen = reachableBlocks(f);
    CHECK(seen[0]);
    CHECK(seen[1]);
  }

  CASE("a region emits its hoists, then a dispatch loop");
  {
    RegionFacts f = chain();
    f.blocks[0].defines = {1};
    f.blocks[1].reads = {1};
    RegionPlan p = planRegion(f);

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, namesFor(), [&](BlockId) { return msl::Block{}; });
    const std::string s = render(out);

    CHECK(s.find("int v1;") != std::string::npos);
    CHECK(s.find("int v1;") < s.find("while ("));
    CHECK(s.find("__state = 0") != std::string::npos);
    CHECK(s.find("while (__state != -1)") != std::string::npos);
    CHECK_EQ(countOf(s, "case "), 2);
  }

  CASE("a branch assigns the state and continues");
  {
    RegionFacts f = chain();
    RegionPlan p = planRegion(f);

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, namesFor(), [&](BlockId) { return msl::Block{}; });
    const std::string s = render(out);
    CHECK(s.find("__state = 1;") != std::string::npos);
    CHECK(s.find("continue;") != std::string::npos);
    CHECK(s.find("__state = 1;") < s.find("continue;"));
  }

  CASE("a return assigns the exit state");
  {
    RegionFacts f = chain();
    RegionPlan p = planRegion(f);

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, namesFor(), [&](BlockId) { return msl::Block{}; });
    const std::string s = render(out);
    CHECK(s.find("__state = -1;") != std::string::npos);
  }

  CASE("a conditional branch emits both edges");
  {
    RegionFacts f;
    f.blocks.resize(3);
    f.blocks[0].term = TermKind::CondBranch;
    f.blocks[0].edges = {Edge{1, {}}, Edge{2, {}}};
    f.blocks[1].term = TermKind::Return;
    f.blocks[2].term = TermKind::Return;
    RegionPlan p = planRegion(f);
    CHECK(regionDecision(f, p).ok());

    msl::Context c;
    msl::Block out;
    emitRegion(
        c, out, f, p, namesFor(), [&](BlockId) { return msl::Block{}; },
        [&](BlockId) { return c.var("cond"); });
    const std::string s = render(out);
    CHECK(s.find("if (cond)") != std::string::npos);
    CHECK(s.find("__state = 1;") != std::string::npos);
    CHECK(s.find("__state = 2;") != std::string::npos);
    CHECK_EQ(countOf(s, "continue;"), 2);
  }

  CASE("an edge copies its arguments before assigning the state");
  {
    RegionFacts f = chain();
    f.blocks[1].params = {20};
    f.blocks[0].edges = {Edge{1, {30}}};
    RegionPlan p = planRegion(f);

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, namesFor(), [&](BlockId) { return msl::Block{}; });
    const std::string s = render(out);
    CHECK(s.find("v20 = v30;") != std::string::npos);
    CHECK(s.find("v20 = v30;") < s.find("__state = 1;"));
    CHECK(s.find("__state = 1;") < s.find("continue;"));
  }

  CASE("overlapping phi copies go through temporaries");
  {
    // Sequential copies would leave both holding the original v2.
    RegionFacts f;
    f.blocks.resize(1);
    f.blocks[0].params = {1, 2};
    f.blocks[0].term = TermKind::Branch;
    f.blocks[0].edges = {Edge{0, {2, 1}}}; // (a, b) -> (b, a)
    RegionPlan p = planRegion(f);
    CHECK(regionDecision(f, p).ok());

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, namesFor(), [&](BlockId) { return msl::Block{}; });
    const std::string s = render(out);

    CHECK(s.find("__phi0 = v2") != std::string::npos);
    CHECK(s.find("__phi1 = v1") != std::string::npos);
    CHECK(s.find("__phi1 = v1") < s.find("v1 = __phi0"));
    CHECK(s.find("v1 = __phi0") != std::string::npos);
    CHECK(s.find("v2 = __phi1") != std::string::npos);
  }

  CASE("non-overlapping copies need no temporaries");
  {
    RegionFacts f = chain();
    f.blocks[1].params = {20, 21};
    f.blocks[0].edges = {Edge{1, {30, 31}}};
    RegionPlan p = planRegion(f);

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, namesFor(), [&](BlockId) { return msl::Block{}; });
    const std::string s = render(out);
    CHECK(s.find("__phi") == std::string::npos);
    CHECK(s.find("v20 = v30;") != std::string::npos);
    CHECK(s.find("v21 = v31;") != std::string::npos);
  }

  CASE("a value held in several registers hoists and copies all of them");
  {
    // A tensor value is N MSL variables. Hoist is per value, declarations
    // and copies are per variable.
    RegionFacts f = chain();
    f.blocks[0].defines = {7};
    f.blocks[1].reads = {7};
    f.blocks[1].params = {20};
    f.blocks[0].edges = {Edge{1, {30}}};
    RegionPlan p = planRegion(f);

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, namesForTensor(3),
               [&](BlockId) { return msl::Block{}; });
    const std::string s = render(out);
    for (const char *reg : {"v7_0", "v7_1", "v7_2"})
      CHECK(s.find(std::string("int ") + reg + ";") != std::string::npos);
    for (int r = 0; r < 3; ++r) {
      const std::string d = "v20_" + std::to_string(r);
      const std::string src = "v30_" + std::to_string(r);
      CHECK(s.find(d + " = " + src + ";") != std::string::npos);
    }
  }

  CASE("a block's own statements precede its terminator");
  {
    RegionFacts f = chain();
    RegionPlan p = planRegion(f);

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, namesFor(), [&](BlockId b) {
      msl::Block body;
      body.push_back(c.assign(c.var("mark"), c.lit(b)));
      return body;
    });
    const std::string s = render(out);
    CHECK(s.find("mark = 0;") < s.find("__state = 1;"));
  }

  CASE("a hoist declares a value's storage");
  {
    // A pointer from `addptr` holds its base's name and owns only the
    // offset, so the hoist asks `storageOf`.
    RegionFacts f = chain();
    f.blocks[0].defines = {7};
    f.blocks[1].reads = {7};
    RegionPlan p = planRegion(f);

    RegionNames nm = namesFor();
    nm.namesOf = [](ValueId) { return ValueNames{"arg0"}; };
    nm.storageOf = [](ValueId v) {
      std::vector<std::pair<msl::Str, msl::Type>> out;
      if (v == 7)
        out.emplace_back("off7", msl::Type::scalar(msl::Scalar::I32));
      return out;
    };

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, nm, [&](BlockId) { return msl::Block{}; });
    const std::string s = render(out);
    CHECK(s.find("int off7;") != std::string::npos);
    CHECK(s.find("int arg0;") == std::string::npos);
  }

  CASE("the shadow-drop follows the storage the hoist actually declared");
  {
    RegionFacts f = chain();
    f.blocks[0].defines = {7};
    f.blocks[1].reads = {7};
    RegionPlan p = planRegion(f);

    RegionNames nm = namesFor();
    nm.namesOf = [](ValueId) { return ValueNames{"borrowed"}; };
    nm.storageOf = [](ValueId v) {
      std::vector<std::pair<msl::Str, msl::Type>> out;
      if (v == 7)
        out.emplace_back("own7", msl::Type::scalar(msl::Scalar::I32));
      return out;
    };

    msl::Context c;
    msl::Block out;
    emitRegion(c, out, f, p, nm, [&](BlockId b) {
      msl::Block body;
      if (b == 0) {
        body.push_back(
            c.declStmt(msl::Type::scalar(msl::Scalar::I32), "own7", c.lit(1)));
        body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::I32),
                                  "borrowed", c.lit(2)));
      }
      return body;
    });
    const std::string s = render(out);
    CHECK(s.find("own7 = 1;") != std::string::npos); // rewritten
    CHECK(s.find("int own7 = 1;") == std::string::npos);
    CHECK(s.find("int borrowed = 2;") != std::string::npos); // untouched
  }

  return ::agpu_test::report("Region");
}
