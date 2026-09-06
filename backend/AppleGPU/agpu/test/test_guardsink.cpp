#include "agpu/msl/GuardFuse.h"
#include "agpu/msl/GuardSink.h"
#include "agpu/msl/Printer.h"
#include "harness.h"

#include <sstream>

using namespace agpu;
using namespace agpu::msl;

namespace {

std::string render(const Block &b) {
  std::ostringstream os;
  Printer p(os);
  p.printBlock(b);
  return os.str();
}

std::size_t countOf(const std::string &s, const std::string &needle) {
  std::size_t n = 0;
  for (std::size_t i = s.find(needle); i != std::string::npos;
       i = s.find(needle, i + needle.size()))
    ++n;
  return n;
}

CoordSet literalCoords(Expr *e) {
  if (e && e->kind == ExprKind::Literal) {
    auto *l = static_cast<Literal *>(e);
    if (l->form == Literal::Form::Int)
      return exactCoord((int32_t)l->intValue);
  }
  return unknownCoords();
}

Stmt *guardedStore(Context &c, Expr *cond, const char *buf, int idx,
                   Expr *value) {
  return c.ifStmt(cond,
                  Block{c.assign(c.subscript(c.var(buf), c.lit(idx)), value)});
}

} // namespace

int main() {
  CASE("a store sinks past an unrelated statement and then fuses");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.declStmt(Context::i32(), "tmp", c.lit(7)),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    CHECK_EQ(sinkGuardedStores(b, literalCoords), 1);
    CHECK(b[0]->kind == StmtKind::Decl);
    CHECK(b[1]->kind == StmtKind::If);
    CHECK(b[2]->kind == StmtKind::If);
    fuseGuards(c, b);
    const std::string out = render(b);
    CHECK_EQ(countOf(out, "if"), 1u);
  }

  CASE("(1) a statement that writes the guard's condition blocks the sink");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.assign(c.var("p"), c.lit(0)),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::ClobbersCond);
    CHECK_EQ(sinkGuardedStores(b, literalCoords), 0);
  }

  CASE("(2) a store to an address that may be the same blocks the sink");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.assign(c.subscript(c.var("out"), c.var("i")), c.lit(1)),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::MayAlias);
  }

  CASE("(2b) a provably disjoint store does not block it");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.assign(c.subscript(c.var("out"), c.lit(9)), c.lit(1)),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::Merged);
    CHECK_EQ(sinkGuardedStores(b, literalCoords), 1);
  }

  CASE("(2c) a store to a different buffer does not block it");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.assign(c.subscript(c.var("scratch"), c.var("i")), c.lit(1)),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::Merged);
  }

  CASE("(3) a statement that writes what the store reads blocks the sink");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.assign(c.var("x"), c.lit(1)),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::FeedsStore);
    CHECK_EQ(sinkGuardedStores(b, literalCoords), 0);
  }

  CASE("(4) anything not understood blocks the sink");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.forStmt(
            c.declStmt(Context::i32(), "k", c.lit(0)),
            c.binary(BinOp::Lt, c.var("k"), c.lit(4)),
            c.assign(c.var("k"), c.binary(BinOp::Add, c.var("k"), c.lit(1))),
            Block{c.assign(c.var("acc"), c.lit(0))}),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::Opaque);
  }

  CASE("a guard with no matching partner does not move");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.declStmt(Context::i32(), "tmp", c.lit(7)),
        guardedStore(c, c.var("q"), "out", 4, c.var("z")),
    };
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::Opaque);
    CHECK_EQ(sinkGuardedStores(b, literalCoords), 0);
  }

  CASE("running out of block with nothing to meet is NoTarget");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.declStmt(Context::i32(), "tmp", c.lit(7)),
    };
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::NoTarget);
    CHECK_EQ(sinkGuardedStores(b, literalCoords), 0);
  }

  CASE("an if/else never sinks, because half a move is not a move");
  {
    Context c;
    auto *ie = c.ifStmt(
        c.var("p"),
        Block{c.assign(c.subscript(c.var("out"), c.lit(0)), c.var("x"))});
    ie->elseBody.push_back(c.assign(c.var("other"), c.lit(1)));
    Block b{ie, c.declStmt(Context::i32(), "tmp", c.lit(7)),
            guardedStore(c, c.var("p"), "out", 4, c.var("z"))};
    std::size_t to = 0;
    CHECK(sinkTarget(b, 0, to, literalCoords) == SinkStop::Opaque);
  }

  CASE("already-adjacent guards are left for Fuse.h");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    CHECK_EQ(sinkGuardedStores(b, literalCoords), 0);
    fuseGuards(c, b);
    CHECK_EQ(countOf(render(b), "if"), 1u);
  }

  CASE("a sink across two statements still preserves order of the rest");
  {
    Context c;
    Block b{
        guardedStore(c, c.var("p"), "out", 0, c.var("x")),
        c.declStmt(Context::i32(), "a", c.lit(1)),
        c.declStmt(Context::i32(), "bb", c.lit(2)),
        guardedStore(c, c.var("p"), "out", 4, c.var("z")),
    };
    CHECK_EQ(sinkGuardedStores(b, literalCoords), 1);
    const std::string out = render(b);
    CHECK(out.find("a") < out.find("bb"));
  }

  CASE("a masked store of n registers ends up under a single guard");
  {
    Context c;
    Block b;
    for (int i = 0; i < 8; ++i) {
      b.push_back(guardedStore(c, c.var("mask"), "out", i,
                               c.var("v" + std::to_string(i))));
      b.push_back(
          c.declStmt(Context::i32(), "t" + std::to_string(i), c.lit(i)));
    }
    for (std::size_t pass = 0; pass < b.size(); ++pass)
      if (sinkGuardedStores(b, literalCoords) == 0)
        break;
    fuseGuards(c, b);
    CHECK_EQ(countOf(render(b), "if ("), 1u);
  }

  CASE("fusing alone merges none of them, which is why the sink exists");
  {
    Context c;
    Block b;
    for (int i = 0; i < 8; ++i) {
      b.push_back(guardedStore(c, c.var("mask"), "out", i,
                               c.var("v" + std::to_string(i))));
      b.push_back(
          c.declStmt(Context::i32(), "t" + std::to_string(i), c.lit(i)));
    }
    fuseGuards(c, b);
    CHECK_EQ(countOf(render(b), "if ("), 8u);
  }

  return ::agpu_test::report("GuardSink");
}
