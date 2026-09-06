// Analysis tests.
//
// A B or F prefix on a case names the bug report it was written for, in a
// tracker outside this tree. The sentence after it stands on its own; the tag
// only ties the case back to its report.
#include "agpu/msl/Analysis.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Printer.h"
#include "harness.h"

#include <sstream>

using namespace agpu::msl;

namespace {
std::string render(const Block &b) {
  std::ostringstream os;
  Printer p(os);
  p.printBlock(b);
  return os.str();
}
} // namespace

int main() {
  CASE("B3: a name used only in an array initialiser stays live");
  {
    Context c;
    Block b;
    b.push_back(c.declStmt(Context::i32(), "seed", c.lit(3)));
    auto *arr = c.arrayDecl(Context::i32(), "tbl", 2);
    arr->init.push_back(c.var("seed"));
    arr->init.push_back(c.lit(0));
    b.push_back(arr);
    b.push_back(c.assign(c.var("out"), c.var("tbl")));

    auto dead = findDeadDecls(b);
    CHECK_EQ(dead.size(), 0u);
  }

  CASE("B7: declarations inside an if body are counted");
  {
    Context c;
    Block b{
        c.ifStmt(c.var("p"), Block{c.declStmt(Context::i32(), "a", c.lit(1)),
                                   c.declStmt(Context::i32(), "b", c.lit(2))})};
    FuncSize s = measure(b);
    CHECK_EQ(s.decls, 2);
    CHECK_EQ(s.branches, 1);
  }

  CASE("F24: a loop's init declaration is counted");
  {
    Context c;
    Block b{c.forStmt(c.declStmt(Context::i32(), "k", c.lit(0)),
                      c.binary(BinOp::Lt, c.var("k"), c.lit(8)),
                      c.assignOp(BinOp::Add, c.var("k"), c.lit(1)),
                      Block{c.declStmt(Context::i32(), "t", c.lit(0))})};
    FuncSize s = measure(b);
    CHECK_EQ(s.decls, 2);
    CHECK_EQ(s.loops, 1);
  }

  CASE("F24: state-machine case bodies are analysed");
  {
    Context c;
    auto *sm = c.stateMachine("state");
    sm->cases.push_back({0, Block{c.declStmt(Context::i32(), "x", c.lit(1))}});
    sm->cases.push_back({1, Block{c.declStmt(Context::i32(), "y", c.lit(2)),
                                  c.assign(c.var("out"), c.var("y"))}});
    Block b{sm};
    FuncSize s = measure(b);
    CHECK_EQ(s.decls, 2);
    auto dead = findDeadDecls(b);
    CHECK_EQ(dead.size(), 1u);
    CHECK_EQ(static_cast<Decl *>(dead[0])->name, std::string("x"));
  }

  CASE("a plain assignment does not read its target and dies with it");
  {
    Context c;
    Block b{c.declStmt(Context::i32(), "v", c.lit(0)),
            c.assign(c.var("v"), c.lit(1))};
    auto dead = findDeadDecls(b);
    CHECK_EQ(dead.size(), 2u);

    PtrSet<Stmt *> drop;
    for (Stmt *s : dead)
      drop.insert(s);
    eraseStmts(b, drop);
    CHECK(b.empty());
  }

  CASE("a compound assignment does read its target");
  {
    Context c;
    Block b{c.declStmt(Context::i32(), "v", c.lit(0)),
            c.assignOp(BinOp::Add, c.var("v"), c.lit(1)),
            c.assign(c.var("out"), c.var("v"))};
    CHECK_EQ(findDeadDecls(b).size(), 0u);

    Block cyc{c.declStmt(Context::i32(), "w", c.lit(0)),
              c.assignOp(BinOp::Add, c.var("w"), c.lit(1))};
    CHECK_EQ(findDeadDecls(cyc).size(), 2u);
  }

  CASE("a subscripted or dereferenced target reads its base");
  {
    Context c;
    Block sub{c.declStmt(Context::i32(), "p", c.lit(0)),
              c.assign(c.subscript(c.var("p"), c.lit(2)), c.lit(1))};
    CHECK_EQ(findDeadDecls(sub).size(), 0u);

    Block der{c.declStmt(Context::i32(), "q", c.lit(0)),
              c.assign(c.deref(c.var("q")), c.lit(1))};
    CHECK_EQ(findDeadDecls(der).size(), 0u);
  }

  CASE("reads through calls, casts and ternaries all count");
  {
    Context c;
    Block b{
        c.declStmt(Context::i32(), "a", c.lit(1)),
        c.declStmt(Context::i32(), "b", c.lit(2)),
        c.declStmt(Context::i32(), "d", c.lit(3)),
        c.exprStmt(c.call("f", {c.var("a")})),
        c.assign(c.var("o1"), c.cast(Context::f32(), c.var("b"))),
        c.assign(c.var("o2"), c.ternary(c.var("p"), c.var("d"), c.lit(0))),
    };
    CHECK_EQ(findDeadDecls(b).size(), 0u);
  }

  CASE("nesting does not hide a read");
  {
    Context c;
    Block b{
        c.declStmt(Context::i32(), "deep", c.lit(1)),
        c.forStmt(c.declStmt(Context::i32(), "i", c.lit(0)),
                  c.binary(BinOp::Lt, c.var("i"), c.lit(4)),
                  c.assignOp(BinOp::Add, c.var("i"), c.lit(1)),
                  Block{c.ifStmt(c.var("p"),
                                 Block{c.assign(c.var("o"), c.var("deep"))})})};
    CHECK_EQ(findDeadDecls(b).size(), 0u);
  }

  CASE("an array of fragments counts as its element count");
  {
    Context c;
    Block b{c.arrayDecl(Type::matrix("simdgroup_float8x8"), "acc", 16)};
    FuncSize s = measure(b);
    CHECK_EQ(s.decls, 16);
  }

  CASE("a threadgroup array is not allocas and does not count");
  {
    Context c;
    Type pool = Type::scalar(Scalar::I8).inAddrSpace(AddrSpace::Threadgroup);
    Block b{c.arrayDecl(pool, "pool", 32768),
            c.declStmt(Context::i32(), "i0", c.lit(0))};
    FuncSize s = measure(b);
    CHECK_EQ(s.decls, 1);

    Block t{c.arrayDecl(Context::i32(), "tmp", 64)};
    CHECK_EQ(measure(t).decls, 64);
  }

  CASE("matrix declarations are tracked separately");
  {
    Context c;
    Block b{c.declStmt(Type::matrix("simdgroup_float8x8"), "f0"),
            c.declStmt(Context::i32(), "i0", c.lit(0))};
    FuncSize s = measure(b);
    CHECK_EQ(s.decls, 2);
    CHECK_EQ(s.fragDecls, 1);
  }

  CASE("an opaque type that is not a fragment does not count as one");
  {
    Context c;
    Block b{c.declStmt(Type::named("callee_ret"), "r"),
            c.declStmt(Type::named("atomic_uint").pointerTo(AddrSpace::Device),
                       "p"),
            c.declStmt(Type::matrix("simdgroup_float8x8"), "f0")};
    FuncSize s = measure(b);
    CHECK_EQ(s.decls, 3);
    CHECK_EQ(s.fragDecls, 1);
  }

  CASE("MMA calls are counted, being the other side of the rolling trade");
  {
    Context c;
    Block b;
    for (int i = 0; i < 3; ++i)
      b.push_back(
          c.exprStmt(c.call(builtin::sg::MultiplyAccumulate,
                            {c.var("a"), c.var("b"), c.var("d"), c.var("a")})));
    b.push_back(c.exprStmt(c.call("something_else", {c.var("x")})));
    FuncSize s = measure(b);
    CHECK_EQ(s.mma, 3);
    CHECK_EQ(s.stmts, 4);
  }

  CASE("barriers and loops are counted at depth");
  {
    Context c;
    Block b{c.forStmt(c.declStmt(Context::i32(), "k", c.lit(0)),
                      c.binary(BinOp::Lt, c.var("k"), c.lit(4)),
                      c.assignOp(BinOp::Add, c.var("k"), c.lit(1)),
                      Block{c.barrier(), c.whileStmt(c.var("p"), Block{})})};
    FuncSize s = measure(b);
    CHECK_EQ(s.loops, 2);
    CHECK_EQ(s.barriers, 1);
  }

  CASE("erasing dead declarations changes the emitted text");
  {
    Context c;
    Block b{c.declStmt(Context::i32(), "keep", c.lit(1)),
            c.declStmt(Context::i32(), "drop", c.lit(2)),
            c.assign(c.var("out"), c.var("keep"))};

    auto dead = findDeadDecls(b);
    CHECK_EQ(dead.size(), 1u);
    PtrSet<Stmt *> kill(dead.begin(), dead.end());
    eraseStmts(b, kill);

    CHECK_EQ(render(b), std::string("int keep = 1;\n"
                                    "out = keep;\n"));
  }

  CASE("erasure reaches into nested blocks");
  {
    Context c;
    auto *inner = c.declStmt(Context::i32(), "gone", c.lit(9));
    Block b{c.ifStmt(c.var("p"), Block{inner, c.assign(c.var("o"), c.lit(1))})};
    auto dead = findDeadDecls(b);
    CHECK_EQ(dead.size(), 1u);
    PtrSet<Stmt *> kill(dead.begin(), dead.end());
    eraseStmts(b, kill);
    CHECK_EQ(render(b), std::string("if (p) o = 1;\n"));
  }

  CASE("a chain of dead decls is found in a single call");
  {
    Context c;
    Block blk{c.declStmt(Context::i32(), "b", c.lit(1)),
              c.declStmt(Context::i32(), "a", c.var("b"))};
    auto dead = findDeadDecls(blk);
    CHECK_EQ(dead.size(), 2u);

    PtrSet<Stmt *> kill(dead.begin(), dead.end());
    eraseStmts(blk, kill);
    CHECK(blk.empty());
  }

  CASE("a long chain converges at any depth");
  {
    Context c;
    Block blk;
    blk.push_back(c.declStmt(Context::i32(), "v0", c.lit(1)));
    for (int i = 1; i < 40; ++i)
      blk.push_back(c.declStmt(Context::i32(), "v" + std::to_string(i),
                               c.var("v" + std::to_string(i - 1))));
    auto dead = findDeadDecls(blk);
    CHECK_EQ(dead.size(), 40u);
  }

  CASE("what a statement writes is asked in one place");
  {
    Context c;
    bool escapes = false;

    CHECK_EQ(writtenName(c.assign(c.var("v"), c.lit(1)), escapes),
             std::string("v"));
    CHECK(!escapes);

    writtenName(c.assign(c.deref(c.var("p")), c.lit(1)), escapes);
    CHECK(escapes);

    CHECK_EQ(
        writtenName(c.assign(c.subscript(c.var("a"), c.var("i")), c.lit(1)),
                    escapes),
        std::string("a"));
    CHECK(!escapes);

    Block store{c.assign(c.subscript(c.var("a"), c.var("i")), c.lit(1))};
    CHECK(writesTo(store[0], PtrSet<Str>{"a"}));
    CHECK(!writesTo(store[0], PtrSet<Str>{"p"}));

    CHECK_EQ(writtenName(c.declStmt(Context::i32(), "v", c.lit(0)), escapes),
             std::string("v"));
  }

  CASE("a compound assignment both reads and writes its target");
  {
    Context c;
    Block b{c.assignOp(BinOp::Add, c.var("x"), c.lit(1))};
    CHECK(collectReads(b).count("x") == 1);

    PtrSet<Str> names;
    names.insert("x");
    CHECK(writesTo(b[0], names));
  }

  CASE("a plain assignment writes without reading");
  {
    Context c;
    Block b{c.assign(c.var("x"), c.lit(1))};
    CHECK(collectReads(b).count("x") == 0);

    PtrSet<Str> names;
    names.insert("x");
    CHECK(writesTo(b[0], names));
  }

  CASE("an unmodelled statement is assumed to write everything");
  {
    Context c;
    PtrSet<Str> names;
    names.insert("anything");
    CHECK(writesTo(c.exprStmt(c.call("f", {})), names));
    CHECK(writesTo(c.barrier(), names));

    CHECK(!writesTo(c.scope(Block{c.assign(c.var("other"), c.lit(1))}), names));
    CHECK(
        writesTo(c.scope(Block{c.assign(c.var("anything"), c.lit(1))}), names));
  }

  CASE("a declaration whose initialiser calls something is not dead");
  {
    Context c;
    Block b{c.declStmt(
        Context::f32(), "unused",
        c.call("__agpu_atomic_rmw_f32", {c.var("p"), c.var("v"), c.lit(0)}))};
    auto dead = findDeadDecls(b);
    CHECK(dead.empty());
  }

  CASE("a call nested inside a larger initialiser still protects it");
  {
    Context c;
    Block b{
        c.declStmt(Context::f32(), "unused",
                   c.binary(BinOp::Add, c.lit(1), c.call("f", {c.var("x")})))};
    CHECK(findDeadDecls(b).empty());
  }

  CASE("a pure initialiser is still eliminated");
  {
    Context c;
    Block b{c.declStmt(Context::i32(), "unused",
                       c.binary(BinOp::Add, c.var("x"), c.lit(1)))};
    CHECK_EQ(findDeadDecls(b).size(), 1u);
  }

  CASE("an assignment of a call result survives its dead target");
  {
    Context c;
    Block b{c.declStmt(Context::i32(), "v", c.lit(0)),
            c.assign(c.var("v"), c.call("f", {c.var("x")}))};
    auto dead = findDeadDecls(b);
    CHECK(dead.empty());
  }

  CASE("deletion reaches every Block the node table declares");
  {
    Context c;
    auto *victim = c.assign(c.var("x"), c.lit(1));

    auto *ifStmt = c.ifStmt(c.var("p"), Block{victim});
    auto *forStmt =
        c.forStmt(c.declStmt(Context::i32(), "i", c.lit(0)), c.var("p"),
                  c.assign(c.var("i"), c.lit(1)), Block{victim});
    auto *whileStmt = c.whileStmt(c.var("p"), Block{victim});
    auto *scope = c.scope(Block{victim});
    auto *sm = c.stateMachine("s");
    sm->cases.push_back({0, Block{victim}});

    PtrSet<Stmt *> drop;
    drop.insert(victim);
    for (Stmt *host : {(Stmt *)ifStmt, (Stmt *)forStmt, (Stmt *)whileStmt,
                       (Stmt *)scope, (Stmt *)sm}) {
      Block body{host};
      eraseStmts(body, drop);
      int survivors = 0;
      visitStmtsOnly(body[0], [&](Stmt *s) {
        if (s == victim)
          ++survivors;
      });
      CHECK_EQ(survivors, 0);
    }

    auto *both = c.ifElse(c.var("p"), Block{c.breakStmt()}, Block{victim});
    Block body{both};
    eraseStmts(body, drop);
    int survivors = 0;
    visitStmtsOnly(body[0], [&](Stmt *s) {
      if (s == victim)
        ++survivors;
    });
    CHECK_EQ(survivors, 0);
  }

  CASE("an unused fragment is not eliminated and that is not a bug");
  {
    Context c;
    Block b;
    b.push_back(c.declStmt(Type::matrix("simdgroup_half8x8"), "f0"));
    b.push_back(c.exprStmt(
        c.call("simdgroup_load", {c.var("f0"), c.var("pA"), c.lit(8)})));
    b.push_back(c.assign(c.var("out"), c.lit(1)));

    CHECK_EQ(measure(b).fragDecls, 1);
    CHECK_EQ(findDeadDecls(b).size(), 0u);

    Block plain;
    plain.push_back(c.declStmt(Context::i32(), "unused", c.lit(1)));
    plain.push_back(c.assign(c.var("out"), c.lit(1)));
    CHECK_EQ(findDeadDecls(plain).size(), 1u);
  }

  CASE("an induction cycle nothing observes is dead");
  {
    Context c;
    Block b;
    b.push_back(c.declStmt(Context::i32(), "i", c.lit(0)));
    b.push_back(
        c.assign(c.var("i"), c.binary(BinOp::Add, c.var("i"), c.lit(1))));
    b.push_back(c.assign(c.var("out"), c.lit(1)));

    CHECK_EQ(findDeadDecls(b).size(), 2u);
  }

  CASE("a cycle with one outside reader is entirely live");
  {
    Context c;
    Block b;
    b.push_back(c.declStmt(Context::i32(), "i", c.lit(0)));
    b.push_back(
        c.assign(c.var("i"), c.binary(BinOp::Add, c.var("i"), c.lit(1))));
    b.push_back(c.assign(c.var("out"), c.var("i")));

    CHECK_EQ(findDeadDecls(b).size(), 0u);
  }

  CASE("a cycle whose assignment has an effect stays, value dead or not");
  {
    Context c;
    Block b;
    b.push_back(c.declStmt(Context::i32(), "i", c.lit(0)));
    b.push_back(c.assign(c.var("i"), c.binary(BinOp::Add, c.var("i"),
                                              c.call("atomic_fetch_add", {}))));
    b.push_back(c.assign(c.var("out"), c.lit(1)));

    CHECK_EQ(findDeadDecls(b).size(), 0u);
  }

  CASE("a loop induction variable is not a dead cycle");
  {
    Context c;
    Block b;
    b.push_back(c.forStmt(
        c.declStmt(Context::i32(), "i", c.lit(0)),
        c.binary(BinOp::Lt, c.var("i"), c.var("n")),
        c.assign(c.var("i"), c.binary(BinOp::Add, c.var("i"), c.lit(1))),
        Block{c.assign(c.var("out"), c.lit(1))}));
    CHECK_EQ(findDeadDecls(b).size(), 0u);
  }

  CASE("a two-variable cycle is dead as a whole");
  {
    Context c;
    Block b;
    b.push_back(c.declStmt(Context::i32(), "a", c.lit(0)));
    b.push_back(c.declStmt(Context::i32(), "b", c.lit(0)));
    b.push_back(c.assign(c.var("a"), c.var("b")));
    b.push_back(c.assign(c.var("b"), c.var("a")));
    b.push_back(c.assign(c.var("out"), c.lit(1)));

    CHECK_EQ(findDeadDecls(b).size(), 4u);
  }

  return ::agpu_test::report("Analysis");
}
