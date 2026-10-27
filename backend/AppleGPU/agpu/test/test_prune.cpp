// Pruning barriers that order nothing the previous one did not.
#include "agpu/emit/EmitPrune.h"
#include "agpu/msl/Context.h"
#include "harness.h"

using namespace agpu;
using namespace agpu::msl;

namespace {

int barriers(const Block &b) {
  int n = 0;
  visitBlock(
      b, [&](Stmt *s) { n += s->kind == StmtKind::Barrier; }, [](Expr *) {});
  return n;
}

Type tgPtr() {
  return Type::scalar(Scalar::F32).pointerTo(AddrSpace::Threadgroup);
}

Block withShared(Context &c) {
  return Block{c.declStmt(tgPtr(), "p", c.var("pool"))};
}

} // namespace

int main() {
  CASE("a barrier with only register work since the last one goes");
  {
    Context c;
    Block b = withShared(c);
    b.push_back(c.barrier());
    b.push_back(c.declStmt(Context::i32(), "x", c.lit(1)));
    b.push_back(c.barrier());
    pruneRedundantBarriers(b);
    CHECK_EQ(barriers(b), 1);
  }

  CASE("threadgroup memory touched between keeps both");
  {
    Context c;
    Block b = withShared(c);
    b.push_back(c.barrier());
    b.push_back(c.assign(c.deref(c.var("p")), c.litF(1.0)));
    b.push_back(c.barrier());
    pruneRedundantBarriers(b);
    CHECK_EQ(barriers(b), 2);
  }

  CASE("a cast to a threadgroup pointer counts as touching");
  {
    Context c;
    Block b = withShared(c);
    b.push_back(c.barrier());
    b.push_back(c.declStmt(Type::scalar(Scalar::F32), "v",
                           c.deref(c.cast(tgPtr(), c.var("pool")))));
    b.push_back(c.barrier());
    pruneRedundantBarriers(b);
    CHECK_EQ(barriers(b), 2);
  }

  CASE("hard and device barriers are never dropped; a hard one still fences");
  {
    Context c;
    Block hard = withShared(c);
    hard.push_back(c.barrier());
    hard.push_back(c.hardBarrier());
    pruneRedundantBarriers(hard);
    CHECK_EQ(barriers(hard), 2);

    Block after = withShared(c);
    after.push_back(c.hardBarrier());
    after.push_back(c.declStmt(Context::i32(), "x", c.lit(1)));
    after.push_back(c.barrier());
    pruneRedundantBarriers(after);
    CHECK_EQ(barriers(after), 1);

    Block device = withShared(c);
    device.push_back(c.barrier());
    device.push_back(c.barrier(Barrier::Scope::Device));
    pruneRedundantBarriers(device);
    CHECK_EQ(barriers(device), 2);
  }

  CASE("a loop leaving threadgroup memory alone is transparent");
  {
    Context c;
    const auto loop = [&](Stmt *inBody) {
      return c.forStmt(c.declStmt(Context::i32(), "i", c.lit(0)),
                       c.binary(BinOp::Lt, c.var("i"), c.lit(4)),
                       c.assignOp(BinOp::Add, c.var("i"), c.lit(1)),
                       Block{inBody});
    };
    Block quiet = withShared(c);
    quiet.push_back(c.barrier());
    quiet.push_back(loop(c.assignOp(BinOp::Add, c.var("x"), c.lit(1))));
    quiet.push_back(c.barrier());
    pruneRedundantBarriers(quiet);
    CHECK_EQ(barriers(quiet), 1);

    Block busy = withShared(c);
    busy.push_back(c.barrier());
    busy.push_back(loop(c.assign(c.deref(c.var("p")), c.litF(0.0))));
    busy.push_back(c.barrier());
    pruneRedundantBarriers(busy);
    CHECK_EQ(barriers(busy), 2);
  }

  CASE("barriers inside a loop body collapse there, not across the back edge");
  {
    Context c;
    Block b = withShared(c);
    b.push_back(c.forStmt(
        c.declStmt(Context::i32(), "i", c.lit(0)),
        c.binary(BinOp::Lt, c.var("i"), c.lit(4)),
        c.assignOp(BinOp::Add, c.var("i"), c.lit(1)),
        Block{c.barrier(), c.assign(c.deref(c.var("p")), c.litF(0.0)),
              c.barrier(), c.assignOp(BinOp::Add, c.var("x"), c.lit(1)),
              c.barrier()}));
    pruneRedundantBarriers(b);
    CHECK_EQ(barriers(b), 2);
  }

  CASE("an if is entered fenced when its condition leaves shared memory alone");
  {
    Context c;
    Block b = withShared(c);
    b.push_back(c.barrier());
    b.push_back(c.ifStmt(
        c.var("keep"),
        Block{c.barrier(), c.assign(c.deref(c.var("p")), c.litF(0.0))}));
    pruneRedundantBarriers(b);
    CHECK_EQ(barriers(b), 1);

    Block reads = withShared(c);
    reads.push_back(c.barrier());
    reads.push_back(c.ifStmt(
        c.binary(BinOp::Lt, c.deref(c.var("p")), c.litF(0.0)),
        Block{c.barrier(), c.assign(c.deref(c.var("p")), c.litF(0.0))}));
    pruneRedundantBarriers(reads);
    CHECK_EQ(barriers(reads), 2);
  }

  CASE("after an if, fenced only when every path is");
  {
    Context c;
    Block quiet = withShared(c);
    quiet.push_back(c.barrier());
    quiet.push_back(c.ifElse(
        c.var("keep"), Block{c.assignOp(BinOp::Add, c.var("x"), c.lit(1))},
        Block{c.assignOp(BinOp::Add, c.var("y"), c.lit(1))}));
    quiet.push_back(c.barrier());
    pruneRedundantBarriers(quiet);
    CHECK_EQ(barriers(quiet), 1);

    Block oneArm = withShared(c);
    oneArm.push_back(c.barrier());
    oneArm.push_back(c.ifStmt(
        c.var("keep"), Block{c.assign(c.deref(c.var("p")), c.litF(0.0))}));
    oneArm.push_back(c.barrier());
    pruneRedundantBarriers(oneArm);
    CHECK_EQ(barriers(oneArm), 2);

    Block closed = withShared(c);
    closed.push_back(c.barrier());
    closed.push_back(c.ifStmt(
        c.var("keep"),
        Block{c.assign(c.deref(c.var("p")), c.litF(0.0)), c.barrier()}));
    closed.push_back(c.barrier());
    pruneRedundantBarriers(closed);
    CHECK_EQ(barriers(closed), 2);
  }

  return ::agpu_test::report("Prune");
}
