// MSL AST, walk and printer tests.
//
// A B or F prefix on a case names the bug report it was written for, in a
// tracker outside this tree. The sentence after it stands on its own; the tag
// only ties the case back to its report.
#include "agpu/msl/AstWalk.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Equal.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <cstdlib>
#include <limits>
#include <sstream>

using namespace agpu::msl;
using agpu_test::countOf;
using agpu_test::renderType;

namespace {

std::string render(const Block &b) {
  std::ostringstream os;
  Printer p(os);
  p.printBlock(b);
  return os.str();
}

std::string renderExpr(const Expr *e) {
  std::ostringstream os;
  Printer p(os);
  p.printExpr(e);
  return os.str();
}

int countAll(Stmt *s) {
  int n = 0;
  visitStmts(s, [&](Stmt *) { ++n; }, [&](Expr *) { ++n; });
  return n;
}

} // namespace

int main() {
  CASE("B3: an array initialiser is visited");
  {
    Context c;
    auto *arr = c.arrayDecl(Context::i32(), "buf", 2);
    arr->init.push_back(c.var("a"));
    arr->init.push_back(c.var("b"));

    std::vector<std::string> seen;
    visitStmts(
        arr, [](Stmt *) {},
        [&](Expr *e) {
          if (e->kind == ExprKind::VarRef)
            seen.push_back(static_cast<VarRef *>(e)->name);
        });
    CHECK_EQ(seen.size(), 2u);
    CHECK_EQ(seen[0], std::string("a"));
    CHECK_EQ(seen[1], std::string("b"));
  }

  CASE("B7: an if body is descended");
  {
    Context c;
    auto *inner = c.declStmt(Context::i32(), "v", c.lit(1));
    auto *guard = c.ifStmt(c.var("p"), Block{inner});
    int decls = 0;
    visitStmtsOnly(guard, [&](Stmt *s) {
      if (s->kind == StmtKind::Decl)
        ++decls;
    });
    CHECK_EQ(decls, 1);
  }

  CASE("F24: for-loop init and step are children");
  {
    Context c;
    auto *init = c.declStmt(Context::i32(), "i", c.lit(0));
    auto *step = c.assignOp(BinOp::Add, c.var("i"), c.lit(1));
    auto *loop = c.forStmt(init, c.binary(BinOp::Lt, c.var("i"), c.lit(8)),
                           step, Block{c.breakStmt()});
    int decls = 0, stmts = 0;
    visitStmtsOnly(loop, [&](Stmt *s) {
      ++stmts;
      if (s->kind == StmtKind::Decl)
        ++decls;
    });
    CHECK_EQ(decls, 1);
    CHECK_EQ(stmts, 4); // for, init, step, break
  }

  CASE("F24: state-machine case bodies are reached");
  {
    Context c;
    auto *sm = c.stateMachine("state");
    sm->cases.push_back({0, Block{c.declStmt(Context::i32(), "x", c.lit(1))}});
    sm->cases.push_back({1, Block{c.declStmt(Context::i32(), "y", c.lit(2))}});
    int decls = 0;
    visitStmtsOnly(sm, [&](Stmt *s) {
      if (s->kind == StmtKind::Decl)
        ++decls;
    });
    CHECK_EQ(decls, 2);
  }

  CASE("a state machine dispatches");
  {
    Context c;
    auto *sm = c.stateMachine("st", /*entry=*/2);
    sm->cases.push_back({2, Block{c.assign(c.var("st"), c.lit(3))}});
    sm->cases.push_back({3, Block{c.assign(c.var("st"), c.lit(-1))}});

    const std::string out = render(Block{sm});
    CHECK(out.find("int st = 2;") != std::string::npos);
    CHECK(out.find("while (st != -1)") != std::string::npos);
    CHECK(out.find("switch (st)") != std::string::npos);
    CHECK(out.find("int st = 2;") < out.find("while (st != -1)"));
    CHECK(out.find("while (st != -1)") < out.find("switch (st)"));
    CHECK_EQ(countOf(out, "case "), 2);
    CHECK_EQ(countOf(out, "break;"), 3); // two cases plus the default
    CHECK(out.find("default: st = -1; break;") != std::string::npos);
  }

  CASE("nested structures are fully traversed");
  {
    Context c;
    auto *deep =
        c.assign(c.var("out"), c.binary(BinOp::Add, c.var("a"), c.var("b")));
    auto *guard = c.ifStmt(c.var("p"), Block{deep});
    auto *loop =
        c.forStmt(c.declStmt(Context::i32(), "i", c.lit(0)),
                  c.binary(BinOp::Lt, c.var("i"), c.lit(4)),
                  c.assignOp(BinOp::Add, c.var("i"), c.lit(1)), Block{guard});
    CHECK(countAll(loop) > 10);
  }

  CASE("constant folding: offsets collapse to one literal");
  {
    Context c;
    Expr *off = c.add(c.mul(c.lit(1 * 8), c.lit(128)), c.lit(2 * 8));
    CHECK(off->kind == ExprKind::Literal);
    CHECK_EQ(static_cast<Literal *>(off)->intValue, 8 * 128 + 16);
    CHECK_EQ(renderExpr(off), std::string("1040"));
  }

  CASE("identity elimination: x + 0 and x * 1 vanish");
  {
    Context c;
    Expr *x = c.var("x");
    CHECK(c.add(x, c.lit(0)) == x);
    CHECK(c.add(c.lit(0), x) == x);
    CHECK(c.mul(x, c.lit(1)) == x);
    CHECK(c.binary(BinOp::Sub, x, c.lit(0)) == x);
    Expr *z = c.mul(x, c.lit(0));
    CHECK(z->kind == ExprKind::Literal);
    CHECK_EQ(static_cast<Literal *>(z)->intValue, 0);
  }

  CASE("a zero-origin tile emits no offset arithmetic at all");
  {
    Context c;
    Expr *off = c.add(c.mul(c.lit(0), c.lit(128)), c.lit(0));
    CHECK_EQ(renderExpr(off), std::string("0"));
  }

  CASE("comparison folds to a bool literal");
  {
    Context c;
    Expr *e = c.binary(BinOp::Lt, c.lit(3), c.lit(8));
    CHECK(e->kind == ExprKind::Literal);
    CHECK_EQ(renderExpr(e), std::string("true"));
  }

  CASE("a provably-false guard drops its body");
  {
    Context c;
    auto *body = c.assign(c.var("x"), c.lit(1));
    Stmt *g = c.guarded(c.binary(BinOp::Gt, c.lit(1), c.lit(2)), body);
    CHECK(g == nullptr);
    Stmt *t = c.guarded(c.binary(BinOp::Lt, c.lit(1), c.lit(2)), body);
    CHECK(t == body);
    CHECK(c.guarded(nullptr, body) == body);
  }

  CASE("a switch case ends in break, so states do not fall through");
  {
    Context c;
    auto *sm = c.stateMachine("state");
    sm->cases.push_back({0, Block{c.assign(c.var("x"), c.lit(1))}});
    sm->cases.push_back({1, Block{c.assign(c.var("y"), c.lit(2))}});
    const std::string out = render(Block{sm});

    CHECK(out.find("switch (state)") != std::string::npos);
    CHECK(out.find("case 0:") != std::string::npos);
    CHECK(out.find("case 1:") != std::string::npos);
    CHECK_EQ(countOf(out, "break;"), 3); // one per case, plus the default
    CHECK(out.find("break;") < out.find("case 1:"));
  }

  CASE("a case already ending in a transfer gets no unreachable break");
  {
    Context c;
    auto *sm = c.stateMachine("st");
    sm->cases.push_back(
        {0, Block{c.assign(c.var("st"), c.lit(1)), c.continueStmt()}});
    sm->cases.push_back({1, Block{c.assign(c.var("x"), c.lit(2))}});
    const std::string out = render(Block{sm});

    CHECK(out.find("continue;\n") != std::string::npos);
    CHECK(out.find("continue;\n    break;") == std::string::npos);
    CHECK_EQ(countOf(out, "break;"), 2);
  }

  CASE("a pending barrier is flushed before the break");
  {
    Context c;
    auto *sm = c.stateMachine("state");
    sm->cases.push_back(
        {0, Block{c.assign(c.var("x"), c.lit(1)), c.barrier()}});
    const std::string out = render(Block{sm});
    CHECK(out.find("threadgroup_barrier") < out.find("break;"));
  }

  CASE("an address space renders as a single name");
  {
    CHECK_EQ(std::string(spell(AddrSpace::Device)), std::string("device"));
    CHECK_EQ(std::string(spell(AddrSpace::Constant)), std::string("constant"));
    CHECK_EQ(std::string(spell(AddrSpace::None)), std::string(""));

    Context c;
    std::ostringstream os;
    Printer p(os);
    p.printType(Type::scalar(Scalar::F32).pointerTo(AddrSpace::Device));
    CHECK_EQ(os.str(), std::string("device float *"));
  }

  CASE("a barrier is one call whose flags vary");
  {
    Context c;
    Block b;
    b.push_back(c.barrier(Barrier::Scope::Simdgroup));
    b.push_back(c.assign(c.var("x"), c.lit(1)));
    b.push_back(c.barrier(Barrier::Scope::Device));
    b.push_back(c.assign(c.var("y"), c.lit(1)));
    const std::string out = render(b);
    CHECK(out.find("simdgroup_barrier(mem_flags::mem_threadgroup);") !=
          std::string::npos);
    // A device barrier still orders threadgroup memory.
    CHECK(out.find("threadgroup_barrier(mem_flags::mem_threadgroup | "
                   "mem_flags::mem_device);") != std::string::npos);
  }

  CASE("a negated negative does not print as a predecrement");
  {
    Context c;
    CHECK_EQ(renderExpr(c.unary(UnOp::Neg, c.lit(-5))), std::string("-(-5)"));
    CHECK_EQ(renderExpr(c.unary(UnOp::Neg, c.unary(UnOp::Neg, c.var("x")))),
             std::string("-(-x)"));
    CHECK_EQ(renderExpr(c.unary(UnOp::Neg, c.var("x"))), std::string("-x"));
    CHECK_EQ(renderExpr(c.unary(UnOp::Neg, c.lit(5))), std::string("-5"));
    CHECK_EQ(renderExpr(c.unary(UnOp::Not, c.unary(UnOp::Not, c.var("x")))),
             std::string("~~x"));
  }

  CASE("collapsing barriers never narrows the scope");
  {
    using S = Barrier::Scope;
    CHECK(Barrier::widest(S::Simdgroup, S::Threadgroup) == S::Threadgroup);
    CHECK(Barrier::widest(S::Threadgroup, S::Simdgroup) == S::Threadgroup);
    CHECK(Barrier::widest(S::Threadgroup, S::Device) == S::Device);
    CHECK(Barrier::widest(S::Device, S::Simdgroup) == S::Device);
    CHECK(Barrier::widest(S::Simdgroup, S::Simdgroup) == S::Simdgroup);

    Context c;
    Block b;
    b.push_back(c.barrier(S::Simdgroup));
    b.push_back(c.barrier(S::Threadgroup));
    b.push_back(c.assign(c.var("x"), c.lit(1)));
    std::ostringstream os;
    Printer p(os);
    p.printBlock(b);
    const std::string out = os.str();
    CHECK(out.find("threadgroup_barrier") != std::string::npos);
    CHECK(out.find("simdgroup_barrier") == std::string::npos);
  }

  CASE("a statement and a loop header spell the same thing the same way");
  {
    Context c;
    struct Case {
      Stmt *s;
    };
    Stmt *decl = c.declStmt(Context::i32(), "v", c.lit(1));
    Stmt *plain = c.assign(c.var("v"), c.lit(2));
    Stmt *compound = c.assignOp(BinOp::Add, c.var("v"), c.lit(3));
    Stmt *call = c.exprStmt(c.call("f", {c.var("v")}));

    for (Stmt *s : {decl, plain, compound, call}) {
      std::ostringstream header;
      Printer hp(header);
      hp.printHeaderStmt(s);
      CHECK_EQ(render(Block{s}), header.str() + ";\n");
    }
  }

  CASE("only three statement kinds can stand in a loop header");
  {
    Context c;
    CHECK(isHeaderStmt(c.declStmt(Context::i32(), "i", c.lit(0))));
    CHECK(isHeaderStmt(c.assign(c.var("i"), c.lit(1))));
    CHECK(isHeaderStmt(c.exprStmt(c.call("f", {}))));
    CHECK(isHeaderStmt(nullptr)); // an absent clause is legal: for (;;)

    CHECK(!isHeaderStmt(c.barrier()));
    CHECK(!isHeaderStmt(c.breakStmt()));
    CHECK(!isHeaderStmt(c.returnStmt()));
    CHECK(!isHeaderStmt(c.scope(Block{})));
  }

  CASE("a hard barrier never merges, in either direction");
  {
    Context c;
    Block b;
    b.push_back(c.hardBarrier());
    b.push_back(c.hardBarrier());
    b.push_back(c.assign(c.var("x"), c.lit(1)));
    CHECK_EQ(countOf(render(b), "threadgroup_barrier"), 2);

    Block soft;
    soft.push_back(c.barrier());
    soft.push_back(c.hardBarrier());
    soft.push_back(c.assign(c.var("x"), c.lit(1)));
    CHECK_EQ(countOf(render(soft), "threadgroup_barrier"), 2);

    Block after;
    after.push_back(c.hardBarrier());
    after.push_back(c.barrier());
    after.push_back(c.assign(c.var("x"), c.lit(1)));
    CHECK_EQ(countOf(render(after), "threadgroup_barrier"), 2);
  }

  CASE("soft barriers still merge, so hardness is not the default");
  {
    Context c;
    Block b;
    b.push_back(c.barrier());
    b.push_back(c.barrier());
    b.push_back(c.barrier());
    b.push_back(c.assign(c.var("x"), c.lit(1)));
    CHECK_EQ(countOf(render(b), "threadgroup_barrier"), 1);
  }

  CASE("a hard barrier keeps its own scope when fused with a softer one");
  {
    using S = Barrier::Scope;
    Context c;
    Block b;
    b.push_back(c.hardBarrier(S::Simdgroup));
    b.push_back(c.barrier(S::Device));
    b.push_back(c.assign(c.var("x"), c.lit(1)));
    const std::string out = render(b);
    CHECK(out.find("simdgroup_barrier") != std::string::npos);
    CHECK(out.find("mem_device") != std::string::npos);
  }

  CASE("guardedInto splices an unguarded block flat into the parent");
  {
    Context c;
    Block inner;
    inner.push_back(c.assign(c.var("x"), c.lit(1)));
    inner.push_back(c.assign(c.var("y"), c.lit(2)));

    Block out;
    c.guardedInto(out, nullptr, inner);
    CHECK_EQ(out.size(), 2u);

    Block guardedOut;
    c.guardedInto(guardedOut, c.binary(BinOp::Eq, c.var("w"), c.lit(0)), inner);
    CHECK_EQ(guardedOut.size(), 1u);

    Block dropped;
    c.guardedInto(dropped, c.binary(BinOp::Gt, c.lit(1), c.lit(2)), inner);
    CHECK_EQ(dropped.size(), 0u);

    Block always;
    c.guardedInto(always, c.binary(BinOp::Lt, c.lit(1), c.lit(2)), inner);
    CHECK_EQ(always.size(), 2u);
  }

  CASE("parens appear only where precedence requires");
  {
    Context c;
    Expr *a = c.var("a"), *b = c.var("b"), *d = c.var("d");
    CHECK_EQ(renderExpr(c.add(a, c.mul(b, d))), std::string("a + b * d"));
    CHECK_EQ(renderExpr(c.mul(c.add(a, b), d)), std::string("(a + b) * d"));
  }

  CASE("left-associativity is preserved on the right operand");
  {
    Context c;
    Expr *a = c.var("a"), *b = c.var("b"), *d = c.var("d");
    CHECK_EQ(renderExpr(c.binary(BinOp::Sub, a, c.binary(BinOp::Sub, b, d))),
             std::string("a - (b - d)"));
    CHECK_EQ(renderExpr(c.binary(BinOp::Sub, c.binary(BinOp::Sub, a, b), d)),
             std::string("a - b - d"));
  }

  CASE("logical chains read without noise");
  {
    Context c;
    Expr *r = c.var("row");
    Expr *lo = c.binary(BinOp::Ge, r, c.lit(16));
    Expr *hi = c.binary(BinOp::Lt, r, c.lit(32));
    CHECK_EQ(renderExpr(c.binary(BinOp::LAnd, lo, hi)),
             std::string("row >= 16 && row < 32"));
  }

  CASE("a small kernel body");
  {
    Context c;
    Block b;
    b.push_back(c.declStmt(Context::i32(), "v0", c.lit(7)));
    b.push_back(c.assignOp(BinOp::Add, c.var("v0"), c.lit(1)));
    b.push_back(c.ifStmt(c.binary(BinOp::Lt, c.var("v0"), c.lit(32)),
                         Block{c.assign(c.var("v0"), c.lit(0))}));
    CHECK_EQ(render(b), std::string("int v0 = 7;\n"
                                    "v0 += 1;\n"
                                    "if (v0 < 32) v0 = 0;\n"));
  }

  CASE("a braced if with an else");
  {
    Context c;
    Block thenB{c.declStmt(Context::i32(), "a", c.lit(1)),
                c.assign(c.var("a"), c.lit(2))};
    Block elseB{c.assign(c.var("a"), c.lit(3))};
    Block b{c.ifElse(c.var("p"), thenB, elseB)};
    CHECK_EQ(render(b), std::string("if (p) {\n"
                                    "  int a = 1;\n"
                                    "  a = 2;\n"
                                    "} else {\n"
                                    "  a = 3;\n"
                                    "}\n"));
  }

  CASE("a loop prints its full header");
  {
    Context c;
    auto *loop =
        c.forStmt(c.declStmt(Context::i32(), "k", c.lit(0)),
                  c.binary(BinOp::Lt, c.var("k"), c.lit(32)),
                  c.assignOp(BinOp::Add, c.var("k"), c.lit(8)),
                  Block{c.exprStmt(c.call("simdgroup_multiply_accumulate",
                                          {c.var("acc"), c.var("fa"),
                                           c.var("fb"), c.var("acc")}))});
    CHECK_EQ(render(Block{loop}),
             std::string("for (int k = 0; k < 32; k += 8) {\n"
                         "  simdgroup_multiply_accumulate(acc, fa, fb, acc);\n"
                         "}\n"));
  }

  CASE("pointer types and address spaces");
  {
    Context c;
    Type tg = Type::scalar(Scalar::F32).pointerTo(AddrSpace::Threadgroup);
    Block b{c.declStmt(tg, "pA", c.var("pool"))};
    CHECK_EQ(render(b), std::string("threadgroup float * pA = pool;\n"));
  }

  CASE("adjacent barriers collapse, device scope wins");
  {
    Context c;
    Block b{c.barrier(), c.barrier(), c.assign(c.var("x"), c.lit(1))};
    CHECK_EQ(render(b),
             std::string("threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                         "x = 1;\n"));

    Block d{c.barrier(), c.barrier(Barrier::Scope::Device),
            c.assign(c.var("x"), c.lit(1))};
    CHECK_EQ(render(d),
             std::string("threadgroup_barrier(mem_flags::mem_threadgroup | "
                         "mem_flags::mem_device);\n"
                         "x = 1;\n"));
  }

  CASE("a struct declaration renders as a proper AST node");
  {
    Context c;
    auto *sd = c.structDecl();
    sd->name = "fn_dot_ret";
    sd->fields.push_back({Context::f32(), "f0"});
    sd->fields.push_back({Context::f32(), "f1"});
    CHECK_EQ(render(Block{sd}), std::string("struct fn_dot_ret {\n"
                                            "  float f0;\n"
                                            "  float f1;\n"
                                            "};\n"));
  }

  CASE("a kernel function with parameters and attributes");
  {
    Context c;
    auto *f = c.function();
    f->isKernel = true;
    f->name = "dot_kernel";
    f->returnType = Type::named("void");
    f->params.push_back({Type::scalar(Scalar::F32).pointerTo(AddrSpace::Device),
                         "A", Attribute::buffer(0)});
    f->params.push_back(
        {Type::vector(Scalar::U32, 3), "tid",
         Attribute::builtin(Attribute::Kind::ThreadPositionInThreadgroup)});
    f->body.push_back(c.returnStmt());
    CHECK_EQ(
        render(Block{f}),
        std::string("kernel void dot_kernel(device float * A [[buffer(0)]], "
                    "uint3 tid [[thread_position_in_threadgroup]]) {\n"
                    "  return;\n"
                    "}\n"));
  }

  CASE("unsigned literals carry their suffix");
  {
    Context c;
    Expr *m = c.lit(31, Context::u32());
    CHECK_EQ(renderExpr(m), std::string("31u"));
    CHECK_EQ(renderExpr(c.lit(31)), std::string("31"));
  }

  CASE("float literals always look like floats");
  {
    Context c;
    CHECK_EQ(renderExpr(c.litF(1.0)), std::string("1.0f"));
    CHECK_EQ(renderExpr(c.litF(0.5)), std::string("0.5f"));
  }

  CASE("the non-finite values are named, since their digits are not MSL");
  {
    // The platform spelling of inf/nan is not valid MSL; these come from
    // metal_stdlib.
    Context c;
    const double inf = std::numeric_limits<double>::infinity();
    CHECK_EQ(renderExpr(c.litF(inf)), std::string("INFINITY"));
    CHECK_EQ(renderExpr(c.litF(-inf)), std::string("(-INFINITY)"));
    CHECK_EQ(renderExpr(c.litF(std::numeric_limits<double>::quiet_NaN())),
             std::string("NAN"));
    CHECK(renderExpr(c.litF(inf)).find('f') == std::string::npos);
  }

  CASE("a literal is spelled with the fewest digits that read back exactly");
  {
    Context c;
    CHECK_EQ(renderExpr(c.litF(0.1)), std::string("0.1f"));
    CHECK_EQ(renderExpr(c.litF(0.25)), std::string("0.25f"));

    // Widening stops as soon as it round-trips at the literal's own width.
    for (double v : {0.1, 1.0 / 3.0, 1e-20, 123456792.0, 3.14159265358979}) {
      const std::string s = renderExpr(c.litF(v));
      CHECK_EQ((float)std::strtod(s.c_str(), nullptr), (float)v);
    }
  }

  CASE("a cast says what it means");
  {
    // `(half)x` converts, `as_type<half>(x)` reinterprets.
    Context c;
    Expr *v = c.var("x");
    const Type h = Type::scalar(Scalar::F16);
    CHECK_EQ(renderExpr(c.cast(h, v)), std::string("(half)x"));
    CHECK_EQ(renderExpr(c.bitcast(h, v)), std::string("as_type<half>(x)"));
    CHECK_EQ(renderExpr(c.construct(h, v)), std::string("half(x)"));
    CHECK_EQ(renderExpr(c.cast(h, v, Cast::Style::Static)),
             std::string("static_cast<half>(x)"));
  }

  CASE("a pointer carries a set of qualifiers");
  {
    Context c;
    const Type f = Type::scalar(Scalar::F32);
    CHECK_EQ(renderType(f.pointerTo(AddrSpace::Device)),
             std::string("device float *"));
    CHECK_EQ(renderType(f.pointerTo(AddrSpace::Device, Type::Coherent)),
             std::string("device coherent(device) float *"));
    CHECK_EQ(renderType(f.pointerTo(AddrSpace::Device, Type::Volatile)),
             std::string("volatile device float *"));
    CHECK_EQ(renderType(f.pointerTo(AddrSpace::Device, Type::Const)),
             std::string("device const float *"));

    // Grammar order: volatile before the address space, coherent after it,
    // const last.
    CHECK_EQ(renderType(f.pointerTo(AddrSpace::Device,
                                    Type::Volatile | Type::Coherent)),
             std::string("volatile device coherent(device) float *"));

    CHECK_EQ(renderType(f.pointerTo(AddrSpace::Threadgroup, Type::Coherent)),
             std::string("threadgroup coherent(threadgroup) float *"));
  }

  CASE("qualifiers are part of a pointer type's identity");
  {
    const Type f = Type::scalar(Scalar::F32);
    CHECK(!(f.pointerTo(AddrSpace::Device) ==
            f.pointerTo(AddrSpace::Device, Type::Coherent)));
    CHECK(!(f.pointerTo(AddrSpace::Device, Type::Volatile) ==
            f.pointerTo(AddrSpace::Device, Type::Coherent)));
    CHECK(f.pointerTo(AddrSpace::Device, Type::Coherent) ==
          f.pointerTo(AddrSpace::Device, Type::Coherent));
  }

  CASE("a matrix type equals itself");
  {
    const Type a = Type::matrix("simdgroup_float8x8");
    const Type b = Type::matrix("simdgroup_float8x8");
    CHECK(a == b);
    CHECK(!(a == Type::matrix("simdgroup_half8x8")));
    CHECK(!(a == Type::named("simdgroup_float8x8")));
  }

  CASE("two casts of one type but different styles are not equal");
  {
    Context c;
    const Type h = Type::scalar(Scalar::F16);
    CHECK(!exprsEqual(c.cast(h, c.var("x")), c.bitcast(h, c.var("x"))));
    CHECK(exprsEqual(c.bitcast(h, c.var("x")), c.bitcast(h, c.var("x"))));
  }

  CASE("an initialised table declares the length its contents have");
  {
    Context c;
    SmallVec<Expr *, 4> vals{c.lit(1), c.lit(2), c.lit(3)};
    ArrayDecl *d = c.arrayDecl(Context::i32(), "tbl", std::move(vals));
    CHECK_EQ(d->count, 3);
    CHECK_EQ(render(Block{d}), std::string("int tbl[3] = {1, 2, 3};\n"));
  }

  CASE("every node kind is reachable from the table");
  {
    CHECK_EQ(static_cast<int>(ExprKind::AddrOf) + 1, 11);
    CHECK_EQ(static_cast<int>(StmtKind::StructDecl) + 1, 15);
  }

  return ::agpu_test::report("MSL");
}
