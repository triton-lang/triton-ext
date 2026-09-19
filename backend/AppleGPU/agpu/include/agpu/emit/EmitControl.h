// EmitControl.h - structured control flow and the SSA it has to unmake.
//
// MSL has no block arguments, so every region result becomes a mutable
// variable declared before the construct and assigned on each path out.
#ifndef AGPU_EMIT_CONTROL_H
#define AGPU_EMIT_CONTROL_H

#include "agpu/core/Decline.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/Elementwise.h"

namespace agpu {

// One value crossing a region boundary.
struct CarriedValue {
  msl::SmallVec<msl::Str, 8> regs;
  ElemType elem = i32();
};

using Carried = msl::SmallVec<CarriedValue, 4>;

// Before the construct: a variable declared inside a region is out of scope
// after it.
inline void declareResults(msl::Context &c, msl::Block &body,
                           const Carried &results) {
  for (const CarriedValue &v : results)
    for (const msl::Str &r : v.regs)
      body.push_back(c.declStmt(mslTypeOf(v.elem), r, nullptr));
}

// `dst = src;` for one register pair.
inline void assignReg(msl::Context &c, msl::Block &arm, const msl::Str &dst,
                      const msl::Str &src) {
  arm.push_back(c.assign(c.var(dst), c.var(src)));
}

// False when the counts disagree: the IR is malformed.
inline bool emitYield(msl::Context &c, msl::Block &arm, const Carried &results,
                      const Carried &yielded) {
  if (results.size() != yielded.size())
    return false;
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results[i].regs.size() != yielded[i].regs.size())
      return false;
    for (std::size_t r = 0; r < results[i].regs.size(); ++r)
      assignReg(c, arm, results[i].regs[r], yielded[i].regs[r]);
  }
  return true;
}

inline Decision emitIf(msl::Context &c, msl::Block &body, const msl::Str &cond,
                       const Carried &results, msl::Block thenArm,
                       const Carried &thenYield, bool hasElse,
                       msl::Block elseArm, const Carried &elseYield) {
  declareResults(c, body, results);

  if (!emitYield(c, thenArm, results, thenYield))
    return Decision::failed();

  if (!hasElse) {
    body.push_back(c.ifStmt(c.var(cond), std::move(thenArm)));
    return Decision::emitted();
  }

  if (!emitYield(c, elseArm, results, elseYield))
    return Decision::failed();
  body.push_back(c.ifElse(c.var(cond), std::move(thenArm), std::move(elseArm)));
  return Decision::emitted();
}

struct LoopBounds {
  msl::Str iv = "i";

  // A GEMM K-loop's bounds are runtime SSA values. Null means unset.
  msl::Expr *lo = nullptr;
  msl::Expr *hi = nullptr;
  msl::Expr *step = nullptr;

  // AGX computes an i64 induction variable in the Gauss-sum closed form, at
  // i65 intermediate width and gets it wrong. `wideIv` counts iterations in
  // a narrow counter and derives the value:
  //
  //     for (int tc = 0; ; ++tc) { long iv = lo + tc*step;
  //                                if (!(iv < hi)) break; ... }
  bool wideIv = false;

  ElemType ivType() const {
    return ElemType{ElemType::Kind::Int, wideIv ? 64u : 32u, false};
  }
};

struct LoopExprs {
  msl::Expr *lo;
  msl::Expr *hi;
  msl::Expr *step;
};

inline LoopExprs boundsOf(msl::Context &c, const LoopBounds &b) {
  return {b.lo ? b.lo : c.lit(0), b.hi ? b.hi : c.lit(0),
          b.step ? b.step : c.lit(1)};
}

inline LoopBounds constBounds(msl::Context &c, msl::Str iv, int64_t lo,
                              int64_t hi, int64_t step = 1,
                              bool wideIv = false) {
  LoopBounds b;
  b.iv = std::move(iv);
  b.lo = c.lit(lo);
  b.hi = c.lit(hi);
  b.step = c.lit(step);
  b.wideIv = wideIv;
  return b;
}

inline Decision emitFor(msl::Context &c, msl::Block &body, const LoopBounds &b,
                        const Carried &carried, const Carried &inits,
                        msl::Block loopBody, const Carried &yielded) {
  if (carried.size() != inits.size())
    return Decision::failed();

  for (std::size_t i = 0; i < carried.size(); ++i) {
    if (carried[i].regs.size() != inits[i].regs.size())
      return Decision::failed();
    for (std::size_t r = 0; r < carried[i].regs.size(); ++r)
      body.push_back(c.declStmt(mslTypeOf(carried[i].elem), carried[i].regs[r],
                                c.var(inits[i].regs[r])));
  }

  if (!emitYield(c, loopBody, carried, yielded))
    return Decision::failed();

  const LoopExprs e = boundsOf(c, b);

  if (b.wideIv) {
    // The test moves into the body: the derived value does not exist until
    // the body computes it.
    const msl::Str tc = b.iv + "_tc";
    msl::Block inner;
    inner.push_back(
        c.declStmt(mslTypeOf(b.ivType()), b.iv,
                   c.binary(msl::BinOp::Add, e.lo,
                            c.binary(msl::BinOp::Mul, c.var(tc), e.step))));
    inner.push_back(c.ifStmt(
        c.unary(msl::UnOp::LNot, c.binary(msl::BinOp::Lt, c.var(b.iv), e.hi)),
        msl::Block{c.breakStmt()}));
    for (msl::Stmt *s : loopBody)
      inner.push_back(s);

    body.push_back(c.forStmt(
        c.declStmt(msl::Context::i32(), tc, c.lit(0)), nullptr,
        c.assignOp(msl::BinOp::Add, c.var(tc), c.lit(1)), std::move(inner)));
    return Decision::emitted();
  }

  body.push_back(c.forStmt(c.declStmt(mslTypeOf(b.ivType()), b.iv, e.lo),
                           c.binary(msl::BinOp::Lt, c.var(b.iv), e.hi),
                           c.assignOp(msl::BinOp::Add, c.var(b.iv), e.step),
                           std::move(loopBody)));
  return Decision::emitted();
}

// scf.while has a `before` region computing the condition and an `after`
// region doing the work. MSL has no such split, so the before region becomes
// the loop body's head with an early break.
inline Decision emitWhile(msl::Context &c, msl::Block &body,
                          const Carried &carried, const Carried &inits,
                          msl::Block beforeArm, const msl::Str &cond,
                          const Carried &results, const Carried &forwarded,
                          msl::Block afterArm, const Carried &yielded) {
  if (carried.size() != inits.size())
    return Decision::failed();

  for (std::size_t i = 0; i < carried.size(); ++i) {
    if (carried[i].regs.size() != inits[i].regs.size())
      return Decision::failed();
    for (std::size_t r = 0; r < carried[i].regs.size(); ++r)
      body.push_back(c.declStmt(mslTypeOf(carried[i].elem), carried[i].regs[r],
                                c.var(inits[i].regs[r])));
  }
  declareResults(c, body, results);

  // Forwarding before the break: after the break nothing runs.
  msl::Block exit;
  if (!emitYield(c, exit, results, forwarded))
    return Decision::failed();
  exit.push_back(c.breakStmt());

  msl::Block loop = std::move(beforeArm);
  loop.push_back(
      c.ifStmt(c.unary(msl::UnOp::LNot, c.var(cond)), std::move(exit)));
  for (msl::Stmt *s : afterArm)
    loop.push_back(s);
  if (!emitYield(c, loop, carried, yielded))
    return Decision::failed();

  body.push_back(c.whileStmt(c.litBool(true), std::move(loop)));
  return Decision::emitted();
}

} // namespace agpu

#endif // AGPU_EMIT_CONTROL_H
