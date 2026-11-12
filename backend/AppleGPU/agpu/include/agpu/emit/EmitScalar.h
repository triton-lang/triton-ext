// EmitScalar.h - the scalar dot: a per-thread K loop, for when no MMA exists.
//
// `simdgroup_matrix` takes only float, half and bfloat, so integer dots land
// here unless `liftsToFloatMma` sends them through the f32 MMA.
#ifndef AGPU_EMIT_SCALAR_H
#define AGPU_EMIT_SCALAR_H

#include "agpu/emit/Emit.h"
#include "agpu/emit/EmitDirect.h"
#include "agpu/plan/DotPlan.h"

namespace agpu {

// One K loop per participating C register:
//
//   out = incoming accumulator, or zero;
//   for (k) out += (acc)A[row][k] * (acc)B[k][col];
//
// Both elements are cast to the plan's accumulator type before the multiply,
// or e.g. i8 x i8 overflows at K = 2.
inline void emitScalarDot(msl::Context &c, msl::Block &body, const Plan &p,
                          const OperandSource &a, const OperandSource &b,
                          const ReadbackInputs &back,
                          const CoordSource &cCoords, const DirectNames &nm) {
  const msl::Type accTy = mslTypeOf(p.scalar().acc);
  const int mDim = p.facts.rank - 2;
  int counter = 0;
  // Merged actions are un-merged one register at a time; the merge is about
  // consecutive pool slots, which a K loop does not use.
  for (const StageAction &merged : back.actions)
    for (int lane = 0; lane < merged.width; ++lane) {
      StageAction act = merged;
      act.reg = merged.reg + lane;
      act.width = 1;
      const msl::Str &name = back.names[act.reg];
      const msl::Str base =
          act.reg < (int)back.bases.size() ? back.bases[act.reg] : msl::Str{};
      const msl::Str kv = nm.kVar + std::to_string(counter++);

      msl::Block stmts;
      stmts.push_back(c.assign(c.var(name), base.empty()
                                                ? (msl::Expr *)c.lit(0, accTy)
                                                : c.var(base)));

      msl::Expr *batchA = nullptr, *batchB = nullptr;
      if (p.facts.batched()) {
        batchA = cCoords.of(c, act.reg, 0);
        batchB = cCoords.of(c, act.reg, 0);
      }
      msl::Block loop;
      msl::Expr *ea = c.subscript(
          c.var(a.buffer),
          a.elemOffsetOf(c, cCoords.of(c, act.reg, mDim), c.var(kv), batchA));
      msl::Expr *eb =
          c.subscript(c.var(b.buffer),
                      b.elemOffsetOf(c, c.var(kv),
                                     cCoords.of(c, act.reg, mDim + 1), batchB));
      loop.push_back(c.assignOp(
          msl::BinOp::Add, c.var(name),
          c.binary(msl::BinOp::Mul, c.cast(accTy, ea), c.cast(accTy, eb))));
      stmts.push_back(c.forStmt(
          c.declStmt(msl::Context::i32(), kv, c.lit(0)),
          c.binary(msl::BinOp::Lt, c.var(kv), c.lit(p.facts.K)),
          c.assignOp(msl::BinOp::Add, c.var(kv), c.lit(1)), std::move(loop)));

      c.guardedInto(body, guardExpr(c, act.guard, act.reg, cCoords),
                    std::move(stmts));
    }
}

} // namespace agpu

#endif // AGPU_EMIT_SCALAR_H
