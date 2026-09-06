// EmitShuffle.h - a layout conversion that stays inside the warp.
#ifndef AGPU_EMIT_SHUFFLE_H
#define AGPU_EMIT_SHUFFLE_H

#include "agpu/core/Names.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/ShufflePlan.h"
#include "agpu/plan/ShuffleType.h"

namespace agpu {

// `simd_shuffle` and friends take only 32-bit scalars and vectors of them; a
// `long`, `bool` or `half` is a compile error at the call, so the value
// travels bitcast to an accepted type and back.
inline msl::Expr *shuffleOf(msl::Context &c, const char *builtin, ElemType elem,
                            const msl::Str &v, msl::Expr *arg) {
  const msl::Scalar s = mslTypeOf(elem).scalarKind();
  const auto call = [&](msl::Expr *x) { return c.call(builtin, {x, arg}); };

  switch (shuffleFormOf(s)) {
  case ShuffleForm::SplitU32Pair:
    return c.bitcast(
        msl::Type::scalar(s),
        call(c.bitcast(msl::Type::vector(msl::Scalar::U32, 2), c.var(v))));
  case ShuffleForm::BoolAsU8:
    return c.cast(msl::Type::scalar(s),
                  call(c.cast(msl::Type::scalar(msl::Scalar::U8), c.var(v))));
  case ShuffleForm::NarrowAsBits:
    return c.bitcast(
        msl::Type::scalar(s),
        call(c.bitcast(msl::Type::scalar(shuffleBitsOf(s)), c.var(v))));
  case ShuffleForm::Direct:
    break;
  }
  return call(c.var(v));
}

struct ShuffleNames : ThreadNames {
  msl::Str srcLane = "sl";
  msl::Str table = "lanetab";
};

// Linear permutations become an XOR chain over the set bits of the lane id;
// anything else needs a lookup table, since there is no closed form.
inline msl::Expr *srcLaneExpr(msl::Context &c, const ShufflePlan &p,
                              const ShuffleNames &nm) {
  if (!p.linearLanePerm)
    return c.subscript(c.var(nm.table), c.var(nm.laneId));

  msl::Expr *acc = p.laneOffset ? c.lit((int64_t)p.laneOffset) : nullptr;

  bool identityBasis = p.laneBasis.size() > 0;
  for (std::size_t b = 0; b < p.laneBasis.size(); ++b)
    identityBasis = identityBasis && p.laneBasis[b] == (int32_t)(1u << b);
  if (identityBasis)
    return acc ? c.binary(msl::BinOp::Xor, c.var(nm.laneId), acc)
               : (msl::Expr *)c.var(nm.laneId);

  for (std::size_t b = 0; b < p.laneBasis.size(); ++b) {
    if (p.laneBasis[b] == 0)
      continue;
    msl::Expr *bit =
        c.binary(msl::BinOp::And,
                 c.binary(msl::BinOp::Shr, c.var(nm.laneId), c.lit((int64_t)b)),
                 c.lit(1));
    msl::Expr *term =
        c.binary(msl::BinOp::Mul, bit, c.lit((int64_t)p.laneBasis[b]));
    acc = acc ? c.binary(msl::BinOp::Xor, acc, term) : term;
  }
  return acc ? acc : c.lit(0);
}

// Returns the names holding the result. A pure rebind returns the source names
// unchanged and emits nothing.
inline msl::SmallVec<msl::Str, 8>
emitShuffle(msl::Context &c, msl::Block &body, const ShufflePlan &p,
            const msl::SmallVec<msl::Str, 8> &srcNames,
            const msl::SmallVec<msl::Str, 8> &dstNames, ElemType elem,
            const ShuffleNames &nm) {
  msl::SmallVec<msl::Str, 8> out;
  if (!p.usable())
    return out;

  if (p.isRebind()) {
    for (const ShuffleStep &s : p.steps)
      out.push_back(srcNames[s.srcReg]);
    return out;
  }

  // `usable()` admits at most one distinct permutation among the shuffling
  // registers, so one table and one lane index serve all of them.
  const std::vector<int32_t> *perm = p.shufflePerm();
  if (!perm)
    return out;
  if (!p.linearLanePerm) {
    msl::SmallVec<msl::Expr *, 4> lanes;
    for (int32_t l : *perm)
      lanes.push_back(c.lit((int64_t)l));
    body.push_back(
        c.arrayDecl(msl::Context::i32(), nm.table, std::move(lanes)));
  }
  body.push_back(
      c.declStmt(msl::Context::i32(), nm.srcLane, srcLaneExpr(c, p, nm)));

  for (std::size_t r = 0; r < p.steps.size(); ++r) {
    const ShuffleStep &s = p.steps[r];
    if (s.identity()) {
      out.push_back(srcNames[s.srcReg]);
      continue;
    }
    body.push_back(
        c.declStmt(mslTypeOf(elem), dstNames[r],
                   shuffleOf(c, msl::builtin::simd::Shuffle, elem,
                             srcNames[s.srcReg], c.var(nm.srcLane))));
    out.push_back(dstNames[r]);
  }
  return out;
}
} // namespace agpu

#endif // AGPU_EMIT_SHUFFLE_H
