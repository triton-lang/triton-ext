// LayoutExpr.h - coordinate expressions from layout bases.
//
// The bases themselves live in LayoutBasis.h, AST-free. `coordExpr` turns
// them into syntax.
#ifndef AGPU_LAYOUT_EXPR_H
#define AGPU_LAYOUT_EXPR_H

#include "agpu/msl/Context.h"
#include "agpu/plan/LayoutBasis.h"

#include <cstdint>
#include <vector>

namespace agpu {

// Build `coord(reg)` as an expression over the runtime id names: xor of a
// folded constant and one term per runtime bit whose basis is non-zero.
inline msl::Expr *coordExpr(msl::Context &c, const LayoutBasis &lb, int reg,
                            const msl::Str &laneId, const msl::Str &warpId,
                            const msl::Str &blockId = {}) {
  msl::SmallVec<msl::Expr *, 4> terms;

  if (int32_t k = lb.registerConstant(reg))
    terms.push_back(c.lit(k));

  auto runtime = [&](const BasisRow &row, const msl::Str &idName) {
    const int n = (int)row.size();
    for (int b = 0; b < n;) {
      const int32_t basis = row[b];
      if (basis == 0) {
        ++b;
        continue;
      }
      // A maximal run of identity bases (basis(k) == 1<<k) is a contiguous
      // bitfield: one mask replaces the chain.
      if (basis == (std::int32_t(1) << b)) {
        int e = b;
        while (e < n && row[e] == (std::int32_t(1) << e))
          ++e;
        const int32_t mask = ((std::int32_t(1) << (e - b)) - 1) << b;
        terms.push_back(c.binary(msl::BinOp::And, c.var(idName), c.lit(mask)));
        b = e;
        continue;
      }
      // (((id >> b) & 1) * basis)
      msl::Expr *bit = c.binary(
          msl::BinOp::And, c.binary(msl::BinOp::Shr, c.var(idName), c.lit(b)),
          c.lit(1));
      terms.push_back(c.binary(msl::BinOp::Mul, bit, c.lit(basis)));
      ++b;
    }
  };
  runtime(lb.lane, laneId);
  runtime(lb.warp, warpId);
  runtime(lb.block, blockId);

  if (terms.empty())
    return c.lit(0);
  return c.chain(msl::BinOp::Xor, {terms.begin(), terms.end()});
}

} // namespace agpu

#endif // AGPU_LAYOUT_EXPR_H
