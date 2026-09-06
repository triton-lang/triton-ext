// CoordHoist - coordinate expressions, emitted once each, so every asker gets
// the same spelling. CoordSource delegates here when a kernel-scoped instance
// exists.
#ifndef AGPU_EMIT_COORD_HOIST_H
#define AGPU_EMIT_COORD_HOIST_H

#include "agpu/core/Names.h"
#include "agpu/emit/LayoutExpr.h"
#include "agpu/msl/Context.h"

#include <map>
#include <string>

namespace agpu {

// The identity of a coordinate expression. The separators matter: without them
// `{1,2}` and `{12}` key alike.
inline std::string coordKey(const LayoutBasis &lb, int reg) {
  std::string key = std::to_string(lb.registerConstant(reg));
  for (const BasisRow *row : {&lb.lane, &lb.warp, &lb.block}) {
    key += "|";
    for (int32_t b : *row)
      key += std::to_string(b) + ",";
  }
  return key;
}

class CoordHoist {
public:
  CoordHoist(const ThreadNames &n, msl::Str prefix = "coord")
      : lane_(n.laneId), warp_(n.warpId), block_(n.blockId),
        prefix_(std::move(prefix)) {}

  // A fresh declaration the first time, a variable reference after that.
  msl::Expr *coord(msl::Context &c, const LayoutBasis &lb, int reg) {
    const std::string key = coordKey(lb, reg);
    const auto it = names_.find(key);
    if (it != names_.end())
      return c.var(it->second);

    msl::Expr *built = coordExpr(c, lb, reg, lane_, warp_, block_);

    if (built && built->kind == msl::ExprKind::Literal)
      return built;

    const msl::Str name = prefix_ + std::to_string(next_++);
    decls.push_back(
        c.declStmt(msl::Type::scalar(msl::Scalar::I32), name, built));
    names_.emplace(key, name);
    return c.var(name);
  }

  msl::Block decls;

  std::size_t distinct() const { return names_.size(); }

private:
  msl::Str lane_, warp_, block_, prefix_;
  std::map<std::string, msl::Str> names_;
  int next_ = 0;
};

} // namespace agpu

#endif // AGPU_EMIT_COORD_HOIST_H
