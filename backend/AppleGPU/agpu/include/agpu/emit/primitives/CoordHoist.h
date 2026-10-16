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
#include <utility>
#include <vector>

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
    for (auto e = epochs_.rbegin(); e != epochs_.rend(); ++e)
      if (const auto it = e->names.find(key); it != e->names.end())
        return c.var(it->second);
    if (epochs_.empty())
      if (const auto it = names_.find(key); it != names_.end())
        return c.var(it->second);

    msl::Expr *built = coordExpr(c, lb, reg, lane_, warp_, block_);

    if (built && built->kind == msl::ExprKind::Literal)
      return built;

    origins_.emplace(key, std::make_pair(lb, reg));
    const msl::Str name = prefix_ + std::to_string(next_++);
    msl::Stmt *decl =
        c.declStmt(msl::Type::scalar(msl::Scalar::I32), name, built);
    if (!epochs_.empty()) {
      epochs_.back().into->push_back(decl);
      epochs_.back().names.emplace(key, name);
    } else {
      decls.push_back(decl);
      names_.emplace(key, name);
    }
    return c.var(name);
  }

  // Metal keeps a kernel-top coordinate live until its last use, across every
  // loop before it. `lane`/`warp` take values LLVM cannot prove equal to their
  // old ones, so the coordinates respelled from them are not merged back.
  void rebase(msl::Context &c, msl::Block &into, msl::Expr *lane,
              msl::Expr *warp) {
    into.push_back(c.assign(c.var(lane_), lane));
    into.push_back(c.assign(c.var(warp_), warp));
    Epoch e;
    e.into = &into;
    for (const auto &[key, origin] : origins_) {
      msl::Expr *built =
          coordExpr(c, origin.first, origin.second, lane_, warp_, block_);
      if (!built || built->kind == msl::ExprKind::Literal)
        continue;
      const msl::Str name = prefix_ + std::to_string(next_++);
      into.push_back(
          c.declStmt(msl::Type::scalar(msl::Scalar::I32), name, built));
      e.names.emplace(key, name);
    }
    epochs_.push_back(std::move(e));
  }

  std::size_t depth() const { return epochs_.size(); }
  void popTo(std::size_t d) {
    if (d < epochs_.size())
      epochs_.resize(d);
  }

  msl::Block decls;

  std::size_t distinct() const { return names_.size(); }

private:
  struct Epoch {
    msl::Block *into = nullptr;
    std::map<std::string, msl::Str> names;
  };

  msl::Str lane_, warp_, block_, prefix_;
  std::map<std::string, msl::Str> names_;
  std::map<std::string, std::pair<LayoutBasis, int>> origins_;
  std::vector<Epoch> epochs_;
  int next_ = 0;
};

} // namespace agpu

#endif // AGPU_EMIT_COORD_HOIST_H
