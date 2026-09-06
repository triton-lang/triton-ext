// CoordGuard - deciding whether a bounds test is needed at all.
#ifndef AGPU_COORD_GUARD_H
#define AGPU_COORD_GUARD_H

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace agpu {

// Half-open window [lo, hi) on one dimension.
struct CoordWindow {
  int dim = 0;
  int64_t lo = 0;
  int64_t hi = 0;

  bool contains(int64_t v) const { return v >= lo && v < hi; }
  int64_t extent() const { return hi - lo; }
};

// The reachable range of a register's coordinate on one dimension, inclusive
// on both ends.
struct CoordRange {
  int dim = 0;
  int64_t lo = 0;
  int64_t hi = 0;

  bool within(const CoordWindow &w) const { return lo >= w.lo && hi < w.hi; }
  bool disjoint(const CoordWindow &w) const { return hi < w.lo || lo >= w.hi; }
};

struct GuardTerm {
  enum class Op { Ge, Lt };
  int dim;
  Op op;
  int64_t bound;

  bool operator==(const GuardTerm &o) const {
    return dim == o.dim && op == o.op && bound == o.bound;
  }
};

// Dead: register cannot land in the window, emit nothing.
// Unguarded: it always lands in the window, emit with no test.
// Needed: emit under the conjunction of `terms`.
class CoordGuard {
public:
  enum class Kind { Dead, Unguarded, Needed };

  static CoordGuard dead() { return CoordGuard(Kind::Dead, {}); }
  static CoordGuard unguarded() { return CoordGuard(Kind::Unguarded, {}); }
  static CoordGuard needed(std::vector<GuardTerm> terms) {
    assert(!terms.empty() && "Needed with no terms is Unguarded");
    return CoordGuard(Kind::Needed, std::move(terms));
  }

  Kind kind() const { return kind_; }
  bool isDead() const { return kind_ == Kind::Dead; }
  bool isUnguarded() const { return kind_ == Kind::Unguarded; }
  bool needsTest() const { return kind_ == Kind::Needed; }
  const std::vector<GuardTerm> &terms() const { return terms_; }

  bool operator==(const CoordGuard &o) const {
    return kind_ == o.kind_ && terms_ == o.terms_;
  }

private:
  CoordGuard(Kind k, std::vector<GuardTerm> t)
      : kind_(k), terms_(std::move(t)) {}
  Kind kind_;
  std::vector<GuardTerm> terms_;
};

// `ranges` and `windows` are matched by position.
inline CoordGuard planGuard(const std::vector<CoordRange> &ranges,
                            const std::vector<CoordWindow> &windows) {
  assert(ranges.size() == windows.size());
  std::vector<GuardTerm> terms;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const CoordRange &r = ranges[i];
    const CoordWindow &w = windows[i];
    assert(r.dim == w.dim && "range and window describe different dimensions");
    if (r.disjoint(w))
      return CoordGuard::dead();
    if (r.lo < w.lo)
      terms.push_back({w.dim, GuardTerm::Op::Ge, w.lo});
    if (r.hi >= w.hi)
      terms.push_back({w.dim, GuardTerm::Op::Lt, w.hi});
  }
  if (terms.empty())
    return CoordGuard::unguarded();
  return CoordGuard::needed(std::move(terms));
}

inline CoordGuard planGuard(std::initializer_list<CoordRange> ranges,
                            std::initializer_list<CoordWindow> windows) {
  return planGuard(std::vector<CoordRange>(ranges),
                   std::vector<CoordWindow>(windows));
}

inline CoordWindow batchWindow(int dim, int64_t bi) {
  return CoordWindow{dim, bi, bi + 1};
}

} // namespace agpu

#endif // AGPU_COORD_GUARD_H
