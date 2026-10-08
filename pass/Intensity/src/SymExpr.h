//===- SymExpr.h - Affine expressions extended with min/max -----*- C++ -*-===//
//
// Triton Intensity extension.
//
//===----------------------------------------------------------------------===//
//
// `mlir::AffineExpr` is a closed kind enum interned in an `MLIRContext`.
// It has no `min` or `max` kind, and a subclass cannot add one:
// `simplifyAffineExpr`, the printer, and the arithmetic operators all
// switch on that enum and return `AffineExpr` by value.
//
// `SymExpr` is a pass-local expression that *contains* an `AffineExpr`.
// An expression is either one affine tree, or `min` / `max` / an arithmetic
// operator over such trees. Two affine children of `+`, `-`, `*`,
// `floorDiv`, `ceilDiv`, or `%` collapse back into a single `AffineExpr`,
// so constant folding and `simplifyAffineExpr` still apply to those
// subtrees. `min` and `max` stay outside.
//
// Identities applied at construction time:
//   - `min(e, e) = e`, and likewise for `max`.
//   - `min` / `max` of two constants fold.
//   - When `a` and `b` are affine and `a - b` is a constant: a positive
//     difference selects `a` for `max` and `b` for `min`; a negative
//     difference selects the other side. Zero means the sides are equal.
//   - `min(x, min(x, y)) = min(x, y)`, `min(x, max(x, y)) = x`, and the
//     `max` duals.
//   - `+` and `-` distribute over `min` / `max`. Subtracting a `min`
//     flips it to `max`.
//   - Multiplication by a non-negative constant distributes; a negative
//     constant distributes and swaps `min` with `max`.
//   - `floorDiv` / `ceilDiv` by a positive constant distribute. `%` does
//     not.
//
// Nodes are interned in a `SymExpr::Context` owned by the caller (the
// pass's per-function symbol table). A `SymExpr` is a pointer into that
// context and is invalidated when the context is destroyed.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_EXT_PASS_INTENSITY_SYM_EXPR_H
#define TRITON_EXT_PASS_INTENSITY_SYM_EXPR_H

#include "mlir/IR/AffineExpr.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <optional>

namespace tint {

namespace detail {
struct BinKey {
  uint8_t kind = 0;
  const void *lhs = nullptr;
  const void *rhs = nullptr;
  bool operator==(const BinKey &other) const {
    return kind == other.kind && lhs == other.lhs && rhs == other.rhs;
  }
};
} // namespace detail

} // namespace tint

namespace llvm {
template <> struct DenseMapInfo<tint::detail::BinKey> {
  static tint::detail::BinKey getEmptyKey() { return {0xFF, nullptr, nullptr}; }
  static tint::detail::BinKey getTombstoneKey() {
    return {0xFE, nullptr, nullptr};
  }
  static unsigned getHashValue(const tint::detail::BinKey &key) {
    return hash_combine(key.kind, key.lhs, key.rhs);
  }
  static bool isEqual(const tint::detail::BinKey &lhs,
                      const tint::detail::BinKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace tint {

// Symbolic integer expression: an affine tree, or min/max around one.
class SymExpr {
public:
  class Context;

  SymExpr() = default;

  explicit operator bool() const { return node != nullptr; }
  bool operator!() const { return node == nullptr; }

  static SymExpr min(SymExpr lhs, SymExpr rhs);
  static SymExpr max(SymExpr lhs, SymExpr rhs);

  SymExpr operator+(SymExpr other) const;
  SymExpr operator-(SymExpr other) const;
  SymExpr operator-() const;
  SymExpr operator*(SymExpr other) const;
  SymExpr operator%(SymExpr other) const;
  SymExpr floorDiv(SymExpr other) const;
  SymExpr ceilDiv(SymExpr other) const;

  // Flatten each affine subtree with `simplifyAffineExpr`. `numSymbols` is
  // the number of `AffineSymbolExpr` positions in use.
  SymExpr simplify(unsigned numSymbols) const;

  void print(llvm::raw_ostream &os) const;

private:
  enum class Kind : uint8_t {
    Affine,
    Add,
    Mul,
    FloorDiv,
    CeilDiv,
    Mod,
    Min,
    Max,
  };

  struct Node {
    Kind kind;
    const Context *ctx;
    mlir::AffineExpr affine;
    const Node *lhs = nullptr;
    const Node *rhs = nullptr;
  };

public:
  // Intern table for one function's equations. Not copyable or movable:
  // every `SymExpr` is a pointer into its allocator.
  class Context {
  public:
    explicit Context(mlir::MLIRContext *ctx) : mlirCtx(ctx) {}
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    SymExpr getAffine(mlir::AffineExpr expr) const;
    SymExpr getConstant(int64_t value) const;

  private:
    friend class SymExpr;
    SymExpr getBin(uint8_t kind, SymExpr lhs, SymExpr rhs) const;

    mlir::MLIRContext *mlirCtx;
    mutable llvm::BumpPtrAllocator allocator;
    mutable llvm::DenseMap<const void *, const Node *> affineNodes;
    mutable llvm::DenseMap<detail::BinKey, const Node *> binNodes;
  };

private:
  // A `min` or `max` node, used while distributing arithmetic over it.
  struct Bound {
    Kind kind;
    const Node *lhs;
    const Node *rhs;
  };

  explicit SymExpr(const Node *n) : node(n) {}

  bool isAffine() const { return node && node->kind == Kind::Affine; }
  mlir::AffineExpr affine() const { return node->affine; }
  const Context *ctx() const { return node->ctx; }

  std::optional<int64_t> tryGetConstant() const;
  std::optional<Bound> asBound() const;

  static SymExpr foldBound(Kind kind, SymExpr lhs, SymExpr rhs);
  static SymExpr mulByConstant(int64_t factor, SymExpr expr);

  enum class Tightness { Weak, Strong };
  void printRec(llvm::raw_ostream &os, Tightness enclosing) const;
  void printAffine(llvm::raw_ostream &os, mlir::AffineExpr expr,
                   Tightness enclosing) const;
  void dump() const;
  static const char *spelling(Kind kind);

  const Node *node = nullptr;
};

inline SymExpr SymExpr::Context::getAffine(mlir::AffineExpr expr) const {
  assert(expr && "null AffineExpr");
  const void *key = expr.getAsOpaquePointer();
  const Node *&slot = affineNodes[key];
  if (!slot) {
    slot = new (allocator.Allocate<Node>())
        Node{Kind::Affine, this, expr, nullptr, nullptr};
  }
  return SymExpr(slot);
}

inline SymExpr SymExpr::Context::getConstant(int64_t value) const {
  return getAffine(mlir::getAffineConstantExpr(value, mlirCtx));
}

inline SymExpr SymExpr::Context::getBin(uint8_t kind, SymExpr lhs,
                                        SymExpr rhs) const {
  assert(lhs && rhs && "null SymExpr operand");
  detail::BinKey key{kind, lhs.node, rhs.node};
  const Node *&slot = binNodes[key];
  if (!slot) {
    slot = new (allocator.Allocate<Node>()) Node{
        static_cast<Kind>(kind), this, mlir::AffineExpr(), lhs.node, rhs.node};
  }
  return SymExpr(slot);
}

inline std::optional<int64_t> SymExpr::tryGetConstant() const {
  if (!isAffine())
    return std::nullopt;
  if (auto cst = mlir::dyn_cast<mlir::AffineConstantExpr>(node->affine))
    return cst.getValue();
  return std::nullopt;
}

inline std::optional<SymExpr::Bound> SymExpr::asBound() const {
  if (!node || (node->kind != Kind::Min && node->kind != Kind::Max))
    return std::nullopt;
  return Bound{node->kind, node->lhs, node->rhs};
}

inline SymExpr SymExpr::foldBound(Kind kind, SymExpr lhs, SymExpr rhs) {
  assert(lhs && rhs && "null SymExpr operand");
  assert(lhs.ctx() == rhs.ctx() && "mixed SymExpr contexts");
  if (lhs.node == rhs.node)
    return lhs;
  if (auto cl = lhs.tryGetConstant()) {
    if (auto cr = rhs.tryGetConstant()) {
      int64_t folded =
          kind == Kind::Min ? std::min(*cl, *cr) : std::max(*cl, *cr);
      return lhs.ctx()->getConstant(folded);
    }
  }

  // When both sides are affine and `a - b` simplifies to a constant, that
  // sign picks the winner: a positive difference is `a` for `max` and `b`
  // for `min`; a negative difference is the other side. Zero means the
  // sides are equal. Local affine folding does not cancel `a - (a + c)`,
  // so simplify the difference before reading the constant.
  if (lhs.isAffine() && rhs.isAffine()) {
    SymExpr diff = lhs - rhs;
    unsigned numDims = 0, numSyms = 0;
    diff.affine().walk([&](mlir::AffineExpr e) {
      if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(e))
        numDims = std::max(numDims, dim.getPosition() + 1);
      else if (auto sym = mlir::dyn_cast<mlir::AffineSymbolExpr>(e))
        numSyms = std::max(numSyms, sym.getPosition() + 1);
    });
    mlir::AffineExpr simplified =
        mlir::simplifyAffineExpr(diff.affine(), numDims, numSyms);
    if (auto cst = mlir::dyn_cast<mlir::AffineConstantExpr>(simplified)) {
      bool lhsWins =
          kind == Kind::Max ? cst.getValue() >= 0 : cst.getValue() <= 0;
      return lhsWins ? lhs : rhs;
    }
  }
  // `min(x, min(x, y)) = min(x, y)` and `min(x, max(x, y)) = x`.
  auto absorb = [&](SymExpr outer, SymExpr inner) -> std::optional<SymExpr> {
    Kind innerKind = inner.node->kind;
    bool sameKind = innerKind == kind;
    bool dual = (kind == Kind::Min && innerKind == Kind::Max) ||
                (kind == Kind::Max && innerKind == Kind::Min);
    if ((sameKind || dual) &&
        (inner.node->lhs == outer.node || inner.node->rhs == outer.node))
      return sameKind ? inner : outer;
    return std::nullopt;
  };
  if (auto folded = absorb(lhs, rhs))
    return *folded;
  if (auto folded = absorb(rhs, lhs))
    return *folded;
  // `max(max(a, k), c) = max(a, k)` when `k >= c`: the inner bound already
  // clears `c`. The `min` dual holds when `k <= c`.
  auto dominated = [&](SymExpr side, int64_t bound) -> std::optional<SymExpr> {
    if (side.node->kind != kind)
      return std::nullopt;
    auto clears = [&](const Node *child) {
      auto cst = SymExpr(child).tryGetConstant();
      if (!cst)
        return false;
      return kind == Kind::Max ? *cst >= bound : *cst <= bound;
    };
    if (clears(side.node->lhs) || clears(side.node->rhs))
      return side;
    return std::nullopt;
  };
  if (auto cst = rhs.tryGetConstant())
    if (auto folded = dominated(lhs, *cst))
      return *folded;
  if (auto cst = lhs.tryGetConstant())
    if (auto folded = dominated(rhs, *cst))
      return *folded;
  return lhs.ctx()->getBin(static_cast<uint8_t>(kind), lhs, rhs);
}

inline SymExpr SymExpr::min(SymExpr lhs, SymExpr rhs) {
  return foldBound(Kind::Min, lhs, rhs);
}

inline SymExpr SymExpr::max(SymExpr lhs, SymExpr rhs) {
  return foldBound(Kind::Max, lhs, rhs);
}

inline SymExpr SymExpr::mulByConstant(int64_t factor, SymExpr expr) {
  assert(expr && "null SymExpr operand");
  if (factor == 0)
    return expr.ctx()->getConstant(0);
  if (factor == 1)
    return expr;
  if (std::optional<Bound> bound = expr.asBound()) {
    SymExpr lhs = mulByConstant(factor, SymExpr(bound->lhs));
    SymExpr rhs = mulByConstant(factor, SymExpr(bound->rhs));
    if (factor > 0)
      return bound->kind == Kind::Min ? min(lhs, rhs) : max(lhs, rhs);
    return bound->kind == Kind::Min ? max(lhs, rhs) : min(lhs, rhs);
  }
  if (expr.node->kind == Kind::Add) {
    return mulByConstant(factor, SymExpr(expr.node->lhs)) +
           mulByConstant(factor, SymExpr(expr.node->rhs));
  }
  if (expr.isAffine())
    return expr.ctx()->getAffine(expr.affine() * factor);
  return expr.ctx()->getBin(static_cast<uint8_t>(Kind::Mul),
                            expr.ctx()->getConstant(factor), expr);
}

inline SymExpr SymExpr::operator+(SymExpr other) const {
  assert(node && other.node && "null SymExpr operand");
  if (auto cst = tryGetConstant()) {
    if (*cst == 0)
      return other;
  }
  if (auto cst = other.tryGetConstant()) {
    if (*cst == 0)
      return *this;
  }
  if (std::optional<Bound> bound = asBound()) {
    SymExpr lhs = SymExpr(bound->lhs) + other;
    SymExpr rhs = SymExpr(bound->rhs) + other;
    return bound->kind == Kind::Min ? min(lhs, rhs) : max(lhs, rhs);
  }
  if (std::optional<Bound> bound = other.asBound()) {
    SymExpr lhs = *this + SymExpr(bound->lhs);
    SymExpr rhs = *this + SymExpr(bound->rhs);
    return bound->kind == Kind::Min ? min(lhs, rhs) : max(lhs, rhs);
  }
  if (isAffine() && other.isAffine())
    return ctx()->getAffine(affine() + other.affine());
  return ctx()->getBin(static_cast<uint8_t>(Kind::Add), *this, other);
}

inline SymExpr SymExpr::operator-() const {
  assert(node && "null SymExpr");
  return mulByConstant(-1, *this);
}

inline SymExpr SymExpr::operator-(SymExpr other) const {
  return *this + (-other);
}

inline SymExpr SymExpr::operator*(SymExpr other) const {
  assert(node && other.node && "null SymExpr operand");
  if (auto cst = tryGetConstant())
    return mulByConstant(*cst, other);
  if (auto cst = other.tryGetConstant())
    return mulByConstant(*cst, *this);
  if (isAffine() && other.isAffine())
    return ctx()->getAffine(affine() * other.affine());
  return ctx()->getBin(static_cast<uint8_t>(Kind::Mul), *this, other);
}

inline SymExpr SymExpr::floorDiv(SymExpr other) const {
  assert(node && other.node && "null SymExpr operand");
  if (auto cst = other.tryGetConstant()) {
    if (*cst > 0) {
      if (std::optional<Bound> bound = asBound()) {
        SymExpr lhs = SymExpr(bound->lhs).floorDiv(other);
        SymExpr rhs = SymExpr(bound->rhs).floorDiv(other);
        return bound->kind == Kind::Min ? min(lhs, rhs) : max(lhs, rhs);
      }
    }
  }
  if (isAffine() && other.isAffine())
    return ctx()->getAffine(affine().floorDiv(other.affine()));
  return ctx()->getBin(static_cast<uint8_t>(Kind::FloorDiv), *this, other);
}

inline SymExpr SymExpr::ceilDiv(SymExpr other) const {
  assert(node && other.node && "null SymExpr operand");
  if (auto cst = other.tryGetConstant()) {
    if (*cst > 0) {
      if (std::optional<Bound> bound = asBound()) {
        SymExpr lhs = SymExpr(bound->lhs).ceilDiv(other);
        SymExpr rhs = SymExpr(bound->rhs).ceilDiv(other);
        return bound->kind == Kind::Min ? min(lhs, rhs) : max(lhs, rhs);
      }
    }
  }
  if (isAffine() && other.isAffine())
    return ctx()->getAffine(affine().ceilDiv(other.affine()));
  return ctx()->getBin(static_cast<uint8_t>(Kind::CeilDiv), *this, other);
}

inline SymExpr SymExpr::operator%(SymExpr other) const {
  assert(node && other.node && "null SymExpr operand");
  if (isAffine() && other.isAffine())
    return ctx()->getAffine(affine() % other.affine());
  return ctx()->getBin(static_cast<uint8_t>(Kind::Mod), *this, other);
}

inline SymExpr SymExpr::simplify(unsigned numSymbols) const {
  if (!node)
    return *this;
  if (isAffine()) {
    return ctx()->getAffine(
        mlir::simplifyAffineExpr(affine(), /*numDims=*/0, numSymbols));
  }
  SymExpr lhs = SymExpr(node->lhs).simplify(numSymbols);
  SymExpr rhs = SymExpr(node->rhs).simplify(numSymbols);
  switch (node->kind) {
  case Kind::Add:
    return lhs + rhs;
  case Kind::Mul:
    return lhs * rhs;
  case Kind::FloorDiv:
    return lhs.floorDiv(rhs);
  case Kind::CeilDiv:
    return lhs.ceilDiv(rhs);
  case Kind::Mod:
    return lhs % rhs;
  case Kind::Min:
    return min(lhs, rhs);
  case Kind::Max:
    return max(lhs, rhs);
  case Kind::Affine:
    break;
  }
  llvm_unreachable("unhandled SymExpr kind");
}

inline const char *SymExpr::spelling(Kind kind) {
  switch (kind) {
  case Kind::Add:
    return " + ";
  case Kind::Mul:
    return " * ";
  case Kind::Mod:
    return " % ";
  default:
    break;
  }
  llvm_unreachable("not an infix arithmetic kind");
}

inline void SymExpr::printAffine(llvm::raw_ostream &os, mlir::AffineExpr expr,
                                 Tightness enclosing) const {
  using mlir::AffineExprKind;
  const char *binopSpelling = nullptr;
  switch (expr.getKind()) {
  case AffineExprKind::SymbolId:
    os << 's' << mlir::cast<mlir::AffineSymbolExpr>(expr).getPosition();
    return;
  case AffineExprKind::DimId:
    os << 'd' << mlir::cast<mlir::AffineDimExpr>(expr).getPosition();
    return;
  case AffineExprKind::Constant:
    os << mlir::cast<mlir::AffineConstantExpr>(expr).getValue();
    return;
  case AffineExprKind::FloorDiv:
    binopSpelling = "floordiv";
    break;
  case AffineExprKind::CeilDiv:
    binopSpelling = "ceildiv";
    break;
  case AffineExprKind::Add:
    binopSpelling = " + ";
    break;
  case AffineExprKind::Mul:
    binopSpelling = " * ";
    break;
  case AffineExprKind::Mod:
    binopSpelling = " % ";
    break;
  }

  auto binOp = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
  mlir::AffineExpr lhsExpr = binOp.getLHS();
  mlir::AffineExpr rhsExpr = binOp.getRHS();

  // `floordiv` and `ceildiv` print as calls, like `min` and `max`. The call
  // is an atom, so a tight parent does not add another pair of parens.
  if (expr.getKind() == AffineExprKind::FloorDiv ||
      expr.getKind() == AffineExprKind::CeilDiv) {
    os << binopSpelling << '(';
    printAffine(os, lhsExpr, Tightness::Weak);
    os << ", ";
    printAffine(os, rhsExpr, Tightness::Weak);
    os << ')';
    return;
  }

  // Tight operators (`*`, `%`) parenthesize when nested in a tight parent.
  // Addition does not, except for the subtraction forms below.
  if (expr.getKind() != AffineExprKind::Add) {
    if (enclosing == Tightness::Strong)
      os << '(';

    auto rhsConst = mlir::dyn_cast<mlir::AffineConstantExpr>(rhsExpr);
    if (rhsConst && expr.getKind() == AffineExprKind::Mul &&
        rhsConst.getValue() == -1) {
      os << '-';
      printAffine(os, lhsExpr, Tightness::Strong);
      if (enclosing == Tightness::Strong)
        os << ')';
      return;
    }

    printAffine(os, lhsExpr, Tightness::Strong);
    os << binopSpelling;
    printAffine(os, rhsExpr, Tightness::Strong);
    if (enclosing == Tightness::Strong)
      os << ')';
    return;
  }

  if (enclosing == Tightness::Strong)
    os << '(';

  // `lhs + (rhs * -1)` prints as `lhs - rhs`.
  if (auto rhs = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(rhsExpr)) {
    if (rhs.getKind() == AffineExprKind::Mul) {
      if (auto factor =
              mlir::dyn_cast<mlir::AffineConstantExpr>(rhs.getRHS())) {
        if (factor.getValue() == -1) {
          printAffine(os, lhsExpr, Tightness::Weak);
          os << " - ";
          Tightness subTight = rhs.getLHS().getKind() == AffineExprKind::Add
                                   ? Tightness::Strong
                                   : Tightness::Weak;
          printAffine(os, rhs.getLHS(), subTight);
          if (enclosing == Tightness::Strong)
            os << ')';
          return;
        }
        if (factor.getValue() < -1) {
          printAffine(os, lhsExpr, Tightness::Weak);
          os << " - ";
          printAffine(os, rhs.getLHS(), Tightness::Strong);
          os << " * " << -factor.getValue();
          if (enclosing == Tightness::Strong)
            os << ')';
          return;
        }
      }
    }
  }

  if (auto rhsConst = mlir::dyn_cast<mlir::AffineConstantExpr>(rhsExpr)) {
    if (rhsConst.getValue() < 0) {
      printAffine(os, lhsExpr, Tightness::Weak);
      os << " - " << -rhsConst.getValue();
      if (enclosing == Tightness::Strong)
        os << ')';
      return;
    }
  }

  printAffine(os, lhsExpr, Tightness::Weak);
  os << " + ";
  printAffine(os, rhsExpr, Tightness::Weak);
  if (enclosing == Tightness::Strong)
    os << ')';
}

inline void SymExpr::printRec(llvm::raw_ostream &os,
                              Tightness enclosing) const {
  assert(node && "null SymExpr");
  const char *call = nullptr;
  switch (node->kind) {
  case Kind::Affine:
    printAffine(os, node->affine, enclosing);
    return;
  case Kind::Min:
    call = "min";
    break;
  case Kind::Max:
    call = "max";
    break;
  case Kind::FloorDiv:
    call = "floordiv";
    break;
  case Kind::CeilDiv:
    call = "ceildiv";
    break;
  default:
    break;
  }
  if (call) {
    os << call << '(';
    SymExpr(node->lhs).printRec(os, Tightness::Weak);
    os << ", ";
    SymExpr(node->rhs).printRec(os, Tightness::Weak);
    os << ')';
    return;
  }
  bool wrap = enclosing == Tightness::Strong;
  if (wrap)
    os << '(';
  Tightness child =
      node->kind == Kind::Add ? Tightness::Weak : Tightness::Strong;
  SymExpr(node->lhs).printRec(os, child);
  os << spelling(node->kind);
  Tightness rhsTight =
      node->kind == Kind::Add ? Tightness::Weak : Tightness::Strong;
  SymExpr(node->rhs).printRec(os, rhsTight);
  if (wrap)
    os << ')';
}

inline void SymExpr::print(llvm::raw_ostream &os) const {
  if (!node) {
    os << "<<null>>";
    return;
  }
  printRec(os, Tightness::Weak);
}

void SymExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, SymExpr expr) {
  expr.print(os);
  return os;
}

} // namespace tint

#endif // TRITON_EXT_PASS_INTENSITY_SYM_EXPR_H
