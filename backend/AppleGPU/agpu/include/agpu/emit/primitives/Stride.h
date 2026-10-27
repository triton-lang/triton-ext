// Stride - the distance between consecutive rows. A device-resident operand's
// is a runtime kernel argument; a staged operand's is compile-time.
#ifndef AGPU_EMIT_STRIDE_H
#define AGPU_EMIT_STRIDE_H

#include "agpu/msl/Context.h"

namespace agpu {

class Stride {
public:
  Stride() = default;

  Stride(int64_t k) : constant_(k) {}

  static Stride runtime(msl::Str name) {
    Stride s;
    s.name_ = std::move(name);
    return s;
  }

  bool isConst() const { return name_.empty(); }

  int64_t constant() const { return constant_; }

  msl::Expr *expr(msl::Context &c) const {
    if (isConst())
      return c.lit(constant_);
    return c.var(name_);
  }

  msl::Expr *scale(msl::Context &c, msl::Expr *term) const {
    return c.binary(msl::BinOp::Mul, term, expr(c));
  }

  msl::Expr *scale(msl::Context &c, int64_t k) const {
    if (isConst())
      return c.lit(k * constant_);
    return scale(c, (msl::Expr *)c.lit(k));
  }

  bool operator==(const Stride &o) const {
    return name_ == o.name_ && constant_ == o.constant_;
  }

private:
  int64_t constant_ = 0;
  msl::Str name_;
};

} // namespace agpu

#endif // AGPU_EMIT_STRIDE_H
