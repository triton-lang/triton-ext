// AgpuFacts - what the walk knows about a value. Read only: emission is
// elsewhere.
#ifndef AGPU_BRIDGE_FACTS_H
#define AGPU_BRIDGE_FACTS_H

#include "agpu/bind/SymbolTable.h"
#include "agpu/msl/Ast.h"
#include "agpu/plan/Elementwise.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir::triton::applegpu::bridge {

// Carried beside an OpView: OpView's int64_t cannot hold a float.
struct ConstantValue {
  int64_t i = 0;
  double f = 0.0;
  bool isFloat = false;
  // False for an attribute type the backend has no representation for.
  bool known = false;
};

// One value's registers. Once ok(), every at(r) has a name.
class Operand {
public:
  Operand(const agpu::SymbolTable &sym, agpu::ValueId v, int64_t regs)
      : sym_(&sym), v_(v), regs_(regs) {
    for (int64_t r = 0; r < regs; ++r)
      if (sym.regAt(v, (std::size_t)r) == nullptr) {
        unnamed_ = r;
        break;
      }
  }

  bool ok() const { return unnamed_ < 0; }
  int64_t registers() const { return regs_; }

  int64_t firstUnnamedRegister() const { return unnamed_; }

  // Only meaningful once ok(), which is vacuously true for a zero-register
  // operand, hence the range check.
  const agpu::msl::Str &at(int64_t r) const {
    static const agpu::msl::Str kNone;
    if (r < 0 || r >= regs_)
      return kNone;
    const agpu::msl::Str *n = sym_->regAt(v_, (std::size_t)r);
    return n ? *n : kNone;
  }

private:
  const agpu::SymbolTable *sym_;
  agpu::ValueId v_;
  int64_t regs_;
  int64_t unnamed_ = -1;
};

// Everything a handler needs before it can emit: arity, result element type,
// register count and the operands resolved to names.
struct Ready {
  agpu::ElemType elem;
  int64_t regs = 0;
  agpu::msl::SmallVec<Operand, 3> ops;
  agpu::Decision why = agpu::Decision::emitted();

  bool ok() const { return why.ok(); }

  const Operand &operator[](std::size_t i) const { return ops[i]; }
};

// What one register holds. With a guard, the declaration takes `init` and a
// guarded assignment of `value` follows.
struct RegValue {
  agpu::msl::Expr *value = nullptr; // null declines the whole op
  agpu::msl::Expr *guard = nullptr; // null: unconditional
  agpu::msl::Expr *init = nullptr;  // the value when the guard fails
};

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_FACTS_H
