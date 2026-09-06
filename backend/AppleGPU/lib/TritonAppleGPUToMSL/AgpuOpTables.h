// AgpuOpTables - which agpu fact each MLIR op name carries, and the predicate
// decoders that read MLIR's own enums.
#ifndef AGPU_BRIDGE_OP_TABLES_H
#define AGPU_BRIDGE_OP_TABLES_H

#include "agpu/plan/Elementwise.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <string_view>
#include <vector>

namespace mlir::triton::applegpu::bridge {

// A handler registered for several ops still has to know which one it got, so
// the test and the registration read one name.
inline constexpr std::string_view kLoad = "tt.load";
inline constexpr std::string_view kGetProgramId = "tt.get_program_id";

// ── asking a table ────────────────────────────────────────────────────────

// The row for an op name, or null when the table has none.
template <class Row, std::size_t N>
const Row *rowFor(const Row (&rows)[N], std::string_view name) {
  for (const Row &r : rows)
    if (r.op == name)
      return &r;
  return nullptr;
}

// Every op name a table covers.
template <class Row, std::size_t N>
std::vector<std::string_view> namesOf(const Row (&rows)[N]) {
  std::vector<std::string_view> out;
  out.reserve(N);
  for (const Row &r : rows)
    out.push_back(r.op);
  return out;
}

// ── the operator family ───────────────────────────────────────────────────

struct EwName {
  std::string_view op;
  agpu::EwOp ew;
};

inline constexpr EwName kEwNames[] = {
    {"arith.addf", agpu::EwOp::Add},       {"arith.subf", agpu::EwOp::Sub},
    {"arith.mulf", agpu::EwOp::Mul},       {"arith.divf", agpu::EwOp::DivF},
    {"tt.precise_divf", agpu::EwOp::DivF},

    {"arith.addi", agpu::EwOp::Add},       {"arith.subi", agpu::EwOp::Sub},
    {"arith.muli", agpu::EwOp::Mul},       {"arith.divsi", agpu::EwOp::DivS},
    {"arith.divui", agpu::EwOp::DivU},     {"arith.remsi", agpu::EwOp::RemS},
    {"arith.remui", agpu::EwOp::RemU},

    {"arith.andi", agpu::EwOp::And},       {"arith.ori", agpu::EwOp::Or},
    {"arith.xori", agpu::EwOp::Xor},       {"arith.shli", agpu::EwOp::Shl},
    {"arith.shrsi", agpu::EwOp::ShrS},     {"arith.shrui", agpu::EwOp::ShrU},
    // `arith.remf` is absent: it lowers to `fmod` in the two-operand math
    // table below.
};

inline bool ewOpFor(std::string_view name, agpu::EwOp &out) {
  const EwName *e = rowFor(kEwNames, name);
  if (e)
    out = e->ew;
  return e != nullptr;
}

inline std::vector<std::string_view> ewOpNames() { return namesOf(kEwNames); }

inline constexpr std::string_view kCmpI = "arith.cmpi";

struct CmpIRow {
  arith::CmpIPredicate pred;
  agpu::EwOp ew;
};

inline constexpr CmpIRow kCmpIRows[] = {
    {arith::CmpIPredicate::eq, agpu::EwOp::CmpEq},
    {arith::CmpIPredicate::ne, agpu::EwOp::CmpNe},
    {arith::CmpIPredicate::slt, agpu::EwOp::CmpLtS},
    {arith::CmpIPredicate::sle, agpu::EwOp::CmpLeS},
    {arith::CmpIPredicate::sgt, agpu::EwOp::CmpGtS},
    {arith::CmpIPredicate::sge, agpu::EwOp::CmpGeS},
    {arith::CmpIPredicate::ult, agpu::EwOp::CmpLtU},
    {arith::CmpIPredicate::ule, agpu::EwOp::CmpLeU},
    {arith::CmpIPredicate::ugt, agpu::EwOp::CmpGtU},
    {arith::CmpIPredicate::uge, agpu::EwOp::CmpGeU},
};

// A table does not warn when MLIR adds a predicate, so the count is asserted.
static_assert(sizeof(kCmpIRows) / sizeof(kCmpIRows[0]) ==
                  arith::getMaxEnumValForCmpIPredicate() + 1,
              "every CmpIPredicate needs a row");

// Integer comparisons. The op name says "compare"; the predicate attribute
// says which.
inline bool cmpOpFor(std::string_view name, int64_t pred, agpu::EwOp &out) {
  if (name != kCmpI)
    return false;
  for (const CmpIRow &r : kCmpIRows)
    if (r.pred == (arith::CmpIPredicate)pred) {
      out = r.ew;
      return true;
    }
  return false;
}

inline std::vector<std::string_view> cmpOpNames() { return {kCmpI}; }

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_OP_TABLES_H
