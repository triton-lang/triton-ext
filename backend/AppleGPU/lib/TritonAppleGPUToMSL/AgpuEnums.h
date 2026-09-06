// AgpuEnums - MLIR's enums in, agpu's enums out. Each numbering is read here
// and never travels further.
#ifndef AGPU_BRIDGE_ENUMS_H
#define AGPU_BRIDGE_ENUMS_H

#include "agpu/plan/AtomicPlan.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/TypeConvert.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <string_view>
#include <vector>

namespace mlir::triton::applegpu::bridge {

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

struct CmpFRow {
  arith::CmpFPredicate pred;
  agpu::FCmp fcmp;
};

inline constexpr CmpFRow kCmpFRows[] = {
    {arith::CmpFPredicate::AlwaysFalse, agpu::FCmp::False},
    {arith::CmpFPredicate::OEQ, agpu::FCmp::OEq},
    {arith::CmpFPredicate::OGT, agpu::FCmp::OGt},
    {arith::CmpFPredicate::OGE, agpu::FCmp::OGe},
    {arith::CmpFPredicate::OLT, agpu::FCmp::OLt},
    {arith::CmpFPredicate::OLE, agpu::FCmp::OLe},
    {arith::CmpFPredicate::ONE, agpu::FCmp::ONe},
    {arith::CmpFPredicate::ORD, agpu::FCmp::Ord},
    {arith::CmpFPredicate::UEQ, agpu::FCmp::UEq},
    {arith::CmpFPredicate::UGT, agpu::FCmp::UGt},
    {arith::CmpFPredicate::UGE, agpu::FCmp::UGe},
    {arith::CmpFPredicate::ULT, agpu::FCmp::ULt},
    {arith::CmpFPredicate::ULE, agpu::FCmp::ULe},
    {arith::CmpFPredicate::UNE, agpu::FCmp::UNe},
    {arith::CmpFPredicate::UNO, agpu::FCmp::Uno},
    {arith::CmpFPredicate::AlwaysTrue, agpu::FCmp::True},
};

static_assert(sizeof(kCmpFRows) / sizeof(kCmpFRows[0]) ==
                  arith::getMaxEnumValForCmpFPredicate() + 1,
              "every CmpFPredicate needs a row");

// Separate from `cmpOpFor`: `FCmp` carries an ordered/unordered distinction
// the integer path has no meaning for.
inline bool fcmpPredFor(int64_t pred, agpu::FCmp &out) {
  for (const CmpFRow &r : kCmpFRows)
    if (r.pred == (arith::CmpFPredicate)pred) {
      out = r.fcmp;
      return true;
    }
  return false;
}

// `tt.fp_to_fp`'s rounding mode.
inline agpu::Rounding roundingFor(int64_t mode) {
  switch ((triton::RoundingMode)mode) {
  case triton::RoundingMode::RTZ:
    return agpu::Rounding::RTZ;
  case triton::RoundingMode::RTNE:
    return agpu::Rounding::RTNE;
  }
  return agpu::Rounding::Default;
}

// Triton's read-modify-write operation, as agpu names it.
inline bool rmwOpFor(triton::RMWOp op, agpu::RmwOp &out) {
  switch (op) {
  case triton::RMWOp::AND:
    out = agpu::RmwOp::And;
    return true;
  case triton::RMWOp::OR:
    out = agpu::RmwOp::Or;
    return true;
  case triton::RMWOp::XOR:
    out = agpu::RmwOp::Xor;
    return true;
  case triton::RMWOp::ADD:
    out = agpu::RmwOp::Add;
    return true;
  case triton::RMWOp::FADD:
    out = agpu::RmwOp::FAdd;
    return true;
  case triton::RMWOp::MAX:
    out = agpu::RmwOp::Max;
    return true;
  case triton::RMWOp::MIN:
    out = agpu::RmwOp::Min;
    return true;
  case triton::RMWOp::UMAX:
    out = agpu::RmwOp::UMax;
    return true;
  case triton::RMWOp::UMIN:
    out = agpu::RmwOp::UMin;
    return true;
  case triton::RMWOp::XCHG:
    out = agpu::RmwOp::Xchg;
    return true;
  }
  return false;
}

// The memory order the IR asked for. Metal's device atomics are relaxed
// only, so this does not reach the atomic call: `fencesFor` turns it into
// device-scope fences around a relaxed operation.
inline agpu::MemOrder memOrderOf(triton::MemSemantic sem) {
  switch (sem) {
  case triton::MemSemantic::RELAXED:
    return agpu::MemOrder::Relaxed;
  case triton::MemSemantic::ACQUIRE:
    return agpu::MemOrder::Acquire;
  case triton::MemSemantic::RELEASE:
    return agpu::MemOrder::Release;
  case triton::MemSemantic::ACQUIRE_RELEASE:
    return agpu::MemOrder::AcquireRelease;
  }
  return agpu::MemOrder::Relaxed;
}

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_ENUMS_H
