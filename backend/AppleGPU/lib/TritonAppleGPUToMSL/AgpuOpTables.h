// AgpuOpTables - which agpu fact each MLIR op name carries.
// Predicate decoders needing MLIR types live separately in AgpuEnums.h.
#ifndef AGPU_BRIDGE_OP_TABLES_H
#define AGPU_BRIDGE_OP_TABLES_H

#include "agpu/plan/Elementwise.h"
#include "agpu/plan/MathFn.h"
#include "agpu/plan/TypeConvert.h"

#include <string_view>
#include <vector>

namespace mlir::triton::applegpu::bridge {

// A handler registered for several ops still has to know which one it got, so
// the test and the registration read one name.
inline constexpr std::string_view kExpandDims = "tt.expand_dims";
inline constexpr std::string_view kBroadcast = "tt.broadcast";
inline constexpr std::string_view kConvertLayout = "ttg.convert_layout";
inline constexpr std::string_view kUnsplat = "tt.unsplat";
inline constexpr std::string_view kFp4ToFp = "ttg.fp4_to_fp";
inline constexpr std::string_view kJoin = "tt.join";
inline constexpr std::string_view kFpToFp = "tt.fp_to_fp";
inline constexpr std::string_view kLoad = "tt.load";
inline constexpr std::string_view kBarrier = "gpu.barrier";
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

// ── the one-operand math family ───────────────────────────────────────────

struct MathName {
  std::string_view op;
  agpu::MathFn fn;
};

inline constexpr MathName kMathNames[] = {
    {"math.exp", agpu::MathFn::Exp},
    {"math.exp2", agpu::MathFn::Exp2},
    {"math.log", agpu::MathFn::Log},
    {"math.log2", agpu::MathFn::Log2},
    {"math.log10", agpu::MathFn::Log10},
    {"math.sqrt", agpu::MathFn::Sqrt},
    {"math.rsqrt", agpu::MathFn::Rsqrt},
    {"math.sin", agpu::MathFn::Sin},
    {"math.cos", agpu::MathFn::Cos},
    {"math.tan", agpu::MathFn::Tan},
    {"math.tanh", agpu::MathFn::Tanh},
    {"math.asin", agpu::MathFn::Asin},
    {"math.acos", agpu::MathFn::Acos},
    {"math.atan", agpu::MathFn::Atan},
    {"math.sinh", agpu::MathFn::Sinh},
    {"math.cosh", agpu::MathFn::Cosh},
    // `metal::abs` covers both int and float.
    {"math.absf", agpu::MathFn::Abs},
    {"math.absi", agpu::MathFn::Abs},
    {"math.floor", agpu::MathFn::Floor},
    {"math.ceil", agpu::MathFn::Ceil},
    // Round is half away from zero, RoundEven half to even.
    {"math.round", agpu::MathFn::Round},
    {"math.roundeven", agpu::MathFn::RoundEven},
    {"math.trunc", agpu::MathFn::Trunc},
    {"math.erf", agpu::MathFn::Erf},
    {"math.cbrt", agpu::MathFn::Cbrt},
    {"tt.precise_sqrt", agpu::MathFn::Sqrt},
};

inline bool mathFnFor(std::string_view name, agpu::MathFn &out) {
  const MathName *m = rowFor(kMathNames, name);
  if (m)
    out = m->fn;
  return m != nullptr;
}

inline std::vector<std::string_view> mathOpNames() {
  return namesOf(kMathNames);
}

// ── the two-operand math family ───────────────────────────────────────────

struct Math2Name {
  std::string_view op;
  agpu::MathFn2 fn;

  // Whether a NaN operand must reach the result. `metal::min` returns the
  // other operand on NaN (IEEE minNum), wrong for `minimumf`.
  bool propagateNan;

  // Whether the operands are read as unsigned. MLIR integers are signless,
  // so only the op name says.
  bool readsUnsigned = false;
};

inline constexpr Math2Name kMath2Names[] = {
    {"arith.minimumf", agpu::MathFn2::Min, true},
    {"arith.maximumf", agpu::MathFn2::Max, true},
    {"arith.minnumf", agpu::MathFn2::Min, false},
    {"arith.maxnumf", agpu::MathFn2::Max, false},
    {"arith.minsi", agpu::MathFn2::Min, false},
    {"arith.maxsi", agpu::MathFn2::Max, false},
    {"arith.minui", agpu::MathFn2::Min, false, true},
    {"arith.maxui", agpu::MathFn2::Max, false, true},
    {"math.powf", agpu::MathFn2::Pow, false},
    {"math.atan2", agpu::MathFn2::Atan2, false},
    {"math.copysign", agpu::MathFn2::Copysign, false},
    {"arith.remf", agpu::MathFn2::Fmod, false},
    {"tt.mulhiui", agpu::MathFn2::Mulhi, false, true},
};

inline const Math2Name *math2For(std::string_view name) {
  return rowFor(kMath2Names, name);
}

inline std::vector<std::string_view> math2OpNames() {
  return namesOf(kMath2Names);
}

struct Math3Name {
  std::string_view op;
  agpu::MathFn3 fn;
};

inline constexpr Math3Name kMath3Names[] = {
    {"tt.clampf", agpu::MathFn3::Clamp},
    // A call, so the product rounds once.
    {"math.fma", agpu::MathFn3::Fma},
};

inline const Math3Name *math3For(std::string_view name) {
  return rowFor(kMath3Names, name);
}

inline std::vector<std::string_view> math3OpNames() {
  return namesOf(kMath3Names);
}

// ── the casts ─────────────────────────────────────────────────────────────

// One cast op and the signedness it names. MLIR integers are signless, so
// only the op name says it (`extui` vs `extsi`).
struct CastName {
  std::string_view op;

  // Bits kept and only the reading changed: `as_type<uint>(1.0f)` is
  // 0x3f800000, `(uint)1.0f` is 1.
  bool reinterpret;

  // Whether the source is read as unsigned.
  bool readsUnsigned = false;

  // Whether the result is unsigned.
  bool writesUnsigned = false;
};

inline constexpr CastName kCastNames[] = {
    {"arith.extf", false},
    {"arith.extsi", false},
    {"arith.extui", false, true},
    {"arith.truncf", false},
    {"arith.trunci", false},
    {"arith.sitofp", false},
    {"arith.uitofp", false, true},
    {"arith.fptosi", false},
    {"arith.fptoui", false, false, true},
    {kFpToFp, false},
    {"arith.bitcast", true},
    {"tt.bitcast", true},
    // A Triton pointer is a 64-bit integer here (`mslTypeOf` spells
    // Kind::Pointer as u64), so these change the type and not the value.
    {"tt.int_to_ptr", true},
    {"tt.ptr_to_int", true},
};

inline const CastName *castFor(std::string_view name) {
  return rowFor(kCastNames, name);
}

inline std::vector<std::string_view> castOpNames() {
  return namesOf(kCastNames);
}

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_OP_TABLES_H
