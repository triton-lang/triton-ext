// Prelude.h - the helper definitions a kernel needs before it can link.
#ifndef AGPU_PRELUDE_H
#define AGPU_PRELUDE_H

#include "agpu/core/EnumBitset.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/AssertPlan.h"
#include "agpu/plan/AtomicPlan.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/MathFn.h"
#include "agpu/plan/PrintPlan.h"
#include "agpu/plan/TypeConvert.h"

#include <cstdint>
#include <sstream>
#include <string>

namespace agpu {

// Emission order. A helper must precede any helper that calls it.
enum class Helper {
#define HELPER(Name) Name,
#include "agpu/emit/Helpers.def"

  Count,
};

namespace hn = msl::builtin::helper;

inline const char *helperName(Helper h) {
  switch (h) {
#define HELPER(Name)                                                           \
  case Helper::Name:                                                           \
    return hn::Name;
#include "agpu/emit/Helpers.def"
  case Helper::Count:
    break;
  }
  return "";
}

// One row per fp8 encoding: which helper packs it and which unpacks it.
struct Fp8Helpers {
  Fp8Kind kind;
  Helper pack;
  Helper unpack;
};

inline constexpr Fp8Helpers kFp8Helpers[] = {
    {Fp8Kind::E4M3, Helper::Fp8PackE4M3, Helper::Fp8UnpackE4M3},
    {Fp8Kind::E5M2, Helper::Fp8PackE5M2, Helper::Fp8UnpackE5M2},
    {Fp8Kind::E4B8, Helper::Fp8PackE4B8, Helper::Fp8UnpackE4B8},
    {Fp8Kind::E5B16, Helper::Fp8PackE5B16, Helper::Fp8UnpackE5B16},
};

inline bool convertHelper(const ConvertPlan &p, Helper &out) {
  const bool brain = p.to.floatKind == FloatKind::Brain;
  switch (p.kind) {
  case ConvertKind::NarrowRtz:
    out = brain ? Helper::RtzBfloat : Helper::RtzHalf;
    return true;
  case ConvertKind::NarrowRtne:
    out = brain ? Helper::RtneBfloat : Helper::RtneHalf;
    return true;
  case ConvertKind::Fp8Pack:
  case ConvertKind::Fp8Unpack:
    for (const Fp8Helpers &h : kFp8Helpers)
      if (h.kind == p.fp8) {
        out = p.kind == ConvertKind::Fp8Pack ? h.pack : h.unpack;
        return true;
      }
    return false;
  default:
    return false;
  }
}

// Which prelude helper a math function lowers to. False for everything Metal
// spells itself.
inline bool mathHelper(MathFn fn, Helper &out) {
  switch (fn) {
  case MathFn::Erf:
    out = Helper::Erf;
    return true;
  case MathFn::Cbrt:
    out = Helper::Cbrt;
    return true;
  default:
    return false;
  }
}

class HelperSet {
public:
  void add(Helper h) { bits_.add(h); }
  bool has(Helper h) const { return bits_.has(h); }
  bool any() const { return bits_.any(); }

  void require(const AtomicPlan &p) {
    switch (p.strategy) {
    case AtomicStrategy::FloatCas:
      add(Helper::AtomicRmwF32);
      return;
    case AtomicStrategy::Packed16:
      add(Helper::AtomicRmwPacked16);
      add(Helper::RtneIntHalf);
      add(Helper::RtneIntBfloat);
      return;
    default:
      return;
    }
  }

  void require(MathFn fn) {
    Helper h;
    if (mathHelper(fn, h))
      add(h);
  }

  void require(const ConvertPlan &p) {
    Helper h;
    if (convertHelper(p, h))
      add(h);
  }

private:
  EnumBitset<Helper, std::uint64_t> bits_;
};

// The `op` ladder both CAS helpers carry, generated from the enum that
// `AtomicPlan.h` selects with. `cur` names the caller's current value.
inline msl::Expr *emuRmwLadderExpr(msl::Context &c, const char *cur) {
  msl::Expr *const value = c.var(cur);
  msl::Expr *const v = c.var("v");
  msl::Expr *const op = c.var("op");

  const auto isOp = [&](EmuRmw which) {
    return c.binary(msl::BinOp::Eq, op, c.lit(emuRmwCode(which)));
  };
  const auto callOn = [&](const char *fn) { return c.call(fn, {value, v}); };

  return c.ternary(isOp(EmuRmw::Add), c.add(value, v),
                   c.ternary(isOp(EmuRmw::Max), callOn(msl::builtin::math::Max),
                             c.ternary(isOp(EmuRmw::Min),
                                       callOn(msl::builtin::math::Min), v)));
}

inline std::string emuRmwLadder(const char *cur) {
  msl::Context c;
  std::ostringstream os;
  msl::Printer(os).printExpr(emuRmwLadderExpr(c, cur));
  return os.str();
}

inline msl::Function *helperFn(msl::Context &c, const char *name,
                               msl::Type ret) {
  msl::Function *fn = c.function();
  fn->isInline = true;
  fn->name = name;
  fn->returnType = std::move(ret);
  return fn;
}

// Trailing newline included: `printPrelude` joins bodies without one.
inline std::string renderHelper(const msl::Function *fn) {
  std::ostringstream os;
  msl::Printer(os).printStmt(fn);
  std::string out = os.str();
  while (!out.empty() && out.back() == '\n')
    out.pop_back();
  return out;
}

// The fp8 packers share one algorithm and differ only in field widths. `satHi`
// is the saturated magnitude and `nanHi` the all-exponent pattern; e4m3fn
// reserves its top slot for NaN, so the two differ there and coincide for e5m2.
inline msl::Function *fp8Packer(msl::Context &c, const char *name, int mantBits,
                                int bias, int satExp, int subnormalShift,
                                int subnormalFloor, int64_t nanHi,
                                int64_t satHi, int64_t quiet,
                                msl::Expr *extraSatCond, bool biasForm) {
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type u8 = msl::Type::scalar(msl::Scalar::U8);
  const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
  msl::Function *fn = helperFn(c, name, u8);
  fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

  msl::Expr *const u = c.var("u");
  msl::Expr *const sgn = c.var("sgn");
  msl::Expr *const mant = c.var("mant");
  msl::Expr *const ex = c.var("ex");
  msl::Expr *const m = c.var("m");
  msl::Expr *const rem = c.var("rem");
  msl::Expr *const sh = c.var("sh");
  msl::Expr *const halfWay = c.var("half_");
  const int tailShift = 23 - mantBits;

  const auto shr = [&](msl::Expr *e, msl::Expr *by) {
    return c.binary(msl::BinOp::Shr, e, by);
  };
  const auto bitAnd = [&](msl::Expr *e, msl::Expr *k) {
    return c.binary(msl::BinOp::And, e, k);
  };
  const auto bitOr = [&](msl::Expr *a, msl::Expr *b) {
    return c.binary(msl::BinOp::Or, a, b);
  };
  const auto roundUp = [&](msl::Expr *target, msl::Expr *mid) {
    return c.ifStmt(
        c.binary(msl::BinOp::LOr, c.binary(msl::BinOp::Gt, rem, mid),
                 c.binary(msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, rem, mid),
                          bitAnd(m, c.lit(1, u32)))),
        {c.assignOp(msl::BinOp::Add, target, c.lit(1))});
  };

  fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
  fn->body.push_back(
      c.declStmt(u32, "sgn", bitAnd(shr(u, c.lit(24)), c.litHex(0x80))));
  fn->body.push_back(c.declStmt(
      i32, "e32", c.cast(i32, bitAnd(shr(u, c.lit(23)), c.litHex(0xff)))));
  fn->body.push_back(c.declStmt(u32, "mant", bitAnd(u, c.litHex(0x7fffff))));
  msl::Expr *nan = bitOr(sgn, c.litHex(nanHi));
  if (biasForm)
    nan = c.ternary(mant, c.litHex(0x80), nan);
  else if (quiet)
    nan = bitOr(nan, c.ternary(mant, c.litHex(quiet), c.lit(0, u32)));
  fn->body.push_back(
      c.ifStmt(c.binary(msl::BinOp::Eq, c.var("e32"), c.litHex(0xff)),
               {c.returnStmt(c.cast(u8, nan))}));
  fn->body.push_back(c.declStmt(
      i32, "ex",
      c.add(c.binary(msl::BinOp::Sub, c.var("e32"), c.lit(127)), c.lit(bias))));

  msl::Expr *satCond = c.binary(msl::BinOp::Ge, ex, c.lit(satExp));
  if (extraSatCond)
    satCond = c.binary(msl::BinOp::LOr, satCond, extraSatCond);
  fn->body.push_back(c.ifStmt(
      satCond, {c.returnStmt(c.cast(u8, bitOr(sgn, c.litHex(satHi))))}));

  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::Le, ex, c.lit(0)),
      {c.ifStmt(c.binary(msl::BinOp::Lt, ex, c.lit(subnormalFloor)),
                {c.returnStmt(c.cast(u8, biasForm ? c.lit(0, u32) : sgn))}),
       c.declStmt(u32, "fm", bitOr(mant, c.litHex(0x800000))),
       c.declStmt(i32, "sh",
                  c.binary(msl::BinOp::Sub, c.lit(subnormalShift), ex)),
       c.declStmt(u32, "m", shr(c.var("fm"), sh)),
       c.declStmt(u32, "rem",
                  bitAnd(c.var("fm"),
                         c.binary(msl::BinOp::Sub,
                                  c.binary(msl::BinOp::Shl, c.lit(1, u32), sh),
                                  c.lit(1, u32)))),
       c.declStmt(u32, "half_",
                  c.binary(msl::BinOp::Shl, c.lit(1, u32),
                           c.binary(msl::BinOp::Sub, sh, c.lit(1)))),
       roundUp(m, halfWay),
       c.returnStmt(c.cast(u8, biasForm
                                   ? c.ternary(m, bitOr(sgn, m), c.lit(0, u32))
                                   : bitOr(sgn, m)))}));

  fn->body.push_back(c.declStmt(u32, "m", shr(mant, c.lit(tailShift))));
  fn->body.push_back(c.declStmt(
      u32, "rem", bitAnd(mant, c.litHex((int64_t(1) << tailShift) - 1))));
  fn->body.push_back(
      c.declStmt(u32, "bits",
                 bitOr(bitOr(sgn, c.binary(msl::BinOp::Shl, c.cast(u32, ex),
                                           c.lit(mantBits))),
                       m)));
  fn->body.push_back(
      roundUp(c.var("bits"), c.litHex(int64_t(1) << (tailShift - 1))));
  fn->body.push_back(c.returnStmt(c.cast(u8, c.var("bits"))));
  return fn;
}

// The fp8 unpackers mirror the packers. `infArm` is null for the FNUZ
// encodings, which have no infinity and instead map 0x80 to NaN up front.
inline msl::Function *fp8Unpacker(msl::Context &c, const char *name,
                                  int mantBits, int bias, bool fnuz,
                                  msl::Expr *infCond, msl::Expr *infValue) {
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
  const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
  msl::Function *fn = helperFn(c, name, f32);
  fn->params.push_back({msl::Type::scalar(msl::Scalar::U8), "b", {}});

  msl::Expr *const b = c.cast(u32, c.var("b"));
  msl::Expr *const sgn = c.var("sgn");
  msl::Expr *const e = c.var("e");
  msl::Expr *const m = c.var("m");
  const int expBits = 7 - mantBits;
  const int64_t mantMask = (int64_t(1) << mantBits) - 1;
  const int mantShift = 23 - mantBits;

  const auto bitOr = [&](msl::Expr *x, msl::Expr *y) {
    return c.binary(msl::BinOp::Or, x, y);
  };
  const auto shl = [&](msl::Expr *x, int64_t by) {
    return c.binary(msl::BinOp::Shl, x, c.lit(by));
  };
  const auto compose = [&](msl::Expr *exp) {
    return c.bitcast(f32, bitOr(bitOr(sgn, shl(exp, 23)), shl(m, mantShift)));
  };

  if (fnuz)
    fn->body.push_back(
        c.ifStmt(c.binary(msl::BinOp::Eq, c.var("b"), c.litHex(0x80)),
                 {c.returnStmt(c.bitcast(f32, c.litHex(0x7fc00000)))}));

  fn->body.push_back(c.declStmt(
      u32, "sgn", shl(c.binary(msl::BinOp::And, b, c.litHex(0x80)), 24)));
  fn->body.push_back(c.declStmt(
      u32, "e",
      c.binary(msl::BinOp::And, c.binary(msl::BinOp::Shr, b, c.lit(mantBits)),
               c.litHex((int64_t(1) << expBits) - 1))));
  fn->body.push_back(
      c.declStmt(u32, "m", c.binary(msl::BinOp::And, b, c.litHex(mantMask))));

  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::Eq, e, c.lit(0, u32)),
      {c.ifStmt(c.binary(msl::BinOp::Eq, m, c.lit(0, u32)),
                {c.returnStmt(c.bitcast(f32, sgn))}),
       c.declStmt(i32, "sh", c.lit(0)),
       c.whileStmt(c.binary(msl::BinOp::Eq,
                            c.binary(msl::BinOp::And, m,
                                     c.litHex(int64_t(1) << mantBits)),
                            c.lit(0, u32)),
                   {c.assignOp(msl::BinOp::Shl, m, c.lit(1)),
                    c.assignOp(msl::BinOp::Add, c.var("sh"), c.lit(1))}),
       c.assignOp(msl::BinOp::And, m, c.litHex(mantMask)),
       c.declStmt(u32, "e32",
                  c.cast(u32, c.binary(msl::BinOp::Sub, c.lit(127 - (bias - 1)),
                                       c.var("sh")))),
       c.returnStmt(compose(c.var("e32")))}));

  if (infCond)
    fn->body.push_back(
        c.ifStmt(infCond, {c.returnStmt(c.bitcast(f32, infValue))}));

  fn->body.push_back(c.declStmt(
      u32, "e32",
      c.binary(msl::BinOp::Sub, c.add(e, c.lit(127, u32)), c.lit(bias, u32))));
  fn->body.push_back(c.returnStmt(compose(c.var("e32"))));
  return fn;
}

// The print and assert buffers share a shape: bump a head counter, drop the
// record if the buffer is full, then store one word per field.
inline msl::Function *recordAppender(msl::Context &c, const char *name,
                                     int64_t headWord, int64_t capacity,
                                     int64_t headerWords, int64_t recordWords,
                                     const std::vector<msl::Str> &fields,
                                     const std::vector<int64_t> &offsets) {
  namespace at = msl::builtin::atomic;
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type atomicPtr =
      msl::Type::named(at::Uint).pointerTo(msl::AddrSpace::Device);
  msl::Function *fn = helperFn(c, name, msl::Type::named("void"));
  fn->params.push_back({atomicPtr, "buf", {}});
  for (const msl::Str &f : fields)
    fn->params.push_back({u32, f, {}});

  msl::Expr *const buf = c.var("buf");
  msl::Expr *const relaxed = c.var(msl::builtin::order::Relaxed);

  fn->body.push_back(
      c.declStmt(u32, "slot",
                 c.call(at::FetchAdd, {c.add(buf, c.lit(headWord)),
                                       c.lit(1, u32), relaxed})));
  fn->body.push_back(
      c.ifStmt(c.binary(msl::BinOp::Ge, c.var("slot"), c.lit(capacity, u32)),
               {c.returnStmt()}));
  fn->body.push_back(
      c.declStmt(atomicPtr, "rec",
                 c.add(c.add(buf, c.lit(headerWords)),
                       c.mul(c.var("slot"), c.lit(recordWords)))));
  for (std::size_t i = 0; i < fields.size(); ++i)
    fn->body.push_back(
        c.exprStmt(c.call(at::Store, {c.add(c.var("rec"), c.lit(offsets[i])),
                                      c.var(fields[i]), relaxed})));
  return fn;
}

inline std::string helperSource(Helper h) {
  switch (h) {
  case Helper::AtomicRmwF32: {
    namespace at = msl::builtin::atomic;
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    msl::Function *fn = helperFn(c, hn::AtomicRmwF32, f32);
    fn->params.push_back(
        {msl::Type::named(at::Uint).pointerTo(msl::AddrSpace::Device),
         "p",
         {}});
    fn->params.push_back({f32, "v", {}});
    fn->params.push_back({msl::Type::scalar(msl::Scalar::I32), "op", {}});

    msl::Expr *const relaxed = c.var(msl::builtin::order::Relaxed);
    msl::Expr *const expected = c.var("expected");

    fn->body.push_back(
        c.declStmt(u32, "expected", c.call(at::Load, {c.var("p"), relaxed})));
    fn->body.push_back(c.declStmt(f32, "old"));
    fn->body.push_back(c.whileStmt(
        c.litBool(true),
        {c.assign(c.var("old"), c.bitcast(f32, expected)),
         c.declStmt(f32, "next", emuRmwLadderExpr(c, "old")),
         c.ifStmt(c.call(at::CompareExchangeWeak,
                         {c.var("p"), c.addrOf(expected),
                          c.bitcast(u32, c.var("next")), relaxed, relaxed}),
                  {c.breakStmt()})}));
    fn->body.push_back(c.returnStmt(c.var("old")));
    return renderHelper(fn);
  }

  // Narrows via `__agpu_rtne_int_*`. Under fast-math a `(half)` cast folds
  // away against the next iteration's re-widening `float()`, which
  // accumulates at f32 precision into a 16-bit slot.
  case Helper::AtomicRmwPacked16: {
    namespace at = msl::builtin::atomic;
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type u16 = msl::Type::scalar(msl::Scalar::U16);
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    const msl::Type tTy = msl::Type::named("T");

    msl::Function *decl = helperFn(c, hn::Narrow16, u16);
    decl->templateParams.push_back("T");
    decl->isPrototype = true;
    decl->params.push_back({f32, "v", {}});

    const auto narrowFor = [&](const char *elem, const char *rtne) {
      msl::Function *fn = helperFn(c, hn::Narrow16, u16);
      fn->isSpecialization = true;
      fn->templateArgs.push_back(elem);
      fn->params.push_back({f32, "v", {}});
      fn->body.push_back(c.returnStmt(c.call(rtne, {c.var("v")})));
      return fn;
    };

    msl::Function *fn = helperFn(c, hn::AtomicRmwPacked16, tTy);
    fn->templateParams.push_back("T");
    fn->params.push_back(
        {msl::Type::named(at::Uint).pointerTo(msl::AddrSpace::Device),
         "word",
         {}});
    fn->params.push_back({msl::Type::scalar(msl::Scalar::Bool), "high", {}});
    fn->params.push_back({f32, "v", {}});
    fn->params.push_back({msl::Type::scalar(msl::Scalar::I32), "op", {}});

    msl::Expr *const expected = c.var("expected");
    msl::Expr *const high = c.var("high");
    msl::Expr *const nb = c.var("nb");
    msl::Expr *const relaxed = c.var(msl::builtin::order::Relaxed);
    const auto shr = [&](msl::Expr *e, int64_t by) {
      return c.binary(msl::BinOp::Shr, e, c.lit(by));
    };
    const auto bitAnd = [&](msl::Expr *e, int64_t m) {
      return c.binary(msl::BinOp::And, e, c.litHex(m));
    };

    fn->body.push_back(c.declStmt(u32, "expected",
                                  c.call(at::Load, {c.var("word"), relaxed})));
    fn->body.push_back(c.declStmt(tTy, "old"));
    fn->body.push_back(c.whileStmt(
        c.litBool(true),
        {c.declStmt(u16, "bits",
                    c.ternary(high, c.construct(u16, shr(expected, 16)),
                              c.construct(u16, bitAnd(expected, 0xffff)))),
         c.assign(c.var("old"), c.bitcast(tTy, c.var("bits"))),
         c.declStmt(f32, "cur", c.construct(f32, c.var("old"))),
         c.declStmt(f32, "next", emuRmwLadderExpr(c, "cur")),
         c.declStmt(u16, "nb", c.call(hn::Narrow16, {"T"}, {c.var("next")})),
         c.declStmt(
             u32, "merged",
             c.ternary(high,
                       c.binary(msl::BinOp::Or, bitAnd(expected, 0x0000ffff),
                                c.binary(msl::BinOp::Shl, c.construct(u32, nb),
                                         c.lit(16))),
                       c.binary(msl::BinOp::Or, bitAnd(expected, 0xffff0000),
                                c.construct(u32, nb)))),
         c.ifStmt(c.call(at::CompareExchangeWeak,
                         {c.var("word"), c.addrOf(expected), c.var("merged"),
                          relaxed, relaxed}),
                  {c.breakStmt()})}));
    fn->body.push_back(c.returnStmt(c.var("old")));

    return renderHelper(decl) + "\n" +
           renderHelper(narrowFor("half", hn::RtneIntHalf)) + "\n" +
           renderHelper(narrowFor("bfloat", hn::RtneIntBfloat)) + "\n\n" +
           renderHelper(fn);
  }

  // Metal has no cbrt. `pow(|x|, 1/3)` needs the sign taken out and restored,
  // plus one Newton step for the last few ulp. The zero guard avoids 0/0 in
  // that step and preserves the sign of -0.0.
  case Helper::Cbrt: {
    msl::Context c;
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    msl::Function *fn = helperFn(c, hn::Cbrt, f32);
    fn->params.push_back({f32, "x", {}});

    msl::Expr *const x = c.var("x");
    msl::Expr *const a = c.var("a");
    msl::Expr *const y = c.var("y");
    msl::Expr *const third =
        c.binary(msl::BinOp::Div, c.litF(1.0), c.litF(3.0));

    fn->body.push_back(
        c.declStmt(f32, "a", c.call(msl::builtin::math::Abs, {x})));
    fn->body.push_back(
        c.ifStmt(c.binary(msl::BinOp::Eq, a, c.litF(0.0)), {c.returnStmt(x)}));
    fn->body.push_back(
        c.declStmt(f32, "y",
                   c.call(spell(msl::builtin::accuracy::Pow, Accuracy::Exact),
                          {a, third})));
    fn->body.push_back(c.assign(
        y, c.binary(msl::BinOp::Sub, y,
                    c.mul(c.binary(msl::BinOp::Sub, y,
                                   c.binary(msl::BinOp::Div, a, c.mul(y, y))),
                          third))));
    fn->body.push_back(
        c.returnStmt(c.ternary(c.binary(msl::BinOp::Lt, x, c.litF(0.0)),
                               c.unary(msl::UnOp::Neg, y), y)));
    return renderHelper(fn);
  }

  // Metal has no erf. Abramowitz-Stegun 7.1.26.
  case Helper::Erf: {
    msl::Context c;
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    msl::Function *fn = helperFn(c, hn::Erf, f32);
    fn->params.push_back({f32, "x", {}});

    msl::Expr *const x = c.var("x");
    msl::Expr *const a = c.var("a");
    msl::Expr *const t = c.var("t");

    // Horner, innermost coefficient first.
    msl::Expr *poly = c.litF(1.061405429);
    for (const double k :
         {-1.453152027, 1.421413741, -0.284496736, 0.254829592}) {
      poly = c.mul(poly, t);
      poly = c.binary(k < 0 ? msl::BinOp::Sub : msl::BinOp::Add, poly,
                      c.litF(k < 0 ? -k : k));
    }

    fn->body.push_back(
        c.declStmt(f32, "s", c.call(msl::builtin::math::Sign, {x})));
    fn->body.push_back(
        c.declStmt(f32, "a", c.call(msl::builtin::math::Abs, {x})));
    fn->body.push_back(
        c.declStmt(f32, "t",
                   c.binary(msl::BinOp::Div, c.litF(1.0),
                            c.add(c.litF(1.0), c.mul(c.litF(0.3275911), a)))));
    fn->body.push_back(c.declStmt(
        f32, "y",
        c.binary(
            msl::BinOp::Sub, c.litF(1.0),
            c.mul(c.mul(poly, t),
                  c.call(spell(msl::builtin::accuracy::Exp, Accuracy::Tolerant),
                         {c.mul(c.unary(msl::UnOp::Neg, a), a)})))));
    fn->body.push_back(c.returnStmt(c.mul(c.var("s"), c.var("y"))));
    return renderHelper(fn);
  }

  // Round-to-nearest-even, spelled out because MSL's `(half)` cast does not
  // guarantee NaN/Inf/subnormal handling under fast-math. Returns bits rather
  // than `half` so the CAS loops' result is not a fold candidate.
  case Helper::RtneIntHalf: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type u16 = msl::Type::scalar(msl::Scalar::U16);
    const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
    msl::Function *fn = helperFn(c, hn::RtneIntHalf, u16);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

    msl::Expr *const u = c.var("u");
    msl::Expr *const sgn = c.var("sgn");
    msl::Expr *const mant = c.var("mant");
    msl::Expr *const ex = c.var("ex");
    msl::Expr *const m = c.var("m");
    msl::Expr *const rem = c.var("rem");
    const auto shr = [&](msl::Expr *e, msl::Expr *by) {
      return c.binary(msl::BinOp::Shr, e, by);
    };
    const auto bitAnd = [&](msl::Expr *e, msl::Expr *k) {
      return c.binary(msl::BinOp::And, e, k);
    };
    const auto bitOr = [&](msl::Expr *a, msl::Expr *b) {
      return c.binary(msl::BinOp::Or, a, b);
    };
    const auto odd = [&] { return bitAnd(m, c.lit(1, u32)); };

    fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
    fn->body.push_back(
        c.declStmt(u32, "sgn", bitAnd(shr(u, c.lit(16)), c.litHex(0x8000))));
    fn->body.push_back(c.declStmt(
        i32, "e32", c.cast(i32, bitAnd(shr(u, c.lit(23)), c.litHex(0xff)))));
    fn->body.push_back(c.declStmt(u32, "mant", bitAnd(u, c.litHex(0x7fffff))));
    fn->body.push_back(c.ifStmt(
        c.binary(msl::BinOp::Eq, c.var("e32"), c.litHex(0xff)),
        {c.returnStmt(c.cast(
            u16, bitOr(bitOr(sgn, c.litHex(0x7c00)),
                       c.ternary(mant, c.litHex(0x200), c.lit(0, u32)))))}));
    fn->body.push_back(c.declStmt(
        i32, "ex",
        c.add(c.binary(msl::BinOp::Sub, c.var("e32"), c.lit(127)), c.lit(15))));
    fn->body.push_back(
        c.ifStmt(c.binary(msl::BinOp::Ge, ex, c.lit(31)),
                 {c.returnStmt(c.cast(u16, bitOr(sgn, c.litHex(0x7c00))))}));

    msl::Expr *const sh = c.var("sh");
    msl::Expr *const fm = c.var("fm");
    msl::Expr *const halfWay = c.var("half_");
    fn->body.push_back(c.ifStmt(
        c.binary(msl::BinOp::Le, ex, c.lit(0)),
        {c.ifStmt(c.binary(msl::BinOp::Lt, ex, c.lit(-10)),
                  {c.returnStmt(c.cast(u16, sgn))}),
         c.declStmt(u32, "fm", bitOr(mant, c.litHex(0x800000))),
         c.declStmt(i32, "sh", c.binary(msl::BinOp::Sub, c.lit(14), ex)),
         c.declStmt(u32, "m", shr(fm, sh)),
         c.declStmt(
             u32, "rem",
             bitAnd(fm, c.binary(msl::BinOp::Sub,
                                 c.binary(msl::BinOp::Shl, c.lit(1, u32), sh),
                                 c.lit(1, u32)))),
         c.declStmt(u32, "half_",
                    c.binary(msl::BinOp::Shl, c.lit(1, u32),
                             c.binary(msl::BinOp::Sub, sh, c.lit(1)))),
         c.ifStmt(
             c.binary(msl::BinOp::LOr, c.binary(msl::BinOp::Gt, rem, halfWay),
                      c.binary(msl::BinOp::LAnd,
                               c.binary(msl::BinOp::Eq, rem, halfWay), odd())),
             {c.assignOp(msl::BinOp::Add, m, c.lit(1))}),
         c.returnStmt(c.cast(u16, bitOr(sgn, m)))}));

    fn->body.push_back(c.declStmt(u32, "m", shr(mant, c.lit(13))));
    fn->body.push_back(c.declStmt(u32, "rem", bitAnd(mant, c.litHex(0x1fff))));
    fn->body.push_back(c.declStmt(
        u32, "bits",
        bitOr(bitOr(sgn, c.binary(msl::BinOp::Shl, c.cast(u32, ex), c.lit(10))),
              m)));
    fn->body.push_back(c.ifStmt(
        c.binary(
            msl::BinOp::LOr, c.binary(msl::BinOp::Gt, rem, c.litHex(0x1000)),
            c.binary(msl::BinOp::LAnd,
                     c.binary(msl::BinOp::Eq, rem, c.litHex(0x1000)), odd())),
        {c.assignOp(msl::BinOp::Add, c.var("bits"), c.lit(1))}));
    fn->body.push_back(c.returnStmt(c.cast(u16, c.var("bits"))));
    return renderHelper(fn);
  }

  case Helper::RtneHalf: {
    msl::Context c;
    msl::Function *fn =
        helperFn(c, hn::RtneHalf, msl::Type::scalar(msl::Scalar::F16));
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});
    fn->body.push_back(
        c.returnStmt(c.bitcast(msl::Type::scalar(msl::Scalar::F16),
                               c.call(hn::RtneIntHalf, {c.var("v")}))));
    return renderHelper(fn);
  }

  // bfloat is f32's top 16 bits. The NaN arm is separate because the round
  // increment can carry a quiet NaN's payload into an infinity.
  case Helper::RtneIntBfloat: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type u16 = msl::Type::scalar(msl::Scalar::U16);
    msl::Function *fn = helperFn(c, hn::RtneIntBfloat, u16);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

    msl::Expr *const u = c.var("u");
    msl::Expr *const mant = c.var("mant");
    const auto shr = [&](msl::Expr *e, int64_t by) {
      return c.binary(msl::BinOp::Shr, e, c.lit(by));
    };
    const auto andHex = [&](msl::Expr *e, int64_t m) {
      return c.binary(msl::BinOp::And, e, c.litHex(m));
    };

    fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
    fn->body.push_back(c.ifStmt(
        c.binary(msl::BinOp::Eq, andHex(shr(u, 23), 0xff), c.litHex(0xff)),
        {c.declStmt(u32, "mant", andHex(u, 0x7fffff)),
         c.returnStmt(c.cast(
             u16, c.binary(msl::BinOp::Or, andHex(shr(u, 16), 0xff80),
                           c.ternary(mant, c.litHex(0x40),
                                     c.lit(0, msl::Type::scalar(
                                                  msl::Scalar::U32))))))}));
    fn->body.push_back(c.declStmt(
        u32, "lsb", c.binary(msl::BinOp::And, shr(u, 16), c.lit(1, u32))));
    fn->body.push_back(c.declStmt(
        u32, "rounded", c.add(c.add(u, c.litHex(0x7fff)), c.var("lsb"))));
    fn->body.push_back(c.returnStmt(c.cast(u16, shr(c.var("rounded"), 16))));
    return renderHelper(fn);
  }

  case Helper::RtneBfloat: {
    msl::Context c;
    msl::Function *fn =
        helperFn(c, hn::RtneBfloat, msl::Type::scalar(msl::Scalar::BF16));
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});
    fn->body.push_back(
        c.returnStmt(c.bitcast(msl::Type::scalar(msl::Scalar::BF16),
                               c.call(hn::RtneIntBfloat, {c.var("v")}))));
    return renderHelper(fn);
  }

  // Round-toward-zero. MSL's cast is always nearest-even, so this truncates
  // the mantissa directly. Overflow saturates to 0x7bff (largest finite
  // half): RTZ cannot turn a finite input into inf (0x7c00).
  case Helper::RtzHalf: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type u16 = msl::Type::scalar(msl::Scalar::U16);
    const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
    const msl::Type f16 = msl::Type::scalar(msl::Scalar::F16);
    msl::Function *fn = helperFn(c, hn::RtzHalf, f16);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

    msl::Expr *const u = c.var("u");
    msl::Expr *const sgn = c.var("sgn");
    msl::Expr *const ex = c.var("ex");
    msl::Expr *const mant = c.var("mant");
    msl::Expr *const bits = c.var("bits");
    const auto shr = [&](msl::Expr *e, int64_t by) {
      return c.binary(msl::BinOp::Shr, e, c.lit(by));
    };
    const auto orAll = [&](std::vector<msl::Expr *> parts) {
      return c.chain(msl::BinOp::Or, parts);
    };
    const auto setBits = [&](msl::Expr *e) {
      return c.assign(bits, c.cast(u16, e));
    };
    msl::Expr *const exponent =
        c.binary(msl::BinOp::And, shr(u, 23), c.litHex(0xff));

    fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
    fn->body.push_back(c.declStmt(u16, "bits"));
    fn->body.push_back(c.declStmt(
        u32, "sgn", c.binary(msl::BinOp::And, shr(u, 16), c.litHex(0x8000))));
    fn->body.push_back(c.declStmt(
        i32, "ex",
        c.binary(msl::BinOp::Sub, c.cast(i32, exponent), c.lit(112))));
    fn->body.push_back(c.declStmt(
        u32, "mant", c.binary(msl::BinOp::And, u, c.litHex(0x7fffff))));

    msl::Block subnormal{c.ifElse(
        c.binary(msl::BinOp::Lt, ex, c.lit(-10)), {setBits(sgn)},
        {c.declStmt(u32, "m",
                    c.binary(msl::BinOp::Shr,
                             c.binary(msl::BinOp::Or, mant, c.litHex(0x800000)),
                             c.binary(msl::BinOp::Sub, c.lit(14), ex))),
         setBits(c.binary(msl::BinOp::Or, sgn, c.var("m")))})};

    fn->body.push_back(c.ifElse(
        c.binary(msl::BinOp::Eq, exponent, c.litHex(0xff)),
        {setBits(orAll({sgn, c.litHex(0x7c00),
                        c.ternary(mant, c.litHex(0x200), c.lit(0, u32))}))},
        {c.ifElse(
            c.binary(msl::BinOp::Ge, ex, c.lit(31)),
            {setBits(c.binary(msl::BinOp::Or, sgn, c.litHex(0x7bff)))},
            {c.ifElse(
                c.binary(msl::BinOp::Le, ex, c.lit(0)), subnormal,
                {setBits(orAll(
                    {sgn, c.binary(msl::BinOp::Shl, c.cast(u32, ex), c.lit(10)),
                     shr(mant, 13)}))})})}));
    fn->body.push_back(c.returnStmt(c.bitcast(f16, bits)));
    return renderHelper(fn);
  }

  // bfloat is f32's top 16 bits, so RTZ is a plain truncation with no
  // exponent rebiasing.
  case Helper::RtzBfloat: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type bf16 = msl::Type::scalar(msl::Scalar::BF16);
    msl::Function *fn = helperFn(c, hn::RtzBfloat, bf16);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

    fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
    fn->body.push_back(c.returnStmt(c.bitcast(
        bf16, c.cast(msl::Type::scalar(msl::Scalar::U16),
                     c.binary(msl::BinOp::And,
                              c.binary(msl::BinOp::Shr, c.var("u"), c.lit(16)),
                              c.litHex(0xffff))))));
    return renderHelper(fn);
  }

  // fp8 has no MSL type and travels as a byte. Rounds to nearest-even and
  // handles subnormals. e4m3fn's top slot is NaN, so saturation lands on
  // the half-way mark below 448, since 0x7f is unavailable as a finite max.
  case Helper::Fp8PackE4M3: {
    msl::Context c;
    msl::Expr *const nearMax = c.binary(
        msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, c.var("ex"), c.lit(15)),
        c.binary(msl::BinOp::Gt, c.var("mant"), c.litHex(0x600000)));
    return renderHelper(fp8Packer(c, hn::Fp8PackE4M3, /*mantBits=*/3,
                                  /*bias=*/7, /*satExp=*/16,
                                  /*subnormalShift=*/21, /*subnormalFloor=*/-6,
                                  /*nanHi=*/0x7f, /*satHi=*/0x7e,
                                  /*quiet=*/0, nearMax, /*biasForm=*/false));
  }

  case Helper::Fp8UnpackE4M3: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    msl::Expr *const isNan = c.binary(
        msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, c.var("e"), c.litHex(0xf)),
        c.binary(msl::BinOp::Eq, c.var("m"), c.litHex(0x7)));
    msl::Expr *const nanBits =
        c.binary(msl::BinOp::Or,
                 c.binary(msl::BinOp::Or, c.var("sgn"), c.litHex(0x7f800000)),
                 c.litHex(0x400000));
    return renderHelper(fp8Unpacker(c, hn::Fp8UnpackE4M3, /*mantBits=*/3,
                                    /*bias=*/7, /*fnuz=*/false, isNan,
                                    nanBits));
  }

  case Helper::Fp8PackE5M2: {
    msl::Context c;
    return renderHelper(fp8Packer(c, hn::Fp8PackE5M2, /*mantBits=*/2,
                                  /*bias=*/15, /*satExp=*/31,
                                  /*subnormalShift=*/22, /*subnormalFloor=*/-2,
                                  /*nanHi=*/0x7c, /*satHi=*/0x7c,
                                  /*quiet=*/0x2, nullptr, /*biasForm=*/false));
  }

  case Helper::Fp8UnpackE5M2: {
    msl::Context c;
    msl::Expr *const isInf =
        c.binary(msl::BinOp::Eq, c.var("e"), c.litHex(0x1f));
    msl::Expr *const infBits =
        c.binary(msl::BinOp::Or,
                 c.binary(msl::BinOp::Or, c.var("sgn"), c.litHex(0x7f800000)),
                 c.binary(msl::BinOp::Shl, c.var("m"), c.lit(21)));
    return renderHelper(fp8Unpacker(c, hn::Fp8UnpackE5M2, /*mantBits=*/2,
                                    /*bias=*/15, /*fnuz=*/false, isInf,
                                    infBits));
  }

  // FNUZ variants e4b8/e5b16: no inf/NaN (0x7f is the finite max), 0x80 is
  // the NaN encoding so signed zero must pack to +0 and the bias is one
  // larger than the encoding they otherwise mirror (8 vs 7, 16 vs 15).
  case Helper::Fp8PackE4B8: {
    msl::Context c;
    msl::Expr *const nearMax = c.binary(
        msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, c.var("ex"), c.lit(15)),
        c.binary(msl::BinOp::Gt, c.var("mant"), c.litHex(0x700000)));
    return renderHelper(fp8Packer(c, hn::Fp8PackE4B8, /*mantBits=*/3,
                                  /*bias=*/8, /*satExp=*/16,
                                  /*subnormalShift=*/21, /*subnormalFloor=*/-6,
                                  /*nanHi=*/0x7f, /*satHi=*/0x7f,
                                  /*quiet=*/0, nearMax, /*biasForm=*/true));
  }

  case Helper::Fp8UnpackE4B8: {
    msl::Context c;
    return renderHelper(fp8Unpacker(c, hn::Fp8UnpackE4B8, /*mantBits=*/3,
                                    /*bias=*/8, /*fnuz=*/true, nullptr,
                                    nullptr));
  }

  case Helper::Fp8PackE5B16: {
    msl::Context c;
    msl::Expr *const nearMax = c.binary(
        msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, c.var("ex"), c.lit(31)),
        c.binary(msl::BinOp::Gt, c.var("mant"), c.litHex(0x600000)));
    return renderHelper(fp8Packer(c, hn::Fp8PackE5B16, /*mantBits=*/2,
                                  /*bias=*/16, /*satExp=*/32,
                                  /*subnormalShift=*/22, /*subnormalFloor=*/-2,
                                  /*nanHi=*/0x7f, /*satHi=*/0x7f,
                                  /*quiet=*/0, nearMax, /*biasForm=*/true));
  }

  case Helper::Fp8UnpackE5B16: {
    msl::Context c;
    return renderHelper(fp8Unpacker(c, hn::Fp8UnpackE5B16, /*mantBits=*/2,
                                    /*bias=*/16, /*fnuz=*/true, nullptr,
                                    nullptr));
  }

  // e2m1: one sign bit, two exponent bits, one mantissa bit, bias 1. All
  // sixteen values fit in a table and there is no inf/NaN.
  case Helper::Fp4Unpack: {
    msl::Context c;
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    msl::Function *fn = helperFn(c, hn::Fp4Unpack, f32);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::U8), "nib", {}});

    msl::SmallVec<msl::Expr *, 4> values;
    for (const double v : {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0})
      values.push_back(c.litF(v));
    for (const double v : {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0})
      values.push_back(c.unary(msl::UnOp::Neg, c.litF(v)));

    fn->body.push_back(
        c.arrayDecl(f32.withQual(msl::Type::Const), "kV", std::move(values)));
    fn->body.push_back(c.returnStmt(c.subscript(
        c.var("kV"),
        c.binary(msl::BinOp::And,
                 c.cast(msl::Type::scalar(msl::Scalar::U32), c.var("nib")),
                 c.litHex(0xf)))));
    return renderHelper(fn);
  }

  // Offsets come from `PrintField`, kept in sync with `PrintPlan`. The head
  // bump happens before the bounds test so the host can count lost records.
  case Helper::PrintAppend: {
    msl::Context c;
    return renderHelper(recordAppender(
        c, hn::PrintAppend, printHeaderWord(PrintHeader::Head),
        kPrintBufferRecords, kPrintHeaderWords, kPrintRecordWords,
        {"site", "pid", "tid", "index", "type", "value", "operand"},
        {printFieldWord(PrintField::Site), printFieldWord(PrintField::Pid),
         printFieldWord(PrintField::Tid), printFieldWord(PrintField::Index),
         printFieldWord(PrintField::Type), printFieldWord(PrintField::Value),
         printFieldWord(PrintField::Operand)}));
  }

  // Offsets come from `AssertField`, kept in sync with `assertLayoutText`.
  // No barrier: only failing threads reach here and a barrier in divergent
  // control flow is undefined in Metal.
  case Helper::AssertRecord: {
    msl::Context c;
    return renderHelper(recordAppender(
        c, hn::AssertRecord, assertHeaderWord(AssertHeader::Head),
        kAssertBufferRecords, kAssertHeaderWords, kAssertRecordWords,
        {"site", "pid", "tid"},
        {assertFieldWord(AssertField::Site), assertFieldWord(AssertField::Pid),
         assertFieldWord(AssertField::Tid)}));
  }
  case Helper::Count:
    break;
  }
  return {};
}

} // namespace agpu

#endif // AGPU_PRELUDE_H
