// EmitAtomicAccess.h - an atomic load or store, emitted.
#ifndef AGPU_EMIT_ATOMIC_ACCESS_H
#define AGPU_EMIT_ATOMIC_ACCESS_H

#include "agpu/core/Names.h"
#include "agpu/emit/Prelude.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/AtomicAccessPlan.h"

#include <functional>

namespace agpu {

struct AtomicAccessNames : ThreadNames {
  msl::Str result = "ld";
  // The byte or half a sub-word access occupies within its word.
  msl::Str shift = "sh";
};

inline msl::Type atomicAccessPtrType(const AtomicAccessPlan &p) {
  if (p.wide)
    return msl::Type::scalar(p.word).pointerTo(msl::AddrSpace::Device,
                                               msl::Type::Volatile);
  return msl::deviceAtomicPtr(p.word);
}

inline int64_t subWordMask(SubWord s) {
  return s == SubWord::Byte ? 0xff : 0xffff;
}

// The whole word, read once. Relaxed: the fences carry any ordering.
inline msl::Expr *atomicWordLoad(msl::Context &c, const AtomicAccessPlan &p,
                                 const msl::Str &ptr) {
  if (p.wide)
    return c.deref(c.var(ptr));
  return c.call(msl::builtin::atomic::Load,
                {c.var(ptr), c.var(msl::builtin::order::Relaxed)});
}

// The integer that holds one element's bits, narrowed to its own width.
inline msl::Type atomicBitsType(const AtomicAccessPlan &p) {
  switch (p.elem.bits) {
  case 8:
    return msl::Type::scalar(msl::Scalar::U8);
  case 16:
    return msl::Type::scalar(msl::Scalar::U16);
  case 64:
    return msl::Type::scalar(msl::Scalar::U64);
  }
  return msl::Type::scalar(msl::Scalar::U32);
}

// One element's load, typed as the element. A float arrives as the bits of
// the word that carried it, so it is reinterpreted and never converted.
inline msl::Expr *atomicLoadValue(msl::Context &c, const AtomicAccessPlan &p,
                                  const msl::Str &ptr,
                                  const AtomicAccessNames &nm) {
  msl::Expr *word = atomicWordLoad(c, p, ptr);
  if (p.sub != SubWord::None)
    word = c.binary(msl::BinOp::And,
                    c.binary(msl::BinOp::Shr, word, c.var(nm.shift)),
                    c.lit(subWordMask(p.sub)));

  const msl::Type elemTy = mslTypeOf(p.elem);
  if (p.elem.kind != ElemType::Kind::Float)
    return c.cast(elemTy, word);
  return c.bitcast(elemTy, c.cast(atomicBitsType(p), word));
}

// One element's value as the bits to write, never a numeric conversion.
inline msl::Expr *atomicStoreBits(msl::Context &c, const AtomicAccessPlan &p,
                                  const msl::Str &value) {
  if (p.elem.kind != ElemType::Kind::Float)
    return c.cast(msl::Type::scalar(p.word), c.var(value));
  return c.cast(msl::Type::scalar(p.word),
                c.bitcast(atomicBitsType(p), c.var(value)));
}

inline void emitAtomicAccessFenceBefore(msl::Context &c, msl::Block &body,
                                        const AtomicAccessPlan &p) {
  if (p.fences.before)
    body.push_back(c.barrier(msl::Barrier::Scope::Device));
}

inline void emitAtomicAccessFenceAfter(msl::Context &c, msl::Block &body,
                                       const AtomicAccessPlan &p) {
  if (p.fences.after)
    body.push_back(c.barrier(msl::Barrier::Scope::Device));
}

// A store of a value narrower than the word has to leave its neighbours
// alone, so it reads the word, replaces its part and retries until no other
// thread changed the rest.
inline void emitSubWordStore(msl::Context &c, msl::Block &body,
                             const AtomicAccessPlan &p, const msl::Str &ptr,
                             const msl::Str &value,
                             const AtomicAccessNames &nm) {
  const msl::Type wordTy = msl::Type::scalar(p.word);
  const int64_t mask = subWordMask(p.sub);

  const msl::Str keep = nm.result + "_keep";
  const msl::Str want = nm.result + "_want";
  const msl::Str seen = nm.result + "_seen";

  body.push_back(
      c.declStmt(wordTy, keep,
                 c.unary(msl::UnOp::Not, c.binary(msl::BinOp::Shl, c.lit(mask),
                                                  c.var(nm.shift)))));
  body.push_back(
      c.declStmt(wordTy, want,
                 c.binary(msl::BinOp::Shl,
                          c.binary(msl::BinOp::And,
                                   atomicStoreBits(c, p, value), c.lit(mask)),
                          c.var(nm.shift))));
  body.push_back(
      c.declStmt(wordTy, seen,
                 c.call(msl::builtin::atomic::Load,
                        {c.var(ptr), c.var(msl::builtin::order::Relaxed)})));

  body.push_back(c.whileStmt(
      c.unary(
          msl::UnOp::LNot,
          c.call(msl::builtin::atomic::CompareExchangeWeak,
                 {c.var(ptr), c.addrOf(c.var(seen)),
                  c.binary(msl::BinOp::Or,
                           c.binary(msl::BinOp::And, c.var(seen), c.var(keep)),
                           c.var(want)),
                  c.var(msl::builtin::order::Relaxed),
                  c.var(msl::builtin::order::Relaxed)})),
      msl::Block{}));
}

inline void emitAtomicStoreValue(msl::Context &c, msl::Block &body,
                                 const AtomicAccessPlan &p, const msl::Str &ptr,
                                 const msl::Str &value,
                                 const AtomicAccessNames &nm) {
  if (p.sub != SubWord::None) {
    emitSubWordStore(c, body, p, ptr, value, nm);
    return;
  }
  if (p.wide) {
    body.push_back(c.assign(c.deref(c.var(ptr)), c.var(value)));
    return;
  }
  body.push_back(c.exprStmt(c.call(msl::builtin::atomic::Store,
                                   {c.var(ptr), atomicStoreBits(c, p, value),
                                    c.var(msl::builtin::order::Relaxed)})));
}

} // namespace agpu

#endif // AGPU_EMIT_ATOMIC_ACCESS_H
