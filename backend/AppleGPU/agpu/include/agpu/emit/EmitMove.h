// EmitMove.h - loads and stores, one guarded access per register.
//
// The site hands over expressions: accesses are `base[off]`, no pointer
// materialised. Callbacks mint fresh AST per call, so
// no node is shared between statements.
#ifndef AGPU_EMIT_MOVE_H
#define AGPU_EMIT_MOVE_H

#include "agpu/emit/EmitPoison.h"
#include "agpu/msl/Context.h"

#include <functional>

namespace agpu {

struct MoveSite {
  // The lvalue for register r's element: `base[off]`, or `*p`.
  std::function<msl::Expr *(int64_t)> elem;

  // The condition register r's access runs under, or null for "always".
  std::function<msl::Expr *(int64_t)> guard;

  // The IR-supplied value for a lane the mask excludes.
  std::function<msl::Expr *(int64_t)> other;

  // One name per register, broadcasting at size one. Loads declare these;
  // stores read them.
  msl::SmallVec<msl::Str, 8> values;
};

struct MoveFacts {
  int64_t regCount = 1;
  bool hasMask = false;
  bool hasOther = false; // a value for lanes the mask excludes
  bool isStore = false;
};

inline const msl::Str &at(const msl::SmallVec<msl::Str, 8> &v, int64_t r) {
  return v[v.size() == 1 ? 0 : (std::size_t)r];
}

inline void emitMove(msl::Context &c, msl::Block &body, const MoveFacts &f,
                     const MoveSite &site, ElemType elem) {
  // Every register must be defined before the mask is consulted: the mask is
  // a runtime value, so a lane it excludes still reads its name.
  const bool seeds = !f.isStore && f.hasMask;
  if (seeds)
    for (int64_t r = 0; r < f.regCount; ++r) {
      msl::Expr *seed = f.hasOther ? site.other(r) : poisonValue(c, elem);
      body.push_back(c.declStmt(mslTypeOf(elem), at(site.values, r), seed));
    }

  for (int64_t r = 0; r < f.regCount; ++r) {
    msl::Expr *cond = f.hasMask && site.guard ? site.guard(r) : nullptr;
    msl::Stmt *access;
    if (f.isStore)
      access = c.assign(site.elem(r), c.var(at(site.values, r)));
    else if (seeds)
      access = c.assign(c.var(at(site.values, r)), site.elem(r));
    else
      access = c.declStmt(mslTypeOf(elem), at(site.values, r), site.elem(r));
    if (msl::Stmt *s = c.guarded(cond, access))
      body.push_back(s);
  }
}

} // namespace agpu

#endif // AGPU_EMIT_MOVE_H
