// EmitPrune.h - deleting the statements nothing reads.
//
// `Analysis.h` finds them. Metal's optimiser drops dead registers but not the
// barriers a redundant `convert_layout` emits ahead of a dot.
#ifndef AGPU_EMIT_PRUNE_H
#define AGPU_EMIT_PRUNE_H

#include "agpu/msl/Analysis.h"
#include "agpu/msl/AstWalk.h"

namespace agpu {

inline void pruneDead(msl::Block &body) {
  const msl::SmallVec<msl::Stmt *, 8> dead = msl::findDeadDecls(body);
  if (dead.empty())
    return;
  msl::eraseStmts(body, msl::PtrSet<msl::Stmt *>(dead.begin(), dead.end()));
}

} // namespace agpu

#endif // AGPU_EMIT_PRUNE_H
