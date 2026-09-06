// GuardSink.h - moving a guarded store down to meet an identically-guarded
// one, so GuardFuse.h can take over. Fusing only merges adjacent identical
// guards:
//
//   if (c) out[i] = x;
//   tmp = y;              <- blocks the fusion
//   if (c) out[j] = z;
//
// The move stops when the intervening statement (1) writes a name the guard
// reads, (2) may write an address the sunk store also writes, (3) writes a
// name the sunk store reads, or (4) is not understood. Case (4) is the
// default, so an unhandled shape stays put.
//
// What it buys: on a masked store of n registers, the shape a vectorised
// store emits, this collapses n guards to 1. GuardFuse alone collapses none
// of them, because the interleaved temps break adjacency.
#ifndef AGPU_SINK_H
#define AGPU_SINK_H

#include "agpu/core/CoordSet.h"
#include "agpu/msl/Analysis.h"
#include "agpu/msl/Ast.h"
#include "agpu/msl/GuardFuse.h"

namespace agpu {
namespace msl {

// Why a sink stopped where it did.
enum class SinkStop {
  Merged,       // reached an identically-guarded statement
  ClobbersCond, // (1) writes a name the guard reads
  MayAlias,     // (2) writes an address the store may write
  FeedsStore,   // (3) writes a name the store reads
  Opaque,       // (4) not understood
  NoTarget,     // ran out of block with no matching guard
};

// The address a store writes, as a set. False for a statement that is not a
// simple `buf[idx] = v` (Opaque case).
template <typename CoordsFn>
inline bool storeTarget(const Stmt *s, Str &buf, CoordSet &where,
                        CoordsFn coordsOf) {
  if (!s || s->kind != StmtKind::Assign)
    return false;
  const auto *a = static_cast<const Assign *>(s);
  if (a->compound || a->target->kind != ExprKind::Subscript)
    return false;
  const auto *sub = static_cast<const Subscript *>(a->target);
  if (sub->base->kind != ExprKind::VarRef)
    return false;
  buf = static_cast<const VarRef *>(sub->base)->name;
  where = coordsOf(sub->index);
  return true;
}

// Returns the index the statement at `from` may move to. `Merged` means that
// index holds a statement with the same guard; any other stop means no move.
template <typename CoordsFn>
inline SinkStop sinkTarget(const Block &b, std::size_t from, std::size_t &to,
                           CoordsFn coordsOf) {
  to = from;
  if (from >= b.size() || !b[from] || b[from]->kind != StmtKind::If)
    return SinkStop::Opaque;
  const auto *guard = static_cast<const If *>(b[from]);
  if (guard->thenBody.size() != 1 || guard->hasElse())
    return SinkStop::Opaque;

  Str buf;
  CoordSet where;
  if (!storeTarget(guard->thenBody[0], buf, where, coordsOf))
    return SinkStop::Opaque;

  const ReadNames cond = namesRead(guard->cond);
  const ReadNames store =
      namesRead(static_cast<Assign *>(guard->thenBody[0])->value);
  if (cond.opaque || store.opaque)
    return SinkStop::Opaque;

  for (std::size_t j = from + 1; j < b.size(); ++j) {
    Stmt *s = b[j];
    if (!s)
      return SinkStop::Opaque;

    if (s->kind == StmtKind::If &&
        exprsEqual(static_cast<const If *>(s)->cond, guard->cond)) {
      to = j;
      return SinkStop::Merged;
    }

    // Before the name checks: `writesTo` treats any subscript target as an
    // escape, so it cannot answer the aliasing question.
    Str otherBuf;
    CoordSet other;
    if (storeTarget(s, otherBuf, other, coordsOf)) {
      if (otherBuf == buf && !provablyDisjoint(where, other))
        return SinkStop::MayAlias;
      continue;
    }

    if (writesTo(s, cond.names))
      return SinkStop::ClobbersCond;
    if (writesTo(s, store.names))
      return SinkStop::FeedsStore;

    if (s->kind != StmtKind::Decl && s->kind != StmtKind::Assign)
      return SinkStop::Opaque;
  }
  return SinkStop::NoTarget;
}

// Sink every guarded store that can reach a matching guard and report how
// many moved. Recurses into nested blocks.
template <typename CoordsFn>
inline int sinkGuardedStores(Block &b, CoordsFn coordsOf) {
  int moved = 0;
  for (Stmt *s : b)
    forEachChildBlock(s, [&](Block &nested) {
      moved += sinkGuardedStores(nested, coordsOf);
    });
  std::size_t i = 0;
  while (i + 1 < b.size()) {
    std::size_t to = 0;
    if (sinkTarget(b, i, to, coordsOf) != SinkStop::Merged || to <= i + 1) {
      ++i;
      continue;
    }
    Stmt *s = b[i];
    b.erase(b.begin() + (long)i);
    b.insert(b.begin() + (long)(to - 1), s);
    ++moved;
  }
  return moved;
}

} // namespace msl
} // namespace agpu

#endif // AGPU_SINK_H
