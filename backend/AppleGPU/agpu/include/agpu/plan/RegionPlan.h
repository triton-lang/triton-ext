// RegionPlan.h - lowering unstructured control flow to a dispatch loop.
//
// A region of basic blocks with arbitrary branches has no MSL goto, so the
// blocks become numbered states and a branch becomes an assignment to the
// state variable, dispatched by an enclosing loop.
//
// A value defined in one block and read in another is declared outside the
// machine. A block argument is a phi: the edge copies into the destination's
// variables before assigning the state.
#ifndef AGPU_REGION_PLAN_H
#define AGPU_REGION_PLAN_H

#include "agpu/core/Decline.h"
#include "agpu/core/ValueId.h"

#include <cstdint>
#include <vector>

namespace agpu {

// `ValueId` and `BlockId` live in core/ValueId.h because bind/SymbolTable.h
// needs them too and neither layer may include the other.

// What one block's terminator does.
enum class TermKind {
  Branch,     // one successor
  CondBranch, // two, chosen by a condition
  Return,     // leaves the region
};

// `args` are the values copied into the destination block's parameters, in
// parameter order. The phi lives on the edge.
struct Edge {
  BlockId to = 0;
  std::vector<ValueId> args;
};

// One block of the region.
struct BlockFacts {
  // Values this block's parameters bind, in order. Every incoming edge
  // supplies exactly this many arguments.
  std::vector<ValueId> params;

  std::vector<ValueId> defines;
  std::vector<ValueId> reads;

  TermKind term = TermKind::Return;
  std::vector<Edge> edges; // 1 for Branch, 2 for CondBranch, 0 for Return
};

struct RegionFacts {
  std::vector<BlockFacts> blocks;
  BlockId entry = 0;
};

// What the emitter must declare before the machine and what each edge copies.
struct RegionPlan {
  // Values declared outside the dispatch loop: read by a block other than
  // the one defining it, plus every block parameter.
  std::vector<ValueId> hoisted;

  // The state each block is dispatched by: index into `blocks`, also the
  // case value.
  std::vector<int64_t> stateOf;

  int64_t exitState = -1;

  bool usable = true;
};

namespace detail {

inline bool contains(const std::vector<ValueId> &v, ValueId x) {
  for (ValueId e : v)
    if (e == x)
      return true;
  return false;
}

inline void addOnce(std::vector<ValueId> &v, ValueId x) {
  if (!contains(v, x))
    v.push_back(x);
}

} // namespace detail

// Whether a value defined in `from` is read anywhere else. Not "read after
// the definition": the machine can re-enter, so an earlier block counts.
inline bool crossesBlocks(const RegionFacts &f, BlockId from, ValueId v) {
  for (BlockId b = 0; b < (BlockId)f.blocks.size(); ++b) {
    if (b == from)
      continue;
    if (detail::contains(f.blocks[b].reads, v))
      return true;
    // An edge argument is a read too: the copy happens in the source block.
    for (const Edge &e : f.blocks[b].edges)
      if (detail::contains(e.args, v))
        return true;
  }
  return false;
}

inline RegionPlan planRegion(const RegionFacts &f) {
  RegionPlan p;
  if (f.blocks.empty()) {
    p.usable = false;
    return p;
  }

  for (BlockId b = 0; b < (BlockId)f.blocks.size(); ++b)
    p.stateOf.push_back(b);

  // Block parameters are always hoisted: an edge writes them from a
  // different case body than the one that reads them.
  for (const BlockFacts &b : f.blocks)
    for (ValueId v : b.params)
      detail::addOnce(p.hoisted, v);

  for (BlockId b = 0; b < (BlockId)f.blocks.size(); ++b)
    for (ValueId v : f.blocks[b].defines)
      if (crossesBlocks(f, b, v))
        detail::addOnce(p.hoisted, v);

  return p;
}

// Edge counts and argument counts must match the terminator and destination.
inline Decision regionDecision(const RegionFacts &f, const RegionPlan &p) {
  if (!p.usable)
    return Decision::declined("region", "no blocks");

  for (const BlockFacts &b : f.blocks) {
    const std::size_t want = b.term == TermKind::CondBranch ? 2
                             : b.term == TermKind::Branch   ? 1
                                                            : 0;
    if (b.edges.size() != want)
      return Decision::failed();

    for (const Edge &e : b.edges) {
      if (e.to < 0 || e.to >= (BlockId)f.blocks.size())
        return Decision::failed();
      if (e.args.size() != f.blocks[e.to].params.size())
        return Decision::failed();
    }
  }
  return Decision::emitted();
}

// An unreachable block is legal to emit, but usually means the caller built
// the edges wrong.
inline std::vector<bool> reachableBlocks(const RegionFacts &f) {
  std::vector<bool> seen(f.blocks.size(), false);
  if (f.blocks.empty())
    return seen;

  std::vector<BlockId> work{f.entry};
  seen[(std::size_t)f.entry] = true;
  while (!work.empty()) {
    const BlockId b = work.back();
    work.pop_back();
    for (const Edge &e : f.blocks[(std::size_t)b].edges)
      if (!seen[(std::size_t)e.to]) {
        seen[(std::size_t)e.to] = true;
        work.push_back(e.to);
      }
  }
  return seen;
}

} // namespace agpu

#endif // AGPU_REGION_PLAN_H
