// Predicated select arms: work only one arm of a select reads runs under that
// arm's condition, so a simdgroup whose lanes all take the other arm skips it.
#include "AgpuEmitter.h"

#include "llvm/ADT/ScopeExit.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

namespace {

llvm::StringRef nameOf(Operation *op) { return op->getName().getStringRef(); }

// Pure per-lane work only. The arm is re-emitted at the select, so a load could
// move past a store, and a barrier or cross-lane op would diverge in the
// branch.
bool isArmOp(Operation *op) {
  if (op->getNumRegions() || op->getNumResults() != 1)
    return false;
  const llvm::StringRef n = nameOf(op);
  if (n.starts_with("arith.") || n.starts_with("math."))
    return true;
  return llvm::is_contained({"tt.splat", "tt.broadcast", "tt.expand_dims",
                             "tt.addptr", "tt.extern_elementwise",
                             "tt.precise_divf", "tt.precise_sqrt", "tt.mulhiui",
                             "tt.fp_to_fp", "tt.bitcast"},
                            n);
}

bool isCostly(Operation *op) {
  const llvm::StringRef n = nameOf(op);
  if (n.starts_with("math."))
    return !llvm::is_contained({"math.absf", "math.absi", "math.fma",
                                "math.floor", "math.ceil", "math.trunc",
                                "math.round", "math.roundeven", "math.copysign",
                                "math.ctlz", "math.cttz", "math.ctpop"},
                               n);
  return llvm::is_contained({"tt.extern_elementwise", "tt.precise_divf",
                             "tt.precise_sqrt", "arith.divf"},
                            n);
}

// Ops of `sel`'s block whose every use ends in operand `arm` of `sel`, in
// block order.
std::vector<Operation *> armCone(Operation *sel, unsigned arm,
                                 const llvm::DenseSet<Operation *> &claimed) {
  llvm::SetVector<Operation *> cone;
  SmallVector<Value> work{sel->getOperand(arm)};
  while (!work.empty()) {
    Operation *def = work.pop_back_val().getDefiningOp();
    if (!def || def->getBlock() != sel->getBlock() || cone.contains(def) ||
        claimed.count(def) || !isArmOp(def))
      continue;
    cone.insert(def);
    for (Value v : def->getOperands())
      work.push_back(v);
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::make_early_inc_range(cone)) {
      const bool exclusive = llvm::all_of(op->getUses(), [&](OpOperand &use) {
        if (use.getOwner() == sel)
          return use.getOperandNumber() == arm;
        return cone.contains(use.getOwner());
      });
      if (!exclusive) {
        cone.remove(op);
        changed = true;
      }
    }
  }
  std::vector<Operation *> out(cone.begin(), cone.end());
  llvm::sort(out,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  if (llvm::none_of(out, isCostly))
    out.clear();
  return out;
}

// A simdgroup covers sizePerThread * threadsPerWarp elements of each dim, so a
// condition constant over aligned runs that long is the same in every lane, as
// a scalar is. Only then does a branch skip work; a lane-varying one runs both
// arms.
bool isSimdgroupUniform(ModuleAxisInfoAnalysis &axis, Value cond) {
  auto ty = dyn_cast<RankedTensorType>(cond.getType());
  if (!ty)
    return true;
  auto blk =
      dyn_cast_or_null<triton::gpu::BlockedEncodingAttr>(ty.getEncoding());
  AxisInfo *ai = axis.getAxisInfo(cond);
  if (!blk || !ai)
    return false;
  for (int d = 0; d < ty.getRank(); ++d) {
    const int64_t span = std::min<int64_t>((int64_t)blk.getSizePerThread()[d] *
                                               blk.getThreadsPerWarp()[d],
                                           ty.getShape()[d]);
    if (ai->getConstancy(d) < span)
      return false;
  }
  return true;
}

} // namespace

// Latest select first, so a select inside an earlier-planned arm stays part of
// that arm. A scalar select stays branch-free although its condition is
// uniform: branching its arms measured no faster and was miscompiled on one
// GPU generation.
void AgpuEmitter::planPredicatedArms(Block &block) {
  llvm::DenseSet<Operation *> claimed;
  for (Operation &op : llvm::reverse(block)) {
    if (nameOf(&op) != "arith.select" || claimed.count(&op))
      continue;
    const Value cond = op.getOperand(0);
    if (!isa<RankedTensorType>(cond.getType()) ||
        !isSimdgroupUniform(axisInfo(), cond))
      continue;
    PredicatedArms p;
    for (unsigned arm : {1u, 2u})
      p.cones[arm - 1] = armCone(&op, arm, claimed);
    if (p.cones[0].empty() && p.cones[1].empty())
      continue;
    for (const std::vector<Operation *> &cone : p.cones)
      for (Operation *o : cone) {
        claimed.insert(o);
        deferred_.insert(o);
      }
    predicated_[&op] = std::move(p);
  }
}

agpu::Decision AgpuEmitter::emitPredicatedSelect(Operation *sel) {
  const PredicatedArms &p = predicated_.find(sel)->second;
  am::Context &c = agpu_.context();

  std::vector<am::Str> conds;
  if (const agpu::ValueNames *n = body_.sym.namesOf(idOf(sel->getOperand(0))))
    for (const am::Str &s : *n)
      if (!llvm::is_contained(conds, s))
        conds.push_back(s);

  for (unsigned i = 0; i < 2; ++i) {
    const std::vector<Operation *> &cone = p.cones[i];
    if (cone.empty())
      continue;
    const Value root = sel->getOperand(i + 1);
    const std::optional<agpu::ElemType> elem = heldTypeFor(root);
    if (conds.empty() || !elem || elem->isPointer()) {
      for (Operation *op : cone)
        if (const agpu::Decision d = walkOp(op); !d.ok())
          return d;
      continue;
    }

    agpu::CarriedValue like;
    like.elem = *elem;
    like.regs.resize(registersHeldByType(root.getType()));
    am::Block arm;
    agpu::CarriedValue yielded;
    {
      const CurBlock here(*this, arm);
      const llvm::scope_exit epochs(
          [this, depth = body_.hoist.depth()] { body_.hoist.popTo(depth); });
      for (Operation *op : cone)
        if (const agpu::Decision d = walkOp(op); !d.ok())
          return d;
      const agpu::Result<agpu::CarriedValue> y = carriedFrom(
          root, like, "arith.select", "a predicated arm has no register names");
      if (!y.ok())
        return y.why;
      yielded = y.value;
    }

    // Assigned only when the arm runs, so defined for when it does not.
    const agpu::CarriedValue result = carriedFresh(root);
    for (const am::Str &r : result.regs)
      cur_->push_back(agpu::poisonDecl(c, r, result.elem));
    agpu::emitYield(c, arm, agpu::Carried{result}, agpu::Carried{yielded});
    am::Expr *pred = nullptr;
    for (const am::Str &s : conds) {
      am::Expr *e = c.var(s);
      if (i == 1)
        e = c.unary(am::UnOp::LNot, e);
      pred = pred ? c.binary(am::BinOp::LOr, pred, e) : e;
    }
    cur_->push_back(c.ifStmt(pred, std::move(arm)));
  }
  return walkOp(sel);
}

} // namespace mlir::triton::applegpu::bridge
