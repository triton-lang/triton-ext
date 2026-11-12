// The fused-dot accumulator cycle: proving a loop-carried accumulator
// reaches a tt.store through a chain the drain can replay. Declarations and
// the small predicates are in AgpuDotChain.h.
#include "AgpuDotChain.h"

namespace mlir::triton::applegpu::bridge {

bool DrainProof::operandPeel(Operation *def) const {
  if (isa<gpu::ConvertLayoutOp>(def))
    return true;
  if (auto ext = dyn_cast<arith::ExtFOp>(def)) {
    auto rty = dyn_cast<RankedTensorType>(ext.getResult().getType());
    return rty && rty.getElementType().isF32();
  }
  return false;
}

Value DrainProof::lookThroughPeels(Value x) const {
  while (Operation *def = x.getDefiningOp()) {
    if (!operandPeel(def))
      return x;
    x = def->getOperand(0);
  }
  return x;
}

Value DrainProof::absorbOperandPeels(Value x) {
  while (Operation *def = x.getDefiningOp()) {
    if (!x.hasOneUse() || !operandPeel(def))
      return x;
    out_.chainOps.push_back(def);
    x = def->getOperand(0);
  }
  return x;
}

// Shape of the chain, down to the store. Operand classification needs the
// store's window and waits.
bool DrainProof::spine(Operation *firstUser) {
  Value v = root_;
  while (!store_) {
    v = pastConverts(v, out_.chainOps);
    Operation *user = nullptr;
    if (v == root_ && firstUser) {
      user = firstUser;
    } else if (v.hasOneUse()) {
      user = *v.getUsers().begin();
    } else if (v == root_ || fanOut_) {
      return fail(v == root_ ? "the loop result has more than one use"
                             : "a second folded value has more than one use");
    } else {
      int best = -1;
      for (Operation *u : v.getUsers()) {
        const int hops = hopsToStore(u);
        if (hops >= 0 && (best < 0 || hops < best)) {
          best = hops;
          user = u;
        }
      }
      if (!user)
        return fail("a folded value has more than one use");
      fanOut_ = v;
      fanOutStep_ = (int)folded_.size();
    }
    if ((store_ = dyn_cast<triton::StoreOp>(user)))
      break;

    if (isa<arith::TruncFOp>(user)) {
      auto rty = dyn_cast<RankedTensorType>(user->getResult(0).getType());
      // f16/bf16 are the elements simdgroup_matrix offers below f32, both
      // sharing the f32 fragment's lane mapping.
      if (!rty ||
          !(rty.getElementType().isF16() || rty.getElementType().isBF16()))
        return fail("the chain narrows to an element the drain cannot "
                    "store");
      pendingRound_ = true;
      out_.chainOps.push_back(user);
      v = user->getResult(0);
      continue;
    }

    // f16 embeds in f32, so this is a rename of an already-rounded value.
    if (isa<arith::ExtFOp>(user)) {
      auto rty = dyn_cast<RankedTensorType>(user->getResult(0).getType());
      if (!rty || !rty.getElementType().isF32())
        return fail("the chain widens to an element the fragments do not "
                    "hold");
      out_.chainOps.push_back(user);
      v = user->getResult(0);
      continue;
    }

    const auto name = user->getName().getStringRef();
    if (!agpu::isEpilogueOp({name.data(), name.size()}) ||
        user->getNumResults() != 1)
      // The op name distinguishes a shape that can never fuse (e.g.
      // tt.reshape) from a gap in the proof.
      return fail("the loop result's use is " + name.str());

    Value other;
    if (agpu::isEpilogueBinary({name.data(), name.size()})) {
      const Value lhs = user->getOperand(0), rhs = user->getOperand(1);
      other = lhs == v ? rhs : lhs;
      if (other == v)
        return fail("the epilogue op combines the accumulator with itself");
      // The drain renders `acc OP operand`; nothing implements the
      // reversed rendering a non-commutative op would need.
      if (lhs != v) {
        agpu::EpilogueBinOp bin{};
        agpu::epilogueBinOpOf({name.data(), name.size()}, bin);
        if (bin == agpu::EpilogueBinOp::Sub || bin == agpu::EpilogueBinOp::Div)
          return fail("the accumulator is the right operand of a "
                      "non-commutative epilogue op");
      }
    }
    folded_.push_back({user, other, pendingRound_});
    // Every element the drain can spell now computes at f32, so no step
    // rounds its own result.
    auto sty = dyn_cast<RankedTensorType>(user->getResult(0).getType());
    if (!sty ||
        !(sty.getElementType().isF32() || sty.getElementType().isF16() ||
          sty.getElementType().isBF16()))
      return fail("an epilogue op computes in an element the drain cannot "
                  "spell");
    pendingRound_ = false;
    out_.chainOps.push_back(user);
    v = user->getResult(0);
  }
  if (store_.getValue() != v)
    return fail("the store's data is not the loop result");
  return true;
}

bool DrainProof::storeWindow() {
  out_.window = deviceWindowOf(store_.getPtr());
  if (!out_.window.base)
    return fail("the store's pointers are not a row-major window");
  auto tileTy = dyn_cast<RankedTensorType>(store_.getValue().getType());
  if (!tileTy || tileTy.getRank() != 2)
    return fail("the store's data is not a 2-D tile");
  // f32 or f16 only. An f32 chain still carrying a pending round has
  // nowhere to replay it.
  if (tileTy.getElementType().isF32()) {
    if (pendingRound_)
      return fail("the chain narrows and widens back with nothing between");
  } else if (!tileTy.getElementType().isF16() &&
             !tileTy.getElementType().isBF16()) {
    return fail("the store's element is not one the drain can spell");
  }
  rows_ = tileTy.getShape()[0];
  cols_ = tileTy.getShape()[1];

  // The mask, when there is one, must be the window's own bounds.
  if (store_.getMask()) {
    const WindowBounds b =
        windowBoundsOf(store_.getMask(), out_.window, rows_, cols_);
    if (!b.ok)
      return fail("the store's mask is not the window's own bounds");
    out_.rowBound = b.row;
    out_.colBound = b.col;
    out_.uniformGuard = b.uniform;
    out_.clamps = b.clamps;
  }
  return true;
}

// Classify one folded operand, for both the spine's steps and an
// accumulator-rooted chain's links. Returns the failure reason, empty on
// success.
template <class Target>
std::string DrainProof::classifyOperand(Value operand, Target &target) {
  using K = std::decay_t<decltype(target.kind)>;
  const Value behind = lookThroughPeels(operand);
  if (Value s = splatScalarOf(behind)) {
    target.kind = K::Splat;
    target.splat = s;
    absorbOperandPeels(operand);
  } else if (splatConstantOf(behind, target.splatConst)) {
    target.kind = K::Splat;
    absorbOperandPeels(operand);
  } else if (DrainAddend a = drainAddendOf(behind, out_.window); a.ok()) {
    // A masked operand's mask must be bounds the store also enforces.
    // Weaker is fine; stricter would read past the operand's tensor.
    if (a.load.getMask()) {
      const WindowBounds lb =
          windowBoundsOf(a.load.getMask(), out_.window, rows_, cols_);
      if (!lb.ok)
        return "a folded operand's mask is not the window's own bounds";
      if ((lb.row.present && !(lb.row == out_.rowBound)) ||
          (lb.col.present && !(lb.col == out_.colBound)))
        return "a folded operand's mask is stricter than the store's";
    }
    target.kind = K::Window;
    target.addend = a;
    // If the fold is its only reader, the load joins chainOps: the drain
    // fetches the same elements from device memory itself.
    if (const Value x = absorbOperandPeels(operand);
        x == a.load.getResult() && x.hasOneUse())
      out_.chainOps.push_back(a.load);
  } else {
    Operation *from = behind.getDefiningOp();
    return "a folded op's operand is neither a splat nor a device "
           "window at the store's coordinates (it comes from " +
           (from ? from->getName().getStringRef().str()
                 : std::string("a block argument")) +
           ")";
  }
  return {};
}

// An operand computed from the accumulator, walked backward to `root_`.
// Every link is a single-use epilogue-table op on f32 tensors, converts
// absorbed. `links` come out root-first. False sets `creason`.
bool DrainProof::accChain(Value operand,
                          agpu::msl::SmallVec<BranchLinkRaw, 4> &links,
                          int &base, std::string &creason) {
  const auto cfail = [&](std::string reason) {
    creason = std::move(reason);
    return false;
  };
  std::vector<Operation *> branchChainOps;
  Value x = operand;
  int length = 0;
  base = 0;
  while (x != root_ && !(fanOut_ && x == fanOut_)) {
    if (!x.hasOneUse())
      return cfail("a folded value has more than one use");
    Operation *def = x.getDefiningOp();
    if (!def)
      return cfail("a folded operand's chain leaves the block");
    if (isa<gpu::ConvertLayoutOp>(def)) {
      branchChainOps.push_back(def);
      x = def->getOperand(0);
      continue;
    }
    // Kept inside `drainChainRootOf`'s depth so the pool pre-pass can
    // re-find the root from each folded op.
    if (++length > 6)
      return cfail("a folded operand's chain is too long");
    const auto name = def->getName().getStringRef();
    if (!agpu::isEpilogueOp({name.data(), name.size()}) ||
        def->getNumResults() != 1)
      return cfail("a folded operand's chain holds " + name.str());
    auto rty = dyn_cast<RankedTensorType>(def->getResult(0).getType());
    if (!rty || !rty.getElementType().isF32())
      return cfail("a folded operand's chain computes off f32");
    Value next = def->getOperand(0), other;
    if (agpu::isEpilogueBinary({name.data(), name.size()})) {
      const Value lhs = def->getOperand(0), rhs = def->getOperand(1);
      const bool lRooted = rootedAt(lhs), rRooted = rootedAt(rhs);
      if (lRooted && rRooted)
        return cfail("the epilogue op combines the accumulator with "
                     "itself");
      if (!lRooted && !rRooted)
        return cfail("a folded operand's chain loses the accumulator");
      next = lRooted ? lhs : rhs;
      other = lRooted ? rhs : lhs;
      if (!lRooted) {
        agpu::EpilogueBinOp bin{};
        agpu::epilogueBinOpOf({name.data(), name.size()}, bin);
        if (bin == agpu::EpilogueBinOp::Sub || bin == agpu::EpilogueBinOp::Div)
          return cfail("the accumulator is the right operand of a "
                       "non-commutative epilogue op");
      }
    }
    links.push_back({def, other});
    branchChainOps.push_back(def);
    x = next;
  }
  base = x == root_ ? 0 : fanOutStep_;
  std::reverse(links.begin(), links.end());
  out_.chainOps.insert(out_.chainOps.end(), branchChainOps.begin(),
                       branchChainOps.end());
  return true;
}

bool DrainProof::foldSteps() {
  for (const auto &[op, operand, roundBefore] : folded_) {
    DrainStepFact step;
    step.op = op;
    step.roundBefore = roundBefore;
    if (!operand) {
      out_.steps.push_back(step);
      continue;
    }
    if (const std::string creason = classifyOperand(operand, step);
        !creason.empty()) {
      // Not a splat and not a window. If the accumulator feeds the
      // operand, it folds as a chain re-rendered from the element (gelu's
      // `(acc*0.5)*(1+erf(acc*0.7071))`).
      if (!rootedAt(operand))
        return fail(creason);
      agpu::msl::SmallVec<BranchLinkRaw, 4> raw;
      std::string breason;
      if (!accChain(operand, raw, step.branchBase, breason))
        return fail(std::move(breason));
      step.kind = DrainStepFact::Operand::AccChain;
      for (const BranchLinkRaw &lk : raw) {
        DrainBranchLinkFact link;
        link.op = lk.op;
        if (lk.operand) {
          if (const std::string lreason = classifyOperand(lk.operand, link);
              !lreason.empty())
            return fail(lreason);
        }
        step.branch.push_back(link);
      }
    }
    out_.steps.push_back(step);
  }
  return true;
}

// Every consumer of a fanned-out root must be part of the drain, else the
// fragments are not the value's only home.
bool DrainProof::checkFanOut() {
  if (root_.hasOneUse() && !fanOut_)
    return true;
  llvm::SmallPtrSet<Operation *, 8> absorbed;
  for (const auto &[op, operand, roundBefore] : folded_)
    absorbed.insert(op);
  for (const DrainStepFact &s : out_.steps)
    for (const DrainBranchLinkFact &l : s.branch)
      absorbed.insert(l.op);
  for (Operation *op : out_.chainOps)
    absorbed.insert(op);
  absorbed.insert(store_.getOperation());
  for (Operation *u : root_.getUsers())
    if (!absorbed.count(u))
      return fail("the loop result has a consumer outside the drain");
  if (fanOut_)
    for (Operation *u : fanOut_.getUsers())
      if (!absorbed.count(u))
        return fail("a folded value has a consumer outside the drain");
  return true;
}

int DrainProof::hopsToStore(Operation *user) {
  int hops = 0;
  while (!isa<triton::StoreOp>(user)) {
    const auto name = user->getName().getStringRef();
    if (user->getNumResults() != 1 || !user->getResult(0).hasOneUse() ||
        !(isa<gpu::ConvertLayoutOp, arith::TruncFOp, arith::ExtFOp>(user) ||
          agpu::isEpilogueOp({name.data(), name.size()})))
      return -1;
    user = *user->getResult(0).getUsers().begin();
    ++hops;
  }
  return hops;
}

FusedDrain DrainProof::run(Operation *firstUser) {
  if (!spine(firstUser) || !storeWindow() || !foldSteps() || !checkFanOut())
    return FusedDrain{};
  out_.store = store_;
  return out_;
}

FusedDrain fusedDrainOf(Value carried, std::string *why) {
  if (!carried) {
    if (why)
      *why = "the accumulator does not outlive the loop";
    return FusedDrain{};
  }

  // Peel single-use converts first, leaving the value the chain hangs off,
  // which is also the one value allowed more than one use.
  std::vector<Operation *> headOps;
  const Value root = pastConverts(carried, headOps);

  // A fan-out root tries each consumer as the spine and takes the first that
  // reaches a store with every other consumer folded back in.
  std::string firstWhy;
  if (root.hasOneUse()) {
    DrainProof proof(carried, root, headOps);
    FusedDrain d = proof.run(nullptr);
    if (!d && why)
      *why = proof.why();
    return d;
  }
  llvm::SmallPtrSet<Operation *, 4> tried;
  for (Operation *u : root.getUsers()) {
    if (!tried.insert(u).second)
      continue;
    DrainProof proof(carried, root, headOps);
    if (FusedDrain d = proof.run(u))
      return d;
    if (firstWhy.empty())
      firstWhy = proof.why();
  }
  if (why)
    *why =
        firstWhy.empty() ? "the loop result has more than one use" : firstWhy;
  return FusedDrain{};
}

// The loop result a drain-chain value hangs off, or null: converts and
// epilogue-table ops peeled upward until an scf.for result appears. At a
// binary op either operand may be the chain, so both are tried.
Value drainChainRootOf(Value v, int depth) {
  if (depth <= 0 || !v)
    return {};
  if (auto res = dyn_cast<OpResult>(v))
    if (isa<scf::ForOp>(res.getOwner()))
      return v;
  Operation *d = v.getDefiningOp();
  if (!d)
    return {};
  if (isa<gpu::ConvertLayoutOp, arith::TruncFOp, arith::ExtFOp>(d))
    return drainChainRootOf(d->getOperand(0), depth - 1);
  const auto name = d->getName().getStringRef();
  if (agpu::isEpilogueOp({name.data(), name.size()}))
    for (Value o : d->getOperands())
      if (Value r = drainChainRootOf(o, depth - 1))
        return r;
  return {};
}

// The dot whose fused accumulator `loopResult` is, or null.
triton::DotOp fusedDotBehind(Value loopResult) {
  auto res = dyn_cast_or_null<OpResult>(loopResult);
  auto forOp = res ? dyn_cast<scf::ForOp>(res.getOwner()) : nullptr;
  if (!forOp)
    return {};
  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield)
    return {};
  auto dot =
      yield.getOperand(res.getResultNumber()).getDefiningOp<triton::DotOp>();
  if (!dot || fusedAccumulator(dot.getC(), dot.getResult()) != loopResult)
    return {};
  return dot;
}

// Fill the C-side device window, when the fused chain ends in a store whose
// window the proof recognises. One function so the pool pre-pass and the
// handler answer identically.
void fillFusedCStore(DotShape &d) {
  const FusedDrain drain = fusedDrainOf(d.cCarried, &d.cStoreWhy);
  if (!drain)
    return;
  d.cDevice = drain.window;
  d.cStore = drain.store;
  d.cRowBound = drain.rowBound;
  d.cColBound = drain.colBound;
  d.cClamps = drain.clamps;
  d.cUniformGuard = drain.uniformGuard;
  d.cSteps = drain.steps;
  d.cChainOps = drain.chainOps;
}

// How this dot's accumulator is carried: whether it arrives as a loop
// iter-arg, whether the loop carries it in fragments and where its fused
// chain stores. The pool pre-pass and the handler reach C and result
// differently, so only the derivation is shared here.
void fillAccumulatorCarry(DotShape &d, Value c, Value result) {
  d.cInput = c;
  d.cResult = result;
  d.cIterArg = carriedAccumulator(c);
  d.cCarried = fusedAccumulator(c, result);
  fillFusedCStore(d);
}

} // namespace mlir::triton::applegpu::bridge
