#include "AgpuEmitter.h"
#include "AgpuLog.h"

#include "agpu/core/Names.h"
#include "agpu/emit/EmitPrune.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/Vestigial.h"

#include <sstream>

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

namespace {

std::string_view opName(Operation *op) {
  const llvm::StringRef n = op->getName().getStringRef();
  return std::string_view(n.data(), n.size());
}

ConstantValue valueOfAttr(Attribute a) {
  if (auto ia = dyn_cast<IntegerAttr>(a))
    return ConstantValue{ia.getInt(), 0.0, false, true};
  if (auto fa = dyn_cast<FloatAttr>(a))
    return ConstantValue{0, fa.getValueAsDouble(), true, true};
  return ConstantValue{};
}

std::vector<ConstantValue> constantsOf(arith::ConstantOp k) {
  const Attribute a = k.getValue();

  auto d = dyn_cast<DenseElementsAttr>(a);
  if (!d) {
    const ConstantValue v = valueOfAttr(a);
    return v.known ? std::vector<ConstantValue>{v}
                   : std::vector<ConstantValue>{};
  }
  if (d.isSplat()) {
    const ConstantValue v = valueOfAttr(d.getSplatValue<Attribute>());
    return v.known ? std::vector<ConstantValue>{v}
                   : std::vector<ConstantValue>{};
  }

  std::vector<ConstantValue> out;
  for (Attribute e : d.getValues<Attribute>()) {
    const ConstantValue v = valueOfAttr(e);
    if (!v.known)
      return {};
    out.push_back(v);
  }
  return out;
}
} // namespace

void AgpuEmitter::registerHandlers() {
  registerArithHandlers();
  registerValueHandlers();
  registerRebindHandler();
  registerCallHandler();
  registerTileHandlers();
  registerInterleaveHandler();
  registerDotHandler();
  registerRangeHandler();
  registerAddPtrHandler();
  registerAtomicHandlers();
  registerMemoryHandler();
  registerMemDescHandler();
  registerBarrierHandler();
}

int64_t AgpuEmitter::numWarps() const {
  if (auto nw = mod_->getAttrOfType<IntegerAttr>(gpu::AttrNumWarpsName))
    return nw.getInt();
  return 1;
}

agpu::ValueId AgpuEmitter::idOf(Value v) {
  const auto it = ids_.find(v);
  if (it != ids_.end())
    return it->second;
  const agpu::ValueId id = nextId_++;
  ids_[v] = id;
  return id;
}

bool AgpuEmitter::willHaveScalarName(Value v, Operation *useSite) const {
  const auto it = ids_.find(v);
  if (it != ids_.end() && body_.sym.scalarName(it->second) != nullptr)
    return true;

  if (isa<RankedTensorType>(v.getType()))
    return false;

  if (auto arg = dyn_cast<BlockArgument>(v))
    return isa<triton::FuncOp>(arg.getOwner()->getParentOp());

  Operation *def = v.getDefiningOp();
  if (!def)
    return false;

  for (Operation *scope = useSite; scope; scope = scope->getParentOp())
    if (def->getParentRegion() == scope->getParentRegion())
      return def->getBlock() != scope->getBlock() ||
             def->isBeforeInBlock(scope);
  return false;
}

agpu::ValueNames AgpuEmitter::freshNames(Value v, int64_t count) {
  agpu::ValueNames out;
  const agpu::ValueId id = idOf(v);
  for (int64_t r = 0; r < count; ++r)
    out.push_back(nameFor('v', id, r));
  return out;
}

LogicalResult AgpuEmitter::bindArgs(triton::FuncOp func,
                                    std::vector<agpu::KernelArg> &args) {
  for (auto [i, argTy] : llvm::enumerate(func.getFunctionType().getInputs())) {
    const BlockArgument arg = func.getArgument(i);

    agpu::KernelArg ka;
    ka.name = "arg" + std::to_string(i);
    ka.isPointer = isa<triton::PointerType>(argTy);

    const std::optional<agpu::ElemType> elem = elemTypeOf(argTy);
    if (!elem) {
      func.emitError("AgpuEmitter: unsupported kernel argument type");
      return failure();
    }
    ka.elem = *elem;
    args.push_back(ka);

    body_.sym.bindScalar(idOf(arg), ka.name);
    if (ka.isPointer)
      markBasePointer(idOf(arg));
    elemFor_[idOf(arg)] = *elem;
    agpu_.noteIfNarrowed(*elem);
    valueFor_[idOf(arg)] = arg;
  }
  return success();
}

agpu::Decision AgpuEmitter::walkOp(Operation *op) {
  if (body_.absorbedOps.count(op))
    return agpu::Decision::emitted();

  if (agpu::isVestigial(opName(op)))
    return agpu::Decision::emitted();

  for (Value v : op->getResults()) {
    const agpu::ValueId id = idOf(v);
    valueFor_[id] = v;
    if (const std::optional<agpu::ElemType> e = elemTypeOf(v.getType())) {
      elemFor_[id] = *e;
      agpu_.noteIfNarrowed(*e);
    }
  }

  // An op lowers here, outside the dispatch table, when it needs a region to
  // walk or an attribute OpView cannot carry. OpView holds no
  // Operation * on purpose, so that agpu/bind/ builds and tests without MLIR,
  // which is what makes these inexpressible as table handlers. The dyn_casts
  // further down read attributes into `ints` and still dispatch; only the ones
  // here return early.
  if (auto typed = dyn_cast<triton::AssertOp>(op))
    return emitted(emitAssertOp(typed), op, "tt.assert");
  if (auto typed = dyn_cast<triton::PrintOp>(op))
    return emitted(emitPrintOp(typed), op, "tt.print");
  if (auto typed = dyn_cast<scf::ForOp>(op))
    return emitted(emitForOp(typed), op, "scf.for");
  if (auto typed = dyn_cast<scf::IfOp>(op))
    return emitted(emitIfOp(typed), op, "scf.if");
  if (auto typed = dyn_cast<scf::WhileOp>(op))
    return emitted(emitWhileOp(typed), op, "scf.while");

  if (isa<triton::ReduceOp, triton::ScanOp>(op)) {
    body_.pool.carve(poolNeedOf(op));
    auto red = dyn_cast<triton::ReduceOp>(op);
    const agpu::Decision d =
        red ? emitReduceOp(red) : emitScanOp(cast<triton::ScanOp>(op));
    return emitted(d, op, opName(op));
  }

  if (auto map = dyn_cast<triton::MapElementwiseOp>(op))
    return emitted(emitMapOp(map), op, "tt.map_elementwise");

  agpu::OpView view;
  view.name = opName(op);
  for (Value v : op->getOperands())
    view.operands.push_back(idOf(v));
  if (auto mr = dyn_cast<triton::MakeRangeOp>(op))
    view.ints.push_back(mr.getStart());
  if (auto cmp = dyn_cast<arith::CmpIOp>(op))
    view.ints.push_back((int64_t)cmp.getPredicate());
  if (auto cmp = dyn_cast<arith::CmpFOp>(op))
    view.ints.push_back((int64_t)cmp.getPredicate());
  if (auto fp = dyn_cast<triton::FpToFpOp>(op))
    if (const std::optional<triton::RoundingMode> rm = fp.getRounding())
      view.ints.push_back((int64_t)*rm);
  if (auto fp4 = dyn_cast<triton::gpu::Fp4ToFpOp>(op))
    view.ints.push_back((int64_t)fp4.getAxis());
  if (auto ga = dyn_cast<triton::GatherOp>(op))
    view.ints.push_back((int64_t)ga.getAxis());
  if (auto cas = dyn_cast<triton::AtomicCASOp>(op))
    view.ints.push_back((int64_t)cas.getSem());
  if (auto rmw = dyn_cast<triton::AtomicRMWOp>(op)) {
    view.ints.push_back((int64_t)rmw.getAtomicRmwOp());
    view.ints.push_back((int64_t)rmw.getSem());
  }
  if (auto pid = dyn_cast<triton::GetProgramIdOp>(op))
    view.ints.push_back((int64_t)pid.getAxisAsInt());
  if (auto np = dyn_cast<triton::GetNumProgramsOp>(op))
    view.ints.push_back((int64_t)np.getAxisAsInt());
  if (auto bar = dyn_cast<gpu::BarrierOp>(op))
    view.ints.push_back((int64_t)(uint32_t)bar.getAddrSpace());
  // metal::clamp is min(max(...)) and drops NaN.
  if (auto cl = dyn_cast<triton::ClampFOp>(op))
    view.ints.push_back(cl.getPropagateNan() == triton::PropagateNan::ALL);
  if (auto call = dyn_cast<triton::CallOp>(op))
    view.text = call.getCallee();
  std::vector<ConstantValue> konst;
  if (auto k = dyn_cast<arith::ConstantOp>(op))
    konst = constantsOf(k);

  for (Value v : op->getResults()) {
    const agpu::ValueId id = idOf(v);
    view.results.push_back(id);
    if (!konst.empty())
      constantFor_[id] = konst;
  }

  body_.pool.carve(poolNeedOf(op));

  if (agpu_.gates.on(agpu::Gate::TraceOps)) {
    std::ostringstream os;
    os << view.name;
    for (Value v : op->getOperands())
      if (auto t = dyn_cast<RankedTensorType>(v.getType()))
        os << "  in=" << t.getShape()[0] << "x"
           << (t.getRank() > 1 ? t.getShape()[1] : 1) << "/"
           << registerCount(t);
    for (Value v : op->getResults())
      if (auto t = dyn_cast<RankedTensorType>(v.getType()))
        os << "  out=" << t.getShape()[0] << "x"
           << (t.getRank() > 1 ? t.getShape()[1] : 1) << "/"
           << registerCount(t);
    os << "\n";
    appendLog(agpu::Gate::TraceOps, os.str());
  }

  std::string who;
  return emitted(table_.runNamed(view, who), op, view.name);
}

agpu::Decision AgpuEmitter::declineOp(Operation *op, const agpu::Decision &d,
                                      std::string_view name) {
  agpu_.declines.record(d, agpu::DeclineSite{std::string(name), ""});

  // compiler.py scans stderr for an out-of-budget message. Full detail goes
  // to AGPU_DECLINE_LOG.
  op->emitError() << "AgpuEmitter: " << d.message();
  appendLog(agpu::Gate::DeclineLog, std::string(name) + "\t" +
                                        (d.isBug() ? "FAILED (handler bug)"
                                         : d.keepLooking() ? "NOT MINE"
                                                           : d.message()) +
                                        "\n");
  return d;
}

agpu::Decision AgpuEmitter::walkBlock(Block &block, am::Block &out) {
  const CurBlock here(*this, out);
  for (Operation &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (const agpu::Decision d = walkOp(&op); !d.ok())
      return d;
  }
  return agpu::Decision::emitted();
}

// Fires only when the panel dots alone clear the shrink thresholds, a lower
// bound of what the measured body would show, so it never rolls a kernel the
// measured path would keep unrolled.
am::RollPrediction AgpuEmitter::predictRollFor(triton::FuncOp func) {
  agpu::PanelMmaSize u, ro;
  func.walk([&](triton::DotOp dot) {
    const DotShape shape = dotShapeOf(dot);
    if (!shape.aTy)
      return;
    const agpu::Plan plan = agpu_.planFor(dotFactsOf(shape));
    if (plan.kind != agpu::Plan::Kind::Panel)
      return;
    const agpu::PanelMmaSize du =
        agpu::predictPanelDotSize(plan.facts, plan.panel().panel, false);
    const agpu::PanelMmaSize dr =
        agpu::predictPanelDotSize(plan.facts, plan.panel().panel, true);
    u.decls += du.decls;
    u.fragDecls += du.fragDecls;
    u.mma += du.mma;
    ro.decls += dr.decls;
    ro.fragDecls += dr.fragDecls;
    ro.mma += dr.mma;
  });

  am::RollPrediction out;
  out.declDelta = u.decls - ro.decls;
  out.fragDelta = u.fragDecls - ro.fragDecls;
  out.mmaDelta = u.mma - ro.mma;
  out.roll = u.load() > am::kDeclBudget && u.fragDecls >= am::kRollFragFloor &&
             u.load() > ro.load();
  return out;
}

// Pool views, then live buffers, then coordinates, then body: each references
// only names declared before it.
agpu::BuiltBody AgpuEmitter::buildKernelBody(Region &region) {
  // MSL has no goto; a multi-block body lowers to a dispatch loop instead.
  am::Block body;
  if (!walkWholeRegion(region, body).ok()) {
    bodyOk_ = false;
    return agpu::BuiltBody{std::move(body)};
  }

  am::Block out = poolDecls();
  for (am::Stmt *s : body_.liveDecls)
    out.push_back(s);
  for (am::Stmt *s : body_.hoist.decls)
    out.push_back(s);
  for (am::Stmt *s : body)
    out.push_back(s);

  // A convert_layout absorbed by a later dot leaves its scatter-barrier-gather
  // emitted but unread. Metal drops the dead registers but not the barriers.
  agpu::pruneDead(out);
  return agpu::BuiltBody{std::move(out), body_.pool.usedBytes()};
}

LogicalResult AgpuEmitter::emit() {
  appendLog(agpu::Gate::DeclineLog, "(enter)\temit\n");
  registerHandlers();

  llvm::DenseSet<StringRef> callTargets;
  mod_.walk([&](triton::CallOp call) {
    callTargets.insert(call.getCalleeAttr().getValue());
  });

  clampOf_.clear();
  clampPoison_.clear();
  cDirectOf_.clear();
  for (auto func : mod_.getOps<triton::FuncOp>())
    scanPool(func);

  // Callees before callers: MSL needs the prototype at the call site.
  if (failed(addDeviceFnsInCallOrder(callTargets)))
    return failure();

  bool any = false;
  for (auto func : mod_.getOps<triton::FuncOp>()) {
    if (callTargets.contains(func.getSymName()))
      continue;

    agpu::KernelFacts facts;
    facts.name = agpu::kernelSymbol(func.getSymName());
    if (failed(bindArgs(func, facts.args))) {
      appendLog(agpu::Gate::DeclineLog,
                "(args)\t" + func.getSymName().str() +
                    ": an argument type has no representation\n");
      return failure();
    }
    facts.numWarps = numWarps();
    facts.debug.print = printBindingOf(func);
    // A callee's assert reaches the buffer through this kernel's parameter, so
    // the binding follows the module and not this body alone.
    facts.debug.assertion = agpu_.asserts.asserts() ? agpu::DebugBinding::Bound
                                                    : assertBindingOf(func);

    const agpu::CoherencePlan coherence =
        agpu::planCoherence(coherenceFactsOf(func));
    coherentArgs_.clear();
    for (std::size_t i = 0; i < facts.args.size(); ++i) {
      facts.args[i].coherent = coherence.needsCoherent((int)i);
      if (facts.args[i].coherent)
        coherentArgs_.insert((int)i);
    }

    facts.predictedRoll = predictRollFor(func);

    // Captured by value: the callback runs during print(), after this loop
    // returns.
    const agpu::SymbolTable afterArgs = body_.sym;
    const std::set<agpu::ValueId> argPtrs = body_.basePtrs;
    Block &entry = func.getBody().front();

    agpu_.addKernel(facts,
                    [this, afterArgs, argPtrs, &entry, coherent = coherentArgs_,
                     declineMark = std::optional<std::size_t>(),
                     printMark = std::optional<std::size_t>(),
                     assertMark = std::optional<std::size_t>()](
                        am::Context &, bool rollK) mutable -> agpu::BuiltBody {
                      coherentArgs_ = coherent;
                      rollK_ = rollK;
                      body_ = BodyState{afterArgs, argPtrs};
                      // Marking keeps earlier kernels' entries through this
                      // kernel's rebuild, and the second pass does not
                      // double-count.
                      if (!declineMark)
                        declineMark = agpu_.declines.size();
                      else
                        agpu_.declines.truncate(*declineMark);
                      if (!printMark)
                        printMark = agpu_.prints.siteCount();
                      else
                        agpu_.prints.truncate(*printMark);
                      if (!assertMark)
                        assertMark = agpu_.asserts.siteCount();
                      else
                        agpu_.asserts.truncate(*assertMark);
                      return buildKernelBody(*entry.getParent());
                    });
    any = true;
  }

  if (!any) {
    appendLog(agpu::Gate::DeclineLog,
              "(module)\tno kernel: every function is a call target\n");
    return failure();
  }

  std::ostringstream out;
  const agpu::ModuleResult mr = agpu_.print(out);

  if (agpu_.gates.on(agpu::Gate::FuncBudgetDebug)) {
    for (const agpu::KernelResult &kr : mr.kernels) {
      if (!kr.fn)
        continue;
      llvm::errs() << "[budget] "
                   << am::budgetReport(std::string_view(kr.fn->name), kr.size,
                                       kr.shrink, kr.reemitted)
                   << "\n";
    }
  }

  if (agpu_.gates.on(agpu::Gate::DeclineLog)) {
    std::ostringstream os;
    agpu_.declines.printSummary(os);
    if (!mr.ok())
      os << "(module)\t" << mr.decision.message() << "\n";
    if (!bodyOk_ && agpu_.declines.empty())
      os << "(module)\ta body failed with no decline recorded\n";
    appendLog(agpu::Gate::DeclineLog, os.str());
  }

  if (!bodyOk_)
    return failure();

  if (!mr.ok()) {
    // The host and autotuner read these attributes without parsing the
    // diagnostic.
    const auto i64 = mlir::IntegerType::get(mod_.getContext(), 64);
    mod_->setAttr(agpu::kPoolNeededAttr,
                  mlir::IntegerAttr::get(i64, mr.pool.total().count()));
    mod_->setAttr(agpu::kPoolLimitAttr,
                  mlir::IntegerAttr::get(i64, agpu::kTGResidentBudgetBytes));
    mod_.emitError() << "AgpuEmitter: " << mr.decision.message();
    return failure();
  }

  os_ << out.str();
  return success();
}

} // namespace mlir::triton::applegpu::bridge
