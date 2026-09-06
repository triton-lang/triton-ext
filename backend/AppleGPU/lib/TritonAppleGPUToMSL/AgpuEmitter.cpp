// AgpuEmitter - the walk core: dispatch registration, the op walk, what the
// walk knows about a value, and the module entry point.
#include "AgpuEmitter.h"

#include "agpu/core/Names.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/Vestigial.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

void AgpuEmitter::markBasePointer(agpu::ValueId v) { body_.basePtrs.insert(v); }

void AgpuEmitter::inheritBasePointer(agpu::ValueId from, agpu::ValueId to) {
  if (body_.basePtrs.count(from))
    body_.basePtrs.insert(to);
}
void AgpuEmitter::inheritOffset(agpu::ValueId from, int64_t fromReg,
                                agpu::ValueId to, int64_t toReg) {
  const auto off = body_.offsetOf.find({from, fromReg});
  if (off == body_.offsetOf.end())
    return;
  PtrOffset copy = off->second;
  copy.owned = false;
  body_.offsetOf[{to, toReg}] = std::move(copy);
}

am::Expr *AgpuEmitter::addressAt(agpu::ValueId ptr, int64_t reg) {
  am::Context &mc = agpu_.context();
  const am::Str *base = body_.sym.regAt(ptr, (std::size_t)reg);
  if (!base)
    return nullptr;
  const auto off = body_.offsetOf.find({ptr, reg});
  if (off != body_.offsetOf.end())
    return mc.subscript(mc.var(*base), mc.var(off->second.name));

  if (body_.basePtrs.count(ptr))
    return mc.subscript(mc.var(*base), mc.lit(0));
  return nullptr;
}

bool AgpuEmitter::addressesAreRedundant(Value ptr) {
  auto ptrTy =
      ptr ? dyn_cast<RankedTensorType>(ptr.getType()) : RankedTensorType();
  if (!ptrTy)
    return true;
  const LinearLayout ll = gpu::toLinearLayout(ptrTy);
  MLIRContext *ctx = ptrTy.getContext();
  return freeBitsOf(ll, ctx, lldim::Lane) != 0 ||
         freeBitsOf(ll, ctx, lldim::Warp) != 0;
}

am::Expr *AgpuEmitter::maskAt(const agpu::OpView &o, std::size_t maskIndex,
                              int64_t reg) {
  if (o.operands.size() <= maskIndex)
    return nullptr;
  const am::Str *m = body_.sym.regAt(o.operands[maskIndex], (std::size_t)reg);
  return m ? agpu_.context().var(*m) : nullptr;
}

agpu::CoordSource AgpuEmitter::coordSourceOf(RankedTensorType ty) {
  agpu::CoordSource cs;
  cs.hoist = &body_.hoist;
  const LinearLayout ll = gpu::toLinearLayout(ty);
  for (int d = 0; d < ty.getRank(); ++d)
    if (const std::optional<StringAttr> dim = outDimAt(ll, d))
      cs.dims.push_back(layoutSourceOf(ll, ty.getContext(), *dim).basis());
  return cs;
}

am::Expr *AgpuEmitter::coordOf(Value v, int reg, int axis) {
  auto rt = dyn_cast<RankedTensorType>(v.getType());
  if (!rt)
    return nullptr;
  const LinearLayout ll = gpu::toLinearLayout(rt);
  const std::optional<StringAttr> dim = outDimAt(ll, axis);
  if (!dim)
    return nullptr;
  const agpu::LayoutBasis lb =
      layoutSourceOf(ll, rt.getContext(), *dim).basis();
  return body_.hoist.coord(agpu_.context(), lb, reg);
}

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
  registerRangeHandler();
  registerAddPtrHandler();
  registerMemoryHandler();
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
  // here return early. An op with neither an arm here nor a table handler
  // declines by name.
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

  std::string who;
  return emitted(table_.runNamed(view, who), op, view.name);
}

agpu::Decision AgpuEmitter::declineOp(Operation *op, const agpu::Decision &d,
                                      std::string_view name) {
  agpu_.declines.record(d, agpu::DeclineSite{std::string(name)});

  op->emitError() << "AgpuEmitter: " << d.message();
  return d;
}

int64_t AgpuEmitter::registersHeldByType(Type t) const {
  auto rt = dyn_cast<RankedTensorType>(t);
  return rt ? registerCount(rt) : 1;
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

// MSL has no goto, so a multi-block body needs the dispatch-loop lowering that
// arrives with control flow.
agpu::Decision AgpuEmitter::walkWholeRegion(Region &region, am::Block &out,
                                            const TerminatorFn &atTerminator) {
  if (!region.hasOneBlock())
    return declined("region", "a multi-block body needs a dispatch loop");
  Block &only = region.front();
  if (const agpu::Decision d = walkBlock(only, out); !d.ok())
    return d;
  return atTerminator ? atTerminator(only, out) : agpu::Decision::emitted();
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

  am::Block out;
  for (am::Stmt *s : body_.hoist.decls)
    out.push_back(s);
  for (am::Stmt *s : body)
    out.push_back(s);

  return agpu::BuiltBody{std::move(out)};
}

LogicalResult AgpuEmitter::emit() {
  registerHandlers();

  bool any = false;
  for (auto func : mod_.getOps<triton::FuncOp>()) {
    agpu::KernelFacts facts;
    facts.name = agpu::kernelSymbol(func.getSymName());
    if (failed(bindArgs(func, facts.args)))
      return failure();
    facts.numWarps = numWarps();

    // Captured by value: the callback runs during print(), after this loop
    // returns.
    const agpu::SymbolTable afterArgs = body_.sym;
    const std::set<agpu::ValueId> argPtrs = body_.basePtrs;
    Block &entry = func.getBody().front();

    agpu_.addKernel(facts,
                    [this, afterArgs, argPtrs, &entry,
                     declineMark = std::optional<std::size_t>()](
                        am::Context &) mutable -> agpu::BuiltBody {
                      body_ = BodyState{afterArgs, argPtrs};
                      // Marking keeps earlier kernels' entries through this
                      // kernel's rebuild, and the second pass does not
                      // double-count.
                      if (!declineMark)
                        declineMark = agpu_.declines.size();
                      else
                        agpu_.declines.truncate(*declineMark);
                      return buildKernelBody(*entry.getParent());
                    });
    any = true;
  }

  if (!any)
    return failure();

  std::ostringstream out;
  const agpu::ModuleResult mr = agpu_.print(out);

  if (!bodyOk_)
    return failure();

  if (!mr.ok()) {
    mod_.emitError() << "AgpuEmitter: " << mr.decision.message();
    return failure();
  }

  os_ << out.str();
  return success();
}

} // namespace mlir::triton::applegpu::bridge
