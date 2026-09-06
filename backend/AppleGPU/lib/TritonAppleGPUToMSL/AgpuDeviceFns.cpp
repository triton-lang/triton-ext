// The unstructured-CFG walk and device functions, plus tt.call.
#include "AgpuEmitter.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

agpu::RegionFacts AgpuEmitter::regionFactsOf(Region &region) {
  agpu::RegionFacts f;
  llvm::DenseMap<Block *, agpu::BlockId> index;
  for (Block &b : region)
    index[&b] = (agpu::BlockId)index.size();
  f.blocks.resize(index.size());
  f.entry = index.lookup(&region.front());

  auto edgeTo = [&](Block *dest, Operation::operand_range args) {
    agpu::Edge e;
    e.to = index.lookup(dest);
    for (Value a : args)
      e.args.push_back(idOf(a));
    return e;
  };

  for (Block &b : region) {
    agpu::BlockFacts &bf = f.blocks[(std::size_t)index.lookup(&b)];
    // The entry block's arguments are the function's parameters.
    if (&b != &region.front())
      for (BlockArgument a : b.getArguments())
        bf.params.push_back(idOf(a));

    // Nested ops included: a value an inner scf.for's body reads is hoisted
    // the same as a top-level read.
    for (Operation &op : b)
      op.walk([&](Operation *inner) {
        for (Value r : inner->getResults())
          bf.defines.push_back(idOf(r));
        for (Value o : inner->getOperands())
          bf.reads.push_back(idOf(o));
      });

    Operation *term = b.getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      bf.term = agpu::TermKind::Branch;
      bf.edges.push_back(edgeTo(br.getDest(), br.getDestOperands()));
    } else if (auto cbr = dyn_cast<cf::CondBranchOp>(term)) {
      bf.term = agpu::TermKind::CondBranch;
      bf.edges.push_back(edgeTo(cbr.getTrueDest(), cbr.getTrueDestOperands()));
      bf.edges.push_back(
          edgeTo(cbr.getFalseDest(), cbr.getFalseDestOperands()));
    } else {
      bf.term = agpu::TermKind::Return;
    }
  }
  return f;
}

LogicalResult AgpuEmitter::appendDeviceReturn(triton::FuncOp func,
                                              const agpu::DeviceFnAbi &abi,
                                              am::Block &body) {
  if (abi.ret == agpu::RetShape::Void)
    return success();

  auto ret = dyn_cast<triton::ReturnOp>(func.getBody().back().getTerminator());
  if (!ret) {
    func.emitError("AgpuEmitter: a device function must end in tt.return");
    return failure();
  }

  std::vector<am::Str> names;
  for (Value v : ret.getOperands()) {
    const agpu::ValueId id = idOf(v);
    const agpu::ValueNames *regs = body_.sym.namesOf(id);
    if (!regs) {
      func.emitError("AgpuEmitter: a returned value was never bound");
      return failure();
    }
    for (const am::Str &n : *regs)
      names.push_back(n);
  }
  if (names.size() != abi.retFields.size()) {
    func.emitError("AgpuEmitter: the returned registers do not match the "
                   "signature");
    return failure();
  }

  body.push_back(agpu::emitDeviceReturn(agpu_.context(), abi, names));
  return success();
}

// Callees-first depth-first walk; `emitting` catches a call cycle.
LogicalResult AgpuEmitter::addDeviceFnsInCallOrder(
    const llvm::DenseSet<StringRef> &callTargets) {
  llvm::DenseSet<StringRef> emitting;

  std::function<LogicalResult(triton::FuncOp)> visit =
      [&](triton::FuncOp func) -> LogicalResult {
    if (deviceFns_.count(func.getSymName().str()))
      return success();
    if (!emitting.insert(func.getSymName()).second) {
      func.emitError("AgpuEmitter: a device function calls itself");
      return failure();
    }

    LogicalResult callees = success();
    func.walk([&](triton::CallOp call) {
      auto target = mod_.lookupSymbol<triton::FuncOp>(call.getCallee());
      if (!target) {
        call.emitError("AgpuEmitter: tt.call names no function in this module");
        callees = failure();
        return;
      }
      if (failed(visit(target)))
        callees = failure();
    });
    emitting.erase(func.getSymName());
    if (failed(callees))
      return failure();
    // The parameters must not outlive the function: the kernel loop snapshots
    // body_.sym, and an owner_ claim is permanent.
    const agpu::SymbolTable outer = body_.sym;
    const LogicalResult added = addDeviceFn(func);
    body_.sym = outer;
    return added;
  };

  for (auto func : mod_.getOps<triton::FuncOp>())
    if (callTargets.contains(func.getSymName()))
      if (failed(visit(func)))
        return failure();
  return success();
}

LogicalResult AgpuEmitter::addDeviceFn(triton::FuncOp func) {
  agpu::DeviceFnFacts f;
  f.name = agpu::deviceFnSymbol(func.getSymName());

  std::vector<am::Str> paramNames;
  for (auto [i, argTy] : llvm::enumerate(func.getFunctionType().getInputs())) {
    const std::optional<agpu::DeviceValue> v =
        deviceValueOf(argTy, registersHeldByType(argTy));
    if (!v) {
      func.emitError("AgpuEmitter: unsupported device function argument type");
      return failure();
    }
    f.params.push_back(*v);
    paramNames.push_back("p" + std::to_string(i));
  }
  for (Type resTy : func.getFunctionType().getResults()) {
    const std::optional<agpu::DeviceValue> v =
        deviceValueOf(resTy, registersHeldByType(resTy));
    if (!v) {
      func.emitError("AgpuEmitter: unsupported device function result type");
      return failure();
    }
    f.results.push_back(*v);
  }

  for (auto [i, argTy] : llvm::enumerate(func.getFunctionType().getInputs())) {
    const BlockArgument arg = func.getArgument(i);
    if (const std::optional<agpu::ElemType> e = elemTypeOf(argTy))
      elemFor_[idOf(arg)] = *e;
    if (f.params[i].isTensor()) {
      agpu::ValueNames regs;
      for (int64_t r = 0; r < f.params[i].regCount; ++r)
        regs.push_back(paramNames[i] + "_" + std::to_string(r));
      body_.sym.bindRegs(idOf(arg), std::move(regs));
    } else {
      body_.sym.bindScalar(idOf(arg), paramNames[i]);
      if (f.params[i].isPointer)
        markBasePointer(idOf(arg));
    }
  }

  // No blockId: a device function has no threadgroup position.
  agpu::ThreadNames tn;
  tn.blockId = {};
  body_.hoist = agpu::CoordHoist(tn);
  body_.pool = PoolLedger{};

  am::Block walked;
  Region &region = func.getBody();
  const bool ok = [&] {
    CurBlock in(*this, walked);
    return walkWholeRegion(region, walked).ok();
  }();
  if (!ok)
    return failure();

  am::Block body;
  agpu::emitLaneWarpPrologue(agpu_.context(), body, agpu::KernelNames{});
  for (am::Stmt *s : poolDecls())
    body.push_back(s);
  for (am::Stmt *s : body_.hoist.decls)
    body.push_back(s);
  for (am::Stmt *s : walked)
    body.push_back(s);

  // Decided after the walk so a function that stages through the pool takes
  // the pool parameter. anyScratch keeps the convention that every function
  // after the first pool user takes it too, so callers forward one pointer.
  const int64_t poolBytes = body_.pool.usedBytes();
  f.moduleNeedsPool = agpu_.pool.anyScratch() || poolBytes > 0;
  f.moduleAsserts = agpu_.asserts.asserts();
  if (poolBytes > 0)
    agpu_.pool.scratch("device fn", agpu::Bytes(poolBytes));

  const agpu::DeviceFnAbi abi = agpu::planDeviceFn(f);
  if (!abi.usable) {
    func.emitError("AgpuEmitter: this device function's shape is not lowered");
    return failure();
  }

  if (failed(appendDeviceReturn(func, abi, body)))
    return failure();

  deviceFns_[func.getSymName().str()] = {f, abi};
  agpu_.addDeviceFn(std::move(f), std::move(paramNames), std::move(body));
  return success();
}

agpu::Decision AgpuEmitter::walkWholeRegion(Region &region, am::Block &out,
                                            const TerminatorFn &atTerminator) {
  if (!region.hasOneBlock())
    return walkRegionCFG(region, out, atTerminator);
  Block &only = region.front();
  if (const agpu::Decision d = walkBlock(only, out); !d.ok())
    return d;
  return atTerminator ? atTerminator(only, out) : agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::walkRegionCFG(Region &region, am::Block &out,
                                          const TerminatorFn &atTerminator) {
  agpu::RegionFacts f = regionFactsOf(region);

  // Checked before any block is walked so a malformed edge set (arity
  // mismatch) is reported, with no extra args silently dropped.
  {
    const agpu::RegionPlan p = agpu::planRegion(f);
    if (!p.usable)
      return agpu::Decision::failed();
    if (const agpu::Decision d = agpu::regionDecision(f, p); !d.ok())
      return d;
  }

  // Block arguments are phis, bound before any block is walked so the first
  // read finds a register.
  for (Block &b : region)
    // Entry block's args are the function's parameters, already bound.
    if (&b != &region.front())
      for (BlockArgument arg : b.getArguments())
        carriedFresh(arg);

  std::vector<am::Block> bodies(f.blocks.size());
  std::vector<am::Expr *> conds(f.blocks.size(), nullptr);
  agpu::BlockId at = 0;
  for (Block &b : region) {
    if (const agpu::Decision d = walkBlock(b, bodies[(std::size_t)at]); !d.ok())
      return d;
    if (atTerminator)
      if (const agpu::Decision d = atTerminator(b, bodies[(std::size_t)at]);
          !d.ok())
        return d;
    if (auto cbr = dyn_cast<cf::CondBranchOp>(b.getTerminator())) {
      const am::Str *name = body_.sym.regAt(idOf(cbr.getCondition()), 0);
      if (!name)
        return agpu::Decision::failed();
      conds[(std::size_t)at] = agpu_.context().var(*name);
    }
    ++at;
  }

  std::map<am::Str, agpu::ValueId> offsetOwner;
  for (const auto &[key, off] : body_.offsetOf)
    if (off.owned)
      offsetOwner.emplace(off.name, key.first);
  const auto heldOwnersOf = [&](agpu::ValueId v,
                                std::vector<agpu::ValueId> &into) {
    if (const agpu::ValueNames *held = body_.sym.namesOf(v))
      for (const am::Str &n : *held)
        if (const agpu::ValueId o = body_.sym.ownerOf(n);
            o != agpu::kNoValue && o != v)
          into.push_back(o);
    for (auto off = body_.offsetOf.lower_bound({v, 0});
         off != body_.offsetOf.end() && off->first.first == v; ++off) {
      if (off->second.owned)
        continue;
      if (const agpu::ValueId o = body_.sym.ownerOf(off->second.name);
          o != agpu::kNoValue) {
        into.push_back(o);
      } else if (const auto own = offsetOwner.find(off->second.name);
                 own != offsetOwner.end()) {
        into.push_back(own->second);
      }
    }
  };
  for (agpu::BlockFacts &bf : f.blocks) {
    std::vector<agpu::ValueId> owners;
    for (agpu::ValueId v : bf.reads)
      heldOwnersOf(v, owners);
    for (const agpu::Edge &e : bf.edges)
      for (agpu::ValueId a : e.args)
        heldOwnersOf(a, owners);
    for (agpu::ValueId o : owners)
      bf.reads.push_back(o);
  }
  const agpu::RegionPlan p = agpu::planRegion(f);

  agpu::RegionNames nm;
  nm.stateVar = "state" + body_.scope;
  nm.namesOf = [this](agpu::ValueId v) {
    const agpu::ValueNames *n = body_.sym.namesOf(v);
    return n ? *n : agpu::ValueNames{};
  };
  nm.typeOf = [this](agpu::ValueId v) {
    const agpu::ElemType *e = elemOf(v);
    return e ? agpu::mslTypeOf(*e) : am::Type::scalar(am::Scalar::I32);
  };
  nm.storageOf = [this](agpu::ValueId v) { return storageOf(v); };

  agpu::emitRegion(
      agpu_.context(), out, f, p, nm,
      [&](agpu::BlockId b) { return std::move(bodies[(std::size_t)b]); },
      [&](agpu::BlockId b) { return conds[(std::size_t)b]; });
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitCallOp(const agpu::OpView &o) {
  const auto it = deviceFns_.find(std::string(o.text));
  if (it == deviceFns_.end())
    return declined("tt.call", "the callee was never emitted");
  const agpu::DeviceFnFacts &f = it->second.facts;
  const agpu::DeviceFnAbi &abi = it->second.abi;

  std::vector<am::Str> args;
  for (auto [i, operand] : llvm::enumerate(o.operands)) {
    if (i >= f.params.size())
      return declined("tt.call", "more arguments than parameters");
    for (int64_t r = 0; r < f.params[i].regCount; ++r) {
      const am::Str *n = body_.sym.regAt(operand, (std::size_t)r);
      if (!n)
        return declined("tt.call", "an argument register has no name");
      args.push_back(*n);
    }
  }

  std::vector<am::Str> resultNames;
  for (std::size_t k = 0; k < o.results.size(); ++k)
    for (int64_t r = 0; r < f.results[k].regCount; ++r)
      resultNames.push_back(nameFor('q', o.results[k], r));

  const agpu::KernelNames knm;
  agpu::CallerContext caller;
  caller.threadgroupPos = knm.threadgroupPos;
  caller.threadId = knm.threadId;
  caller.gridSize = knm.gridSize;
  caller.pool = knm.pool;
  caller.assertBuffer = knm.assertBuffer;

  const am::Str tmp = "callret" + std::to_string(body_.callSeq++) + body_.scope;
  agpu::emitDeviceCall(agpu_.context(), *cur_, f, abi, args, caller,
                       resultNames, tmp);

  std::size_t at = 0;
  for (std::size_t k = 0; k < o.results.size(); ++k) {
    agpu::ValueNames regs;
    for (int64_t r = 0; r < f.results[k].regCount; ++r)
      regs.push_back(resultNames[at++]);
    body_.sym.bindRegs(o.results[k], std::move(regs));
  }
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerCallHandler() {
  // The callee's signature/ABI is looked up.
  table_.add("call", agpu::forOps({"tt.call"}, [this](const agpu::OpView &o) {
               return emitCallOp(o);
             }));
}

} // namespace mlir::triton::applegpu::bridge
