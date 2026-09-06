// EmitKernel.h - the whole function: signature, prologue, body.
#ifndef AGPU_EMIT_KERNEL_H
#define AGPU_EMIT_KERNEL_H

#include "agpu/emit/KernelAbi.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/FuncSize.h"
#include "agpu/msl/GuardFuse.h"
#include "agpu/msl/GuardSink.h"

#include <algorithm>
#include <functional>

namespace agpu {

// The statements a body build produced and the threadgroup pool bytes they
// address; the pool declaration is sized from this after the body is built.
struct BuiltBody {
  msl::Block stmts;
  int64_t poolBytes = 0;

  BuiltBody() = default;
  BuiltBody(msl::Block b) : stmts(std::move(b)) {}
  BuiltBody(msl::Block b, int64_t pool)
      : stmts(std::move(b)), poolBytes(pool) {}
};

// Called by `emitKernel`, possibly twice: once to measure, again if the size
// policy asks for the rolled form (`rollK`).
using BodyFn = std::function<BuiltBody(msl::Context &, bool rollK)>;

struct KernelFacts {
  msl::Str name;
  std::vector<KernelArg> args;
  int64_t numWarps = 1;
  int64_t poolBytes = 0;
  int64_t coreBudget = kTGCoreBudgetBytes;
  DebugChannels debug;
  msl::RollPrediction predictedRoll;
};

struct KernelResult {
  msl::Function *fn = nullptr;
  msl::FuncSize size;
  msl::ShrinkPlan shrink;
  bool reemitted = false;
  int64_t poolBytes = 0;
  Decision decision = Decision::failed();

  bool ok() const { return fn != nullptr && decision.ok(); }
};

inline void emitLaneWarpPrologue(msl::Context &c, msl::Block &body,
                                 const KernelNames &nm) {
  msl::Expr *flat = c.member(c.var(nm.threadId), msl::builtin::comp::X);
  body.push_back(
      c.declStmt(msl::Context::i32(), nm.laneId,
                 c.binary(msl::BinOp::And, flat, c.lit(kWarpSize - 1))));
  body.push_back(c.declStmt(msl::Context::i32(), nm.warpId,
                            c.binary(msl::BinOp::Div, flat, c.lit(kWarpSize))));
}

inline void emitArgUnpack(msl::Context &c, msl::Block &body,
                          const std::vector<KernelArg> &args,
                          const KernelAbi &abi, const KernelNames &nm) {
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (abi.placements[i].slot != ArgSlot::ArgBuffer)
      continue;
    const msl::Type ty = mslTypeOf(args[i].elem);
    msl::Expr *addr = c.binary(msl::BinOp::Add, c.var(nm.argBuffer),
                               c.lit(abi.placements[i].offset));
    body.push_back(c.declStmt(
        ty, args[i].name,
        c.deref(c.cast(ty.pointerTo(msl::AddrSpace::Constant), addr))));
  }
}

inline void addParams(msl::Function *fn, const std::vector<KernelArg> &args,
                      const KernelAbi &abi, const KernelNames &nm) {
  using A = msl::Attribute;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (abi.placements[i].slot != ArgSlot::Buffer)
      continue;
    const unsigned quals =
        args[i].coherent ? msl::Type::Coherent : msl::Type::QualNone;
    fn->params.push_back(msl::Function::Param{
        mslTypeOf(args[i].elem).pointerTo(msl::AddrSpace::Device, quals),
        args[i].name, A::buffer(abi.placements[i].index)});
  }
  if (abi.hasArgBuffer)
    fn->params.push_back(msl::Function::Param{
        msl::Type::scalar(msl::Scalar::U8).pointerTo(msl::AddrSpace::Constant),
        nm.argBuffer, A::buffer(abi.argBufferIndex)});

  // Metal requires atomic_uint* for the atomic fetch-add the allocator uses.
  if (abi.hasPrintBuffer)
    fn->params.push_back(
        msl::Function::Param{msl::Type::named(msl::builtin::atomic::Uint)
                                 .pointerTo(msl::AddrSpace::Device),
                             nm.printBuffer, A::buffer(abi.printBufferIndex)});

  if (abi.hasAssertBuffer)
    fn->params.push_back(msl::Function::Param{
        msl::Type::named(msl::builtin::atomic::Uint)
            .pointerTo(msl::AddrSpace::Device),
        nm.assertBuffer, A::buffer(abi.assertBufferIndex)});

  const msl::Type u3 = msl::Type::vector(msl::Scalar::U32, 3);
  fn->params.push_back(msl::Function::Param{
      u3, nm.threadgroupPos, A::builtin(A::Kind::ThreadgroupPositionInGrid)});
  fn->params.push_back(msl::Function::Param{
      u3, nm.threadId, A::builtin(A::Kind::ThreadPositionInThreadgroup)});
  fn->params.push_back(msl::Function::Param{
      u3, nm.gridSize, A::builtin(A::Kind::ThreadgroupsPerGrid)});
}

// Declared as bytes and cast at each use: the pool holds tiles of several
// element types at once.
inline void emitPoolDecl(msl::Context &c, msl::Block &body, int64_t poolBytes,
                         const KernelNames &nm) {
  if (poolBytes <= 0)
    return;
  body.push_back(c.arrayDecl(msl::Type::scalar(msl::Scalar::I8)
                                 .inAddrSpace(msl::AddrSpace::Threadgroup),
                             nm.pool, poolBytes));
}

inline KernelResult emitKernel(msl::Context &c, const KernelFacts &f,
                               const BodyFn &buildBody,
                               const KernelNames &nm = {}) {
  KernelResult r;
  const KernelAbi abi = planKernelAbi(f.args, f.numWarps, f.debug);
  r.decision = abiDecision(abi);
  if (!r.decision.ok())
    return r;

  auto assemble = [&](bool rollK) {
    BuiltBody built = buildBody(c, rollK);
    msl::Block body;
    emitArgUnpack(c, body, f.args, abi, nm);
    emitLaneWarpPrologue(c, body, nm);
    for (msl::Stmt *s : built.stmts)
      body.push_back(s);
    built.stmts = std::move(body);
    return built;
  };

  msl::Block body;
  int64_t bodyPool = 0;
  // A predicted roll builds the rolled form first and checks the same
  // predicates against the inferred unrolled size. A wrong prediction falls
  // through to the measured path below, so the answer is unchanged.
  if (f.predictedRoll.roll) {
    BuiltBody rolled = assemble(/*rollK=*/true);
    const msl::FuncSize after = msl::measure(rolled.stmts);
    const msl::FuncSize before = msl::unrolledFrom(after, f.predictedRoll);
    const msl::ShrinkPlan plan = msl::planShrink(before);
    if (plan.needsReemit() && msl::shrinkHelped(before, after)) {
      body = std::move(rolled.stmts);
      bodyPool = rolled.poolBytes;
      r.size = after;
      r.shrink = plan;
      r.reemitted = true;
    }
  }

  if (!r.reemitted) {
    BuiltBody flat = assemble(/*rollK=*/false);
    body = std::move(flat.stmts);
    bodyPool = flat.poolBytes;
    r.size = msl::measure(body);
    r.shrink = msl::planShrink(r.size);

    if (r.shrink.needsReemit()) {
      BuiltBody rolled = assemble(/*rollK=*/true);
      const msl::FuncSize after = msl::measure(rolled.stmts);
      if (msl::shrinkHelped(r.size, after)) {
        body = std::move(rolled.stmts);
        bodyPool = rolled.poolBytes;
        r.size = after;
        r.reemitted = true;
      }
    }
  }

  if (r.shrink.fuseGuards) {
    // Sink first: fusion only merges adjacent guards. Folded literals only;
    // anything else yields an invalid set (not disjoint from anything) and the
    // sink stops.
    auto literalCoords = [](msl::Expr *e) {
      if (e && e->kind == msl::ExprKind::Literal) {
        auto *l = static_cast<msl::Literal *>(e);
        if (l->form == msl::Literal::Form::Int)
          return exactCoord((std::int32_t)l->intValue);
      }
      return unknownCoords();
    };
    // To a fixpoint, bounded by statement count so a non-converging shape
    // stops.
    for (std::size_t pass = 0; pass < body.size(); ++pass)
      if (msl::sinkGuardedStores(body, literalCoords) == 0)
        break;
    msl::fuseGuards(c, body);
  }

  r.poolBytes = std::max(f.poolBytes, bodyPool);
  if (r.poolBytes > 0) {
    msl::Block withPool;
    emitPoolDecl(c, withPool, r.poolBytes, nm);
    for (msl::Stmt *s : body)
      withPool.push_back(s);
    body = std::move(withPool);
  }

  msl::Function *fn = c.function();
  fn->isKernel = true;
  fn->name = f.name;
  addParams(fn, f.args, abi, nm);
  if (shouldPinThreadgroupSize(r.poolBytes, f.coreBudget, abi.launchThreads))
    fn->qualifier = msl::Attribute::maxThreads(abi.launchThreads);
  fn->body = std::move(body);

  r.fn = fn;
  r.decision = Decision::emitted();
  return r;
}

} // namespace agpu

#endif // AGPU_EMIT_KERNEL_H
