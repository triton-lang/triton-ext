// EmitKernel.h - the whole function: signature, prologue, body.
#ifndef AGPU_EMIT_KERNEL_H
#define AGPU_EMIT_KERNEL_H

#include "agpu/emit/KernelAbi.h"
#include "agpu/msl/Context.h"

#include <algorithm>
#include <functional>

namespace agpu {

// The statements a body build produced and the threadgroup pool bytes they
// address; the pool declaration is sized from this after the body is built.
struct BuiltBody {
  msl::Block stmts;

  BuiltBody() = default;
  BuiltBody(msl::Block b) : stmts(std::move(b)) {}
};

using BodyFn = std::function<BuiltBody(msl::Context &)>;

struct KernelFacts {
  msl::Str name;
  std::vector<KernelArg> args;
  int64_t numWarps = 1;
};

struct KernelResult {
  msl::Function *fn = nullptr;
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
    fn->params.push_back(msl::Function::Param{
        mslTypeOf(args[i].elem).pointerTo(msl::AddrSpace::Device), args[i].name,
        A::buffer(abi.placements[i].index)});
  }
  if (abi.hasArgBuffer)
    fn->params.push_back(msl::Function::Param{
        msl::Type::scalar(msl::Scalar::U8).pointerTo(msl::AddrSpace::Constant),
        nm.argBuffer, A::buffer(abi.argBufferIndex)});

  const msl::Type u3 = msl::Type::vector(msl::Scalar::U32, 3);
  fn->params.push_back(msl::Function::Param{
      u3, nm.threadgroupPos, A::builtin(A::Kind::ThreadgroupPositionInGrid)});
  fn->params.push_back(msl::Function::Param{
      u3, nm.threadId, A::builtin(A::Kind::ThreadPositionInThreadgroup)});
  fn->params.push_back(msl::Function::Param{
      u3, nm.gridSize, A::builtin(A::Kind::ThreadgroupsPerGrid)});
}

// Declared as bytes and cast at each use: the pool holds tiles of several
inline KernelResult emitKernel(msl::Context &c, const KernelFacts &f,
                               const BodyFn &buildBody,
                               const KernelNames &nm = {}) {
  KernelResult r;
  const KernelAbi abi = planKernelAbi(f.args, f.numWarps);
  r.decision = abiDecision(abi);
  if (!r.decision.ok())
    return r;

  BuiltBody built = buildBody(c);
  msl::Block body;
  emitArgUnpack(c, body, f.args, abi, nm);
  emitLaneWarpPrologue(c, body, nm);
  for (msl::Stmt *s : built.stmts)
    body.push_back(s);

  msl::Function *fn = c.function();
  fn->isKernel = true;
  fn->name = f.name;
  addParams(fn, f.args, abi, nm);
  fn->body = std::move(body);

  r.fn = fn;
  r.decision = Decision::emitted();
  return r;
}

} // namespace agpu

#endif // AGPU_EMIT_KERNEL_H
