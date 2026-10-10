// AgpuAssume - llvm.intr.assume, handed to the Metal compiler as a hint.
#include "AgpuEmitter.h"

#include "agpu/msl/Builtins.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

agpu::Decision AgpuEmitter::emitAssumeOp(LLVM::AssumeOp as) {
  const Value cond = as.getCond();
  const int64_t count = registersHeldBy(idOf(cond));
  if (count != 1)
    return declined("llvm.intr.assume", "the condition is not a scalar");

  const am::Str *name = body_.sym.regAt(idOf(cond), 0);
  if (!name)
    return declined("llvm.intr.assume", "the condition has no register name");

  am::Context &mc = agpu_.context();
  cur_->push_back(
      mc.exprStmt(mc.call(am::builtin::clang::Assume, {mc.var(*name)})));
  return agpu::Decision::emitted();
}

} // namespace mlir::triton::applegpu::bridge
