// AgpuDebug - tt.assert and tt.print, recorded into a module-level plan the
// host reads back.
#include "AgpuEmitter.h"

#include "agpu/emit/EmitAssert.h"
#include "agpu/emit/EmitPrint.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

template <class Op> static agpu::DebugBinding bindingOf(triton::FuncOp func) {
  bool any = false;
  func.walk([&](Op) { any = true; });
  return any ? agpu::DebugBinding::Bound : agpu::DebugBinding::None;
}

agpu::DebugBinding AgpuEmitter::printBindingOf(triton::FuncOp func) {
  return bindingOf<triton::PrintOp>(func);
}

agpu::DebugBinding AgpuEmitter::assertBindingOf(triton::FuncOp func) {
  return bindingOf<triton::AssertOp>(func);
}

// Source location for a diagnostic the host raises.
static std::pair<std::string, int> sourceLocOf(Operation *op) {
  Location loc = op->getLoc();
  while (true) {
    if (auto c = dyn_cast<CallSiteLoc>(loc)) {
      loc = c.getCallee();
      continue;
    }
    if (auto n = dyn_cast<NameLoc>(loc)) {
      loc = n.getChildLoc();
      continue;
    }
    if (auto f = dyn_cast<FusedLoc>(loc)) {
      if (!f.getLocations().empty()) {
        loc = f.getLocations().front();
        continue;
      }
    }
    break;
  }
  if (auto fl = dyn_cast<FileLineColLoc>(loc))
    return {fl.getFilename().getValue().str(), (int)fl.getLine()};
  return {"unknown", 0};
}

// Whether anything after this op synchronises the threadgroup. Conservative:
// a barrier some threads skip is undefined in Metal.
static bool barrierSynchronises(Operation *o) {
  return isa<mlir::gpu::BarrierOp, triton::gpu::BarrierOp, triton::ReduceOp,
             triton::ScanOp>(o);
}

static bool barrierFollowsIn(Operation *op) {
  auto func = op->getParentOfType<triton::FuncOp>();
  if (!func)
    return true;

  // Inside a loop, "after" includes what the next iteration reaches, so any
  // barrier in the function counts.
  const bool inLoop =
      op->getParentOfType<scf::ForOp>() || op->getParentOfType<scf::WhileOp>();

  bool seenOp = false;
  bool barrier = false;
  func.walk([&](Operation *o) {
    if (o == op) {
      seenOp = true;
      return;
    }
    if (!seenOp && !inLoop)
      return;
    if (barrierSynchronises(o))
      barrier = true;
  });
  return barrier;
}

agpu::Decision AgpuEmitter::emitAssertOp(triton::AssertOp as) {
  agpu::AssertSite site;
  site.message = as.getMessage().str();
  const std::pair<std::string, int> where = sourceLocOf(as);
  site.file = where.first;
  site.line = where.second;

  agpu::AssertContext ctx;
  ctx.barrierFollows = barrierFollowsIn(as);
  site.halt = agpu::assertHaltFor(ctx);

  const Value cond = as.getCondition();
  const int64_t count = registersHeldBy(idOf(cond));
  std::vector<am::Str> names;
  for (int64_t r = 0; r < count; ++r) {
    const am::Str *name = body_.sym.regAt(idOf(cond), (std::size_t)r);
    if (!name)
      return declined("tt.assert", "the condition has no register name");
    names.push_back(*name);
  }

  site.site = agpu_.asserts.add(site);
  agpu_.helpers.add(agpu::Helper::AssertRecord);
  agpu::emitAssert(agpu_.context(), *cur_, site, names, agpu::KernelNames{});
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitPrintOp(triton::PrintOp pr) {
  am::Context &mc = agpu_.context();
  agpu::PrintSite site;
  site.prefix = pr.getPrefix().str();
  site.hex = pr.getHex();

  // MLIR integer types are signless; `isSigned` from the IR is the only
  // source of signedness.
  const llvm::ArrayRef<int32_t> signedness = pr.getIsSigned();

  for (auto [i, arg] : llvm::enumerate(pr.getArgs())) {
    std::optional<agpu::ElemType> elem = elemTypeOf(arg.getType());
    if (!elem)
      return declined("tt.print", "an operand type has no representation");
    if (i < signedness.size())
      elem->isUnsigned = signedness[i] == 0;

    agpu::PrintOperand operand;
    operand.elem = *elem;

    const int64_t regs = tensorRegisterCountOf(idOf(arg));
    operand.distributed = regs > 0;
    const int64_t count = registersHeldBy(idOf(arg));
    for (int64_t r = 0; r < count; ++r) {
      const am::Str *name = body_.sym.regAt(idOf(arg), (std::size_t)r);
      if (!name)
        return declined("tt.print", "an operand has no register name");
      operand.regs.push_back(*name);
    }
    site.operands.push_back(std::move(operand));
  }

  site.site = agpu_.prints.add(site);
  agpu_.helpers.add(agpu::Helper::PrintAppend);

  std::vector<std::vector<am::Expr *>> indices(site.operands.size());
  for (std::size_t o = 0; o < site.operands.size(); ++o) {
    const agpu::PrintOperand &operand = site.operands[o];
    indices[o].resize(operand.regs.size());
    for (std::size_t r = 0; r < operand.regs.size(); ++r) {
      if (!operand.distributed) {
        indices[o][r] = mc.lit(0);
        continue;
      }
      indices[o][r] = coordOf(pr.getArgs()[o], (int)r);
      if (!indices[o][r])
        return declined("tt.print",
                        "an operand's layout has no element coordinate");
    }
  }

  agpu::emitPrint(
      mc, *cur_, site,
      [&](std::size_t o, std::size_t r) { return indices[o][r]; },
      agpu::KernelNames{});
  return agpu::Decision::emitted();
}

} // namespace mlir::triton::applegpu::bridge
