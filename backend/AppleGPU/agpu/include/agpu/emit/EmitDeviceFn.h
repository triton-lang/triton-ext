// EmitDeviceFn.h - a called function, its prototype and the calls to it.
//
// Definition, prototype and calls all build from one `DeviceFnAbi`.
#ifndef AGPU_EMIT_DEVICE_FN_H
#define AGPU_EMIT_DEVICE_FN_H

#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/DeviceFn.h"

namespace agpu {

inline msl::Type deviceValueType(const DeviceValue &v) {
  const msl::Type base = mslTypeOf(v.elem);
  return v.isPointer ? base.pointerTo(msl::AddrSpace::Device) : base;
}

inline msl::Stmt *emitRetStruct(msl::Context &c, const DeviceFnFacts &f,
                                const DeviceFnAbi &abi,
                                const DeviceFnNames &nm = {}) {
  if (!abi.returnsStruct())
    return nullptr;
  msl::StructDecl *d = c.structDecl();
  d->name = retTypeName(f, nm);
  for (std::size_t i = 0; i < abi.retFields.size(); ++i)
    d->fields.push_back(
        {mslTypeOf(abi.retFields[i]), retFieldName((int64_t)i, nm)});
  return d;
}

inline msl::Type deviceRetType(const DeviceFnFacts &f, const DeviceFnAbi &abi,
                               const DeviceFnNames &nm = {}) {
  switch (abi.ret) {
  case RetShape::Void:
    return msl::Type::named("void");
  case RetShape::Scalar:
    return mslTypeOf(abi.retFields[0]);
  case RetShape::Struct:
    return msl::Type::named(retTypeName(f, nm));
  }
  return msl::Type::named("void");
}

inline msl::Type implicitType(ImplicitArg a) {
  if (a == ImplicitArg::Pool)
    return msl::Type::scalar(msl::Scalar::I8)
        .pointerTo(msl::AddrSpace::Threadgroup);
  if (a == ImplicitArg::Asserts)
    return msl::Type::named(msl::builtin::atomic::Uint)
        .pointerTo(msl::AddrSpace::Device);
  return msl::Type::vector(msl::Scalar::U32, 3);
}

// Shared by the definition and the prototype: a short `paramNames` leaves the
// rest unnamed, which a prototype allows.
//
// Thread-context parameters carry no attribute:
// `[[thread_position_in_threadgroup]]` on a non-kernel function does not
// compile, so these are bare `uint3`s the caller fills in.
inline void addDeviceParams(msl::Function *fn, const DeviceFnFacts &f,
                            const DeviceFnAbi &abi,
                            const std::vector<msl::Str> &paramNames,
                            const DeviceFnNames &nm = {}) {
  for (std::size_t i = 0; i < f.params.size(); ++i)
    fn->params.push_back(msl::Function::Param{
        deviceValueType(f.params[i]),
        i < paramNames.size() ? paramNames[i] : msl::Str{}, msl::Attribute{}});

  for (ImplicitArg a : abi.implicit)
    fn->params.push_back(
        msl::Function::Param{implicitType(a), nm.of(a), msl::Attribute{}});
}

inline msl::Function *emitDeviceProto(msl::Context &c, const DeviceFnFacts &f,
                                      const DeviceFnAbi &abi,
                                      const DeviceFnNames &nm = {}) {
  msl::Function *fn = c.function();
  fn->isPrototype = true;
  fn->name = f.name;
  fn->returnType = deviceRetType(f, abi, nm);
  addDeviceParams(fn, f, abi, {}, nm);
  return fn;
}

inline msl::Stmt *emitDeviceReturn(msl::Context &c, const DeviceFnAbi &abi,
                                   const std::vector<msl::Str> &values) {
  switch (abi.ret) {
  case RetShape::Void:
    return c.returnStmt();
  case RetShape::Scalar:
    return c.returnStmt(c.var(values.empty() ? msl::Str{} : values[0]));
  case RetShape::Struct: {
    msl::SmallVec<msl::Expr *, 4> fields;
    for (const msl::Str &v : values)
      fields.push_back(c.var(v));
    return c.returnStructStmt(std::move(fields));
  }
  }
  return c.returnStmt();
}

inline msl::Function *emitDeviceFn(msl::Context &c, const DeviceFnFacts &f,
                                   const DeviceFnAbi &abi,
                                   const std::vector<msl::Str> &paramNames,
                                   msl::Block body,
                                   const DeviceFnNames &nm = {}) {
  msl::Function *fn = c.function();
  fn->name = f.name;
  fn->returnType = deviceRetType(f, abi, nm);
  addDeviceParams(fn, f, abi, paramNames, nm);
  fn->body = std::move(body);
  return fn;
}

// ── the call side ─────────────────────────────────────────────────────────

// Names the caller holds for the implicit arguments it passes down. Order
// comes from the ABI.
struct CallerContext {
  msl::Str threadgroupPos;
  msl::Str threadId;
  msl::Str gridSize;
  msl::Str pool;
  msl::Str assertBuffer;

  const msl::Str &of(ImplicitArg a) const {
    switch (a) {
    case ImplicitArg::ThreadgroupPos:
      return threadgroupPos;
    case ImplicitArg::ThreadId:
      return threadId;
    case ImplicitArg::ThreadgroupCount:
      return gridSize;
    case ImplicitArg::Pool:
      return pool;
    case ImplicitArg::Asserts:
      return assertBuffer;
    }
    return pool;
  }
};

inline msl::Expr *deviceCallExpr(msl::Context &c, const DeviceFnFacts &f,
                                 const DeviceFnAbi &abi,
                                 const std::vector<msl::Str> &args,
                                 const CallerContext &caller) {
  msl::SmallVec<msl::Expr *, 4> all;
  for (const msl::Str &a : args)
    all.push_back(c.var(a));
  for (ImplicitArg a : abi.implicit)
    all.push_back(c.var(caller.of(a)));
  return c.call(f.name, all);
}

// `resultNames` are the names the results get bound to; for a struct return
// there is one per field and `tmp` names the struct itself.
inline void emitDeviceCall(msl::Context &c, msl::Block &body,
                           const DeviceFnFacts &f, const DeviceFnAbi &abi,
                           const std::vector<msl::Str> &args,
                           const CallerContext &caller,
                           const std::vector<msl::Str> &resultNames,
                           const msl::Str &tmp = "callret",
                           const DeviceFnNames &nm = {}) {
  msl::Expr *call = deviceCallExpr(c, f, abi, args, caller);

  switch (abi.ret) {
  case RetShape::Void:
    body.push_back(c.exprStmt(call));
    return;

  case RetShape::Scalar:
    body.push_back(c.declStmt(mslTypeOf(abi.retFields[0]),
                              resultNames.empty() ? tmp : resultNames[0],
                              call));
    return;

  case RetShape::Struct: {
    body.push_back(c.declStmt(deviceRetType(f, abi, nm), tmp, call));
    for (std::size_t i = 0; i < abi.retFields.size(); ++i) {
      if (i >= resultNames.size())
        break;
      body.push_back(
          c.declStmt(mslTypeOf(abi.retFields[i]), resultNames[i],
                     c.member(c.var(tmp), retFieldName((int64_t)i, nm))));
    }
    return;
  }
  }
}

} // namespace agpu

#endif // AGPU_EMIT_DEVICE_FN_H
