// DeviceFn.h - the shape of a non-kernel function, decided once.
//
// A called function is an ordinary MSL function: its thread context
// parameters carry no attributes and arrive as bare `uint3`; it cannot
// declare threadgroup memory, so the pool is threaded down as a
// `threadgroup char *`; it must be forward-declared; and several results
// become a struct.
//
// The definition, the prototype and the call all read this one value.
#ifndef AGPU_DEVICE_FN_H
#define AGPU_DEVICE_FN_H

#include "agpu/core/Decline.h"
#include "agpu/core/Names.h"
#include "agpu/plan/Elementwise.h"

#include <cstdint>
#include <string>
#include <vector>

namespace agpu {

// A tensor is a value with more than one register; there is no separate flag.
struct DeviceValue {
  ElemType elem = i32();
  bool isPointer = false;
  int64_t regCount = 1;

  bool isTensor() const { return regCount > 1; }
};

// The call site makes the same decision to unpack the result.
enum class RetShape {
  Void,   // no results
  Scalar, // one non-tensor result: returned directly
  Struct, // one tensor, or two or more results: returned as a struct
};

// Appended to every device function's parameters, in this order and passed
// by every call in the same order.
enum class ImplicitArg {
  ThreadgroupPos,
  ThreadId,
  ThreadgroupCount,
  Pool,    // only when the module needs one; see DeviceFnAbi::needsPool
  Asserts, // only when the module asserts; see DeviceFnAbi::needsAsserts
};

// Must match `KernelNames`: the same walk lowers a device function's body and
// a kernel's. test_builtins checks the agreement.
struct DeviceFnNames : ThreadNames {
  msl::Str threadgroupPos = "tgid";
  msl::Str gridSize = "tgcount";
  msl::Str pool = "pool";
  msl::Str assertBuffer = "asserts";

  // Prefix for a returned struct's fields and for the struct type itself.
  msl::Str retFieldPrefix = "f";
  msl::Str retTypeSuffix = "_ret";

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

struct DeviceFnFacts {
  msl::Str name;
  std::vector<DeviceValue> params;
  std::vector<DeviceValue> results;

  // Whether the module needs a threadgroup pool. The pool parameter is part
  // of the calling convention, so every
  // device function in a module with a pool takes it.
  bool moduleNeedsPool = false;

  // Same convention as the pool: a module that asserts passes the buffer to
  // every device function, since the assert can sit in any of them.
  bool moduleAsserts = false;
};

struct DeviceFnAbi {
  RetShape ret = RetShape::Void;

  // A tensor result contributes `regCount` fields of its element type; a
  // scalar contributes one. Empty when `ret` is Void.
  std::vector<ElemType> retFields;

  // The implicit parameters, in the order they are appended and passed.
  std::vector<ImplicitArg> implicit;

  bool usable = true;
  bool needsPool() const {
    for (ImplicitArg a : implicit)
      if (a == ImplicitArg::Pool)
        return true;
    return false;
  }

  bool needsAsserts() const {
    for (ImplicitArg a : implicit)
      if (a == ImplicitArg::Asserts)
        return true;
    return false;
  }

  bool returnsStruct() const { return ret == RetShape::Struct; }
};

// The name of the struct a function returns, when it returns one.
inline msl::Str retTypeName(const DeviceFnFacts &f,
                            const DeviceFnNames &nm = {}) {
  return f.name + nm.retTypeSuffix;
}

// Used by both the definition and the call.
inline msl::Str retFieldName(int64_t i, const DeviceFnNames &nm = {}) {
  return nm.retFieldPrefix + std::to_string(i);
}

inline DeviceFnAbi planDeviceFn(const DeviceFnFacts &f) {
  DeviceFnAbi abi;

  if (f.results.empty()) {
    abi.ret = RetShape::Void;
  } else if (f.results.size() == 1 && !f.results[0].isTensor()) {
    abi.ret = RetShape::Scalar;
    abi.retFields.push_back(f.results[0].elem);
  } else {
    // MSL functions return one value, so the registers travel as a struct.
    abi.ret = RetShape::Struct;
    for (const DeviceValue &r : f.results)
      for (int64_t i = 0; i < r.regCount; ++i)
        abi.retFields.push_back(r.elem);
  }

  abi.implicit = {ImplicitArg::ThreadgroupPos, ImplicitArg::ThreadId,
                  ImplicitArg::ThreadgroupCount};
  if (f.moduleNeedsPool)
    abi.implicit.push_back(ImplicitArg::Pool);
  if (f.moduleAsserts)
    abi.implicit.push_back(ImplicitArg::Asserts);

  return abi;
}

inline Decision deviceFnDecision(const DeviceFnAbi &abi) {
  if (abi.usable)
    return Decision::emitted();
  return Decision::declined("deviceFn", "signature cannot be expressed");
}

// Module-wide pool sizing lives in plan/PoolPlan.h.

} // namespace agpu

#endif // AGPU_DEVICE_FN_H
