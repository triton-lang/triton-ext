// KernelAbi.h - a kernel's signature and how the host reaches its arguments.
// The host side must match this.
#ifndef AGPU_KERNEL_ABI_H
#define AGPU_KERNEL_ABI_H

#include "agpu/core/Decline.h"
#include "agpu/core/Names.h"
#include "agpu/core/Units.h"
#include "agpu/plan/Elementwise.h"

#include <cstdint>
#include <vector>

namespace agpu {

struct KernelNames : ThreadNames {
  msl::Str threadgroupPos = "tgid";
  msl::Str gridSize = "tgcount";
  msl::Str argBuffer = "args";
  msl::Str pool = "pool";
  msl::Str printBuffer = "prints";
  msl::Str assertBuffer = "asserts";
};

// Metal binds at most 31 buffers, indices 0-30. Checked here so the failure is
// one message.
inline constexpr int64_t kMaxBuffers = 31;

struct KernelArg {
  msl::Str name;
  ElemType elem = i32();
  bool isPointer = false;

  // Qualifies the parameter type, so it covers every access through it.
};

enum class ArgSlot {
  Buffer,
  ArgBuffer,
};

struct ArgPlacement {
  ArgSlot slot = ArgSlot::Buffer;
  int64_t index = 0;
  int64_t offset = 0;
};

// Rounds up: a sub-byte argument occupies one byte.
inline int64_t argSizeOf(ElemType e) { return byteWidthOf(e); }

struct KernelAbi {
  std::vector<ArgPlacement> placements;
  int64_t bufferCount = 0;
  int64_t argBufferBytes = 0;
  bool hasArgBuffer = false;
  int64_t argBufferIndex = 0;

  int64_t launchThreads = 0;

  bool usable() const { return bufferCount <= kMaxBuffers; }
};

// Pointers take buffer bindings in order; scalars pack into one constant
// buffer at their natural alignment. Any later buffer appends after these, so
// adding one cannot renumber a pointer binding. The launcher's `ptr_args`
// order is positional.
inline KernelAbi planKernelAbi(const std::vector<KernelArg> &args,
                               int64_t numWarps) {
  KernelAbi abi;
  abi.launchThreads = threadsFor(numWarps);
  abi.placements.resize(args.size());

  int64_t buffer = 0;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (!args[i].isPointer)
      continue;
    abi.placements[i] = ArgPlacement{ArgSlot::Buffer, buffer++, 0};
  }

  int64_t off = 0;
  bool anyScalar = false;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i].isPointer)
      continue;
    anyScalar = true;
    const int64_t size = argSizeOf(args[i].elem);
    // A misaligned constant read is undefined on Apple silicon.
    if (size > 0)
      off = (off + size - 1) / size * size;
    abi.placements[i] = ArgPlacement{ArgSlot::ArgBuffer, 0, off};
    off += size;
  }

  abi.hasArgBuffer = anyScalar;
  abi.argBufferBytes = off;
  if (anyScalar)
    abi.argBufferIndex = buffer++;

  abi.bufferCount = buffer;
  return abi;
}

inline Decision abiDecision(const KernelAbi &abi) {
  if (abi.usable())
    return Decision::emitted();
  return Decision::declined("kernelAbi",
                            "more buffer bindings than Metal allows");
}

// Unpinned, the compiler budgets registers for a worst-case threadgroup and
// can cap the pipeline below the actual launch, failing at dispatch with
} // namespace agpu

#endif // AGPU_KERNEL_ABI_H
