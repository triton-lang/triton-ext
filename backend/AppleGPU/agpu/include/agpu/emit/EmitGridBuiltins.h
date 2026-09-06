// EmitGridBuiltins.h - program id and grid size, read off the kernel's names.
#ifndef AGPU_EMIT_GRID_BUILTINS_H
#define AGPU_EMIT_GRID_BUILTINS_H

#include "agpu/emit/KernelAbi.h"
#include "agpu/msl/Builtins.h"
#include "agpu/msl/Context.h"

namespace agpu {

enum class GridQuery { ProgramId, NumPrograms };

// Metal names the grid dimensions by component. Null past axis 2.
inline const char *axisComponent(int axis) {
  return msl::builtin::comp::of(axis);
}

// A view of KernelNames.
struct GridNames {
  msl::Str threadgroupPos;
  msl::Str gridSize;

  explicit GridNames(const KernelNames &k = KernelNames{})
      : threadgroupPos(k.threadgroupPos), gridSize(k.gridSize) {}

  const msl::Str &of(GridQuery q) const {
    return q == GridQuery::ProgramId ? threadgroupPos : gridSize;
  }
};

// Cast to i32: the builtin is unsigned and a signed comparison against it
// downstream would promote the wrong way.
inline msl::Stmt *emitGridQuery(msl::Context &c, GridQuery q, int axis,
                                const msl::Str &name,
                                const GridNames &nm = GridNames{}) {
  const char *comp = axisComponent(axis);
  if (!comp)
    return nullptr;
  return c.declStmt(msl::Type::scalar(msl::Scalar::I32), name,
                    c.cast(msl::Type::scalar(msl::Scalar::I32),
                           c.member(c.var(nm.of(q)), comp)));
}

} // namespace agpu

#endif // AGPU_EMIT_GRID_BUILTINS_H
