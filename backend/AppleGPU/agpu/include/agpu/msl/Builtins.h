// Builtins.h - the names Metal spells for us, in one place. Strings only: a
// planner may name a builtin without being able to build a call.
#ifndef AGPU_MSL_BUILTINS_H
#define AGPU_MSL_BUILTINS_H

namespace agpu::msl::builtin {

namespace comp {
inline constexpr const char *X = "x";
inline constexpr const char *Y = "y";
inline constexpr const char *Z = "z";

// Null for an axis Metal has no component for.
inline const char *of(int axis) {
  switch (axis) {
  case 0:
    return X;
  case 1:
    return Y;
  case 2:
    return Z;
  }
  return nullptr;
}
} // namespace comp

} // namespace agpu::msl::builtin

#endif // AGPU_MSL_BUILTINS_H
