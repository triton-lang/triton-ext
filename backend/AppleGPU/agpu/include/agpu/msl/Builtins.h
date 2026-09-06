// Builtins.h - the MSL names the language provides. Strings only, no nodes,
// so plan/ may include it.
#ifndef AGPU_MSL_BUILTINS_H
#define AGPU_MSL_BUILTINS_H

#include "agpu/msl/Containers.h"

namespace agpu::msl::builtin {

namespace math {
inline constexpr const char *Abs = "metal::abs";
inline constexpr const char *Fabs = "metal::fabs";
inline constexpr const char *Floor = "metal::floor";
inline constexpr const char *Ceil = "metal::ceil";
inline constexpr const char *Isnan = "metal::isnan";
inline constexpr const char *Isinf = "metal::isinf";
inline constexpr const char *Min = "metal::min";
inline constexpr const char *Max = "metal::max";
inline constexpr const char *Fma = "metal::fma";
inline constexpr const char *Copysign = "metal::copysign";
inline constexpr const char *Clamp = "metal::clamp";
inline constexpr const char *Sign = "metal::sign";
inline constexpr const char *Mulhi = "metal::mulhi";

// `round` rounds half away from zero; `rint` rounds half to even. Distinct
// MLIR ops: math.round and math.roundeven.
inline constexpr const char *Round = "metal::round";
inline constexpr const char *RoundEven = "metal::rint";
inline constexpr const char *Trunc = "metal::trunc";
} // namespace math

// In Metal the namespace controls accuracy.
struct AccuracyPair {
  const char *fast;
  const char *precise;
};

namespace accuracy {
inline constexpr AccuracyPair Sin{"metal::sin", "metal::precise::sin"};
inline constexpr AccuracyPair Cos{"metal::cos", "metal::precise::cos"};
inline constexpr AccuracyPair Tanh{"metal::tanh", "metal::precise::tanh"};
inline constexpr AccuracyPair Exp{"metal::exp", "metal::precise::exp"};
inline constexpr AccuracyPair Exp2{"metal::exp2", "metal::precise::exp2"};
inline constexpr AccuracyPair Exp10{"metal::exp10", "metal::precise::exp10"};
inline constexpr AccuracyPair Log{"metal::log", "metal::precise::log"};
inline constexpr AccuracyPair Log2{"metal::log2", "metal::precise::log2"};
inline constexpr AccuracyPair Log10{"metal::log10", "metal::precise::log10"};
inline constexpr AccuracyPair Pow{"metal::pow", "metal::precise::pow"};
inline constexpr AccuracyPair Atan2{"metal::atan2", "metal::precise::atan2"};
inline constexpr AccuracyPair Fmod{"metal::fmod", "metal::precise::fmod"};
// metal::sqrt(12.25) answers 3.5000002, so an exact square does not round trip.
inline constexpr AccuracyPair Sqrt{"metal::sqrt", "metal::precise::sqrt"};
inline constexpr AccuracyPair Rsqrt{"metal::rsqrt", "metal::precise::rsqrt"};

inline constexpr AccuracyPair Sinh{"metal::sinh", "metal::precise::sinh"};
inline constexpr AccuracyPair Cosh{"metal::cosh", "metal::precise::cosh"};
inline constexpr AccuracyPair Tan{"metal::tan", "metal::precise::tan"};
inline constexpr AccuracyPair Asin{"metal::asin", "metal::precise::asin"};
inline constexpr AccuracyPair Acos{"metal::acos", "metal::precise::acos"};
inline constexpr AccuracyPair Atan{"metal::atan", "metal::precise::atan"};
} // namespace accuracy

// Cross-lane shuffles.
namespace simd {
inline constexpr const char *Shuffle = "simd_shuffle";
inline constexpr const char *ShuffleUp = "simd_shuffle_up";
inline constexpr const char *ShuffleDown = "simd_shuffle_down";
inline constexpr const char *ShuffleXor = "simd_shuffle_xor";

// Whole-simdgroup folds. Metal has no bfloat or 64-bit overload; see
// `simdFoldable`.
inline constexpr const char *Sum = "simd_sum";
inline constexpr const char *Product = "simd_product";
inline constexpr const char *Min = "simd_min";
inline constexpr const char *Max = "simd_max";
inline constexpr const char *And = "simd_and";
inline constexpr const char *Or = "simd_or";
inline constexpr const char *Xor = "simd_xor";
inline constexpr const char *PrefixInclusiveSum = "simd_prefix_inclusive_sum";
inline constexpr const char *PrefixExclusiveSum = "simd_prefix_exclusive_sum";
inline constexpr const char *PrefixInclusiveProduct =
    "simd_prefix_inclusive_product";
inline constexpr const char *PrefixExclusiveProduct =
    "simd_prefix_exclusive_product";
} // namespace simd

// simdgroup_matrix intrinsics. The type names live in CanonicalFragment.
namespace sg {
inline constexpr const char *Load = "simdgroup_load";
inline constexpr const char *Store = "simdgroup_store";
inline constexpr const char *MultiplyAccumulate =
    "simdgroup_multiply_accumulate";
inline constexpr const char *TypePrefix = "simdgroup_";
} // namespace sg

// Explicit-ordering atomics.
namespace atomic {
inline constexpr const char *Uint = "atomic_uint";
inline constexpr const char *Load = "atomic_load_explicit";
inline constexpr const char *Store = "atomic_store_explicit";
inline constexpr const char *Exchange = "atomic_exchange_explicit";
inline constexpr const char *ThreadFence = "atomic_thread_fence";
inline constexpr const char *FetchAdd = "atomic_fetch_add_explicit";
inline constexpr const char *FetchAnd = "atomic_fetch_and_explicit";
inline constexpr const char *FetchMax = "atomic_fetch_max_explicit";
inline constexpr const char *FetchMin = "atomic_fetch_min_explicit";
inline constexpr const char *FetchOr = "atomic_fetch_or_explicit";
inline constexpr const char *FetchXor = "atomic_fetch_xor_explicit";
inline constexpr const char *CompareExchangeWeak =
    "atomic_compare_exchange_weak_explicit";
} // namespace atomic

namespace order {
inline constexpr const char *Relaxed = "memory_order_relaxed";
inline constexpr const char *SeqCst = "memory_order_seq_cst";
} // namespace order

namespace memflags {
inline constexpr const char *Device = "mem_flags::mem_device";
inline constexpr const char *Threadgroup = "mem_flags::mem_threadgroup";
inline constexpr const char *DeviceAndThreadgroup =
    "mem_flags::mem_threadgroup | mem_flags::mem_device";
} // namespace memflags

namespace barrier {
inline constexpr const char *Threadgroup = "threadgroup_barrier";
inline constexpr const char *Simdgroup = "simdgroup_barrier";
} // namespace barrier

// Metal has none of these; the prelude defines them. Using a name here means
// also emitting the body, through Prelude.h's `HelperSet`. Named here because
// plan/ picks some as a lowering and may not include emit/.
namespace helper {
inline constexpr const char *AtomicRmwF32 = "__agpu_atomic_rmw_f32";
inline constexpr const char *AtomicRmwPacked16 = "__agpu_atomic_rmw_packed16";
// Defined inside AtomicRmwPacked16's body, which is its only caller.
inline constexpr const char *Narrow16 = "__agpu_narrow16";
inline constexpr const char *Erf = "__agpu_erf";
inline constexpr const char *Cbrt = "__agpu_cbrt";
inline constexpr const char *RtneHalf = "__agpu_rtne_half";
inline constexpr const char *RtneBfloat = "__agpu_rtne_bfloat";
inline constexpr const char *RtneIntHalf = "__agpu_rtne_int_half";
inline constexpr const char *RtneIntBfloat = "__agpu_rtne_int_bfloat";
inline constexpr const char *RtzHalf = "__agpu_rtz_half";
inline constexpr const char *RtzBfloat = "__agpu_rtz_bfloat";
inline constexpr const char *Fp8PackE4M3 = "__agpu_f32_to_e4m3";
inline constexpr const char *Fp8UnpackE4M3 = "__agpu_e4m3_to_f32";
inline constexpr const char *Fp8PackE5M2 = "__agpu_f32_to_e5m2";
inline constexpr const char *Fp8UnpackE5M2 = "__agpu_e5m2_to_f32";
inline constexpr const char *Fp8PackE4B8 = "__agpu_f32_to_e4b8";
inline constexpr const char *Fp8UnpackE4B8 = "__agpu_e4b8_to_f32";
inline constexpr const char *Fp8PackE5B16 = "__agpu_f32_to_e5b16";
inline constexpr const char *Fp8UnpackE5B16 = "__agpu_e5b16_to_f32";
inline constexpr const char *Fp4Unpack = "__agpu_e2m1_to_f32";
inline constexpr const char *PrintAppend = "__agpu_print_append";
inline constexpr const char *AssertRecord = "__agpu_assert_record";
} // namespace helper

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
