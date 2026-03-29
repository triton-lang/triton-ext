"""uTLX Plugin barrier ops.

These ops call plugin custom ops registered by uTLXPlugin.cpp:
  - utlx_alloc_barriers(numBarriers, arriveCount) -> MemDesc Value
  - utlx_barrier_wait(mbarrier, phase, pred)
  - utlx_barrier_arrive(mbarrier, arriveCount)
  - utlx_barrier_expect(mbarrier, expectBytes, pred)
  - utlx_named_barrier_wait(barrier, numThreads)
  - utlx_named_barrier_arrive(barrier, numThreads)
"""

import triton.language.core as tl

from . import types as tlx
from .utility import is_hip


@tl.builtin
def cluster_barrier(_semantic=None):
    _semantic.builder.create_cluster_barrier()


@tl.builtin
def alloc_barriers(
    num_barriers: tl.constexpr,
    arrive_count: tl.constexpr = tl.constexpr(1),
    _semantic=None,
) -> tlx.mbarrier:
    """Allocate mbarriers in shared memory."""
    num_barriers_val = tl._unwrap_if_constexpr(num_barriers)
    arrive_count_val = tl._unwrap_if_constexpr(arrive_count)

    num_barriers_ir = _semantic.builder.get_int32(int(num_barriers_val))
    arrive_count_ir = _semantic.builder.get_int32(int(arrive_count_val))

    args = [num_barriers_ir, arrive_count_ir]
    handle = _semantic.builder.utlx_alloc_barriers(args)

    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
    return tlx.mbarrier(handle, num_barriers_val, layout)


@tl.builtin
def alloc_warp_barrier(
    num_barriers: tl.constexpr,
    num_warps: tl.constexpr = tl.constexpr(1),
    num_arrivals: tl.constexpr = tl.constexpr(1),
    _semantic=None,
) -> tlx.mbarrier:
    """Allocate warp barriers where all threads arrive independently."""
    num_barriers_val = tl._unwrap_if_constexpr(num_barriers)
    num_warps_val = tl._unwrap_if_constexpr(num_warps)
    num_arrivals_val = tl._unwrap_if_constexpr(num_arrivals)
    arrive_count = num_warps_val * 32 * num_arrivals_val

    num_barriers_ir = _semantic.builder.get_int32(int(num_barriers_val))
    arrive_count_ir = _semantic.builder.get_int32(int(arrive_count))

    args = [num_barriers_ir, arrive_count_ir]
    handle = _semantic.builder.utlx_alloc_barriers(args)

    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
    return tlx.mbarrier(handle, num_barriers_val, layout, is_warp_barrier=True)


@tl.builtin
def barrier_expect_bytes(
    bar: tlx.mbarrier,
    size: tl.constexpr,
    pred: tl.tensor = None,
    _semantic=None,
) -> None:
    """Signal a barrier of an expected number of bytes to be copied."""
    if pred is None:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        pred_handle = pred.handle

    size_val = tl._unwrap_if_constexpr(size)
    size_ir = _semantic.builder.get_int32(int(size_val))

    _semantic.builder.utlx_barrier_expect([bar.handle, size_ir, pred_handle])


@tl.builtin
def barrier_wait(
    bar: tlx.buffered_tensor,
    phase,
    pred: tl.tensor = None,
    _semantic=None,
) -> None:
    """Wait until the mbarrier phase completes."""
    assert bar.type.storage == tlx.storage_kind.smem, (
        "barrier_wait does not support remote_view of mbarrier.")

    if pred is None:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        pred_handle = pred.handle

    if isinstance(phase, tl.tensor):
        phase_handle = phase.handle
    elif isinstance(phase, tl.constexpr):
        phase_handle = _semantic._convert_elem_to_ir_value(phase.value,
                                                           require_i64=False)
    else:
        raise RuntimeError(
            f"`phase` must be tl.tensor or tl.constexpr, got {type(phase)}")

    _semantic.builder.utlx_barrier_wait(
        [bar.handle, phase_handle, pred_handle])


@tl.builtin
def barrier_arrive(
    bar: tlx.buffered_tensor,
    arrive_count: tl.constexpr = tl.constexpr(1),
    remote_cta_rank: tl.tensor = None,
    _semantic=None,
) -> None:
    """Perform the arrive operation on an mbarrier."""
    assert bar.type.storage == tlx.storage_kind.smem
    arrive_count_val = tl._unwrap_if_constexpr(arrive_count)
    assert arrive_count_val == 1 or not is_hip(
    ), "AMD backend currently only supports arrive_count == 1"

    if remote_cta_rank is not None:
        from .mem_ops import remote_view
        bar = remote_view(bar, remote_cta_rank, _semantic=_semantic)

    arrive_count_ir = _semantic.builder.get_int32(int(arrive_count_val))

    _semantic.builder.utlx_barrier_arrive([bar.handle, arrive_count_ir])


@tl.builtin
def named_barrier_wait(
    bar: int,
    arrive_count: int,
    _semantic=None,
) -> None:
    """Wait until arrive_count threads have reached the named barrier."""
    bar_handle = _semantic._convert_elem_to_ir_value(bar, require_i64=False)
    arrive_count_handle = _semantic._convert_elem_to_ir_value(
        arrive_count, require_i64=False)
    _semantic.builder.utlx_named_barrier_wait(
        [bar_handle, arrive_count_handle])


@tl.builtin
def named_barrier_arrive(
    bar: tl.constexpr,
    arrive_count: tl.constexpr,
    _semantic=None,
) -> None:
    """Signal arrival at a named barrier."""
    bar_handle = _semantic._convert_elem_to_ir_value(bar, require_i64=False)
    arrive_count_handle = _semantic._convert_elem_to_ir_value(
        arrive_count, require_i64=False)
    _semantic.builder.utlx_named_barrier_arrive(
        [bar_handle, arrive_count_handle])
