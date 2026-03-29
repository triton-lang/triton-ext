"""uTLX CLC (Cluster Launch Control) operations for Blackwell dynamic launch."""

import triton.language.core as tl

from . import types as tlx
from .mem_ops import local_view
from .barrier import alloc_barriers, barrier_expect_bytes, barrier_wait, barrier_arrive


@tl.builtin
def _alloc_clc_responses(num_responses: tl.constexpr,
                         _semantic=None) -> tlx.clc_response:
    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
    num_responses_val = tl._unwrap_if_constexpr(num_responses)
    handle = _semantic.builder.utlx_alloc_clc_responses(
        [_semantic.builder.get_int32(int(num_responses_val))])
    return tlx.clc_response(handle, num_responses_val, layout)


@tl.builtin
def _clc_issue(clc_response_addr, barrier, _semantic=None):
    assert isinstance(clc_response_addr, tlx.clc_response)
    _semantic.builder.utlx_async_clc_try_cancel(
        [barrier.handle, clc_response_addr.handle])


@tl.builtin
def _clc_query(clc_response_addr, _semantic=None):
    assert isinstance(clc_response_addr, tlx.clc_response)
    x = _semantic.builder.utlx_clc_query([clc_response_addr.handle])
    return tl.tensor(x, tl.int32)


@tl.builtin
def clc_create_context(num_consumers,
                       num_stages=1,
                       _semantic=None) -> tlx.CLCPipelineContext:
    if not isinstance(num_stages, tl.constexpr):
        num_stages = tl.constexpr(num_stages)
    if not isinstance(num_consumers, tl.constexpr):
        num_consumers = tl.constexpr(num_consumers)
    return tlx.CLCPipelineContext(
        clc_mbars_empty=alloc_barriers(num_barriers=num_stages,
                                       arrive_count=num_consumers,
                                       _semantic=_semantic),
        clc_mbars_full=alloc_barriers(num_barriers=num_stages,
                                      _semantic=_semantic),
        clc_responses=_alloc_clc_responses(num_responses=num_stages,
                                           _semantic=_semantic),
    )


@tl.builtin
def clc_producer(context,
                 p_producer=None,
                 multi_ctas=False,
                 k=0,
                 _semantic=None):
    """Issue a CLC try_cancel request from the first CTA in the cluster."""
    bar_empty = local_view(context._clc_mbars_empty, k, _semantic=_semantic)
    bar_full = local_view(context._clc_mbars_full, k, _semantic=_semantic)
    response = local_view(context._clc_responses, k, _semantic=_semantic)

    if multi_ctas:
        cta_rank = _semantic.builder.utlx_cluster_cta_rank([])
        zero = _semantic.builder.get_int32(0)
        pred_cta0_handle = _semantic.builder.create_icmpEQ(cta_rank, zero)
        pred_cta0 = tl.tensor(pred_cta0_handle, tl.int1)
    else:
        pred_cta0 = None

    barrier_wait(bar_empty, p_producer, pred_cta0, _semantic=_semantic)
    barrier_expect_bytes(bar_full, tl.constexpr(16), _semantic=_semantic)
    _clc_issue(response, bar_full, _semantic=_semantic)


@tl.builtin
def clc_consumer(context,
                 p_consumer=None,
                 multi_ctas=False,
                 k=0,
                 _semantic=None):
    """Decode tile ID from CLC response and signal completion."""
    bar_empty = local_view(context._clc_mbars_empty, k, _semantic=_semantic)
    bar_full = local_view(context._clc_mbars_full, k, _semantic=_semantic)
    response = local_view(context._clc_responses, k, _semantic=_semantic)

    barrier_wait(bar_full, p_consumer, _semantic=_semantic)
    stolen_tile_id = _clc_query(response, _semantic=_semantic)

    if multi_ctas:
        barrier_arrive(bar_empty, tl.constexpr(1), 0, _semantic=_semantic)
    else:
        barrier_arrive(bar_empty, _semantic=_semantic)

    return stolen_tile_id
