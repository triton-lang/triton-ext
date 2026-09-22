"""uTLX warp-specialization helpers.

Imported as ``triton.language.extra.tlx.warp_spec``, matching the module path
Meta's TLX exposes, because kernels written against TLX reach for it directly:

    from triton.language.extra.tlx.warp_spec import get_bufidx_phase

Deliberately not re-exported from ``__init__``, so the qualified import above
stays the only spelling and matches TLX.
"""

import triton
import triton.language as tl


@triton.jit
def get_bufidx_phase(accum_cnt, NUM_BUFFERS: tl.constexpr):
    """Map a monotonic accumulation count onto a buffer slot and mbarrier phase.

    ``accum_cnt`` counts every pass through a multi-buffered stage, so the slot
    cycles every ``NUM_BUFFERS`` passes and the phase flips each time the ring
    wraps -- which is the parity an mbarrier wait needs to distinguish this
    lap's arrival from the previous one's.
    """
    buf_idx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return buf_idx, phase
