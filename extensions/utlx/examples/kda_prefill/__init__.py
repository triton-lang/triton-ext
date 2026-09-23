"""gfx950 TLX KDA kernels.

The kernels import ``triton.language.extra.tlx``, which only exists in a
TLX-capable triton (fbtriton). Import is therefore lazy: ``is_available()``
answers without pulling the kernels in, so callers on stock AMD triton can
probe and fall back.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


# The decode kernel only needs the async-copy primitives, which have been in
# TLX since 3.7.0. The prefill kernel additionally needs the layout API, which
# fbtriton 3.7.4 does not ship -- it compiles against a newer TLX. Probing the
# two separately lets decode run on the released wheel while prefill falls
# back, instead of failing the whole backend.
_DECODE_REQUIRES = ("async_load", "local_alloc", "local_view", "local_load")
_PREFILL_REQUIRES = (
    "amd_mfma_layout",
    "dot_operand_layout",
    "release_layout",
    "slice_layout",
    "zeros",
)


def _tlx_module():
    try:
        from triton.language.extra import tlx
    except ImportError:
        return None
    return tlx


def is_available() -> bool:
    """Whether this triton exposes the TLX primitives the decode kernel uses."""
    tlx = _tlx_module()
    return tlx is not None and all(hasattr(tlx, n) for n in _DECODE_REQUIRES)


def is_prefill_available() -> bool:
    """Whether TLX is new enough for the chunk prefill kernel as well."""
    tlx = _tlx_module()
    return tlx is not None and all(
        hasattr(tlx, n) for n in (*_DECODE_REQUIRES, *_PREFILL_REQUIRES)
    )


def missing_prefill_ops() -> tuple[str, ...]:
    """Names the prefill kernel needs that this TLX does not provide."""
    tlx = _tlx_module()
    if tlx is None:
        return _PREFILL_REQUIRES
    return tuple(n for n in _PREFILL_REQUIRES if not hasattr(tlx, n))


_EXPORTS = {
    "kda_recurrent_decode": "kimi_k3_kda_decode",
    "kda_paged_prefill": "kimi_k3_kda_prefill",
    # plain Triton, no TLX needed -- builds the prepared inputs the two above
    # require in one pass instead of a dozen torch launches.
    "prepare_kda_inputs": "kimi_k3_kda_prepare",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)


__all__ = [
    "is_available",
    "is_prefill_available",
    "missing_prefill_ops",
    *_EXPORTS,
]
