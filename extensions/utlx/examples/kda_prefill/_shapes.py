"""Shape sets the upstream TLX benchmarks sweep.

Only the two ``*_FOCUS`` tuples the kernels import are kept; upstream pulls
these from a ``FocusRegistry`` that exists purely for its benchmark harness.
"""

from __future__ import annotations

from typing import NamedTuple


class KDADecodeShape(NamedTuple):
    batch: int
    heads: int
    key_dim: int
    value_dim: int
    dtype: str


class KDAPrefillShape(NamedTuple):
    total_tokens: int
    sequences: int
    heads: int
    key_dim: int
    value_dim: int
    dtype: str


GFX950_DECODE_FOCUS: tuple[KDADecodeShape, ...] = tuple(
    KDADecodeShape(batch, heads, 128, 128, "bf16")
    for heads in (4, 12)
    for batch in (1, 2, 4, 8, 16, 32)
)

GFX950_PREFILL_FOCUS: tuple[KDAPrefillShape, ...] = tuple(
    KDAPrefillShape(total_tokens, sequences, heads, 128, 128, "bf16")
    for heads in (4, 12)
    for total_tokens, sequences in ((4096, 1), (4096, 4), (131072, 1), (131072, 8))
)

__all__ = [
    "GFX950_DECODE_FOCUS",
    "GFX950_PREFILL_FOCUS",
    "KDADecodeShape",
    "KDAPrefillShape",
]
