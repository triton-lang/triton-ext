"""Public TLX surface, vendored onto stock Triton + the triton-utlx plugin.

Three adaptations are needed versus the in-tree build:

1. `utlx_plugin` must be imported first -- that is what registers the TLX DSL
   as `triton.language.extra.tlx`, which every kernel under `ops/` imports.
2. Stock `triton.Config` has no `ctas_per_cga` keyword (it is a Meta addition).
   The kernels only ever pass `(n, 1, 1)` or None, which upstream spells
   `num_ctas=n`, so translate it.
3. A config the host Triton cannot lower -- currently any NUM_CTAS=2 kernel,
   which needs `mapa.shared::cluster` and the `.cta_group::2` TMA modifier --
   is reported as `UnsupportedOp`, the catalog's "no entry for this target"
   signal, rather than as a hard error. Callers already handle that by
   declining the shape.
"""

import functools
import inspect

import triton as _triton

import utlx_plugin as _utlx_plugin  # noqa: F401  (registers triton.language.extra.tlx)

if "ctas_per_cga" not in inspect.signature(_triton.Config.__init__).parameters:

    class _Config(_triton.Config):
        """`triton.Config` that accepts `ctas_per_cga` and maps it to `num_ctas`."""

        def __init__(self, kwargs, *args, ctas_per_cga=None, **kw):
            if ctas_per_cga is not None:
                kw.setdefault("num_ctas", int(ctas_per_cga[0]))
            super().__init__(kwargs, *args, **kw)

    _triton.Config = _Config

from . import ops  # noqa: E402


def _unsupported_by_host(exc):
    """The host-capability error behind `exc`, if that is what it is.

    Triton wraps a frontend exception in `CompilationError`, so walk the chain.
    """
    from utlx_plugin.host_caps import HostFeatureUnavailable

    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, HostFeatureUnavailable):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def _decline_when_host_cannot_lower(fn):
    """Turn "this Triton build cannot lower that" into `UnsupportedOp`."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ops.UnsupportedOp:
            raise
        except Exception as exc:
            unsupported = _unsupported_by_host(exc)
            if unsupported is None:
                raise
            raise ops.UnsupportedOp(str(unsupported)) from exc

    return wrapper


for _name in ("mm", "addmm", "bmm", "flash_attn", "hstu_attn",
              "kimi_delta_attention"):
    _impl = getattr(ops, _name, None)
    if _impl is not None:
        setattr(ops, _name, _decline_when_host_cannot_lower(_impl))

__all__ = ["ops"]
