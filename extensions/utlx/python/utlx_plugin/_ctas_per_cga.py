"""Support TLX's ``ctas_per_cga`` launch option on upstream Triton.

TLX kernels ask for thread-block clusters CUDA-style: the grid is the total CTA
count and ``ctas_per_cga`` regroups it into clusters::

    Config({...}, ctas_per_cga=(2, 1, 1))
    kernel[(total_ctas, 1, 1)](..., ctas_per_cga=(2, 1, 1))

Upstream spells clusters ``num_ctas`` and gives it the opposite grid
convention -- ``driver.c`` launches ``gridDimX * num_ctas`` CTAs, so the grid is
per-cluster rather than total. Both end up at the same
``CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION``; only the bookkeeping differs::

    TLX:      ctas_per_cga=(N,1,1), grid=G    ->  clusterDim.x=N, gridDimX=G
    upstream: num_ctas=N,           grid=G/N  ->  clusterDim.x=N, gridDimX=G

So the two are exactly interconvertible, and this module converts. Meta's fork
instead threads ``ctas_per_cga`` into ``CUDAOptions.cluster_dims`` and reads it
from a multi-dimensional cluster path in its own ``driver.c``; neither the
option field nor that path exists upstream, and a plugin cannot add them --
with ``num_ctas`` left at 1 upstream emits no cluster attribute at all. Hence
the conversion rather than a port.

Only ``(N, 1, 1)`` is expressible: upstream hardcodes ``clusterDim.y`` and
``clusterDim.z`` to 1. Anything else raises rather than silently launching a
different cluster shape.
"""

import triton.runtime.autotuner as _autotuner
import triton.runtime.jit as _jit


def _cluster_size(ctas_per_cga):
    """Validate a ``(N, 1, 1)`` cluster shape and return N."""
    dims = tuple(int(d) for d in ctas_per_cga)
    if len(dims) != 3:
        raise ValueError(
            f"ctas_per_cga must be a 3-tuple, got {ctas_per_cga!r}")
    if dims[1] != 1 or dims[2] != 1:
        raise ValueError(
            f"uTLX supports only ctas_per_cga=(N, 1, 1); got {dims!r}. "
            "Upstream Triton launches clusters through num_ctas, which fixes "
            "clusterDim.y and clusterDim.z at 1.")
    return dims[0]


def _rescale_grid(grid, n):
    """Convert a total-CTA grid to upstream's per-cluster grid."""

    def scale(resolved):
        dims = list(resolved)
        if dims[0] % n:
            raise ValueError(
                f"grid[0]={dims[0]} is not divisible by the cluster size {n}; "
                "with ctas_per_cga the grid is the total CTA count, so it must "
                "be a whole number of clusters.")
        dims[0] //= n
        return tuple(dims)

    if callable(grid):
        return lambda meta: scale(grid(meta))
    return scale(grid)


def install():
    """Teach Config and JITFunction.run about ``ctas_per_cga``. Idempotent."""
    if getattr(_jit.JITFunction, "_utlx_ctas_per_cga", False):
        return

    # Config carries the value; it is not translated here, so that one place
    # owns the conversion and autotuning still hashes the option.
    _orig_config_init = _autotuner.Config.__init__

    def _config_init(self,
                     *args,
                     ctas_per_cga=None,
                     preferred_ctas_per_cga=None,
                     **kwargs):
        if preferred_ctas_per_cga is not None:
            raise ValueError(
                "preferred_ctas_per_cga is not supported: it maps to "
                "CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, which "
                "upstream Triton's launcher never sets.")
        _orig_config_init(self, *args, **kwargs)
        if ctas_per_cga is not None:
            _cluster_size(ctas_per_cga)  # fail at construction, not at launch
        self.ctas_per_cga = ctas_per_cga

    _autotuner.Config.__init__ = _config_init

    # all_kwargs() forwards a fixed allowlist, so ctas_per_cga needs adding or
    # it never reaches the launch.
    _orig_all_kwargs = _autotuner.Config.all_kwargs

    def _all_kwargs(self):
        kwargs = _orig_all_kwargs(self)
        if getattr(self, "ctas_per_cga", None) is not None:
            kwargs["ctas_per_cga"] = self.ctas_per_cga
        return kwargs

    _autotuner.Config.all_kwargs = _all_kwargs

    # Both launch paths funnel through JITFunction.run: Autotuner.run calls it
    # with the chosen config's all_kwargs(), and fn[grid](...) calls it directly
    # through __getitem__.
    _orig_run = _jit.JITFunction.run

    def _run(self, *args, grid, warmup, **kwargs):
        ctas_per_cga = kwargs.pop("ctas_per_cga", None)
        if ctas_per_cga is not None:
            n = _cluster_size(ctas_per_cga)
            if n != 1:
                if kwargs.get("num_ctas", 1) != 1:
                    raise ValueError(
                        "num_ctas must be 1 when ctas_per_cga is set; the two "
                        "express the same cluster with opposite grid "
                        "conventions.")
                kwargs["num_ctas"] = n
                grid = _rescale_grid(grid, n)
        return _orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)

    _jit.JITFunction.run = _run
    _jit.JITFunction._utlx_ctas_per_cga = True
