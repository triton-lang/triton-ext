"""pytest plugin that supplies fork-only helpers to ``triton._internal_testing``.

fbtriton's TLX op tests (``python/test/unit/tlx_ops``) import test helpers from
``triton._internal_testing`` that only Meta's fork defines. The ``pytest11``
entry point loads this plugin into every pytest run in an environment with uTLX
installed, so those tests run unmodified on stock Triton; outside pytest nothing
changes. Disable it with ``-p no:utlx``.

pytest imports plugins at startup, so this lives outside ``utlx_plugin`` (whose
import loads Triton) and imports nothing heavy. Helpers are added when
``triton._internal_testing`` is first imported, and only those it lacks: a
Triton that defines them keeps its own.
"""

import importlib.abc
import importlib.machinery
import sys

INTERNAL_TESTING = "triton._internal_testing"


def swizzle_scale_to_5d(scale, outer_chunks, k_chunks):
    """Convert raw block scales to the swizzled 5D TMA layout.

    Copied from fbtriton's ``triton/_internal_testing.py``.

    Applies the cuBLAS block scaling layout within each 128x4 block.
    dest[row%32 * 16 + row//32 * 4 + col] = src[row, col]

    Args:
        scale: Raw scale tensor of shape (batch, rows, scale_cols).
        outer_chunks: Number of 128-row chunks (rows // 128).
        k_chunks: Number of 4-column scale chunks (ceil(scale_cols / 4)).

    Returns:
        Swizzled 5D tensor of shape (batch, outer_chunks, k_chunks, 2, 256).
    """
    import torch  # type: ignore[import-not-found]

    batch = scale.shape[0]
    cols = scale.shape[2]
    padded_cols = k_chunks * 4

    if cols < padded_cols:
        scale = torch.nn.functional.pad(scale, (0, padded_cols - cols))

    blocks = (scale.reshape(batch, outer_chunks, 128, k_chunks,
                            4).permute(0, 1, 3, 2,
                                       4).reshape(batch, outer_chunks,
                                                  k_chunks, 512))

    _r = torch.arange(128)
    _c = torch.arange(4)
    _rg, _cg = torch.meshgrid(_r, _c, indexing="ij")
    idx = ((_rg % 32) * 16 + (_rg // 32) * 4 + _cg).reshape(-1)
    idx = idx.to(scale.device).expand_as(blocks)
    output = torch.empty_like(blocks)
    output.scatter_(-1, idx, blocks)

    return output.reshape(batch, outer_chunks, k_chunks, 2, 256)


HELPERS = {"swizzle_scale_to_5d": swizzle_scale_to_5d}


def backfill(module):
    """Add each helper *module* lacks."""
    for name, helper in HELPERS.items():
        if not hasattr(module, name):
            setattr(module, name, helper)


class _BackfillLoader(importlib.abc.Loader):
    """Run the real loader, then backfill the module it produced."""

    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        backfill(module)


class _BackfillFinder(importlib.abc.MetaPathFinder):
    """Wrap the loader of ``triton._internal_testing`` in ``_BackfillLoader``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != INTERNAL_TESTING:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _BackfillLoader(spec.loader)
        return spec


if INTERNAL_TESTING in sys.modules:
    backfill(sys.modules[INTERNAL_TESTING])
else:
    sys.meta_path.insert(0, _BackfillFinder())
