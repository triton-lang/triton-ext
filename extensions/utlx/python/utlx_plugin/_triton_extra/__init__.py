"""Core-Triton modules that Meta's fork ships and upstream Triton does not.

Nothing here is TLX. These are pieces of ``triton.language.extra`` that TLX
kernels import by their core paths:

    from triton.language.extra.subtile_ops import _split_n_2D
    from triton.language.extra.cuda.inline_ptx_lib import _fma_f32x2

Upstream has no such modules, so those imports fail and the kernel never gets
as far as any TLX op. Both are small and written entirely against stock Triton
(``core.inline_asm_elementwise``, ``tl.join``/``permute``/``split``/
``reshape``), so uTLX can supply them rather than leaving the kernels
unimportable.

``install()`` registers each one in ``sys.modules`` under its core path, and
only when that path does not already resolve. On Meta's fork, and on any future
upstream that grows them, the real module wins and uTLX stays out of the way --
these are a fallback for a namespace uTLX does not own, not a replacement.

Registration is by ``sys.modules`` alone, which is what the ``from <path>
import <name>`` form above needs. Binding the module as an attribute of its
parent is deliberately not attempted: ``triton.language.extra`` loads its
submodules by hand (``module_from_spec`` + ``exec_module``) without binding
them either, so more than one module object for e.g.
``triton.language.extra.cuda`` can be live at once and an attribute set on one
is invisible from the other.
"""

import importlib
import sys

# Core module path -> submodule of this package that stands in for it.
_SHIMS = {
    "triton.language.extra.subtile_ops": "subtile_ops",
    "triton.language.extra.cuda.inline_ptx_lib": "inline_ptx_lib",
}


def _already_available(path):
    """True when Triton itself provides *path*."""
    if path in sys.modules:
        return True
    try:
        return importlib.util.find_spec(path) is not None
    except (ImportError, AttributeError, ValueError):
        # A parent package that does not exist or refuses inspection means
        # there is nothing to defer to.
        return False


def install():
    """Fill in each missing core module. Returns the paths uTLX supplied."""
    installed = []
    for path, name in _SHIMS.items():
        if _already_available(path):
            continue
        sys.modules[path] = importlib.import_module(f".{name}", __name__)
        installed.append(path)
    return installed
