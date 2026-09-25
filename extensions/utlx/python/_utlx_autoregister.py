"""Point Triton at the bundled uTLX plugin before anything imports Triton.

Run at interpreter startup from ``utlx_plugin.pth``, because a Triton older
than ``extend_with`` reads ``TRITON_PLUGIN_PATHS`` only while its own
``libtriton`` is imported -- by the time ``import utlx_plugin`` runs, it is
already too late to register the plugin's ops. Newer Tritons ignore the
variable and register explicitly instead (see ``utlx_plugin/_compat.py``).

Set ``UTLX_NO_AUTOREGISTER=1`` to skip, and import ``utlx_plugin`` before
``triton`` instead.

It also installs the finder for uTLX's two Triton-namespaced aliases (see
``install_alias_finder``). That has to happen here rather than in
``utlx_plugin`` because the importers are third parties -- PyTorch reaches for
``triton.language.extra.tlx.inductor.registry`` while building its own
heuristics table, long before anything has imported uTLX.

This runs in every interpreter that has uTLX installed, including ones that
never touch Triton, so it imports nothing but ``os``, ``sys`` and the
``importlib`` machinery, and imports no module until one of the aliases is
actually requested.
"""

import importlib
import importlib.abc
import importlib.util
import os
import sys

PLUGIN_PATHS_ENV = "TRITON_PLUGIN_PATHS"
OPT_OUT_ENV = "UTLX_NO_AUTOREGISTER"

HERE = os.path.dirname(os.path.abspath(__file__))

# Module paths Meta's TLX occupies inside the ``triton`` package, which uTLX
# cannot create for real: it ships beside Triton rather than inside it.
TLX_LANGUAGE = "triton.language.extra.tlx"
TLX_OPS_ROOT = "triton.tlx"


def register():
    """Append the installed ``libutlx.so`` to ``TRITON_PLUGIN_PATHS``."""
    library = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "utlx_plugin", "libutlx.so")
    if not os.path.isfile(library):
        return

    paths = [
        p for p in os.environ.get(PLUGIN_PATHS_ENV, "").split(os.pathsep) if p
    ]
    if library not in paths:
        paths.append(library)
        os.environ[PLUGIN_PATHS_ENV] = os.pathsep.join(paths)


class _AliasLoader(importlib.abc.Loader):
    """Bind an alias to a module uTLX ships under a different name."""

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass  # already executed under its real name


class _PackageLoader(importlib.abc.Loader):
    """Create an empty package rooted at *path*, for a purely namespace alias."""

    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None  # default module object; __path__ comes from the spec

    def exec_module(self, module):
        pass


class UtlxAliasFinder(importlib.abc.MetaPathFinder):
    """Resolve the ``triton``-namespaced module paths uTLX stands in for.

    ``triton.language.extra.tlx`` is uTLX itself. ``utlx_plugin`` already
    publishes that name in ``sys.modules`` once imported, but a consumer that
    gets there first -- PyTorch's Inductor does -- would see a plain
    ``ModuleNotFoundError``, because ``triton.language.extra`` only discovers
    packages physically inside the installed Triton. Importing uTLX on demand
    makes the two orders equivalent.

    ``triton.tlx`` is the TLX op library, vendored under ``_triton_tlx``. It is
    aliased as a package rather than a module so that ``triton.tlx.ops`` and
    everything below it load under their real names, rather than a second
    identity for the same files.
    """

    _OPS_ROOT = os.path.join(HERE, "utlx_plugin", "_triton_tlx")

    def find_spec(self, fullname, path=None, target=None):
        if fullname == TLX_LANGUAGE:
            return importlib.util.spec_from_loader(fullname,
                                                   _AliasLoader("utlx_plugin"),
                                                   is_package=True)
        if fullname == TLX_OPS_ROOT and os.path.isdir(self._OPS_ROOT):
            spec = importlib.util.spec_from_loader(fullname,
                                                   _PackageLoader(
                                                       self._OPS_ROOT),
                                                   is_package=True)
            spec.submodule_search_locations = [self._OPS_ROOT]
            return spec
        return None


def install_alias_finder():
    """Append the alias finder, once, behind every real finder."""
    if any(isinstance(finder, UtlxAliasFinder) for finder in sys.meta_path):
        return
    # Appended, not prepended: a Triton that really does ship these modules
    # (Meta's fork) must keep resolving them itself.
    sys.meta_path.append(UtlxAliasFinder())


if not os.environ.get(OPT_OUT_ENV):
    register()
    install_alias_finder()
