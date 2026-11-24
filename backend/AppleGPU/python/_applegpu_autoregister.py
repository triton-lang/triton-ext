"""Point Triton at the bundled Apple GPU plugin before anything imports Triton.

Run at interpreter startup from ``applegpu_backend.pth``, because a Triton
older than ``extend_with`` reads ``TRITON_PLUGIN_PATHS`` only while its own
``libtriton`` is imported. By the time ``import triton_apple_backend`` runs it
is too late for that Triton to see the plugin. Newer Tritons ignore the
variable and are registered explicitly by the package itself.

Set ``TRITON_APPLE_NO_AUTOREGISTER=1`` to skip, and import
``triton_apple_backend`` before ``triton`` instead.

This runs in every interpreter that has the backend installed, including ones
that never touch Triton, so it imports nothing but ``os``.
"""

import os

PLUGIN_PATHS_ENV = "TRITON_PLUGIN_PATHS"
OPT_OUT_ENV = "TRITON_APPLE_NO_AUTOREGISTER"


def register():
    here = os.path.dirname(os.path.abspath(__file__))
    library = os.path.join(here, "triton_apple_backend",
                           "libapplegpu_backend.dylib")
    if not os.path.isfile(library):
        return

    paths = [
        p for p in os.environ.get(PLUGIN_PATHS_ENV, "").split(os.pathsep) if p
    ]
    if library not in paths:
        paths.append(library)
        os.environ[PLUGIN_PATHS_ENV] = os.pathsep.join(paths)


if not os.environ.get(OPT_OUT_ENV):
    register()
