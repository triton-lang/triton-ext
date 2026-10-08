"""Register the intensity pass as a Triton plugin.

Importing this package loads the compiled plugin library that is bundled
alongside this file and hands it to Triton's plugin API. Triton must already be
imported (so ``libtriton`` is loaded); the plugin resolves its MLIR/LLVM symbols
from that already-loaded library, so no ``LD_LIBRARY_PATH`` or
``TRITON_PLUGIN_PATHS`` is required.

Importing also installs the pass into Triton's compilation pipeline: it runs
at the end of every ``ttir`` stage and records its per-argument
``tint.load_bytes`` / ``tint.store_bytes`` / ``tint.op_count`` equations in the
kernel's compilation
metadata (see :mod:`.custom_stages`). Use :class:`IntensityListener`
(or :func:`enable`) to gather those equations per kernel function as kernels
are compiled, and :func:`intensity` /
:meth:`IntensityEquations.evaluate` to turn them into FLOP and byte
counts for a concrete launch.

:mod:`.utilities` (re-exported here) adds the plumbing benchmarks need on
top: :func:`launch` / :func:`record` / :func:`recording` to capture the
launches an operation performs and :class:`Work` to sum their work,
:func:`compile_with_config` to compile one ``triton.Config`` of an autotuned
kernel without autotuning (:class:`AutotuneRecorder` and
:class:`IntensityPruner` hook the autotuner to capture, respectively prune,
its configurations), and :func:`print_equations` to show the equations with
``args[i]`` resolved to parameter names.
"""

from __future__ import annotations

from pathlib import Path

import triton._C.libtriton as libtriton

PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libintensity.so"
libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes

from . import custom_stages, utilities  # noqa: E402
from .intensity import (  # noqa: E402
    LOAD_BYTES, METADATA_KEY, OP_COUNT, STORE_BYTES, Intensity,
    IntensityEquations, IntensityListener, intensity, enable,
)
from .utilities import (  # noqa: E402
    DEFAULT_TITLE, RESOURCE_ERRORS, AutotuneRecorder, ConfigLaunch,
    IntensityPruner, KernelLaunch, Work, arg_labels, compile_with_config,
    detach, format_equations, jit_function, last_launches, launch,
    print_equations, record, recording,
)

custom_stages.install()  # run the pass after every `ttir` stage

__all__ = [
    "LOAD_BYTES",
    "METADATA_KEY",
    "OP_COUNT",
    "STORE_BYTES",
    "PLUGIN_DIR",
    "PLUGIN_LIBRARY",
    "Intensity",
    "IntensityEquations",
    "IntensityListener",
    "intensity",
    "custom_stages",
    "enable",
    "utilities",
    # utilities.launches
    "RESOURCE_ERRORS",
    "AutotuneRecorder",
    "ConfigLaunch",
    "IntensityPruner",
    "KernelLaunch",
    "Work",
    "compile_with_config",
    "detach",
    "jit_function",
    "last_launches",
    "launch",
    "record",
    "recording",
    # utilities.report
    "DEFAULT_TITLE",
    "arg_labels",
    "format_equations",
    "print_equations",
]
