"""Helpers for benchmarking and reporting with the intensity pass.

Everything here is re-exported by :mod:`triton_intensity`.

* :mod:`.launches` -- record kernel launches (:func:`launch`, :func:`record`,
  :func:`recording`, :class:`KernelLaunch`), sum the work of several launches
  (:class:`Work`) and compile one ``triton.Config`` of an autotuned kernel
  without autotuning (:func:`compile_with_config`, :class:`ConfigLaunch`);
  hook the autotuner to capture every configuration it benchmarks
  (:class:`AutotuneRecorder`) or to prune low-intensity configurations
  before it benchmarks them (:class:`IntensityPruner`).
* :mod:`.report` -- render the equations with ``args[i]`` resolved to
  parameter names (:func:`format_equations`, :func:`print_equations`).
"""

from __future__ import annotations

from . import launches, report
from .launches import (RESOURCE_ERRORS, AutotuneRecorder, ConfigLaunch,
                       IntensityPruner, KernelLaunch, Work,
                       compile_with_config, detach, jit_function,
                       last_launches, launch, record, recording)
from .report import (DEFAULT_TITLE, arg_labels, format_equations,
                     print_equations)

__all__ = [
    # launches
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
    # report
    "DEFAULT_TITLE",
    "arg_labels",
    "format_equations",
    "print_equations",
    # submodules
    "launches",
    "report",
]
