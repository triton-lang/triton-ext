"""Human-readable rendering of the pass equations.

The equations are written over ``args[i]`` (``tt.func`` argument indices).
:func:`format_equations` renders them per kernel function with the indices
resolved to parameter names (``a_desc.shape[0]``, ``stride_am``, ...) and the
``constexpr`` / specialized parameters folded into the kernel listed in the
header, so that e.g. the dependence of the per-program bytes on the tile shape
is visible in the equations themselves::

    matmul_kernel [BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, ...]
        a_ptr  load_bytes: ((args[5] + 63) / 64) * 16384
        b_ptr  load_bytes: ((args[5] + 63) / 64) * 32768
        c_ptr store_bytes: 65536
        c_ptr    op_count: ((args[5] + 63) / 64) * 4194304
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

from ..arithmetic_intensity import (LOAD_BYTES, OP_COUNT, STORE_BYTES,
                                    ArithmeticIntensityEquations)
from .launches import KernelLaunch

__all__ = [
    "DEFAULT_TITLE",
    "arg_labels",
    "format_equations",
    "print_equations",
]

#: Header printed by :func:`print_equations`.
DEFAULT_TITLE = ("Arithmetic-intensity equations per kernel function "
                 "(per program):")


def arg_labels(equations: ArithmeticIntensityEquations, *args: Any,
               **kwargs: Any) -> Dict[int, str]:
    """``{tt.func arg index: parameter label}`` of the entry function.

    Launch arguments are optional; they only refine the labels of tuple
    parameters whose compile-time type is not recorded.
    """
    _, names = equations.ttir_args(equations.bind_launch_args(*args, **kwargs))
    return names


def format_equations(equations: ArithmeticIntensityEquations,
                     *args: Any,
                     indent: str = "    ",
                     **kwargs: Any) -> str:
    """Render the entry function's equations, one line per argument and kind.

    The first line names the kernel function and lists its
    :attr:`~ArithmeticIntensityEquations.folded_args`; ``args`` / ``kwargs``
    are the launch arguments, if known (see :func:`arg_labels`).
    """
    labels = arg_labels(equations, *args, **kwargs)
    folded = ", ".join(f"{k}={v}" for k, v in equations.folded_args.items())
    lines = [f"{equations.name} [{folded}]" if folded else equations.name]
    rows: List[Tuple[str, str, str]] = []
    for index, per_arg in sorted(equations.entry.items()):
        label = labels.get(index, f"args[{index}]")
        for kind in (LOAD_BYTES, STORE_BYTES, OP_COUNT):
            if kind in per_arg:
                rows.append((label, kind, per_arg[kind]))
    width = max((len(label) for label, _, _ in rows), default=0)
    kind_width = max((len(kind) for _, kind, _ in rows), default=0)
    for label, kind, equation in rows:
        lines.append(
            f"{indent}{label:>{width}} {kind:>{kind_width}}: {equation}")
    return "\n".join(lines)


def _as_equations(
        item: Any) -> Tuple[ArithmeticIntensityEquations, tuple, dict]:
    """``(equations, launch args, launch kwargs)`` of a printable item."""
    if isinstance(item, KernelLaunch):
        return item.equations(), tuple(item.args), dict(item.kwargs)
    if isinstance(item, ArithmeticIntensityEquations):
        return item, (), {}
    return ArithmeticIntensityEquations.from_kernel(item), (), {}


def print_equations(items: Any,
                    *,
                    title: Optional[str] = DEFAULT_TITLE,
                    indent: str = "    ",
                    file: Optional[TextIO] = None) -> None:
    """Print the equations of one or more kernels.

    ``items`` is a :class:`~.launches.KernelLaunch`, an
    :class:`ArithmeticIntensityEquations`, a ``CompiledKernel``, or an
    iterable of those (e.g. ``tai.last_launches.values()`` or the launches of
    a :func:`~.launches.recording`). When several items belong to the same
    kernel function only the last one is printed, in order of first
    appearance. ``title=None`` omits the header.
    """
    if isinstance(
            items,
        (KernelLaunch,
         ArithmeticIntensityEquations)) or not isinstance(items, Iterable):
        items = [items]
    last: Dict[str, Tuple[ArithmeticIntensityEquations, tuple, dict]] = {}
    for item in items:
        equations, args, kwargs = _as_equations(item)
        last[equations.name] = (equations, args, kwargs)
    out = sys.stdout if file is None else file
    if title:
        print(title, file=out)
    # Under a title, each kernel's block is indented by one more level.
    block_indent = indent if title else ""
    for equations, args, kwargs in last.values():
        text = format_equations(equations, *args, indent=indent, **kwargs)
        for line in text.splitlines():
            print(block_indent + line, file=out)
