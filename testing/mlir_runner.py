"""
Shared helper for MLIR driver scripts.

Extensions that run list-style MLIR tests provide a driver script under their
``test/`` directory; they call :func:`run_passes` to run a pass pipeline. This
reads an MLIR file (path given as ``argv[1]``), runs the pipeline, and prints
the resulting MLIR to stdout -- ready to be piped into FileCheck.
"""

from __future__ import annotations

import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from triton._C.libtriton import ir

# A pass constructor mutates a ``pass_manager`` in place, e.g.
# ``passes.common.add_cse``.
AddPass = Callable[..., None]

# Matches the ``// -----`` separator used by ``--split-input-file`` tests: a
# line consisting solely of the separator (leading/trailing whitespace okay).
_SPLIT_RE = re.compile(r"^//\s*-----\s*$", re.MULTILINE)

# Separator emitted between transformed chunks so downstream FileCheck sees the
# same document structure it would from ``triton-opt --split-input-file``.
_SPLIT_OUT = "\n// -----\n"


def _run_chunk(mlir: str, adders: Sequence[AddPass]) -> str:
    """Parse a single MLIR chunk, run ``adders``, and return the printed IR."""
    # ``parse_mlir_module`` reads from a file path, so stage the chunk on disk.
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as tmp:
        tmp.write(mlir)
        path = tmp.name
    try:
        ctx = ir.context()
        ir.load_dialects(ctx)
        module = ir.parse_mlir_module(path, ctx)
        pm = ir.pass_manager(ctx)
        for add in adders:
            add(pm)
        pm.run(module, "mlir_runner")
        return module.str_nodebug()
    finally:
        Path(path).unlink(missing_ok=True)


def run_passes(adders: Sequence[AddPass],
               argv: Sequence[str] | None = None) -> None:
    """Run ``adders`` over the MLIR file named in ``argv`` and print the result.

    The input is split on ``// -----`` (mirroring ``--split-input-file``); each
    chunk is processed independently and the results are re-joined with the same
    separator so a single FileCheck invocation can validate the whole document.
    """
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2:
        prog = Path(argv[0]).name if argv else "driver"
        raise SystemExit(f"usage: {prog} <input.mlir>")

    text = Path(argv[1]).read_text()
    outputs = [
        _run_chunk(chunk, adders) for chunk in _SPLIT_RE.split(text)
        if chunk.strip()
    ]
    sys.stdout.write(_SPLIT_OUT.join(outputs))
    if outputs:
        sys.stdout.write("\n")
