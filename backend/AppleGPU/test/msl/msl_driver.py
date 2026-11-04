#!/usr/bin/env python3
"""Opt driver for the emit-msl pass.

Runs ``emit-msl`` over a TTGIR input and prints the Metal Shading Language it
produces. The pass leaves the module alone and writes its output to a path
given as a pass argument, so what this prints is that file.
"""

import sys
import tempfile
from pathlib import Path

import triton_apple_backend  # noqa: F401  registers the plugin on import
from triton._C.libtriton import ir, passes


def emit(path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".metal", delete=False) as out:
        msl_path = out.name
    try:
        ctx = ir.context()
        ir.load_dialects(ctx)
        module = ir.parse_mlir_module(path, ctx)
        module.context = ctx
        pm = ir.pass_manager(ctx)
        pm.enable_debug()
        passes.plugin.add_emit_msl(pm, [msl_path])
        pm.run(module, "msl_driver")
        return Path(msl_path).read_text()
    finally:
        Path(msl_path).unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <input.mlir>")
    sys.stdout.write(emit(sys.argv[1]))
