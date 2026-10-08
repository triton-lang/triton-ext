"""Evaluate an equation whose trip count is a remainder.

The pass prints modulo as ``%``. This checks that form against the evaluator,
which treats ``%`` as Python's modulo.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing"))

import triton_intensity as tint  # noqa: E402
from mlir_runner import _run_chunk  # noqa: E402
from triton._C.libtriton import passes  # noqa: E402

# 256 f32 elements = 1024 bytes per iteration. The loop runs ``N % M`` times.
# The modulus is a symbol so `simplifyAffineExpr` leaves the `%` in place
# (a constant modulus is rewritten as a product of `floordiv`).
_MOD_TRIP = """
tt.func @mod_trip(%in: !tt.ptr<f32>, %N: i32, %M: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %bound = arith.remsi %N, %M : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}
"""

_LOAD_BYTES = re.compile(r'tint\.load_bytes = "([^"]*)"')


def test_modulo_equation_evaluates():
    printed = _run_chunk(_MOD_TRIP, [passes.plugin.add_intensity])
    match = _LOAD_BYTES.search(printed)
    assert match, printed
    equation = match.group(1)
    assert equation == "max((args[1] % args[2]) * 1024, 0)"

    equations = tint.IntensityEquations(
        name="mod_trip",
        functions={"mod_trip": {
            0: {
                tint.LOAD_BYTES: equation
            }
        }},
    )
    # 10 % 8 = 2 iterations, 1024 bytes each.
    assert equations.evaluate(1, arg_values={1: 10, 2: 8}).load_bytes == 2048
    # An exact multiple of the modulus runs zero times.
    assert equations.evaluate(1, arg_values={1: 16, 2: 8}).load_bytes == 0
