"""Pytest tests for the arithmetic-intensity plugin pass.

The tests load Triton's Python bindings, register the
``libarithmetic_intensity.so`` plugin via ``TRITON_PLUGIN_PATHS``, parse
a small MLIR module, run the pass through ``ir.pass_manager``, and
assert that the expected ``tt.bandwidth`` and ``tt.compute`` attributes
have been attached to the function's arguments.

Plugin discovery (in order of priority):
  * ``ARITHMETIC_INTENSITY_PLUGIN`` - explicit path to
    ``libarithmetic_intensity.so``
  * ``<repo>/build*/lib/libarithmetic_intensity.so`` (most recently
    built)

If the plugin cannot be located or Triton's bindings cannot be imported
the whole module is skipped, so it is safe to collect in environments
that haven't built the pass.
"""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
import textwrap

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _resolve_plugin() -> pathlib.Path | None:
    if env := os.environ.get("ARITHMETIC_INTENSITY_PLUGIN"):
        p = pathlib.Path(env)
        return p if p.exists() else None
    for build_dir in ("build", "build-release", "build-debug"):
        cand = PROJECT_ROOT / build_dir / "lib" / "libarithmetic_intensity.so"
        if cand.exists():
            return cand
    return None


_plugin = _resolve_plugin()
_skip_reason: str | None = None
if _plugin is None:
    _skip_reason = ("arithmetic-intensity plugin not found; build the pass or "
                    "set ARITHMETIC_INTENSITY_PLUGIN to override")
else:
    # Plugins are enumerated by libtriton at import time, so we must set
    # TRITON_PLUGIN_PATHS *before* importing triton.
    existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    if str(_plugin) not in existing.split(":"):
        os.environ["TRITON_PLUGIN_PATHS"] = (f"{_plugin}:{existing}"
                                             if existing else str(_plugin))

if _skip_reason is None:
    try:
        from triton._C.libtriton import ir, passes  # noqa: E402
    except ImportError as exc:
        _skip_reason = f"triton._C.libtriton import failed: {exc}"
    else:
        if not hasattr(passes.plugin, "arithmetic_intensity"):
            _skip_reason = ("arithmetic_intensity pass not registered on "
                            "passes.plugin (plugin failed to load?)")

pytestmark = pytest.mark.skipif(_skip_reason is not None,
                                reason=_skip_reason or "")


@pytest.fixture(scope="module")
def run_pass():
    """Return a callable that parses MLIR text, runs the pass, returns IR."""

    ctx = ir.context()
    ir.load_dialects(ctx)

    def _run(mlir: str) -> str:
        # parse_mlir_module only accepts a file path, so spill to a tmp file.
        with tempfile.NamedTemporaryFile(mode="w",
                                         suffix=".mlir",
                                         delete=False) as f:
            f.write(textwrap.dedent(mlir))
            path = f.name
        try:
            mod = ir.parse_mlir_module(path, ctx)
        finally:
            os.unlink(path)

        pm = ir.pass_manager(ctx)
        passes.plugin.arithmetic_intensity(pm)
        pm.run(mod, "arithmetic_intensity_test")
        return mod.str_nodebug()

    return _run


_ATTR_KV = re.compile(r'([\w.]+)\s*=\s*"((?:[^"\\]|\\.)*)"')


def _parse_arg_attrs(ir_text: str,
                     func_name: str) -> dict[int, dict[str, str]]:
    """Return ``{arg_index: {attr_name: attr_value}}`` for ``func_name``.

    Parses the textual MLIR with bracket balancing rather than a single
    regex so that nested ``<...>`` in types don't confuse the scan.
    """
    head = re.search(
        rf'tt\.func\s+(?:public\s+)?@{re.escape(func_name)}\s*\(',
        ir_text,
    )
    assert head is not None, f"function @{func_name} not found in:\n{ir_text}"
    i = head.end()
    depth = 1  # we are immediately after the opening '(' of the arg list
    args_text_start = i
    while i < len(ir_text) and depth > 0:
        c = ir_text[i]
        if c == '(' or c == '<' or c == '[' or c == '{':
            depth += 1
        elif c == ')' or c == '>' or c == ']' or c == '}':
            depth -= 1
        i += 1
    assert depth == 0, "unbalanced parens parsing function signature"
    sig = ir_text[args_text_start:i - 1]

    pieces, depth, start = [], 0, 0
    for j, ch in enumerate(sig):
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        elif ch == ',' and depth == 0:
            pieces.append(sig[start:j])
            start = j + 1
    pieces.append(sig[start:])

    out: dict[int, dict[str, str]] = {}
    for piece in pieces:
        m = re.match(r'\s*%arg(\d+)\s*:', piece)
        if not m:
            continue
        idx = int(m.group(1))
        brace = piece.find('{')
        if brace == -1:
            out[idx] = {}
            continue
        end = piece.rfind('}')
        attrs_text = piece[brace + 1:end] if end > brace else ""
        out[idx] = {k: v for k, v in _ATTR_KV.findall(attrs_text)}
    return out


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_static_loop_emits_constant_bandwidth(run_pass) -> None:
    """A loop with a constant trip count folds to a literal byte count.

    This case keeps the ``tensor<256x!tt.ptr<f32>>`` parameter shape (and
    threads the pointer tensor through ``scf.for`` ``iter_args``) so we
    keep regression coverage of ``findPointerParam`` walking back through
    a loop-carried tensor-of-pointers to its function-arg base.
    """
    ir_text = run_pass("""
        tt.func @const_loop(%out: tensor<256x!tt.ptr<f32>>,
                            %in: tensor<256x!tt.ptr<f32>>) {
          %c0 = arith.constant 0 : i32
          %c1 = arith.constant 1 : i32
          %c4 = arith.constant 4 : i32
          %off = tt.splat %c1 : i32 -> tensor<256xi32>
          %r:2 = scf.for %i = %c0 to %c4 step %c1
              iter_args(%pin = %in, %pout = %out)
              -> (tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>) : i32 {
            %v = tt.load %pin : tensor<256x!tt.ptr<f32>>
            tt.store %pout, %v : tensor<256x!tt.ptr<f32>>
            %nin = tt.addptr %pin, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
            %nout = tt.addptr %pout, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
            scf.yield %nin, %nout : tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
          }
          tt.return
        }
    """)
    attrs = _parse_arg_attrs(ir_text, "const_loop")
    # 4 iterations * 256 elements * 4 bytes/elem = 4096 bytes per pointer.
    assert attrs[0].get("tt.bandwidth") == "4096", attrs
    assert attrs[1].get("tt.bandwidth") == "4096", attrs


def test_param_driven_loop_emits_symbolic_bandwidth(run_pass) -> None:
    """The dynamic loop range becomes a symbol on the function arg.

    Uses the typical kernel idiom: scalar ``!tt.ptr<f32>`` parameters
    with the address tensor built from ``tt.make_range`` + ``tt.splat``
    + ``tt.addptr``.
    """
    ir_text = run_pass("""
        tt.func @param_loop(%out: !tt.ptr<f32>,
                            %in: !tt.ptr<f32>,
                            %N: i32) {
          %c0 = arith.constant 0 : i32
          %c1 = arith.constant 1 : i32
          %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
          %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
          %outb = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
          %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
          %outp = tt.addptr %outb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
          scf.for %i = %c0 to %N step %c1 : i32 {
            %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
            tt.store %outp, %v : tensor<256x!tt.ptr<f32>>
          }
          tt.return
        }
    """)
    attrs = _parse_arg_attrs(ir_text, "param_loop")
    # Per-iter bytes = 256 * 4 = 1024, trip count = args[2].
    for idx in (0, 1):
        bw = attrs[idx].get("tt.bandwidth", "")
        assert "args[2]" in bw, (
            f"expected args[2] as trip count in arg{idx} bandwidth; got {bw!r}"
        )
        assert "1024" in bw, (
            f"expected per-iter bytes 1024 in arg{idx} bandwidth; got {bw!r}")


def test_iter_arg_inner_bound_is_symbolic(run_pass) -> None:
    """Inner loop bound is a loop-carried iter_arg of the outer loop.

    The pass should trace the iter_arg back to its init value and emit a
    bandwidth equation in both function parameters that drive the loops.
    """
    ir_text = run_pass("""
        tt.func @nested_iter_arg(%in: !tt.ptr<f32>, %M: i32, %N: i32) {
          %c0 = arith.constant 0 : i32
          %c1 = arith.constant 1 : i32
          %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
          %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
          %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
          %r = scf.for %i = %c0 to %M step %c1
              iter_args(%bound = %N) -> (i32) : i32 {
            scf.for %j = %c0 to %bound step %c1 : i32 {
              %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
            }
            scf.yield %bound : i32
          }
          tt.return
        }
    """)
    attrs = _parse_arg_attrs(ir_text, "nested_iter_arg")
    bw = attrs[0].get("tt.bandwidth", "")
    assert "args[1]" in bw, f"expected outer trip count args[1] in {bw!r}"
    assert "args[2]" in bw, (
        f"expected inner trip count args[2] (from iter_arg init) in {bw!r}")
    assert "1024" in bw, f"expected per-iter bytes 1024 in {bw!r}"


def test_iv_inner_bound_falls_back_to_upper(run_pass) -> None:
    """Inner loop bound IS an outer induction variable.

    The pass substitutes the IV with the outer loop's upper bound,
    yielding a quadratic upper-bound estimate.
    """
    ir_text = run_pass("""
        tt.func @triangular(%in: !tt.ptr<f32>, %M: i32) {
          %c0 = arith.constant 0 : i32
          %c1 = arith.constant 1 : i32
          %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
          %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
          %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
          scf.for %i = %c0 to %M step %c1 : i32 {
            scf.for %j = %c0 to %i step %c1 : i32 {
              %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
            }
          }
          tt.return
        }
    """)
    attrs = _parse_arg_attrs(ir_text, "triangular")
    bw = attrs[0].get("tt.bandwidth", "")
    assert bw.count("args[1]") == 2, (
        f"expected args[1] to appear twice (outer trip * IV-substituted inner "
        f"trip); got {bw!r}")
    assert "1024" in bw, f"expected per-iter bytes 1024 in {bw!r}"


def test_store_emits_compute_metric(run_pass) -> None:
    """Stored values report a compute metric in addition to bandwidth."""
    ir_text = run_pass("""
        tt.func @add_kernel(%out: !tt.ptr<f32>,
                            %a: !tt.ptr<f32>,
                            %b: !tt.ptr<f32>) {
          %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
          %ab = tt.splat %a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
          %bb = tt.splat %b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
          %ob = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
          %ap = tt.addptr %ab, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
          %bp = tt.addptr %bb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
          %op = tt.addptr %ob, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
          %va = tt.load %ap : tensor<256x!tt.ptr<f32>>
          %vb = tt.load %bp : tensor<256x!tt.ptr<f32>>
          %s = arith.addf %va, %vb : tensor<256xf32>
          tt.store %op, %s : tensor<256x!tt.ptr<f32>>
          tt.return
        }
    """)
    attrs = _parse_arg_attrs(ir_text, "add_kernel")
    # Loaded arrays get bandwidth; the stored array additionally accumulates
    # the chain's compute (one elementwise addf over 256 elements = 256 ops).
    for idx in (0, 1, 2):
        assert "1024" in attrs[idx].get("tt.bandwidth", ""), attrs
    compute = attrs[0].get("tt.compute", "")
    assert "256" in compute, f"expected 256 ops in store-side compute; got {compute!r}"


def test_no_attrs_when_no_loads_or_stores(run_pass) -> None:
    """A function with no memory traffic should not gain any attributes."""
    ir_text = run_pass("""
        tt.func @empty(%p: !tt.ptr<f32>) {
          tt.return
        }
    """)
    attrs = _parse_arg_attrs(ir_text, "empty")
    assert attrs.get(0, {}).get("tt.bandwidth") is None
    assert attrs.get(0, {}).get("tt.compute") is None


def test_matmul_scalar_ptr_params(run_pass) -> None:
    """Block-matmul kernel with scalar ``!tt.ptr<f16>`` parameters.

    Builds 64x64 address tensors from ``tt.make_range`` + ``tt.expand_dims``
    + ``tt.broadcast`` + ``tt.splat`` + ``tt.addptr`` (the usual Triton
    address arithmetic), iterates the K loop ``K/64`` times, accumulates
    a ``tt.dot``, then truncs to f16 and stores once. The pass should:

    * attribute ``A``/``B`` bandwidth = ``8192 * (K/64)`` (per-iter
      tensor<64x64xf16> = 8192 bytes, times the loop trip count).
    * attribute ``C`` bandwidth = ``8192`` (single out-of-loop store).
    * attribute ``C`` compute = dot FLOPs scaled by trip count
      (``2*M*N*K`` per dot block) plus the epilogue truncf
      (``64*64=4096`` ops).
    """
    ir_text = run_pass("""
        tt.func @matmul_ptr(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f16>,
                            %M: i32, %N: i32, %K: i32,
                            %sam: i32, %sak: i32,
                            %sbk: i32, %sbn: i32,
                            %scm: i32, %scn: i32) {
          %c0 = arith.constant 0 : i32
          %c64 = arith.constant 64 : i32
          %zero = arith.constant 0.0 : f32
          %acc0 = tt.splat %zero : f32 -> tensor<64x64xf32>
          %rm = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
          %rn = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
          %rk = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
          %rm_col = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
          %rn_row = tt.expand_dims %rn {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
          %rk_row = tt.expand_dims %rk {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
          %rk_col = tt.expand_dims %rk {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
          %sam_t = tt.splat %sam : i32 -> tensor<64x1xi32>
          %sak_t = tt.splat %sak : i32 -> tensor<1x64xi32>
          %a_m = arith.muli %rm_col, %sam_t : tensor<64x1xi32>
          %a_k = arith.muli %rk_row, %sak_t : tensor<1x64xi32>
          %a_m_b = tt.broadcast %a_m : tensor<64x1xi32> -> tensor<64x64xi32>
          %a_k_b = tt.broadcast %a_k : tensor<1x64xi32> -> tensor<64x64xi32>
          %a_off = arith.addi %a_m_b, %a_k_b : tensor<64x64xi32>
          %ab = tt.splat %A : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
          %ap = tt.addptr %ab, %a_off : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
          %sbk_t = tt.splat %sbk : i32 -> tensor<64x1xi32>
          %sbn_t = tt.splat %sbn : i32 -> tensor<1x64xi32>
          %b_k = arith.muli %rk_col, %sbk_t : tensor<64x1xi32>
          %b_n = arith.muli %rn_row, %sbn_t : tensor<1x64xi32>
          %b_k_b = tt.broadcast %b_k : tensor<64x1xi32> -> tensor<64x64xi32>
          %b_n_b = tt.broadcast %b_n : tensor<1x64xi32> -> tensor<64x64xi32>
          %b_off = arith.addi %b_k_b, %b_n_b : tensor<64x64xi32>
          %bb = tt.splat %B : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
          %bp = tt.addptr %bb, %b_off : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
          %final = scf.for %k = %c0 to %K step %c64
              iter_args(%acc = %acc0) -> (tensor<64x64xf32>) : i32 {
            %a = tt.load %ap : tensor<64x64x!tt.ptr<f16>>
            %b = tt.load %bp : tensor<64x64x!tt.ptr<f16>>
            %d = tt.dot %a, %b, %acc :
                tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
            scf.yield %d : tensor<64x64xf32>
          }
          %scm_t = tt.splat %scm : i32 -> tensor<64x1xi32>
          %scn_t = tt.splat %scn : i32 -> tensor<1x64xi32>
          %c_m = arith.muli %rm_col, %scm_t : tensor<64x1xi32>
          %c_n = arith.muli %rn_row, %scn_t : tensor<1x64xi32>
          %c_m_b = tt.broadcast %c_m : tensor<64x1xi32> -> tensor<64x64xi32>
          %c_n_b = tt.broadcast %c_n : tensor<1x64xi32> -> tensor<64x64xi32>
          %c_off = arith.addi %c_m_b, %c_n_b : tensor<64x64xi32>
          %cb = tt.splat %C : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
          %cp = tt.addptr %cb, %c_off : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
          %final_f16 = arith.truncf %final : tensor<64x64xf32> to tensor<64x64xf16>
          tt.store %cp, %final_f16 : tensor<64x64x!tt.ptr<f16>>
          tt.return
        }
    """)
    attrs = _parse_arg_attrs(ir_text, "matmul_ptr")
    # 64x64xf16 = 8192 bytes per load/store.
    a_bw = attrs[0].get("tt.bandwidth", "")
    b_bw = attrs[1].get("tt.bandwidth", "")
    c_bw = attrs[2].get("tt.bandwidth", "")
    for name, bw in (("A", a_bw), ("B", b_bw)):
        assert "args[5]" in bw, (
            f"expected K (args[5]) trip count in {name} bandwidth; got {bw!r}")
        assert "8192" in bw, (
            f"expected per-iter bytes 8192 in {name} bandwidth; got {bw!r}")
    assert c_bw == "8192", (
        f"expected exactly one 64x64xf16 store on C (8192 bytes); got {c_bw!r}"
    )
    # tt.dot M*N*K*2 with M=N=K=64 -> 524288 flops/iter, times K/64, plus the
    # epilogue 64x64 truncf (4096) outside the loop.
    c_compute = attrs[2].get("tt.compute", "")
    assert "args[5]" in c_compute, (
        f"expected K (args[5]) in C compute; got {c_compute!r}")
    assert "524288" in c_compute, (
        f"expected dot block flops 524288 in C compute; got {c_compute!r}")
    assert "4096" in c_compute, (
        f"expected epilogue truncf 4096 in C compute; got {c_compute!r}")


def test_matmul_tensordesc_params(run_pass) -> None:
    """Block-matmul kernel with ``!tt.tensordesc`` parameters.

    The kernel signature still carries explicit shape (``M``, ``N``, ``K``)
    and stride parameters next to the descriptors, mirroring kernels that
    take both forms.  The compute-density pass treats
    ``tt.descriptor_load`` / ``tt.descriptor_store`` like their pointer
    counterparts, attributing bandwidth/compute to the descriptor arg
    (``isPointerLikeFuncArgType`` accepts ``!tt.tensordesc<>``).
    """
    ir_text = run_pass("""
        tt.func @matmul_tensordesc(%A: !tt.tensordesc<64x64xf16>,
                                   %B: !tt.tensordesc<64x64xf16>,
                                   %C: !tt.tensordesc<64x64xf16>,
                                   %M: i32, %N: i32, %K: i32,
                                   %sam: i32, %sak: i32,
                                   %sbk: i32, %sbn: i32,
                                   %scm: i32, %scn: i32) {
          %c0 = arith.constant 0 : i32
          %c64 = arith.constant 64 : i32
          %zero = arith.constant 0.0 : f32
          %acc0 = tt.splat %zero : f32 -> tensor<64x64xf32>
          %final = scf.for %k = %c0 to %K step %c64
              iter_args(%acc = %acc0) -> (tensor<64x64xf32>) : i32 {
            %a = tt.descriptor_load %A[%c0, %k] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
            %b = tt.descriptor_load %B[%k, %c0] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
            %d = tt.dot %a, %b, %acc :
                tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
            scf.yield %d : tensor<64x64xf32>
          }
          %final_f16 = arith.truncf %final : tensor<64x64xf32> to tensor<64x64xf16>
          tt.descriptor_store %C[%c0, %c0], %final_f16 :
              !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
          tt.return
        }
    """)
    attrs = _parse_arg_attrs(ir_text, "matmul_tensordesc")
    a_bw = attrs[0].get("tt.bandwidth", "")
    b_bw = attrs[1].get("tt.bandwidth", "")
    c_bw = attrs[2].get("tt.bandwidth", "")
    for name, bw in (("A", a_bw), ("B", b_bw)):
        assert "args[5]" in bw, (
            f"expected K (args[5]) trip count in {name} bandwidth; got {bw!r}")
        assert "8192" in bw, (
            f"expected per-iter bytes 8192 in {name} bandwidth; got {bw!r}")
    assert c_bw == "8192", (
        f"expected exactly one 64x64xf16 store on C (8192 bytes); got {c_bw!r}"
    )
    c_compute = attrs[2].get("tt.compute", "")
    assert "args[5]" in c_compute, (
        f"expected K (args[5]) in C compute; got {c_compute!r}")
    assert "524288" in c_compute, (
        f"expected dot block flops 524288 in C compute; got {c_compute!r}")
    assert "4096" in c_compute, (
        f"expected epilogue truncf 4096 in C compute; got {c_compute!r}")
