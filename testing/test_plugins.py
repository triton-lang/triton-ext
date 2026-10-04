"""Plugin integration tests.

Auto-discovers every extension declared by a ``pyproject.toml`` carrying a
``[tool.triton-ext]`` stanza and exercises it via a direct import.  Each
enabled extension is imported using ``import <package>``, which loads and
registers the compiled plugin library. No ``TRITON_PLUGIN_PATHS``,
``PYTHONPATH``, or ``LD_LIBRARY_PATH`` overrides are needed; the extensions
must be installed beforehand (e.g. with ``make build && make install``).

Tests:
  - test_plugins_discovered                -- guard: at least one plugin exists.
  - test_plugin_loads[<name>]              -- ``import <package>`` succeeds.
  - test_plugin_compiles_kernel[<name>]    -- JIT-decorate and lower a basic
                                             kernel through the plugin pipeline
                                             (in a fresh interpreter, so plugins
                                             do not affect each other).
  - test_utlx_registers_tlx_dsl           -- utlx registers
                                             ``triton.language.extra.tlx``.
  - test_arithmetic_intensity_*           -- the arithmetic-intensity pass
                                             hooks the ``ttir`` stage, its
                                             listener gathers per-kernel work
                                             and its utilities record launches,
                                             compile single configs, prune
                                             autotune configs by intensity and
                                             print equations.

Adding a new plugin: drop a ``pyproject.toml`` with ``[tool.triton-ext]``;
both parametrized tests pick it up automatically.  To exempt a plugin from a
parametrized test mark it at parametrize time with
``pytest.param(..., marks=pytest.mark.skip(...))``.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "ci"))
import extension  # noqa: E402  (ci/ is added to sys.path above)

# Map from extension *name* (``Manifest.name``: the ``pyproject.toml`` project
# name without its ``triton-`` prefix, normalized to underscores) to the
# importable Python package name.
_PACKAGE_MAP: dict[str, str] = {
    "arithmetic_intensity": "triton_arithmetic_intensity",
    "loop_split": "triton_loop_split",
    "example": "triton_example",
    "utlx": "utlx_plugin",
    "apple-backend": "triton_apple_backend",
}


def _package_name(ext_name: str) -> str:
    """Return the importable Python package name for an extension."""
    key = ext_name.replace("-", "_").replace(" ", "_")
    return _PACKAGE_MAP.get(key, key)


def _discover_plugins() -> list[ParameterSet]:
    plugins: list[ParameterSet] = []
    for cfg in extension.discover():
        if cfg.enabled:
            plugins.append(pytest.param(cfg.name, id=cfg.name))
    plugins.sort(key=lambda p: p.id)
    return plugins


PLUGINS = _discover_plugins()

# ---------------------------------------------------------------------------
# Generic per-plugin tests (auto-discovered)
# ---------------------------------------------------------------------------


def test_plugins_discovered() -> None:
    """Guard against silently testing nothing if discovery breaks."""
    assert PLUGINS, f"No triton-ext extensions found under {REPO_ROOT}"


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_loads(name: str) -> None:
    """Smoke: ``import <package>`` succeeds with the plugin registered."""
    pkg = _package_name(name)
    try:
        importlib.import_module(pkg)
    except ImportError as exc:
        pytest.skip(
            f"Package {pkg!r} not installed (run `make build && make install`): {exc}"
        )


# example dialect is scaffolding-only — its Dialect::initialize() doesn't
# register StringAttr, so kernel compile aborts with an LLVM storage-uniquer
# error.  Tag it as skip at parametrize time.
_COMPILE_PLUGINS = [
    pytest.param(p.values[0],
                 marks=pytest.mark.skip(reason="scaffolding-only dialect"),
                 id=p.id) if p.id == "example" else p for p in PLUGINS
]

# Run in a fresh interpreter: importing a plugin installs global compilation
# hooks (e.g. ``knobs.runtime.add_stages_inspection_hook``), and a plugin
# imported earlier in this process must not leak into another plugin's test.
_COMPILE_SCRIPT = """
import importlib
importlib.import_module({pkg!r})

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, y_ptr, n: tl.constexpr):
    offs = tl.arange(0, n)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, x)


n = 32
x = torch.ones(n, device="cuda")
y = torch.zeros(n, device="cuda")
_kernel[(1, )](x, y, n)
torch.cuda.synchronize()
assert torch.allclose(x, y), "kernel output mismatch"
"""


@pytest.mark.parametrize("name", _COMPILE_PLUGINS)
def test_plugin_compiles_kernel(name: str, tmp_path: Path) -> None:
    """User scenario: with the plugin imported, JIT-decorate and lower a basic kernel."""
    pkg = _package_name(name)
    if importlib.util.find_spec(pkg) is None:
        pytest.skip(
            f"Package {pkg!r} not installed (run `make build && make install`)"
        )

    # Compiling the kernel (lowering to PTX/HSACO) requires a GPU device.
    # Skip gracefully if no GPU is available rather than failing.
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed; skipping kernel compile test")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available; skipping kernel compile test")

    # `@triton.jit` reads the kernel source from disk, so run from a file.
    script = tmp_path / f"compile_with_{pkg}.py"
    script.write_text(_COMPILE_SCRIPT.format(pkg=pkg))
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"kernel compile with {pkg!r} imported failed "
        f"(exit {result.returncode}):\n{result.stderr}")


# ---------------------------------------------------------------------------
# Plugin-specific tests
# ---------------------------------------------------------------------------


def test_utlx_registers_tlx_dsl() -> None:
    """utlx registers ``triton.language.extra.tlx`` with local_alloc/view/store/load.

    The namespace is set up by ``extensions/utlx/python/utlx_plugin/__init__.py``
    when the package is imported.
    """
    try:
        import utlx_plugin  # noqa: F401
    except ImportError:
        pytest.skip(
            "utlx_plugin not installed (run `make build && make install`)")

    import triton.language.extra as extra
    assert hasattr(extra, "tlx"), "triton.language.extra.tlx not registered"

    import triton.language.extra.tlx as tlx
    for attr in ("local_alloc", "local_view", "local_store", "local_load"):
        assert hasattr(tlx,
                       attr), f"tlx.{attr} missing after import utlx_plugin"


def _import_arithmetic_intensity(monkeypatch=None):
    """Import the plugin; with ``monkeypatch``, also isolate its pipeline hook.

    ``knobs.runtime.add_stages_inspection_hook`` is a single global slot that
    every plugin imported in this process competes for, so put ours on top
    and (for the duration of the test) detach it from any other plugin's hook.
    """
    try:
        import triton_arithmetic_intensity as tai
    except ImportError:
        pytest.skip("triton_arithmetic_intensity not installed "
                    "(run `make build && make install`)")
    if monkeypatch is not None:
        tai.custom_stages.install()
        monkeypatch.setattr(tai.custom_stages, "_previous", None)
        monkeypatch.setattr(tai.custom_stages, "_cache_key", None)
    return tai


def test_arithmetic_intensity_installs_ttir_hook(monkeypatch) -> None:
    """Importing the package appends the pass to every ``ttir`` stage."""
    tai = _import_arithmetic_intensity(monkeypatch)
    from triton import knobs

    hook = knobs.runtime.add_stages_inspection_hook
    assert hook is tai.custom_stages.inspect_stages_hook

    # Without arguments the hook contributes a (key, hash) to the cache key.
    key, digest = hook()
    assert key and len(digest) == 64

    # With a stages table it wraps `ttir` (and only `ttir`).
    original = object()
    stages = {"ttir": original, "ttgir": "unchanged"}
    hook(None, stages, object(), object(), 120)
    assert stages["ttir"] is not original
    assert stages["ttgir"] == "unchanged"
    gluon = {"glir": "g", "ttgir": "t"}
    hook(None, gluon, object(), object(), 120)
    assert gluon == {"glir": "g", "ttgir": "t"}


def test_arithmetic_intensity_parses_arg_attrs() -> None:
    """Equations are recovered from the printed ``tt.func`` signature."""
    tai = _import_arithmetic_intensity()
    parse = tai.custom_stages.parse_function_equations

    sig = (
        'tt.func public @"k(x, y"(%arg0: !tt.ptr<f32, 3> '
        '{tai.load_bytes = "(args[5] / 64) * 8192", tt.divisibility = 16 : i32}, '
        '%arg1: !tt.tensordesc<tensor<64x64xf16>>, '
        '%arg2: tensor<64x!tt.ptr<f32>> {tai.store_bytes = "8192", '
        'tai.op_count = "a \\"q\\" , eq"}, '
        '%arg3: i32 {tai.op_count = "num_programs[0] % 3"}) '
        'attributes {noinline = false, tai.load_bytes = "not an arg"} {\n'
        '  tt.return\n}\n')
    assert parse(sig) == {
        0: {
            "load_bytes": "(args[5] / 64) * 8192"
        },
        2: {
            "store_bytes": "8192",
            "op_count": 'a "q" , eq'
        },
        3: {
            "op_count": "num_programs[0] % 3"
        },
    }
    assert parse("tt.func @empty() {\n  tt.return\n}\n") == {}


def test_arithmetic_intensity_listener_gathers_per_kernel(monkeypatch) -> None:
    """User scenario: launch a kernel, read its work from the listener."""
    tai = _import_arithmetic_intensity(monkeypatch)
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed; skipping kernel compile test")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available; skipping kernel compile test")

    import triton
    import triton.language as tl

    @triton.jit
    def _axpy(x_ptr, y_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, alpha * x + y, mask=mask)

    listener = tai.ArithmeticIntensityListener().install()
    try:
        n, block = 4096, 1024
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, block), )
        compiled = _axpy[grid](x, y, out, n, 2.0, BLOCK=block)
        torch.cuda.synchronize()
        assert torch.allclose(out, 2.0 * x + y)

        # Equations were recorded in the kernel metadata by the ttir stage...
        assert tai.METADATA_KEY in compiled.metadata._asdict()
        # ...and gathered per kernel function by the listener.
        assert compiled.name in listener
        equations = listener[compiled]
        assert listener.get(_axpy) is equations
        assert equations.entry[0] == {"load_bytes": str(block * 4)}
        assert equations.load_bytes(0) == str(block * 4)
        assert equations.store_bytes(2) == str(block * 4)
        assert equations.op_count(2) == str(2 * block)

        work = tai.arithmetic_intensity(compiled,
                                        grid,
                                        x,
                                        y,
                                        out,
                                        n,
                                        2.0,
                                        BLOCK=block)
        assert work.grid == (n // block, 1, 1)
        assert work.bytes == 3 * n * 4
        assert work.load_bytes == 2 * n * 4
        assert work.store_bytes == n * 4
        assert work.flops == 2 * n
        assert set(work.per_arg) == {"x_ptr", "y_ptr", "out_ptr"}
        assert work.per_arg["x_ptr"]["load_bytes"] == n * 4
        assert work.per_arg["x_ptr"]["store_bytes"] == 0
        assert work.per_arg["out_ptr"]["store_bytes"] == n * 4
        assert work == equations.evaluate(grid, x, y, out, n, 2.0, BLOCK=block)
    finally:
        listener.uninstall()


def _fake_kernel(tai,
                 name: str,
                 equations: dict,
                 arg_names: list[str],
                 signature: dict,
                 constants: dict | None = None):
    """A stand-in ``CompiledKernel`` carrying arithmetic-intensity metadata."""
    from collections import namedtuple
    from types import SimpleNamespace

    assert tai.METADATA_KEY == "arithmetic_intensity"
    Metadata = namedtuple("Metadata", ["name", "arithmetic_intensity", "hash"])
    src = SimpleNamespace(fn=SimpleNamespace(arg_names=arg_names,
                                             signature=None),
                          constants=constants or {},
                          signature=signature)
    return SimpleNamespace(name=name,
                           src=src,
                           metadata=Metadata(name, {name: equations}, "h"))


def test_arithmetic_intensity_utilities_record_and_report(capsys) -> None:
    """Launch recording, summed work and equation printing (no GPU needed)."""
    tai = _import_arithmetic_intensity()
    kernel = _fake_kernel(
        tai,
        "k",
        {
            "0": {
                "load_bytes": "args[1] * 4",
                "op_count": "2 * args[1]"
            },
            "2": {
                "store_bytes": "program_id[0] * 8"
            },
        },
        ["x", "n", "y", "BLOCK"],
        {
            "x": "*fp32",
            "n": "i32",
            "y": "*fp32",
            "BLOCK": "constexpr"
        },
        constants={(3, ): 128},
    )

    # `kernel[grid](*args, **kwargs)` returning the compiled kernel.
    class Jit:

        def __getitem__(self, grid):
            return lambda *args, **kwargs: kernel

    # Recorded launches are detached from tensors (if torch is around).
    x = object()
    try:
        import torch
        x = torch.zeros(2)
    except ImportError:
        pass

    tai.last_launches.clear()
    with tai.recording() as outer:
        result, inner = tai.record(
            lambda: tai.launch(Jit(), (4, ), x, 256, None, BLOCK=128))
        tai.launch(Jit(), (2, ), x, 16, None, BLOCK=128)
    assert result is kernel
    assert len(inner) == 1 and len(outer) == 2
    assert tai.last_launches["k"] is outer[-1]
    first = inner[0]
    assert isinstance(first, tai.KernelLaunch)
    assert first.name == "k"
    assert first.args[0] is None or first.args[0] is x  # detached tensor
    assert first.bound_args() == {
        "x": first.args[0],
        "n": 256,
        "y": None,
        "BLOCK": 128
    }

    # Per-launch and summed work.
    work = first.work()
    assert work.grid == (4, 1, 1)
    assert work.load_bytes == 4 * 256 * 4
    assert work.store_bytes == 8 * (0 + 1 + 2 + 3)
    assert work.bytes == work.load_bytes + work.store_bytes
    assert work.flops == 4 * 2 * 256
    total = tai.Work.of(outer)
    assert total.flops == work.flops + 2 * 2 * 16
    assert total.load_bytes == work.load_bytes + 2 * 16 * 4
    assert total.store_bytes == work.store_bytes + 8
    assert total.bytes == total.load_bytes + total.store_bytes
    assert total.num_programs == 6
    assert total.intensity == total.flops / total.bytes
    assert total.tflops(1.0) == pytest.approx(total.flops * 1e-9)
    assert list(total.per_kernel) == ["k"]
    assert total.per_kernel["k"].flops == total.flops

    # Equations rendered with `args[i]` resolved to parameter names.
    text = tai.format_equations(first.equations(), *first.args, **first.kwargs)
    assert text.splitlines() == [
        "k [BLOCK=128]",
        "    x  load_bytes: args[1] * 4",
        "    x    op_count: 2 * args[1]",
        "    y store_bytes: program_id[0] * 8",
    ]
    tai.print_equations(outer)  # one block per kernel function
    out = capsys.readouterr().out
    assert out.startswith(tai.DEFAULT_TITLE + "\n    k [BLOCK=128]\n")
    assert out.count("k [BLOCK=128]") == 1
    tai.print_equations(kernel, title=None)
    assert capsys.readouterr().out == text + "\n"


def test_arithmetic_intensity_compile_with_config(monkeypatch) -> None:
    """User scenario: evaluate and run one config of an autotuned kernel."""
    tai = _import_arithmetic_intensity(monkeypatch)
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed; skipping kernel compile test")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available; skipping kernel compile test")

    import triton
    import triton.language as tl

    configs = [
        triton.Config({"BLOCK": 256}, num_warps=2),
        triton.Config({"BLOCK": 1024}, num_warps=4),
    ]

    @triton.autotune(configs=configs, key=["n"])
    @triton.jit
    def _scale(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs,
                 alpha * tl.load(x_ptr + offs, mask=mask),
                 mask=mask)

    assert tai.jit_function(_scale) is _scale.fn
    n = 4096
    x = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(n, META["BLOCK"]), )  # noqa: E731
    for config in configs:
        block = config.kwargs["BLOCK"]
        launch = tai.compile_with_config(_scale, config, grid, x, out, n, 2.0)
        assert isinstance(launch, tai.ConfigLaunch)
        assert launch.config is config
        assert launch.kwargs["num_warps"] == config.num_warps
        # The equations are known before the kernel is ever launched.
        work = launch.work()
        assert work.grid == (n // block, 1, 1)
        assert work.bytes == 2 * n * 4
        assert work.flops == n
        out.zero_()
        launch.run()
        torch.cuda.synchronize()
        assert torch.allclose(out, 2.0 * x)
    with pytest.raises(ValueError, match="Conflicting"):
        tai.compile_with_config(_scale,
                                configs[0],
                                grid,
                                x,
                                out,
                                n,
                                2.0,
                                BLOCK=64)

    # `launch` records the autotuner's pick, `record` gathers it.
    tai.last_launches.clear()
    compiled, launches = tai.record(
        lambda: tai.launch(_scale, grid, x, out, n, 2.0))
    assert [kl.kernel for kl in launches] == [compiled]
    assert launches[0].args[:2] == (None, None)  # tensors detached
    assert launches[0].work().flops == n
    assert tai.last_launches[compiled.name] is launches[0]

    # Kernel parameters may also be given by keyword (as a hook sees them).
    by_name = tai.compile_with_config(_scale,
                                      configs[0],
                                      grid,
                                      x_ptr=x,
                                      out_ptr=out,
                                      n=n,
                                      alpha=2.0)
    assert by_name.work() == tai.compile_with_config(_scale, configs[0], grid,
                                                     x, out, n, 2.0).work()
    out.zero_()
    by_name.run()
    torch.cuda.synchronize()
    assert torch.allclose(out, 2.0 * x)

    # An AutotuneRecorder as the autotuner's post_hook captures a launch per
    # benchmarked config while the autotuner tunes.
    recorder = tai.AutotuneRecorder()

    @triton.autotune(configs=configs, key=["n"], post_hook=recorder)
    @triton.jit
    def _scale_recorded(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs,
                 alpha * tl.load(x_ptr + offs, mask=mask),
                 mask=mask)

    recorder.attach(_scale_recorded)
    _scale_recorded[grid](x, out, n, 2.0)
    assert set(recorder.launches) == set(configs)
    assert not recorder.errors
    for config, recorded in recorder.launches.items():
        assert recorded.config is config
        assert recorded.kwargs["n"] == n
        assert recorded.kwargs["BLOCK"] == config.kwargs["BLOCK"]
        assert recorded.work().grid == (n // config.kwargs["BLOCK"], 1, 1)
        assert recorded.work().flops == n
        out.zero_()
        recorded.run()
        torch.cuda.synchronize()
        assert torch.allclose(out, 2.0 * x)
    # `install` chains onto the existing hook; a new tuning key fills both.
    second = tai.AutotuneRecorder().install(_scale_recorded)
    recorder.clear()
    half = n // 2
    grid_half = lambda META: (triton.cdiv(half, META["BLOCK"]), )  # noqa: E731
    _scale_recorded[grid_half](x[:half], out[:half], half, 2.0)
    assert set(second.launches) == set(configs) == set(recorder.launches)
    assert second.launches[configs[0]].work().flops == half


def test_arithmetic_intensity_pruner(monkeypatch) -> None:
    """User scenario: drop low-intensity configs before the autotuner times them."""
    tai = _import_arithmetic_intensity(monkeypatch)
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed; skipping kernel compile test")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available; skipping kernel compile test")

    import triton
    import triton.language as tl

    # The intensity of a matmul tile grows with its size: every program
    # streams BLOCK_M x K of `a` and K x BLOCK_N of `b` for 2*BLOCK_M*BLOCK_N*K
    # FLOPs, so the pruner has something to discriminate on.
    configs = [
        triton.Config({
            "BLOCK_M": 16,
            "BLOCK_N": 16
        }, num_warps=1),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 32
        }, num_warps=2),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64
        }, num_warps=4),
    ]
    pruner = tai.IntensityPruner(min_intensity=None)
    recorder = tai.AutotuneRecorder()

    @triton.autotune(configs=configs,
                     key=["M", "N", "K"],
                     prune_configs_by={"early_config_prune": pruner},
                     post_hook=recorder)
    @triton.jit
    def _matmul(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, 16)
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, 16):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
            a_ptrs += 16
            b_ptrs += 16 * N
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc.to(tl.float16))

    pruner.attach(_matmul)
    recorder.attach(_matmul)
    M = N = K = 128
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    def grid(META):
        return triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"])

    def tune():
        _matmul.cache.clear()
        recorder.clear()
        _matmul[grid](a, b, c, M, N, K)
        torch.cuda.synchronize()
        assert torch.allclose(c, a @ b, atol=1e-1, rtol=1e-2)

    # Disabled (`None`): everything passes through to the autotuner untouched.
    tune()
    assert not pruner.launches and not pruner.pruned
    assert set(recorder.launches) == set(configs)
    intensity = {
        cfg: launch.work().intensity
        for cfg, launch in recorder.launches.items()
    }
    assert intensity[configs[0]] < intensity[configs[1]] < intensity[
        configs[2]]

    # Between the two smallest tiles: the autotuner only times the others.
    pruner.min_intensity = (intensity[configs[0]] + intensity[configs[1]]) / 2
    tune()
    assert set(pruner.launches) == set(configs)  # all compiled and evaluated
    assert pruner.pruned == [configs[0]]
    assert not pruner.errors
    assert set(recorder.launches) == set(configs[1:])
    assert _matmul.best_config in configs[1:]
    for cfg in configs:
        assert pruner.work[cfg] == pruner.launches[cfg].work()
        assert pruner.work[cfg].intensity == intensity[cfg]
    # The pruned configuration can still be launched by hand.
    c.zero_()
    pruner.launches[configs[0]].run()
    torch.cuda.synchronize()
    assert torch.allclose(c, a @ b, atol=1e-1, rtol=1e-2)

    # Nothing reaches the threshold: the most intense configuration is kept
    # (the autotuner needs one) and, alone, is not benchmarked at all.
    pruner.min_intensity = 2 * intensity[configs[2]]
    tune()
    assert pruner.pruned == configs[:2]
    assert not recorder.launches
    assert _matmul.best_config is configs[2]
    pruner.keep_at_least = 2
    tune()
    assert pruner.pruned == [configs[0]]
    assert set(recorder.launches) == set(configs[1:])

    # `install` chains after an existing `early_config_prune`.
    def drop_last(configs, named_args, **kwargs):
        assert named_args["M"] == M and "grid" in kwargs
        return configs[:-1]

    _matmul.early_config_prune = drop_last
    installed = tai.IntensityPruner(intensity[configs[1]]).install(_matmul)
    assert _matmul.early_config_prune is installed
    tune()
    assert set(installed.launches) == set(configs[:2])
    assert installed.pruned == [configs[0]]
    assert not recorder.launches  # a single config left: nothing to time
    assert _matmul.best_config is configs[1]

    with pytest.raises(RuntimeError, match="attach"):
        tai.IntensityPruner(1.0)(configs, {"M": M}, grid=grid)
