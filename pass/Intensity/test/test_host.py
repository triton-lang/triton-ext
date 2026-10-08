"""Host-side tests for the intensity plugin.

Covers the ``ttir`` stage hook, equation parsing, the per-kernel listener,
launch recording, single-config compilation, and intensity pruning.
"""

from __future__ import annotations

import pytest


def _import_intensity(monkeypatch=None):
    """Import the plugin; with ``monkeypatch``, also isolate its pipeline hook.

    ``knobs.runtime.add_stages_inspection_hook`` is a single global slot that
    every plugin imported in this process competes for, so put ours on top
    and (for the duration of the test) detach it from any other plugin's hook.
    """
    try:
        import triton_intensity as tint
    except ImportError:
        pytest.skip("triton_intensity not installed "
                    "(run `make build && make install`)")
    if monkeypatch is not None:
        tint.custom_stages.install()
        monkeypatch.setattr(tint.custom_stages, "_previous", None)
        monkeypatch.setattr(tint.custom_stages, "_cache_key", None)
    return tint


def test_intensity_installs_ttir_hook(monkeypatch) -> None:
    """Importing the package appends the pass to every ``ttir`` stage."""
    tint = _import_intensity(monkeypatch)
    from triton import knobs

    hook = knobs.runtime.add_stages_inspection_hook
    assert hook is tint.custom_stages.inspect_stages_hook

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


def test_intensity_parses_arg_attrs() -> None:
    """Equations are recovered from the printed ``tt.func`` signature."""
    tint = _import_intensity()
    parse = tint.custom_stages.parse_function_equations

    sig = (
        'tt.func public @"k(x, y"(%arg0: !tt.ptr<f32, 3> '
        '{tint.load_bytes = "(args[5] / 64) * 8192", tt.divisibility = 16 : i32}, '
        '%arg1: !tt.tensordesc<tensor<64x64xf16>>, '
        '%arg2: tensor<64x!tt.ptr<f32>> {tint.store_bytes = "8192", '
        'tint.op_count = "a \\"q\\" , eq"}, '
        '%arg3: i32 {tint.op_count = "num_programs[0] % 3"}) '
        'attributes {noinline = false, tint.load_bytes = "not an arg"} {\n'
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


def test_intensity_listener_gathers_per_kernel(monkeypatch) -> None:
    """User scenario: launch a kernel, read its work from the listener."""
    tint = _import_intensity(monkeypatch)
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

    listener = tint.IntensityListener().install()
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
        assert tint.METADATA_KEY in compiled.metadata._asdict()
        # ...and gathered per kernel function by the listener.
        assert compiled.name in listener
        equations = listener[compiled]
        assert listener.get(_axpy) is equations
        assert equations.entry[0] == {"load_bytes": str(block * 4)}
        assert equations.load_bytes(0) == str(block * 4)
        assert equations.store_bytes(2) == str(block * 4)
        assert equations.op_count(2) == str(2 * block)

        work = tint.intensity(compiled, grid, x, y, out, n, 2.0, BLOCK=block)
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


def _fake_kernel(tint,
                 name: str,
                 equations: dict,
                 arg_names: list[str],
                 signature: dict,
                 constants: dict | None = None):
    """A stand-in ``CompiledKernel`` carrying intensity metadata."""
    from collections import namedtuple
    from types import SimpleNamespace

    assert tint.METADATA_KEY == "intensity"
    Metadata = namedtuple("Metadata", ["name", "intensity", "hash"])
    src = SimpleNamespace(fn=SimpleNamespace(arg_names=arg_names,
                                             signature=None),
                          constants=constants or {},
                          signature=signature)
    return SimpleNamespace(name=name,
                           src=src,
                           metadata=Metadata(name, {name: equations}, "h"))


def test_intensity_utilities_record_and_report(capsys) -> None:
    """Launch recording, summed work and equation printing (no GPU needed)."""
    tint = _import_intensity()
    kernel = _fake_kernel(
        tint,
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

    tint.last_launches.clear()
    with tint.recording() as outer:
        result, inner = tint.record(
            lambda: tint.launch(Jit(), (4, ), x, 256, None, BLOCK=128))
        tint.launch(Jit(), (2, ), x, 16, None, BLOCK=128)
    assert result is kernel
    assert len(inner) == 1 and len(outer) == 2
    assert tint.last_launches["k"] is outer[-1]
    first = inner[0]
    assert isinstance(first, tint.KernelLaunch)
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
    total = tint.Work.of(outer)
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
    text = tint.format_equations(first.equations(), *first.args,
                                 **first.kwargs)
    assert text.splitlines() == [
        "k [BLOCK=128]",
        "    x  load_bytes: args[1] * 4",
        "    x    op_count: 2 * args[1]",
        "    y store_bytes: program_id[0] * 8",
    ]
    tint.print_equations(outer)  # one block per kernel function
    out = capsys.readouterr().out
    assert out.startswith(tint.DEFAULT_TITLE + "\n    k [BLOCK=128]\n")
    assert out.count("k [BLOCK=128]") == 1
    tint.print_equations(kernel, title=None)
    assert capsys.readouterr().out == text + "\n"


def test_intensity_compile_with_config(monkeypatch) -> None:
    """User scenario: evaluate and run one config of an autotuned kernel."""
    tint = _import_intensity(monkeypatch)
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

    assert tint.jit_function(_scale) is _scale.fn
    n = 4096
    x = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(n, META["BLOCK"]), )  # noqa: E731
    for config in configs:
        block = config.kwargs["BLOCK"]
        launch = tint.compile_with_config(_scale, config, grid, x, out, n, 2.0)
        assert isinstance(launch, tint.ConfigLaunch)
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
        tint.compile_with_config(_scale,
                                 configs[0],
                                 grid,
                                 x,
                                 out,
                                 n,
                                 2.0,
                                 BLOCK=64)

    # `launch` records the autotuner's pick, `record` gathers it.
    tint.last_launches.clear()
    compiled, launches = tint.record(
        lambda: tint.launch(_scale, grid, x, out, n, 2.0))
    assert [kl.kernel for kl in launches] == [compiled]
    assert launches[0].args[:2] == (None, None)  # tensors detached
    assert launches[0].work().flops == n
    assert tint.last_launches[compiled.name] is launches[0]

    # Kernel parameters may also be given by keyword (as a hook sees them).
    by_name = tint.compile_with_config(_scale,
                                       configs[0],
                                       grid,
                                       x_ptr=x,
                                       out_ptr=out,
                                       n=n,
                                       alpha=2.0)
    assert by_name.work() == tint.compile_with_config(_scale, configs[0], grid,
                                                      x, out, n, 2.0).work()
    out.zero_()
    by_name.run()
    torch.cuda.synchronize()
    assert torch.allclose(out, 2.0 * x)

    # An AutotuneRecorder as the autotuner's post_hook captures a launch per
    # benchmarked config while the autotuner tunes.
    recorder = tint.AutotuneRecorder()

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
    second = tint.AutotuneRecorder().install(_scale_recorded)
    recorder.clear()
    half = n // 2
    grid_half = lambda META: (triton.cdiv(half, META["BLOCK"]), )  # noqa: E731
    _scale_recorded[grid_half](x[:half], out[:half], half, 2.0)
    assert set(second.launches) == set(configs) == set(recorder.launches)
    assert second.launches[configs[0]].work().flops == half


def test_intensity_pruner(monkeypatch) -> None:
    """User scenario: drop low-intensity configs before the autotuner times them."""
    tint = _import_intensity(monkeypatch)
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
    pruner = tint.IntensityPruner(min_intensity=None)
    recorder = tint.AutotuneRecorder()

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
    installed = tint.IntensityPruner(intensity[configs[1]]).install(_matmul)
    assert _matmul.early_config_prune is installed
    tune()
    assert set(installed.launches) == set(configs[:2])
    assert installed.pruned == [configs[0]]
    assert not recorder.launches  # a single config left: nothing to time
    assert _matmul.best_config is configs[1]

    with pytest.raises(RuntimeError, match="attach"):
        tint.IntensityPruner(1.0)(configs, {"M": M}, grid=grid)
