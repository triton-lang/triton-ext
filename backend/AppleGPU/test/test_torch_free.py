"""The backend must import and dispatch on a box without torch. Torch is
hidden from a subprocess with a meta-path finder that refuses it, which is
what a real torch-less install looks like to Triton (no `torch` key in
sys.modules)."""

import importlib.util
import subprocess
import sys
import textwrap

import pytest

_NO_TORCH_PRELUDE = textwrap.dedent("""
    import sys, importlib.abc

    class _NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'torch' or fullname.startswith('torch.'):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    sys.meta_path.insert(0, _NoTorch())
""")


def _run_without_torch(body):
    got = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=600)
    assert got.returncode == 0, got.stdout + got.stderr
    return got.stdout


def test_import_triton_without_torch():
    out = _run_without_torch("""
        import triton
        import triton_apple_backend.driver as d
        assert 'torch' not in sys.modules
        print('apple' in triton.backends.backends)
    """)
    assert out.strip() == "True"


@pytest.mark.skipif(sys.platform != "darwin", reason="needs Metal")
@pytest.mark.skipif(
    importlib.util.find_spec("triton_apple_backend.metal_native") is None,
    reason="metal_native is not built")
def test_dispatch_without_torch():
    out = _run_without_torch("""
        import numpy as np
        import triton
        import triton.language as tl
        from triton.runtime import driver
        from triton_apple_backend import metal_native
        from triton_apple_backend.driver import _NativeRuntime, _runtime

        assert type(driver.active).__name__ == 'MetalDriver'
        assert isinstance(_runtime(), _NativeRuntime)

        @triton.jit
        def add(x_ptr, y_ptr, out_ptr, n, scale, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, (x + y) * scale, mask=mask)

        @triton.jit
        def show(x_ptr):
            tl.device_print("v", tl.load(x_ptr + tl.program_id(0)))

        n = 1000
        x = np.random.rand(n).astype(np.float32)
        y = np.random.rand(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        grid = (triton.cdiv(n, 256),)

        add[grid](metal_native.wrap(x), metal_native.wrap(y), metal_native.wrap(out), n, 2.0, BLOCK=256)
        metal_native.synchronize()
        assert np.allclose(out, (x + y) * 2.0)

        ob = metal_native.alloc(n * 4, np.dtype('float32'))
        add[grid](metal_native.wrap(x), metal_native.wrap(y), ob, n, 0.5, BLOCK=256)
        assert np.allclose(np.frombuffer(ob, dtype=np.float32), (x + y) * 0.5)

        show[(2,)](metal_native.wrap(np.array([1.5, 2.5], dtype=np.float32)))
        metal_native.synchronize()
        assert 'torch' not in sys.modules
    """)
    assert "v:1.5" in out and "v:2.5" in out


@pytest.mark.skipif(sys.platform != "darwin", reason="needs Metal")
@pytest.mark.skipif(
    importlib.util.find_spec("triton_apple_backend.metal_native") is None,
    reason="metal_native is not built")
def test_address_table_without_torch():
    out = _run_without_torch("""
        import numpy as np
        import triton
        import triton.language as tl
        from triton_apple_backend import metal_native
        from triton_apple_backend.address import address_table, gpu_address

        @triton.jit
        def through(tab, out, n, BLOCK: tl.constexpr):
            e = tl.program_id(0)
            offs = tl.arange(0, BLOCK)
            mask = offs < n
            src = tl.load(tab + e).to(tl.pointer_type(tl.float32))
            tl.store(out + e * n + offs,
                     tl.load(src + offs, mask=mask, other=0.), mask=mask)

        n, experts = 256, 4
        data = [np.arange(i * n, (i + 1) * n, dtype=np.float32)
                for i in range(experts)]
        bufs = []
        for d in data:
            b = metal_native.alloc(n * 4, np.dtype('float32'))
            np.frombuffer(b, dtype=np.float32)[:] = d
            bufs.append(b)

        ob = metal_native.alloc(experts * n * 4, np.dtype('float32'))
        through[(experts,)](address_table(bufs), ob, n, BLOCK=n)
        metal_native.synchronize()
        assert np.array_equal(np.frombuffer(ob, dtype=np.float32),
                              np.concatenate(data))

        assert gpu_address(bufs[0]) != bufs[0].data_ptr()

        try:
            gpu_address(metal_native.wrap(data[0]))
        except RuntimeError:
            pass
        else:
            raise AssertionError("a wrapped buffer should have no gpu address")

        assert 'torch' not in sys.modules
        print('ok')
    """)
    assert out.strip() == "ok"
