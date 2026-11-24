"""The fork's extra HIPOptions reach the AMDGPU code generator."""

import pytest
import torch
import triton
import triton.language as tl

from conftest import DEVICE, is_hip

pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires AMD")


@triton.jit
def _copy(x_ptr, y_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))


def _codegen_flags(monkeypatch, **options):
    """The LLVM flags each compile of ``_copy`` hands the code generator."""
    import triton.backends.amd.compiler as hip
    seen = []
    if hasattr(hip, "compile_amdgpu"):  # Triton 3.9
        load = hip._load_amd_codegen

        class Recorder:

            def __init__(self, library):
                self._library = library

            def __getattr__(self, name):
                return getattr(self._library, name)

            def triton_amdgpu_compile(self, src, size, options, *rest):
                seen.append(options._obj.flags.decode().split(","))
                return self._library.triton_amdgpu_compile(
                    src, size, options, *rest)

        monkeypatch.setattr(hip, "_load_amd_codegen",
                            lambda path: Recorder(load(path)))
    else:  # Triton 3.8: llvm.translate_to_asm
        llvm = hip.llvm
        orig = llvm._llvm.translate_to_asm

        def record(*args):
            seen.append(list(args[4]))
            return orig(*args)

        monkeypatch.setattr(llvm._llvm, "translate_to_asm", record)
    x = torch.arange(64, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    _copy[(1, )](x, y, 64, **options)
    torch.testing.assert_close(y, x)
    return seen


def test_llvm_cl_opts_and_codegen_flags(monkeypatch):
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    [flags] = _codegen_flags(
        monkeypatch,
        reverse_local_assignment=True,
        llvm_cl_opts="amdgpu-use-amdgpu-trackers,unroll-threshold=150")
    assert flags[-3:] == [
        "greedy-reverse-local-assignment",
        "amdgpu-use-amdgpu-trackers=true",
        "unroll-threshold=150",
    ]
