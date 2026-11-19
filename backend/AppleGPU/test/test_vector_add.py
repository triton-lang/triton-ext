"""Runs the vector-add kernel on the GPU and checks the result.

This is the end-to-end path: compile, build a metallib, dispatch through
torch's MPS stream, read the tensor back. It needs an Apple GPU, so it skips
everywhere else; the compile-only coverage lives in `test_backend_discovery`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="needs torch")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off,
             tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m),
             mask=m)


def _add(x, y):
    out = torch.empty_like(x)
    n = out.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK"]), )

    add_kernel[grid](x, y, out, n, BLOCK=1024)
    torch.mps.synchronize()
    return out


def test_the_active_backend_is_ours():
    from triton.runtime.driver import driver
    assert type(driver.active).__name__ == "MetalDriver"


@pytest.mark.parametrize("n", [1, 1023, 1024, 98432])
def test_vector_add_matches_torch(n):
    torch.manual_seed(0)
    x = torch.rand(n, device="mps")
    y = torch.rand(n, device="mps")
    assert torch.equal(_add(x, y), x + y)


def test_a_ragged_tail_leaves_the_rest_alone():
    # The mask has to reject exactly the tail: 1000 of a 1024 block.
    torch.manual_seed(0)
    n = 1000
    x = torch.rand(n, device="mps")
    y = torch.rand(n, device="mps")
    assert torch.equal(_add(x, y), x + y)


def test_the_kernel_compiled_through_our_backend():
    torch.manual_seed(0)
    x = torch.rand(256, device="mps")
    compiled = add_kernel[(1, )](x, x, torch.empty_like(x), 256, BLOCK=256)
    torch.mps.synchronize()
    assert "kernel void" in compiled.asm["msl"]
    assert compiled.asm["metallib"], "no metallib was produced"
