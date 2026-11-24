"""A failed device assert surfaces at the next synchronize, as on CUDA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton_apple_backend.device_assert import DeviceAssertError  # noqa: E402


@triton.jit(debug=True)
def bounded_kernel(x_ptr, LIMIT: tl.constexpr):
    i = tl.arange(0, 64)
    tl.device_assert(tl.load(x_ptr + i) < LIMIT, "value past the limit")


def test_the_launch_returns_and_the_synchronize_raises():
    x = torch.arange(64, dtype=torch.int32, device="mps")
    bounded_kernel[(1, )](x, LIMIT=10)
    with pytest.raises(DeviceAssertError, match="value past the limit"):
        torch.mps.synchronize()


def test_a_reported_failure_does_not_repeat():
    x = torch.arange(64, dtype=torch.int32, device="mps")
    bounded_kernel[(1, )](x, LIMIT=10)
    with pytest.raises(DeviceAssertError):
        torch.mps.synchronize()
    bounded_kernel[(1, )](x, LIMIT=100)
    torch.mps.synchronize()


def test_the_next_launch_raises_once_the_failure_has_landed():
    x = torch.arange(64, dtype=torch.int32, device="mps")
    bounded_kernel[(1, )](x, LIMIT=10)
    # Waits for the stream without going through torch.mps.synchronize.
    torch.ones(1, device="mps").sum().item()
    with pytest.raises(DeviceAssertError, match="value past the limit"):
        bounded_kernel[(1, )](x, LIMIT=100)
    torch.mps.synchronize()


def test_a_passing_assert_never_raises():
    x = torch.arange(64, dtype=torch.int32, device="mps")
    for _ in range(3):
        bounded_kernel[(1, )](x, LIMIT=100)
    torch.mps.synchronize()
