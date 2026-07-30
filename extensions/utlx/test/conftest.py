"""Shared fixtures and helpers for uTLX plugin tests."""

import pytest

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import triton

import utlx_plugin  # noqa: F401  registers passes/dialects/builder and tlx DSL
import triton.language.extra.tlx as tlx  # noqa: F401


def pytest_ignore_collect(collection_path, config):
    """Skip collection of test files when torch is not installed.

    Test files import torch at the top level for GPU device detection;
    prevent collection errors in environments without torch installed.
    """
    if not _HAS_TORCH and collection_path.suffix == ".py" \
            and collection_path.name.startswith("test_"):
        return True
    return None


if _HAS_TORCH:
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
else:
    DEVICE = None


def is_hip():
    if not _HAS_TORCH:
        return False
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def is_cuda():
    if not _HAS_TORCH:
        return False
    return torch.cuda.is_available() and not is_hip()


def is_hopper_or_newer():
    try:
        return is_cuda() and torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


def is_hopper():
    try:
        return is_cuda() and torch.cuda.get_device_capability() == (9, 0)
    except Exception:
        return False


def is_blackwell():
    try:
        return is_cuda() and torch.cuda.get_device_capability()[0] >= 10
    except Exception:
        return False


def is_hip_cdna2():
    if not is_hip():
        return False
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target.arch in ("gfx90a", )
    except Exception:
        return False


def get_current_target():
    return triton.runtime.driver.active.get_current_target()


@pytest.fixture
def device():
    return DEVICE
