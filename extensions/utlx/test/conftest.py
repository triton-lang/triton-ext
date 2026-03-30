"""Shared fixtures and helpers for uTLX plugin tests."""

import os
import sys

import pytest
import torch

import triton
from triton import knobs

# Add the plugin python dir to sys.path
_plugin_python_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
if _plugin_python_dir not in sys.path:
    sys.path.insert(0, _plugin_python_dir)

from utlx_plugin.utility import ensure_plugin_on_path  # noqa: E402

ensure_plugin_on_path()
import triton.language.extra.tlx as tlx  # noqa: E402, F401

from utlx_plugin.custom_stages import inspect_stages_hook  # noqa: E402

# Activate the plugin's custom compilation stages
knobs.runtime.add_stages_inspection_hook = inspect_stages_hook

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def is_cuda():
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
