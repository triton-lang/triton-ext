"""AMD workgroup-barrier code generation; no GPU is needed."""

import re

import pytest

import triton
import triton.language as tl
from conftest import tlx
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _barrier_kernel(output):
    tlx.workgroup_barrier()
    tl.store(output, 1)


@pytest.mark.parametrize("arch,warp_size", [("gfx950", 64), ("gfx1250", 32)])
def test_workgroup_barrier_has_lds_and_scheduler_fences(arch, warp_size):
    kernel = triton.compile(
        ASTSource(_barrier_kernel, signature={"output": "*i32"}),
        target=GPUTarget("hip", arch, warp_size),
    )
    assert re.search(
        r"rocdl\.sched\.barrier none\b.*\n\s*ttg\.barrier local\b.*\n"
        r"\s*rocdl\.sched\.barrier none\b", kernel.asm["ttgir"])
    assert "llvm.amdgcn.sched.barrier" in kernel.asm["llir"]
    assert "s_barrier" in kernel.asm["amdgcn"]
