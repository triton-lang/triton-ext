"""Constants compiler.py and driver.py share, in a separate module because
neither can import the other.

The hardware ones mirror kWarpSize, kSgFragDim and kTGResidentBudgetBytes in
agpu/include/agpu/core/Units.h, which owns them. test_argbuf_abi.py reads that
header and fails if these drift.
"""

WARP_SIZE = 32
SG_FRAG_DIM = 8
TG_BUDGET_BYTES = 32768

# Torch calls the Apple GPU device "mps", so Triton's target must agree.
TARGET = "mps"


def target_arch(arch):
    """The `backend:arch` string Triton's target parser expects."""
    return f"{TARGET}:{arch}"
