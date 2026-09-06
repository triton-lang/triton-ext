"""Constants compiler.py and driver.py share, in a separate module because
neither can import the other.

kWarpSize in agpu/include/agpu/core/Units.h owns the warp size;
test_argbuf_abi.py reads that header and fails if the two drift. The other two
have no C++ owner in this stage: Triton requires the option fields and the
device property regardless of what the emitter uses.
"""

WARP_SIZE = 32
SG_FRAG_DIM = 8
TG_BUDGET_BYTES = 32768

# Torch calls the Apple GPU device "mps", so Triton's target must agree.
TARGET = "mps"


def target_arch(arch):
    """The `backend:arch` string Triton's target parser expects."""
    return f"{TARGET}:{arch}"
