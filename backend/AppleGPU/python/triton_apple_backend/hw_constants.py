"""Constants compiler.py and driver.py share, in a separate module because
neither can import the other.

The hardware ones mirror kWarpSize, kSgFragDim and kTGResidentBudgetBytes in
agpu/include/agpu/core/Units.h, which owns them, and MAX_PIPELINE_STAGES
mirrors kMaxPipelineStages in agpu/include/agpu/cost/Pipeline.h.
"""

WARP_SIZE = 32
SG_FRAG_DIM = 8
TG_BUDGET_BYTES = 32768
MAX_PIPELINE_STAGES = 2

# Torch calls the Apple GPU device "mps", so Triton's target must agree.
TARGET = "mps"


def target_arch(arch):
    """The `backend:arch` string Triton's target parser expects."""
    return f"{TARGET}:{arch}"
