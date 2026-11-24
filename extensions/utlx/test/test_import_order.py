"""uTLX can be imported before PyTorch."""

import subprocess
import sys


def test_tlx_before_torch():
    # Loading the plugin first used to abort the process when PyTorch's ROCm
    # libraries loaded: "LLVM ERROR: Option 'default' already exists!".
    result = subprocess.run([
        sys.executable, "-c", "import triton.language.extra.tlx; import torch"
    ],
                            capture_output=True,
                            text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
