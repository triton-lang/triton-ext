#!/usr/bin/env python3
"""Lower a basic kernel through whichever plugin is loaded via TRITON_PLUGIN_PATHS.

Run manually to debug a plugin's compile pipeline:

    TRITON_PLUGIN_PATHS=/path/to/lib<name>.so python testing/scripts/compile_kernel.py

Exits 0 on success (or if no target is available, e.g. no GPU present).
"""

import sys

import triton
import triton.language as tl


@triton.jit
def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(in_ptr + offs))


def main() -> int:
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception as e:
        print(f"No target ({type(e).__name__}: {e}); skipping compile.")
        return 0
    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={
            "in_ptr": "*fp32",
            "out_ptr": "*fp32"
        },
        constexprs={"BLOCK": 128},
    )
    triton.compile(src, target=target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
