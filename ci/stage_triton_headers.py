#!/usr/bin/env python3
"""Stage Triton's C++ headers into an installed Triton wheel.

Building an extension needs Triton's headers, which a wheel only carries if it
was built with TRITON_EXT_ENABLED=1 *and* its setup.py runs the
`wheel_headers` cmake install component. Nightly wheels do; Triton releases up
to and including v3.8.0 (`c01b6774`) predate that component, so a wheel built
from a release tag has `triton/_C/libtriton.so` but no `triton/include`, and
`find_triton_wheel` rejects it with "Triton wheel is missing C++ headers".

This reproduces the same layout by hand from the source tree plus the CMake
binary directory, which holds the TableGen-generated `*.h.inc`, so it has to
run after a full Triton build.

Usage:
    python ci/stage_triton_headers.py <triton-source-dir> [<dest-include-dir>]

With no destination it writes to the `triton/include` of the Triton importable
by the current interpreter.
"""

import pathlib
import shutil
import sys

SRC_ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "triton380").resolve()
DST = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None

if DST is None:
    import triton
    DST = pathlib.Path(triton.__file__).parent / "include"

build_dirs = sorted((SRC_ROOT / "build").glob("cmake.*"))
if not build_dirs:
    sys.exit(f"no CMake build dir under {SRC_ROOT / 'build'}; build Triton first")
bld = build_dirs[0]

if DST.exists():
    shutil.rmtree(DST)

staged = 0
for base in (SRC_ROOT / "include", bld / "include"):
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".h", ".td") and not path.name.endswith(".h.inc"):
            continue
        out = DST / path.relative_to(base)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        staged += 1

for name in ("ir.h", "passes.h"):
    path = SRC_ROOT / "python" / "src" / name
    if path.exists():
        out = DST / "python" / "src" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        staged += 1

print(f"staged {staged} headers into {DST}")
