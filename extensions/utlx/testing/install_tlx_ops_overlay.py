#!/usr/bin/env python3
"""Make `triton.tlx.ops` importable on a stock Triton + triton-utlx install.

The uTLX plugin ships the TLX *language*. The TLX *op library* -- `tlx.ops.mm`,
`tlx.ops.flash_attn`, ... and the tests that cover them -- lives in Meta's
Triton fork under `third_party/tlx/ops`, and is not part of any wheel. Point
this script at a checkout of

    https://github.com/facebookexperimental/triton

and it installs, into the active interpreter's site-packages:

  triton/tlx/ops/          a copy of that tree
  triton/tlx/__init__.py   imports utlx_plugin first (which is what registers
                           the DSL as triton.language.extra.tlx), adapts
                           `triton.Config(ctas_per_cga=...)` to upstream's
                           `num_ctas`, and reports configs this Triton cannot
                           lower as `UnsupportedOp`
  utlx_plugin/warp_spec.py a small helper the plugin's TLX snapshot predates

Nothing from the fork is vendored into this repository; it is copied from the
checkout you supply, at the version you supply.

Usage:
    python install_tlx_ops_overlay.py --fb-triton /path/to/facebookexperimental-triton
"""

import argparse
import pathlib
import shutil
import site
import sys


def site_packages():
    paths = site.getsitepackages()
    for path in paths:
        if path.endswith("site-packages"):
            return pathlib.Path(path)
    return pathlib.Path(paths[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fb-triton", required=True, type=pathlib.Path,
                        help="checkout of github.com/facebookexperimental/triton")
    args = parser.parse_args()

    source = args.fb_triton.expanduser().resolve()
    ops_src = source / "third_party" / "tlx" / "ops"
    warp_spec_src = source / "third_party" / "tlx" / "language" / "tlx" / "warp_spec.py"
    for path in (ops_src, warp_spec_src):
        if not path.exists():
            sys.exit(f"not found: {path}\n--fb-triton must point at a checkout of "
                     "github.com/facebookexperimental/triton")

    packages = site_packages()
    triton_dir = packages / "triton"
    plugin_dir = packages / "utlx_plugin"
    for path in (triton_dir, plugin_dir):
        if not path.is_dir():
            sys.exit(f"not found: {path}\ninstall `torch` (for triton) and the "
                     "triton-utlx wheel into this interpreter first")

    here = pathlib.Path(__file__).resolve().parent
    tlx_dir = triton_dir / "tlx"
    if tlx_dir.exists():
        shutil.rmtree(tlx_dir)
    tlx_dir.mkdir(parents=True)
    shutil.copytree(ops_src, tlx_dir / "ops")
    shutil.copy2(here / "overlay" / "triton" / "tlx" / "__init__.py",
                 tlx_dir / "__init__.py")
    shutil.copy2(warp_spec_src, plugin_dir / "warp_spec.py")

    for cache in tlx_dir.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    print(f"installed triton.tlx.ops into {packages}")


if __name__ == "__main__":
    main()
