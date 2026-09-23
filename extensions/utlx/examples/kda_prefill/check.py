#!/usr/bin/env python3
"""Standalone check: does this triton have the TLX APIs the KDA kernels need?

    python3 check.py

Self-contained -- only torch and triton. Copy it anywhere. It reports which
TLX APIs are present and compiles/runs a minimal kernel using the async-copy
set, so a green run means those APIs actually work on this GPU, not just that
the symbols exist.

Exit codes
    0  TLX usable (decode-class APIs present and a kernel using them ran)
    1  TLX missing, or a kernel that should have worked failed
"""

import sys
import traceback

import torch
import triton
import triton.language as tl

# What the two KDA kernels import. Keep in sync with the kernels themselves.
DECODE_APIS = (
    "async_load",
    "async_load_commit_group",
    "async_load_wait_group",
    "local_alloc",
    "local_view",
    "local_load",
)
PREFILL_APIS = (
    "amd_mfma_layout",
    "dot_operand_layout",
    "release_layout",
    "slice_layout",
    "zeros",
)

try:
    from triton.language.extra import tlx

    HAVE_TLX = True
except ImportError as _exc:
    tlx = None
    HAVE_TLX = False
    _TLX_ERR = _exc


def hdr(text):
    print(f"\n{text}\n" + "-" * len(text))


# --------------------------------------------------------------------------
# Minimal kernels. The plain one is a control: if it fails, the problem is
# triton or the GPU, not TLX.
# --------------------------------------------------------------------------


@triton.jit
def _plain_tile_scale(src, dst, M: tl.constexpr, N: tl.constexpr, SCALE: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    offs = rows * N + cols
    tl.store(dst + offs, tl.load(src + offs) * SCALE)


if HAVE_TLX:

    @triton.jit
    def _tlx_tile_scale(
        src, dst, M: tl.constexpr, N: tl.constexpr, SCALE: tl.constexpr
    ):
        """Same result as above, but the tile goes through LDS via an async
        copy -- exercises the exact primitives the KDA decode kernel uses."""
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        offs = rows * N + cols

        buf = tlx.local_alloc((M, N), src.dtype.element_ty, 1)
        token = tlx.async_load(src + offs, tlx.local_view(buf, 0))
        tlx.async_load_commit_group([token])
        done = tlx.async_load_wait_group(0)
        tile = tlx.local_load(tlx.local_view(buf, 0), token=done)

        tl.store(dst + offs, tile * SCALE)


def check_env():
    hdr("environment")
    print(f"  triton   {triton.__version__}")
    print(f"           {triton.__file__}")
    print(f"  torch    {torch.__version__}")
    if not torch.cuda.is_available():
        print("  GPU      none")
        return False
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU      {getattr(props, 'gcnArchName', props.name)}")
    if HAVE_TLX:
        print(f"  TLX      {tlx.__file__}")
    else:
        print(f"  TLX      MISSING -- {_TLX_ERR}")
    return True


def _plugin_installed():
    try:
        import utlx_plugin  # noqa: F401

        return True
    except ImportError:
        return False


def check_plugin():
    """Diagnose the uTLX plugin route, which fails in two distinct ways that
    look identical from the outside."""
    import os

    try:
        import utlx_plugin
    except ImportError:
        return  # not using the plugin route at all

    hdr("uTLX plugin")
    so = os.path.join(os.path.dirname(utlx_plugin.__file__), "libutlx.so")
    print(f"  installed  {so}")
    env = os.environ.get("TRITON_PLUGIN_PATHS")
    print(f"  TRITON_PLUGIN_PATHS  {env or '(unset)'}")

    if not env:
        print(
            "\n  Installing the package is not enough -- triton only consults a\n"
            "  plugin when this variable points at the .so:\n"
            "    export TRITON_PLUGIN_PATHS=$(python3 -c \\\n"
            '      "import utlx_plugin, os; print(os.path.join('
            'os.path.dirname(utlx_plugin.__file__), \'libutlx.so\'))")'
        )
    elif not HAVE_TLX:
        print(
            "\n  Variable is set but tlx still did not load. Either this triton\n"
            "  was built without TRITON_EXT_ENABLED (look for that warning above),\n"
            "  or the plugin rejected the triton revision -- each libutlx.so\n"
            "  pins one exact <version>+git<sha>."
        )


def check_apis():
    hdr("TLX APIs")
    if not HAVE_TLX:
        print("  (no tlx module)")
        return False, False
    ok = {}
    for label, names in (("decode needs", DECODE_APIS), ("prefill needs", PREFILL_APIS)):
        present = [n for n in names if hasattr(tlx, n)]
        ok[label] = len(present) == len(names)
        print(f"  {label}: {len(present)}/{len(names)}")
        for n in names:
            print(f"    {'ok     ' if hasattr(tlx, n) else 'MISSING'}  {n}")
    return ok["decode needs"], ok["prefill needs"]


def run_kernel(fn, label, M=64, N=64):
    src = torch.randn(M, N, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    fn[(1,)](src, dst, M=M, N=N, SCALE=2.0)
    torch.cuda.synchronize()
    expected = src * 2.0
    if not torch.equal(dst, expected):
        raise AssertionError(
            f"wrong result, max diff {(dst - expected).abs().max().item()}"
        )
    print(f"  PASS  {label}")


def main():
    if not check_env():
        return 1
    check_plugin()
    decode_ok, prefill_ok = check_apis()

    hdr("kernels")
    failures = 0

    try:
        run_kernel(_plain_tile_scale, "plain triton (control)")
    except Exception as exc:
        print("  FAIL  plain triton (control)")
        traceback.print_exc()
        print("\n  Plain triton cannot compile here, so nothing below is meaningful.")
        if "utlx" in str(exc) or _plugin_installed():
            print(
                "\n  triton-utlx is installed. It hooks itself into triton's\n"
                "  compile pipeline whether or not you set TRITON_PLUGIN_PATHS,\n"
                "  and if libutlx.so never loads (triton built without\n"
                "  TRITON_EXT_ENABLED, or a version/sha mismatch) that hook\n"
                "  points at a missing symbol and EVERY triton kernel fails.\n"
                "  `pip uninstall triton-utlx` restores the environment."
            )
        return 1

    if decode_ok:
        try:
            run_kernel(_tlx_tile_scale, "tlx async_load -> LDS -> local_load")
        except Exception:
            failures += 1
            print("  FAIL  tlx async_load -> LDS -> local_load")
            traceback.print_exc()
    else:
        print("  SKIP  tlx async copy (decode APIs missing)")

    # No kernel for the prefill set: those are layout descriptors whose
    # signatures differ across TLX versions, so presence is the useful signal
    # and a hand-written probe would just test our guess at the API.
    print(
        f"  {'ok  ' if prefill_ok else 'SKIP'}  prefill layout APIs "
        f"({'present' if prefill_ok else 'absent'}; presence check only)"
    )

    hdr("result")
    if not HAVE_TLX:
        print("  TLX not available in this triton")
        return 1
    if failures:
        print(f"  {failures} TLX kernel(s) failed despite the APIs being present")
        return 1
    if not decode_ok:
        print("  TLX present but missing the async-copy set -- KDA decode won't run")
        return 1
    if not prefill_ok:
        missing = ", ".join(n for n in PREFILL_APIS if not hasattr(tlx, n))
        print("  KDA decode: OK")
        print(f"  KDA prefill: needs {missing}")
        print("  -> those landed after fbtriton 3.7.4; build main (3.8.0+fb)")
        return 0
    print("  all TLX APIs present, kernels run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
