"""Reporting for `tlx.async_tasks(...)` options the host Triton cannot act on.

The options are recorded as attributes on the generated
`ttg.warp_specialize` op, but acting on them happens outside this plugin:

* `tlx.exclusive` and `tlx.no_ending_cluster_sync` are read by the TLX Fixup
  pass, which turns them into module markers that only Meta's NVGPUToLLVM
  lowering understands (`hasUserPostWsSync`, `setHasSingleWarpSpecialize`).
* `tlx.mbarrier_try_wait_suspend_ns` is read by Meta's BarrierOpToLLVM.

None of those live in the TLX dialect, so on a stock Triton wheel the
attributes are inert. Each dropped option is conservative rather than unsound
-- the compiler keeps its own cluster sync, does not switch to single-warp-
specialize codegen, and spins instead of suspending on mbarrier waits -- but a
kernel tuned with them will not perform as intended, and silently ignoring a
correctness-adjacent request is worse than saying so. Warn once per process.

Set ``UTLX_SILENCE_WARP_SPEC_OPTION_WARNINGS=1`` to suppress.
"""

import os
import warnings

# Options whose effect is implemented entirely outside the plugin.
HOST_IMPLEMENTED_OPTIONS = (
    "exclusive",
    "no_ending_cluster_sync",
    "mbarrier_try_wait_suspend_ns",
    "less_reg_mma",
)

_warned = set()


def host_acts_on_options():
    """True when the host Triton is a build whose lowering reads the markers."""
    from .host_caps import host_is_meta_intree
    return host_is_meta_intree()


def warn_if_unsupported(requested):
    """Warn once for each requested option the host will ignore.

    `requested` maps option name -> value as written by the kernel; options
    left at their default are not reported.
    """
    if os.environ.get("UTLX_SILENCE_WARP_SPEC_OPTION_WARNINGS"):
        return
    if host_acts_on_options():
        return

    unsupported = sorted(name for name, value in requested.items()
                         if value not in (False, None) and name in HOST_IMPLEMENTED_OPTIONS and name not in _warned)
    if not unsupported:
        return
    _warned.update(unsupported)
    warnings.warn(
        "tlx.async_tasks option(s) " + ", ".join(unsupported) +
        " are recorded on the warp_specialize op but this Triton build does not"
        " act on them; they need the lowering support that ships with Meta's"
        " in-tree TLX. Compilation continues with the compiler's default"
        " behaviour (conservative, not unsound). Set"
        " UTLX_SILENCE_WARP_SPEC_OPTION_WARNINGS=1 to silence.",
        RuntimeWarning,
        stacklevel=3,
    )
