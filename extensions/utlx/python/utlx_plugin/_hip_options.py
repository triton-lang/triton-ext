"""Accept the AMD compile options TLX's gfx950 kernels pass, on upstream Triton.

Meta's fork adds fields to ``HIPOptions`` that the TLX op library passes as
launch options, e.g. ``kernel[grid](..., reverse_local_assignment=True)``.
Upstream rejects any of them with "Keyword argument ... was specified but
unrecognised". This module adds them:

* ``reverse_local_assignment``, ``sink_insts_to_avoid_spills``,
  ``regclass_priority_trumps_globalness`` and
  ``disable_unclustered_high_rp_reschedule`` are LLVM codegen flags. They are
  forwarded to the AMDGPU backend exactly as the fork does, by appending them to
  the flags ``make_amdgcn`` hands each ``llvm.translate_*`` call (Triton 3.8)
  or ``compile_amdgpu`` (Triton 3.9).
* ``llvm_cl_opts``, ``"name[=value],..."`` (a bare name means true) or
  ``((name, value), ...)``, adds LLVM command-line options the same way, after
  the codegen flags so they override them, as in the fork.
* ``enable_sched_group_barrier_scheduler`` (with
  ``sched_group_barrier_mfma_per_dwordx4`` and
  ``sched_group_barrier_required_region_count``) selects the fork's TTGIR
  sched-group-barrier annotation pass, which needs matching changes inside
  libtriton's TritonGPU-to-LLVM lowering that a plugin cannot make. The pass
  only reorders instructions -- its output is bitwise identical -- so the
  options are validated and accepted, but have no effect.

The options are fields on a ``HIPOptions`` subclass bound in place of the
original, so they reach the compile-cache key like any other option. Inert on a
Triton that already has them.
"""

import contextvars
import dataclasses

# Option name -> LLVM flag, as in the fork's _get_codegen_flags.
_CODEGEN_FLAGS = {
    "reverse_local_assignment":
    "greedy-reverse-local-assignment",
    "sink_insts_to_avoid_spills":
    "sink-insts-to-avoid-spills",
    "regclass_priority_trumps_globalness":
    "greedy-regclass-priority-trumps-globalness",
    "disable_unclustered_high_rp_reschedule":
    "amdgpu-disable-unclustered-high-rp-reschedule",
}

# The llvm entry points make_amdgcn passes its flag list to, as the fifth
# positional argument (after src/path, triple, arch, features), as
# compile_amdgpu takes it too.
_FLAGGED_LLVM_FUNCS = ("translate_to_mir", "dump_sched_dag",
                       "translate_mir_to_asm", "translate_to_asm")
_FLAGS_ARG = 4

_extra_flags = contextvars.ContextVar("utlx_extra_llvm_flags", default=())


def _with_extras(args, kwargs):
    """``args``/``kwargs`` of a codegen call with the active extras added."""
    args = list(args)
    flags = _extra_flags.get()
    if flags:
        if "flags" in kwargs:
            kwargs["flags"] = [*kwargs["flags"], *flags]
        elif len(args) > _FLAGS_ARG:
            args[_FLAGS_ARG] = [*args[_FLAGS_ARG], *flags]
    return args, kwargs


def _parse_llvm_cl_opts(opts):
    """``llvm_cl_opts`` as ``((name, value), ...)``, as the fork parses it."""
    if not isinstance(opts, str):
        return tuple((str(name), str(value)) for name, value in opts)
    parsed = []
    for opt in opts.split(","):
        name, sep, value = opt.partition("=")
        if name.strip():
            parsed.append((name.strip(), value.strip() if sep else "true"))
    return tuple(parsed)


class _FlagForwardingLLVM:
    """Stand-in for ``libtriton.llvm`` that appends the active extra flags."""

    def __init__(self, llvm):
        self._llvm = llvm

    def __getattr__(self, name):
        fn = getattr(self._llvm, name)
        if name not in _FLAGGED_LLVM_FUNCS:
            return fn

        def call(*args, **kwargs):
            args, kwargs = _with_extras(args, kwargs)
            return fn(*args, **kwargs)

        return call


def install():
    """Add the fork's HIPOptions fields and forward the codegen flags.

    Idempotent; a no-op without the AMD backend or when the fields exist.
    """
    try:
        import triton.backends.amd.compiler as hip
    except ImportError:
        return
    base = hip.HIPOptions
    if "reverse_local_assignment" in base.__dataclass_fields__:
        return

    @dataclasses.dataclass(frozen=True)
    class HIPOptions(base):
        reverse_local_assignment: bool = False
        sink_insts_to_avoid_spills: bool = False
        regclass_priority_trumps_globalness: bool = False
        disable_unclustered_high_rp_reschedule: bool = False
        # Accepted for compatibility; see the module docstring.
        enable_sched_group_barrier_scheduler: bool = False
        sched_group_barrier_mfma_per_dwordx4: int = 4
        sched_group_barrier_required_region_count: int = 0
        llvm_cl_opts: str | tuple[tuple[str, str], ...] = ""

        def __post_init__(self):
            super().__post_init__()
            object.__setattr__(self, "llvm_cl_opts",
                               _parse_llvm_cl_opts(self.llvm_cl_opts))
            if self.sched_group_barrier_mfma_per_dwordx4 <= 0:
                raise ValueError(
                    "sched_group_barrier_mfma_per_dwordx4 must be positive")
            if self.sched_group_barrier_required_region_count < 0:
                raise ValueError("sched_group_barrier_required_region_count "
                                 "must be non-negative")

    HIPOptions.__qualname__ = base.__qualname__
    HIPOptions.__module__ = base.__module__
    # parse_options reads the module global, so rebinding it is enough.
    hip.HIPOptions = HIPOptions

    hip.llvm = _FlagForwardingLLVM(hip.llvm)
    # Triton 3.9's make_amdgcn hands its flags to compile_amdgpu instead.
    if hasattr(hip, "compile_amdgpu"):
        orig_compile_amdgpu = hip.compile_amdgpu

        def compile_amdgpu(*args, **kwargs):
            args, kwargs = _with_extras(args, kwargs)
            return orig_compile_amdgpu(*args, **kwargs)

        hip.compile_amdgpu = compile_amdgpu
    orig_make_amdgcn = hip.HIPBackend.make_amdgcn

    def make_amdgcn(src, metadata, options):
        flags = tuple(flag for name, flag in _CODEGEN_FLAGS.items()
                      if getattr(options, name, False))
        # Last, so they override the flags above, as in the fork.
        flags += tuple(f"{name}={value}"
                       for name, value in options.llvm_cl_opts)
        token = _extra_flags.set(flags)
        try:
            return orig_make_amdgcn(src, metadata, options)
        finally:
            _extra_flags.reset(token)

    hip.HIPBackend.make_amdgcn = staticmethod(make_amdgcn)
