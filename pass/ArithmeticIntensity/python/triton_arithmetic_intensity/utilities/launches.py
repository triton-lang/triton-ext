"""Record kernel launches so their arithmetic-intensity equations can be evaluated.

The equations the pass stores on a ``CompiledKernel`` are evaluated for a
concrete launch: the grid and the launch arguments (see
:func:`~triton_arithmetic_intensity.arithmetic_intensity`). Benchmarks usually
launch kernels from deep inside library code (an ``autograd.Function``, a
wrapper picking one of several kernels, ...), so this module offers two ways
to get hold of the launches that happened:

* :func:`launch` is a drop-in for ``kernel[grid](*args, **kwargs)`` that
  records a :class:`KernelLaunch` -- in :data:`last_launches` (per kernel
  function) and in every active :func:`recording` window. Tensors are
  replaced by placeholders in the recorded arguments (see :func:`detach`) so
  a recording holds no GPU memory; the equations only need the integer
  arguments and the shape/strides of host tensor descriptors.
* :func:`record` / :func:`recording` gather the launches performed while a
  function runs; :meth:`Work.of` sums their work, e.g. for the several
  kernels of an attention backward pass.

:func:`compile_with_config` compiles an autotuned kernel for one fixed
``triton.Config`` without autotuning, as a :class:`ConfigLaunch` whose
equations can be evaluated before (or without) launching it -- the way to
compare a whole tuning space rather than only the configuration the
autotuner picks. :class:`AutotuneRecorder` does that from a ``post_hook`` of
the autotuner, capturing a :class:`ConfigLaunch` for every configuration it
benchmarks, and :class:`IntensityPruner` from its ``early_config_prune``,
evaluating every configuration *before* the autotuner benchmarks them so the
ones of low arithmetic intensity are not timed at all.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Tuple, Type, TypeVar)

from triton import knobs
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources, PTXASError
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor

from ..arithmetic_intensity import (_DEFAULT_MAX_PROGRAMS, _flops_per_byte,
                                    ArithmeticIntensity,
                                    ArithmeticIntensityEquations)

__all__ = [
    "RESOURCE_ERRORS",
    "AutotuneRecorder",
    "ConfigLaunch",
    "IntensityPruner",
    "KernelLaunch",
    "Work",
    "compile_with_config",
    "detach",
    "jit_function",
    "last_launches",
    "launch",
    "record",
    "recording",
]

T = TypeVar("T")

#: Exceptions meaning "this configuration does not fit the device" rather than
#: "the kernel is wrong": raised when compiling (``CompileTimeAssertionFailure``,
#: ``PTXASError``) or launching (``OutOfResources``) a ``triton.Config``.
RESOURCE_ERRORS: Tuple[Type[Exception],
                       ...] = (OutOfResources, CompileTimeAssertionFailure,
                               PTXASError)

# ---------------------------------------------------------------------------
# Launches
# ---------------------------------------------------------------------------


@dataclass
class KernelLaunch:
    """A kernel launch: the compiled kernel plus what it was launched with.

    ``kernel`` is the ``CompiledKernel`` that ran (for an autotuned kernel,
    the one compiled with the winning configuration), whose metadata carries
    the equations; ``grid``, ``args`` and ``kwargs`` are what
    ``kernel[grid](*args, **kwargs)`` was called with. ``kwargs`` may include
    launch options (``num_warps``, ...) and autotuned meta-parameters.
    """

    kernel: Any
    grid: Any
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Name of the kernel function."""
        return str(self.kernel.name)

    def equations(self) -> ArithmeticIntensityEquations:
        """The symbolic equations recorded on the compiled kernel."""
        return ArithmeticIntensityEquations.from_kernel(self.kernel)

    def bound_args(self) -> Dict[str, Any]:
        """Launch arguments bound to parameter names (plus folded ones)."""
        return self.equations().bind_launch_args(*self.args, **self.kwargs)

    def work(self,
             max_programs: int = _DEFAULT_MAX_PROGRAMS) -> ArithmeticIntensity:
        """Evaluate the equations for this launch's grid and arguments."""
        return self.equations().evaluate(self.grid,
                                         *self.args,
                                         max_programs=max_programs,
                                         **self.kwargs)

    def detached(self) -> KernelLaunch:
        """A copy whose arguments hold no tensors (see :func:`detach`)."""
        return replace(self,
                       args=tuple(detach(a) for a in self.args),
                       kwargs={
                           k: detach(v)
                           for k, v in self.kwargs.items()
                       })


def detach(value: Any) -> Any:
    """Replace tensors by placeholders so a recorded launch holds no memory.

    The equations only reference integer arguments (and, for host tensor
    descriptors, their ``shape`` / ``strides``); pointer arguments never
    appear in them. ``torch.Tensor`` becomes ``None``, a
    ``TensorDescriptor`` keeps only its ``shape`` and ``strides``, and tuples
    are detached member by member. Everything else is returned unchanged.
    """
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return None
    if isinstance(value, TensorDescriptor):
        return SimpleNamespace(shape=list(value.shape),
                               strides=list(value.strides))
    if isinstance(value, tuple):
        return tuple(detach(v) for v in value)
    return value


#: Last recorded launch of each kernel function, by name.
last_launches: Dict[str, KernelLaunch] = {}
#: Launch lists of the active :func:`recording` windows (innermost last).
_recording: List[List[KernelLaunch]] = []


def launch(kernel: Any, grid: Any, *args: Any, **kwargs: Any) -> Any:
    """``kernel[grid](*args, **kwargs)`` that records the launch.

    ``kernel`` is a ``JITFunction`` or an ``Autotuner`` wrapping one. Returns
    the ``CompiledKernel`` that ran, like the launch itself does. The
    recorded :class:`KernelLaunch` (with detached arguments) is stored in
    :data:`last_launches` and appended to every active :func:`recording`.
    """
    compiled = kernel[grid](*args, **kwargs)
    entry = KernelLaunch(compiled, grid, args, kwargs).detached()
    last_launches[entry.name] = entry
    for window in _recording:
        window.append(entry)
    return compiled


@contextmanager
def recording() -> Iterator[List[KernelLaunch]]:
    """Context manager collecting the launches made through :func:`launch`.

    ::

        with tai.recording() as launches:
            attention(q, k, v)
        work = tai.Work.of(launches)

    Windows nest: an inner window sees only the launches made while it is
    active, an outer one also sees those of the inner windows.
    """
    window: List[KernelLaunch] = []
    _recording.append(window)
    try:
        yield window
    finally:
        # By identity: nested windows can compare equal (same launches).
        for i in range(len(_recording) - 1, -1, -1):
            if _recording[i] is window:
                del _recording[i]
                break


def record(fn: Callable[[], T]) -> Tuple[T, List[KernelLaunch]]:
    """Run ``fn()`` and return its result with the launches it performed."""
    with recording() as launches:
        result = fn()
    return result, launches


# ---------------------------------------------------------------------------
# Aggregated work
# ---------------------------------------------------------------------------


@dataclass
class Work:
    """Pass-derived work summed over several launches.

    A high-level operation (an attention step, a matmul with a preprocessing
    kernel) may launch several kernels; :meth:`of` evaluates each recorded
    launch and sums the FLOPs, bytes (in total and split into ``load_bytes``
    / ``store_bytes``) and programs, keeping the per-launch
    :class:`~triton_arithmetic_intensity.ArithmeticIntensity` results in
    ``per_launch`` (as ``(kernel name, work)`` pairs, in launch order).
    """

    flops: int = 0
    bytes: int = 0
    load_bytes: int = 0
    store_bytes: int = 0
    num_programs: int = 0
    per_launch: List[Tuple[str,
                           ArithmeticIntensity]] = field(default_factory=list)

    @classmethod
    def of(cls,
           launches: Any,
           max_programs: int = _DEFAULT_MAX_PROGRAMS) -> Work:
        """Sum the work of an iterable of :class:`KernelLaunch`."""
        total = cls()
        for kernel_launch in launches:
            total.add(kernel_launch.name,
                      kernel_launch.work(max_programs=max_programs))
        return total

    def add(self, name: str, work: ArithmeticIntensity) -> None:
        """Account for one more launch of kernel ``name``."""
        self.flops += work.flops
        self.bytes += work.bytes
        self.load_bytes += work.load_bytes
        self.store_bytes += work.store_bytes
        self.num_programs += work.num_cores
        self.per_launch.append((name, work))

    @property
    def per_kernel(self) -> Dict[str, Work]:
        """The work broken down by kernel function name."""
        result: Dict[str, Work] = {}
        for name, work in self.per_launch:
            result.setdefault(name, Work()).add(name, work)
        return result

    @property
    def intensity(self) -> float:
        """Arithmetic intensity in FLOPs per byte.

        FLOPs and no bytes is infinite. No FLOPs and no bytes is zero.
        """
        return _flops_per_byte(self.flops, self.bytes)

    def tflops(self, ms: float) -> float:
        """Achieved TFLOP/s given a measured runtime in milliseconds."""
        return self.flops * 1e-12 / (ms * 1e-3)

    def gbps(self, ms: float) -> float:
        """Achieved GB/s given a measured runtime in milliseconds."""
        return self.bytes * 1e-9 / (ms * 1e-3)


# ---------------------------------------------------------------------------
# Compiling one configuration of an autotuned kernel
# ---------------------------------------------------------------------------


def jit_function(kernel: Any) -> JITFunction:
    """The ``JITFunction`` behind ``kernel`` (unwrapping ``Autotuner`` etc.)."""
    fn = kernel
    while not isinstance(fn, JITFunction):
        inner = getattr(fn, "fn", None)
        if inner is None or inner is fn:
            raise TypeError(f"{kernel!r} does not wrap a JITFunction")
        fn = inner
    return fn


@dataclass
class ConfigLaunch(KernelLaunch):
    """A kernel compiled for one fixed ``triton.Config``, not yet launched.

    Produced by :func:`compile_with_config` (directly or through an
    :class:`AutotuneRecorder`). The equations are in the compiled kernel's
    metadata as soon as it is compiled, whether or not the configuration fits
    the device: a configuration can compile fine and still raise
    ``OutOfResources`` when launched with :meth:`run`.
    """

    #: The ``JITFunction`` the kernel was compiled from.
    fn: Any = None
    #: The configuration it was compiled with.
    config: Any = None

    def run(self) -> Any:
        """Launch the compiled kernel (a cache hit); returns it."""
        return self.fn[self.grid](*self.args, **self.kwargs)


def compile_with_config(kernel: Any, config: Any, grid: Any, *args: Any,
                        **kwargs: Any) -> ConfigLaunch:
    """Compile ``kernel`` for ``config`` without autotuning.

    Mirrors what the autotuner does when it benchmarks ``config``: its
    meta-parameters and launch options (``config.all_kwargs()``) are added to
    ``kwargs``, its ``pre_hook`` (if any) is called with the bound arguments
    -- e.g. to set the block shape of host tensor descriptors -- and the
    kernel is compiled with ``JITFunction.warmup``. ``kernel`` may be the
    ``Autotuner`` itself or the underlying ``JITFunction``; a ``triton.Config``
    can come from ``kernel.configs``. Kernel parameters may be passed
    positionally or by keyword.
    """
    fn = jit_function(kernel)
    conflicts = kwargs.keys() & config.kwargs.keys()
    if conflicts:
        raise ValueError(
            f"Conflicting meta-parameters: {', '.join(conflicts)}."
            " Make sure that you don't re-define auto-tuned symbols.")
    meta = dict(kwargs, **config.all_kwargs())
    if config.pre_hook is not None:
        config.pre_hook({**dict(zip(fn.arg_names, args)), **meta})
    compiled = fn.warmup(*args, grid=grid, **meta)
    return ConfigLaunch(compiled, grid, args, meta, fn=fn, config=config)


#: Launch options the autotuner adds to the arguments its hooks receive.
_LAUNCH_OPTIONS = ("grid", "warmup")


def _hook_params(
        nargs: Mapping[str, Any],
        exclude: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Kernel parameters among the arguments an autotuner hook receives.

    Drops the launch options (``grid``, ``warmup``) and the keys of
    ``exclude`` (a configuration's meta-parameters), leaving what
    :func:`compile_with_config` takes as keyword arguments.
    """
    exclude = exclude or {}
    return {
        k: v
        for k, v in nargs.items()
        if k not in exclude and k not in _LAUNCH_OPTIONS
    }


class AutotuneRecorder:
    """Autotuner ``post_hook`` capturing a :class:`ConfigLaunch` per config.

    While the autotuner benchmarks its configurations it calls its
    ``post_hook`` after every launch with the full argument dictionary (kernel
    arguments, the configuration's meta-parameters and launch options, and
    the ``grid``). This hook recognizes the ``triton.Config`` being
    benchmarked and, the first time it sees it, calls
    :func:`compile_with_config` -- a cache hit, since the autotuner has just
    compiled that configuration -- so that afterwards :attr:`launches` holds
    the compiled kernel (and thus the equations) of every configuration the
    autotuner tried, and :attr:`errors` the exception of every configuration
    that did not compile or fit the device.

    Pass the recorder as ``post_hook`` to ``triton.autotune`` and
    :meth:`attach` the resulting ``Autotuner`` (needed to identify configs)::

        RECORDER = tai.AutotuneRecorder()

        @triton.autotune(configs=..., key=[...], post_hook=RECORDER)
        @triton.jit
        def kernel(...): ...

        RECORDER.attach(kernel)

    or :meth:`install` it on an existing autotuner (chaining onto whatever
    ``post_hook`` it has). The hook only runs for configurations the
    autotuner actually benchmarks: not when tuning results come from its disk
    cache, nor for configurations removed by ``prune_configs_by``.
    """

    def __init__(self, kernel: Any = None) -> None:
        self.kernel = kernel
        #: ``{triton.Config: ConfigLaunch}`` of every compiled configuration.
        self.launches: Dict[Any, ConfigLaunch] = {}
        #: ``{triton.Config: exception}`` of every configuration that failed
        #: to compile or to launch (see :data:`RESOURCE_ERRORS`).
        self.errors: Dict[Any, BaseException] = {}
        self._previous: Any = None

    def attach(self, kernel: Any) -> AutotuneRecorder:
        """Set the ``Autotuner`` whose configurations are recorded."""
        self.kernel = kernel
        return self

    def install(self, kernel: Any) -> AutotuneRecorder:
        """Become ``kernel.post_hook``, still calling the hook it replaces."""
        self.attach(kernel)
        if kernel.post_hook is not self:
            self._previous = kernel.post_hook
        kernel.post_hook = self
        kernel.user_defined_post_hook = True
        return self

    def clear(self) -> None:
        self.launches.clear()
        self.errors.clear()

    def config_of(self, nargs: Mapping[str, Any]) -> Any:
        """The attached autotuner's ``Config`` whose kwargs are in ``nargs``."""
        if self.kernel is None:
            raise RuntimeError("AutotuneRecorder: attach() the autotuner "
                               "before it runs")
        for config in self.kernel.configs:
            if all(k in nargs and nargs[k] == v
                   for k, v in config.all_kwargs().items()):
                return config
        raise LookupError("no configuration matches the benchmarked "
                          "arguments")

    def __call__(self,
                 nargs: Mapping[str, Any],
                 exception: Any = None,
                 **_: Any) -> None:
        if self._previous is not None:
            self._previous(nargs, exception)
        config = self.config_of(nargs)
        if config in self.errors:
            return
        if exception is not None and not isinstance(exception, OutOfResources):
            # Did not compile (e.g. a compile-time assertion); nothing to
            # record but the failure.
            self.errors[config] = exception
            return
        if config not in self.launches:
            params = _hook_params(nargs, exclude=config.all_kwargs())
            try:
                self.launches[config] = compile_with_config(
                    self.kernel, config, nargs.get("grid"), **params)
            except RESOURCE_ERRORS as exc:
                self.errors[config] = exc
                return
        if exception is not None:  # compiled, but does not fit the device
            self.errors[config] = exception


class IntensityPruner:
    """Autotuner ``early_config_prune`` dropping low-intensity configurations.

    Before benchmarking a tuning key the autotuner calls
    ``early_config_prune(configs, named_args, **kwargs)`` with the kernel
    arguments of the launch being tuned. This hook compiles every
    ``triton.Config`` for that launch (:func:`compile_with_config`, which
    also runs the configuration's ``pre_hook``), evaluates the pass equations
    for the launch's grid and arguments, and returns only the configurations
    whose arithmetic intensity reaches :attr:`min_intensity` FLOP/byte -- the
    autotuner never times the others. The kept configurations are compiled
    anyway when benchmarked (the autotuner's compilations become cache hits);
    only the pruned ones cost an extra compilation, the price of knowing their
    work without running them. Configurations that fail to compile (see
    :data:`RESOURCE_ERRORS`) are dropped as well.

    After the autotuner ran, :attr:`launches` holds the :class:`ConfigLaunch`
    of every configuration that compiled (kept or pruned), :attr:`work` its
    evaluated :class:`~triton_arithmetic_intensity.ArithmeticIntensity`,
    :attr:`pruned` the configurations that were dropped for their intensity
    and :attr:`errors` the exception of every configuration that did not
    compile. They describe the last tuning key pruned; the hook only runs for
    tuning keys the autotuner has not tuned (or loaded from its disk cache)
    yet. At least :attr:`keep_at_least` configurations are always kept (the
    most intense ones), since the autotuner requires one.

    Pass the pruner in ``prune_configs_by`` and :meth:`attach` the resulting
    ``Autotuner`` (needed to compile its configurations)::

        PRUNER = tai.IntensityPruner(min_intensity=40)

        @triton.autotune(configs=..., key=[...],
                         prune_configs_by={"early_config_prune": PRUNER})
        @triton.jit
        def kernel(...): ...

        PRUNER.attach(kernel)

    or :meth:`install` it on an existing autotuner, where it runs after the
    ``early_config_prune`` the autotuner already has. :attr:`min_intensity`
    may be changed between launches; ``None`` disables the hook (the
    configurations pass through without being compiled or evaluated).
    Combine with an :class:`AutotuneRecorder` as ``post_hook`` to also learn
    which kept configurations the autotuner could run.
    """

    def __init__(self,
                 min_intensity: Optional[float],
                 kernel: Any = None,
                 *,
                 keep_at_least: int = 1,
                 max_programs: int = _DEFAULT_MAX_PROGRAMS) -> None:
        #: Threshold in FLOP/byte; configurations below it are pruned.
        self.min_intensity = min_intensity
        self.kernel = kernel
        #: Never return fewer configurations than this (most intense first).
        self.keep_at_least = keep_at_least
        self.max_programs = max_programs
        #: ``{triton.Config: ConfigLaunch}`` of every compiled configuration.
        self.launches: Dict[Any, ConfigLaunch] = {}
        #: ``{triton.Config: ArithmeticIntensity}`` of every compiled one.
        self.work: Dict[Any, ArithmeticIntensity] = {}
        #: Configurations dropped for their intensity, in tuning-space order.
        self.pruned: List[Any] = []
        #: ``{triton.Config: exception}`` of configurations that did not
        #: compile.
        self.errors: Dict[Any, BaseException] = {}
        self._previous: Any = None

    def attach(self, kernel: Any) -> IntensityPruner:
        """Set the ``Autotuner`` whose configurations are pruned."""
        self.kernel = kernel
        return self

    def install(self, kernel: Any) -> IntensityPruner:
        """Become ``kernel.early_config_prune``, after the hook it replaces."""
        self.attach(kernel)
        if kernel.early_config_prune is not self:
            self._previous = kernel.early_config_prune
        kernel.early_config_prune = self
        return self

    def clear(self) -> None:
        self.launches.clear()
        self.work.clear()
        self.pruned.clear()
        self.errors.clear()

    def evaluate(self, configs: Any, grid: Any,
                 **params: Any) -> Dict[Any, ArithmeticIntensity]:
        """Compile and evaluate ``configs`` for a launch, filling the state.

        ``params`` are the kernel parameters by name (as
        :func:`compile_with_config` takes them). Returns :attr:`work`.
        """
        if self.kernel is None:
            raise RuntimeError("IntensityPruner: attach() the autotuner "
                               "before it runs")
        self.clear()
        for config in configs:
            try:
                launch = compile_with_config(self.kernel, config, grid,
                                             **params)
            except RESOURCE_ERRORS as exc:
                self.errors[config] = exc
                continue
            self.launches[config] = launch
            self.work[config] = launch.work(max_programs=self.max_programs)
        return self.work

    def select(self, configs: Any) -> List[Any]:
        """The configurations of ``configs`` to keep, given :attr:`work`.

        Keeps those (that compiled and) reach :attr:`min_intensity`, in
        their original order, topped up to :attr:`keep_at_least` with the
        most intense ones; the rest go to :attr:`pruned`.
        """
        threshold = self.min_intensity or 0.0
        compiled = [c for c in configs if c in self.work]
        kept = {c for c in compiled if self.work[c].intensity >= threshold}
        if len(kept) < self.keep_at_least:
            ranked = sorted(compiled,
                            key=lambda c: self.work[c].intensity,
                            reverse=True)
            kept.update(ranked[:self.keep_at_least])
        self.pruned = [c for c in compiled if c not in kept]
        return [c for c in compiled if c in kept]

    def __call__(self, configs: Any, named_args: Mapping[str, Any],
                 **kwargs: Any) -> List[Any]:
        if self._previous is not None:
            configs = self._previous(configs, named_args, **kwargs)
        if self.min_intensity is None:
            return list(configs)
        self.evaluate(configs, kwargs.get("grid"), **named_args,
                      **_hook_params(kwargs))
        kept = self.select(configs)
        if knobs.autotuning.print:
            for config in self.pruned:
                print(f"Pruning config {config}: arithmetic intensity "
                      f"{self.work[config].intensity:.1f} FLOP/B < "
                      f"{self.min_intensity}")
            for config, exc in self.errors.items():
                print(f"Pruning config {config}: {exc}")
        return kept
