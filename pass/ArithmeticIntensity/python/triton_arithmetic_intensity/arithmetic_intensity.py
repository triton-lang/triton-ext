"""Arithmetic-intensity introspection for compiled Triton kernels.

The ``triton-arithmetic-intensity`` pass annotates every ``tt.func`` argument
with string attributes describing the *per-program* (per-CTA) work the kernel
performs against that argument:

  * ``tai.load_bytes``:  bytes loaded per program by loads rooted at the
    argument.
  * ``tai.store_bytes``: bytes stored per program by stores rooted at the
    argument.
  * ``tai.op_count``:    op count (FLOPs) feeding the stores rooted at the
    argument.

The equations are symbolic, written over ``args[i]`` (the i-th ``tt.func``
argument), ``program_id[i]`` and ``num_programs[i]``, with ``/`` and ``%``
denoting floor-division and modulo.

:mod:`triton_arithmetic_intensity.custom_stages` runs the pass at the end of
every ``ttir`` compilation stage and records the equations of each function
under ``metadata["arithmetic_intensity"]``, which Triton persists alongside the
compiled kernel. This module turns those equations into numbers:

  * :class:`ArithmeticIntensityEquations` wraps the recorded equations of one
    kernel (plus the argument layout needed to bind ``args[i]`` to launch
    arguments) and evaluates them for a concrete launch.
  * :class:`ArithmeticIntensityListener` is a Triton ``CompilationListener``
    that gathers an :class:`ArithmeticIntensityEquations` per kernel function
    as kernels are compiled (or loaded from the cache).
  * :func:`arithmetic_intensity` evaluates the equations stored on a
    ``CompiledKernel`` for a given grid and launch arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

#: Key under which the equations are stored in kernel compilation metadata.
METADATA_KEY = "arithmetic_intensity"
#: Keys of the per-argument equation dictionaries.
LOAD_BYTES = "load_bytes"
STORE_BYTES = "store_bytes"
OP_COUNT = "op_count"

#: ``{function name: {ttir arg index:
#:     {"load_bytes": eq, "store_bytes": eq, "op_count": eq}}}``.
FunctionEquations = Dict[str, Dict[int, Dict[str, str]]]

_DEFAULT_MAX_PROGRAMS = 1 << 20

# ---------------------------------------------------------------------------
# Equation evaluation
# ---------------------------------------------------------------------------


class _BoundArgs:
    """``args[i]`` accessor used while evaluating an equation.

    Raises a descriptive error when an equation references an argument that
    was not bound to a value (e.g. because the launch arguments were not
    passed to :meth:`ArithmeticIntensityEquations.evaluate`).
    """

    def __init__(self, values: Mapping[int, Any], names: Mapping[int, str]):
        self._values = values
        self._names = names

    def __getitem__(self, index: int) -> Any:
        try:
            return self._values[index]
        except KeyError:
            name = self._names.get(index)
            what = f"args[{index}]" + (f" (parameter {name!r})"
                                       if name else "")
            raise ValueError(
                f"equation references {what} but no value was bound for it; "
                "pass the launch arguments or `arg_values`") from None


def _eval_equation(equation: str, args: _BoundArgs, program_id: Tuple[int, int,
                                                                      int],
                   num_programs: Tuple[int, int, int]) -> int:
    """Evaluate one equation for a single program.

    The equation is plain Python once ``args`` / ``program_id`` /
    ``num_programs`` are in scope; the only fix-up is mapping the affine
    printer's ``/`` (floor-division) to Python's ``//`` so op counts stay
    integral (``%`` already matches Python's modulo).
    """
    expr = equation.replace("/", "//")
    try:
        return eval(expr, {"__builtins__": {}}, {
            "args": args,
            "program_id": program_id,
            "num_programs": num_programs,
        })
    except NameError as exc:
        # The pass binds values it cannot express (e.g. results of
        # unsupported integer ops) to opaque symbols `s<N>`.
        raise ValueError(f"equation {equation!r} contains an opaque symbol "
                         "the pass could not resolve to a kernel argument "
                         f"({exc})") from None


def _sum_equation_over_grid(equation: Optional[str], args: _BoundArgs,
                            grid: Tuple[int, int,
                                        int], max_programs: int) -> int:
    """Sum a per-program equation across every program in ``grid``.

    Equations that do not reference ``program_id`` describe uniform
    per-program work, so they are evaluated once and multiplied by the number
    of programs. Otherwise every program is enumerated (bounded by
    ``max_programs``).
    """
    if not equation:
        return 0
    gx, gy, gz = grid
    n_programs = gx * gy * gz
    if "program_id" not in equation:
        return _eval_equation(equation, args, (0, 0, 0), grid) * n_programs
    if n_programs > max_programs:
        raise ValueError(
            f"equation {equation!r} depends on program_id and the grid has "
            f"{n_programs} programs (> max_programs={max_programs}); raise "
            "max_programs to evaluate it.")
    total = 0
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                total += _eval_equation(equation, args, (x, y, z), grid)
    return total


def _flops_per_byte(flops: int, nbytes: int) -> float:
    """FLOPs per byte.

    FLOPs and no bytes is infinite. No FLOPs and no bytes is zero, so an
    empty kernel does not outrank real work.
    """
    if nbytes:
        return flops / nbytes
    return float("inf") if flops else 0.0


def _normalize_grid(grid: Any,
                    bound_args: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Resolve ``grid`` the way a kernel launch does and pad it to 3-D."""
    if callable(grid):
        grid = grid(dict(bound_args))
    if isinstance(grid, int):
        grid = (grid, )
    grid = tuple(int(g) for g in grid)
    if not 1 <= len(grid) <= 3:
        raise ValueError("grid must have between 1 and 3 dimensions")
    return grid + (1, ) * (3 - len(grid))  # type: ignore[return-value]


@dataclass
class ArithmeticIntensity:
    """Required work for a kernel launch.

    ``flops`` and ``bytes`` are the overall op count and total bytes moved
    across the whole grid, ``load_bytes`` / ``store_bytes`` split the latter
    into bytes read and written; the ``*_per_core`` properties give the
    per-program (per-CTA) requirement. Pair these with a ``do_bench`` time
    (in ms) via :meth:`tflops` / :meth:`gbps` to report achieved throughput.

    ``per_arg`` breaks the totals down by kernel parameter (keyed by parameter
    name when known, ``args[i]`` otherwise) into ``flops``, ``bytes``,
    ``load_bytes`` and ``store_bytes``.
    """

    flops: int
    bytes: int
    grid: Tuple[int, int, int]
    load_bytes: int = 0
    store_bytes: int = 0
    per_arg: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def num_cores(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2]

    @property
    def flops_per_core(self) -> float:
        return self.flops / self.num_cores if self.num_cores else 0.0

    @property
    def bytes_per_core(self) -> float:
        return self.bytes / self.num_cores if self.num_cores else 0.0

    @property
    def load_bytes_per_core(self) -> float:
        return self.load_bytes / self.num_cores if self.num_cores else 0.0

    @property
    def store_bytes_per_core(self) -> float:
        return self.store_bytes / self.num_cores if self.num_cores else 0.0

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
# Per-kernel equations
# ---------------------------------------------------------------------------

_MISSING = object()


def _member(value: Any, index: int) -> Any:
    """``value[index]`` for tuple-like launch values, ``_MISSING`` otherwise."""
    if value is _MISSING or isinstance(value, (str, bytes)):
        return _MISSING
    try:
        return value[index]
    except (TypeError, IndexError, KeyError):
        return _MISSING


def _ir_leaves(ty: Any, value: Any, label: str, path: Tuple[int, ...],
               constants: Mapping[Tuple[int, ...],
                                  Any], out: List[Tuple[str, Any]]) -> None:
    """Append the ``tt.func`` arguments a parameter of type ``ty`` expands to.

    Mirrors ``base_type._flatten_ir_types`` structurally: ``constexpr``
    parameters (annotated or specialized, i.e. present in ``constants``)
    produce no argument, tuples flatten member by member, host tensor
    descriptors expand to ``(descriptor, *shape, *strides)``, and anything
    else is a single argument. ``value`` may be ``_MISSING``.
    """
    if path in constants or type(ty).__name__ == "constexpr_type":
        return
    members = getattr(ty, "types", None)
    if members is not None:  # tuple_type (and named tuples / aggregates)
        fields = getattr(ty, "fields", None) or []
        for j, member_ty in enumerate(members):
            name = f"{label}.{fields[j]}" if j < len(
                fields) else f"{label}[{j}]"
            _ir_leaves(member_ty, _member(value, j), name, path + (j, ),
                       constants, out)
        return
    out.append((label, value))
    shape_ty = getattr(ty, "shape_type", None)
    strides_ty = getattr(ty, "strides_type", None)
    if shape_ty is not None and strides_ty is not None:  # host descriptor
        _ir_leaves(shape_ty, getattr(value, "shape", _MISSING),
                   f"{label}.shape", path + (1, ), constants, out)
        _ir_leaves(strides_ty, getattr(value, "strides", _MISSING),
                   f"{label}.strides", path + (2, ), constants, out)


def _value_leaves(value: Any, label: str, path: Tuple[int, ...],
                  constants: Mapping[Tuple[int, ...],
                                     Any], out: List[Tuple[str, Any]]) -> None:
    """Fallback flattening (no compile-time type): expand tuples by value."""
    if path in constants:
        return
    if isinstance(value, tuple):
        for j, member in enumerate(value):
            _value_leaves(member, f"{label}[{j}]", path + (j, ), constants,
                          out)
        return
    out.append((label, value))


@dataclass
class ArithmeticIntensityEquations:
    """The symbolic equations recorded for one compiled kernel.

    ``functions`` holds the equations of every ``tt.func`` in the kernel's
    TTIR module, keyed by symbol name, then by ``tt.func`` argument index.
    ``arg_names`` / ``arg_types`` / ``constants`` describe the kernel's
    Python signature as it was compiled -- the parameter types and the
    (``constexpr`` or specialized) parameters that were folded into the IR --
    which is what is needed to map ``args[i]`` back to launch arguments.
    """

    #: Entry kernel name (``metadata["name"]``).
    name: str
    functions: FunctionEquations = field(default_factory=dict)
    #: Python parameter names of the kernel, in declaration order.
    arg_names: List[str] = field(default_factory=list)
    #: Parameters folded into the IR (``ASTSource.constants``): paths of
    #: parameter indices (``(i,)`` or ``(i, j, ...)`` for tuple members).
    constants: Dict[Tuple[int, ...], Any] = field(default_factory=dict)
    #: Compile-time parameter types (``ASTSource.signature``): Triton type
    #: strings (or nested tuples of them) keyed by parameter name.
    arg_types: Dict[str, Any] = field(default_factory=dict)
    #: Cache hash of the compilation, when known.
    hash: Optional[str] = None
    #: ``inspect.Signature`` of the kernel, used to bind launch arguments.
    signature: Any = field(default=None, repr=False, compare=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_metadata(
            cls,
            metadata: Mapping[str, Any],
            src: Any = None) -> Optional[ArithmeticIntensityEquations]:
        """Build from compilation ``metadata`` (and its ``ASTSource``).

        Returns ``None`` when the metadata carries no arithmetic-intensity
        entry (e.g. the pass was not installed when the kernel was compiled).
        """
        entry = metadata.get(METADATA_KEY)
        if entry is None:
            return None
        functions: FunctionEquations = {}
        for func_name, per_arg in entry.items():
            functions[func_name] = {
                int(index): dict(equations)
                for index, equations in per_arg.items()
            }
        fn = getattr(src, "fn", None)
        constants = getattr(src, "constants", None) or {}
        arg_types = getattr(src, "signature", None) or {}
        return cls(
            name=str(metadata.get("name") or getattr(src, "name", "")),
            functions=functions,
            arg_names=list(getattr(fn, "arg_names", []) or []),
            constants={
                tuple(k): v
                for k, v in constants.items()
            },
            arg_types=dict(arg_types)
            if isinstance(arg_types, Mapping) else {},
            hash=metadata.get("hash"),
            signature=getattr(fn, "signature", None),
        )

    @classmethod
    def from_kernel(cls, kernel: Any) -> ArithmeticIntensityEquations:
        """Build from a ``CompiledKernel`` (raises if nothing was recorded)."""
        metadata = getattr(kernel, "metadata", None)
        if metadata is None:
            raise ValueError("kernel does not expose compilation metadata")
        result = cls.from_metadata(metadata._asdict(),
                                   getattr(kernel, "src", None))
        if result is None:
            raise ValueError(
                f"no {METADATA_KEY!r} entry in the metadata of kernel "
                f"{getattr(kernel, 'name', kernel)!r}; import "
                "`triton_arithmetic_intensity` before compiling it")
        return result

    # -- accessors ------------------------------------------------------------

    @property
    def entry(self) -> Dict[int, Dict[str, str]]:
        """Equations of the entry kernel function, keyed by ``args`` index."""
        if self.name in self.functions:
            return self.functions[self.name]
        if len(self.functions) == 1:
            return next(iter(self.functions.values()))
        if not self.functions:
            return {}
        raise KeyError(f"entry function {self.name!r} not found among "
                       f"{sorted(self.functions)}")

    def load_bytes(self, index: int) -> Optional[str]:
        """The ``tai.load_bytes`` equation of ``args[index]``, if any."""
        return self.entry.get(index, {}).get(LOAD_BYTES)

    def store_bytes(self, index: int) -> Optional[str]:
        """The ``tai.store_bytes`` equation of ``args[index]``, if any."""
        return self.entry.get(index, {}).get(STORE_BYTES)

    def op_count(self, index: int) -> Optional[str]:
        """The ``tai.op_count`` equation of ``args[index]``, if any."""
        return self.entry.get(index, {}).get(OP_COUNT)

    # -- argument binding -------------------------------------------------------

    @property
    def folded_args(self) -> Dict[str, Any]:
        """Values of whole parameters folded into the IR, by parameter name.

        These are the ``constexpr`` parameters and the integer parameters
        Triton specialized (e.g. strides equal to 1) when the kernel was
        compiled, so they need not be repeated when evaluating.
        """
        result: Dict[str, Any] = {}
        for path, value in self.constants.items():
            if len(path) == 1 and path[0] < len(self.arg_names):
                result[self.arg_names[path[0]]] = value
        return result

    def bind_launch_args(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Bind launch arguments to parameter names.

        Defaults and the compile-time :attr:`folded_args` fill in anything
        not passed, so the result resembles the ``bound_args`` Triton hands
        to a callable grid (e.g. it contains the autotuned ``BLOCK_*``).
        Keyword arguments that are not kernel parameters (launch options such
        as ``num_warps`` or ``num_stages``) are ignored.
        """
        if self.arg_names:
            kwargs = {k: v for k, v in kwargs.items() if k in self.arg_names}
        result: Dict[str, Any]
        if self.signature is not None:
            bound = self.signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            result = dict(bound.arguments)
        else:
            if len(args) > len(self.arg_names):
                raise TypeError(
                    "too many positional arguments for the kernel signature")
            result = dict(zip(self.arg_names, args))
            for name, value in kwargs.items():
                if name not in self.arg_names:
                    raise TypeError(f"unexpected keyword argument {name!r}")
                result[name] = value
        for name, value in self.folded_args.items():
            result.setdefault(name, value)
        return result

    def ttir_args(
            self,
            bound: Mapping[str, Any]) -> Tuple[Dict[int, Any], Dict[int, str]]:
        """Map ``tt.func`` argument indices to launch values and names.

        The TTIR arguments are the kernel's parameters in declaration order,
        minus those folded into the IR (``constants``), flattened the way
        Triton flattens them (tuples member by member, host tensor descriptors
        to ``descriptor, *shape, *strides``); see :func:`_ir_leaves`. Only
        parameters present in ``bound`` get a value; every other leaf still
        consumes an index.
        """
        try:
            from triton.language import str_to_ty
        except ImportError:  # pragma: no cover - triton is always present
            str_to_ty = None
        leaves: List[Tuple[str, Any]] = []
        for i, name in enumerate(self.arg_names):
            value = bound.get(name, _MISSING)
            type_str = self.arg_types.get(name)
            if type_str is not None and str_to_ty is not None:
                _ir_leaves(str_to_ty(type_str, None), value, name, (i, ),
                           self.constants, leaves)
            else:
                _value_leaves(value, name, (i, ), self.constants, leaves)
        values = {
            index: value
            for index, (_, value) in enumerate(leaves) if value is not _MISSING
        }
        names = {index: label for index, (label, _) in enumerate(leaves)}
        return values, names

    # -- evaluation -------------------------------------------------------------

    def evaluate(
        self,
        grid: Any,
        *args: Any,
        arg_values: Optional[Union[Mapping[int, Any], Sequence[Any]]] = None,
        max_programs: int = _DEFAULT_MAX_PROGRAMS,
        **kwargs: Any,
    ) -> ArithmeticIntensity:
        """Evaluate the entry function's equations for a concrete launch.

        :param grid: The launch grid: an int, a 1-3 element sequence, or a
            callable taking the bound launch arguments (as ``kernel[grid]``
            does).
        :param args: The launch arguments, positional and/or keyword, used to
            bind ``args[i]`` symbols. Tensors are accepted and ignored; only
            the integer scalars referenced by the equations matter.
        :param arg_values: Bind ``args[i]`` directly instead, either as an
            ``{index: value}`` mapping or a positional sequence.
        :param max_programs: Safety bound on grid enumeration for equations
            that depend on ``program_id``.
        """
        bound = self.bind_launch_args(*args, **kwargs)
        values, names = self.ttir_args(bound)
        if arg_values is not None:
            if isinstance(arg_values, Mapping):
                values = dict(arg_values)
            else:
                values = dict(enumerate(arg_values))
        bound_args = _BoundArgs(values, names)
        grid3 = _normalize_grid(grid, bound)

        total_flops = 0
        total_load = 0
        total_store = 0
        per_arg: Dict[str, Dict[str, int]] = {}
        for index, equations in sorted(self.entry.items()):
            arg_load = _sum_equation_over_grid(equations.get(LOAD_BYTES),
                                               bound_args, grid3, max_programs)
            arg_store = _sum_equation_over_grid(equations.get(STORE_BYTES),
                                                bound_args, grid3,
                                                max_programs)
            arg_flops = _sum_equation_over_grid(equations.get(OP_COUNT),
                                                bound_args, grid3,
                                                max_programs)
            label = names.get(index, f"args[{index}]")
            per_arg[label] = {
                OP_COUNT: arg_flops,
                "bytes": arg_load + arg_store,
                LOAD_BYTES: arg_load,
                STORE_BYTES: arg_store,
            }
            total_load += arg_load
            total_store += arg_store
            total_flops += arg_flops
        return ArithmeticIntensity(flops=total_flops,
                                   bytes=total_load + total_store,
                                   grid=grid3,
                                   load_bytes=total_load,
                                   store_bytes=total_store,
                                   per_arg=per_arg)


def arithmetic_intensity(
    kernel: Any,
    grid: Any,
    *args: Any,
    arg_values: Optional[Union[Mapping[int, Any], Sequence[Any]]] = None,
    max_programs: int = _DEFAULT_MAX_PROGRAMS,
    **kwargs: Any,
) -> ArithmeticIntensity:
    """Compute the overall op count and bytes moved by a kernel launch.

    Reads the equations recorded on ``kernel`` (a ``CompiledKernel`` compiled
    with :mod:`triton_arithmetic_intensity` imported) and evaluates them for
    ``grid`` and the launch arguments; see
    :meth:`ArithmeticIntensityEquations.evaluate` for the parameters.

    Example with ``triton.testing.do_bench``::

        import triton_arithmetic_intensity as tai

        compiled = matmul_kernel[grid](a, b, c, M, N, K, ...)
        ms = triton.testing.do_bench(
            lambda: matmul_kernel[grid](a, b, c, M, N, K, ...))
        work = tai.arithmetic_intensity(compiled, grid, a, b, c, M, N, K, ...)
        print(f"{work.tflops(ms):.1f} TFLOP/s, {work.gbps(ms):.0f} GB/s")
    """
    equations = ArithmeticIntensityEquations.from_kernel(kernel)
    return equations.evaluate(grid,
                              *args,
                              arg_values=arg_values,
                              max_programs=max_programs,
                              **kwargs)


# ---------------------------------------------------------------------------
# Compilation listener
# ---------------------------------------------------------------------------


class ArithmeticIntensityListener:
    """A Triton ``CompilationListener`` gathering equations per kernel.

    Install it with :meth:`install` (or :func:`enable`); Triton then calls it
    at the end of every ``triton.compile`` -- including cache hits, whose
    metadata is reloaded from disk -- and it records an
    :class:`ArithmeticIntensityEquations` for the compiled kernel:

    * :attr:`results` maps each kernel function name to the equations of its
      most recent compilation.
    * :attr:`history` keeps every compilation in order (one kernel function
      may be compiled several times with different specializations).

    Any listener previously installed on ``knobs.compilation.listener`` keeps
    being called.
    """

    def __init__(self) -> None:
        self.results: Dict[str, ArithmeticIntensityEquations] = {}
        self.history: List[ArithmeticIntensityEquations] = []
        self._previous: Optional[Callable[..., None]] = None
        self._installed = False

    # -- CompilationListener protocol --------------------------------------------

    def __call__(self, *, src: Any, metadata: Dict[str, Any],
                 metadata_group: Dict[str, str], times: Any,
                 cache_hit: bool) -> None:
        if self._previous is not None:
            self._previous(src=src,
                           metadata=metadata,
                           metadata_group=metadata_group,
                           times=times,
                           cache_hit=cache_hit)
        equations = ArithmeticIntensityEquations.from_metadata(metadata, src)
        if equations is None:
            return
        self.results[equations.name] = equations
        self.history.append(equations)

    # -- container helpers -----------------------------------------------------------

    def __getitem__(self, kernel: Any) -> ArithmeticIntensityEquations:
        return self.results[self._key(kernel)]

    def __contains__(self, kernel: Any) -> bool:
        return self._key(kernel) in self.results

    def __len__(self) -> int:
        return len(self.results)

    def __bool__(self) -> bool:
        # Triton guards calls with `if listener:`; an empty listener must
        # still be truthy or it would never receive its first compilation.
        return True

    def get(self,
            kernel: Any,
            default: Any = None) -> Optional[ArithmeticIntensityEquations]:
        return self.results.get(self._key(kernel), default)

    def clear(self) -> None:
        self.results.clear()
        self.history.clear()

    @staticmethod
    def _key(kernel: Any) -> str:
        """Accept a kernel name, a ``JITFunction`` or a ``CompiledKernel``."""
        if isinstance(kernel, str):
            return kernel
        metadata = getattr(kernel, "metadata", None)
        if metadata is not None and hasattr(metadata, "name"):
            return str(metadata.name)
        return str(getattr(kernel, "__name__", kernel))

    # -- installation --------------------------------------------------------------

    def install(self) -> ArithmeticIntensityListener:
        """Register on ``knobs.compilation.listener`` (chaining any existing one)."""
        from triton import knobs
        if self._installed:
            return self
        current = knobs.compilation.listener
        if current is not self:
            self._previous = current
        knobs.compilation.listener = self
        self._installed = True
        return self

    def uninstall(self) -> None:
        """Restore the listener that was installed before :meth:`install`."""
        from triton import knobs
        if not self._installed:
            return
        if knobs.compilation.listener is self:
            knobs.compilation.listener = self._previous
        self._previous = None
        self._installed = False


def enable() -> ArithmeticIntensityListener:
    """Install a global :class:`ArithmeticIntensityListener` and return it.

    Returns the already-installed instance when ``knobs.compilation.listener``
    is an :class:`ArithmeticIntensityListener`.
    """
    from triton import knobs
    existing = knobs.compilation.listener
    if isinstance(existing, ArithmeticIntensityListener):
        return existing
    return ArithmeticIntensityListener().install()
