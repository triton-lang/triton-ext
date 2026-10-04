"""Run the arithmetic-intensity pass at the end of every ``ttir`` stage.

Triton lets extensions customize the compilation pipeline through
``knobs.runtime.add_stages_inspection_hook``: each backend calls the hook with
its freshly built ``stages`` table (see ``add_stages`` in the NVIDIA and AMD
backends), and the hook may replace or wrap any stage. :func:`install` (called
when :mod:`triton_arithmetic_intensity` is imported) registers
:func:`inspect_stages_hook`, which wraps the ``ttir`` stage so that:

1. the original stage runs (``make_ttir`` or whatever a previously installed
   hook replaced it with),
2. ``triton-arithmetic-intensity`` runs on the resulting module, annotating
   every ``tt.func`` argument with ``tai.load_bytes`` / ``tai.store_bytes`` /
   ``tai.op_count``,
3. the equations are copied into ``metadata["arithmetic_intensity"]`` as
   ``{function name: {arg index:
   {"load_bytes": eq, "store_bytes": eq, "op_count": eq}}}``.

Triton serializes ``metadata`` next to the compiled binary, so the equations
survive cache hits and are exposed on ``CompiledKernel.metadata``; see
:mod:`triton_arithmetic_intensity.arithmetic_intensity` for the consumers.

The hook takes part in Triton's cache key: it returns a ``(key, hash)`` pair
derived from this file and the plugin library (combined with the pair of any
previously installed hook), so kernels compiled with or without the pass do
not share cache entries.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from triton import knobs
from triton._C.libtriton import ir, passes

from .arithmetic_intensity import LOAD_BYTES, METADATA_KEY, OP_COUNT, STORE_BYTES

#: Name of the compilation stage after which the pass is run.
STAGE = "ttir"
#: Name given to the pass-manager run (used for reproducer files).
PASS_NAME = "arithmetic_intensity"

#: ``tt.func`` argument attributes written by the pass -> metadata keys.
_ARG_ATTRS = {
    "tai.load_bytes": LOAD_BYTES,
    "tai.store_bytes": STORE_BYTES,
    "tai.op_count": OP_COUNT,
}
_ATTR_RE = {
    key: re.compile(r'\b' + re.escape(attr) + r'\s*=\s*"((?:[^"\\]|\\.)*)"')
    for attr, key in _ARG_ATTRS.items()
}

#: Hook that was installed before ours; it keeps being called.
_previous: Optional[Callable[..., Any]] = None
_cache_key: Optional[Tuple[str, str]] = None

# ---------------------------------------------------------------------------
# Running the pass and harvesting its results
# ---------------------------------------------------------------------------


def run_pass(mod: Any) -> Any:
    """Run ``triton-arithmetic-intensity`` on ``mod`` (in place)."""
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.plugin.add_arithmetic_intensity(pm)
    pm.run(mod, PASS_NAME)
    return mod


def _unescape(value: str) -> str:
    return value.replace('\\"', '"').replace("\\\\", "\\")


def _split_signature_args(text: str) -> list[str]:
    """Split the argument list of a printed ``tt.func`` into per-arg text.

    ``text`` starts at (or before) the function's opening parenthesis. The
    scan tracks string literals and bracket nesting so commas inside types
    (``!tt.ptr<f32, 3>``), attribute dictionaries and equation strings do not
    split an argument.
    """
    depth = {"(": 0, "<": 0, "{": 0, "[": 0}
    closer = {")": "(", ">": "<", "}": "{", "]": "["}
    in_string = False
    start: Optional[int] = None
    args: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif start is None:
            if ch == "(":
                depth["("] = 1
                start = i + 1
        elif ch in depth:
            depth[ch] += 1
        elif ch in closer:
            if ch == ">" and i > 0 and text[i - 1] == "-":
                pass  # `->` in a function type, not a closing bracket
            else:
                depth[closer[ch]] -= 1
                if ch == ")" and depth["("] == 0:
                    args.append(text[start:i])
                    break
        elif ch == "," and depth["("] == 1 and not any(depth[k]
                                                       for k in "<{["):
            args.append(text[start:i])
            start = i + 1
        i += 1
    return [arg for arg in args if arg.strip()]


def parse_function_equations(func_text: str) -> Dict[int, Dict[str, str]]:
    """Extract the pass's arg attrs (see ``_ARG_ATTRS``) from printed IR.

    Triton's Python bindings expose no reader for function argument
    attributes, so the equations are recovered from the textual form of the
    ``tt.func`` (as produced by ``str_nodebug()``). Returns
    ``{arg index: {"load_bytes": eq, "store_bytes": eq, "op_count": eq}}`` for
    every argument that carries at least one of the attributes.
    """
    result: Dict[int, Dict[str, str]] = {}
    for index, arg in enumerate(_split_signature_args(func_text)):
        equations: Dict[str, str] = {}
        for key, pattern in _ATTR_RE.items():
            match = pattern.search(arg)
            if match:
                equations[key] = _unescape(match.group(1))
        if equations:
            result[index] = equations
    return result


def collect_module_equations(mod: Any) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Read the equations of every ``tt.func`` in ``mod``.

    Returns ``{function name: {str(arg index): {...}}}``; indices are
    stringified so the value is JSON-serializable as compilation metadata.
    """
    names: list[str] = []

    def visit(op: Any) -> None:
        if op.get_name() == "tt.func":
            name = op.get_str_attr("sym_name")
            if name is not None:
                names.append(name)

    mod.walk(visit)
    equations: Dict[str, Dict[str, Dict[str, str]]] = {}
    for name in names:
        func = mod.get_function(name)
        per_arg = parse_function_equations(func.str_nodebug())
        equations[name] = {str(i): eq for i, eq in sorted(per_arg.items())}
    return equations


def annotate(mod: Any, metadata: Dict[str, Any]) -> Any:
    """Run the pass on ``mod`` and record its results in ``metadata``."""
    try:
        run_pass(mod)
    except RuntimeError as exc:  # pass failure: keep compiling the kernel
        warnings.warn(f"triton-arithmetic-intensity failed: {exc}",
                      RuntimeWarning,
                      stacklevel=2)
        return mod
    metadata[METADATA_KEY] = collect_module_equations(mod)
    return mod


def wrap_stage(
    original: Callable[[Any, Dict[str, Any]], Any]
) -> Callable[[Any, Dict[str, Any]], Any]:
    """Return a ``ttir`` stage running ``original`` and then the pass."""

    def make_ttir(mod: Any, metadata: Dict[str, Any]) -> Any:
        return annotate(original(mod, metadata), metadata)

    return make_ttir


# ---------------------------------------------------------------------------
# Pipeline hook
# ---------------------------------------------------------------------------


def _self_key() -> str:
    here = Path(__file__).resolve()
    key = here.read_text()
    library = here.parent / "libarithmetic_intensity.so"
    if library.exists():
        key += "\n" + hashlib.sha256(library.read_bytes()).hexdigest()
    return key


def cache_key() -> Tuple[str, str]:
    """``(key, hash)`` identifying this pipeline customization."""
    global _cache_key
    if _cache_key is None:
        key = _self_key()
        if _previous is not None:
            previous_key, _ = _previous()
            key = previous_key + "\n" + key
        _cache_key = key, hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _cache_key


def inspect_stages_hook(self: Any = None,
                        stages: Optional[Dict[str, Any]] = None,
                        options: Any = None,
                        language: Any = None,
                        capability: Any = None) -> Tuple[str, str]:
    """``PipelineStagesHook`` appending the pass to the ``ttir`` stage.

    Called without arguments it only returns the cache key; called by a
    backend's ``add_stages`` it first defers to the previously installed hook
    (if any) and then wraps whatever ``stages["ttir"]`` is at that point.
    Pipelines without a ``ttir`` stage (e.g. Gluon) are left alone.
    """
    if all(arg is None for arg in (stages, options, language, capability)):
        return cache_key()
    assert stages is not None
    if _previous is not None:
        _previous(self, stages, options, language, capability)
    original = stages.get(STAGE)
    if original is not None:
        stages[STAGE] = wrap_stage(original)
    return cache_key()


def install() -> None:
    """Register :func:`inspect_stages_hook`, chaining any existing hook."""
    global _previous, _cache_key
    current = knobs.runtime.add_stages_inspection_hook
    if current is inspect_stages_hook:
        return
    _previous = current
    _cache_key = None
    knobs.runtime.add_stages_inspection_hook = inspect_stages_hook


def uninstall() -> None:
    """Restore the hook that was installed before :func:`install`."""
    global _previous, _cache_key
    if knobs.runtime.add_stages_inspection_hook is inspect_stages_hook:
        knobs.runtime.add_stages_inspection_hook = _previous
    _previous = None
    _cache_key = None
