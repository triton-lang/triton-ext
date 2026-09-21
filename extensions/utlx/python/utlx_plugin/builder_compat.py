"""Call-shape adapters for builder bindings that differ between Triton builds.

A few op builders are bound with different Python signatures by Meta's in-tree
Triton and by upstream. Where the difference is only a defaulted trailing
argument, adapt the call here instead of forking the DSL.

Larger differences (an extra result type to compute, a missing dependency
operand) are handled by dedicated `utlx_*` plugin ops instead, since those need
access to the source value's MLIR type.
"""

import re

_arity_cache = {}


def binding_arity(builder, name):
    """(min, max) positional arity of a binding, excluding `self`.

    Reads the signature nanobind/pybind embeds in `__doc__`; returns None when
    the method is absent or its signature cannot be parsed (a pure-Python
    override, for instance), in which case callers should assume the in-tree
    shape.
    """
    cls = type(builder)
    key = (cls, name)
    if key in _arity_cache:
        return _arity_cache[key]

    result = None
    fn = getattr(cls, name, None)
    doc = getattr(fn, "__doc__", None) or ""
    for line in doc.splitlines():
        match = re.match(rf"\s*(?:\d+\.\s*)?{re.escape(name)}\((.*?)\)\s*->", line)
        if not match:
            continue
        params, depth, current = [], 0, ""
        for char in match.group(1):
            if char in "[(":
                depth += 1
            elif char in "])":
                depth -= 1
            if char == "," and depth == 0:
                params.append(current)
                current = ""
            else:
                current += char
        if current.strip():
            params.append(current)
        params = [p.strip() for p in params if p.strip() and not p.strip().startswith("self")]
        required = sum(0 if "=" in p else 1 for p in params)
        result = ((required, len(params)) if result is None else
                  (min(result[0], required), max(result[1], len(params))))

    _arity_cache[key] = result
    return result


def async_tma_store_wait(builder, pendings):
    """Wait for all but `pendings` outstanding TMA stores.

    Upstream's binding takes a second `readOnly` flag (whether the wait only
    needs the source buffer to be readable again, rather than the store to have
    fully landed); the in-tree one always waits for completion, which is
    `readOnly=False`.
    """
    arity = binding_arity(builder, "create_async_tma_store_wait")
    if arity is not None and arity[1] >= 2:
        builder.create_async_tma_store_wait(pendings, False)
    else:
        builder.create_async_tma_store_wait(pendings)
