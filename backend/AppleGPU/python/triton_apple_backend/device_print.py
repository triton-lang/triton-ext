"""Reads the buffer `tl.device_print` writes and prints it to stdout. Record
layout comes from the ``AGPU-PRINT-LAYOUT`` block the emitter
(agpu/plan/PrintPlan.h) puts in the .metal module; the block's presence also
says whether a kernel prints.
"""

import re as _re
import struct as _struct

_TAG = "AGPU-PRINT-LAYOUT"
_LINE = _re.compile(r"^\s*" + _TAG + r"\s+([A-Za-z0-9_.]+)=(.*)$", _re.M)

_REQUIRED = (
    "headerWords",
    "headWord",
    "recordWords",
    "records",
    "bytes",
    "field.site",
    "field.pid",
    "field.tid",
    "field.index",
    "field.type",
    "field.value",
    "field.operand",
    "type.sint",
    "type.uint",
    "type.float",
    "sites",
)


class PrintLayout:
    """How to read one kernel's print buffer, as the emitter described it."""

    def __init__(self, nums, prefixes, hexflags, nops):
        self._nums = nums
        self.prefixes = prefixes
        self.hexflags = hexflags
        self.nops = nops

    @property
    def nbytes(self):
        return self._nums["bytes"]

    @property
    def capacity(self):
        return self._nums["records"]

    def head(self, words):
        """How many records the kernel tried to write (may exceed capacity)."""
        return words[self._nums["headWord"]]

    def field(self, words, slot, name):
        base = (self._nums["headerWords"] + slot * self._nums["recordWords"] +
                self._nums["field." + name])
        return words[base]

    def type_name(self, code):
        for name in ("sint", "uint", "float"):
            if code == self._nums["type." + name]:
                return name
        return "uint"


def extract_print_layout_text(msl):
    """The layout block of an emitted module, or None if it has none."""
    lines = [m.group(0).strip() for m in _LINE.finditer(msl)]
    return "\n".join(lines) if lines else None


def parse_print_layout(text):
    """Turn a layout block back into something that can decode a buffer."""
    if not text:
        return None
    nums, prefixes, hexflags, nops = {}, {}, {}, {}
    for key, value in _LINE.findall(text):
        if key.startswith("site."):
            _, idx, base = key.split(".", 2)
            prefixes[int(idx)] = value
            hexflags[int(idx)] = base == "hex"
        elif key.startswith("nops."):
            nops[int(key.split(".", 1)[1])] = int(value)
        else:
            nums[key] = int(value)

    if not nums:
        return None

    missing = [k for k in _REQUIRED if k not in nums]
    if missing:
        raise RuntimeError(
            f"emitted print layout is missing {missing}; the emitter and "
            "this decoder are out of sync (see agpu/plan/PrintPlan.h)")
    return PrintLayout(nums, prefixes, hexflags, nops)


def _format_value(raw, kind, as_hex):
    """One value, from its 32-bit word and the type code beside it. Floats
    travel as bits; converting on the device would round.
    """
    raw = int(raw)
    if kind == "float":
        value = _struct.unpack("<f", _struct.pack("<I", raw))[0]
        return f"0x{raw:08x}" if as_hex else repr(value)
    if kind == "sint":
        value = raw - (1 << 32) if raw >= (1 << 31) else raw
        return f"0x{raw:08x}" if as_hex else str(value)
    return f"0x{raw:08x}" if as_hex else str(raw)


def _decode(layout, words):
    """The records the buffer holds, in the order the kernel wrote them."""
    attempted = layout.head(words)
    stored = min(attempted, layout.capacity)
    out = []
    for slot in range(stored):
        site = layout.field(words, slot, "site")
        out.append((
            site,
            layout.field(words, slot, "pid"),
            layout.field(words, slot, "tid"),
            layout.field(words, slot, "index"),
            layout.type_name(layout.field(words, slot, "type")),
            layout.field(words, slot, "value"),
            layout.field(words, slot, "operand"),
        ))
    return out, attempted, stored


def format_records(layout, words):
    """The lines a user sees: `pid (X, 0, 0) idx (N)prefix value`, matching
    Triton's PrintOpToLLVM lowering.

    Sorted and deduplicated on (site, operand, pid, index, value): a layout
    replicates values across threads, so duplicate records are dropped. A
    differing value at the same key is kept.
    """
    records, attempted, stored = _decode(layout, words)

    ordered = sorted((site, operand, pid, index, kind, raw)
                     for site, pid, _tid, index, kind, raw, operand in records)

    seen = set()
    lines = []
    for key in ordered:
        if key in seen:
            continue
        seen.add(key)
        site, operand, pid, index, kind, raw = key
        prefix = layout.prefixes.get(site, "")
        value = _format_value(raw, kind, layout.hexflags.get(site, False))
        tag = f" (operand {operand})" if layout.nops.get(site, 1) > 1 else ""
        lines.append(f"pid ({pid}, 0, 0) idx ({index}){prefix}{tag}{value}")

    if attempted > stored:
        lines.append(
            f"[triton] device_print dropped {attempted - stored} of "
            f"{attempted} records: the print buffer holds {layout.capacity}")
    return lines
