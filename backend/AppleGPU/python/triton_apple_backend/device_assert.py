"""Reads the buffer a failed `tl.device_assert` writes and raises. Record
layout comes from the ``AGPU-ASSERT-LAYOUT`` block the emitter
(agpu/plan/AssertPlan.h) puts in the .metal module; the block's presence also
says whether a kernel asserts.

As on CUDA, a failure surfaces later, not at the launch: reading the buffer
right after the launch would stall the queue on every kernel that indexes
indirectly. It surfaces at the next synchronize, or at the next launch once the
failing kernel has finished when the buffer is host-visible shared memory.
"""

import re as _re

_TAG = "AGPU-ASSERT-LAYOUT"
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
    "sites",
)


class DeviceAssertError(RuntimeError):
    """A `tl.device_assert` failed on the device."""


class AssertLayout:
    """How to read one kernel's assert buffer, as the emitter described it."""

    def __init__(self, nums, messages, wheres):
        self._nums = nums
        self.messages = messages
        self.wheres = wheres

    @property
    def nbytes(self):
        return self._nums["bytes"]

    @property
    def capacity(self):
        return self._nums["records"]

    def head(self, words):
        """How many threads failed an assert."""
        return int(words[self._nums["headWord"]])

    def field(self, words, slot, name):
        base = (self._nums["headerWords"] + slot * self._nums["recordWords"] +
                self._nums["field." + name])
        return int(words[base])


def extract_assert_layout_text(msl):
    """The layout block of an emitted module, or None if it has none."""
    lines = [m.group(0).strip() for m in _LINE.finditer(msl)]
    return "\n".join(lines) if lines else None


def parse_assert_layout(text):
    """Turn a layout block back into something that can decode a buffer."""
    if not text:
        return None
    nums, messages, wheres = {}, {}, {}
    for key, value in _LINE.findall(text):
        if key.startswith("msg."):
            messages[int(key[4:])] = value
        elif key.startswith("where."):
            wheres[int(key[6:])] = value
        else:
            nums[key] = int(value)

    if not nums:
        return None

    missing = [k for k in _REQUIRED if k not in nums]
    if missing:
        raise RuntimeError(
            f"emitted assert layout is missing {missing}; the emitter and "
            "this decoder are out of sync (see agpu/plan/AssertPlan.h)")
    return AssertLayout(nums, messages, wheres)


_pending = {}


def defer(key, layout, buffer):
    """Check `buffer` at the next synchronize."""
    _pending[key] = (layout, buffer)


def _clear(rt, buffer):
    words = rt.peek_u32(buffer)
    if words is None:
        rt.clear_cache(buffer)
    else:
        words[:] = 0


def check_landed(rt):
    """Before a launch: raise for an earlier launch whose failure the host can
    already see, without waiting for the device."""
    for key, (layout, buffer) in list(_pending.items()):
        words = rt.peek_u32(buffer)
        if words is None or layout.head(words) == 0:
            continue
        words = words.copy()
        del _pending[key]
        _clear(rt, buffer)
        check(layout, words)


def check_pending(rt):
    """After a synchronize: raise for the first launch whose assert failed."""
    pending = list(_pending.values())
    _pending.clear()
    failed = None
    for layout, buffer in pending:
        words = rt.peek_u32(buffer)
        if words is None:
            words = rt.as_u32(buffer)
        if layout.head(words) == 0:
            continue
        if failed is None:
            failed = (layout, words.copy())
        _clear(rt, buffer)
    if failed is not None:
        check(*failed)


def check(layout, words):
    """Raise if any thread failed an assert. Returns None otherwise."""
    failures = layout.head(words)
    if failures == 0:
        return

    stored = min(failures, layout.capacity)
    # Slot 0 is whichever thread won the race to record.
    site = layout.field(words, 0, "site") if stored else 0
    pid = layout.field(words, 0, "pid") if stored else 0
    tid = layout.field(words, 0, "tid") if stored else 0

    message = layout.messages.get(site, "")
    where = layout.wheres.get(site, "unknown:0")
    summary = f"device assert failed: {message}" if message \
        else "device assert failed"

    detail = ""
    if failures > 1:
        detail = f"\n  {failures} threads failed; the first recorded is below"

    raise DeviceAssertError(
        f"{summary}{detail}\n"
        f"  at {where}\n"
        f"  pid ({pid}, 0, 0), thread {tid}\n"
        f"  kernel outputs are undefined after a failed assert")
