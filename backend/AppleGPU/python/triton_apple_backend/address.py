"""Addresses a kernel can read through, for tables of device pointers.

Neither runtime's `data_ptr()` is one: on MPS it is the MTLBuffer object,
and in metal_native it is the CPU mapping.
"""


def gpu_address(buffer):
    """The address of an MPS tensor or a metal_native.MetalBuffer.

    Valid while `buffer` is alive: once freed, the allocator may reissue the
    storage and the address then names whatever took its place. A stale
    address raises nothing in either direction, since the read can return the
    freed bytes and look correct, or the newcomer's.
    """
    own = getattr(buffer, "gpu_address", None)
    if own is not None:
        return own()
    from . import metal_torch
    return metal_torch.gpu_address(buffer)


def address_table(buffers, device=None):
    """An int64 table of addresses, in whichever buffer type `buffers` are.

    The table is built in the type it was given, which is wider than what
    `gpu_address` answers for: a host holding buffers it wrapped itself has to
    build the table from addresses a kernel captured.
    """
    if not buffers:
        raise ValueError("address_table needs at least one buffer")
    addrs = [gpu_address(b) for b in buffers]

    if hasattr(buffers[0], "gpu_address"):
        import numpy as np
        from . import metal_native
        return metal_native.wrap(np.array(addrs, dtype=np.int64))

    import torch
    return torch.tensor(
        addrs,
        dtype=torch.int64,
        device=device if device is not None else buffers[0].device)
