class warp_pipeline_stage:
    """Context manager marking an explicit warp-pipeline stage (AMD only).

    This is a partitioning marker, not an automatic optimization. The compiler
    splits the loop body at stage boundaries and inserts conditional barriers
    so that one warp group executes one stage ahead of the other. Correctness
    depends on the user's buffering and synchronization structure -- the marker
    only defines where stages begin and end.

    Usage inside @triton.jit::

        for k in tl.range(NUM_BUFFERS - 1, K_ITERS):
            with tlx.warp_pipeline_stage("lds_load", priority=1):
                a_tile = tlx.local_load(tlx.local_view(buf_A, consumer))
            tlx.async_load_wait_group(0)
            with tlx.warp_pipeline_stage("compute_and_load", priority=0):
                tlx.async_load(a_ptrs, tlx.local_view(buf_A, producer), ...)
                acc = tl.dot(a_tile, b_tile, acc)

    The body is emitted inline, followed by the same border marker Gluon's
    ``warp_pipeline_stage`` emits; the AMD warp-pipeline passes consume it.

    ``scope="intra_wave"`` stages are accepted, with the fork's validation,
    and their bodies emitted inline with no marker. In the fork they bracket
    an instruction-scheduling window for a pass upstream Triton does not have;
    that pass only reorders instructions and adds no barriers or data
    movement, so the results are the same.

    Args:
        label: Stage name for diagnostics (e.g. "load", "compute").
        scope: ``"inter_wave"`` (the default) phase-shifts warp groups;
            ``"intra_wave"`` schedules instructions within each wave.
        priority: Hardware scheduling hint (0-3), maps to s_setprio.
            Higher values indicate more urgent scheduling. Inter-wave only.
        pair: Identifier shared by two adjacent intra-wave regions in the
            fork. Intra-wave only.
    """

    __slots__ = ("label", "scope", "priority", "pair")

    def __init__(self,
                 label=None,
                 *,
                 scope="inter_wave",
                 priority=None,
                 pair=None):
        if label is not None and not isinstance(label, str):
            raise ValueError(
                f"label must be a string or None, got {type(label).__name__}")
        if scope not in ("inter_wave", "intra_wave"):
            raise ValueError("scope must be 'inter_wave' or 'intra_wave', "
                             f"got {scope!r}")
        if priority is not None and not (0 <= priority <= 3):
            raise ValueError(f"priority must be 0-3, got {priority}")
        if pair is not None and (isinstance(pair, bool)
                                 or not isinstance(pair, int) or pair < 0):
            raise ValueError("pair must be a non-negative integer or None, "
                             f"got {pair!r}")
        if scope == "inter_wave" and pair is not None:
            raise ValueError("pair is only meaningful with scope='intra_wave'")
        if scope == "intra_wave" and priority is not None:
            raise ValueError(
                "priority is only meaningful with scope='inter_wave'")
        self.label = label
        self.scope = scope
        self.priority = priority
        self.pair = pair

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
