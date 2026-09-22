from triton.language import core


class async_task:
    """Context manager to run code fragments asynchronously."""

    def __init__(self, *args, _builder=None, **kwargs):
        self.builder = _builder
        self.is_default = False
        self.is_explict = False
        self.task_ids = None
        self.num_warps = None
        self.num_regs = None
        self.replicate = None
        self.warp_group_start_id = None
        if args:
            assert len(args) == 1
            if isinstance(args[0], core.constexpr) and args[0] == "default":
                self.is_explict = True
                self.is_default = True
                self.num_regs = core._unwrap_if_constexpr(
                    kwargs.get("num_regs", kwargs.get("registers", None)))
                self.replicate = core._unwrap_if_constexpr(
                    kwargs.get("replicate", 1))
                self.warp_group_start_id = core._unwrap_if_constexpr(
                    kwargs.get("warp_group_start_id", None))
            else:
                self.task_ids = list(
                    {core._unwrap_if_constexpr(tid)
                     for tid in args[0]})
        else:
            self.is_explict = True
            self.num_warps = core._unwrap_if_constexpr(
                kwargs.get("num_warps", None))
            self.num_regs = core._unwrap_if_constexpr(
                kwargs.get("num_regs", kwargs.get("registers", None)))
            self.replicate = core._unwrap_if_constexpr(
                kwargs.get("replicate", 1))
            self.warp_group_start_id = core._unwrap_if_constexpr(
                kwargs.get("warp_group_start_id", None))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class async_tasks:
    """Context manager grouping the `async_task` regions of one kernel.

    Normally intercepted at the AST level by ``visit_withAsyncTasks`` before it
    is ever instantiated. Upstream Triton's ``visit_With`` constructs every
    context manager as ``fn(*args, _semantic=..., **kws)``, so the signature has
    to absorb ``_semantic`` and the option keywords for the paths that do build
    one (a nested `with`, or a Triton whose ``visit_With`` we did not patch).
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
