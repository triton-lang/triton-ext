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
    """Context manager grouping the `async_task` blocks of one kernel.

    The keyword options are recorded as attributes on the generated
    `ttg.warp_specialize` op (see `compiler/code_generator.py`); the codegen
    reads them off the AST rather than from an instance, so this constructor
    only has to accept and validate them.

    Not every host Triton acts on every option -- see
    `warp_spec_options.unsupported_options`.
    """

    def __init__(
        self,
        *args,
        exclusive=False,
        no_ending_cluster_sync=False,
        mbarrier_try_wait_suspend_ns=None,
        less_reg_mma=False,
        **kwargs,
    ):
        self.exclusive = core._unwrap_if_constexpr(exclusive)
        self.no_ending_cluster_sync = core._unwrap_if_constexpr(
            no_ending_cluster_sync)
        self.mbarrier_try_wait_suspend_ns = core._unwrap_if_constexpr(
            mbarrier_try_wait_suspend_ns)
        # Lower each MMA's SMEM operand descriptor with its own side-effecting
        # address computation so LLVM cannot CSE it across distant MMAs.
        self.less_reg_mma = core._unwrap_if_constexpr(less_reg_mma)
        if self.mbarrier_try_wait_suspend_ns is not None:
            if (not isinstance(self.mbarrier_try_wait_suspend_ns, int)
                    or self.mbarrier_try_wait_suspend_ns < 0):
                raise ValueError(
                    "mbarrier_try_wait_suspend_ns must be a non-negative integer"
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
