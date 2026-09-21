import ast
import threading
from typing import List


def _get_tlx():
    import triton.language.extra.tlx as tlx
    return tlx


from contextlib import contextmanager  # noqa: E402

_tlx_state = threading.local()


def _get_region_replica_id_stack() -> List[int]:
    if not hasattr(_tlx_state, 'region_replica_id_stack'):
        _tlx_state.region_replica_id_stack = []
    return _tlx_state.region_replica_id_stack


def _get_task_num_warps_stack() -> List[int]:
    """Warp count of each enclosing `async_task` region, innermost last.

    A warp-specialized region runs on its own warp count, not the kernel's, and
    some layouts (a TMEM load's register layout, for one) can only be computed
    once that is known. The codegen is the only place that knows it, so record
    it while the region's body is being built.
    """
    if not hasattr(_tlx_state, 'task_num_warps_stack'):
        _tlx_state.task_num_warps_stack = []
    return _tlx_state.task_num_warps_stack


def current_task_num_warps(default=None):
    """Warp count of the innermost enclosing `async_task`, else `default`."""
    stack = _get_task_num_warps_stack()
    return stack[-1] if stack else default


def _get_sub_region_has_exception() -> bool:
    if not hasattr(_tlx_state, 'sub_region_has_exception'):
        _tlx_state.sub_region_has_exception = False
    return _tlx_state.sub_region_has_exception


def _set_sub_region_has_exception(value: bool) -> None:
    _tlx_state.sub_region_has_exception = value


@contextmanager
def tlx_enter_sub_region():
    region_replica_id_stack = _get_region_replica_id_stack()
    replica_id_stack_backup = region_replica_id_stack.copy()
    try:
        _set_sub_region_has_exception(False)
        yield
    except Exception as e:
        _set_sub_region_has_exception(True)
        raise e
    finally:
        if not _get_sub_region_has_exception():
            current_stack = _get_region_replica_id_stack()
            assert current_stack == replica_id_stack_backup, "region_replica_id_stack is not restored"


def _is_async_task(self, node) -> bool:
    if isinstance(node, ast.With):
        context = node.items[0].context_expr
        if isinstance(context, ast.Call):
            withitemClass = self.visit(context.func)
            if withitemClass == _get_tlx().async_task:
                return True
    return False


def _resolve_async_task_stmts(self, stmts):
    from triton.language.core import _unwrap_if_constexpr

    resolved = []
    for stmt in stmts:
        if _is_async_task(self, stmt):
            resolved.append(stmt)
        elif isinstance(stmt, ast.If):
            cond = self.visit(stmt.test)
            cond = _unwrap_if_constexpr(cond)
            active_block = stmt.body if cond else stmt.orelse
            for inner_stmt in active_block:
                assert _is_async_task(self, inner_stmt), (
                    "Statements inside a constexpr if-guard within async_tasks() "
                    "must be `with tlx.async_task(...)` blocks")
                resolved.append(inner_stmt)
        else:
            assert False, (
                "Statements inside async_tasks() must be `with tlx.async_task(...)` "
                "blocks or constexpr if-guards around them")
    return resolved


def _get_async_task(self, node):
    context = node.items[0].context_expr
    args = [self.visit(arg) for arg in context.args]
    kwargs = {kw.arg: self.visit(kw.value) for kw in context.keywords}
    with _get_tlx().async_task(*args, _builder=self.builder, **kwargs) as task:
        return task


def visit_withAsyncTask(self, node):
    self.visit_compound_statement(node.body)


def _validate_warp_group_start_ids(
    start_ids: List[int],
    num_warps: List[int],
    task_replicates: List[int],
    default_num_warps: int,
) -> None:
    assert len(start_ids) == len(num_warps) == len(task_replicates)

    for i, start_id in enumerate(start_ids):
        assert start_id >= 0, f"warp_group_start_id[{i}] = {start_id} must be non-negative"

    ranges = [(start_ids[i], start_ids[i] + num_warps[i] * task_replicates[i])
              for i in range(len(start_ids))]
    default_range = (0, default_num_warps)

    for i, (start_i, end_i) in enumerate(ranges):
        if start_i < default_range[1] and default_range[0] < end_i:
            assert False, (
                f"Overlapping warp ranges: task {i} uses warps [{start_i}, {end_i}) "
                f"which overlaps with default region warps [{default_range[0]}, {default_range[1]})"
            )

    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            start_i, end_i = ranges[i]
            start_j, end_j = ranges[j]
            if start_i < end_j and start_j < end_i:
                assert False, (
                    f"Overlapping warp ranges: task {i} uses warps [{start_i}, {end_i}) "
                    f"and task {j} uses warps [{start_j}, {end_j})")


@tlx_enter_sub_region()
def visit_withAsyncTasks(self, node):
    from triton.compiler.code_generator import enter_sub_region, _is_list_like, _is_constexpr
    from triton.language.core import _unwrap_if_constexpr

    from .. import warp_spec_options

    # Async-task policy is written as keywords on the `with` expression. Read
    # it off the AST (the context manager instance is never materialised) and
    # mark the warp_specialize op, so a Fixup pass can propagate it to the
    # module attributes that later lowering consumes.
    ws_context = node.items[0].context_expr
    exclusive = False
    no_ending_cluster_sync = False
    mbarrier_try_wait_suspend_ns = None
    less_reg_mma = False
    for kw in getattr(ws_context, "keywords", []):
        if kw.arg == "exclusive":
            exclusive = bool(_unwrap_if_constexpr(self.visit(kw.value)))
        elif kw.arg == "no_ending_cluster_sync":
            no_ending_cluster_sync = bool(
                _unwrap_if_constexpr(self.visit(kw.value)))
        elif kw.arg == "mbarrier_try_wait_suspend_ns":
            mbarrier_try_wait_suspend_ns = _unwrap_if_constexpr(
                self.visit(kw.value))
            if (not isinstance(mbarrier_try_wait_suspend_ns, int)
                    or mbarrier_try_wait_suspend_ns < 0):
                raise ValueError(
                    "mbarrier_try_wait_suspend_ns must be a non-negative integer"
                )
        elif kw.arg == "less_reg_mma":
            less_reg_mma = bool(_unwrap_if_constexpr(self.visit(kw.value)))

    warp_spec_options.warn_if_unsupported({
        "exclusive": exclusive,
        "no_ending_cluster_sync": no_ending_cluster_sync,
        "mbarrier_try_wait_suspend_ns": mbarrier_try_wait_suspend_ns,
        "less_reg_mma": less_reg_mma,
    })

    with enter_sub_region(self) as sr:
        liveins, _ = sr
        ip, last_loc = self._get_insertion_point_and_loc()

        region_replica_id_stack = _get_region_replica_id_stack()

        def _flatten_value_handles(val):
            handles = []
            if hasattr(val, "_flatten_ir"):
                val._flatten_ir(handles)
            else:
                handles.append(val.handle)
            return handles

        stmts = node.body
        if not _is_list_like(stmts):
            stmts = [stmts]

        stmts = _resolve_async_task_stmts(self, stmts)

        has_non_default = False
        for stmt in stmts:
            task_check = _get_async_task(self, stmt)
            if not task_check.is_default:
                has_non_default = True
                break

        if not has_non_default:
            for stmt in stmts:
                self.visit(stmt)
            return

        with tlx_enter_sub_region():
            block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(block)
            taskNumWarps = []
            taskNumRegs = []
            taskReplica = []
            taskWarpGroupStartIds = []

            perTaskNumWarps = []
            perTaskStartIds = []
            perTaskReplicates = []

            region_replica_id_stack.append(-1)

            num_default = 0
            for stmt in stmts:
                task = _get_async_task(self, stmt)
                assert task.is_explict
                assert task.replicate is not None
                if task.is_default:
                    num_default += 1
                    if task.replicate > 1:
                        taskReplica.append(task.replicate - 1)
                        taskNumWarps.extend([self.builder.options.num_warps] *
                                            (task.replicate - 1))
                        if task.num_regs:
                            taskNumRegs.extend([task.num_regs] *
                                               (task.replicate - 1))
                        if task.warp_group_start_id is not None:
                            taskWarpGroupStartIds.extend(
                                [task.warp_group_start_id] *
                                (task.replicate - 1))
                else:
                    taskReplica.append(task.replicate)
                    taskNumWarps.extend([task.num_warps] * task.replicate)
                    if task.num_regs:
                        taskNumRegs.extend([task.num_regs] * task.replicate)
                    if task.warp_group_start_id is not None:
                        for r in range(task.replicate):
                            taskWarpGroupStartIds.append(
                                task.warp_group_start_id + r * task.num_warps)
                        perTaskNumWarps.append(task.num_warps)
                        perTaskStartIds.append(task.warp_group_start_id)
                        perTaskReplicates.append(task.replicate)

            region_replica_id_stack.pop()

        assert num_default == 1, "Default task must be one and only one"
        block.erase()

        assert len(taskNumRegs) in [0, len(taskNumWarps)]
        assert len(taskWarpGroupStartIds) in [0, len(taskNumWarps)]

        if len(perTaskStartIds) > 0:
            _validate_warp_group_start_ids(perTaskStartIds, perTaskNumWarps,
                                           perTaskReplicates,
                                           self.builder.options.num_warps)

        if not hasattr(self.builder, "create_warp_specialize_op"):
            _build_warp_specialize_upstream(
                self, stmts, liveins, ip, last_loc, taskNumWarps, taskNumRegs,
                taskReplica, taskWarpGroupStartIds, region_replica_id_stack,
                _flatten_value_handles, {
                    "tlx.exclusive": exclusive,
                    "tlx.no_ending_cluster_sync": no_ending_cluster_sync,
                    "tlx.mbarrier_try_wait_suspend_ns":
                    mbarrier_try_wait_suspend_ns,
                    "tlx.less_reg_mma": less_reg_mma,
                })
            return

        self._set_insertion_point_and_loc(ip, last_loc)
        ws_op = self.builder.create_warp_specialize_op(
            taskNumWarps,
            taskNumRegs if len(taskNumRegs) > 0 else None,
            sum(taskReplica),
            taskWarpGroupStartIds if len(taskWarpGroupStartIds) > 0 else None,
        )
        if exclusive:
            ws_op.set_attr("tlx.exclusive", self.builder.get_unit_attr())
        if no_ending_cluster_sync:
            ws_op.set_attr("tlx.no_ending_cluster_sync",
                           self.builder.get_unit_attr())
        if mbarrier_try_wait_suspend_ns is not None:
            ws_op.set_attr(
                "tlx.mbarrier_try_wait_suspend_ns",
                self.builder.get_int32_attr(mbarrier_try_wait_suspend_ns),
            )
        if less_reg_mma:
            ws_op.set_attr("tlx.less_reg_mma", self.builder.get_unit_attr())

        index = 0
        for stmt in stmts:
            task = _get_async_task(self, stmt)
            assert task.is_explict
            task_replicate = (task.replicate -
                              1) if task.is_default else task.replicate
            if task_replicate > 0:
                task_body = ws_op.get_partition_region(index)
                block = self.builder.create_block_with_parent(task_body, [])
                region_replica_id_stack.append(0)
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)
                region_replica_id_stack.pop()
                index += task_replicate
                block.erase()

        captures = sorted(v for v in (liveins.keys() & self.used_vars)
                          if not _is_constexpr(liveins[v]))
        for name in captures:
            val = liveins[name]
            if getattr(val, "__triton_aggregate__", False):
                for field in val.type.fields:
                    v = getattr(val, field[0])
                    for h in _flatten_value_handles(v):
                        ws_op.append_operand(h)
            else:
                for h in _flatten_value_handles(val):
                    ws_op.append_operand(h)

        index = 0
        for stmt in stmts:
            task = _get_async_task(self, stmt)
            if task.is_default:
                region_replica_id_stack.append(0)
                # The default region runs on the kernel's own warps.
                _get_task_num_warps_stack().append(
                    self.builder.options.num_warps)
                task_body = ws_op.get_default_region()
                block = self.builder.create_block_with_parent(task_body, [])
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)
                self.builder.create_warp_yield_op()
                _get_task_num_warps_stack().pop()
                region_replica_id_stack.pop()

            replicate_start = 1 if task.is_default else 0
            for i in range(replicate_start, task.replicate):
                region_replica_id_stack.append(i)
                _get_task_num_warps_stack().append(
                    task.num_warps if task.num_warps else
                    self.builder.options.num_warps)
                task_body = ws_op.get_partition_region(index)
                index += 1
                block = self.builder.create_block_with_parent(task_body, [])
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)
                for name in captures:
                    val = liveins[name]
                    if getattr(val, "__triton_aggregate__", False):
                        for field in val.type.fields:
                            v = getattr(val, field[0])
                            for h in _flatten_value_handles(v):
                                arg = task_body.add_argument(h.get_type())
                                block.replace_use_in_block_with(h, arg)
                    else:
                        for h in _flatten_value_handles(val):
                            arg = task_body.add_argument(h.get_type())
                            block.replace_use_in_block_with(h, arg)
                self.builder.create_warp_return_op()
                _get_task_num_warps_stack().pop()
                region_replica_id_stack.pop()


def _build_warp_specialize_upstream(self, stmts, liveins, ip, last_loc,
                                    task_num_warps, task_num_regs,
                                    task_replica, task_warp_group_start_ids,
                                    region_replica_id_stack,
                                    flatten_value_handles, marker_attrs):
    """Emit `ttg.warp_specialize` using upstream Triton's builder API.

    Meta's Triton exposes one fused `create_warp_specialize_op` that takes the
    warp counts, register counts, partition count and warp-group start ids, and
    lets the caller append captures afterwards. Upstream splits the same thing
    up: the op is created with just its result types and warp counts, the
    default region is a detached block pushed into it, and the captures are
    operands of a nested `ttg.warp_specialize_partitions` whose regions all
    take them as block arguments. Mirrors the sequence in gluon's own
    `warp_specialize` lowering.
    """
    from triton.compiler.code_generator import enter_sub_region, _is_constexpr

    if task_warp_group_start_ids:
        raise NotImplementedError(
            "tlx.async_task(warp_group_start_id=...) needs Meta's Triton: "
            "upstream's ttg.warp_specialize has no warp-group start ids, it "
            "packs partitions after the default region automatically.")

    builder = self.builder
    num_partitions = sum(task_replica)

    # The caller's scan block has been erased, leaving the insertion point
    # dangling; put it back before creating anything.
    self._set_insertion_point_and_loc(ip, last_loc)

    # Create the op first: upstream only needs its result types and warp
    # counts up front. tlx tasks yield nothing, so there are no results. Having
    # it early gives us a real region to build throwaway blocks in -- a
    # detached block has no parent, and any control flow inside a task body
    # would try to create a sibling block and assert.
    ws_op = builder.create_warp_specialize([], task_num_warps)
    if task_num_regs:
        ws_op.set_requested_registers(task_num_regs)
    for name, value in marker_attrs.items():
        if value is True:
            ws_op.set_attr(name, builder.get_unit_attr())
        elif isinstance(value, int) and not isinstance(value, bool):
            ws_op.set_attr(name, builder.get_int32_attr(value))

    # 1. Dry-visit each partition body, only to record which live-in values it
    #    reads -- those become the captures.
    region_replica_id_stack.append(0)
    for stmt in stmts:
        task = _get_async_task(self, stmt)
        replicate = (task.replicate - 1) if task.is_default else task.replicate
        if replicate <= 0:
            continue
        probe = builder.create_block_with_parent(ws_op.get_default_region(),
                                                 [])
        builder.set_insertion_point_to_start(probe)
        with enter_sub_region(self):
            self.visit(stmt)
        probe.erase()
    region_replica_id_stack.pop()

    # Meta's code generator records which names a region read (`used_vars`)
    # so only those are captured. Upstream has no such bookkeeping, and its own
    # `warp_specialize` sidesteps the question by making the caller pass the
    # arguments explicitly. Fall back to capturing every non-constexpr live-in:
    # a superset is still correct -- a partition that ignores an argument just
    # leaves it unused -- it only costs a few extra operands.
    used = getattr(self, "used_vars", None)
    names = liveins.keys() if used is None else (liveins.keys() & used)
    captures = sorted(v for v in names if not _is_constexpr(liveins[v]))
    capture_handles = []
    for name in captures:
        val = liveins[name]
        if getattr(val, "__triton_aggregate__", False):
            for field in val.type.fields:
                capture_handles += flatten_value_handles(
                    getattr(val, field[0]))
        else:
            capture_handles += flatten_value_handles(val)

    # 2. The default region, for real.
    default_stmt = next(stmt for stmt in stmts
                        if _get_async_task(self, stmt).is_default)
    default_block = builder.create_block_with_parent(
        ws_op.get_default_region(), [])
    builder.set_insertion_point_to_start(default_block)
    region_replica_id_stack.append(0)
    _get_task_num_warps_stack().append(builder.options.num_warps)
    with enter_sub_region(self):
        self.visit(default_stmt)
    builder.create_warp_yield([])
    _get_task_num_warps_stack().pop()
    region_replica_id_stack.pop()

    # 4. Partitions: every region takes the whole capture list as arguments.
    builder.create_block_with_parent(ws_op.get_partition_op_holder(), [])
    partitions_op = builder.create_warp_specialize_partitions(
        capture_handles, num_partitions)
    arg_types = [handle.get_type() for handle in capture_handles]

    index = 0
    for stmt in stmts:
        task = _get_async_task(self, stmt)
        first = 1 if task.is_default else 0
        for replica in range(first, task.replicate):
            block = builder.create_block_with_parent(
                partitions_op.get_region(index), arg_types)
            index += 1
            builder.set_insertion_point_to_start(block)
            region_replica_id_stack.append(replica)
            _get_task_num_warps_stack().append(task.num_warps if task.num_warps
                                               else builder.options.num_warps)
            with enter_sub_region(self):
                self.visit(stmt)
            for position, handle in enumerate(capture_handles):
                block.replace_use_in_block_with(handle,
                                                block.get_argument(position))
            builder.create_warp_return()
            _get_task_num_warps_stack().pop()
            region_replica_id_stack.pop()

    builder.set_insertion_point_after(ws_op.get_operation())
