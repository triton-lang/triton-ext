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


def _get_async_task_num_warps_stack() -> List[int]:
    if not hasattr(_tlx_state, 'async_task_num_warps_stack'):
        _tlx_state.async_task_num_warps_stack = []
    return _tlx_state.async_task_num_warps_stack


def current_num_warps(builder) -> int:
    """Warp count of the enclosing tlx.async_task, else the kernel's.

    A warp-specialized partition runs on its own warp count, and TMEM register
    layouts depend on it, so `builder.options.num_warps` is the wrong answer
    inside `with tlx.async_tasks(...)`. The fork reads this off the partition
    being generated; upstream has no equivalent, so the emitter maintains it.
    """
    stack = _get_async_task_num_warps_stack()
    if stack:
        return int(stack[-1])
    return int(builder.options.num_warps)


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

    with enter_sub_region(self) as sr:
        liveins, _ = sr
        ip, last_loc = self._get_insertion_point_and_loc()

        region_replica_id_stack = _get_region_replica_id_stack()
        num_warps_stack = _get_async_task_num_warps_stack()

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

        if len(taskWarpGroupStartIds) > 0:
            raise NotImplementedError(
                "tlx.async_task(warp_group_start_id=...) has no upstream "
                "equivalent. ttg.warp_specialize does carry a warpGroupStartIds "
                "attribute, but only the AllocateWarpGroups pass assigns it and "
                "no builder API exposes it. Drop the argument and let the "
                "compiler place the warp groups.")

        # Upstream has no `self.used_vars`, so capture every non-constexpr
        # live-in that carries IR. That is a superset of what used_vars would
        # select: a spurious capture costs an unused block argument, while a
        # missing one is a verifier error, because a partition region is
        # isolated from above and cannot reference an outer value directly.
        captures = []
        capture_handles = []
        for name in sorted(liveins):
            val = liveins[name]
            if _is_constexpr(val):
                continue
            if getattr(val, "__triton_aggregate__", False):
                handles = []
                for field in val.type.fields:
                    handles.extend(
                        _flatten_value_handles(getattr(val, field[0])))
            elif hasattr(val, "_flatten_ir") or hasattr(val, "handle"):
                handles = _flatten_value_handles(val)
            else:
                # A module, a JIT function, a plain Python object: not IR.
                continue
            if handles:
                captures.append(name)
                capture_handles.extend(handles)

        # Upstream splits the fork's single fused op in two: ttg.warp_specialize
        # holds the default region and the warp counts, and a nested
        # ttg.warp_specialize.partitions holds the worker regions and owns the
        # captures. The captures are operands fixed at construction, so unlike
        # the fork we cannot emit the workers first and append operands after --
        # the capture set has to be known up front, which is why it is computed
        # above rather than discovered by a throwaway codegen pass.
        #
        # Gluon's own driver emits the default body into a detached new_block()
        # to infer result types before creating the op. We must not: a detached
        # block has no parent region, and any nested control flow in the body
        # aborts the process, because _find_carries -> builder.create_block()
        # needs one. tlx.async_tasks yields no values, so the result types are
        # always empty and the op can be created first and filled in place.
        self._set_insertion_point_and_loc(ip, last_loc)
        ws_op = self.builder.create_warp_specialize([], taskNumWarps)
        if len(taskNumRegs) > 0:
            ws_op.set_requested_registers(taskNumRegs)

        for stmt in stmts:
            if not _get_async_task(self, stmt).is_default:
                continue
            region_replica_id_stack.append(0)
            num_warps_stack.append(self.builder.options.num_warps)
            self.builder.create_block_with_parent(ws_op.get_default_region(),
                                                  [])
            with enter_sub_region(self):
                self.visit(stmt)
            self.builder.create_warp_yield([])
            num_warps_stack.pop()
            region_replica_id_stack.pop()

        self.builder.create_block_with_parent(ws_op.get_partition_op_holder(),
                                              [])
        partitions_op = self.builder.create_warp_specialize_partitions(
            capture_handles, sum(taskReplica))
        arg_types = [handle.get_type() for handle in capture_handles]

        index = 0
        for stmt in stmts:
            task = _get_async_task(self, stmt)
            assert task.is_explict
            replicate_start = 1 if task.is_default else 0
            for i in range(replicate_start, task.replicate):
                region_replica_id_stack.append(i)
                num_warps_stack.append(taskNumWarps[index])
                block = self.builder.create_block_with_parent(
                    partitions_op.get_region(index), arg_types)
                index += 1
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)
                # Every partition takes the whole capture list, so argument j
                # always corresponds to capture_handles[j].
                for j, handle in enumerate(capture_handles):
                    block.replace_use_in_block_with(handle,
                                                    block.get_argument(j))
                self.builder.create_warp_return()
                num_warps_stack.pop()
                region_replica_id_stack.pop()

        self.builder.set_insertion_point_after(ws_op.get_operation())
