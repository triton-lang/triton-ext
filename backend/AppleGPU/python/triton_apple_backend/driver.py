"""Apple GPU Triton backend driver. Dispatch pipeline:
  metallib bytes -> metal_utils.load_metallib(bytes) -> MetalLibrary.get_function(name)
  -> MetalKernel (PSO) -> kernel(*tensors, threads=, group_size=)
"""

import os as _os
import re as _re
import struct as _struct
import torch
from triton.backends.driver import DriverBase, decompose_descriptor, expand_signature
from triton.runtime.errors import OutOfResources
from triton.tools.tensor_descriptor import TensorDescriptor
from triton_apple_backend.device_assert import check as _check_asserts
from triton_apple_backend.device_assert import parse_assert_layout
from triton_apple_backend.device_print import format_records, parse_print_layout
from triton_apple_backend.hw_constants import TARGET as _TARGET
from triton_apple_backend.hw_constants import TG_BUDGET_BYTES as _TG_BUDGET_BYTES
from triton_apple_backend.hw_constants import WARP_SIZE as _WARP_SIZE
from triton_apple_backend.tables import SCALAR_PACK_INFO as _SCALAR_PACK_INFO
from triton_apple_backend.tables import TY_TO_CPP as _TY_TO_CPP


def _load_metal_utils():
    from triton_apple_backend import metal_utils
    return metal_utils


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    try:
        return _TY_TO_CPP[ty]
    except KeyError:
        raise KeyError(
            f"ty_to_cpp has no entry for {ty!r}. Scalar kernel-argument types "
            "must appear in BOTH TY_TO_CPP and SCALAR_PACK_INFO in tables.py; "
            f"SCALAR_PACK_INFO {'has' if ty in _SCALAR_PACK_INFO else 'also lacks'} it."
        ) from None


_SETBYTES_LIMIT = 4096


def _is_pointer_type(ty):
    return isinstance(ty, str) and ty.startswith('*')


def _compute_scalar_layout(scalar_types):
    """Returns (total_size, field_offsets), field_offsets[i] the byte offset
    for scalar i in the packed buffer."""
    offsets = []
    current = 0
    for ty in scalar_types:
        info = _SCALAR_PACK_INFO.get(ty)
        if info is None:
            raise ValueError(f"Unknown scalar type for packing: {ty}")
        _, size, align = info
        padding = (align - (current % align)) % align
        current += padding
        offsets.append(current)
        current += size
    return current, offsets


def _pack_scalars(scalar_types, scalar_values, total_size, offsets):
    buf = bytearray(total_size)
    for ty, val, offset in zip(scalar_types, scalar_values, offsets):
        fmt, size, _ = _SCALAR_PACK_INFO[ty]
        if ty in ("i1", "u1"):
            val = 1 if val else 0
        elif ty == "bf16":
            bits = _struct.unpack('<I', _struct.pack('<f', float(val)))[0]
            bf16_bits = bits >> 16
            _struct.pack_into("H", buf, offset, bf16_bits)
            continue
        _struct.pack_into(fmt, buf, offset, val)
    return bytes(buf)


class _NoCompileFunction:
    """Stands in for a PSO under TRITON_MSL_NO_COMPILE. Thread budget is large
    enough for the launcher's shape checks to pass."""

    max_total_threads_per_threadgroup = 1024

    def __init__(self, name):
        self.name = name


class MetalUtils:
    """Metal GPU utils. JIT-compiles metal_utils.m for zero-copy MPS tensor
    dispatch."""

    def __init__(self):
        self._metal = _load_metal_utils()

    def load_binary(self, name, metallib_bytes, shared_mem, device):
        """Returns (module, function, n_regs, n_spills, n_max_threads)."""
        # Returns a handle, so a dump-only run reaches every kernel. Refusal
        # moves to launch.
        if _os.environ.get('TRITON_MSL_NO_COMPILE') == '1':
            return (None, _NoCompileFunction(name), 0, 0, 1024)
        try:
            module = self._metal.load_metallib(bytes(metallib_bytes))
            function = module.get_function(name)
            # Report the PSO's real maxTotalThreadsPerThreadgroup so triton
            # drops configs needing more threads. A cross-warp-smem kernel
            # launched with fewer leaves smem slots unwritten.
            max_threads = getattr(function,
                                  'max_total_threads_per_threadgroup', 1024)
            return module, function, 0, 0, max_threads
        except RuntimeError as e:
            msg = str(e)
            m = _re.search(
                r'Threadgroup (?:memory )?size \((\d+)\) exceeds the maximum .+ \((\d+)\)',
                msg)
            if m:
                raise OutOfResources(int(m.group(1)), int(m.group(2)),
                                     "Metal PSO") from e
            if 'exceeds available stack space' in msg:
                raise OutOfResources(0, 0, "Metal PSO stack space") from e
            raise

    def unload_module(self, module):
        del module

    def get_device_properties(self, device):
        return {
            "warpSize":
            _WARP_SIZE,
            "max_shared_mem":
            _TG_BUDGET_BYTES,
            "multiprocessorCount":
            getattr(torch._C, '_mps_get_core_count', lambda: 10)(),
        }

    def get_current_device(self):
        return 0

    def set_current_device(self, device):
        pass

    def get_current_stream(self, device):
        return 0


class MetalLauncher:
    """Called by Triton's JIT runtime to dispatch a compiled kernel.
    `function` = the MetalKernel returned by load_binary."""

    def __init__(self, src, metadata):
        self.signature = dict(src.signature)
        self.constants = getattr(src, "constants", {})

        # Constexpr args appear in Python *args but not the compiled IR;
        # strip them so Metal buffer slots match IR arg positions.
        self.constexpr_py_slots = frozenset(
            i for i, (k, ty) in enumerate(self.signature.items())
            if ty == 'constexpr')

        # Apple GPUs have no hardware TMA, so tensordesc_meta is always None.
        # Descriptors decompose to (ptr, *shape, *strides, padding, tf32,
        # *shape, *strides).
        non_constexpr_sig = [
            ty for ty in self.signature.values() if ty != 'constexpr'
        ]
        expanded = expand_signature(non_constexpr_sig, None, None)

        # Scalars pack into one device buffer, so IR param order is
        # [pointers, packed_scalar_buf, system_values]. Tuple args flatten
        # recursively; _flat_arg_keep[i] says whether flat_arg[i] is
        # forwarded (constexpr-in-tuple values are not).
        self.ptr_indices = []  # indices into the kept slice of flat_args
        self.scalar_indices = []  # indices into the kept slice of flat_args
        self.scalar_types = []  # type strings for scalars (for packing)

        def _flatten_types_with_mask(types):
            """Recursively flatten tuple types; return (all_types, keep_mask).

            all_types: leaf type for every position (including constexpr)
            keep_mask: True where the leaf is a real GPU arg (not constexpr)
            """
            all_tys, keep = [], []
            for ty in types:
                if isinstance(ty, tuple):
                    sub_tys, sub_keep = _flatten_types_with_mask(ty)
                    all_tys.extend(sub_tys)
                    keep.extend(sub_keep)
                else:
                    all_tys.append(ty)
                    keep.append(ty != 'constexpr')
            return all_tys, keep

        all_types, keep_mask = _flatten_types_with_mask(expanded)
        # A signature may omit constexpr params altogether, leaving them
        # untyped. Those are not in constexpr_py_slots, so they survive
        # the strip in __call__ and need one `False` each in the mask.
        fn = getattr(src, 'fn', None)
        declared = set(self.signature.keys())
        n_undeclared = sum(1 for name in (getattr(fn, 'arg_names', ()) or ())
                           if name not in declared)
        keep_mask = keep_mask + [False] * n_undeclared
        self._flat_arg_keep = keep_mask

        kept_slot = 0
        for ty, keep in zip(all_types, keep_mask):
            if not keep:
                continue
            if _is_pointer_type(ty):
                self.ptr_indices.append(kept_slot)
            else:
                self.scalar_indices.append(kept_slot)
                self.scalar_types.append(ty)
            kept_slot += 1

        if self.scalar_types:
            self.total_size, self.field_offsets = _compute_scalar_layout(
                self.scalar_types)
            if self.total_size > _SETBYTES_LIMIT:
                raise RuntimeError(
                    f"packed scalar args are {self.total_size} bytes, over "
                    f"Metal's {_SETBYTES_LIMIT}-byte setBytes limit")
        else:
            self.total_size = 0
            self.field_offsets = []

        self._requested_threads = getattr(metadata, "num_warps",
                                          4) * _WARP_SIZE
        self._smem_bytes = int(getattr(metadata, "shared", 0) or 0)
        self._cross_tg_barrier = bool(
            getattr(metadata, "cross_tg_barrier", False))
        # None when the kernel does not print/assert, which also says not to
        # bind a buffer.
        self._print_layout = parse_print_layout(
            getattr(metadata, "print_layout", None))
        self._assert_layout = parse_assert_layout(
            getattr(metadata, "assert_layout", None))
        self.lx = self._requested_threads
        self.ly = 1
        self.lz = 1

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *args):

        # Under TRITON_MSL_NO_COMPILE the dispatch is skipped, so outputs are
        # never written and the caller's own assertion fails afterwards.
        if isinstance(function, _NoCompileFunction):
            return None

        # load_binary already drops over-large configs, so a deficit here is
        # a bug.
        max_threads = getattr(function, 'max_total_threads_per_threadgroup',
                              1024)
        if self._requested_threads > max_threads:
            raise RuntimeError(
                f"kernel needs {self._requested_threads} threads/threadgroup "
                f"but PSO supports only {max_threads}; this config should have "
                f"been rejected at load_binary (OutOfResources)")

        # Grid-barrier kernels need every threadgroup co-resident. Metal has no
        # cooperative launch and does not preempt spinning threadgroups, so a
        # grid larger than what fits deadlocks or watchdog-aborts.
        if self._cross_tg_barrier:
            total_tgs = gridX * gridY * gridZ
            cores = getattr(torch._C, '_mps_get_core_count', lambda: 10)()
            capacity = cores * max(1, max_threads // self._requested_threads)
            if total_tgs > capacity:
                raise OutOfResources(
                    total_tgs, capacity,
                    "co-resident threadgroups (cross-threadgroup barrier)")

        if launch_enter_hook:
            launch_enter_hook(launch_metadata)

        # Strip constexpr args and decompose TensorDescriptors. Tuple args
        # flatten recursively to match _flat_arg_keep from __init__.
        from triton.runtime.jit import TensorWrapper

        def _flatten_arg(a, out):
            """Recursively flatten an arg value, expanding tuples to leaves."""
            if isinstance(a, TensorWrapper):
                out.append(a.base)
            elif isinstance(a, torch.Tensor):
                # Don't unwrap to `a._base`: that drops storage_offset, which
                # the launcher applies separately via setBuffer:offset:.
                out.append(a)
            elif isinstance(a, TensorDescriptor):
                out.extend(decompose_descriptor(a))
            elif isinstance(a, tuple):
                for elem in a:
                    _flatten_arg(elem, out)
            else:
                out.append(a)

        all_flat_args = []
        for i, a in enumerate(args):
            if i in self.constexpr_py_slots:
                continue
            _flatten_arg(a, all_flat_args)

        # A mismatch would pass wrong buffers via zip() truncation.
        if len(all_flat_args) != len(self._flat_arg_keep):
            raise RuntimeError(
                f"flat arg count {len(all_flat_args)} does not match signature "
                f"keep-mask length {len(self._flat_arg_keep)}; kernel call "
                f"signature is out of sync with the compiled IR")
        flat_args = [
            v for v, keep in zip(all_flat_args, self._flat_arg_keep) if keep
        ]

        # Emitted kernel signature is [ptr0, ptr1, ..., packed_scalar_buf].
        # See emitFunc's argbuf packing in EmitMSLFunc.cpp.
        ptr_args = [flat_args[i] for i in self.ptr_indices]
        scalar_values = [flat_args[i] for i in self.scalar_indices]

        if scalar_values:
            # setBytes inline; staging through an MPS tensor would cost a
            # device alloc + H2D copy per launch.
            packed_bytes = _pack_scalars(self.scalar_types, scalar_values,
                                         self.total_size, self.field_offsets)
            reordered_args = tuple(ptr_args) + (packed_bytes, )
        else:
            reordered_args = tuple(ptr_args)

        # Binds last, per planKernelAbi, so adding a print cannot renumber an
        # existing pointer binding. Must be zeroed: the head is a running
        # count the kernel bumps.
        print_buffer = None
        if self._print_layout is not None:
            print_buffer = torch.zeros(self._print_layout.nbytes // 4,
                                       dtype=torch.int32,
                                       device='mps')
            reordered_args = reordered_args + (print_buffer, )

        # Print first, then assert: the order planKernelAbi fixed.
        assert_buffer = None
        if self._assert_layout is not None:
            assert_buffer = torch.zeros(self._assert_layout.nbytes // 4,
                                        dtype=torch.int32,
                                        device='mps')
            reordered_args = reordered_args + (assert_buffer, )

        if _os.environ.get('TRITON_MSL_DEBUG'):
            _threads = [gridX * self.lx, gridY * self.ly, gridZ * self.lz]
            _gs = [self.lx, self.ly, self.lz]
            print(
                f'[MSL] threads={_threads} group_size={_gs} grid=({gridX},{gridY},{gridZ})'
            )
            print(f'[MSL] reordered_args={reordered_args}')
            if scalar_values:
                print(
                    f'[MSL] scalar_types={self.scalar_types} scalar_values={scalar_values}'
                )
                print(
                    f'[MSL] packed_bytes={packed_bytes.hex()} total_size={self.total_size}'
                )
        function(
            *reordered_args,
            threads=[gridX * self.lx, gridY * self.ly, gridZ * self.lz],
            group_size=[self.lx, self.ly, self.lz],
            threadgroup_mem=self._smem_bytes,
        )

        # The copy to CPU synchronises; the records do not exist until the
        # kernel has run.
        if print_buffer is not None:
            words = print_buffer.cpu().numpy().view('uint32')
            for line in format_records(self._print_layout, words):
                print(line)

        # Before the assert check, so a kernel that trips a device assert still
        # closes the profiler's span.
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)

        # Asserts last, so any print is already on stdout when this throws.
        if assert_buffer is not None:
            _check_asserts(self._assert_layout,
                           assert_buffer.cpu().numpy().view('uint32'))


class MetalDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher

    @staticmethod
    def is_active():
        try:
            return torch.backends.mps.is_available()
        except Exception:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_device_interface(self):
        return torch.mps

    def get_current_target(self):
        from triton.backends.compiler import GPUTarget
        return GPUTarget(_TARGET, "apple_m", _WARP_SIZE)

    def get_active_torch_device(self):
        return torch.device("mps", 0)

    def get_current_device(self):
        return 0

    def get_current_stream(self, device):
        # The default stream's id. ATen has had a stream
        # pool for a while; where `torch.mps` exposes `current_stream`/`stream`
        # a caller can be in a non-default one and dispatch ignores it.
        return 0

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        return torch.empty(256 * 1024 * 1024 // 4,
                           dtype=torch.int32,
                           device='mps')

    def clear_cache(self, cache):
        cache.zero_()
