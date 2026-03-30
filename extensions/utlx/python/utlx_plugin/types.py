"""uTLX types for plugin operations.

Provides the full TLX type system including buffered_tensor, mbarrier,
storage_alias_spec, reuse_group, tensor_memory_layout, etc.
"""

import enum
from abc import abstractmethod
from typing import List, Optional

import triton.language.core as tl
from triton._C.libtriton import ir
from triton.language.core import _aggregate as aggregate


class layout_encoding:

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__}.to_ir() must be overridden in subclasses"
        )


class shared_layout_encoding(layout_encoding):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def make_permute(self, dims):
        raise NotImplementedError(
            f"{self.__class__.__name__}.make_permute() must be overridden in subclasses"
        )

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__}.to_ir() must be overridden in subclasses"
        )


class swizzled_shared_layout_encoding(shared_layout_encoding):

    def __init__(
        self,
        vectorSize,
        perPhase,
        maxPhase,
        order,
        numCTAs,
        numCTAsPerCGA,
        numCTASplit,
        numCTAOrder,
    ):
        super().__init__()
        self.vectorSize = vectorSize
        self.perPhase = perPhase
        self.maxPhase = maxPhase
        self.order = order
        self.numCTAs = numCTAs
        self.numCTAsPerCGA = numCTAsPerCGA
        self.numCTASplit = numCTASplit
        self.numCTAOrder = numCTAOrder

    @classmethod
    def make_default(cls, rank):
        return cls(
            vectorSize=1,
            perPhase=1,
            maxPhase=1,
            order=list(reversed(range(rank))),
            numCTAs=[1] * rank,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=[1] * rank,
        )

    def make_permute(self, dims):
        permuted_order = tuple(self.order[d] for d in dims)
        return swizzled_shared_layout_encoding(
            self.vectorSize,
            self.perPhase,
            self.maxPhase,
            permuted_order,
            self.numCTAs,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
        )

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_swizzled_shared_encoding_attr(
            self.vectorSize,
            self.perPhase,
            self.maxPhase,
            self.order,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
        )


class tensor_memory_layout_encoding(shared_layout_encoding):

    def __init__(self, blockM, blockN, colStride, CTASplitM, CTASplitN):
        super().__init__()
        self.blockM = blockM
        self.blockN = blockN
        self.colStride = colStride
        self.CTASplitM = CTASplitM
        self.CTASplitN = CTASplitN

    @classmethod
    def make_default(cls, shape):
        return cls(
            blockM=shape[0],
            blockN=shape[1],
            colStride=1,
            CTASplitM=1,
            CTASplitN=1,
        )

    def make_permute(self, dims):
        return self

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_tensor_memory_encoding_attr(
            self.blockM,
            self.blockN,
            self.colStride,
            self.CTASplitM,
            self.CTASplitN,
        )


class nv_mma_shared_layout_encoding(shared_layout_encoding):

    def __init__(
        self,
        shape,
        order,
        elemType,
        numCTAsPerCGA,
        numCTASplit,
        numCTAOrder,
        fp4Padded,
        swizzled,
    ):
        super().__init__()
        self.shape = shape
        self.order = order
        self.elemType = elemType
        self.numCTAsPerCGA = numCTAsPerCGA
        self.numCTASplit = numCTASplit
        self.numCTAOrder = numCTAOrder
        self.fp4Padded = fp4Padded
        self.swizzled = swizzled

    @classmethod
    def make_default(cls, shape, elemType, fp4Padded=False):
        rank = len(shape)
        return cls(
            shape=shape,
            order=list(reversed(range(rank))),
            elemType=elemType,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=[1] * rank,
            fp4Padded=fp4Padded,
            swizzled=True,
        )

    def make_permute(self, dims):
        permuted_order = tuple(self.order[d] for d in dims)
        return nv_mma_shared_layout_encoding(
            self.shape,
            permuted_order,
            self.elemType,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
            self.fp4Padded,
            self.swizzled,
        )

    def __str__(self) -> str:
        return (
            f"nv_mma_shared_layout_encoding<{self.shape}, {self.order}, {self.elemType}, "
            f"{self.numCTAsPerCGA}, {self.numCTASplit}, {self.numCTAOrder}, "
            f"{self.fp4Padded}, {self.swizzled}>")

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape
                and self.order == other.order
                and self.elemType == other.elemType
                and self.numCTAsPerCGA == other.numCTAsPerCGA
                and self.numCTASplit == other.numCTASplit
                and self.numCTAOrder == other.numCTAOrder
                and self.fp4Padded == other.fp4Padded
                and self.swizzled == other.swizzled)

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_nv_mma_shared_encoding_attr(
            [int(x) for x in self.shape],
            self.order,
            self.elemType.to_ir(builder),
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
            self.fp4Padded,
            self.swizzled,
        )


class tensor_memory_scales_layout_encoding:

    def __init__(self, CTASplitM: int = 1, CTASplitN: int = 1):
        self.CTASplitM = CTASplitM
        self.CTASplitN = CTASplitN

    @classmethod
    def make_default(cls):
        return cls(CTASplitM=1, CTASplitN=1)

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_tensor_memory_scales_encoding_attr(
            self.CTASplitM, self.CTASplitN)


class DummyRegisterLayoutEncoding(layout_encoding):

    def __init__(self,
                 shape: List[int],
                 element_type: tl.dtype,
                 tmem_compatible: bool = False):
        super().__init__()
        self.shape = shape
        self.element_type = element_type
        self.tmem_compatible = tmem_compatible

    def to_ir(self, builder: ir.builder):
        return builder.make_dummy_register_layout_attr(
            self.shape, self.element_type.to_ir(builder), self.tmem_compatible)

    def __repr__(self):
        return f"DummyRegisterLayoutEncoding<{self.shape}, {self.element_type}, tmem_compatible={self.tmem_compatible}>"

    def __eq__(self, other):
        return (isinstance(other, DummyRegisterLayoutEncoding)
                and self.shape == other.shape
                and self.element_type == other.element_type
                and self.tmem_compatible == other.tmem_compatible)

    def __hash__(self):
        return hash(
            (tuple(self.shape), self.element_type, self.tmem_compatible))


class storage_kind(enum.Enum):
    smem = "smem"
    tmem = "tmem"
    smemCluster = "smemCluster"


class DummyTMEMLayoutEncoding(layout_encoding):

    def __init__(self):
        super().__init__()

    def to_ir(self, builder: ir.builder):
        return builder.make_dummy_tmem_layout_attr()

    def __repr__(self):
        return "DummyTMEMLayoutEncoding<>"

    def __eq__(self, other):
        return isinstance(other, DummyTMEMLayoutEncoding)

    def __hash__(self):
        return hash(self.__class__.__name__)


class reuse_group_type(enum.Enum):
    shared = "shared"
    distinct = "distinct"


class reuse_group:
    """Defines buffer overlap relationships for memory allocations."""

    def __init__(
        self,
        *args,
        group_type: "reuse_group_type" = reuse_group_type.shared,
        group_size: int = 1,
    ):
        if len(args) == 0:
            raise ValueError("reuse_group requires at least one element")

        group_size = tl._unwrap_if_constexpr(group_size)
        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError(
                f"group_size must be a positive integer, got {group_size}")

        args = tuple(tl._unwrap_if_constexpr(elem) for elem in args)
        for elem in args:
            if not isinstance(elem, (reuse_group, buffered_tensor)):
                raise TypeError(
                    f"reuse_group elements must be buffered_tensor or reuse_group, "
                    f"got {type(elem).__name__}")

        self._args = args
        self._group_type = group_type
        self._group_size = group_size

    @property
    def args(self):
        return self._args

    @property
    def group_type(self):
        return self._group_type

    @property
    def group_size(self):
        return self._group_size

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self._args)
        return f"reuse_group({args_str}, type={self._group_type.value}, size={self._group_size})"

    def _flatten_ir(self, handles):
        for elem in self._args:
            elem._flatten_ir(handles)

    def to_ir(self, builder):
        # Collect IR values for elements
        ir_elements = []
        for elem in self._args:
            if isinstance(elem, reuse_group):
                ir_elements.append(elem.to_ir(builder))
            elif isinstance(elem, buffered_tensor):
                ir_elements.append(elem.handle)

        group_kind_val = 0 if self._group_type == reuse_group_type.shared else 1
        group_kind_ir = builder.get_int32(group_kind_val)
        group_size_ir = builder.get_int32(self._group_size)

        args = [group_kind_ir, group_size_ir] + ir_elements
        return builder.utlx_reuse_group(args)


class buffered_tensor(tl.base_value):
    """A tensor allocated in a manually managed buffer (SMEM or TMEM)."""

    def __init__(
        self,
        handle,
        element_ty: tl.dtype,
        shape: List,
        num: int,
        storage: storage_kind,
        layout: Optional[shared_layout_encoding] = None,
    ):
        super().__init__()
        self.handle = handle
        self.shape = shape
        self.type = buffered_tensor_type(element_ty, shape, num, storage,
                                         layout)
        self.dtype = element_ty

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def make_permute(self, handle, dims):
        permuted_layout = self.type.layout.make_permute(dims)
        return buffered_tensor(
            handle,
            self.dtype,
            [self.shape[d] for d in dims],
            self.type.num,
            self.type.storage,
            permuted_layout,
        )


class buffered_tensor_type(tl.block_type):

    def __init__(
        self,
        element_ty: tl.dtype,
        shape: List,
        num: int,
        storage: storage_kind,
        layout: Optional[shared_layout_encoding] = None,
    ):
        super().__init__(element_ty, shape)
        self.storage = storage
        self.layout = layout
        self.num = num

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = "_".join(map(str, self.shape))
        if self.num > 0:
            shape += f"_{self.num}"
        return f"buffered_{elt}S{shape}"

    def __str__(self) -> str:
        return (
            f"buffered_tensor_type<{self.scalar}, {list(self.shape)}, "
            f"num={self.num}, storage={self.storage.value}, layout={self.layout}>"
        )

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.scalar == other.scalar
                and list(self.shape) == list(other.shape)
                and self.num == other.num and self.storage == other.storage
                and self.layout == other.layout)

    def to_ir(self, builder):
        shape = self.shape
        if self.num >= 1:
            shape = [self.num] + list(shape)
        assert self.layout is not None
        layout_ir = self.layout.to_ir(builder)
        alloc_shape = list(shape)
        if self.storage == storage_kind.tmem:
            return builder.get_tensor_mem_desc_ty(
                self.element_ty.to_ir(builder), shape, layout_ir, alloc_shape)
        else:
            return builder.get_shared_mem_desc_ty(
                self.element_ty.to_ir(builder), shape, layout_ir, alloc_shape)

    def _flatten_ir_types(self, builder, out) -> None:
        out.append(self.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        value = buffered_tensor(
            handles[cursor],
            self.scalar,
            self.shape,
            self.num,
            self.storage,
            self.layout,
        )
        return value, cursor + 1


class mbarrier(tl.base_value):
    """An mbarrier allocated in shared memory."""

    def __init__(
        self,
        handle,
        num: int,
        layout: Optional[shared_layout_encoding] = None,
        storage: storage_kind = storage_kind.smem,
        is_warp_barrier: bool = False,
    ):
        assert storage == storage_kind.smem or storage == storage_kind.smemCluster, (
            "mbarrier requires storage to be smem or smemCluster")
        self.handle = handle
        self.type = mbarrier_type(num, layout, storage, is_warp_barrier)
        self.num = num
        self.is_warp_barrier = is_warp_barrier

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)


class mbarrier_type(buffered_tensor_type):

    def __init__(self,
                 num: int,
                 layout: Optional[shared_layout_encoding],
                 storage,
                 is_warp_barrier: bool = False):
        super().__init__(tl.int64, [1], num, storage, layout)
        self.is_warp_barrier = is_warp_barrier

    def to_ir(self, builder):
        if self.num >= 1:
            shape = [self.num]
        else:
            shape = self.shape
        assert self.layout is not None
        layout_ir = self.layout.to_ir(builder)
        return builder.get_shared_mem_desc_ty(self.element_ty.to_ir(builder),
                                              shape, layout_ir, shape)

    def _unflatten_ir(self, handles, cursor):
        value = mbarrier(handles[cursor],
                         self.num,
                         self.layout,
                         self.storage,
                         is_warp_barrier=self.is_warp_barrier)
        return value, cursor + 1


class clc_response(tl.base_value):
    """A CLC response object."""

    def __init__(self, handle, num: int,
                 layout: Optional[shared_layout_encoding]):
        self.handle = handle
        self.type = clc_response_type(num, layout)
        self.num = num

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)


class clc_response_type(buffered_tensor_type):

    def __init__(self, num: int, layout: Optional[shared_layout_encoding]):
        super().__init__(tl.int64, [1], num, storage_kind.smem, layout)

    def to_ir(self, builder):
        if self.num >= 1:
            shape = [self.num]
        else:
            shape = self.shape
        assert self.layout is not None
        layout_ir = self.layout.to_ir(builder)
        return builder.get_shared_mem_desc_ty(self.element_ty.to_ir(builder),
                                              shape, layout_ir, shape)

    def _unflatten_ir(self, handles, cursor):
        value = clc_response(handles[cursor], self.num, self.layout)
        return value, cursor + 1


@aggregate
class CLCPipelineContext:
    _clc_mbars_empty: mbarrier
    _clc_mbars_full: mbarrier
    _clc_responses: clc_response

    def __init__(
        self,
        clc_mbars_empty: mbarrier,
        clc_mbars_full: mbarrier,
        clc_responses: clc_response,
    ):
        self._clc_mbars_empty = clc_mbars_empty
        self._clc_mbars_full = clc_mbars_full
        self._clc_responses = clc_responses


class reuse_group_ir_type(tl.base_type):

    def __init__(self, group_kind: reuse_group_type):
        self._group_kind = group_kind

    @property
    def group_kind(self) -> reuse_group_type:
        return self._group_kind

    def __eq__(self, other):
        return (isinstance(other, reuse_group_ir_type)
                and self._group_kind == other._group_kind)

    def mangle(self) -> str:
        return f"reuse_group_{self._group_kind.value}"


class storage_alias_spec(tl.base_value):
    """A storage alias specification for buffer sharing."""

    def __init__(
        self,
        handle,
        storage: storage_kind,
        buffer_size_bytes: Optional[int] = None,
    ):
        super().__init__()
        if storage == storage_kind.smemCluster:
            raise ValueError(
                "smemCluster storage is not supported for storage_alias_spec")
        self._handle = handle
        self._storage = storage
        self._buffer_size_bytes = buffer_size_bytes
        self.type = storage_alias_spec_type(storage, buffer_size_bytes)

    @property
    def handle(self):
        return self._handle

    @property
    def storage(self):
        return self._storage

    @property
    def buffer_size_bytes(self):
        return self._buffer_size_bytes

    def __repr__(self) -> str:
        return (f"storage_alias_spec(storage={self._storage.value}, "
                f"buffer_size_bytes={self._buffer_size_bytes})")

    @tl.builtin
    def set_buffer_overlap(self, overlap_def, _semantic=None):
        """Define the buffer overlap scheme for this storage alias spec."""
        overlap_def = tl._unwrap_if_constexpr(overlap_def)
        if not isinstance(overlap_def, reuse_group):
            raise TypeError(
                f"overlap_def must be a reuse_group, got {type(overlap_def).__name__}"
            )

        overlap_def_ir = overlap_def.to_ir(_semantic.builder)
        # Call the plugin custom op
        _semantic.builder.utlx_set_buffer_overlap(
            [self._handle, overlap_def_ir])

    def _flatten_ir(self, handles):
        handles.append(self._handle)


class storage_alias_spec_type(tl.base_type):

    def __init__(self, storage, buffer_size_bytes=None):
        self._storage = storage
        self._buffer_size_bytes = buffer_size_bytes

    @property
    def storage(self):
        return self._storage

    @property
    def buffer_size_bytes(self):
        return self._buffer_size_bytes

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self._storage == other._storage
                and self._buffer_size_bytes == other._buffer_size_bytes)

    def __repr__(self) -> str:
        if self._buffer_size_bytes is not None:
            return (f"storage_alias_spec_type(storage={self._storage.value}, "
                    f"buffer_bytes={self._buffer_size_bytes})")
        return f"storage_alias_spec_type(storage={self._storage.value})"

    def mangle(self):
        size_part = f"_{self._buffer_size_bytes}" if self._buffer_size_bytes else ""
        return f"storage_alias_spec_{self._storage.value}{size_part}"

    def to_ir(self, builder):
        # TODO: Requires get_storage_alias_spec_type builder method which
        # is TLX-specific and not available through the plugin interface.
        raise NotImplementedError(
            "storage_alias_spec_type.to_ir() requires TLX-specific builder methods "
            "not available in the plugin interface")

    def _flatten_ir_types(self, builder, out) -> None:
        out.append(self.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        value = storage_alias_spec(
            handles[cursor],
            self._storage,
            self._buffer_size_bytes,
        )
        return value, cursor + 1


class async_token(tl.base_value):
    """Tracks and synchronizes asynchronous operations."""

    def __init__(self, handle):
        self.handle = handle
        self.type = async_token_type(handle)

    def _flatten_ir(self, handles):
        handles.append(self.handle)


class async_token_type(tl.base_type):

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, async_token_type)

    def __repr__(self) -> str:
        return "async_token_type"

    def mangle(self):
        return "async_token_type"

    def _flatten_ir_types(self, builder, out) -> None:
        return  # No-op: async tokens don't contribute IR types

    def _unflatten_ir(self, handles, cursor):
        return async_token(handles[cursor]), cursor + 1


class tensor_descriptor_ptr(tl.base_value):

    def __init__(self, handle, num: int, descriptor_size: int):
        super().__init__()
        self.handle = handle
        self.type = tensor_descriptor_ptr_type(num, descriptor_size)

    @property
    def num(self) -> int:
        return self.type.num

    @property
    def descriptor_size(self) -> int:
        return self.type.size

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)


class tensor_descriptor_ptr_type(tl.pointer_type):

    def __init__(self, num: int, size: int = 128):
        element_type = tl.block_type(tl.int8, [size])
        super().__init__(element_type, address_space=1)
        self.num = num
        self.size = size

    def __eq__(self, other):
        return isinstance(
            other, tensor_descriptor_ptr_type
        ) and self.num == other.num and self.size == other.size

    def __repr__(self) -> str:
        return f"tensor_descriptor_ptr_type(num={self.num}, size={self.size})"

    def mangle(self) -> str:
        if self.num > 1:
            return f"tensor_desc_ptr_{self.num}_{self.size}"
        return f"tensor_desc_ptr_{self.size}"

    def _unflatten_ir(self, handles, cursor):
        return tensor_descriptor_ptr(handles[cursor], self.num,
                                     self.size), cursor + 1
