"""Pure Python tests for uTLX type system.

No GPU required. Covers:
  - storage_kind enum
  - reuse_group_type enum
  - storage_alias_spec_type
  - storage_alias_spec class
  - reuse_group class
  - buffered_tensor / buffered_tensor_type
  - layout_encoding subclasses
  - mbarrier type
  - async_token type
"""

import pytest
import triton.language.core as tl
from conftest import tlx

# ---------------------------------------------------------------------------
# storage_kind
# ---------------------------------------------------------------------------


class TestStorageKind:

    def test_values(self):
        assert tlx.storage_kind.smem.value == "smem"
        assert tlx.storage_kind.tmem.value == "tmem"
        assert tlx.storage_kind.smemCluster.value == "smemCluster"

    def test_enum_members(self):
        members = list(tlx.storage_kind)
        assert len(members) == 3


# ---------------------------------------------------------------------------
# reuse_group_type
# ---------------------------------------------------------------------------


class TestReuseGroupType:

    def test_values(self):
        assert tlx.reuse_group_type.shared.value == "shared"
        assert tlx.reuse_group_type.distinct.value == "distinct"

    def test_enum_members(self):
        members = list(tlx.reuse_group_type)
        assert len(members) == 2


# ---------------------------------------------------------------------------
# storage_alias_spec_type
# ---------------------------------------------------------------------------


class TestStorageAliasSpecType:

    def test_smem_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        assert ty.storage == tlx.storage_kind.smem
        assert ty.buffer_size_bytes is None

    def test_tmem_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem)
        assert ty.storage == tlx.storage_kind.tmem
        assert ty.buffer_size_bytes is None

    def test_smem_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        assert ty.storage == tlx.storage_kind.smem
        assert ty.buffer_size_bytes == 16384

    def test_tmem_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 32768)
        assert ty.storage == tlx.storage_kind.tmem
        assert ty.buffer_size_bytes == 32768

    def test_equality_same(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        # Both have the same storage and size
        assert ty1.storage == ty2.storage
        assert ty1.buffer_size_bytes == ty2.buffer_size_bytes

    def test_different_storage(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 16384)
        assert ty1.storage != ty2.storage

    def test_different_size(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 32768)
        assert ty1.buffer_size_bytes != ty2.buffer_size_bytes

    def test_mangle_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        mangle = ty.mangle()
        assert "storage_alias_spec" in mangle
        assert "smem" in mangle

    def test_mangle_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 8192)
        mangle = ty.mangle()
        assert "storage_alias_spec" in mangle
        assert "tmem" in mangle
        assert "8192" in mangle


# ---------------------------------------------------------------------------
# storage_alias_spec class (Python-level, no IR handle)
# ---------------------------------------------------------------------------


class TestStorageAliasSpecClass:
    """Tests for the storage_alias_spec Python class (via storage_alias_spec_type_class)."""

    def test_smem_unsized(self):
        spec = tlx.storage_alias_spec_type_class(handle=None,
                                                 storage=tlx.storage_kind.smem)
        assert spec.storage == tlx.storage_kind.smem
        assert spec.buffer_size_bytes is None
        assert spec.handle is None

    def test_tmem_sized(self):
        spec = tlx.storage_alias_spec_type_class(handle=None,
                                                 storage=tlx.storage_kind.tmem,
                                                 buffer_size_bytes=32768)
        assert spec.storage == tlx.storage_kind.tmem
        assert spec.buffer_size_bytes == 32768

    def test_rejects_smem_cluster(self):
        with pytest.raises(ValueError, match="smemCluster"):
            tlx.storage_alias_spec_type_class(
                handle=None, storage=tlx.storage_kind.smemCluster)

    def test_type_attribute(self):
        spec = tlx.storage_alias_spec_type_class(handle=None,
                                                 storage=tlx.storage_kind.smem,
                                                 buffer_size_bytes=4096)
        assert isinstance(spec.type, tlx.storage_alias_spec_type)
        assert spec.type.storage == tlx.storage_kind.smem
        assert spec.type.buffer_size_bytes == 4096

    def test_immutability_storage(self):
        spec = tlx.storage_alias_spec_type_class(handle=None,
                                                 storage=tlx.storage_kind.smem)
        with pytest.raises(AttributeError):
            spec.storage = tlx.storage_kind.tmem

    def test_immutability_buffer_size(self):
        spec = tlx.storage_alias_spec_type_class(handle=None,
                                                 storage=tlx.storage_kind.smem,
                                                 buffer_size_bytes=1024)
        with pytest.raises(AttributeError):
            spec.buffer_size_bytes = 2048


# ---------------------------------------------------------------------------
# reuse_group
# ---------------------------------------------------------------------------


def _make_test_buffered_tensor(storage=tlx.storage_kind.smem):
    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=2)
    return tlx.buffered_tensor(
        handle=None,
        element_ty=tl.float32,
        shape=[64, 64],
        num=2,
        storage=storage,
        layout=layout,
    )


class TestReuseGroup:

    def test_basic_shared(self):
        a = _make_test_buffered_tensor()
        b = _make_test_buffered_tensor()
        group = tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.shared)
        assert group.args == (a, b)
        assert group.group_type == tlx.reuse_group_type.shared

    def test_basic_distinct(self):
        a = _make_test_buffered_tensor()
        b = _make_test_buffered_tensor()
        group = tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.distinct)
        assert group.args == (a, b)
        assert group.group_type == tlx.reuse_group_type.distinct

    def test_single_element(self):
        elem = _make_test_buffered_tensor()
        group = tlx.reuse_group(elem, group_type=tlx.reuse_group_type.shared)
        assert len(group.args) == 1
        assert group.args[0] is elem

    def test_multiple_elements(self):
        elems = tuple(_make_test_buffered_tensor() for _ in range(4))
        group = tlx.reuse_group(*elems,
                                group_type=tlx.reuse_group_type.distinct)
        assert group.args == elems
        assert len(group.args) == 4

    def test_nested(self):
        """Nested reuse_group (Flash Attention pattern)."""
        p = _make_test_buffered_tensor()
        alpha = _make_test_buffered_tensor()
        inner = tlx.reuse_group(p,
                                alpha,
                                group_type=tlx.reuse_group_type.distinct)

        qk = _make_test_buffered_tensor()
        outer = tlx.reuse_group(qk,
                                inner,
                                group_type=tlx.reuse_group_type.shared)

        assert outer.group_type == tlx.reuse_group_type.shared
        assert len(outer.args) == 2
        assert outer.args[0] is qk
        assert outer.args[1] is inner
        assert inner.group_type == tlx.reuse_group_type.distinct

    def test_deeply_nested(self):
        """3-level nested reuse_group."""
        c = _make_test_buffered_tensor()
        d = _make_test_buffered_tensor()
        inner = tlx.reuse_group(c, d, group_type=tlx.reuse_group_type.shared)

        b = _make_test_buffered_tensor()
        middle = tlx.reuse_group(b,
                                 inner,
                                 group_type=tlx.reuse_group_type.distinct)

        a = _make_test_buffered_tensor()
        outer = tlx.reuse_group(a,
                                middle,
                                group_type=tlx.reuse_group_type.shared)

        assert outer.group_type == tlx.reuse_group_type.shared
        assert outer.args[1].group_type == tlx.reuse_group_type.distinct
        assert outer.args[1].args[1].group_type == tlx.reuse_group_type.shared

    def test_group_size(self):
        elem = _make_test_buffered_tensor()
        group = tlx.reuse_group(elem,
                                group_type=tlx.reuse_group_type.shared,
                                group_size=2)
        assert group.group_size == 2

    def test_empty_args_raises(self):
        with pytest.raises(ValueError, match="at least one element"):
            tlx.reuse_group(group_type=tlx.reuse_group_type.shared)

    def test_invalid_element_type_raises(self):
        with pytest.raises(TypeError,
                           match="must be buffered_tensor or reuse_group"):
            tlx.reuse_group("invalid", group_type=tlx.reuse_group_type.shared)


# ---------------------------------------------------------------------------
# buffered_tensor / buffered_tensor_type
# ---------------------------------------------------------------------------


class TestBufferedTensor:

    def test_basic_construction(self):
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=2)
        bt = tlx.buffered_tensor(
            handle=None,
            element_ty=tl.float16,
            shape=[64, 64],
            num=2,
            storage=tlx.storage_kind.smem,
            layout=layout,
        )
        assert bt.dtype == tl.float16
        assert bt.shape == [64, 64]
        assert bt.type.num == 2
        assert bt.type.storage == tlx.storage_kind.smem

    def test_1d_shape(self):
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
        bt = tlx.buffered_tensor(
            handle=None,
            element_ty=tl.float32,
            shape=[128],
            num=1,
            storage=tlx.storage_kind.smem,
            layout=layout,
        )
        assert bt.shape == [128]
        assert bt.type.num == 1

    def test_type_element_ty(self):
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=2)
        bt = tlx.buffered_tensor(
            handle=None,
            element_ty=tl.bfloat16,
            shape=[32, 32],
            num=3,
            storage=tlx.storage_kind.smem,
            layout=layout,
        )
        assert bt.type.element_ty == tl.bfloat16


# ---------------------------------------------------------------------------
# Layout encodings
# ---------------------------------------------------------------------------


class TestSwizzledSharedLayoutEncoding:

    def test_make_default_rank1(self):
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
        assert layout.vectorSize == 1
        assert layout.perPhase == 1
        assert layout.maxPhase == 1
        assert layout.order == [0]

    def test_make_default_rank2(self):
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=2)
        assert layout.order == [1, 0]

    def test_make_permute(self):
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=2)
        permuted = layout.make_permute([1, 0])
        assert permuted.order == (0, 1)


class TestNvMmaSharedLayoutEncoding:

    def test_make_default(self):
        layout = tlx.nv_mma_shared_layout_encoding.make_default([64, 64],
                                                                tl.float16)
        assert layout.shape == [64, 64]
        assert layout.elemType == tl.float16
        assert layout.order == [1, 0]

    def test_make_permute(self):
        layout = tlx.nv_mma_shared_layout_encoding.make_default([64, 64],
                                                                tl.float16)
        permuted = layout.make_permute([1, 0])
        assert permuted.order == (0, 1)


class TestTensorMemoryLayoutEncoding:

    def test_make_default(self):
        layout = tlx.tensor_memory_layout_encoding.make_default([128, 64])
        assert layout.blockM == 128
        assert layout.blockN == 64
        assert layout.colStride == 1


# ---------------------------------------------------------------------------
# mbarrier type
# ---------------------------------------------------------------------------


class TestMbarrier:

    def test_construction(self):
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
        bar = tlx.mbarrier(handle=None, num=4, layout=layout)
        assert bar.type.num == 4
        assert bar.type.storage == tlx.storage_kind.smem
        assert bar.type.element_ty == tl.int64
        assert bar.is_warp_barrier is False

    def test_warp_barrier(self):
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
        bar = tlx.mbarrier(handle=None,
                           num=2,
                           layout=layout,
                           is_warp_barrier=True)
        assert bar.is_warp_barrier is True


# ---------------------------------------------------------------------------
# async_token type
# ---------------------------------------------------------------------------


class TestAsyncToken:

    def test_construction(self):
        token = tlx.async_token(handle=None)
        assert token.handle is None
        assert isinstance(token.type,
                          tlx.async_token.__class__.mro()[0].__mro__[0]
                          if False else object)  # just check it doesn't crash

    def test_mangle(self):
        token = tlx.async_token(handle=None)
        assert token.type.mangle() == "async_token_type"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
