"""The `make_*` layout-attribute factories that stock Triton does not provide.

Meta's fork adds a family of `make_*_encoding_attr` methods to its
`GluonOpBuilder`, and the TLX layout encodings in `tlx/language/tlx/types.py`
call them from `to_ir`. Stock Triton has none of them, so every `to_ir` fails
with `'TLXOpBuilder' object has no attribute 'make_..._encoding_attr'`.

Upstream does expose the same attributes, under different names and a different
calling convention: `get_swizzled_shared_layout`, `get_nvmma_shared_layout`,
`get_tensor_memory_layout` and `get_tensor_memory_scales_layout`. Two
differences have to be bridged here rather than at the call sites:

* **CGA layout.** The fork passes `numCTAsPerCGA` / `numCTASplit` /
  `numCTAOrder` as three lists; upstream takes `cgaBases`, a list of basis
  vectors, where the empty list means one CTA per CGA. uTLX only ever emits
  single-CTA layouts, so the conversion is empty-list-or-reject.
* **NVMMA swizzling.** The fork's attribute builder derives
  `swizzlingByteWidth` and `transposed` from the shape; upstream's Python
  getter wants them precomputed. `_nvmma_swizzle` below reproduces the
  derivation in `TritonGPUAttrDefs.td`'s `NVMMASharedEncodingAttr` builder
  exactly, which is safe only because the CTA layout is known to be trivial
  (so `shapePerCTA == shape`).

These are installed onto the synthesized `TLXOpBuilder` class; see
`_install_layout_compat` in `__init__.py`.
"""

from __future__ import annotations

import re


def _cga_bases(num_ctas_per_cga, num_cta_split=None, num_cta_order=None):
    """Convert the fork's three CTA lists into upstream's cgaBases.

    Upstream builds the CGA layout from basis vectors and treats an empty list
    as the 1-CTA layout. uTLX emits nothing else, so anything asking for a real
    cluster is rejected rather than silently flattened to one CTA.
    """
    for name, value in (("numCTAsPerCGA", num_ctas_per_cga), ("numCTASplit",
                                                              num_cta_split)):
        if value is None:
            continue
        if any(int(v) != 1 for v in value):
            raise NotImplementedError(
                f"uTLX builds single-CTA layouts only, but {name}={list(value)}."
                " Upstream's Python layout getters take cgaBases basis vectors;"
                " a multi-CTA layout needs those to be constructed explicitly."
            )
    return []


def _nvmma_swizzle(shape, order, elem_bitwidth, fp4_padded):
    """Reproduce NVMMASharedEncodingAttr's shape-driven builder.

    Mirrors the AttrBuilder in TritonGPUAttrDefs.td. Valid here because
    `_cga_bases` has already established a 1-CTA layout, so shapePerCTA is
    shape.
    """
    shape = [int(d) for d in shape]
    order = [int(o) for o in order]
    packing_factor = 2 if fp4_padded else 1
    contig_dim_bytes = shape[order[0]] * packing_factor * elem_bitwidth // 8

    if contig_dim_bytes >= 128 and contig_dim_bytes % 128 == 0:
        swizzle_byte_width = 128
    elif contig_dim_bytes >= 64 and contig_dim_bytes % 64 == 0:
        swizzle_byte_width = 64
    elif contig_dim_bytes >= 32 and contig_dim_bytes % 32 == 0:
        swizzle_byte_width = 32
    else:
        swizzle_byte_width = 0

    flatten_outer_dim = 1
    for i in range(1, len(shape)):
        flatten_outer_dim *= shape[order[i]]
    if len(shape) < 2 or flatten_outer_dim < 8:
        swizzle_byte_width = 0

    transposed = len(order) > 1 and order[0] == 0
    return swizzle_byte_width, transposed


def _elem_bitwidth(elem_type):
    """Bit width of a TLX layout's element type.

    Callers pass whatever the fork's builder took, which is usually an already
    lowered MLIR type (`elemType.to_ir(builder)`) but is sometimes the
    `tl.dtype` itself. The MLIR type exposes no width accessor to Python, so
    fall back to parsing its name: Triton spells these `f16`, `bf16`, `f32`,
    `i8`, `f8E4M3FN`, `f4E2M1FN` ... where the first run of digits is the
    width.
    """
    width = getattr(elem_type, "primitive_bitwidth", None)
    if width is not None:
        return int(width)

    match = re.match(r"[a-zA-Z]*?(\d+)", str(elem_type))
    if not match:
        raise TypeError(
            f"cannot determine the element bit width of {elem_type!r}; "
            "needed to derive the NVMMA swizzling byte width")
    return int(match.group(1))


def make_swizzled_shared_encoding_attr(self, vectorSize, perPhase, maxPhase,
                                       order, numCTAsPerCGA, numCTASplit,
                                       numCTAOrder):
    return self.get_swizzled_shared_layout(
        int(vectorSize), int(perPhase), int(maxPhase), [int(o) for o in order],
        _cga_bases(numCTAsPerCGA, numCTASplit, numCTAOrder))


def make_nv_mma_shared_encoding_attr(self, shape, order, elemType,
                                     numCTAsPerCGA, numCTASplit, numCTAOrder,
                                     fp4Padded, swizzled):
    cga_bases = _cga_bases(numCTAsPerCGA, numCTASplit, numCTAOrder)
    elem_bitwidth = _elem_bitwidth(elemType)
    swizzle_byte_width, transposed = _nvmma_swizzle(shape, order,
                                                    elem_bitwidth,
                                                    bool(fp4Padded))
    if not swizzled:
        swizzle_byte_width = 0
    return self.get_nvmma_shared_layout(swizzle_byte_width, elem_bitwidth,
                                        transposed, bool(fp4Padded), cga_bases,
                                        len(shape))


def _next_pow2(x):
    """llvm::NextPowerOf2: the smallest power of two strictly above ``x``."""
    return 1 << int(x).bit_length()


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def _prod(values):
    result = 1
    for v in values:
        result *= v
    return result


def _out_sizes(bases, rank):
    """Out-dim sizes a LinearLayout infers from its bases."""
    sizes = [1] * rank
    for basis in bases:
        for d, v in enumerate(basis):
            sizes[d] = max(sizes[d], _next_pow2(v))
    return sizes


def _grow_to(bases, dims, shape):
    """ensureLayoutNotSmallerThan for an offset-only layout.

    ``dims`` is the layout's out-dim order, which decides which dimension's
    extra offset bits come first.
    """
    rank = len(shape)
    for d in dims:
        size = _out_sizes(bases, rank)[d]
        target = shape[d] if _is_pow2(shape[d]) else _next_pow2(shape[d])
        while size < target:
            basis = [0] * rank
            basis[d] = size
            bases.append(basis)
            size *= 2
    return bases


def _shrink_to(bases, dims, shape):
    """ensureLayoutNotLargerThan for an offset-only layout.

    Offsets past ``shape`` along a dim broadcast instead: its largest
    components are zeroed, one power of two at a time, until it fits.
    """
    for d in dims:
        span = _out_sizes(bases, len(shape))[d]
        for value in sorted({b[d] for b in bases if b[d]}, reverse=True):
            if span <= shape[d]:
                break
            for basis in bases:
                if basis[d] == value:
                    basis[d] = 0
            span //= 2
    return bases


def _fit_to(bases, dims, shape):
    """combineCtaCgaWithShape for one CTA: grow, then clamp, to ``shape``."""
    return _shrink_to(_grow_to(bases, dims, shape), dims, shape)


def _transpose(bases, dims, order):
    """transposeLinearLayout: permute out-dim positions, keeping the names."""
    out = []
    for basis in bases:
        positional = [basis[d] for d in dims]
        permuted = [positional[i] for i in order]
        new = [0] * len(basis)
        for pos, d in enumerate(dims):
            new[d] = permuted[pos]
        out.append(new)
    return out


def _reshape(bases, src_shape, dst_shape):
    """reshapeLayout: reinterpret row-major coordinates in a new shape."""
    out = []
    for basis in bases:
        flat = 0
        for d, v in enumerate(basis):
            flat = flat * src_shape[d] + v
        new = [0] * len(dst_shape)
        for d in reversed(range(len(dst_shape))):
            new[d] = flat % dst_shape[d]
            flat //= dst_shape[d]
        out.append(new)
    return out


def _nvmma_tiled_offset_bases(shape, elem_bitwidth, swizzle_byte_width,
                              transposed, fp4_padded):
    """``offset`` bases of an NVMMA layout over ``shape``, TMA tiled mode.

    A port of ``nvmmaSharedToLinearLayout(shape, atom, TMAMode::Tiled)``
    (LinearLayoutConversions.cpp) for the single-CTA layouts uTLX emits, so
    shape is shapePerCTA and there are no block bases.

    Upstream does expose this conversion, as ``GluonOpBuilder.to_linear_layout``,
    but its result goes through ``layoutToGluon``, which caches the Gluon layout
    classes in a C++ static it never releases. They keep ``ir.builder`` and
    ``GluonOpBuilder`` alive past interpreter shutdown, and nanobind reports
    them, every builder method and four ``ir.attribute`` default arguments as
    leaked when the process exits.
    """
    rank = len(shape)
    if not all(_is_pow2(d) for d in shape):
        raise ValueError(
            f"tile_to_shape needs a power-of-two tile shape, got {shape}")

    # getTMABlockShape, tiled mode, packedSize=true.
    tma = list(shape)
    contig = 0 if transposed else rank - 1
    if fp4_padded:
        tma[contig] *= 2
    tma = [min(d, 256) for d in tma]
    if swizzle_byte_width:
        contig_size = 8 * swizzle_byte_width // elem_bitwidth
        if tma[contig] < contig_size:
            raise ValueError(
                f"tile shape {shape} is too small along dim {contig} for a "
                f"{swizzle_byte_width}B swizzled NVMMA layout: need at least "
                f"{contig_size} elements")
        tma[contig] = contig_size
    if fp4_padded:
        tma[contig] //= 2
    # Moves the outer dim innermost, the order the transposed layout is built in.
    rotated = tma[1:] + tma[:1] if transposed else list(tma)
    transpose_order = [rank - 1] + list(range(rank - 1))

    if swizzle_byte_width == 0:
        # Row-major identity, built innermost dim first, so the layout's out
        # dims run dim<rank-1> .. dim0.
        dims = list(reversed(range(rank)))
        bases = []
        stride = [1] * rank
        for d in dims:
            for _ in range(rotated[d].bit_length() - 1):
                basis = [0] * rank
                basis[d] = stride[d]
                bases.append(basis)
                stride[d] *= 2
        if transposed:
            bases = _transpose(bases, dims, transpose_order)
        return _fit_to(bases, dims, shape)

    # getCoreMatrixLinearLayout: one 8-row core matrix, swizzled per row.
    vec = 128 // elem_bitwidth
    per_phase = 128 // swizzle_byte_width
    max_phase = swizzle_byte_width // 16
    tile_cols = 8 * max(16, swizzle_byte_width) // elem_bitwidth
    core = []
    col = 1
    while col < tile_cols:
        core.append([0, col // 16 * 8 + col % 8 if fp4_padded else col])
        col *= 2
    row = 1
    while row < 8:
        swizzle = vec * ((row // per_phase) % max_phase)
        if fp4_padded:
            swizzle = swizzle // 16 * 8 + swizzle % 8
        core.append([row, swizzle])
        row *= 2

    # Collapse the outer dims into one, tile the core matrix over it, then
    # reshape back to the N-d (pre-transpose) shape.
    collapsed = [1, tma[-1]]
    for d in tma[:-1]:
        collapsed[0] *= d
    if transposed:
        collapsed.reverse()
    core_rows, core_cols = _out_sizes(core, 2)
    packing = 2 if fp4_padded else 1
    if collapsed[1] * packing < core_cols or collapsed[0] < core_rows:
        raise ValueError(f"tile shape {shape} is smaller than one {core_rows}x"
                         f"{core_cols // packing} NVMMA core matrix")
    bases = _grow_to(core, [0, 1], collapsed)
    sizes = _out_sizes(bases, 2)
    if sizes[0] * sizes[1] != _prod(rotated):
        raise ValueError(f"cannot tile an NVMMA layout over shape {shape}")
    bases = _reshape(bases, sizes, rotated)
    dims = list(range(rank))
    if transposed:
        bases = _transpose(bases, dims, transpose_order)
    return _fit_to(bases, dims, shape)


def make_nv_mma_tiled_shared_linear_encoding_attr(self, atomShape, order,
                                                  elemType, numCTAsPerCGA,
                                                  numCTASplit, numCTAOrder,
                                                  fp4Padded, swizzled,
                                                  tileShape, alignment):
    """An NVMMA atom tiled out to ``tileShape``, as a #ttg.shared_linear.

    The fork converts the atom with ``nvmmaSharedToLinearLayout(...,
    TMAMode::Tiled)``; ``_nvmma_tiled_offset_bases`` does the same here, and the
    bases are wrapped with the requested alignment instead of the atom's own.
    """
    _cga_bases(numCTAsPerCGA, numCTASplit, numCTAOrder)
    elem_bitwidth = _elem_bitwidth(elemType)
    swizzle_byte_width, transposed = _nvmma_swizzle(atomShape, order,
                                                    elem_bitwidth,
                                                    bool(fp4Padded))
    if not swizzled:
        swizzle_byte_width = 0
    bases = _nvmma_tiled_offset_bases([int(d) for d in tileShape],
                                      elem_bitwidth, swizzle_byte_width,
                                      transposed, bool(fp4Padded))
    return self.get_shared_linear_layout(bases, [], int(alignment))


def make_tensor_memory_encoding_attr(self, blockM, blockN, colStride,
                                     CTASplitM, CTASplitN):
    cga_bases = _cga_bases([CTASplitM, CTASplitN])
    return self.get_tensor_memory_layout(
        [int(blockM), int(blockN)], int(colStride), cga_bases, False, False)


def make_tensor_memory_scales_encoding_attr(self, CTASplitM, CTASplitN):
    cga_bases = _cga_bases([CTASplitM, CTASplitN])
    # The fork's attribute has no repOrder field; upstream's does and offers no
    # default. "mnThenK" is the layout TMEM scale operands use.
    return self.get_tensor_memory_scales_layout(cga_bases, "mnThenK")


def make_linear_encoding_attr(self, reg_bases, lane_bases, warp_bases, shape):
    """The fork's register-layout factory, on upstream's getter.

    Upstream spells the same #ttg.linear attribute
    ``get_distributed_linear_layout`` and takes a fourth basis list for the CTA
    level, between warp and shape. uTLX emits single-CTA layouts, which is the
    empty list -- the same convention ``_cga_bases`` enforces above.
    """
    return self.get_distributed_linear_layout(reg_bases, lane_bases,
                                              warp_bases, [],
                                              [int(d) for d in shape])


#: Installed onto TLXOpBuilder. The remaining fork factories
#: (make_dot_operand_encoding_attr, make_nv_mma_encoding_attr,
#: make_dummy_register_layout_attr, make_dummy_tmem_layout_attr) are TLX-dialect
#: placeholders or layouts with no upstream equivalent reachable this way; they
#: are deliberately absent so the AttributeError still names them rather than a
#: wrong layout being built.
FACTORIES = {
    "make_swizzled_shared_encoding_attr": make_swizzled_shared_encoding_attr,
    "make_nv_mma_shared_encoding_attr": make_nv_mma_shared_encoding_attr,
    "make_nv_mma_tiled_shared_linear_encoding_attr":
    make_nv_mma_tiled_shared_linear_encoding_attr,
    "make_tensor_memory_encoding_attr": make_tensor_memory_encoding_attr,
    "make_tensor_memory_scales_encoding_attr":
    make_tensor_memory_scales_encoding_attr,
    "make_linear_encoding_attr": make_linear_encoding_attr,
}
