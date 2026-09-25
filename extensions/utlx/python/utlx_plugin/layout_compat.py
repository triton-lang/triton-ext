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


#: Installed onto TLXOpBuilder. The remaining fork factories
#: (make_dot_operand_encoding_attr, make_nv_mma_encoding_attr,
#: make_dummy_register_layout_attr, make_dummy_tmem_layout_attr) are register
#: layouts or TLX-dialect placeholders with no upstream equivalent reachable
#: this way; they are deliberately absent so the AttributeError still names
#: them rather than a wrong layout being built.
FACTORIES = {
    "make_swizzled_shared_encoding_attr":
    make_swizzled_shared_encoding_attr,
    "make_nv_mma_shared_encoding_attr":
    make_nv_mma_shared_encoding_attr,
    "make_tensor_memory_encoding_attr":
    make_tensor_memory_encoding_attr,
    "make_tensor_memory_scales_encoding_attr":
    make_tensor_memory_scales_encoding_attr,
}
