"""Layout-encoding factories for Triton builds that do not provide them.

Meta's in-tree TLX registers a family of ``make_*_encoding_attr`` methods onto
``TritonOpBuilder`` from ``third_party/tlx/dialect/triton_tlx.cc``.  A stock
Triton wheel has no such methods, so ``types.py`` fails with
``'TLXOpBuilder' object has no attribute 'make_nv_mma_shared_encoding_attr'``
as soon as a tlx kernel materialises a buffer type.

Stock Triton does expose the same MLIR attribute constructors, under gluon's
``get_*_layout`` names, with two differences this module bridges:

* the CGA layout is passed as explicit linear "block" bases rather than as
  ``(CTAsPerCGA, CTASplitNum, CTAOrder)`` split parameters, and
* ``get_nvmma_shared_layout`` only takes an explicit swizzle byte width, while
  the in-tree builder also accepts ``(shape, order)`` and infers the width.

Both gaps are pure arithmetic, mirrored here from
``CGAEncodingAttr::fromSplitParams`` and the ``NVMMASharedEncodingAttr``
shape/order ``AttrBuilder`` respectively.

The adapters are installed only for names the host builder does not already
have, so on an in-tree Meta build this module is inert.
"""

import re

# Matches the leading mnemonic of an MLIR scalar type -- ``f16``, ``bf16``,
# ``i8``, ``f8E4M3FN``, ``f4E2M1FN`` -- whose first run of digits is the
# element bit width.  The Python `ir.type` binding exposes no bit-width
# accessor, so the printed form is the only thing to go on.
_ELEMENT_BITWIDTH = re.compile(r"^(?:bf|fp|f|ui|si|i|u)(\d+)")


def element_bitwidth(elem_type):
    """Bit width of an MLIR scalar type, from its printed mnemonic."""
    text = str(elem_type)
    match = _ELEMENT_BITWIDTH.match(text)
    if match is None:
        raise ValueError(f"cannot determine the element bit width of {text!r}")
    return int(match.group(1))


def cga_bases(ctas_per_cga, cta_split_num, cta_order):
    """Split parameters -> the "block" bases of a ``#ttg.cga`` layout.

    Mirrors ``CGAEncodingAttr::fromSplitParams``, which builds

        prod over i of  identity1D(split[d]) * zeros1D(total[d] / split[d])

    with ``d = CTAOrder[i]``, each factor mapping the "block" input dim onto
    output dim ``d``.  Because every iteration targets a distinct output dim,
    the direct sum degenerates to concatenating each factor's bases, so an
    identity factor contributes ``log2(split)`` bases ``1, 2, 4, ...`` placed
    at index ``d`` and a zeros factor contributes ``log2(total / split)``
    all-zero bases.  An all-ones ``CTAsPerCGA`` is the 1-CTA layout, which is
    spelled as an empty basis list.
    """
    rank = len(cta_order)
    if all(int(c) == 1 for c in ctas_per_cga):
        return []

    bases = []
    for dim in cta_order:
        dim = int(dim)
        split = int(cta_split_num[dim])
        total = int(ctas_per_cga[dim])
        if split == 0 or total % split != 0:
            raise ValueError(f"invalid CGA parameters: CTAsPerCGA={list(ctas_per_cga)}, "
                             f"CTASplitNum={list(cta_split_num)}")
        basis = 1
        while basis < split:
            vec = [0] * rank
            vec[dim] = basis
            bases.append(vec)
            basis <<= 1
        broadcast = 1
        while broadcast < total // split:
            bases.append([0] * rank)
            broadcast <<= 1
    return bases


def shape_per_cta(cta_split_num, shape):
    """Mirrors ``triton::gpu::getShapePerCTA``."""
    rank = len(shape)
    split = [int(s) for s in cta_split_num]
    if len(split) <= rank:
        split = [1] * (rank - len(split)) + split
    else:
        split = split[len(split) - rank:]
    return [shape[i] // min(shape[i], split[i]) for i in range(rank)]


def default_swizzle_byte_width(shape, order, cta_split_num, elem_bitwidth, fp4_padded):
    """Mirrors the ``NVMMASharedEncodingAttr`` (shape, order) ``AttrBuilder``.

    Picks the widest swizzle the contiguous dimension can fill, then falls back
    to unswizzled for shapes too small to tile.
    """
    per_cta = shape_per_cta(cta_split_num, shape)
    packing_factor = 2 if fp4_padded else 1
    contig_bytes = per_cta[order[0]] * packing_factor * elem_bitwidth // 8

    if contig_bytes >= 128 and contig_bytes % 128 == 0:
        width = 128
    elif contig_bytes >= 64 and contig_bytes % 64 == 0:
        width = 64
    elif contig_bytes >= 32 and contig_bytes % 32 == 0:
        width = 32
    else:
        width = 0

    flattened_outer = 1
    for i in range(1, len(per_cta)):
        flattened_outer *= per_cta[order[i]]
    if len(per_cta) < 2 or flattened_outer < 8:
        width = 0
    return width


def _make_swizzled_shared_encoding_attr(self, vector_size, per_phase, max_phase, order, ctas_per_cga, cta_split_num,
                                        cta_order):
    return self.get_swizzled_shared_layout(vector_size, per_phase, max_phase, list(order),
                                           cga_bases(ctas_per_cga, cta_split_num, cta_order))


def _make_nv_mma_shared_encoding_attr(self, shape, order, elem_type, ctas_per_cga, cta_split_num, cta_order,
                                      fp4_padded, swizzled):
    shape = [int(s) for s in shape]
    order = [int(o) for o in order]
    bitwidth = element_bitwidth(elem_type)
    # For 1D there is no transpose to speak of; the in-tree builder pins it to
    # false so that isTMACompatibleEncoding still accepts the encoding.
    transposed = len(order) > 1 and order[0] == 0
    width = (default_swizzle_byte_width(shape, order, cta_split_num, bitwidth, fp4_padded) if swizzled else 0)
    return self.get_nvmma_shared_layout(width, bitwidth, transposed, fp4_padded,
                                        cga_bases(ctas_per_cga, cta_split_num, cta_order), len(shape))


def _make_tensor_memory_encoding_attr(self, block_m, block_n, col_stride, cta_split_m, cta_split_n):
    splits = [int(cta_split_m), int(cta_split_n)]
    return self.get_tensor_memory_layout([int(block_m), int(block_n)], int(col_stride),
                                         cga_bases(splits, splits, [0, 1]), False, False)


def _make_tensor_memory_scales_encoding_attr(self, cta_split_m, cta_split_n):
    splits = [int(cta_split_m), int(cta_split_n)]
    return self.get_tensor_memory_scales_layout(cga_bases(splits, splits, [0, 1]))


ADAPTERS = {
    "make_swizzled_shared_encoding_attr": _make_swizzled_shared_encoding_attr,
    "make_nv_mma_shared_encoding_attr": _make_nv_mma_shared_encoding_attr,
    "make_tensor_memory_encoding_attr": _make_tensor_memory_encoding_attr,
    "make_tensor_memory_scales_encoding_attr": _make_tensor_memory_scales_encoding_attr,
}


def install(namespace, base):
    """Add every adapter the host builder is missing to a builder namespace."""
    for name, adapter in ADAPTERS.items():
        if not hasattr(base, name):
            namespace[name] = adapter
