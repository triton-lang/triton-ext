"""Kernel-argument type tables.

A scalar type must appear in both: ty_to_cpp spells the launcher's parameter
and _SCALAR_PACK_INFO says how to pack it, so a type in one and not the other
is a launch-time failure.
"""

TY_TO_CPP = {
    "i1": "int32_t",
    "u1": "int32_t",
    "i8": "int8_t",
    "u8": "uint8_t",
    "i16": "int16_t",
    "u16": "uint16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "fp16": "float",
    "bf16": "float",
    "fp32": "float",
    "fp64": "float",
}

# Scalar type -> (struct.pack format char, byte size, alignment)
SCALAR_PACK_INFO = {
    "i1": ("b", 1, 1),
    "u1": ("B", 1, 1),
    "i8": ("b", 1, 1),
    "u8": ("B", 1, 1),
    "i16": ("h", 2, 2),
    "u16": ("H", 2, 2),
    "i32": ("i", 4, 4),
    "i64": ("q", 8, 8),
    "u32": ("I", 4, 4),
    "u64": ("Q", 8, 8),
    "fp16": ("e", 2, 2),
    # _pack_scalars handles bf16 itself; the pack char here is only used for
    # size/alignment.
    "bf16": ("e", 2, 2),
    "fp32": ("f", 4, 4),
    # MSL has no double: the emitter reads fp64 args as a 4-byte float but
    # still advances the layout cursor by 8.
    "fp64": ("f", 8, 8),
}
