"""Apple GPU Triton backend. Compiles Triton kernels through
TTIR -> TTGIR -> MSL -> metallib, then dispatches via
MTLComputeCommandEncoder.
"""

from dataclasses import dataclass, field, fields
import hashlib
import os
import re
import subprocess
import tempfile

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from triton_apple_backend.hw_constants import SG_FRAG_DIM as _SG_FRAG_DIM
from triton_apple_backend.hw_constants import TARGET as _TARGET
from triton_apple_backend.hw_constants import target_arch as _target_arch
from triton_apple_backend import PLUGIN_LIBRARY
from triton_apple_backend.hw_constants import WARP_SIZE as _WARP_SIZE

_plugin = passes.plugin

# Set by the emit-msl pass; also spelled in agpu/plan/LaunchPlan.h.

# Also spelled in agpu/plan/PoolPlan.h.

_MSL_PREAMBLE_END = 'using namespace metal;\n'


def _disable_fp_contraction(msl):
    """Insert the fp-contract pragma for enable_fp_fusion=False. Metal
    contracts `a*b+c` into an FMA by default and MTLCompileOptions has no knob
    for it.
    """
    pragma = '#pragma METAL fp contract(off)\n'
    if pragma in msl:
        return msl
    at = msl.find(_MSL_PREAMBLE_END)
    if at < 0:
        raise RuntimeError(
            "cannot honour enable_fp_fusion=False: no MSL preamble to anchor "
            "the fp-contract pragma to")
    at += len(_MSL_PREAMBLE_END)
    return msl[:at] + pragma + msl[at:]


def _pmaybe_enable_debug(pm):
    if os.environ.get('TRITON_MSL_DEBUG'):
        pm.enable_debug()


def _metallib_from_source(msl):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'k.metal')
        air = os.path.join(d, 'k.air')
        lib = os.path.join(d, 'k.metallib')
        with open(src, 'w') as f:
            f.write(msl)
        for argv in ([
                'xcrun', 'metal', '-c', '-fmetal-math-mode=safe',
                '-fmetal-math-fp32-functions=fast', src, '-o', air
        ], ['xcrun', 'metallib', air, '-o', lib]):
            got = subprocess.run(argv, capture_output=True, text=True)
            if got.returncode != 0:
                raise RuntimeError(f"{argv[1]} failed: {got.stderr.strip()}")
        with open(lib, 'rb') as f:
            return f.read()


def _inert(default):
    """An option this backend accepts but never reads. Stays out of the cache
    key."""
    return field(default=default, metadata={"codegen": False})


@dataclass(frozen=True)
class MetalOptions:
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    arch: str = "apple_m"
    backend_name: str = _TARGET
    # SIMD width is 32 on every Apple GPU family; the emitter hardcodes it.
    warp_size: int = _inert(_WARP_SIZE)

    # Triton requires these fields; the emitter does not read them here.
    simdgroup_m: int = _inert(_SG_FRAG_DIM)
    simdgroup_n: int = _inert(_SG_FRAG_DIM)
    simdgroup_k: int = _inert(_SG_FRAG_DIM)

    debug: bool = False
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = _inert(False)
    instrumentation_mode: str = "none"
    fpsan_homomorphic_casts: bool = _inert(False)
    # Not inert: `semantic.binary_op_sanitize_overflow_impl` reads it and
    # changes the IR, so it has to key the cache.
    sanitize_overflow: bool = False
    allowed_dot_input_precisions: tuple = ("ieee", )
    default_dot_input_precision: str = "ieee"
    supported_fp8_dtypes: tuple = ("fp8e4nv", "fp8e5", "fp8e4b8", "fp8e5b16")
    # Read by semantic.dot for any fp8e4b8/fp8e5b16 operand. Empty: both bias
    # variants are implemented directly.
    deprecated_fp8_dot_operand_dtypes: tuple = ()
    # Hopper-only imprecise fp8 accumulation; 0 means never, as on AMD.
    max_num_imprecise_acc_default: int = 0
    extern_libs: tuple = _inert(())

    def __post_init__(self):
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"
        assert self.warp_size == _WARP_SIZE, \
               f"warp_size is fixed at {_WARP_SIZE} by the hardware"
        for name in ("simdgroup_m", "simdgroup_n", "simdgroup_k"):
            assert getattr(self, name) == _SG_FRAG_DIM, \
                   f"{name} is fixed at {_SG_FRAG_DIM} by the hardware"

    def hash(self):
        keyed = {
            f.name: getattr(self, f.name)
            for f in fields(self) if f.metadata.get("codegen", True)
        }
        return hashlib.md5(str(keyed).encode()).hexdigest()


class MetalBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == _TARGET

    def __init__(self, target: GPUTarget):
        super().__init__(target)
        self.target = target
        self.binary_ext = "metallib"

    def parse_options(self, opts) -> MetalOptions:
        args = {
            k: opts[k]
            for k in MetalOptions.__dataclass_fields__ if k in opts
        }
        return MetalOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self, options):

        def min_dot_size(lhs_type, rhs_type):
            return (8, 8, 8)

        return {"min_dot_size": min_dot_size}

    def get_module_map(self):
        return {}

    def get_target_name(self, options) -> str:
        return _target_arch(options.arch)

    def load_dialects(self, ctx):
        ir.load_dialects(ctx)

    def hash(self):
        h = hashlib.sha256()
        if PLUGIN_LIBRARY is not None:
            st = PLUGIN_LIBRARY.stat()
            h.update(
                f"{PLUGIN_LIBRARY}:{st.st_size}:{st.st_mtime_ns}".encode())
        h.update(__file__.encode())
        try:
            h.update(str(os.stat(__file__).st_mtime_ns).encode())
        except OSError:
            pass
        return f"msl-v0.1-{h.hexdigest()[:16]}"

    def make_ttir(self, mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        _pmaybe_enable_debug(pm)
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    def make_ttgir(self, mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        _pmaybe_enable_debug(pm)

        passes.ttir.add_convert_to_ttgpuir(pm, _target_arch(options.arch),
                                           options.num_warps,
                                           options.warp_size, options.num_ctas)

        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)

        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)

        # No software pipelining: the ttgpuir pipeliner needs an async copy to
        # overlap and staging here lowers to a synchronous threadgroup copy.
        pm.run(mod, 'make_ttgir')
        metadata["shared"] = mod.get_int_attr("ttg.shared") or 0
        return mod

    def make_msl(self, mod, metadata, options):
        dump = os.environ.get('TRITON_MSL_DUMP')

        # The TTGIR written alongside the MSL below. Captured here because the
        # pass rewrites `mod` in place. `#loc`s are dropped: they quote the
        # cache directory and differ run to run.
        _ttgir_for_dump = ''
        if dump:
            _ttgir_for_dump = '\n'.join(line for line in str(mod).splitlines()
                                        if not line.startswith('#loc'))
        with tempfile.NamedTemporaryFile(suffix='.metal', delete=False) as f:
            msl_path = f.name
        pm = ir.pass_manager(mod.context)
        _pmaybe_enable_debug(pm)
        _plugin.add_emit_msl(pm, [msl_path])
        try:
            pm.run(mod, 'make_msl')
        except Exception:
            os.unlink(msl_path)
            raise
        with open(msl_path, 'r') as f:
            msl = f.read()
        os.unlink(msl_path)
        if not options.enable_fp_fusion:
            msl = _disable_fp_contraction(msl)
        if os.environ.get('TRITON_MSL_DEBUG'):
            print("=== emitted MSL ===")
            print(msl)
        m = re.search(r'kernel void (\w+)\(', msl)
        if not m:
            raise RuntimeError("no 'kernel void' entry found in emitted MSL")
        if dump:
            if os.path.isdir(dump):
                # Keyed on the input (ttgir + options), so two emitters' runs
                # of one kernel get the same name.
                import hashlib
                key = hashlib.sha1((str(_ttgir_for_dump) +
                                    options.hash()).encode()).hexdigest()[:8]
                dump = os.path.join(dump, f'{m.group(1)}.{key}')
            else:
                dump = os.path.splitext(dump)[0]
            for suffix, text in (('.metal', msl), ('.ttgir', _ttgir_for_dump)):
                with open(dump + suffix, 'w') as df:
                    df.write(text)
        metadata["name"] = m.group(1)
        metadata["shared"] = 0
        return msl

    def make_msl_metallib(self, msl, metadata, options):
        return _metallib_from_source(msl)

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, meta: self.make_ttir(src, meta, options)
        stages["ttgir"] = lambda src, meta: self.make_ttgir(src, meta, options)
        stages["msl"] = lambda src, meta: self.make_msl(src, meta, options)
        stages["metallib"] = lambda src, meta: self.make_msl_metallib(
            src, meta, options)
