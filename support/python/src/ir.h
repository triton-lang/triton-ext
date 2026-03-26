// This is a forward declaration to avoid a build error introduced in
// https://github.com/triton-lang/triton/pull/9626. The problem is that
// `PluginUtils.h` depends on this class in `python/src/ir.h` for the builder
// necessary to add custom operations. Including this file temporarily resolves
// the missing header; when that is fixed upstream (e.g.,
// https://github.com/triton-lang/triton/pull/9847), this file can be removed.
class TritonOpBuilder;
