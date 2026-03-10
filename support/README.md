# Pass Extension Infrastructure

All pass extensions depend on this shared infrastructure library. This library provides:
- **`Export.{h,cpp}`**: an implementation for Triton's extension API (`tritonAddPluginPass()`,
  `tritonRegisterPluginPass()`, `tritonEnumeratePluginPasses()`, etc.)
- **`Export{Pass,Dialect}.cpp`**: automatic registration code that extensions include at the end of their implementation
  files to enable automatic discovery and registration.

## Build

This library is built as part of the project build; see top-level [README.md](../README.md).

## Use

1. Link against `TritonExtensionSupport` when building your extension (see an [example](../pass/LoopSplit/CMakeLists.txt)).

2. Include `ExportPass.cpp` at the end of your pass implementation file to enable automatic registration (see an
   [example](../pass/LoopSplit/LoopSplit.cpp)).

3. Inject your extension into Triton at runtime (e.g., `TRITON_PASS_PLUGIN_PATH=/path/to/libmy_pass.so python ...`);
   this library ensures Triton can discover and load your extension.
