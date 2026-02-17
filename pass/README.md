# Pass Extension Development

Pass extensions allow you to create custom MLIR passes that can be dynamically loaded into the Triton compiler at
runtime. These extensions are built as shared libraries and can extend Triton's compilation pipeline without modifying
the core Triton codebase.

## Dependencies

All pass extensions depend on the `TritonExtPassInfra` shared library located in [`infra/pass/`](../infra/pass/). See
it's [documentation](./infra/pass/README.md) for more details. Pass extension libraries may also depend on LLVM/MLIR and
Triton shared libraries; see the top-level [README](../README.md) for more details on setting up these dependencies.

## Example

The `LoopSplit` extension in `pass/LoopSplit/` demonstrates how to create a pass extension:

1. **Pass Definition**: Use MLIR TableGen to define the pass metadata (see [`Passes.td`](./LoopSplit/Passes.td)).

2. **Pass Implementation**: Implement the pass logic (see [`LoopSplit.cpp`](./LoopSplit/LoopSplit.cpp)). At the end of
   the file, include the extension registration from the infrastructure library:
   ```cpp
   // Include the MLIR pass extension registry implementation
   #include "TritonExtPass.cpp"
   ```

3. **Extension Configuration**: Specify the extension name and status in a `triton-ext.conf` file
   (see [`triton-ext.conf`](./LoopSplit/triton-ext.conf)):
   ```
   loop_split;experimental
   ```

4. **CMake Configuration**: Set up the build as a shared library with proper dependencies (see
   [`CMakeLists.txt`](./LoopSplit/CMakeLists.txt)). The `CMakeLists.txt` should define:
   - `TRITON_EXT_CLASS`: The pass class name (e.g., `LoopBisectPass`)
   - `TRITON_EXT_NAME`: The extension name (e.g., `loop_bisect`); this is automatically set via
     the `triton_ext_pass_setup` CMake function
