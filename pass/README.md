# Pass Extension Development

Pass extensions allow you to create custom MLIR passes that can be dynamically loaded into the Triton compiler at runtime. These extensions are built as shared libraries and can extend Triton's compilation pipeline without modifying the core Triton codebase.

## Infrastructure Dependency

All pass extensions depend on the shared infrastructure library located in `infra/pass/`. This library provides:

- **`TritonExtPassInfra.h`**: Header file defining the pass extension API, including functions for registering and loading passes (`tritonAddPluginPass`, `tritonRegisterPluginPass`, `tritonEnumeratePluginPasses`, `TritonExtLoadPass`).

- **`TritonExtPassInfra.cpp`**: Implementation of the pass extension infrastructure that bridges between Triton's plugin system and MLIR's pass management.

- **`TritonExtPass.cpp`**: Automatic registration code that pass extensions include at the end of their implementation files to enable automatic discovery and registration.

When building a pass extension, you must link against `TritonExtPassInfra` (as shown in the CMakeLists.txt example below). This shared library handles the integration with Triton's plugin system and provides the necessary registration mechanisms.

## Example: LoopSplit Extension

The `LoopSplit` extension in `pass/LoopSplit/` demonstrates how to create a pass extension:

1. **Pass Definition** (`Passes.td`): Uses MLIR TableGen to define the pass metadata.
   ```tablegen
   def TritonLoopSplit : Pass<"triton-loop-split", "mlir::ModuleOp"> {
     let summary = "Loop splitter";
     let description = [{
       Bisect a loop when beneficial.
     }];
     let constructor = "mlir::triton_ext::passes::createLoopSplitPass()";
   }
   ```

2. **Pass Implementation** (`LoopSplit.cpp`): Implements the pass logic. At the end of the file, include the extension registration from the infrastructure library:
   ```cpp
   // Include the MLIR pass extension registry implementation
   #include "TritonExtPass.cpp"
   ```

3. **Extension Configuration** (`triton_ext.conf`): Specifies the extension name and status:
   ```
   loop_split;experimental
   ```

4. **CMake Configuration** (`CMakeLists.txt`): Sets up the build for the extension as a shared library with proper dependencies. The CMakeLists.txt should define:
   - `TRITON_EXT_CLASS`: The pass class name (e.g., `LoopBisectPass`)
   - The extension will automatically use `TRITON_EXT_NAME` from the project name

### Creating a New Pass Extension

To create a new pass extension:

1. Create a new directory under `pass/` (e.g., `pass/MyPass/`).

2. Create `CMakeLists.txt`:
   ```cmake
   project(my_pass)
   set(TRITON_EXT_CLASS MyPassClass)
   
   # Only build this extension if it is enabled in TRITON_EXT_NAMES
   list(FIND TRITON_EXT_NAMES ${PROJECT_NAME} _index)
   list(LENGTH TRITON_EXT_NAMES _size)
   if(NOT _index EQUAL -1 OR _size EQUAL 0)
       message(STATUS "TRITON EXT: building ${PROJECT_NAME}")
       
       # Add Pass tablegen
       set(EXT_Include "TritonExt${TRITON_EXT_CLASS}IncGen")
       set(LLVM_TARGET_DEFINITIONS Passes.td)
       mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
       add_public_tablegen_target(${EXT_Include})
       
       add_compile_definitions(
           TRITON_EXT_NAME=${PROJECT_NAME} 
           TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
       )
       
       add_mlir_library(${PROJECT_NAME}
           MyPass.cpp
           SHARED
           DEPENDS
           TritonTableGen
           TritonExtPassInfra
           ${EXT_Include}
           LINK_LIBS PUBLIC
           TritonExtPassInfra
           MLIRPass
           LLVMSupport
           MLIRSupport
       )
       
       set_target_properties(${PROJECT_NAME} PROPERTIES
           LIBRARY_OUTPUT_DIRECTORY
           "${TRITON_EXT_LIBRARY_DIR}")
   endif()
   ```

3. Create `Passes.td` with your pass definition.

4. Create the implementation file (e.g., `MyPass.cpp`) that:
   - Defines the pass class (e.g., `MyPassClass`)
   - Implements the pass logic
   - Includes `TritonExtPass.cpp` (from `infra/pass/`) at the end of the file for automatic registration

5. Create `triton-ext.conf` with the extension name and status:
   ```
   my_pass;experimental
   ```

6. The extension will be automatically discovered by the build system (no need to manually add to `pass/CMakeLists.txt`).

