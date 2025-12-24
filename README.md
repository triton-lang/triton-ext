# triton-plugins

A collection of out-of-tree plugins for the Triton compiler, including passes, dialects, backends, and language extensions.

## Overview

This repository provides a framework for developing and building Triton compiler plugins that can extend functionality without modifying the core Triton codebase. Plugins are built as shared libraries that can be dynamically loaded by Triton at runtime.

## Project Structure

```
triton-plugins/
├── backend/              # Backend configuration (legacy)
│   ├── name.conf        # Plugin name configuration
│   ├── compiler.py      # Backend compiler implementation
│   └── driver.py        # Backend driver implementation
├── backends/            # Backend plugins directory
│   └── CMakeLists.txt
├── passes/              # Pass plugins directory
│   ├── api/             # Pass plugin API infrastructure
│   │   ├── CMakeLists.txt
│   │   ├── PassPlugin.h
│   │   └── PassPlugin.cpp
│   ├── LoopSplit/       # Example: Loop splitting pass plugin
│   │   ├── CMakeLists.txt
│   │   ├── LoopSplit.cpp
│   │   └── Passes.td
│   └── CMakeLists.txt
├── dialects/            # Dialect plugins directory
│   └── CMakeLists.txt
├── lang-ext/            # Language extension plugins directory
│   └── CMakeLists.txt
├── CMakeLists.txt       # Root CMake configuration
└── README.md
```

### Directory Descriptions

- **`backend/`**: Contains backend configuration files, including `name.conf` which specifies the plugin name that Triton will recognize.

- **`passes/`**: Contains MLIR pass plugins. Each pass plugin is implemented as a shared library that can be loaded dynamically. The `api/` subdirectory provides the plugin API infrastructure that pass plugins use.

- **`backends/`**: Intended for backend-specific plugin implementations (currently scaffolding only).

- **`dialects/`**: Intended for custom MLIR dialect plugins (currently scaffolding only).

- **`lang-ext/`**: Intended for language extension plugins (currently scaffolding only).

## Build Process

### Prerequisites

- Triton compiler source code
- CMake (3.15 or later)
- LLVM/MLIR development shared libraries
- C++ compiler with C++17 support

### Building as Part of Triton

This plugin repository is designed to be built as part of the Triton build system. To build:

1. **Set up Triton build environment**: Ensure you have Triton's dependencies installed.

2. **Configure Triton build with plugin directory**:
   ```bash
   cmake -DTRITON_PLUGIN_DIRS=/path/to/triton-plugins ...
   ```

   Or when building via Python setup:
   ```bash
   TRITON_PLUGIN_DIRS=/path/to/triton-plugins python setup.py develop
   ```

3. **Build**: The plugins will be built automatically as part of the Triton build process.

4. **Output location**: Built plugins (shared libraries) are placed in:
   ```
   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins/
   ```

### Building Individual Plugin Components

The build system processes plugins in the following order:
1. `backends/`
2. `passes/`
3. `dialects/`
4. `lang-ext/`

Each subdirectory's `CMakeLists.txt` is responsible for building its respective plugins.

## Pass Plugin Development

### Example: LoopSplit Plugin

The `LoopSplit` plugin in `passes/LoopSplit/` demonstrates how to create a pass plugin:

1. **Pass Definition** (`Passes.td`): Uses MLIR TableGen to define the pass metadata.
   ```tablegen
   def TritonLoopSplit : Pass<"triton-loop-split", "mlir::ModuleOp"> {
     let summary = "Loop splitter";
     let description = [{
       Bisect a loop when beneficial.
     }];
     let constructor = "mlir::triton_plugin::passes::createLoopSplitPass()";
   }
   ```

2. **Pass Implementation** (`LoopSplit.cpp`): Implements the pass logic and registers it using the plugin API:
   ```cpp
   #define PLUGIN_NAME "LoopSplit"
   #define PLUGIN_PASS_CLASS mlir::triton_plugin::createLoopSplitPass
   #include "PassPlugin.h"
   ```

3. **CMake Configuration** (`CMakeLists.txt`): Sets up the build for the plugin as a shared library with proper dependencies.

### Creating a New Pass Plugin

To create a new pass plugin:

1. Create a new directory under `passes/` (e.g., `passes/MyPass/`).

2. Create `CMakeLists.txt`:
   ```cmake
   set(PLUGIN "MyPass")
   set(PLUGIN_STATUS "experimental")
   
   # Add Pass tablegen
   set(PLUGIN_Include "TritonPlugin${PLUGIN}IncGen")
   set(LLVM_TARGET_DEFINITIONS Passes.td)
   mlir_tablegen(Passes.h.inc -gen-pass-decls -name Plugins)
   add_public_tablegen_target(${PLUGIN_Include})
   
   add_mlir_library(${PLUGIN}
       MyPass.cpp
       SHARED
       DEPENDS
       TritonTableGen
       TPAPIPassInfra
       ${PLUGIN_Include}
       LINK_LIBS PUBLIC
       TPAPIPassInfra
       MLIRPass
       LLVMSupport
       MLIRSupport
   )
   ```

3. Create `Passes.td` with your pass definition.

4. Create the implementation file (e.g., `MyPass.cpp`) that defines:
   - The pass class
   - A factory function (e.g., `createMyPass()`)
   - `PLUGIN_NAME` and `PLUGIN_PASS_CLASS` macros
   - Include `PassPlugin.h`

5. Add the subdirectory to `passes/CMakeLists.txt`:
   ```cmake
   add_subdirectory(MyPass)
   ```

## Plugin API

The pass plugin API (`passes/api/PassPlugin.h`) provides:

Triton Plugin API:

- `tritonAddPluginPass()`: Add a plugin pass to a pass manager
- `tritonRegisterPluginPass()`: Register a plugin pass
- `tritonEnumeratePluginPasses()`: Enumerate available plugin passes

Local registration for the Triton Plugin API:

- `TPAPILoadPass()`: Load and register a pass plugin (Plugin side)

Pass plugins should define `PLUGIN_NAME` and `PLUGIN_PASS_CLASS` macros before including `PassPlugin.h` to enable automatic registration.

## Using Plugins at Runtime

Plugins can be loaded by Triton at runtime using the `TRITON_PASS_PLUGIN_PATH` environment variable:

```bash
export TRITON_PASS_PLUGIN_PATH=/path/to/libMyPass.so
python your_script.py
```

For more information on integrating plugins into the Triton compilation pipeline, see the Triton documentation on plugin hooks and runtime integration.

## Dependencies

- **MLIR/LLVM**: Required for pass infrastructure and IR manipulation
- **Triton**: Core Triton compiler libraries and headers. Use commit hash in `triton.txt` to guarantee compatibility.
- **CMake**: Build system
- **TableGen**: For pass definition metadata generation

## Notes

- Plugins are built as shared libraries (`SHARED`) to enable dynamic loading.
- The build system automatically handles dependencies and linking requirements.
- Plugin names are collected and propagated to the parent CMake scope via `TRITON_PLUGIN_NAMES`.
- The plugin name is read from `backend/name.conf` when the plugin is included in Triton's build.
