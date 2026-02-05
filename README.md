# Triton Extensions

A collection of out-of-tree extensions for the Triton compiler, including passes, dialects, backends, and language extensions. 

## Overview

This repository provides a framework for developing and building Triton compiler extensions that can extend functionality without modifying the core Triton codebase. Extensions are built as shared libraries that can be dynamically loaded by Triton at runtime. 

Extensions are built on top of the upstream Triton infrastructure documented here: https://github.com/triton-lang/triton/tree/main/examples/plugins

## Project Structure

```
triton-ext/
├── backend/              # Backend extensions directory
│   └── CMakeLists.txt
├── dialect/              # Dialect extensions directory
│   └── CMakeLists.txt
├── language/             # Language extensions directory
│   └── CMakeLists.txt
├── pass/                 # Pass extensions directory
│   ├── LoopSplit/        # Example: Loop splitting pass extension
│   │   ├── CMakeLists.txt
│   │   ├── LoopSplit.cpp
│   │   ├── Passes.td
│   │   └── test          # Lit tests 
│   │       └── loop-split.mlir
│   └── CMakeLists.txt
│
├── infra/                # Extension infrastructure
│   └── pass/             # Pass extension API infrastructure
│       ├── CMakeLists.txt
│       ├── TritonExtPassInfra.h
│       ├── TritonExtPassInfra.cpp
│       └── TritonExtPass.cpp
├── CMakeLists.txt        # Root CMake configuration
└── README.md
```

### Directory Descriptions

- **`backend/`**: Contains backend extension implementations.

- **`dialect/`**: Intended for custom MLIR dialect extensions (currently scaffolding only).

- **`language/`**: Intended for language extension implementations (currently scaffolding only).

- **`pass/`**: Contains MLIR pass extensions. Each pass extension is implemented as a shared library that can be loaded dynamically. Pass extensions include a `triton-ext.conf` file that specifies the extension name and status.

- **`infra/`**: Contains extension infrastructure code. The `infra/pass/` subdirectory provides the pass extension API infrastructure (`TritonExtPassInfra.h`, `TritonExtPassInfra.cpp`, `TritonExtPass.cpp`) that pass extensions use.

## Build Process

### Prerequisites

- Triton compiler source code
- CMake (3.15 or later)
- LLVM/MLIR development shared libraries
- C++ compiler with C++17 support

### Building as Part of Triton

This extension repository is designed to be built as part of the Triton build system. To build:

1. **Set up Triton build environment**: Ensure you have Triton's dependencies installed, including LLVM shared libraries (required for building extensions).

2. **Configure Triton build with extension directory**:
   ```bash
   cmake -DTRITON_EXT_DIRS=/path/to/triton-ext ...
   ```

   Or when building via Python setup:
   ```bash
   TRITON_EXT_DIRS=/path/to/triton-ext python setup.py develop
   ```

3. **Build**: The extensions will be built automatically as part of the Triton build process.

4. **Output location**: Built extensions (shared libraries) are placed in:
   ```
   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../extensions/
   # for editable build this is in: <triton>/python/triton/extensions
   ```

### Building Individual Extension Components

The build system processes extensions in the following order:
1. `dialect/`
2. `pass/`
3. `backend/`
4. `language/`

Each subdirectory's `CMakeLists.txt` is responsible for building its respective extensions.

## Extension API

The pass extension API (`infra/pass/TritonExtPassInfra.h`) provides:

Triton Extension API (provided by Triton):

- `tritonAddPluginPass()`: Add an extension pass to a pass manager
- `tritonRegisterPluginPass()`: Register an extension pass
- `tritonEnumeratePluginPasses()`: Enumerate available extension passes

Extension-side registration API:

- `TritonExtLoadPass()`: Load and register a pass extension (Extension side)

Pass extensions should define `TRITON_EXT_NAME` and `TRITON_EXT_CLASS` compile definitions in their CMakeLists.txt, and include `TritonExtPass.cpp` at the end of their implementation file to enable automatic registration.

## Using Extensions at Runtime

Extensions can be loaded by Triton at runtime using the `TRITON_PASS_PLUGIN_PATH` environment variable:

```bash
export TRITON_PASS_PLUGIN_PATH=/path/to/libmy_pass.so
python your_script.py
```

For more information on integrating extensions into the Triton compilation pipeline, see the Triton documentation on extension hooks and runtime integration.

## Dependencies

- **MLIR/LLVM**: Required for pass infrastructure and IR manipulation
- **Triton**: Core Triton compiler libraries and headers. Use commit hash in `triton.txt` to guarantee compatibility.
- **CMake**: Build system
- **TableGen**: For pass definition metadata generation

## Notes

- Extensions are built as shared libraries (`SHARED`) to enable dynamic loading.
- The build system automatically handles dependencies and linking requirements.
- Extension names are collected and propagated to the parent CMake scope via `TRITON_EXT_NAMES`.
- The extension name and status are read from `triton_ext.conf` in each extension directory (format: `extension_name;status`).
- The naming conventions need to be aligned. The `triton-ext` project uses EXT for extension but triton uses PLUGIN naming convention.
