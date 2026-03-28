# Triton Extensions

<a href="https://github.com/triton-lang/triton-ext/actions?query=workflow%3ACI"><img src="https://github.com/triton-lang/triton-ext/workflows/CI/badge.svg" alt="build status" /></a>
<a href="https://discord.com/channels/1189498204333543425/1483539998702567587"><img src="https://img.shields.io/badge/discord-join_chat-blue.svg" alt="discord chat" /></a>



A collection of out-of-tree extensions for the Triton compiler, including passes, dialects, backends, and language extensions.

> NOTE: this project is *under construction*. It is currently in the early stages of development and parts of it will
> likely change. Contributions are welcome but be aware that the foundations may change rapidly.

## Overview

This repository provides a framework for developing and building Triton compiler extensions that can extend
functionality without modifying the core Triton codebase. Extensions are built as shared libraries that can be
dynamically loaded by Triton at runtime.

Extensions are built on top of the upstream Triton infrastructure documented here:
https://github.com/triton-lang/triton/tree/main/examples/plugins

### Structure

Each subdirectory's `CMakeLists.txt` is responsible for building its respective extensions.

- **[`backend/`](./backend/)**: Intended for backend extension implementations (currently scaffolding only).

- **[`dialect/`](./dialect/)**: Intended for custom MLIR dialect extensions (currently scaffolding only).

- **[`language/`](./language/)**: Intended for language extension implementations (currently scaffolding only).

- **[`pass/`](./pass/)**: Contains MLIR pass extensions. Each pass extension is implemented as a shared library that can
  be loaded dynamically. Pass extensions include a `triton-ext.conf` file that specifies the extension name and status.

- **[`extensions/`](./extensions/)**: Contains standalone plugin extensions that bundle dialects, passes, and language
  bindings into self-contained shared libraries loadable by Triton at runtime.

  - **[`utlx` (µTLX)](./extensions/utlx/)**: Triton Language Extensions plugin that provides most of Meta's TLX
    functionality without modifying the Triton fork. Includes local memory operations (`local_alloc`, `local_view`,
    `local_store`, `local_load`, `alloc_barriers`), custom passes (e.g., PingPong, PruneUnusedBarriers), the TLX
    dialect, conversion patterns, and a Python DSL. Builds `libutlx.so`.

- **[`support/`](./support/)**: Contains extension infrastructure code to automatically register extensions with Triton.

## Prerequisites

- C++ compiler with C++17 support
- CMake (3.15 or later)
- LLVM/MLIR: development shared libraries, headers, binaries (e.g., `mlir-tblgen`)
- Triton: shared libraries and headers (depending on the above LLVM/MLIR libraries)

## Build

This extension repository is designed to be built out-of-tree. It expects to be pointed to both LLVM
(`LLVM_INSTALL_DIR`) and Triton (`TRITON_INSTALL_DIR`). Both directories should be installation directories, i.e.,
built and then packaged with `make install`.

To build the extensions:

1. **Build LLVM**: Build LLVM as shared libraries and install it to a known location; see the CI
   [action](./.github/actions/build-llvm/action.yml) for reference.

2. **Build Triton**: Build Triton and install it to a known location; see the CI
   [action](./.github/actions/build-triton/action.yml) for reference.

3. **Configure and build extensions**:

   ```bash
   mkdir build
   export LLVM_INSTALL_DIR=/path/to/llvm/install
   export TRITON_INSTALL_DIR=/path/to/triton/install
   cmake -S . -B build -G Ninja
   cmake --build build
   ```

   See the CI [workflow](./.github/workflows/ci.yml) for reference. Extensions are built as shared libraries under
   `build/lib`.

## Use

Pass extensions can be loaded by Triton at runtime using the `TRITON_PASS_PLUGIN_PATH` environment variable (see
[Triton plugins](https://github.com/triton-lang/triton/tree/main/examples/plugins)):

```bash
export TRITON_PASS_PLUGIN_PATH=/path/to/libmy_pass.so
python your_script.py
```

### Extension Plugins (µTLX)

Extension plugins are loaded via `TRITON_PLUGIN_PATHS`. The µTLX plugin automatically registers itself as
`triton.language.extra.tlx` when imported, so no filesystem symlinks are needed:

```bash
export TRITON_PLUGIN_PATHS=/path/to/triton-ext/build/lib/libutlx.so
python your_script.py
```

To load multiple plugins, separate paths with `:`:

```bash
export TRITON_PLUGIN_PATHS=build/lib/libutlx.so:build/lib/libother_plugin.so
```
