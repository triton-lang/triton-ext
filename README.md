# Triton Extensions

<a href="https://github.com/triton-lang/triton-ext/actions?query=workflow%3ACI">
   <!-- markdownlint-disable MD013 -->
   <img src="https://github.com/triton-lang/triton-ext/workflows/CI/badge.svg" alt="build status" />
</a>
<a href="https://discord.com/channels/1189498204333543425/1483539998702567587">
   <img src="https://img.shields.io/badge/discord-join_chat-blue.svg" alt="discord chat" />
    <!-- markdownlint-enable MD013 -->
</a>

A collection of out-of-tree extensions for the Triton compiler, including
passes, dialects, backends, and language extensions.

> NOTE: this project is *under construction*. It is currently in the early
> stages of development and parts of it will likely change. Contributions are
> welcome but be aware that the foundations may change rapidly.

## Overview

This repository provides a framework for developing and building Triton compiler
extensions that can extend functionality without modifying the core Triton
codebase. Extensions are built as shared libraries that can be dynamically
loaded by Triton at runtime.

Extensions are built on top of the Triton plugin infrastructure documented
[upstream][triton-plugins].

Slides from the January 2026 Triton Community Meetup:
[Triton Community Meetup: Triton Extensions Jan 2026](https://docs.google.com/presentation/d/1dnm8uhvabdwqsQAsaPM7IRpEh2tktQ91E1d40r91n1M/edit?usp=sharing)

### Structure

Each subdirectory's `CMakeLists.txt` is responsible for building its respective
extensions.

- **[`backend/`](./backend/)**: Intended for backend extension implementations
  (currently scaffolding only).

- **[`dialect/`](./dialect/)**: Intended for custom MLIR dialect extensions
  (currently scaffolding only).

- **[`language/`](./language/)**: Intended for language extension
  implementations (currently scaffolding only).

- **[`pass/`](./pass/)**: Contains MLIR pass extensions. Each pass extension is
  implemented as a shared library that can be loaded dynamically. Pass
  extensions include a `triton-ext.conf` file that specifies the extension name
  and status.

- **[`extensions/`](./extensions/)**: Contains standalone plugin extensions that
  bundle dialects, passes, and language bindings into self-contained shared
  libraries loadable by Triton at runtime.

  - **[`utlx` (µTLX)](./extensions/utlx/)**: Triton Language Extensions plugin
    that provides most of Meta's TLX functionality without modifying the Triton
    fork. Includes local memory operations (`local_alloc`, `local_view`,
    `local_store`, `local_load`, `alloc_barriers`), custom passes (e.g.,
    PingPong, PruneUnusedBarriers), the TLX dialect, conversion patterns, and a
    Python DSL. Builds `libutlx.so`.

- **[`support/`](./support/)**: Contains extension infrastructure code to
  automatically register extensions with Triton.

## Prerequisites

- C++ compiler with C++17 support
- CMake
- GitHub CLI ([`gh`]), for downloading pre-built dependencies (optional)
- Ninja
- Python 3, for tests and build scripts; install dependencies with
  `pip install -r requirements.txt`
- Triton, built with `TRITON_EXT_ENABLED=ON`

Note: Extensions are enabled by default in Triton releases 3.7 and beyond.

## Build

This extension repository is designed to be built out-of-tree. It expects to be
pointed to both LLVM (`LLVM_INSTALL_DIR`) and Triton (`TRITON_INSTALL_DIR`).
Both directories should be installation directories, i.e., built and then
packaged with `make install`.

To build the extensions:

1. **Build LLVM**: Build LLVM as shared libraries and install it to a known
   location; see the CI [action][build-llvm] for reference. Alternately,
   download pre-built LLVM binaries from GitHub: run
   `ci/download-artifact.py llvm` [^list-artifacts].

1. **Build Triton**: Build Triton and install it to a known location; see the CI
   [action][build-triton] for reference. Alternately, download pre-built Triton
   binaries from GitHub: run `ci/download-artifact.py triton` [^list-artifacts].

\[^list-artifacts\]: GitHub artifacts are only available for a limited set of
commits, OSes, and HW architectures. To list available artifacts, run
`ci/fetch-artifacts.py`.

1. **Build extensions**:

   ```bash
   export LLVM_INSTALL_DIR=/path/to/llvm/install
   export TRITON_INSTALL_DIR=/path/to/triton/install
   make build
   ```

   Note that if `LLVM_INSTALL_DIR` and `TRITON_INSTALL_DIR` are not set, the
   `Makefile` will helpfully [search] for them in the project directory. See the
   CI [workflow](./.github/workflows/ci.yml) for reference. Extensions are built
   as shared libraries under `build/lib`.

## Test

Run the test suite to verify the extensions are working correctly:

```bash
make test
```

## Use

Extensions are loaded by Triton at runtime using the `TRITON_PLUGIN_PATHS`
environment variable (see [Triton plugins][triton-plugins]):

```bash
export TRITON_PLUGIN_PATHS=/path/to/libmy_pass.so
python your_script.py
```

Some extensions are accessible from Python: e.g., the µTLX plugin automatically
registers itself as `triton.language.extra.tlx` when imported, so no filesystem
symlinks are needed:

```bash
export TRITON_PLUGIN_PATHS=/path/to/triton-ext/build/lib/libutlx.so
python your_script.py
```

To load multiple plugins, separate paths with `:`:

```bash
export TRITON_PLUGIN_PATHS=build/lib/libutlx.so:build/lib/libother_plugin.so
```

[build-llvm]: ./.github/actions/build-llvm/action.yml
[build-triton]: ./.github/actions/build-triton/action.yml
[search]: ./ci/pick-local-artifact.py
[triton-plugins]: https://github.com/triton-lang/triton/tree/main/examples/plugins
[`gh`]: https://cli.github.com/
