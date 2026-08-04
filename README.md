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

This repository provides a framework for developing Triton compiler extensions
that extend Triton without modifying the upstream codebase. Extensions are built
as shared libraries that are dynamically loaded by Triton at runtime. Extensions
are distributed as Python wheels.

The upstream infrastructure is documented:

- [upstream][triton-plugins].
- in [slides][slides-jan-2026] from the January 2026 Triton Community Meetup
- in more [slides][slides-jul-2026] from the July 2026 Triton Community Meetup.

### Structure

Extensions live in subdirectories, each built as a separate Triton wheel:

- **[`backend/`](./backend/)**: For new Triton backends (e.g., a new device
  target).

- **[`dialect/`](./dialect/)**: For adding MLIR dialects to Triton.

- **[`pass/`](./pass/)**: For adding MLIR passes to Triton (e.g.,
  [`arithmetic-intensity`][arithmetic-intensity]).

- **[`extensions/`](./extensions/)**: For extensions that bundle dialects,
  passes, and language bindings together (e.g., [`utlx`][utlx]).

- **[`support/`](./support/)**: Contains infrastructure code to automatically
  register extensions with Triton.

## Prerequisites

- C++ compiler with C++17 support
- CMake
- GitHub CLI ([`gh`]), for downloading pre-built dependencies (optional)
- Ninja
- Python 3, for tests and build scripts; install dependencies with
  `pip install -r requirements.txt`
- Triton, built with `TRITON_EXT_ENABLED=ON`, see
  [`download_triton_wheel.py`][download_triton]. Note: Extensions are enabled by
  default in Triton releases 3.7 and beyond.
- LLVM compilation artifacts, see [`download_llvm.py`][download_llvm]

## Build

This extension repository is designed to be built out-of-tree. It expects to be
pointed to both LLVM compilation artifacts and an installed Triton wheel. List
available extensions with [`list_extensions.py`][list_extensions].

To build the extensions:

1. **Build LLVM**, one of the following ways:

   - *download pre-built*: run [`download_llvm.py`][download_llvm], optionally
     specifying a specific upstream build.
   - *build locally*: run something like the upstream
     [`build-llvm-project.sh`][build_llvm] and set `LLVM_INSTALL_DIR` in the
     environment.

1. **Build Triton**, one of the following ways:

   - *download pre-built*: run [`download_triton_wheel.py`][download_triton],
     optionally with a specific `<version>+git<commit>` pattern, and
     `pip install triton-*.whl`. To list available wheels, run
     [`ci/list_triton_wheels.py`][list_triton].
   - *build locally*: run `TRITON_EXT_ENABLED=1 python setup.py bdist_wheel` in
     the Triton source tree, then `pip install triton-*.whl` here. To use a
     non-installed version, set `TRITON_WHEEL_DIR` in the environment.

1. **Build extensions**:

   ```bash
   make build
   ```

   Optionally set `LLVM_INSTALL_DIR` and `TRITON_WHEEL_DIR` to custom locations;
   by default triton-ext will helpfully [search] for them in the project
   directory.

A sample build might look like:

```bash
python -m venv --prompt triton-ext .venv
source .venv/bin/activate
pip install -r requirements.txt

ci/download_triton_wheel.py
pip install triton-*.whl
ci/download_llvm.py

make build
```

## Test

Run the test suite to verify the extensions are working correctly:

```bash
make test
```

## Use

Extensions are loaded by Triton by their `__init__.py` file (see
`libtriton.extend_with(...))`:

```bash
pip install <extension>.whl
python your_script.py
```

And in your script:

```python
import triton
import triton-<extension>
...
```

[arithmetic-intensity]: ./pass/ArithmeticIntensity/
[build_llvm]: https://github.com/triton-lang/triton/blob/main/scripts/build-llvm-project.sh
[download_llvm]: ./ci/download_llvm.py
[download_triton]: ./ci/download_triton_wheel.py
[list_extensions]: ./ci/list_extensions.py
[list_triton]: ./ci/list_triton_wheels.py
[search]: ./ci/pick_local_artifact.py
[slides-jan-2026]: https://docs.google.com/presentation/d/1dnm8uhvabdwqsQAsaPM7IRpEh2tktQ91E1d40r91n1M
[slides-jul-2026]: https://docs.google.com/presentation/d/1QwuwCZbhwUnFKA9VSxR0Lww0guwnmJtt6TvdUkMQooE
[triton-plugins]: https://github.com/triton-lang/triton/tree/main/examples/plugins
[utlx]: ./extensions/utlx/
[`gh`]: https://cli.github.com/
