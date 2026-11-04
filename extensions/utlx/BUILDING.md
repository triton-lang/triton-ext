# Building `triton-utlx` locally

[`scripts/build-wheel-local.sh`](scripts/build-wheel-local.sh) builds the
`triton-utlx` wheel from this checkout and, optionally, installs it into a
virtual environment. It is for local development and testing; released wheels
come from the [`publish-utlx.yml`](../../.github/workflows/publish-utlx.yml)
workflow (see [PUBLISHING.md](PUBLISHING.md)).

## TL;DR

```bash
# From the triton-ext checkout, with a venv that has torch and triton installed:
extensions/utlx/scripts/build-wheel-local.sh --install /path/to/venv
```

The script does not pick up the active venv by itself. If the venv is activated,
pass its path from `$VIRTUAL_ENV`:

```bash
source /path/to/venv/bin/activate
extensions/utlx/scripts/build-wheel-local.sh --install "$VIRTUAL_ENV"

# Or pick explicitly which libtriton to build against (see below):
LIBTRITON=pip   extensions/utlx/scripts/build-wheel-local.sh --install "$VIRTUAL_ENV"   # the venv's pip libtriton
LIBTRITON=local extensions/utlx/scripts/build-wheel-local.sh --install "$VIRTUAL_ENV"   # Triton built locally from source
```

Leaving `LIBTRITON` unset is the same as `LIBTRITON=auto`, which picks `pip` for
a venv with Triton 3.9 or newer (its wheel ships the C++ headers the build
needs) and `local` otherwise.

If only the venv's `bin/` is on `PATH`, without `activate`, `$VIRTUAL_ENV` is
not set; derive the path from `python` instead:

```bash
extensions/utlx/scripts/build-wheel-local.sh --install "$(dirname "$(dirname "$(command -v python)")")"
```

The last two lines of output confirm what you got:

```text
triton-utlx 3.9.0 | triton 3.9.0
libtriton: /path/to/venv/lib/python3.11/site-packages/triton/_C/libtriton.so
```

## Prerequisites

- `git`, `clang`, `clang++`, `curl` and `sha256sum` on `PATH`. The plugin is
  always compiled with clang (see the workflow for why it matters).

- `python3.11` (or set `PYTHON`) for the script's private build venv.

- A target venv with `torch` and the `triton` release that
  [`pyproject.toml`](pyproject.toml) pins (`dependencies = ["triton~=X.Y.Z"]`).
  For example:

  ```bash
  python3.11 -m venv /path/to/venv
  /path/to/venv/bin/pip install torch "triton~=3.9.0"
  ```

The target venv's Triton must be from the pinned release line. A plugin built
for one Triton release line corrupts the heap of another's `libtriton.so`
instead of failing cleanly, so the script refuses to build for a mismatched
venv.

## Choosing which `libtriton` to build against

`libutlx.so` compiles against Triton's C++ headers and links against
`libtriton.so`. The `LIBTRITON` environment variable picks where both come from:

| `LIBTRITON`      | Builds against                                                               | Notes                                                                                |
| ---------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `auto` (default) | `pip` if the target venv's Triton wheel ships headers, else `local`          | PyPI Triton wheels ship headers from 3.9 on.                                         |
| `pip`            | The target venv's pip-installed Triton (`<venv>/.../triton/_C/libtriton.so`) | Needs `--install VENV` and a wheel with headers (Triton 3.9+); fails otherwise.      |
| `local`          | Triton built from source in `~/.cache/utlx-build/triton-X.Y.Z`               | The first run builds Triton with `TRITON_EXT_ENABLED=1` (slow); later runs reuse it. |

```bash
# Build against the venv's pip-installed Triton
LIBTRITON=pip extensions/utlx/scripts/build-wheel-local.sh --install /path/to/venv

# Build against a Triton built from source
LIBTRITON=local extensions/utlx/scripts/build-wheel-local.sh --install /path/to/venv

# Same, but rebuild that Triton from scratch first
LIBTRITON=local extensions/utlx/scripts/build-wheel-local.sh --rebuild-triton --install /path/to/venv
```

`pip` builds the plugin against the exact `libtriton.so` it loads at runtime.
`local` builds it against a separate binary of the same Triton release, as the
publish workflow does.

At runtime the plugin always uses the target venv's own `libtriton.so`,
whichever mode built it: `libutlx.so` needs `libtriton.so` by soname, with no
RPATH, so it binds to the copy Triton has already loaded. `LIBTRITON` only
changes what the plugin is compiled and linked against.

## What the script does

1. Reads the Triton version from `pyproject.toml`'s `triton~=X.Y.Z` pin and
   checks that the wheel version is `X.Y.Z` or `X.Y.Z.postN`.
1. With `--install VENV`, checks that `VENV`'s Triton is from that release line
   and decides between `pip` and `local` (see above).
1. Shallow-clones Triton `vX.Y.Z` into `~/.cache/utlx-build/triton-X.Y.Z`. In
   `pip` mode this is only used to read the LLVM version the release pins.
1. Gets the matching prebuilt LLVM/MLIR (headers, `mlir-tblgen`, CMake modules)
   into `~/.triton/llvm`:
   - `local`: Triton's own setup downloads it while building Triton.
   - `pip`: the script downloads the almalinux build, which PyPI's Triton wheels
     are built with, and verifies its sha256 against Triton's
     `cmake/llvm-info.json`.
1. `local` only: builds Triton from source, unless a previous build is reused.
1. Builds the plugin wheel into `~/.cache/utlx-build/dist/`.
1. With `--install VENV`: runs `pip install --force-reinstall --no-deps` into
   `VENV`, so the venv keeps its own Triton and torch, then imports
   `utlx_plugin` and prints the versions and the loaded `libtriton.so`.

## Checking which `libtriton` was used

- **Build time:** look for this CMake line in the output:

  ```text
  -- Found Triton library at /path/to/venv/lib64/python3.11/site-packages/triton/_C/libtriton.so
  ```

  In `local` mode it points into `~/.cache/utlx-build/triton-X.Y.Z` instead.

- **Runtime:** the final `libtriton:` line shows the `libtriton.so` the process
  actually loaded. It should always be the target venv's.

## Building without installing

```bash
extensions/utlx/scripts/build-wheel-local.sh
```

Without `--install` there is no target venv, so the script always uses the
`local` source build, and leaves the wheel in `~/.cache/utlx-build/dist/`. The
wheel keeps the host's native linux tag (it is not built in the manylinux image
the workflow uses), so it is for local use only.

## Environment variables

| Variable                    | Default                     | Meaning                                                                             |
| --------------------------- | --------------------------- | ----------------------------------------------------------------------------------- |
| `LIBTRITON`                 | `auto`                      | Which Triton to build against: `auto`, `pip` or `local` (see above).                |
| `WORK`                      | `~/.cache/utlx-build`       | Build dir: Triton checkout and build venv.                                          |
| `DIST`                      | `$WORK/dist`                | Wheel output dir.                                                                   |
| `VERSION`                   | `pyproject.toml`'s version  | Wheel version; must be the pinned Triton version or `.postN` of it.                 |
| `PYTHON`                    | `python3.11`                | Interpreter for the build venv.                                                     |
| `TRITON_REPO`               | GitHub `triton-lang/triton` | Where to clone Triton from, e.g. a local checkout that has the release tag.         |
| `TRITON_HOME`               | `$HOME`                     | Triton's download cache, LLVM included, lives in `$TRITON_HOME/.triton`.            |
| `LIBSTDCXX_DIR`             | (searched)                  | A `libstdc++.so.6` new enough for the ubuntu-x64 LLVM tools (`GLIBCXX_3.4.30`).     |
| `TRITON_LLVM_SYSTEM_SUFFIX` | host's / `almalinux-<arch>` | Which prebuilt LLVM to use, e.g. `almalinux-x64`. `pip` mode defaults to almalinux. |

`extensions/utlx/scripts/build-wheel-local.sh --help` prints the full reference.

## Running the tests

utlx is not enabled in the repo-level test configuration, so the top-level
`conftest.py` skips `extensions/utlx/test` and `make test` collects no utlx
tests. Run them directly, bypassing that conftest:

```bash
source /path/to/venv/bin/activate
cd extensions/utlx/test
TRITON_ALWAYS_COMPILE=1 python -m pytest -c /dev/null --rootdir=. --confcutdir=. -q .
```

`TRITON_ALWAYS_COMPILE=1` matters: Triton's cache key ignores the plugin, so
without it a rebuilt plugin can silently reuse kernels the previous one
compiled.

## Troubleshooting

| Symptom                                                                   | Cause and fix                                                                                                                                                                                           |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Segmentation fault` in `register_plugin` on `import utlx_plugin`         | The installed plugin was built for a different Triton release line than the venv's. Rebuild with this script; it now refuses to build for a mismatched venv.                                            |
| `error: <venv> has triton A.B.C, not the Triton X.Y line this builds for` | The venv's Triton doesn't match `pyproject.toml`'s pin. Install the pinned release in the venv, or change the pin.                                                                                      |
| `error: ... ships no C++ headers, which LIBTRITON=pip needs`              | The venv's Triton wheel predates 3.9. Use `LIBTRITON=local` (or the default `auto`).                                                                                                                    |
| `mlir-tblgen does not run and no newer libstdc++ was found`               | The ubuntu-x64 LLVM needs `GLIBCXX_3.4.30`. Set `LIBSTDCXX_DIR` to a newer `libstdc++.so.6` (a conda `lib/` works), or use `TRITON_LLVM_SYSTEM_SUFFIX=almalinux-x64`.                                   |
| `uTLX: AMDGPU dialect headers not found ... AMD support disabled`         | Expected in `pip` mode: PyPI wheels don't ship the AMDGPU dialect headers. Use `LIBTRITON=local`, or set `TRITON_SOURCE_DIR`/`TRITON_BUILD_DIR` to a Triton source build, for a wheel with AMD support. |
