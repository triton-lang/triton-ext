#!/bin/bash
# Build the triton-utlx wheel from this checkout locally, the way
# .github/workflows/publish-utlx.yml builds it in CI.
#
# libutlx.so compiles against Triton's C++ headers, which PyPI Triton wheels do
# not ship. So, like the workflow, this builds the Triton release that
# extensions/utlx/pyproject.toml pins (`triton~=X.Y.Z`) from source with
# TRITON_EXT_ENABLED=1, installs its headers into that source tree and builds
# the plugin against it, with clang (see the workflow for why clang matters).
# The resulting wheel runs against the stock PyPI Triton of that release.
#
# The Triton build (the slow part) is done once and reused: later runs only
# rebuild the plugin. Pass --rebuild-triton to force it.
#
# Unlike CI this runs on the host, not in the manylinux_2_28 image, so the
# wheel keeps its native linux tag and is meant for local use only.
#
# Usage: ./build-wheel-local.sh [--rebuild-triton] [--install VENV]
#   --install VENV   pip install the wheel into VENV (--force-reinstall
#                    --no-deps, so VENV keeps its own Triton and torch)
#
# Env overrides:
#   WORK         build dir: Triton checkout, build venv
#                (default: ~/.cache/utlx-build)
#   DIST         wheel output dir         (default: $WORK/dist)
#   VERSION      wheel version            (default: pyproject.toml's)
#   PYTHON       interpreter for the build venv (default: python3.11)
#   TRITON_REPO  where to clone Triton from, e.g. a local checkout that has
#                the release tag (default: https://github.com/triton-lang/triton)
#   TRITON_HOME  Triton's download cache, LLVM included, lives in
#                $TRITON_HOME/.triton (default: $HOME, shared with other builds)
#   LIBSTDCXX_DIR  dir (or file) with a libstdc++.so.6 new enough for
#                Triton's prebuilt LLVM tools. The ubuntu-x64 LLVM needs
#                GLIBCXX_3.4.30, which e.g. CentOS Stream 9's system libstdc++
#                lacks. If the tools do not run as is, the script tries this,
#                then $CONDA_PREFIX/lib, the conda next to the `python3` on
#                PATH, ~/miniconda3/lib, ~/miniconda/lib, ~/anaconda3/lib and
#                /packages/*/conda/lib, and uses the first that works. Only
#                libstdc++ itself goes on LD_LIBRARY_PATH, not its neighbours.

set -euo pipefail

REBUILD_TRITON=0
INSTALL_VENV=
while [ $# -gt 0 ]; do
    case $1 in
        --rebuild-triton) REBUILD_TRITON=1 ;;
        --install) INSTALL_VENV=$(realpath "$2"); shift ;;
        -h | --help) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;;
        *) echo "error: unknown argument $1" >&2; exit 1 ;;
    esac
    shift
done

UTLX=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${WORK:-$HOME/.cache/utlx-build}
DIST=${DIST:-$WORK/dist}
PYTHON=${PYTHON:-python3.11}
PYPROJECT=$UTLX/pyproject.toml

die() { echo "error: $*" >&2; exit 1; }
for tool in git clang clang++ "$PYTHON"; do
    command -v "$tool" > /dev/null || die "$tool not found"
done

TRITON_VERSION=$(sed -nE 's/.*"triton~=([0-9.]+)".*/\1/p' "$PYPROJECT")
[ -n "$TRITON_VERSION" ] || die "no \"triton~=X.Y.Z\" dependency in $PYPROJECT"
VERSION=${VERSION:-$(sed -nE 's/^version = "(.*)"/\1/p' "$PYPROJECT")}
case "$VERSION" in
    "$TRITON_VERSION" | "$TRITON_VERSION".post[0-9]*) ;;
    *) die "version $VERSION must be $TRITON_VERSION or $TRITON_VERSION.postN" ;;
esac
echo "Building triton-utlx $VERSION against Triton v$TRITON_VERSION in $WORK"

mkdir -p "$WORK"
TRITON=$WORK/triton-$TRITON_VERSION
VENV=$WORK/venv
TRITON_REPO=${TRITON_REPO:-https://github.com/triton-lang/triton}
export TRITON_HOME=${TRITON_HOME:-$HOME}

if [ ! -d "$TRITON/.git" ]; then
    git clone --depth 1 --branch "v$TRITON_VERSION" "$TRITON_REPO" "$TRITON"
fi
if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
fi
PY=$VENV/bin/python

HEADERS=$TRITON/python/triton/include
# The LLVM build the Triton release pins, as Triton's setup names it.
LLVM_NAME=$(sed -nE 's/.*"llvm_hash": *"([0-9a-f]{8}).*/llvm-\1/p' "$TRITON/cmake/llvm-info.json")
find_llvm() { ls -d "$TRITON_HOME/.triton/llvm/$LLVM_NAME"-* 2>/dev/null | head -1; }
# Both builds run LLVM's mlir-tblgen, which may need a newer libstdc++ than
# the system's. Find one that works and expose just that library: the dir it
# lives in (a conda lib/, say) holds plenty that must not shadow system libs.
LLVM_LD_PATH=${LD_LIBRARY_PATH:-}
tblgen_runs() {  # tblgen_runs <LD_LIBRARY_PATH>
    LD_LIBRARY_PATH=$1 "$LLVM_INSTALL_DIR/bin/mlir-tblgen" --version > /dev/null 2>&1
}
pick_libstdcxx() {
    local candidate lib shim=$WORK/libstdcxx
    local py_prefix
    py_prefix=$(dirname "$(dirname "$(command -v python3 || echo /x/x)")")
    for candidate in "${LIBSTDCXX_DIR:-}" "${CONDA_PREFIX:+$CONDA_PREFIX/lib}" \
        "$py_prefix/lib" "$HOME/miniconda3/lib" "$HOME/miniconda/lib" \
        "$HOME/anaconda3/lib" /packages/*/conda/lib; do
        [ -n "$candidate" ] || continue
        lib=$candidate
        [ -d "$lib" ] && lib=$lib/libstdc++.so.6
        [ -e "$lib" ] || continue
        rm -rf "$shim" && mkdir -p "$shim"
        ln -s "$(realpath "$lib")" "$shim/libstdc++.so.6"
        if tblgen_runs "$shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; then
            echo "Using $(realpath "$lib") for LLVM's tools"
            LLVM_LD_PATH=$shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            return 0
        fi
    done
    rm -rf "$shim"
    return 1
}
check_llvm_tools() {
    LLVM_INSTALL_DIR=$(find_llvm)
    # Not downloaded yet: Triton's setup fetches it, and we check afterwards.
    [ -n "$LLVM_INSTALL_DIR" ] || return 0
    tblgen_runs "$LLVM_LD_PATH" && return 0
    pick_libstdcxx && return 0
    LD_LIBRARY_PATH=$LLVM_LD_PATH "$LLVM_INSTALL_DIR/bin/mlir-tblgen" --version || true
    die "$LLVM_INSTALL_DIR/bin/mlir-tblgen does not run and no newer libstdc++ was found; set LIBSTDCXX_DIR to a dir with a libstdc++.so.6 that has GLIBCXX_3.4.30"
}
check_llvm_tools

if [ "$REBUILD_TRITON" = 1 ] || [ ! -d "$HEADERS/triton" ]; then
    echo "=== Building Triton v$TRITON_VERSION with extension headers"
    rm -rf "$HEADERS"
    "$PY" -m pip install -r "$TRITON/python/requirements.txt"
    # Only the headers and libtriton matter here; skipping Triton's C++ unit
    # tests also skips fetching googletest from GitHub.
    (cd "$TRITON" && TRITON_EXT_ENABLED=1 MAX_JOBS=${MAX_JOBS:-$(nproc)} \
        CC=clang CXX=clang++ LD_LIBRARY_PATH=$LLVM_LD_PATH \
        TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF ${TRITON_APPEND_CMAKE_ARGS:-}" \
        "$PY" -m pip install -e . --no-build-isolation -v)
    # The headers component copies python/ to include/python, so installing
    # straight into python/triton would recurse into its own output.
    STAGE=$(mktemp -d)
    cmake --install "$(ls -d "$TRITON"/build/cmake.* | head -1)" \
        --component headers --prefix "$STAGE" > /dev/null
    mv "$STAGE/include" "$HEADERS"
    rm -rf "$STAGE"
    [ -d "$HEADERS/triton" ] || die "Triton headers missing from $HEADERS"
else
    echo "=== Reusing Triton build in $TRITON (--rebuild-triton to redo)"
fi

LLVM_INSTALL_DIR=$(find_llvm)
[ -d "$LLVM_INSTALL_DIR" ] || die "no $LLVM_NAME build under $TRITON_HOME/.triton/llvm"
check_llvm_tools

echo "=== Building the wheel"
"$PY" -m pip install "scikit-build-core>=0.10" "cmake>=3.20" ninja wheel
# CMakeLists.txt includes ../../cmake, so build in place; stamp the version
# into pyproject.toml for the build only.
cp "$PYPROJECT" "$WORK/pyproject.toml.orig"
trap 'cp "$WORK/pyproject.toml.orig" "$PYPROJECT"' EXIT
sed -i -E "s/^version = \".*\"/version = \"$VERSION\"/" "$PYPROJECT"
mkdir -p "$DIST"
rm -f "$DIST"/triton_utlx-"$VERSION"-*.whl
CC=clang CXX=clang++ LD_LIBRARY_PATH=$LLVM_LD_PATH \
LLVM_INSTALL_DIR=$LLVM_INSTALL_DIR \
TRITON_WHEEL_DIR=$TRITON/python/triton \
    "$PY" -m pip wheel "$UTLX" --no-build-isolation --no-deps -v -w "$DIST"
WHEEL=$(ls "$DIST"/triton_utlx-"$VERSION"-*.whl)
echo "built $WHEEL"

if [ -n "$INSTALL_VENV" ]; then
    echo "=== Installing into $INSTALL_VENV"
    "$INSTALL_VENV/bin/python" -m pip install --force-reinstall --no-deps "$WHEEL"
    (cd /tmp && "$INSTALL_VENV/bin/python" -c \
        'import importlib.metadata as m, utlx_plugin; print("triton-utlx", m.version("triton-utlx"), "| triton", m.version("triton"))')
fi
