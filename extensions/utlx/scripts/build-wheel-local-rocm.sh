#!/bin/bash
# Build the triton-utlx-rocm wheel from this checkout locally, the way
# .github/workflows/publish-utlx-rocm.yml builds it in CI. build-wheel-local.sh
# is the same for triton-utlx.
#
# As in the workflow, the package metadata comes from pyproject-rocm.toml,
# copied over pyproject.toml for the build; pyproject.toml is put back when the
# script exits.
#
# libutlx.so compiles against Triton's C++ headers, which the triton-rocm
# wheels do not ship. So, like the workflow, this builds the ROCm/triton
# release that pyproject-rocm.toml pins (`triton-rocm~=X.Y.Z`) from source with
# TRITON_EXT_ENABLED=1, installs its headers into that source tree and builds
# the plugin against it, with clang (see the workflow for why clang matters).
# The package name makes uTLX's CMake define UTLX_TRITON_ROCM. The resulting
# wheel runs against the triton-rocm of that release on PyTorch's ROCm index.
#
# The Triton build (the slow part) is done once and reused: later runs only
# rebuild the plugin. Pass --rebuild-triton to force it.
#
# Unlike CI this runs on the host, not in the manylinux_2_28 image, so the
# wheel keeps its native linux tag and is meant for local use only.
#
# Usage: ./build-wheel-local-rocm.sh [--rebuild-triton] [--install VENV]
#   --install VENV   pip install the wheel into VENV (--force-reinstall
#                    --no-deps, so VENV keeps its own Triton and torch),
#                    replacing triton-utlx if VENV has it
#
# Env overrides:
#   WORK         build dir: Triton checkout, build venv
#                (default: ~/.cache/utlx-build)
#   DIST         wheel output dir         (default: $WORK/dist)
#   VERSION      wheel version            (default: pyproject-rocm.toml's)
#   PYTHON       interpreter for the build venv (default: python3.11)
#   TRITON_REPO  where to clone ROCm/triton from, e.g. a local checkout that
#                has the release ref (default: https://github.com/ROCm/triton)
#   TRITON_REF   branch or tag of TRITON_REPO to build
#                (default: release/internal/X.Y.x, as the workflow's default)
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
#                If none works, the script builds against the almalinux-x64
#                LLVM instead, which runs on an older libstdc++.
#   TRITON_LLVM_SYSTEM_SUFFIX  which prebuilt LLVM Triton's setup downloads,
#                e.g. almalinux-x64 (default: Triton's choice for the host)

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
PYPROJECT_ROCM=$UTLX/pyproject-rocm.toml

die() { echo "error: $*" >&2; exit 1; }
for tool in git clang clang++ "$PYTHON"; do
    command -v "$tool" > /dev/null || die "$tool not found"
done

mkdir -p "$WORK"
# Build from pyproject-rocm.toml, as the workflow does, and put pyproject.toml
# back on exit, however the script ends.
cp "$PYPROJECT" "$WORK/pyproject.toml.orig"
trap 'cp "$WORK/pyproject.toml.orig" "$PYPROJECT"' EXIT
cp "$PYPROJECT_ROCM" "$PYPROJECT"

NAME=$(sed -nE 's/^name = "(.*)"/\1/p' "$PYPROJECT")
TRITON_VERSION=$(sed -nE 's/.*"triton-rocm~=([0-9.]+)".*/\1/p' "$PYPROJECT")
[ -n "$TRITON_VERSION" ] ||
    die "no \"triton-rocm~=X.Y.Z\" dependency in $PYPROJECT_ROCM"
VERSION=${VERSION:-$(sed -nE 's/^version = "(.*)"/\1/p' "$PYPROJECT")}
case "$VERSION" in
    "$TRITON_VERSION" | "$TRITON_VERSION".post[0-9]*) ;;
    *) die "version $VERSION must be $TRITON_VERSION or $TRITON_VERSION.postN" ;;
esac
TRITON_REPO=${TRITON_REPO:-https://github.com/ROCm/triton}
TRITON_REF=${TRITON_REF:-release/internal/${TRITON_VERSION%.*}.x}
echo "Building $NAME $VERSION against triton-rocm $TRITON_VERSION" \
    "($TRITON_REPO $TRITON_REF) in $WORK"

TRITON=$WORK/triton-rocm-$TRITON_VERSION
VENV=$WORK/venv
export TRITON_HOME=${TRITON_HOME:-$HOME}

if [ ! -d "$TRITON/.git" ]; then
    git clone --depth 1 --branch "$TRITON_REF" "$TRITON_REPO" "$TRITON"
fi
# The ref must build the release the pin names.
BUILT=$(sed -nE 's/^TRITON_VERSION = "([0-9.]+)".*/\1/p' "$TRITON/setup.py")
[ "$BUILT" = "$TRITON_VERSION" ] ||
    die "$TRITON ($TRITON_REPO $TRITON_REF) builds Triton $BUILT, not $TRITON_VERSION"
if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
fi
PY=$VENV/bin/python

HEADERS=$TRITON/python/triton/include
# The LLVM build the Triton release pins, as Triton's setup names it.
LLVM_NAME=$(sed -nE 's/.*"llvm_hash": *"([0-9a-f]{8}).*/llvm-\1/p' "$TRITON/cmake/llvm-info.json")
# Prints nothing, and succeeds, before Triton's setup has downloaded it.
find_llvm() {
    ls -d "$TRITON_HOME/.triton/llvm/$LLVM_NAME-${TRITON_LLVM_SYSTEM_SUFFIX:-}"* \
        2>/dev/null | head -1 || true
}
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
# Make Triton's setup use the LLVM found to work, rather than download the one
# it would pick for this host. Its directory is <LLVM_NAME>-<suffix>-<build>.
pin_llvm() {
    local suffix=${LLVM_INSTALL_DIR##*/}
    suffix=${suffix#"$LLVM_NAME"-}
    export TRITON_LLVM_SYSTEM_SUFFIX=${suffix%-*}
}
check_llvm_tools() {
    LLVM_INSTALL_DIR=$(find_llvm)
    # Not downloaded yet: Triton's setup fetches it, and we check afterwards.
    [ -n "$LLVM_INSTALL_DIR" ] || return 0
    if tblgen_runs "$LLVM_LD_PATH" || pick_libstdcxx; then
        pin_llvm
        return 0
    fi
    # The manylinux (almalinux) build of the same LLVM, which PyPI's Triton
    # wheels use too, needs no newer libstdc++. Triton's setup downloads it.
    if [ -z "${TRITON_LLVM_SYSTEM_SUFFIX:-}" ] && [ "$(uname -m)" = x86_64 ]; then
        echo "$LLVM_INSTALL_DIR/bin/mlir-tblgen does not run; using the almalinux-x64 LLVM"
        export TRITON_LLVM_SYSTEM_SUFFIX=almalinux-x64
        # A build configured against the unusable LLVM cannot have finished,
        # but its CMake cache would keep pointing at that LLVM's tools.
        rm -rf "$TRITON"/build
        check_llvm_tools
        return
    fi
    LD_LIBRARY_PATH=$LLVM_LD_PATH "$LLVM_INSTALL_DIR/bin/mlir-tblgen" --version || true
    die "$LLVM_INSTALL_DIR/bin/mlir-tblgen does not run and no newer libstdc++ was found; set LIBSTDCXX_DIR to a dir with a libstdc++.so.6 that has GLIBCXX_3.4.30"
}
check_llvm_tools

if [ "$REBUILD_TRITON" = 1 ] || [ ! -d "$HEADERS/triton" ]; then
    echo "=== Building triton-rocm $TRITON_VERSION with extension headers"
    rm -rf "$HEADERS"
    "$PY" -m pip install -r "$TRITON/python/requirements.txt"
    # Only the headers and libtriton matter here; skipping Triton's C++ unit
    # tests also skips fetching googletest from GitHub.
    (cd "$TRITON" && TRITON_WHEEL_NAME=triton-rocm TRITON_EXT_ENABLED=1 \
        MAX_JOBS=${MAX_JOBS:-$(nproc)} \
        CC=clang CXX=clang++ LD_LIBRARY_PATH=$LLVM_LD_PATH \
        TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF ${TRITON_APPEND_CMAKE_ARGS:-}" \
        "$PY" -m pip install -e . --no-build-isolation -v)
    # Triton 3.9+ installs its headers there itself (the wheel_headers
    # component, run by setup.py); earlier releases need the headers component.
    if [ ! -d "$HEADERS/triton" ]; then
        # The headers component copies python/ to include/python, so installing
        # straight into python/triton would recurse into its own output.
        STAGE=$(mktemp -d)
        cmake --install "$(ls -d "$TRITON"/build/cmake.* | head -1)" \
            --component headers --prefix "$STAGE" > /dev/null
        mv "$STAGE/include" "$HEADERS"
        rm -rf "$STAGE"
    fi
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
# into the build's pyproject.toml (put back on exit, see above).
sed -i -E "s/^version = \".*\"/version = \"$VERSION\"/" "$PYPROJECT"
mkdir -p "$DIST"
WHEEL_PREFIX=${NAME//-/_}-$VERSION
rm -f "$DIST"/"$WHEEL_PREFIX"-*.whl
CC=clang CXX=clang++ LD_LIBRARY_PATH=$LLVM_LD_PATH \
LLVM_INSTALL_DIR=$LLVM_INSTALL_DIR \
TRITON_WHEEL_DIR=$TRITON/python/triton \
    "$PY" -m pip wheel "$UTLX" --no-build-isolation --no-deps -v -w "$DIST"
WHEEL=$(ls "$DIST"/"$WHEEL_PREFIX"-*.whl)
echo "built $WHEEL"

if [ -n "$INSTALL_VENV" ]; then
    echo "=== Installing into $INSTALL_VENV"
    # triton-utlx ships utlx_plugin too; uninstall it first so it cannot later
    # take this wheel's files with it.
    "$INSTALL_VENV/bin/python" -m pip uninstall -y -q triton-utlx 2> /dev/null || true
    "$INSTALL_VENV/bin/python" -m pip install --force-reinstall --no-deps "$WHEEL"
    # Ask the module, not a distribution: VENV's Triton may be triton-rocm.
    (cd /tmp && "$INSTALL_VENV/bin/python" -c \
        'import importlib.metadata as m, sys, triton, utlx_plugin; print(sys.argv[1], m.version(sys.argv[1]), "| triton", triton.__version__)' \
        "$NAME")
fi
