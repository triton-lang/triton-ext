#!/usr/bin/env bash
# build_wheel.sh — Build a pip wheel for the utlx package.
#
# Bundles:
#   - utlx_plugin (Python DSL, registers as triton.language.extra.tlx)
#   - utlx        (legacy re-export shim)
#   - tlx         (tlx/language/tlx operators)
#   - libutlx.so  (native plugin, loaded via TRITON_PLUGIN_PATHS)
#
# Usage:
#   ./build_wheel.sh                      # uses ./build as the CMake build dir
#   ./build_wheel.sh /path/to/build_dir   # custom build dir
#
# Output: dist/utlx-*.whl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/build}"
STAGE_DIR="${SCRIPT_DIR}/_wheel_stage"
DIST_DIR="${SCRIPT_DIR}/dist"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -f "${BUILD_DIR}/lib/libutlx.so" ]; then
    echo "ERROR: ${BUILD_DIR}/lib/libutlx.so not found."
    echo "Build utlx first (cmake + ninja), then re-run this script."
    exit 1
fi

# ---------------------------------------------------------------------------
# Clean previous staging area
# ---------------------------------------------------------------------------
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

# ---------------------------------------------------------------------------
# Stage Python packages
# ---------------------------------------------------------------------------

# utlx_plugin  (authoritative DSL — from python/utlx_plugin/)
cp -r "${SCRIPT_DIR}/python/utlx_plugin" "${STAGE_DIR}/utlx_plugin"

# utlx shim    (legacy re-export — from python/utlx/)
cp -r "${SCRIPT_DIR}/python/utlx" "${STAGE_DIR}/utlx"

# tlx          (tlx/language/tlx/ operators)
cp -r "${SCRIPT_DIR}/tlx/language/tlx" "${STAGE_DIR}/tlx"
# Remove tutorials from the wheel — they are not library code
rm -rf "${STAGE_DIR}/tlx/tutorials"

# ---------------------------------------------------------------------------
# Stage native shared library into utlx_plugin/
# ---------------------------------------------------------------------------
cp "${BUILD_DIR}/lib/libutlx.so" "${STAGE_DIR}/utlx_plugin/libutlx.so"

# ---------------------------------------------------------------------------
# Generate a minimal pyproject.toml for the staging area
# ---------------------------------------------------------------------------
PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

cat > "${STAGE_DIR}/pyproject.toml" << 'TOML'
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "utlx"
version = "0.1.0"
description = "uTLX: Triton Language Extensions distributed as a Plugin"
requires-python = ">=3.9"

[tool.setuptools]
packages = [
    "utlx",
    "utlx_plugin",
    "utlx_plugin.compiler",
    "tlx",
    "tlx.compiler",
]

[tool.setuptools.package-data]
utlx_plugin = ["libutlx.so"]
TOML

# ---------------------------------------------------------------------------
# Build the wheel
# ---------------------------------------------------------------------------
echo "==> Building wheel in ${STAGE_DIR} ..."
python3 -m pip install --quiet --upgrade pip setuptools wheel build 2>/dev/null \
    || python3 -m pip install --quiet --upgrade pip setuptools wheel build

python3 -m build --wheel --outdir "${DIST_DIR}" "${STAGE_DIR}"

# ---------------------------------------------------------------------------
# Clean up staging area
# ---------------------------------------------------------------------------
rm -rf "${STAGE_DIR}"

echo ""
echo "==> Wheel built successfully:"
ls -lh "${DIST_DIR}"/utlx-*.whl
echo ""
echo "Install with:"
echo "  pip install ${DIST_DIR}/utlx-*.whl"
echo ""
echo "Then set the plugin path at runtime:"
echo "  export TRITON_PLUGIN_PATHS=\$(python3 -c 'import utlx_plugin, os; print(os.path.join(os.path.dirname(utlx_plugin.__file__), \"libutlx.so\"))')"
