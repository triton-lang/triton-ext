#!/usr/bin/env bash
# Regenerate PythonStubs.c from the CPython symbols libtriton.so imports.
# Run when the Triton pin changes if the imported symbol set drifts.
#
# Usage: gen-python-stubs.sh <path/to/libtriton.so> > PythonStubs.c
set -euo pipefail

LIBTRITON="${1:?usage: gen-python-stubs.sh <libtriton.so>}"

cat <<'HEADER'
// Empty stubs for the CPython C-API symbols libtriton.so imports (it is also a
// Python extension module). triton-ext-opt only runs MLIR passes, so these let
// libtriton load without a real libpython. Regenerate with gen-python-stubs.sh.

HEADER

nm -D --undefined-only "$LIBTRITON" | grep -E ' _?Py' | awk '{print $2}' | sort -u |
  while read -r sym; do
    echo "void ${sym}(void) {}"
  done
