#!/bin/bash
# Run every fbtriton test that covers third_party/tlx/ops/kernels or
# third_party/tlx/language/tlx/inductor, against stock Triton + the installed
# triton-utlx wheel.
#
# Neither directory holds tests itself; these are the fbtriton tests that
# import from them. uTLX's pytest plugin (_utlx_pytest) sees that the tests
# live in an fbtriton checkout and serves `triton.tlx`, the fork-only
# `triton._internal_testing` helpers and (on branches that have it) the
# inductor `*_torch` callables from that checkout.
#
# Some tests also import names the plugin does not map yet. A small extra
# plugin, generated below, maps them to the checkout's own files as well:
#   triton.language.extra.tlx.tutorials -> third_party/tlx/tutorials
#   triton.language.extra.tlx.ops       -> third_party/tlx/language/tlx/ops
#   triton.tlx.pytorch                  -> python/triton/tlx/pytorch.py
#
# Each test file runs in its own pytest process, so a crash in one file does
# not hide the results of the others. Triton's cache key does not include the
# plugin, so every run uses fresh Triton and Inductor caches.
#
# Usage: ./run-fbtriton-tlx-tests.sh [--collect-only] [extra pytest args...]
#   e.g. ./run-fbtriton-tlx-tests.sh -x
#        ./run-fbtriton-tlx-tests.sh -k gfx950
#
# Env overrides:
#   FBTRITON   fbtriton checkout       (default: ~/tmp4/fbtriton2)
#   VENV       venv with triton-utlx   (default: ~/tmp4/triton-clean-venv)
#   TIMEOUT    seconds per test file   (default: 3600)
#   RESULTS    log/junit directory     (default: ~/tmp4/tlx-test-results/<time>)

set -u

FBTRITON=${FBTRITON:-`pwd`/fbtriton2}
VENV=${VENV:-`pwd`/triton-clean-venv}
TIMEOUT=${TIMEOUT:-3600}
RESULTS=${RESULTS:-`pwd`/tlx-test-results/$(date +%Y%m%d-%H%M%S)}
PY=$VENV/bin/python

UNIT=$FBTRITON/python/test/unit
TESTS=(
    # ops/kernels and inductor (the test_torchtlx_* files)
    "$UNIT"/tlx_ops/test_*.py
    # ops/kernels
    "$UNIT/language/test_tlx_layout_gfx950.py"
    "$UNIT/language/test_tlx_amd_gfx950.py"
    "$UNIT/language/test_tlx_amd_hip.py"
    "$UNIT/language/test_tlx_amd_fa_bwd_gfx950.py"
    "$FBTRITON/third_party/tlx/tutorials/testing/test_correctness.py"
    # inductor
    "$UNIT/language/test_torchtlx_templates.py"
    "$UNIT/language/test_torchtlx_fusions.py"
)
# Not run:
#   language/test_tlx_codegen.py  needs triton.backends.amd.amdgc_hazard_repair,
#                                 which only fbtriton's own Triton build has.
#   tutorials/testing/test_amd_gemm_perf.py
#                                 a benchmark; run it with python, not pytest.

die() { echo "error: $*" >&2; exit 1; }
[ -x "$PY" ] || die "no python at $PY (set VENV)"
[ -d "$UNIT/tlx_ops" ] || die "no fbtriton checkout at $FBTRITON (set FBTRITON)"
"$PY" -c "import utlx_plugin" 2>/dev/null ||
    die "triton-utlx is not installed in $VENV"
if ! "$PY" -c "import expecttest" 2>/dev/null; then
    echo "warning: expecttest is missing, so test_torchtlx_{templates,fusions}" \
        "will fail to import; run: $PY -m pip install expecttest" >&2
fi

mkdir -p "$RESULTS"
SHIM=$(mktemp -d)
WORK=$(mktemp -d)
export TRITON_CACHE_DIR=$(mktemp -d)
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
trap 'rm -rf "$SHIM" "$WORK" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"' EXIT

cat > "$SHIM/fbtriton_tlx_names.py" <<EOF
"""Map the remaining fbtriton-only TLX module names to the checkout's files."""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

FB = "$FBTRITON"
TUTORIALS = os.path.join(FB, "third_party", "tlx", "tutorials")
PYTORCH = os.path.join(FB, "python", "triton", "tlx", "pytorch.py")
# The fork language layer's kernel package (e.g. amd_pa_decode), not the op
# library that triton.tlx.ops is.
LANGUAGE_OPS = os.path.join(FB, "third_party", "tlx", "language", "tlx", "ops")


class _Finder(importlib.abc.MetaPathFinder):
    # Appended, so names that already resolve keep resolving as before.
    def find_spec(self, name, path=None, target=None):
        if name == "triton.language.extra.tlx.tutorials":
            spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
            spec.submodule_search_locations = [TUTORIALS]
            return spec
        if name == "triton.language.extra.tlx.ops":
            return importlib.util.spec_from_file_location(
                name, os.path.join(LANGUAGE_OPS, "__init__.py"),
                submodule_search_locations=[LANGUAGE_OPS])
        if name == "triton.tlx.pytorch":
            return importlib.util.spec_from_file_location(name, PYTORCH)
        return None


sys.meta_path.append(_Finder())
EOF

# Run outside any checkout so nothing on the cwd shadows installed packages.
cd "$WORK" || exit 1
echo "triton-utlx $("$PY" -c 'import importlib.metadata as m; print(m.version("triton-utlx"))')," \
    "triton $("$PY" -c 'import importlib.metadata as m; print(m.version("triton"))')," \
    "fbtriton $(git -C "$FBTRITON" rev-parse --short HEAD 2>/dev/null)"
echo "results: $RESULTS"
echo

summary=()
status=0
for test in "${TESTS[@]}"; do
    name=$(basename "$test" .py)
    log=$RESULTS/$name.log
    PYTHONPATH=$SHIM timeout "$TIMEOUT" "$PY" -m pytest "$test" \
        -p fbtriton_tlx_names -p no:cacheprovider -q -rfE \
        --junitxml="$RESULTS/$name.xml" "$@" > "$log" 2>&1
    rc=$?
    case $rc in
        0 | 5) ;;  # 5: nothing collected/selected (e.g. by -k)
        124) status=1; echo "TIMEOUT after ${TIMEOUT}s" >> "$log" ;;
        *) status=1 ;;
    esac
    last=$(grep -E '(passed|failed|error|skipped|deselected|no tests ran|collected)' "$log" | tail -1)
    [ $rc -gt 128 ] && last="CRASHED (signal $((rc - 128))) ${last}"
    [ $rc -eq 124 ] && last="TIMEOUT ${last}"
    summary+=("$(printf '%-40s %s' "$name" "${last:-see $log}")")
    echo "${summary[-1]}"
done

echo
echo "=== summary (logs and junit xml in $RESULTS) ==="
printf '%s\n' "${summary[@]}"
exit $status
