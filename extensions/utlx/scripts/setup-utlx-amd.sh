
echo "Setup Environment..."
# D=$(mktemp -d -t utlx-tests.XXXXXX)
D=`pwd`

python3.11 -m venv $D/triton-clean-venv --prompt triton-clean
source $D/triton-clean-venv/bin/activate

pip install --upgrade pip
pip3 install torch  --index-url https://download.pytorch.org/whl/rocm7.2

pip install pytest numpy
pip install triton-utlx
# pip uninstall -y triton

echo "Cloning fbtriton and triton-ext..."
git clone https://github.com/facebookexperimental/triton fbtriton
git clone https://github.com/triton-lang/triton-ext

echo "Running Kimi KDA..."

# Never reuse cached results: Triton's cache key ignores the uTLX plugin, so
# always recompile kernels, and keep stale bytecode out of runs.
export TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR=$D/triton-cache

# --import-mode=importlib: fbtriton/python/conftest.py would otherwise put the
# unbuilt fbtriton/python/triton ahead of the installed Triton on sys.path.
echo "Running Kimi KDA from fbtriton..."
python -m pytest fbtriton/python/test/unit/tlx_ops/test_kda_gfx950.py -q --import-mode=importlib
python -m pytest fbtriton/python/test/unit/tlx_ops/test_kda_gfx950.py -k decode -q --import-mode=importlib

echo "Running Kimi KDA from triton-ext..."
python -m pytest triton-ext/extensions/utlx/test/tlx_ops/test_kda_gfx950.py -q
python -m pytest triton-ext/extensions/utlx/test/tlx_ops/test_kda_gfx950.py -k decode -q

echo "Running Kimi KDA prefill run.py..."
python triton-ext/extensions/utlx/examples/kda_prefill/run.py
