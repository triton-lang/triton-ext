
echo "Setup Environment..."
D=$(mktemp -d -t utlx-tests.XXXXXX)

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

echo "Running Kimi KDA from fbtriton..."
python -m pytest fbtriton/python/test/unit/tlx_ops/test_kda_gfx950.py -q
python -m pytest fbtriton/python/test/unit/tlx_ops/test_kda_gfx950.py -k decode -q

echo "Running Kimi KDA from triton-ext..."
python -m pytest triton-ext/extensions/utlx/test/tlx_ops/test_kda_gfx950.py -q
python -m pytest triton-ext/extensions/utlx/test/tlx_ops/test_kda_gfx950.py -k decode -q

echo "Running Kimi KDA prefill run.py..."
python triton-ext/extensions/utlx/examples/kda_prefill/run.py



