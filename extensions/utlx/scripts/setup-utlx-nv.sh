
echo ""
echo ""
echo "Setup Environment..."
# D=$(mktemp -d -t utlx-tests.XXXXXX)
D=`pwd`

python3.11 -m venv $D/triton-clean-venv --prompt triton-clean
source $D/triton-clean-venv/bin/activate

pip install --upgrade pip
pip3 install torch

pip install pytest numpy
pip install triton-utlx
pip install expecttest
# pip uninstall -y triton

echo ""
echo ""
echo "Cloning fbtriton and triton-ext..."
git clone https://github.com/facebookexperimental/triton fbtriton
git clone https://github.com/triton-lang/triton-ext


echo ""
echo ""
echo "Reseting fbtriton to a stable hash"
cd fbtriton
git reset --hard  d0fce1bbdb0087eb73b1e46cd6cb5b7aaa4a0cb9
cd -

# Never reuse cached results: Triton's cache key ignores the uTLX plugin, so
# always recompile kernels, and keep stale bytecode out of runs.
export TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR=$D/triton-cache

echo ""
echo ""
echo "Running TorchTLX Unit Tests..."
python -m pytest fbtriton/python/test/unit/tlx_ops/test_torchtlx_*.py -q --import-mode=importlib \
    --rootdir=$D/fbtriton/python/test/unit/tlx_ops \
    -o pythonpath=$D/fbtriton/python/test/unit/tlx_ops

python -m pytest fbtriton/python/test/unit/language/test_torchtlx_*.py -q --import-mode=importlib \
    --rootdir=$D/fbtriton/python/test/unit/language \
    -o pythonpath=$D/fbtriton/python/test/unit/language \
    --deselect test_torchtlx_templates.py::TestTLXTemplates::test_tlx_bmm_shared_a_fused_row_bias \
    --deselect test_torchtlx_templates.py::TestTLXTemplates::test_tlx_bmm_shared_a_shape0 \
    --deselect test_torchtlx_templates.py::TestLocalBufferRetention::test_inductor_codegen \
    --deselect test_torchtlx_templates.py::TestHeuristicConfigSelection::test_config_selection_rule_1b \
    --deselect test_torchtlx_templates.py::TestHeuristicConfigSelection::test_config_selection_rule_3 \
    --deselect test_torchtlx_templates.py::TestHeuristicConfigSelection::test_config_selection_rule_4

echo ""
echo ""
echo "Running ALL TLX.ops Tests..."
python -m pytest fbtriton/python/test/unit/tlx_ops -q --import-mode=importlib \
    --rootdir=$D/fbtriton/python/test/unit/tlx_ops \
    -o pythonpath=$D/fbtriton/python/test/unit/tlx_ops
