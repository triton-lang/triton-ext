python -m pytest fbtriton/python/test/unit/tlx_ops -q

python -m pytest fbtriton/python/test/unit/tlx_ops/test_torchtlx_*.py
python -m pytest fbtriton/python/test/unit/language/test_torchtlx_*.py -q

python -m pytest fbtriton/python/test/unit/tlx_ops/test_kda_gfx950.py -q
python -m pytest fbtriton/python/test/unit/tlx_ops/test_kda_gfx950.py -k decode -q
python -m pytest triton-ext/extensions/utlx/test/tlx_ops/test_kda_gfx950.py -q
python -m pytest triton-ext/extensions/utlx/test/tlx_ops/test_kda_gfx950.py -k decode -q
python triton-ext/extensions/utlx/examples/kda_prefill/run.py

