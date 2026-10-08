"""_utlx_fbtriton backfills triton._internal_testing in fresh interpreters.

A fresh interpreter is what run_in_process's forkserver child is: it never
loads the pytest plugin, so only utlx_plugin.pth's import of _utlx_fbtriton
can reach it. No GPU required.
"""
import os
import subprocess
import sys

PROBE = """
import triton._internal_testing as it
print(getattr(it, "utlx_loader_probe", lambda: "absent")(), it.is_cuda.__module__)
"""


def _run(tmp_path, root):
    helpers = tmp_path / "python" / "triton"
    helpers.mkdir(parents=True, exist_ok=True)
    (helpers / "_internal_testing.py").write_text(
        "def utlx_loader_probe():\n    return 'fbtriton'\n\n"
        "def is_cuda():\n    return 'fbtriton'\n")
    env = {k: v for k, v in os.environ.items() if k != "UTLX_FBTRITON_ROOT"}
    if root:
        env["UTLX_FBTRITON_ROOT"] = str(tmp_path)
    out = subprocess.run([sys.executable, "-c", PROBE],
                         env=env,
                         cwd=tmp_path,
                         capture_output=True,
                         text=True,
                         check=True)
    return out.stdout.split()


def test_fresh_interpreter_is_backfilled(tmp_path):
    # Missing helpers come from the checkout; stock ones are kept.
    assert _run(tmp_path,
                root=True) == ["fbtriton", "triton._internal_testing"]


def test_unset_root_changes_nothing(tmp_path):
    assert _run(tmp_path, root=False) == ["absent", "triton._internal_testing"]
