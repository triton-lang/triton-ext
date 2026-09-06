from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_AGPU = Path(__file__).resolve().parents[1] / "agpu"


def _run(*argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def agpu_build(tmp_path_factory):
    for tool in ("cmake", "ctest"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is not installed")

    build = tmp_path_factory.mktemp("agpu-build")
    argv = ["cmake", "-S", str(_AGPU), "-B", str(build)]
    if shutil.which("ninja"):
        argv += ["-G", "Ninja"]
    got = _run(*argv)
    if got.returncode != 0:
        pytest.fail(f"cmake configure failed:\n{got.stdout}\n{got.stderr}")

    got = _run("cmake", "--build", str(build))
    if got.returncode != 0:
        pytest.fail(f"cmake build failed:\n{got.stdout}\n{got.stderr}")
    return build


def test_agpu_ctest_suite_passes(agpu_build):
    got = _run("ctest", "--output-on-failure", cwd=agpu_build)
    assert got.returncode == 0, got.stdout + got.stderr
