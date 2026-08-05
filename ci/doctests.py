#!/usr/bin/env python3
"""
Run the doctests for all scripts that have them in this directory.

Every script with doctests can also be run with `DOCTEST=1 python ...`. After
finding all scripts `import doctest`, this script runs each one as a separate
subprocess.

Usage:
    python ci/doctests.py
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def has_doctests(script: Path) -> bool:
    """Return whether a script opts in to doctests by importing `doctest`."""
    if script.resolve() == Path(__file__).resolve():
        return False
    text = script.read_text(encoding="utf-8")
    has_doctest = "import doctest" in text
    assert not has_doctest or 'common.env2bool("DOCTEST")' in text
    return has_doctest


def test(script: Path) -> bool:
    print(f"Testing {script}", file=sys.stderr)
    env = dict(os.environ, DOCTEST="1")
    result = subprocess.run([sys.executable, script], env=env, check=False)
    return result.returncode == 0


def main() -> int:
    scripts = sorted(s for s in HERE.glob("*.py") if has_doctests(s))
    for script in scripts:
        if not test(script):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
