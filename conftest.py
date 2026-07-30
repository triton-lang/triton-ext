"""
Pytest integration for lit-style MLIR tests.

Each ``.mlir`` file carrying a ``// RUN:`` directive is collected as a pytest
item and executed the way ``lit`` would: the RUN command is run from the file's
directory.

RUN-line substitutions:
  * ``%s``          -> the ``.mlir`` file path
  * ``%filecheck``  -> ``<python> -m filecheck --enable-var-scope``
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

# Directories that never contain first-party tests but may hold stray ``.mlir``
# files (downloaded artifacts, build trees, caches, virtualenvs).
_EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "build", "dist", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "__pycache__"
}
_EXCLUDE_PREFIXES = ("llvm-", "triton-")

_RUN_RE = re.compile(r"//\s*RUN:\s*(.+?)\s*$", re.MULTILINE)
# Bare ``name.py`` tokens (not preceded by ``/`` or the python ``-m`` form).
_PY_TOKEN_RE = re.compile(r"(?<![\w./])([\w.-]+\.py)\b")

_FILECHECK = f"{shlex.quote(sys.executable)} -m filecheck --enable-var-scope"

_REPO_ROOT = Path(__file__).resolve().parent


def excluded(path: Path) -> bool:
    # Only consider components *below* the repo root: the repo directory itself
    # is named ``triton-ext``, which would otherwise match ``triton-`` and
    # exclude everything.
    try:
        rel = path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return False
    for part in rel.parts:
        if part in _EXCLUDE_DIRS or part.startswith(_EXCLUDE_PREFIXES):
            return True
    return False


def pytest_collect_file(parent, file_path):
    path = Path(file_path)
    if path.suffix == ".mlir" and not excluded(path):
        return MlirFile.from_parent(parent, path=file_path)
    return None


class MlirFile(pytest.File):
    """A ``.mlir`` file; yields one item per ``// RUN:`` directive."""

    def collect(self):
        text = self.path.read_text()
        runs = _RUN_RE.findall(text)
        for index, command in enumerate(runs):
            name = "RUN" if len(runs) == 1 else f"RUN[{index}]"
            yield MlirItem.from_parent(self, name=name, command=command)


class MlirItem(pytest.Item):

    def __init__(self, *, command: str, **kwargs):
        super().__init__(**kwargs)
        self.command = command

    def _resolve(self) -> str:
        testdir = self.path.parent
        command = self.command.replace("%s", shlex.quote(str(self.path)))
        command = command.replace("%filecheck", _FILECHECK)

        def _sub(match: re.Match) -> str:
            token = match.group(1)
            script = testdir / token
            if script.exists():
                return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
            return token

        return _PY_TOKEN_RE.sub(_sub, command)

    def runtest(self) -> None:
        command = self._resolve()
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(self.path.parent),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise MlirFailure(command, result.returncode, result.stdout,
                              result.stderr)

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, MlirFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style=style)

    def reportinfo(self):
        return self.path, 0, f"lit: {self.name}"


class MlirFailure(Exception):

    def __init__(self, command: str, returncode: int, stdout: str,
                 stderr: str):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        parts = [
            f"RUN command failed (exit {self.returncode}):",
            f" {self.command}",
        ]
        if self.stdout.strip():
            parts += ["", "--- stdout ---", self.stdout.rstrip()]
        if self.stderr.strip():
            parts += ["", "--- stderr ---", self.stderr.rstrip()]
        return "\n".join(parts)
