#!/usr/bin/env python3
"""
Parse the `pyproject.toml` manifest: see :load: and :discover:.

This also understands parsing a CODEOWNERS file to associate a list of owners
with an extension: see :parse_codeowners: and :owners_for:.
"""

import doctest
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import common
import tomllib

VALID_STATUS = {"experimental", "stable"}
TRITON_PREFIX = "triton-"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODEOWNERS = REPO_ROOT / ".github" / "CODEOWNERS"
LOG = logging.getLogger(os.path.basename(__file__))


@dataclass(frozen=True)
class Manifest:
    """
    Structured metadata for a Triton extension, parsed from a `pyproject.toml`
    manifest.

    `name` is the short extension name used for display, CMake variables, and
    the plugin ABI; `wheel` is the `pyproject.toml` project name and
    determines the wheel filename.

    >>> Manifest(name="foo", wheel="triton-foo", path=Path("foo"), status="stable", enabled=True, version="1.2.3")
    Manifest(name='foo', wheel='triton-foo', path=PosixPath('foo'), status='stable', enabled=True, version='1.2.3', owners=[])
    >>> Manifest(name="bar", wheel="bar", path=Path("bar"))
    Manifest(name='bar', wheel='bar', path=PosixPath('bar'), status='experimental', enabled=True, version='0.0.0', owners=[])
    """
    name: str
    wheel: str
    path: Path
    status: str = "experimental"
    enabled: bool = True
    version: str = "0.0.0"
    owners: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate the extension metadata."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError(
                f"{self.path}: missing required string field 'name'")
        if not self.wheel or not isinstance(self.wheel, str):
            raise ValueError(
                f"{self.path}: missing required string field 'wheel_name'")
        if self.status not in VALID_STATUS:
            raise ValueError(
                f"{self.path}: invalid status '{self.status}', must be one of: {sorted(VALID_STATUS)}"
            )
        if not isinstance(self.enabled, bool):
            raise TypeError(
                f"{self.path}: 'enabled' must be a boolean, got {self.enabled!r}"
            )
        if not isinstance(self.version, str):
            raise TypeError(f"{self.path}: 'version' field must be a string")
        if not isinstance(self.owners, list) or not all(
                isinstance(o, str) for o in self.owners):
            raise ValueError(
                f"{self.path}: 'owners' must be a list of strings, got {self.owners!r}"
            )


@dataclass(frozen=True)
class CodeOwnerRule:
    """A single rule from a CODEOWNERS file."""
    pattern: str
    owners: list[str] = field(default_factory=list)

    def matches(self, query: str) -> bool:
        """
        Match this rule's pattern against a repo-relative POSIX path.

        >>> CodeOwnerRule("*").matches("foo/bar/baz.py")
        True
        >>> CodeOwnerRule("*.py").matches("foo/bar/baz.py")
        True
        >>> CodeOwnerRule("/foo").matches("foo/bar/baz.py")
        True
        >>> CodeOwnerRule("bar").matches("foo/bar/baz.py")
        False
        >>> CodeOwnerRule("baz.py").matches("foo/bar/baz.py")
        True
        >>> CodeOwnerRule("foo/bar").matches("foo/bar/baz.py")
        True
        """
        query = query.strip("/")
        if self.pattern[0] == "/":
            # If the pattern is anchored, anchor the query as well.
            query = "/" + query
        p = Path(query)
        return p.match(self.pattern) or p.is_relative_to(self.pattern)


def parse_codeowners(codeowners_path: Path) -> list[CodeOwnerRule]:
    """Parse a CODEOWNERS file into a list of CodeOwnerRule objects."""
    rules: list[CodeOwnerRule] = []
    for raw in codeowners_path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        pattern, owners = parts[0], parts[1:]
        rules.append(CodeOwnerRule(pattern, owners))
    return rules


def owners_for(path: str, rules: list[CodeOwnerRule]) -> list[str]:
    """
    Return the owners for a repo-relative path (last matching rule wins).

    >>> owners_for("foo/bar/baz.py", [CodeOwnerRule("*", ["@a"]), CodeOwnerRule("*.py", ["@b"])])
    ['@b']
    """
    result: list[str] = []
    for rule in rules:
        if rule.matches(path):
            result = rule.owners
    return result


def load(manifest_path: Path) -> Manifest:
    """
    Read and validate a `pyproject.toml` manifest attaching any associated code
    owners from the CODEOWNERS file.
    """
    with open(manifest_path, "rb") as f:
        data = tomllib.load(f)
    rules = parse_codeowners(DEFAULT_CODEOWNERS)
    # Resolve symlinks so the comparison is consistent with REPO_ROOT (which is
    # also resolved); otherwise a symlinked checkout breaks `relative_to`.
    path = manifest_path.resolve().relative_to(REPO_ROOT)
    wheel = data["project"]["name"]
    if not wheel.startswith(TRITON_PREFIX):
        LOG.warning("%s: wheel %r has no %r prefix; consider renaming to %r",
                    path, wheel, TRITON_PREFIX, TRITON_PREFIX + wheel)
    name = wheel.removeprefix(TRITON_PREFIX)
    owners = owners_for(path.as_posix(), rules)
    return Manifest(name=name,
                    wheel=wheel,
                    version=data["project"]["version"],
                    status=data["tool"]["triton-ext"].get(
                        "status", "experimental"),
                    enabled=data["tool"]["triton-ext"].get("enabled", True),
                    path=path.parent,
                    owners=owners)


def _out_of_place(path: Path) -> bool:
    """
    Return True if the path is not in a valid extension directory.

    >>> _out_of_place(REPO_ROOT / "extensions" / "foo" / "triton-ext.toml")
    False
    >>> _out_of_place(REPO_ROOT / "triton-7a5d6a3d-linux-x64" / "triton-ext.toml")
    True
    """
    SEARCH_DIRECTORIES = ("backend", "dialect", "extensions", "language",
                          "pass")
    return not any(
        path.is_relative_to(REPO_ROOT / d) for d in SEARCH_DIRECTORIES)


def discover() -> list[Manifest]:
    """Find all `pyproject.toml` files and load them into structured metadata."""
    extensions = []
    for manifest in sorted(REPO_ROOT.rglob("pyproject.toml")):
        if manifest.parent == REPO_ROOT:
            # Skip the root pyproject.toml (for the triton-ext repo itself).
            continue
        if _out_of_place(manifest):
            raise ValueError(
                f"extension manifest found in unexpected location: {manifest}")
        cfg = load(manifest)
        extensions.append(cfg)
    return extensions


if __name__ == "__main__":
    if common.env2bool("DOCTEST"):
        results = doctest.testmod()  # type: ignore[attr-defined]
        sys.exit(int(results.failed > 0))
