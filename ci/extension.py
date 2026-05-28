#!/usr/bin/env python3
"""
Parse the `triton-ext.toml` manifest: see :load:.

This also understands parsing a CODEOWNERS file to associate a list of owners
with an extension: see :parse_codeowners: and :owners_for:.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VALID_STATUS = {"experimental", "stable"}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODEOWNERS = REPO_ROOT / ".github" / "CODEOWNERS"


@dataclass(frozen=True)
class Manifest:
    """
    Structured metadata for a Triton extension, parsed from a `triton-ext.toml`
    manifest.

    >>> Manifest(name="foo", path=Path("foo"), status="stable", enabled=True, version="1.2.3")
    Manifest(name='foo', path=PosixPath('foo'), status='stable', enabled=True, version='1.2.3', owners=[])
    >>> Manifest(name="bar", path=Path("bar"))
    Manifest(name='bar', path=PosixPath('bar'), status='experimental', enabled=True, version='0.0.0', owners=[])
    """
    name: str
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
        if self.status not in VALID_STATUS:
            raise ValueError(
                f"{self.path}: invalid status '{self.status}', must be one of: {sorted(VALID_STATUS)}"
            )
        if not isinstance(self.enabled, bool):
            raise ValueError(
                f"{self.path}: 'enabled' must be a boolean, got {self.enabled!r}"
            )
        if not isinstance(self.version, str):
            raise ValueError(f"{self.path}: 'version' field must be a string")
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
    Read and validate a `triton-ext.toml` manifest attaching any associated code
    owners from the CODEOWNERS file.
    """
    with open(manifest_path, "rb") as f:
        data = tomllib.load(f)
    rules = parse_codeowners(DEFAULT_CODEOWNERS)
    path = manifest_path.absolute().relative_to(REPO_ROOT)
    owners = owners_for(path.as_posix(), rules)
    return Manifest(name=data["name"],
                    path=path,
                    status=data.get("status", "experimental"),
                    enabled=data.get("enabled", True),
                    version=data.get("version", "0.0.0"),
                    owners=owners)


if __name__ == "__main__":
    """Running this file will run the inline doctests."""
    import doctest
    doctest.testmod()
