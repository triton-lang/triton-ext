#!/usr/bin/env python3
"""
This script returns the youngest artifact directory it can find. It will search in the current directory by default;
override this with `SEARCH_DIR`. It is passed a single, hyphen-delimited argument that can be incrementally more
specific (e.g., `<project>`, `<project>-<commit>`, `<project>-<commit>-<os>`, etc.):
- `<project>`: either `llvm` or `triton` (required)
- `<commit>`: a full or short commit hash (optional; defaults to latest Triton commit and corresponding LLVM commit)
- `<os>`: the operating system (optional; defaults to current system)
- `<arch>`: the architecture (optional; defaults to current system)

Usage:
    [SEARCH_DIR=<path>] python pick-local-artifact.py <project>[-<commit>[-<os>[-<arch>]]]
"""

import sys
import os


def run(search_dir: str, criteria: str) -> str:
    """Get the youngest artifact directory matching the given criteria."""
    candidates = []
    for entry in os.listdir(search_dir):
        if os.path.isdir(os.path.join(
                search_dir, entry)) and entry.lower().startswith(criteria):
            candidates.append(entry)

    if not candidates:
        print(
            f"No artifact directory found matching criteria: {criteria}. "
            f"Try downloading the artifact from GitHub (if cached): "
            f"`python ci/download-artifact.py {criteria}`.",
            file=sys.stderr)
        sys.exit(1)

    candidates.sort(
        key=lambda x: os.path.getmtime(os.path.join(search_dir, x)),
        reverse=True)
    return candidates[0]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pick-local-artifact.py "
              "<project>[-<commit>[-<os>[-<arch>]]]")
        sys.exit(1)
    criteria = sys.argv[1].lower()
    search_dir = os.getenv("SEARCH_DIR", ".")
    artifact_dir = run(search_dir, criteria)
    print(artifact_dir)
