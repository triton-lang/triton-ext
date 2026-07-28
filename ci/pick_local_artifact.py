#!/usr/bin/env python3
"""
This script returns the youngest artifact it can find that matches a pattern. It
will search in the current directory by default; override this with
`SEARCH_DIR`. It is passed a single glob pattern (i.e., Unix filename matching);
see `fnmatch`_ for details.

.. _fnmatch: https://docs.python.org/3/library/fnmatch.html

Usage (NOTE: quote the pattern to avoid shell expansion):
    [SEARCH_DIR=<path>] python pick-local-artifact.py '<pattern>'
"""

import logging
import os
import sys
from fnmatch import fnmatch

import common

LOG = logging.getLogger(os.path.basename(__file__))


def run(search_dir: str, pattern: str, only_dir: bool = False) -> str:
    """Get the youngest artifact directory matching the given glob pattern."""
    LOG.debug(f"Searching for pattern: {pattern}")
    candidates = []
    for entry in os.listdir(search_dir):
        is_dir = os.path.isdir(os.path.join(search_dir, entry))
        if only_dir and not is_dir:
            continue
        if fnmatch(entry, pattern):
            LOG.debug(f"Found candidate: {entry}")
            candidates.append(entry)

    if not candidates:
        print(
            f"No artifact found matching pattern: {pattern}. "
            f"Try downloading the artifact; see `ci/download*.py`.",
            file=sys.stderr)
        sys.exit(1)

    candidates.sort(
        key=lambda x: os.path.getmtime(os.path.join(search_dir, x)),
        reverse=True)
    LOG.debug(f"Picked: {candidates[0]}")
    return candidates[0]


if __name__ == "__main__":
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python pick-local-artifact.py <pattern>",
              file=sys.stderr)
        sys.exit(1)
    pattern = sys.argv[1].lower()
    search_dir = os.getenv("SEARCH_DIR", ".")
    artifact = run(search_dir, pattern)
    print(artifact)
