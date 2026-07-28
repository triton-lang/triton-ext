#!/usr/bin/env python3
"""
This script retrieves the latest Triton commit hash with a matching nightly
wheel.

Usage:
    python fetch_triton_hash.py
"""

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime

import common
import list_triton_wheels as wheels

LOG = logging.getLogger(os.path.basename(__file__))


def clone(url: str, depth: int, dest: str):
    """
    Clone some number of commits from a repository `url` to the `dest`
    directory.
    """
    LOG.debug(f"Cloning: {url} (depth={depth})")
    result = subprocess.run(
        ["git", "clone", "--depth",
         str(depth), "--no-checkout", url, dest],
        capture_output=True,
        text=True,
        check=True)
    if result.stdout:
        LOG.debug(result.stdout.strip())
    if result.stderr:
        LOG.debug(result.stderr.strip())


@dataclass
class Commit:
    """A Git commit with its hash and date."""
    hash: str
    date: datetime

    def __init__(self, hash: str, date: str) -> None:
        self.hash = hash
        self.date = datetime.fromisoformat(date)


def scan(dir: str) -> list[Commit]:
    """
    Scan the available commits and return a list of `(commit, date)`.
    """
    result = subprocess.run(["git", "log", "--pretty=format:%H %ci"],
                            cwd=dir,
                            capture_output=True,
                            text=True,
                            check=True)
    lines = result.stdout.strip().splitlines()
    commits = [
        Commit(hash, date)
        for (hash, date) in (line.split(" ", 1) for line in lines)
    ]
    LOG.debug(f"Parsed {len(commits)} commits")
    return commits


def run(repo: str, depth: int) -> str | None:
    """Retrieve the latest Triton commit hash from GitHub."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clone(repo, depth, tmpdir)
        commits = scan(tmpdir)
        sorted_by_date = sorted(commits, key=lambda x: x.date, reverse=True)

    candidates = wheels.run("nightly")
    LOG.debug(f"Found {len(candidates)} nightly wheels")
    for commit in sorted_by_date:
        LOG.debug(f"Checking commit: {commit.hash} ({commit.date})")
        short = commit.hash[:8]
        for wheel in candidates:
            if short in wheel.filename:
                LOG.info(f"Found matching wheel for commit: {commit.hash}")
                return commit.hash

    LOG.error("No matching wheel found")
    return None


if __name__ == "__main__":
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    repo = os.getenv("REPO", "https://github.com/triton-lang/triton")
    depth = int(os.getenv("DEPTH", "100"))
    sha = run(repo, depth)
    if sha:
        print(sha)
    else:
        sys.exit(1)
