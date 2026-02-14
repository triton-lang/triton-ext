#!/usr/bin/env python3
"""
This script retrieves the LLVM commit hash used by Triton. By default, it will retrieve the version pinned in Triton's
`main` branch on GitHub. But, if passed a single argument--`<triton-rev>`--it will retrieve the LLVM commit hash for
that specific revision (i.e., `https://github.com/triton-lang/triton/blob/${triton_rev}/cmake/llvm-hash.txt`).

Usage:
    python fetch-llvm-hash.py [<triton-rev>]
"""

import sys
import requests
import logging


def run(triton_rev):
    """Retrieve the LLVM commit hash pinned at this Triton revision."""
    url = f'https://raw.githubusercontent.com/triton-lang/triton/{triton_rev}/cmake/llvm-hash.txt'
    hash = requests.get(url).text.strip()
    return hash


if __name__ == "__main__":
    triton_rev = 'main'
    if len(sys.argv) > 1:
        triton_rev = sys.argv[1]
    hash = run(triton_rev)
    print(hash)
