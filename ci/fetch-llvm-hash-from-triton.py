#!/usr/bin/env python3
"""
This script retrieves the LLVM commit hash used by Triton. By default, it will retrieve the version pinned in the `main`
branch on GitHub. But, if passed a single argument--`<triton-rev>`--it will retrieve the LLVM commit hash for that
specific revision (i.e., `https://github.com/triton-lang/triton/blob/${triton_rev}/cmake/llvm-hash.txt`).

Usage:
    ./fetch-llvm-hash-from-triton.py [<triton-rev>]
"""

import sys
import requests

triton_rev = 'main'
if len(sys.argv) > 1:
    triton_rev = sys.argv[1]
url = f'https://raw.githubusercontent.com/triton-lang/triton/{triton_rev}/cmake/llvm-hash.txt'
hash = requests.get(url).text.strip()
print(hash)
