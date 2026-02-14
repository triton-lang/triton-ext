#!/usr/bin/env python3
"""
This script retrieves the latest Triton commit hash from GitHub.

Usage:
    python fetch-triton-hash.py
"""

import requests


def run():
    """Retrieve the latest Triton commit hash from GitHub."""
    json = requests.get(
        'https://api.github.com/repos/triton-lang/triton/commits/main').json()
    return json['sha']


if __name__ == "__main__":
    sha = run()
    print(sha)
