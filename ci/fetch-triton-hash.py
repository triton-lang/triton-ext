#!/usr/bin/env python3
"""
This script retrieves the latest Triton commit hash from GitHub.

Usage:
    ./fetch-triton-hash.py
"""

import requests

json = requests.get(
    'https://api.github.com/repos/triton-lang/triton/commits/main').json()
print(json['sha'])
