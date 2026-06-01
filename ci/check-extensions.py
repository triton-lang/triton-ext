#!/usr/bin/env python3
"""
Report every extension's metadata, or fail if any extension is misconfigured.

Usage:
    python ci/check_extensions.py            # human-readable table
    python ci/check_extensions.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import extension


def path_as_str(p):
    """When serializing to JSON, convert Path objects to strings."""
    return str(p) if isinstance(p, Path) else p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    extensions = extension.discover()
    if args.json:
        # Serializing a @dataclass with Path fields requires some special
        # handling.
        ext_dicts = [ext.__dict__ for ext in extensions]
        print(json.dumps(ext_dicts, indent=2, default=path_as_str))
    else:
        for ext in extensions:
            flag = "enabled" if ext.enabled else "DISABLED"
            owners = " ".join(ext.owners) or "(no owners)"
            print(
                f"{ext.name:<14} {flag:<9} {str(ext.path.parent):<24} owners: {owners}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
