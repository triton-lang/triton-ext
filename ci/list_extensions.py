#!/usr/bin/env python3
"""
Report every extension's metadata, or fail if any extension is misconfigured.

Optionally, pass a `<field>` argument to print only the field for each
extension: `name`, `path`, `status`, `enabled`, `version`, or `owners`.

Usage:
    python ci/list_extensions.py [<field>]
"""

from __future__ import annotations

import logging
import sys

import common
import extension


def as_str(ext: extension.Manifest, field: str) -> str:
    """Return the string representation of a field of an extension."""
    if not hasattr(ext, field):
        raise ValueError(f"Invalid field '{field}', must be one of: "
                         f"{', '.join(ext.__dataclass_fields__.keys())}")
    value = getattr(ext, field)
    if field == "path":
        value = str(value)
    elif field == "owners":
        value = " ".join(value) or "(no owners)"
    elif field == "enabled":
        value = "enabled" if value else "DISABLED"
    return value


def run(field: str | None):
    """
    Print a table of all extensions and their metadata, or just a column if a
    single field is specified.
    """
    extensions = extension.discover()
    for ext in extensions:
        if field is not None:
            if not hasattr(ext, field):
                raise ValueError(
                    f"Invalid field '{field}', must be one of: "
                    f"{', '.join(ext.__dataclass_fields__.keys())}")
            print(as_str(ext, field))
        else:
            print(
                f"{as_str(ext, 'name'):<20} {as_str(ext, 'version'):<7} {as_str(ext, 'enabled'):<9} {as_str(ext, 'path'):<24} owners: {as_str(ext, 'owners'):<20}"
            )


if __name__ == "__main__":
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    field = sys.argv[1] if len(sys.argv) > 1 else None
    run(field)
