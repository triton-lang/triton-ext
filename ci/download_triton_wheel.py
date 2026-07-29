#!/usr/bin/env python3
"""
Download a pre-built Triton wheel to the current directory.

The wheel is selected based on:

- `channel`: "nightly" or "release" (default: nightly)
- `wheel-pattern`: the name of a Triton wheel, with glob wildcards allowed
  (default: on nightly, match the pinned Triton commit, Python tags, and current
  architecture)

For ease of use, the wheel can be specified as a glob pattern, e.g.:

- `triton-*`: will match the first wheel fetched by `list_triton_wheels.py`
- `triton-*+git0d7dc8626*`: will match a nightly wheel with the given commit
  hash
- `triton-3.8.0-*x86_64.whl`: will match an x86 release wheel at version 3.8.0

This is helpful given that wheel filenames may be quite long, e.g.:

- `triton-3.8.0-cp314-cp314-linux_x86_64.whl`
- `triton-3.5.0+git07dc8626-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`

Usage (NOTE: quote the pattern to avoid shell expansion):
    python ci/download_triton_wheel.py [channel] ['wheel-pattern']
"""

import logging
import os
import sys
from fnmatch import fnmatch

import common
import list_triton_wheels as wheels
import probe_sysinfo

USAGE = "Usage: python ci/download_triton_wheel.py [channel] ['wheel-pattern']"
LOG = logging.getLogger(os.path.basename(__file__))


def read_triton_hash():
    """Read the pinned Triton commit hash from ci/triton-hash.txt."""
    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, "triton-hash.txt")
    return open(file).read().strip()


def probe_python_tag():
    """Return the CPython tag for the current Python version."""
    major, minor = sys.version_info[:2]
    return f"cp{major}{minor}"


def normalize_arch(arch):
    """
    Normalise an arch name and return the substring that identifies it in
    Triton wheel platform tags ("x86_64" or "aarch64").
    """
    if arch in ("x64", "x86_64", "amd64"):
        return "x86_64"
    if arch in ("arm64", "aarch64"):
        return "aarch64"
    LOG.error(f"Unrecognised arch: {arch!r}; expected 'x64' or 'arm64'")
    sys.exit(1)


def select_wheel(candidates: list[wheels.Wheel],
                 pattern: str) -> wheels.Wheel | None:
    """
    Return the first wheel in `candidates` that matches the `pattern`.

    >>> select_wheel([], "triton-*") is None
    True
    >>> select_wheel([wheels.Wheel("triton-3.8.0-cp314-...whl", "https://...")], "triton-*").version()
    '3.8.0'
    >>> select_wheel([
    ...   wheels.Wheel("triton-3.8.0+gitf6ef5434-...x86_64.whl", "https://..."),
    ...   wheels.Wheel("triton-3.8.0+gitf6ef5434-...aarch64.whl", "https://..."),
    ... ], "triton-*f6ef5434*aarch64*").filename
    'triton-3.8.0+gitf6ef5434-...aarch64.whl'
    >>>
    """
    LOG.debug(f"Searching for pattern: {pattern}")
    for wheel in candidates:
        if fnmatch(wheel.filename, pattern):
            LOG.debug(f"Found: {wheel}")
            return wheel
    return None


def download_wheel(url, filename, channel):
    """Stream-download a wheel file with a progress bar; skip if present."""
    # The Azure DevOps feed only serves the full index to requests that look
    # like pip; otherwise authentication is required.
    headers = wheels.HEADERS if channel == "nightly" else {}
    common.download_file(url, filename, headers=headers)


def main(
    channel: str,
    pattern: str | None,
    dry_run: bool,
):
    """
    Download a Triton wheel from the given `channel`, optionally matching a
    wheel `pattern`.
    """
    if channel not in wheels.CHANNELS:
        LOG.error(f"Invalid channel: {channel}")
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    # Construct pattern if missing on nightly channel.
    if channel == "nightly" and pattern is None:
        ref = read_triton_hash()
        pytag = probe_python_tag()
        os, probed_arch = probe_sysinfo.run()
        arch = normalize_arch(probed_arch)
        pattern = f"triton-*+git{ref[:8]}-{pytag}-{pytag}-*{os}*_{arch}*.whl"

    if not pattern:
        LOG.error("No wheel pattern specified.")
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    candidates = wheels.run(channel)
    wheel = select_wheel(candidates, pattern)
    if not wheel:
        LOG.error(f"No wheel found matching pattern: {pattern}")
        sys.exit(1)

    if not dry_run:
        download_wheel(wheel.url, wheel.filename, channel)
        if wheel.sha256:
            common.verify_checksum(wheel.filename, wheel.sha256)

    print(wheel.filename)


if __name__ == "__main__":
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    if common.env2bool("TEST"):
        import doctest
        results = doctest.testmod()
        sys.exit(int(results.failed > 0))

    dry_run = common.env2bool("DRY_RUN")
    channel = sys.argv[1] if len(sys.argv) > 1 else "nightly"
    wheel = sys.argv[2] if len(sys.argv) > 2 else None
    main(channel, wheel, dry_run)
