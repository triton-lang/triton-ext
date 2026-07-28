#!/usr/bin/env python3
"""
Retrieve a pre-built LLVM artifact from the Triton project's Azure blob storage
and expand it in the current directory. Triton's CI builds LLVM and uploads the
resulting tarballs to a public Azure blob container to Azure:
https://oaitriton.blob.core.windows.net/public/llvm-builds/.

Artifact names follow the pattern `<commit>-<os>-<arch>-<build-number>`.
This script accepts the same pattern as its single argument, which can be
incrementally more specific (e.g., `<commit>`, `<commit>-<os>`, etc.):

- `<commit>`: a short (8-char) LLVM commit hash (optional; defaults to the hash
  pinned in Triton)
- `<os>`: the target OS (optional; defaults to current system; one of: ubuntu,
  almalinux, macos, windows)
- `<arch>`: the architecture (optional; defaults to current system; one of: x64,
  arm64)
- `<build-number>`: the build number (optional; defaults to the number pinned in
  Triton)

Usage:
    python ci/download_llvm.py [<commit>[-<os>[-<arch>[-<build-number>]]]]
"""

import logging
import os
import sys

import common
import probe_sysinfo
import requests

LOG = logging.getLogger(os.path.basename(__file__))
AZURE_BASE_URL = "https://oaitriton.blob.core.windows.net/public/llvm-builds"


def read_triton_hash():
    """Read the pinned Triton hash from the `triton-hash.txt` file."""
    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, "triton-hash.txt")
    hash = open(file).read().strip()
    return hash


def fetch_llvm_build_info(triton_rev):
    """Fetch the LLVM build info (hash, build number, checksums) for a given Triton revision."""
    url = f"https://raw.githubusercontent.com/triton-lang/triton/{triton_rev}/cmake/llvm-info.json"
    LOG.debug(f"Fetching LLVM build info from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_artifact_name(commit, os_name, arch, build_number):
    """Build the artifact base name (without .tar.gz)."""
    short_hash = commit[:8]
    return f"llvm-{short_hash}-{os_name}-{arch}-{build_number}"


def download_artifact(artifact_name):
    """Download an LLVM artifact tarball from Azure blob storage."""
    artifact_file = f"{artifact_name}.tar.gz"
    url = f"{AZURE_BASE_URL}/{artifact_file}"
    LOG.debug(f"Downloading artifact from: {url}")
    common.download_file(url, artifact_file)
    return artifact_file


def main(commit: str | None, os_name: str | None, arch: str | None,
         build_number: str | None, dry_run: bool):

    # Fetch LLVM build info from the pinned Triton revision if any fields are missing.
    build_info = None
    if not commit or not build_number:
        triton_rev = read_triton_hash()
        LOG.debug(f"Found Triton hash: {triton_rev}")
        build_info = fetch_llvm_build_info(triton_rev)
        LOG.debug(f"Fetched LLVM build info: {build_info}")
        assert build_info is not None

    if not commit:
        assert build_info is not None
        commit = build_info["llvm_hash"]
        LOG.debug(f"Found LLVM hash: {commit}")
    if not build_number:
        assert build_info is not None
        build_number = str(build_info["build_number"])
        LOG.debug(f"Found LLVM build number: {build_number}")

    # If no OS or architecture is provided, probe the current system.
    probed_os, probed_arch = probe_sysinfo.run(refine_os=True)
    LOG.debug(f"Probed system: {probed_os}-{probed_arch}")
    if not os_name:
        os_name = probed_os
    if not arch:
        arch = probed_arch

    artifact = get_artifact_name(commit, os_name, arch, build_number)
    if not dry_run:
        tar_gz = download_artifact(artifact)
        if build_info:
            common.verify_checksum(
                tar_gz, build_info["sha256sum"][f"{os_name}-{arch}"])
        common.extract_artifact(tar_gz)

    print(f"Successfully downloaded and installed: {artifact}/",
          file=sys.stderr)


if __name__ == "__main__":
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    dry_run = common.env2bool("DRY_RUN")

    if len(sys.argv) == 1:
        commit, os_name, arch, build_number = (None, None, None, None)
    elif len(sys.argv) == 2:
        parts = sys.argv[1].split('-')
        commit, os_name, arch, build_number = (parts + [None] * 4)[:4]
    else:
        USAGE = "python ci/download_llvm.py [<commit>[-<os>[-<arch>[-<build-number>]]]]"
        LOG.error(f"Usage: {USAGE}")
        sys.exit(1)

    main(commit, os_name, arch, build_number, dry_run)
