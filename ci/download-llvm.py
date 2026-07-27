#!/usr/bin/env python3
"""
Retrieve a pre-built LLVM artifact from the Triton project's Azure blob storage
and expand it in the current directory. Triton's CI builds LLVM and uploads the
resulting tarballs to a public Azure blob container to Azure:
https://oaitriton.blob.core.windows.net/public/llvm-builds/.

Artifact names follow the pattern `llvm-<commit>-<os>-<arch>-<build-number>`.
This script accepts the same pattern as its single argument, which can be
incrementally more specific (e.g., `<llvm>`, `llvm-<commit>`, etc.):

- `llvm` (required)
- `<commit>`: a short (8-char) LLVM commit hash (optional; defaults to the hash
  pinned in Triton)
- `<os>`: the target OS (optional; defaults to current system; one of: ubuntu,
  almalinux, macos, windows)
- `<arch>`: the architecture (optional; defaults to current system; one of: x64,
  arm64)
- `<build-number>`: the build number (optional; defaults to the number pinned in
  Triton)

Usage:
    python ci/download-llvm.py llvm[-<commit>[-<os>[-<arch>[-<build-number>]]]]
"""

import hashlib
import logging
import os
import sys

import common
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


def probe_sysinfo():
    """Get the current OS and architecture."""
    import importlib
    module = importlib.import_module("probe-sysinfo")
    return module.run()


def get_artifact_name(commit, os_name, arch, build_number):
    """Build the artifact base name (without .tar.gz)."""
    short_hash = commit[:8]
    return f"llvm-{short_hash}-{os_name}-{arch}-{build_number}"


def download_artifact(artifact_name):
    """Download an LLVM artifact tarball from Azure blob storage."""
    artifact_file = f"{artifact_name}.tar.gz"
    if os.path.exists(artifact_file):
        LOG.debug(f"Artifact already exists locally: {artifact_file}")
        return artifact_file

    url = f"{AZURE_BASE_URL}/{artifact_file}"
    LOG.debug(f"Downloading artifact from: {url}")

    response = requests.get(url, stream=True)
    if response.status_code == 404:
        LOG.error(f"Artifact not found: {url}")
        sys.exit(1)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(artifact_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(
                    f"\r  {pct}% ({downloaded // (1 << 20)} / {total // (1 << 20)} MiB)",
                    end="",
                    flush=True,
                    file=sys.stderr)
    if total:
        print(file=sys.stderr)  # newline after progress

    return artifact_file


def verify_checksum(artifact_file, expected_sha256):
    """Verify the SHA-256 checksum of a downloaded file."""
    LOG.debug(f"Verifying checksum: {artifact_file}")
    sha256 = hashlib.sha256()
    with open(artifact_file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        LOG.error(f"Checksum mismatch for {artifact_file}:")
        LOG.error(f"  expected: {expected_sha256}")
        LOG.error(f"  actual:   {actual}")
        os.remove(artifact_file)
        sys.exit(1)
    LOG.debug(f"Checksum verified: {actual}")


def main(project: str, commit: str | None, os_name: str | None,
         arch: str | None, build_number: str | None, dry_run: bool):
    assert project == "llvm", f"Only 'llvm' is supported; got '{project}'"

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
    probed_os, probed_arch = probe_sysinfo()
    LOG.debug(f"Probed system: {probed_os}-{probed_arch}")
    if not os_name:
        os_name = probed_os
    if not arch:
        arch = probed_arch

    artifact = get_artifact_name(commit, os_name, arch, build_number)
    if not dry_run:
        tar_gz = download_artifact(artifact)
        if build_info:
            verify_checksum(tar_gz,
                            build_info["sha256sum"][f"{os_name}-{arch}"])
        common.extract_artifact(tar_gz)

    print(f"Successfully downloaded and installed: {artifact}/",
          file=sys.stderr)


if __name__ == "__main__":
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    dry_run = common.env2bool("DRY_RUN")

    USAGE = "python download-llvm.py llvm[-<commit>[-<os>[-<arch>[-<build-number>]]]]"
    if len(sys.argv) != 2:
        LOG.error(f"Usage: {USAGE}")
        sys.exit(1)

    parts = sys.argv[1].split('-')
    project, commit, os_name, arch, build_number = (parts + [None] * 5)[:5]
    if project != "llvm":
        LOG.error("Invalid project name; expected 'llvm'.")
        LOG.error(f"Usage: {USAGE}")
        sys.exit(1)

    main(project, commit, os_name, arch, build_number, dry_run)
