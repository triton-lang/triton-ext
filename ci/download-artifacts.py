#!/usr/bin/env python3
"""
For a given Triton commit hash, retrieve pre-built Triton and LLVM artifacts and expand them in the current directory.
If no Triton commit hash is provided, this will retrieve the latest version from the `main` branch.

This script requires the `gh` CLI tool to interact with the GitHub API. This is available in GitHub Actions, but users
running this locally will need to install it (https://cli.github.com) and authenticate (`gh auth login`) prior to use.

Usage:
    python ci/download-artifacts.py [<triton-rev>]
"""

import sys
import os
import subprocess
import tarfile
from pathlib import Path
import logging
import shutil


def fetch_triton_hash():
    """See fecth-triton-hash.py."""
    import importlib
    module = importlib.import_module("fetch-triton-hash")
    return module.run()


def fetch_llvm_hash(triton_rev):
    """Fetch the LLVM hash for a given Triton revision."""
    import importlib
    module = importlib.import_module("fetch-llvm-hash")
    return module.run(triton_rev)


def probe_sysinfo():
    """Get the current OS and architecture in lowercase."""
    import importlib
    module = importlib.import_module("probe-sysinfo")
    return module.run()


def get_artifact_name(artifact_type, commit_hash, os_name, arch):
    """Generate artifact name based on type and commit info."""
    short_hash = commit_hash[:8]
    return f"{artifact_type}-{short_hash}-{os_name}-{arch}"


def exec(cmd):
    """Run a shell command."""
    logging.debug(f"> {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {cmd}")
        if e.stdout:
            logging.error(e.stdout)
        if e.stderr:
            logging.error(e.stderr)
        sys.exit(1)


def download_artifact(repository, artifact_name):
    """Download an artifact using the gh CLI tool."""
    logging.debug(f"Downloading artifact: {artifact_name}")
    artifact_file = f"{artifact_name}.tar.gz"
    if os.path.exists(artifact_file):
        logging.debug(f"Artifact already exists: {artifact_file}")
        return artifact_file

    exec(f"gh run download --repo {repository} --name {artifact_name}")

    if not os.path.exists(artifact_file):
        logging.error(
            f"Error: Downloaded artifact file not found: {artifact_file}")
        sys.exit(1)

    return artifact_file


def filter_data_no_symlinks(tarinfo, path):
    """Apply 'data' filter behavior but skip symlinks."""
    # Skip symlinks.
    if tarinfo.issym() or tarinfo.islnk():
        return None

    # Apply 'data' filter behavior: strip dangerous metadata.
    tarinfo.mode = 0o755 if tarinfo.isdir() else 0o644
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = ""

    # Block absolute paths and paths with '..'.
    if tarinfo.name.startswith('/') or '..' in tarinfo.name:
        return None

    return tarinfo


def extract_artifact(artifact_file):
    """Extract a tar.gz artifact."""
    output_dir = artifact_file.replace(".tar.gz", "")
    if os.path.exists(output_dir):
        logging.debug(f"Deleting existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    logging.debug(f"Extracting artifact: {artifact_file}")
    try:
        with tarfile.open(artifact_file, "r:gz") as tar:
            tar.extractall(filter=filter_data_no_symlinks)
    except Exception as e:
        logging.error(f"Error extracting artifact: {e}")
        sys.exit(1)


def main(repository: str, dry_run: bool):
    triton_rev = sys.argv[1] if len(sys.argv) > 1 else None
    if not triton_rev:
        triton_rev = fetch_triton_hash()
    logging.debug(f"Found Triton hash: {triton_rev}")

    llvm_hash = fetch_llvm_hash(triton_rev)
    logging.debug(f"Found LLVM hash: {llvm_hash}")

    os_name, arch = probe_sysinfo()
    logging.debug(f"Analyzed system: {os_name}-{arch}")

    llvm_artifact = get_artifact_name("llvm", llvm_hash, os_name, arch)
    triton_artifact = get_artifact_name("triton", triton_rev, os_name, arch)
    if not dry_run:
        llvm_file = download_artifact(repository, llvm_artifact)
        extract_artifact(llvm_file)

        triton_file = download_artifact(repository, triton_artifact)
        extract_artifact(triton_file)

    print("", file=sys.stderr)
    print("Successfully downloaded and extracted artifacts.", file=sys.stderr)
    print(f"LLVM installation: {llvm_artifact}/", file=sys.stderr)
    print(f"Triton installation: {triton_artifact}/", file=sys.stderr)


def env2bool(variable: str) -> bool:
    """Convert an environment variable string to a boolean."""
    return os.getenv(variable, 'false').lower() in ('true', '1', 't')


if __name__ == "__main__":
    logging.getLogger().name = os.path.basename(__file__)
    if env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    repository = os.getenv('GITHUB_REPOSITORY', 'triton-lang/triton-ext')
    dry_run = env2bool("DRY_RUN")
    main(repository, dry_run)
