#!/usr/bin/env python3
"""
Retrieve a pre-built Triton or LLVM artifact and expand it in the current directory.

This expects a single argument that can be incrementally more specific:
- `<project>`: either `llvm` or `triton` (required)
- `<commit>`: a full or short commit hash (optional; defaults to latest Triton commit and corresponding LLVM commit)
- `<os>`: the operating system (optional; defaults to current system)
- `<arch>`: the architecture (optional; defaults to current system)

This script requires the `gh` CLI tool to interact with the GitHub API. This is available in GitHub Actions, but users
running this locally will need to install it (https://cli.github.com) and authenticate (`gh auth login`) prior to use.

Usage:
    python ci/download-artifact.py <project>[-<commit>[-<os>[-<arch>]]]
"""

import sys
import os
import subprocess
import tarfile
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


def main(repository: str, project: str, commit: str, os_name: str, arch: str,
         dry_run: bool):
    assert project in ("llvm", "triton")

    # If no commit hash is provided, fetch the latest Triton hash and corresponding LLVM hash.
    if not commit:
        triton_rev = fetch_triton_hash()
        logging.debug(f"Found Triton hash: {triton_rev}")
        if project == "triton":
            commit = triton_rev
        else:
            commit = fetch_llvm_hash(triton_rev)
            logging.debug(f"Found LLVM hash: {commit}")

    # If no OS or architecture is provided, probe the current system.
    probed_os_name, probed_arch = probe_sysinfo()
    logging.debug(f"Probed system: {probed_os_name}-{probed_arch}")
    if not os_name:
        os_name = probed_os_name
    if not arch:
        arch = probed_arch

    artifact = f"{project}-{commit[:8]}-{os_name}-{arch}"
    if not dry_run:
        tar_gz = download_artifact(repository, artifact)
        extract_artifact(tar_gz)

    print(f"Successfully downloaded and installed: {artifact}/",
          file=sys.stderr)


def env2bool(variable: str) -> bool:
    """Convert an environment variable string to a boolean."""
    return os.getenv(variable, 'false').lower() in ('true', '1', 't')


if __name__ == "__main__":
    logging.getLogger().name = os.path.basename(__file__)
    if env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    repository = os.getenv('GITHUB_REPOSITORY', 'triton-lang/triton-ext')
    dry_run = env2bool("DRY_RUN")

    USAGE = "python download-artifact.py <project>[-<commit>[-<os>[-<arch>]]]"
    if len(sys.argv) != 2:
        logging.error(f"Usage: {USAGE}")
        sys.exit(1)

    project, commit, os_name, arch = (sys.argv[1].split('-') + [None] * 4)[:4]
    if project not in ("llvm", "triton"):
        logging.error("Invalid project name; expected 'llvm' or 'triton'.")
        logging.error(f"Usage: {USAGE}")
        sys.exit(1)
    main(repository, project, commit, os_name, arch, dry_run)
