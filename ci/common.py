"""
Common utility functions for CI scripts.
"""

import hashlib
import logging
import os
import shutil
import sys
import tarfile

LOG = logging.getLogger(os.path.basename(__file__))


def env2bool(variable: str) -> bool:
    """Convert an environment variable string to a boolean."""
    return os.getenv(variable, 'false').lower() in ('true', '1', 't')


def is_contained_path(path):
    """Check if a path is contained (i.e., does not contain '..' or start with '/')."""
    return not (path.startswith('/') or '..' in path)


def filter_data(tarinfo, path):
    """Apply 'data' filter behavior but skip certain symlinks."""
    # Skip symlinks if they point outside the extraction directory.
    if tarinfo.issym() or tarinfo.islnk():
        if not is_contained_path(tarinfo.linkname):
            logging.warning(
                f"Skipping symlink: {tarinfo.name} -> {tarinfo.linkname}")
            return None

    # Apply 'data' filter behavior: strip dangerous metadata, but preserve
    # executability for files that were executable in the archive.
    if tarinfo.isdir():
        tarinfo.mode = 0o755
    else:
        tarinfo.mode = 0o755 if (tarinfo.mode & 0o111) else 0o644
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = ""

    # Block paths outside the extraction directory.
    if not is_contained_path(tarinfo.name):
        logging.warning(f"Skipping potentially unsafe path: {tarinfo.name}")
        return None

    return tarinfo


def verify_checksum(filename, expected_sha256):
    """Verify the SHA-256 checksum of a downloaded file."""
    LOG.debug(f"Verifying checksum: {filename}")
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        LOG.error(f"Checksum mismatch for {filename}:")
        LOG.error(f"  expected: {expected_sha256}")
        LOG.error(f"  actual:   {actual}")
        os.remove(filename)
        sys.exit(1)
    LOG.debug(f"Checksum verified: {actual}")


def download_file(url, filename, headers=None, auth=None):
    """Stream-download a file with a progress bar; skip if already present."""
    import requests

    if os.path.exists(filename):
        LOG.debug(f"File already exists locally: {filename}")
        return

    LOG.debug(f"Downloading: {url}")
    kwargs = {}
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    response = requests.get(url, stream=True, timeout=120, **kwargs)
    if response.status_code == 404:
        LOG.error(f"File not found: {url}")
        sys.exit(1)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(
                    f"\r  {pct}% ({downloaded // (1 << 20)} / {total // (1 << 20)} MiB)",
                    end="",
                    flush=True,
                    file=sys.stderr,
                )
    if total:
        print(file=sys.stderr)  # newline after progress bar


def extract_artifact(artifact_file):
    """Extract a tar.gz artifact."""
    output_dir = artifact_file.replace(".tar.gz", "")
    if os.path.exists(output_dir):
        LOG.debug(f"Deleting existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    LOG.debug(f"Extracting artifact: {artifact_file}")
    try:
        with tarfile.open(artifact_file, "r:gz") as tar:
            tar.extractall(filter=filter_data)
    except Exception as e:
        LOG.error(f"Error extracting artifact: {e}")
        sys.exit(1)
