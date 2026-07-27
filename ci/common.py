"""
Common utility functions for CI scripts.
"""

import logging
import os
import shutil
import sys
import tarfile


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


def extract_artifact(artifact_file):
    """Extract a tar.gz artifact."""
    output_dir = artifact_file.replace(".tar.gz", "")
    if os.path.exists(output_dir):
        logging.debug(f"Deleting existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    logging.debug(f"Extracting artifact: {artifact_file}")
    try:
        with tarfile.open(artifact_file, "r:gz") as tar:
            tar.extractall(filter=filter_data)
    except Exception as e:
        logging.error(f"Error extracting artifact: {e}")
        sys.exit(1)
