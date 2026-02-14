#!/usr/bin/env python3
"""
Retrieves the artifact ID from GitHub Actions for a given artifact name: `<llvm|triton>-<commit hash>-<os>-<arch>`. If
the name is omitted, this lists all available artifacts and their IDs, returning a zero exit code. If the name is
provided but not found, this will return a non-zero exit code and print the available artifacts to stderr.

This script requires the `gh` CLI tool to interact with the GitHub API. This is available in GitHub Actions, but users
running this locally will need to install it (https://cli.github.com) and authenticate (`gh auth login`) prior to use.

This script can be configured with environment variables:
- `GITHUB_REPOSITORY`: The repository to query for artifacts (default: `triton-lang/triton-ext`).
- `VERBOSE`: if set, enables debug logging (e.g., `VERBOSE=1`).

Usage:
    [GITHUB_TOKEN=<relative-path>] [VERBOSE={1|0}] python fetch-artifact-id.py [artifact-name]
"""

from dataclasses import dataclass
import sys
import os
import json
import subprocess
import logging


@dataclass
class Artifact:
    id: int
    name: str
    created_at: str

    def __str__(self):
        return f"{self.name} = {self.id}"


def fetch(repository) -> list[Artifact]:
    """Retrieve the list of artifacts for the given repository using the GitHub CLI."""
    cmd = ["gh", "api", f"repos/{repository}/actions/artifacts"]
    logging.debug(f"> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    artifacts_data = json.loads(result.stdout)
    logging.debug(f"Fetched artifacts: {json.dumps(artifacts_data, indent=2)}")
    return [
        Artifact(id=a['id'], name=a['name'], created_at=a['created_at'])
        for a in artifacts_data['artifacts']
    ]


def print_all(artifacts):
    """Print all artifacts to stderr."""
    for artifact in artifacts:
        print(f"{artifact}", file=sys.stderr)


def choose(artifacts, artifact_name):
    """Find an artifact by name and return its ID; if not found, print all artifacts and exit with an error."""
    artifact_id = None
    for artifact in artifacts:
        if artifact.name == artifact_name:
            return artifact.id
    if not artifact_id:
        print(
            f"[ERROR] Artifact {artifact_name} not found; available artifacts:",
            file=sys.stderr)
        print_all(artifacts)
        sys.exit(1)


def main(repository: str, artifact_name: str | None):
    artifacts = fetch(repository)
    if artifact_name:
        artifact_id = choose(artifacts, artifact_name)
        print(artifact_id)
    else:
        print_all(artifacts)


def env2bool(variable: str) -> bool:
    """Convert an environment variable string to a boolean."""
    return os.getenv(variable, 'false').lower() in ('true', '1', 't')


if __name__ == "__main__":
    logging.getLogger().name = os.path.basename(__file__)
    if env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    repository = os.getenv('GITHUB_REPOSITORY', 'triton-lang/triton-ext')
    artifact_name = sys.argv[1] if len(sys.argv) > 1 else None
    main(repository, artifact_name)
