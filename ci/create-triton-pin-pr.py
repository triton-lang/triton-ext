#!/usr/bin/env python3
"""
Create an automation PR for a Triton pin change. It looks at the `triton-hash.txt` file and, if it has changes, pushes
a branch and a PR for that branch to the `origin` remote.

This script requires the `gh` CLI tool to interact with the GitHub API. This is available in GitHub Actions, but users
running this locally will need to install it (https://cli.github.com) and authenticate (`gh auth login`) prior to use.

This script takes no input can be configured with environment variables:
- `DRY_RUN`: if set, the script will not push changes or create a PR (e.g., `DRY_RUN=1`).
- `VERBOSE`: if set, enables debug logging (e.g., `VERBOSE=1`).

Usage:
	[DRY_RUN={1|0}] [VERBOSE={1|0}] python ci/create-triton-pin-pr.py
"""

import subprocess
import sys
import os
import logging


def exec(args: list[str],
         check: bool = True,
         capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command and return the completed process."""
    logging.debug(f"> {' '.join(args)}")
    try:
        return subprocess.run(args,
                              check=check,
                              text=True,
                              capture_output=capture_output)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e}")
        if e.stdout:
            logging.error(f"stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"stderr: {e.stderr}")
        raise


def is_changed(path: str) -> bool:
    """Check if a file has changes compared to the Git index."""
    return exec(["git", "diff", "--quiet", "--", path],
                check=False).returncode != 0


def push_pin_branch(hash: str, file: str, dry_run: bool) -> str:
    """Create and push a branch for the Triton hash update."""
    branch = f"ci/update-triton-pin-{hash}"
    title = f"ci: update Triton pin to {hash}"
    exec(["git", "checkout", "-B", branch])
    exec(["git", "add", file])
    exec(["git", "commit", "-m", title])
    if not dry_run:
        exec(["git", "push", "--force", "--set-upstream", "origin", branch])
    return branch


def find_matching_pr(branch: str) -> str:
    """Find an existing PR that matches the given base and head branches."""
    return exec(
        [
            "gh",
            "pr",
            "list",
            "--base",
            "main",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "number",
            "--jq",
            ".[0].number // empty",
        ],
        capture_output=True,
    ).stdout.strip()


def open_pr(branch, hash, dry_run: bool):
    """Open a new PR for the Triton pin update."""
    if dry_run:
        return

    title = f"ci: update Triton pin to {hash}"
    body = "Automated update of `triton-hash.txt` to Triton commit "
    body += f"[{hash}](https://github.com/triton-lang/triton/commit/{hash})."
    exec([
        "gh",
        "pr",
        "create",
        "--base",
        "main",
        "--head",
        branch,
        "--title",
        title,
        "--body",
        body,
    ])


def main(dry_run: bool):
    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, "triton-hash.txt")
    if not is_changed(file):
        print(f"No change in {file}; nothing to commit.")
        return

    hash = open(file).read().strip()
    logging.debug(f"Updating Triton pin to {hash}")
    if not hash:
        logging.error(f"Error: Triton hash is empty in {file}")
        sys.exit(1)

    branch = push_pin_branch(hash, file, dry_run)
    existing_pr = find_matching_pr(branch)
    if existing_pr:
        print(f"PR #{existing_pr} already exists for {branch}")
        return
    open_pr(branch, hash, dry_run)


def env2bool(variable: str) -> bool:
    """Convert an environment variable string to a boolean."""
    return os.getenv(variable, 'false').lower() in ('true', '1', 't')


if __name__ == "__main__":
    logging.getLogger().name = os.path.basename(__file__)
    if env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)
    dry_run = env2bool("DRY_RUN")

    main(dry_run)
