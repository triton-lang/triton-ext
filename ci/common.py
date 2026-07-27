"""
Common utility functions for CI scripts.
"""

import os


def env2bool(variable: str) -> bool:
    """Convert an environment variable string to a boolean."""
    return os.getenv(variable, 'false').lower() in ('true', '1', 't')
