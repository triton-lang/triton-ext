#!/usr/bin/env python3
"""
List all available Triton wheel versions.

- **nightly** wheels come from the Triton's Azure `feed`_ published to by CI
  (default)
- **release** wheels come from `PyPI`_

This module exports the :class:`Wheel` class and the :func:`run` function, which
fetches a list of wheels for a given channel (a `PEP 503`_ index). When run as a
script it prints the retrieved wheel names to stdout.

Usage:
    python ci/list_triton_wheels.py [nightly|release]

.. _feed: https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/triton/
.. _PEP 503: https://peps.python.org/pep-0503/
.. _PyPI: https://pypi.org/simple/triton/
"""

import logging
import os
import sys
from dataclasses import dataclass
from html.parser import HTMLParser

import requests

LOG = logging.getLogger(os.path.basename(__file__))
CHANNELS = {
    "nightly":
    "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/triton/",
    "release": "https://pypi.org/simple/triton/"
}
HEADERS = {"User-Agent": "pip/24.0"}


@dataclass
class Wheel:
    """A Triton wheel retrieved from a PEP 503 index."""
    filename: str
    url: str
    sha256: str | None

    def __init__(self, filename: str, url: str):
        self.filename = filename
        if "#sha256=" in url:
            self.sha256 = url.split("#sha256=", 1)[1]
            self.url = url.split("#", 1)[0]
        else:
            self.sha256 = None
            self.url = url

    def __str__(self):
        return f"{self.filename}"

    def version(self) -> str:
        """
        Return the base version string from a wheel filename.

        >>> Wheel("triton-3.8.0-cp314-cp314-linux_x86_64.whl", "https://...").version()
        '3.8.0'
        >>> Wheel("triton-3.8.0+gitf6ef5434-cp314-cp314-linux_x86_64.whl", "https://...").version()
        '3.8.0+gitf6ef5434'
        >>> Wheel("triton-3.7.1+gitf6ef5434-cp314-cp314-linux_x86_64.whl", "https://...").version()
        '3.7.1+gitf6ef5434'
        """
        return self.filename.split("-")[1]

    def tags(self) -> tuple[str, str, str]:
        """
        Return the Python tags (interpreter, ABI, platform) from a wheel
        filename.

        >>> Wheel("triton-3.8.0-cp314-cp314-linux_x86_64.whl", "https://...").tags()
        ('cp314', 'cp314', 'linux_x86_64')
        >>> Wheel("triton-3.8.0+gitf6ef5434-cp314-cp314-linux_x86_64.whl", "https://...").tags()
        ('cp314', 'cp314', 'linux_x86_64')
        """
        stem = self.filename.rsplit(".", 1)[0]
        interpreter, abi, platform, remaining = stem.split("-")[2:]
        assert not remaining
        return (interpreter, abi, platform)


def _fetch_index(channel: str) -> str:
    """Fetch the PEP 503 simple index for the given channel; return raw HTML."""
    url = CHANNELS[channel]
    # The Azure DevOps feed only serves the full index to requests that look
    # like pip; otherwise authentication is required.
    headers = HEADERS if channel == "nightly" else {}
    LOG.debug(f"Fetching index: {url}")
    response = requests.get(url, timeout=60, headers=headers)
    response.raise_for_status()
    return response.text


class _WheelIndexParser(HTMLParser):
    """Parse a PEP 503 simple index; see `_parse_anchors`."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[Wheel] = []
        self._href: str | None = None
        self._text: str | None = None
        self._in_anchor = False

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        self._in_anchor = True
        self._href = dict(attrs).get("href", "")

    def handle_data(self, data: str) -> None:
        if self._in_anchor and self._href and data.strip().endswith(".whl"):
            self.results.append(Wheel(data.strip(), self._href))

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_anchor = False
            self._href = None


def _parse_anchors(html: str) -> list[Wheel]:
    """Parse a PEP 503 simple-index HTML page."""
    parser = _WheelIndexParser()
    parser.feed(html)
    return parser.results


def run(channel) -> list[Wheel]:
    """Return all wheel entries for the given channel."""
    html = _fetch_index(channel)
    wheels = _parse_anchors(html)
    LOG.debug(f"Parsed {len(wheels)} wheel anchors from {channel} index")
    return wheels


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    import common
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    if common.env2bool("TEST"):
        import doctest
        results = doctest.testmod()
        sys.exit(int(results.failed > 0))

    channel = sys.argv[1] if len(sys.argv) > 1 else "nightly"
    if channel not in CHANNELS:
        print("Usage: python ci/list_triton_wheels.py [nightly|release]",
              file=sys.stderr)
        sys.exit(1)

    for wheel in run(channel):
        print(wheel)
