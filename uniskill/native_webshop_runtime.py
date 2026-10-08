"""Ray worker logging setup for WebShop."""

import sys


def setup_worker_logging() -> None:
    """Forward worker stdout to stderr."""
    sys.stdout = sys.stderr
