"""
Base configuration utilities and constants.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root directory by looking for pyproject.toml.

    Returns:
        Path to project root directory.

    Raises:
        ValueError: If pyproject.toml is not found.
    """
    root_dir = Path(__file__).resolve()

    while not (root_dir / "pyproject.toml").exists():
        if root_dir != root_dir.parent:
            root_dir = root_dir.parent
        else:
            raise ValueError("Folder containing pyproject.toml is not found")

    logger.debug(f"Found project root: {root_dir}")
    return root_dir


PROJECT_ROOT = find_project_root()
"""Project root directory path."""
