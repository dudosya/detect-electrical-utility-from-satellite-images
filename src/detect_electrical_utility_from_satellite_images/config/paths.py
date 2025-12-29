"""
Paths configuration for the project.
"""

import logging
from pathlib import Path
from typing import Any

import pydantic
from pydantic import ValidationInfo, field_validator

from .base import PROJECT_ROOT

logger = logging.getLogger(__name__)


class PathsConfig(pydantic.BaseModel):
    """Configuration for project paths."""

    output_dir: Path
    """Output directory for training results."""

    data_dir: Path
    """Directory containing input data."""

    logging_dir_name: Path
    """Directory for log files."""

    @pydantic.field_validator("*", mode="after")
    @classmethod
    def convert_to_absolute_path(cls, v: Any) -> Path:
        """Convert relative paths to absolute paths relative to project root.

        Args:
            v: Path value (could be string or Path object).

        Returns:
            Absolute Path object.
        """
        if isinstance(v, Path) and v.is_absolute():
            return v
        # Convert to Path if it's a string, then make absolute relative to project root
        return (PROJECT_ROOT / Path(v)).resolve()

    @field_validator("*", mode="after")
    @classmethod
    def validate_paths(cls, v: Path, info: ValidationInfo) -> Path:
        """Validate paths and provide warnings for non-existent directories.

        Args:
            v: Path to validate.
            info: Validation context containing field name.

        Returns:
            Validated Path object.
        """
        field_name = info.field_name

        # Skip validation for output_dir as it will be created during training
        if field_name == "output_dir":
            # Check if parent directory is writable
            try:
                v.parent.mkdir(parents=True, exist_ok=True)
                test_file = v.parent / ".write_test"
                test_file.touch()
                test_file.unlink()
            except (OSError, PermissionError):
                logger.warning(f"Output directory may not be writable: {v.parent}")
            return v

        # For data_dir, warn if it doesn't exist (but don't fail - mock data can be used)
        if field_name == "data_dir" and not v.exists():
            logger.warning(
                f"Data directory does not exist: {v}. "
                "Mock data will be used if real data is not available.",
            )

        # For logging_dir_name, ensure parent directory exists
        if field_name == "logging_dir_name":
            try:
                v.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                logger.warning(f"Cannot create logging directory: {v.parent}")

        return v
