"""
Legacy configuration module for backward compatibility.

This module re-exports all configuration classes from the new modular
configuration structure. New code should import directly from the
`detect_electrical_utility_from_satellite_images.config` package.

DEPRECATED: This monolithic config file will be removed in a future version.
Please update imports to use the modular configuration structure.
"""

import warnings

from .config import (
    PROJECT_ROOT,
    AppConfig,
    AugmentationConfig,
    CheckpointsConfig,
    LoggingConfig,
    MetricsConfig,
    ModelConfig,
    PathsConfig,
    PreprocessConfig,
    TrainingConfig,
    WandbConfig,
    find_project_root,
)

# Warn users about the deprecated import path
warnings.warn(
    "Importing from 'detect_electrical_utility_from_satellite_images.config' "
    "is deprecated. Please import from 'detect_electrical_utility_from_satellite_images.config' "
    "package directly.",
    DeprecationWarning,
    stacklevel=2,
)
