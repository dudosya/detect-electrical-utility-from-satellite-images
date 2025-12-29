"""
Configuration package for the electrical utility detection project.

This package contains modular configuration classes for different aspects
of the project, following the single responsibility principle.
"""

from .app import AppConfig
from .augmentation import AugmentationConfig
from .base import PROJECT_ROOT, find_project_root
from .logging import LoggingConfig
from .metrics import MetricsConfig
from .model import ModelConfig
from .paths import PathsConfig
from .preprocessing import PreprocessConfig
from .training import CheckpointsConfig, TrainingConfig
from .wandb import WandbConfig

__all__ = [
    "PROJECT_ROOT",
    "AppConfig",
    "AugmentationConfig",
    "CheckpointsConfig",
    "LoggingConfig",
    "MetricsConfig",
    "ModelConfig",
    "PathsConfig",
    "PreprocessConfig",
    "TrainingConfig",
    "WandbConfig",
    "find_project_root",
]
