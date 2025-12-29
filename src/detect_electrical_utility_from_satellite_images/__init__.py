"""
Electrical utility detection from satellite images.

This package provides tools for training and evaluating segmentation models
to detect electrical utilities in satellite imagery.
"""

from .config import AppConfig, ModelConfig, PathsConfig, TrainingConfig
from .dataset import SatteliteImgsDataset
from .losses import calculate_class_weights, create_loss_function
from .metrics import calculate_all_metrics, visualize_predictions
from .model import create_model, load_checkpoint
from .preprocess import create_patches, jpg_paths_to_patches, pad_to_patch_size
from .train import train_model

__version__ = "0.1.0"
__author__ = "dudosya"
__email__ = "kenaykay@gmail.com"

__all__ = [
    # Configuration
    "AppConfig",
    "ModelConfig",
    "TrainingConfig",
    "PathsConfig",
    # Model
    "create_model",
    "load_checkpoint",
    # Data
    "SatteliteImgsDataset",
    # Training
    "train_model",
    # Metrics
    "calculate_all_metrics",
    "visualize_predictions",
    # Preprocessing
    "create_patches",
    "pad_to_patch_size",
    "jpg_paths_to_patches",
    # Loss functions
    "create_loss_function",
    "calculate_class_weights",
]


def main() -> None:
    """Entry point for the package."""
    print("Electrical Utility Detection from Satellite Images")
    print(f"Version: {__version__}")
    print(f"Author: {__author__} ({__email__})")
    print("\nAvailable components:")
    for component in __all__:
        print(f"  - {component}")
