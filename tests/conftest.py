"""
Shared pytest fixtures for testing.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from detect_electrical_utility_from_satellite_images.config import AppConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict():
    """Return a minimal valid configuration dictionary."""
    return {
        "preprocessing": {
            "patch_size": 256,
            "background_fraction": 0.1,
        },
        "paths": {
            "output_dir": "output",
            "data_dir": "data",
            "logging_dir_name": "logs",
        },
        "model": {
            "architecture": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 5,
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 10,
            "device": "cpu",
        },
        "training": {
            "experiment_name": "test_experiment",
            "seed": 42,
            "mixed_precision": False,
            "loss_function": "cross_entropy",
        },
        "augmentation": {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "color_jitter_prob": 0.5,
            "gaussian_blur_prob": 0.5,
        },
        "metrics": {
            "track_iou": True,
            "track_dice": True,
            "track_accuracy": True,
        },
        "wandb": {
            "project": "test_project",
            "entity": "test_entity",
            "run_name": "test_run",
            "tags": ["test"],
            "log_model": False,
            "save_code": False,
        },
        "logging": {
            "logger_lvl": "info",
            "console_handler_lvl": "info",
            "file_handler_lvl": "debug",
        },
    }


@pytest.fixture
def sample_config(sample_config_dict):
    """Create an AppConfig instance from sample config dict."""
    return AppConfig(**sample_config_dict)


@pytest.fixture
def config_yaml_file(temp_dir, sample_config_dict):
    """Create a temporary YAML config file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def mock_image_paths(temp_dir):
    """Create mock image files for testing."""
    img_dir = temp_dir / "img_patches"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for i in range(5):
        img_path = img_dir / f"sample_img_{i}.png"
        # Create empty file (in real tests, would be actual images)
        img_path.touch()
        img_paths.append(img_path)

    return img_paths


@pytest.fixture
def mock_mask_paths(temp_dir):
    """Create mock mask files for testing."""
    mask_dir = temp_dir / "mask_patches"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = []
    for i in range(5):
        mask_path = mask_dir / f"sample_msk_{i}.png"
        # Create empty file (in real tests, would be actual masks)
        mask_path.touch()
        mask_paths.append(mask_path)

    return mask_paths
