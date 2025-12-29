"""
Tests for the configuration system.
"""

from pathlib import Path

import pytest
import yaml

from detect_electrical_utility_from_satellite_images.config import AppConfig
from detect_electrical_utility_from_satellite_images.config.base import (
    find_project_root,
)


def test_find_project_root():
    """Test project root discovery."""
    root = find_project_root()
    assert root.exists()
    assert root.name == "detect-electrical-utility-from-satellite-images"
    assert (root / "pyproject.toml").exists()


def test_config_creation(sample_config_dict):
    """Test creating AppConfig from dictionary."""
    config = AppConfig(**sample_config_dict)

    # Test basic attributes
    assert config.preprocessing.patch_size == 256
    assert config.model.architecture == "unet"
    assert config.model.device == "cpu"
    assert config.training.experiment_name == "test_experiment"
    assert config.wandb.project == "test_project"


def test_config_validation():
    """Test configuration validation."""
    # Test invalid architecture
    invalid_config = {
        "preprocessing": {"patch_size": 256, "background_fraction": 0.1},
        "paths": {
            "output_dir": "output",
            "data_dir": "data",
            "logging_dir_name": "logs",
        },
        "model": {
            "architecture": "invalid_arch",  # Invalid
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
            "experiment_name": "test",
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
            "enabled": False,
            "project": "test",
            "entity": "test",
        },
        "logging": {
            "logger_lvl": "INFO",
            "console_handler_lvl": "INFO",
            "file_handler_lvl": "DEBUG",
        },
    }

    with pytest.raises(ValueError):
        AppConfig(**invalid_config)


def test_config_from_yaml(config_yaml_file):
    """Test loading configuration from YAML file."""
    with open(config_yaml_file) as f:
        config_dict = yaml.safe_load(f)

    config = AppConfig(**config_dict)
    assert config.preprocessing.patch_size == 256
    assert config.model.architecture == "unet"


def test_config_paths_absolute(sample_config_dict):
    """Test that paths are converted to absolute paths."""
    config = AppConfig(**sample_config_dict)

    # Paths should be absolute
    assert config.paths.output_dir.is_absolute()
    assert config.paths.data_dir.is_absolute()
    assert config.paths.logging_dir_name.is_absolute()


def test_config_device_validation(sample_config_dict):
    """Test device validation."""
    # Test CPU device
    config_dict = sample_config_dict.copy()
    config_dict["model"]["device"] = "cpu"
    config = AppConfig(**config_dict)
    assert config.model.device == "cpu"

    # Test CUDA device (if available)
    import torch

    if torch.cuda.is_available():
        config_dict["model"]["device"] = "cuda"
        config = AppConfig(**config_dict)
        assert config.model.device == "cuda"
    else:
        # Should still accept "cuda" even if not available
        config_dict["model"]["device"] = "cuda"
        config = AppConfig(**config_dict)
        assert config.model.device == "cuda"


def test_config_validation_errors(sample_config_dict):
    """Test various validation errors."""
    config_dict = sample_config_dict.copy()

    # Test invalid architecture
    config_dict["model"]["architecture"] = "invalid_arch"
    with pytest.raises(ValueError):
        AppConfig(**config_dict)

    # Test invalid encoder name
    config_dict = sample_config_dict.copy()
    config_dict["model"]["encoder_name"] = "invalid_encoder"
    with pytest.raises(ValueError):
        AppConfig(**config_dict)

    # Test invalid in_channels
    config_dict = sample_config_dict.copy()
    config_dict["model"]["in_channels"] = 2  # Should be 3 or 4
    with pytest.raises(ValueError):
        AppConfig(**config_dict)


def test_config_mixed_precision_validation(sample_config_dict):
    """Test mixed precision validation."""
    config_dict = sample_config_dict.copy()

    # Mixed precision with CPU - config accepts it but warns
    config_dict["model"]["device"] = "cpu"
    config_dict["training"]["mixed_precision"] = True
    config = AppConfig(**config_dict)
    assert (
        config.training.mixed_precision is True
    )  # Config accepts it, training will handle

    # Mixed precision with CUDA can be True
    config_dict["model"]["device"] = "cuda"
    config_dict["training"]["mixed_precision"] = True
    config = AppConfig(**config_dict)
    assert config.training.mixed_precision is True


def test_config_module_imports():
    """Test that all config modules can be imported."""
    from detect_electrical_utility_from_satellite_images.config import (
        app,
        augmentation,
        base,
        logging,
        metrics,
        model,
        paths,
        preprocessing,
        training,
        wandb,
    )

    # Just test imports work
    assert base is not None
    assert preprocessing is not None
    assert paths is not None
    assert model is not None
    assert training is not None
    assert augmentation is not None
    assert metrics is not None
    assert wandb is not None
    assert logging is not None
    assert app is not None


def test_config_legacy_wrapper():
    """Test the legacy config.py wrapper."""
    from detect_electrical_utility_from_satellite_images import config as legacy_config

    # Test that AppConfig is available
    assert legacy_config.AppConfig is AppConfig

    # Test that the module can be imported without errors
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Should not raise warnings if used correctly
