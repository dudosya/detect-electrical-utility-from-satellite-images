"""Tests for configuration loading."""

from pathlib import Path

import pytest

from detect_electrical_utility_from_satellite_images.config import (
    Config,
    PreprocessingConfig,
    load_config,
)


def test_default_config() -> None:
    """Test that default config values are set correctly."""
    config = Config()

    assert config.preprocessing.patch_size == 500
    assert config.preprocessing.background_fraction == 0.1
    assert config.training.batch_size == 5
    assert config.training.seed == 42


def test_preprocessing_config_validation() -> None:
    """Test that preprocessing config validates correctly."""
    # Valid config
    config = PreprocessingConfig(patch_size=256, background_fraction=0.5)
    assert config.patch_size == 256
    assert config.background_fraction == 0.5

    # Invalid background_fraction should raise
    with pytest.raises(ValueError):
        PreprocessingConfig(patch_size=256, background_fraction=1.5)


def test_load_config_from_yaml(tmp_path: Path) -> None:
    """Test loading config from a YAML file."""
    yaml_content = """
preprocessing:
  patch_size: 256
  background_fraction: 0.2

training:
  batch_size: 8
  seed: 123
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    config = load_config(config_file)

    assert config.preprocessing.patch_size == 256
    assert config.preprocessing.background_fraction == 0.2
    assert config.training.batch_size == 8
    assert config.training.seed == 123


def test_load_config_missing_file() -> None:
    """Test that missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent.yaml"))


def test_config_paths_are_pathlib() -> None:
    """Test that path configs are pathlib.Path objects."""
    config = Config()

    assert isinstance(config.paths.output_dir, Path)
    assert isinstance(config.paths.data_dir, Path)
