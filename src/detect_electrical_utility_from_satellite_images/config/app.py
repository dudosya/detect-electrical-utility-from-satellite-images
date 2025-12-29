"""
Main application configuration that combines all configuration modules.
"""

import logging
from pathlib import Path

import pydantic
import pydantic_settings
from pydantic import model_validator

from .augmentation import AugmentationConfig
from .base import PROJECT_ROOT
from .logging import LoggingConfig
from .metrics import MetricsConfig
from .model import ModelConfig
from .paths import PathsConfig
from .preprocessing import PreprocessConfig
from .training import CheckpointsConfig, TrainingConfig
from .wandb import WandbConfig

logger = logging.getLogger(__name__)


class AppConfig(pydantic_settings.BaseSettings):
    """Main application configuration combining all sub-configurations."""

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        frozen=False,  # Changed from True to allow path formatting
    )

    preprocessing: PreprocessConfig
    """Preprocessing configuration."""

    paths: PathsConfig
    """Paths configuration."""

    logging: LoggingConfig
    """Logging configuration."""

    model: ModelConfig
    """Model configuration."""

    wandb: WandbConfig = pydantic.Field(default_factory=WandbConfig)
    """Weights & Biases configuration."""

    metrics: MetricsConfig = pydantic.Field(default_factory=MetricsConfig)
    """Metrics tracking configuration."""

    training: TrainingConfig = pydantic.Field(default_factory=TrainingConfig)
    """Training configuration."""

    augmentation: AugmentationConfig = pydantic.Field(
        default_factory=AugmentationConfig
    )
    """Data augmentation configuration."""

    checkpoints: CheckpointsConfig = pydantic.Field(default_factory=CheckpointsConfig)
    """Checkpoint configuration."""

    @pydantic.model_validator(mode="after")
    def format_paths(self) -> "AppConfig":
        """Format path placeholders with actual values after validation.

        Returns:
            Updated AppConfig instance with formatted paths.
        """
        # Format output_dir with patch_size and experiment_name
        output_dir_str = str(self.paths.output_dir)
        if "{patch_size}" in output_dir_str or "{experiment_name}" in output_dir_str:
            formatted_output_dir = output_dir_str.format(
                patch_size=self.preprocessing.patch_size,
                experiment_name=self.training.experiment_name,
            )
            # Update the output_dir path
            self.paths.output_dir = Path(formatted_output_dir)

        # Format logging_dir_name with experiment_name
        logging_dir_str = str(self.paths.logging_dir_name)
        if "{experiment_name}" in logging_dir_str:
            formatted_logging_dir = logging_dir_str.format(
                experiment_name=self.training.experiment_name,
            )
            # Update the logging_dir_name path
            self.paths.logging_dir_name = Path(formatted_logging_dir)

        # Re-run the field validator to convert to absolute paths
        # Create a new PathsConfig instance with updated values
        updated_paths = PathsConfig(
            output_dir=self.paths.output_dir,
            data_dir=self.paths.data_dir,
            logging_dir_name=self.paths.logging_dir_name,
        )

        # Use object.__setattr__ to bypass frozen instance restriction
        object.__setattr__(self, "paths", updated_paths)

        return self

    @model_validator(mode="after")
    def validate_cross_configuration(self) -> "AppConfig":
        """Validate cross-configuration dependencies.

        Returns:
            Updated AppConfig instance.
        """
        # Check if class_weights length matches classes
        if (
            self.training.class_weights is not None
            and len(self.training.class_weights) != self.model.classes
        ):
            logger.warning(
                f"class_weights length ({len(self.training.class_weights)}) "
                f"does not match model.classes ({self.model.classes})",
            )

        # Check if mixed precision is enabled but device is CPU
        if self.training.mixed_precision and self.model.device == "cpu":
            logger.warning(
                "mixed_precision is enabled but device is 'cpu'. "
                "Mixed precision only works with CUDA.",
            )

        # Check if gradient accumulation makes sense with batch size
        if self.training.gradient_accumulation_steps > 1 and self.model.batch_size == 1:
            logger.info(
                f"gradient_accumulation_steps={self.training.gradient_accumulation_steps} "
                f"with batch_size=1. Effective batch size is "
                f"{self.training.gradient_accumulation_steps}.",
            )

        # Check patch size compatibility with typical GPU memory
        patch_size = self.preprocessing.patch_size
        batch_size = self.model.batch_size
        channels = self.model.in_channels

        # Rough memory estimate (bytes): batch_size * patch_size² * channels * 4 (float32)
        # For mixed precision: half that
        memory_per_sample = patch_size * patch_size * channels * 4
        total_memory = batch_size * memory_per_sample

        if self.training.mixed_precision:
            total_memory /= 2

        if total_memory > 4e9:  # 4GB
            logger.warning(
                f"Estimated GPU memory usage: {total_memory / 1e9:.1f}GB. "
                f"Consider reducing batch_size or patch_size if you encounter OOM errors.",
            )

        return self
