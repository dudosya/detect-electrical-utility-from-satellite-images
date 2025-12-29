"""
Model configuration.
"""

import logging
import typing

import pydantic
from pydantic import ValidationInfo, field_validator, model_validator

logger = logging.getLogger(__name__)


class ModelConfig(pydantic.BaseModel):
    """Configuration for the segmentation model."""

    architecture: typing.Literal["unet", "deeplabv3", "fpn"]
    """Model architecture to use."""

    encoder_name: typing.Literal[
        "resnet18", "resnet34", "resnet50", "efficientnet-b0", "mobilenet_v2"
    ]
    """Backbone encoder name."""

    encoder_weights: typing.Literal["imagenet", None]
    """Pretrained weights for the encoder."""

    in_channels: typing.Annotated[int, pydantic.Field(ge=3, le=4)]
    """Number of input channels (3 for RGB, 4 for RGBA)."""

    classes: typing.Annotated[int, pydantic.Field(ge=2)]
    """Number of output classes (must be ≥ 2)."""

    learning_rate: typing.Annotated[float, pydantic.Field(gt=0.0)]
    """Learning rate for optimizer."""

    batch_size: typing.Annotated[int, pydantic.Field(ge=1)]
    """Batch size for training."""

    epochs: typing.Annotated[int, pydantic.Field(ge=1)]
    """Number of training epochs."""

    device: typing.Literal["cpu", "cuda"]
    """Device to use for training ('cpu' or 'cuda')."""

    @model_validator(mode="after")
    def validate_device_compatibility(self) -> "ModelConfig":
        """Validate device compatibility and adjust if needed.

        Returns:
            Updated ModelConfig instance.
        """
        try:
            import torch

            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available. Changing device to 'cpu'."
                )
                self.device = "cpu"
        except ImportError:
            logger.warning("torch not available for device validation")

        return self

    @field_validator("batch_size", mode="after")
    @classmethod
    def validate_batch_size(cls, v: int, info: ValidationInfo) -> int:
        """Validate batch size considering typical GPU memory.

        Args:
            v: Batch size value.
            info: Validation context.

        Returns:
            Validated batch size.
        """
        # Only warn for very large batch sizes
        if v > 32:
            logger.warning(f"batch_size={v} is large. Ensure sufficient GPU memory.")
        if v == 1:
            logger.info(
                "batch_size=1 may lead to unstable gradients. Consider gradient_accumulation_steps."
            )

        return v

    @field_validator("learning_rate", mode="after")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate range.

        Args:
            v: Learning rate value.

        Returns:
            Validated learning rate.
        """
        if v > 0.1:
            logger.warning(
                f"learning_rate={v} is very high. Typical values are 0.001 or lower."
            )
        if v < 1e-6:
            logger.warning(f"learning_rate={v} is very low. Training may be very slow.")

        return v

    @field_validator("epochs", mode="after")
    @classmethod
    def validate_epochs(cls, v: int, info: ValidationInfo) -> int:
        """Validate epochs relative to early stopping patience.

        Args:
            v: Number of epochs.
            info: Validation context containing parent configuration data.

        Returns:
            Validated number of epochs.
        """
        # Check if early_stopping_patience is available in parent context
        if "training" in info.data:
            training_config = info.data["training"]
            if "early_stopping_patience" in training_config:
                patience = training_config["early_stopping_patience"]
                if patience >= v:
                    logger.warning(
                        f"early_stopping_patience={patience} is >= epochs={v}. "
                        f"Training may stop before completing all epochs.",
                    )

        return v
