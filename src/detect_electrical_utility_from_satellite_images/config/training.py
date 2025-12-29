"""
Training configuration.
"""

import logging
import typing

import pydantic
from pydantic import ValidationInfo, field_validator, model_validator

logger = logging.getLogger(__name__)


class TrainingConfig(pydantic.BaseModel):
    """Enhanced training configuration."""

    experiment_name: str = "baseline"
    """Experiment name used for folder naming and W&B grouping."""

    seed: int = 42
    """Random seed for reproducibility."""

    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients before updating weights."""

    mixed_precision: bool = False
    """Whether to use mixed precision training (requires CUDA)."""

    early_stopping_patience: int = 10
    """Number of epochs to wait for improvement before early stopping."""

    lr_scheduler: typing.Literal["cosine", "reduce_on_plateau", "step", "none"] = (
        "cosine"
    )
    """Learning rate scheduler type."""

    lr_scheduler_params: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {
            "T_max": 50,
            "eta_min": 1e-6,
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
            "step_size": 10,
            "gamma": 0.1,
        },
    )
    """Parameters for the learning rate scheduler."""

    optimizer: typing.Literal["adam", "sgd", "adamw"] = "adam"
    """Optimizer type."""

    optimizer_params: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
        },
    )
    """Parameters for the optimizer."""

    loss_function: typing.Literal["cross_entropy", "dice", "focal", "combined"] = (
        "cross_entropy"
    )
    """Loss function to use."""

    class_weights: list[float] | None = None
    """Manual class weights for loss function."""

    auto_class_weights: bool = False
    """Whether to automatically calculate class weights from dataset."""

    loss_alpha: float = 0.5
    """Alpha parameter for combined losses (weight between CE and Dice)."""

    loss_gamma: float = 2.0
    """Gamma parameter for focal loss."""

    @model_validator(mode="after")
    def validate_loss_parameters(self) -> "TrainingConfig":
        """Validate loss function and related parameters.

        Returns:
            Updated TrainingConfig instance.
        """
        if self.loss_function != "combined" and self.loss_alpha != 0.5:
            logger.warning(
                f"loss_alpha={self.loss_alpha} is only used with 'combined' loss function "
                f"(current: {self.loss_function}). Consider setting loss_alpha=0.5.",
            )

        if self.loss_function != "focal" and self.loss_gamma != 2.0:
            logger.warning(
                f"loss_gamma={self.loss_gamma} is only used with 'focal' loss function "
                f"(current: {self.loss_function}). Consider setting loss_gamma=2.0.",
            )

        # Validate learning rate range
        if (
            self.optimizer == "sgd"
            and self.lr_scheduler_params.get("min_lr", 1e-6) > 0.01
        ):
            logger.warning(
                f"SGD optimizer with min_lr={self.lr_scheduler_params.get('min_lr')} may be too high. "
                f"Typical SGD learning rates are lower than Adam.",
            )

        return self

    @field_validator("class_weights", mode="after")
    @classmethod
    def validate_class_weights(
        cls,
        v: list[float] | None,
        info: ValidationInfo,
    ) -> list[float] | None:
        """Validate class weights if provided.

        Args:
            v: List of class weights or None.
            info: Validation context.

        Returns:
            Validated class weights or None.

        Raises:
            ValueError: If class weights are invalid.
        """
        if v is not None:
            if len(v) < 2:
                raise ValueError(
                    f"class_weights must have at least 2 values (got {len(v)})",
                )
            if any(w <= 0 for w in v):
                raise ValueError("class_weights must be positive values")
            if sum(v) == 0:
                raise ValueError("class_weights sum cannot be zero")

        return v

    @field_validator("gradient_accumulation_steps", mode="after")
    @classmethod
    def validate_gradient_accumulation(cls, v: int) -> int:
        """Validate gradient accumulation steps.

        Args:
            v: Gradient accumulation steps.

        Returns:
            Validated gradient accumulation steps.

        Raises:
            ValueError: If gradient accumulation steps is less than 1.
        """
        if v < 1:
            raise ValueError("gradient_accumulation_steps must be ≥ 1")
        if v > 100:
            logger.warning(f"gradient_accumulation_steps={v} is unusually high")
        return v


class CheckpointsConfig(pydantic.BaseModel):
    """Checkpoint configuration."""

    save_best: bool = True
    """Whether to save the best model based on monitored metric."""

    save_last: bool = True
    """Whether to save the last model at the end of training."""

    metric_to_monitor: str = "val_loss"
    """Metric to monitor for saving best model."""

    mode: typing.Literal["max", "min"] = "min"
    """Whether to maximize or minimize the monitored metric."""

    min_delta: float = 0.001
    """Minimum improvement required to save checkpoint."""

    save_every_n_epochs: int | None = None
    """Save every N epochs regardless of improvement."""
