"""
Metrics tracking configuration.
"""

import typing

import pydantic


class MetricsConfig(pydantic.BaseModel):
    """Metrics tracking configuration."""

    track: list[
        typing.Literal[
            "iou", "dice", "accuracy", "precision", "recall", "f1", 
            "boundary_iou", "boundary_f1"
        ]
    ] = pydantic.Field(
        default_factory=lambda: [
            "iou",
            "dice",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ],
    )
    """List of metrics to track during training."""

    iou_average: typing.Literal["macro", "micro", "weighted"] = "macro"
    """Averaging method for IoU (Intersection over Union)."""

    dice_average: typing.Literal["macro", "micro", "weighted"] = "macro"
    """Averaging method for Dice coefficient."""

    f1_average: typing.Literal["macro", "micro", "weighted"] = "macro"
    """Averaging method for F1 score."""

    boundary_iou_average: typing.Literal["macro", "micro", "weighted"] = "macro"
    """Averaging method for Boundary IoU."""

    boundary_f1_average: typing.Literal["macro", "micro", "weighted"] = "macro"
    """Averaging method for Boundary F1."""

    boundary_dilation: int = pydantic.Field(default=2, ge=1, le=10)
    """Dilation radius for boundary extraction (1-10)."""

    log_samples: int = 4
    """Number of sample predictions to log."""

    sample_indices: list[int] = pydantic.Field(default_factory=lambda: [0, 1, 2, 3])
    """Indices of samples to log predictions for."""
