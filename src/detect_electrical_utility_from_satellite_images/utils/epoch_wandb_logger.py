"""W&B logger utilities.

This module provides a small WandbLogger wrapper that uses epoch as the logged
step value. This makes W&B charts naturally use epochs on the x-axis instead of
trainer global step.
"""

from __future__ import annotations

from typing import Any

from pytorch_lightning.loggers import WandbLogger


class EpochWandbLogger(WandbLogger):
    """A WandbLogger that uses epoch as the W&B step.

    PyTorch Lightning passes `trainer.global_step` as the step parameter when
    calling `log_metrics()`. For workflows where metrics are logged only per
    epoch (recommended here), it is more intuitive to have the W&B x-axis be
    epoch number.
    """

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        epoch_value = None
        for key in ("epoch", "trainer/epoch"):
            if key in metrics:
                epoch_value = metrics.get(key)
                break

        if epoch_value is not None:
            try:
                step = int(epoch_value)
            except (TypeError, ValueError):
                # Fall back to the provided step if conversion fails.
                pass

        super().log_metrics(metrics, step=step)
