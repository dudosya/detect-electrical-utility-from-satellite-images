"""PyTorch Lightning module for line segmentation training (Stage 2)."""

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchmetrics import JaccardIndex

from detect_electrical_utility_from_satellite_images.config import Config
from detect_electrical_utility_from_satellite_images.datasets.line_dataset import (
    LineSegmentationDataset,
    line_collate_fn,
)
from detect_electrical_utility_from_satellite_images.logging_config import get_logger
from detect_electrical_utility_from_satellite_images.models.line_segmentor import (
    create_line_segmentor,
)


class DiceLoss(torch.nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        """Initialize Dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            predictions: Predicted probabilities (B, 1, H, W).
            targets: Ground truth masks (B, 1, H, W).

        Returns:
            Dice loss value.
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class CombinedLoss(torch.nn.Module):
    """Combined BCE + Dice loss for better segmentation."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        """Initialize combined loss.

        Args:
            bce_weight: Weight for BCE loss.
            dice_weight: Weight for Dice loss.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Args:
            logits: Raw model output (B, 1, H, W).
            targets: Ground truth masks (B, 1, H, W).

        Returns:
            Tuple of (total_loss, bce_loss, dice_loss).
        """
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, targets)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total, bce_loss, dice_loss


class LineSegmentorModule(pl.LightningModule):
    """PyTorch Lightning module for U-Net line segmentation."""

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the module.

        Args:
            config: Configuration instance.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["config"])

        self.log_fn = get_logger()

        # Create model
        self.model = create_line_segmentor(
            in_channels=3,
            base_features=64,
            bilinear=True,
        )

        # Loss function (BCE + Dice)
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

        # Metrics
        self.train_iou = JaccardIndex(task="binary")
        self.val_iou = JaccardIndex(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Logits tensor (B, 1, H, W).
        """
        return self.model(x)

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dictionary with 'images' and 'masks'.
            batch_idx: Batch index.

        Returns:
            Total loss.
        """
        images = batch["images"]
        masks = batch["masks"]
        batch_size = images.size(0)

        # Forward pass
        logits = self.model(images)

        # Compute loss
        total_loss, bce_loss, dice_loss = self.criterion(logits, masks)

        # Compute IoU for logging
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        self.train_iou.update(preds, masks.long())

        # Log losses (on_step=False so x-axis is epochs)
        self.log(
            "train/loss", total_loss,
            on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "train/bce_loss", bce_loss,
            on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            "train/dice_loss", dice_loss,
            on_step=False, on_epoch=True, batch_size=batch_size
        )

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        iou = self.train_iou.compute()
        self.log("train/IoU", iou, prog_bar=True)
        self.train_iou.reset()

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Validation step.

        Args:
            batch: Dictionary with 'images' and 'masks'.
            batch_idx: Batch index.
        """
        images = batch["images"]
        masks = batch["masks"]
        batch_size = images.size(0)

        # Forward pass
        logits = self.model(images)

        # Compute loss
        total_loss, bce_loss, dice_loss = self.criterion(logits, masks)

        # Compute IoU
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        self.val_iou.update(preds, masks.long())

        # Log losses
        self.log(
            "val/loss", total_loss,
            on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "val/bce_loss", bce_loss,
            on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val/dice_loss", dice_loss,
            on_step=False, on_epoch=True, batch_size=batch_size
        )

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        iou = self.val_iou.compute()
        self.log("val/IoU", iou, prog_bar=True)
        self.val_iou.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Uses AdamW with cosine annealing (modern best practice).
        Paper uses SGD with step decay, but AdamW often works better for U-Net.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.initial_lr,
            weight_decay=1e-4,
        )

        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class LineSegmentationDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for line segmentation."""

    def __init__(
        self,
        config: Config,
        patches_dirs: list[Path] | Path | str,
        val_split: float = 0.1,
        num_workers: int = 4,
    ) -> None:
        """Initialize the data module.

        Args:
            config: Configuration instance.
            patches_dirs: Directory or list of directories containing preprocessed patches.
            val_split: Fraction of data to use for validation.
            num_workers: Number of data loading workers.
        """
        super().__init__()
        self.config = config
        # Normalize to list of Paths
        if isinstance(patches_dirs, (str, Path)):
            self.patches_dirs = [Path(patches_dirs)]
        else:
            self.patches_dirs = [Path(d) for d in patches_dirs]
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = config.training.batch_size
        self.line_width = config.line_segmentation.line_width_train

        self.train_dataset: Any = None
        self.val_dataset: Any = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ("fit", "validate") or stage is None:
            # Create datasets for all regions and combine
            datasets = []
            for patches_dir in self.patches_dirs:
                ds = LineSegmentationDataset(
                    patches_dir=patches_dir,
                    line_width=self.line_width,
                    include_empty=False,  # Only patches with lines
                )
                datasets.append(ds)

            # Combine all region datasets
            if len(datasets) == 1:
                full_dataset = datasets[0]
            else:
                full_dataset = ConcatDataset(datasets)

            # Split into train/val
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.config.training.seed),
            )

            get_logger().info(
                "line_data_split",
                num_regions=len(self.patches_dirs),
                train_size=train_size,
                val_size=val_size,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create training dataloader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=line_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create validation dataloader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=line_collate_fn,
            pin_memory=True,
        )
