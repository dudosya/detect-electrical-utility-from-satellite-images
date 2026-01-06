"""PyTorch Lightning module for tower detection training."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchmetrics.detection import MeanAveragePrecision

from detect_electrical_utility_from_satellite_images.config import Config
from detect_electrical_utility_from_satellite_images.datasets.tower_dataset import (
    TowerDetectionDataset,
    collate_fn,
)
from detect_electrical_utility_from_satellite_images.logging_config import get_logger
from detect_electrical_utility_from_satellite_images.models.tower_detector import (
    create_tower_detector,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TowerDetectorModule(pl.LightningModule):
    """PyTorch Lightning module for Faster R-CNN tower detection."""

    def __init__(
        self,
        config: Config,
        num_classes: int = 2,
    ) -> None:
        """Initialize the module.

        Args:
            config: Configuration instance.
            num_classes: Number of classes (background + tower).
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["config"])

        self.log_fn = get_logger()

        # Create model
        self.model = create_tower_detector(
            num_classes=num_classes,
            anchor_areas=config.tower_detection.anchor_areas,
            anchor_ratios=config.tower_detection.anchor_ratios,
            box_score_thresh=config.tower_detection.confidence_threshold,
            box_nms_thresh=config.tower_detection.nms_threshold,
            min_size=config.preprocessing.patch_size,
            max_size=config.preprocessing.patch_size,
        )

        # Detection metrics
        self.val_map = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
        
        # Store validation predictions for visualization
        self.val_predictions: list[dict[str, torch.Tensor]] = []
        self.val_targets: list[dict[str, torch.Tensor]] = []
        self.val_images: list[torch.Tensor] = []

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> Any:
        """Forward pass.

        Args:
            images: List of (3, H, W) tensors.
            targets: Optional list of target dicts for training.

        Returns:
            Loss dict during training, detections during inference.
        """
        return self.model(images, targets)

    def training_step(
        self,
        batch: tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (images, targets).
            batch_idx: Batch index.

        Returns:
            Total loss.
        """
        images, targets = batch
        batch_size = len(images)

        # Faster R-CNN returns losses when targets are provided
        loss_dict = self.model(images, targets)

        # Sum all losses
        total_loss = sum(loss for loss in loss_dict.values())

        # Log individual losses (on_step=False so x-axis is epochs)
        for name, loss in loss_dict.items():
            self.log(f"train/{name}", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

        self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Log learning rate
        opt = self.optimizers()
        if hasattr(opt, 'param_groups'):
            self.log("train/lr", opt.param_groups[0]["lr"], on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(
        self,
        batch: tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]],
        batch_idx: int,
    ) -> None:
        """Validation step.

        Args:
            batch: Tuple of (images, targets).
            batch_idx: Batch index.
        """
        images, targets = batch
        batch_size = len(images)

        # Get losses (need train mode for Faster R-CNN to return losses)
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)

        total_loss = sum(loss for loss in loss_dict.values())

        for name, loss in loss_dict.items():
            self.log(f"val/{name}", loss, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)

        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        # Get predictions (need eval mode)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)

        # Update mAP metric
        preds_formatted = [
            {"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]}
            for p in predictions
        ]
        targets_formatted = [
            {"boxes": t["boxes"], "labels": t["labels"]}
            for t in targets
        ]
        self.val_map.update(preds_formatted, targets_formatted)

        # Store for visualization (limit to prevent memory issues)
        if len(self.val_images) < 16:
            for img, pred, tgt in zip(images, predictions, targets, strict=False):
                if len(self.val_images) >= 16:
                    break
                self.val_images.append(img.cpu())
                self.val_predictions.append({k: v.cpu() for k, v in pred.items()})
                self.val_targets.append({k: v.cpu() for k, v in tgt.items()})

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at end of epoch."""
        # Compute mAP
        map_results = self.val_map.compute()
        
        self.log("val/mAP", map_results["map"], prog_bar=True, sync_dist=True)
        self.log("val/mAP_50", map_results["map_50"], prog_bar=True, sync_dist=True)
        self.log("val/mAP_75", map_results["map_75"], sync_dist=True)
        
        if "mar_100" in map_results:
            self.log("val/recall", map_results["mar_100"], sync_dist=True)

        # Log to wandb with visualizations
        if self.logger and hasattr(self.logger, "experiment"):
            self._log_visualizations()

        # Reset for next epoch
        self.val_map.reset()
        self.val_predictions.clear()
        self.val_targets.clear()
        self.val_images.clear()

    def _log_visualizations(self) -> None:
        """Log visualization images to wandb."""
        try:
            import wandb
            from detect_electrical_utility_from_satellite_images.evaluation import (
                create_detection_grid,
            )

            if len(self.val_images) > 0:
                fig = create_detection_grid(
                    self.val_images,
                    self.val_predictions,
                    self.val_targets,
                    max_images=9,
                    score_threshold=self.config.tower_detection.confidence_threshold,
                )
                self.logger.experiment.log({
                    "val_detections": wandb.Image(fig),
                    "epoch": self.current_epoch,
                })
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception as e:
            self.log_fn.warning("visualization_failed", error=str(e))

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Dict with optimizer and scheduler config.
        """
        # Paper: SGD with lr=3e-3, decay by 0.1 every 10k steps
        optimizer = SGD(
            self.parameters(),
            lr=self.config.training.initial_lr,
            momentum=0.9,
            weight_decay=0.0005,
        )

        # Step scheduler: decay by factor every N steps
        scheduler = StepLR(
            optimizer,
            step_size=self.config.training.lr_decay_steps,
            gamma=self.config.training.lr_decay_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class TowerDetectionDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for tower detection."""

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

        self.train_dataset: ConcatDataset[Any] | TowerDetectionDataset | None = None
        self.val_dataset: ConcatDataset[Any] | TowerDetectionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ("fit", "validate") or stage is None:
            # Create datasets for all regions and combine
            datasets = []
            for patches_dir in self.patches_dirs:
                ds = TowerDetectionDataset(
                    patches_dir=patches_dir,
                    include_background=False,  # Only patches with towers
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
                "data_split",
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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
            pin_memory=True,
        )
