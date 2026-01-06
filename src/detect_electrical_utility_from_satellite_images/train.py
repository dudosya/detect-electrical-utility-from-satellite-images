import logging
import random
import time
import typing
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

import wandb

from .config import AppConfig
from .dataset import SatteliteImgsDataset
from .losses import calculate_class_weights, create_loss_function


# ImageNet statistics for normalization (used with pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_train_transforms(cfg: AppConfig) -> torchvision.transforms.v2.Compose:
    """Create training transforms with augmentation based on config.

    Args:
        cfg: Application configuration containing augmentation settings.

    Returns:
        Composed transforms for training with augmentation, including:
        - Geometric transforms (flips, rotation) if enabled
        - Photometric transforms (color jitter, blur) if enabled
        - ImageNet normalization for pretrained models
        - Type conversions for PyTorch
    """
    from torchvision.transforms import v2, InterpolationMode

    transforms_list = []

    if cfg.augmentation.enabled:
        # Geometric augmentations (applied to both image and mask)
        if cfg.augmentation.horizontal_flip > 0:
            transforms_list.append(v2.RandomHorizontalFlip(p=cfg.augmentation.horizontal_flip))
        if cfg.augmentation.vertical_flip > 0:
            transforms_list.append(v2.RandomVerticalFlip(p=cfg.augmentation.vertical_flip))
        if cfg.augmentation.rotation > 0:
            transforms_list.append(
                v2.RandomRotation(
                    degrees=cfg.augmentation.rotation,
                    interpolation=InterpolationMode.BILINEAR,
                )
            )

        # Photometric augmentations (applied only to image)
        if any([cfg.augmentation.brightness, cfg.augmentation.contrast, cfg.augmentation.saturation]):
            transforms_list.append(
                v2.ColorJitter(
                    brightness=cfg.augmentation.brightness,
                    contrast=cfg.augmentation.contrast,
                    saturation=cfg.augmentation.saturation,
                )
            )
        if cfg.augmentation.gaussian_blur:
            transforms_list.append(v2.GaussianBlur(kernel_size=cfg.augmentation.blur_kernel_size))

    # Convert to float32 and scale to [0, 1]
    transforms_list.append(v2.ToDtype(dtype=torch.float32, scale=True))

    # Apply ImageNet normalization (for pretrained models)
    if cfg.model.encoder_weights == "imagenet":
        transforms_list.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    # Convert mask to long dtype (must be last)
    transforms_list.append(
        v2.Lambda(
            lambda x: x.to(torch.long)
            if hasattr(x, "__class__") and x.__class__.__name__ == "Mask"
            else x,
        )
    )

    return v2.Compose(transforms_list)


def create_val_transforms(cfg: AppConfig) -> torchvision.transforms.v2.Compose:
    """Create validation transforms without augmentation.

    Args:
        cfg: Application configuration.

    Returns:
        Composed transforms for validation (no augmentation).
    """
    from torchvision.transforms import v2

    transforms_list = [
        v2.ToDtype(dtype=torch.float32, scale=True),
    ]

    # Apply ImageNet normalization (for pretrained models)
    if cfg.model.encoder_weights == "imagenet":
        transforms_list.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    # Convert mask to long dtype
    transforms_list.append(
        v2.Lambda(
            lambda x: x.to(torch.long)
            if hasattr(x, "__class__") and x.__class__.__name__ == "Mask"
            else x,
        )
    )

    return v2.Compose(transforms_list)
from .metrics import calculate_all_metrics
from .model import count_parameters, create_model, save_checkpoint
from .utils import split_by_image, validate_split_by_image

logger = logging.getLogger(__name__)


def create_mock_dataloader(
    batch_size: int,
    num_samples: int = 100,
    image_size: tuple[int, int] = (1024, 1024),
    num_classes: int = 5,
    device: str = "cpu",
) -> DataLoader:
    """Create a mock dataloader for testing training pipeline.

    Args:
        batch_size: Batch size
        num_samples: Number of mock samples to generate
        image_size: (height, width) of mock images
        num_classes: Number of classes for segmentation masks
        device: Device to place tensors on

    Returns:
        DataLoader yielding (images, masks) as torch tensors
    """

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, image_size, num_classes, device):
            self.num_samples = num_samples
            self.image_size = image_size
            self.num_classes = num_classes
            self.device = device

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random image (3 channels, normalized to [0, 1])
            image = torch.rand(
                3, self.image_size[0], self.image_size[1], device=self.device,
            )
            # Generate random mask with class indices
            mask = torch.randint(
                0,
                self.num_classes,
                (1, self.image_size[0], self.image_size[1]),
                device=self.device,
            )
            return image, mask.squeeze(0)  # Remove channel dimension for mask

    dataset = MockDataset(num_samples, image_size, num_classes, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    cfg,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, dict[str, float | dict[int, float]]]:
    """Train for one epoch with metrics tracking.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        cfg: Configuration object
        gradient_accumulation_steps: Steps for gradient accumulation

    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Initialize metrics accumulators
    all_predictions = []
    all_targets = []

    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        # Ensure masks have correct shape (batch_size, height, width)
        if masks.ndim == 4:  # If masks have channel dimension
            masks = masks.squeeze(1)  # Remove channel dimension

        outputs = model(images)
        loss = criterion(outputs, masks)

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Store predictions and targets for metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())

        # Update weights only after accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if batch_idx % 10 == 0:
            logger.debug(
                f"Epoch {epoch}, Batch {batch_idx}: loss = {loss.item() * gradient_accumulation_steps:.4f}",
            )

    # Handle remaining gradients if any
    if len(dataloader) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Calculate metrics
    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = calculate_all_metrics(
            all_predictions,
            all_targets,
            cfg.model.classes,
            cfg.metrics.dict(),
        )
    else:
        metrics = {}

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch {epoch} training complete: avg loss = {avg_loss:.4f}")

    # Log per-class IoU to console for easy monitoring
    if "iou_per_class" in metrics:
        iou_strs = [f"c{k}={v:.3f}" for k, v in metrics["iou_per_class"].items()]
        logger.info(f"  Train IoU per class: {', '.join(iou_strs)}")
    if "iou" in metrics:
        logger.info(f"  Train mIoU: {metrics['iou']:.4f}")

    # Log to W&B
    log_data = {"train/loss": avg_loss}
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, dict):
            for class_idx, class_value in metric_value.items():
                log_data[f"train/{metric_name}_class_{class_idx}"] = class_value
        else:
            log_data[f"train/{metric_name}"] = metric_value

    wandb.log(log_data, step=epoch)

    return avg_loss, metrics  # type: ignore


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    cfg,
) -> tuple[float, dict[str, float | dict[int, float]]]:
    """Validate model with metrics tracking.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        cfg: Configuration object

    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Initialize metrics accumulators
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Ensure masks have correct shape (batch_size, height, width)
            if masks.ndim == 4:  # If masks have channel dimension
                masks = masks.squeeze(1)  # Remove channel dimension

            outputs = model(images)
            loss = criterion(outputs, masks)

            # Store predictions and targets for metrics
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())

            total_loss += loss.item()
            num_batches += 1

    # Calculate metrics
    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = calculate_all_metrics(
            all_predictions,
            all_targets,
            cfg.model.classes,
            cfg.metrics.dict(),
        )
    else:
        metrics = {}

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Validation complete: avg loss = {avg_loss:.4f}")

    # Log per-class IoU to console for easy monitoring
    if "iou_per_class" in metrics:
        iou_strs = [f"c{k}={v:.3f}" for k, v in metrics["iou_per_class"].items()]
        logger.info(f"  Val IoU per class: {', '.join(iou_strs)}")
    if "iou" in metrics:
        logger.info(f"  Val mIoU: {metrics['iou']:.4f}")

    return avg_loss, metrics


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(cfg: AppConfig, use_mock_data: bool = True) -> None:
    """Main training function with W&B integration and mixed precision support.

    Args:
        cfg: Application configuration
        use_mock_data: If True, use mock data; if False, use real dataset
    """
    start_time = time.perf_counter()

    # Set random seed for reproducibility
    set_seed(cfg.training.seed)

    # Setup device
    device = cfg.model.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Initialize mixed precision if enabled and device is CUDA
    scaler = None
    if cfg.training.mixed_precision and device == "cuda":
        try:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision (AMP) enabled")
        except Exception as e:
            logger.warning(f"Failed to enable mixed precision: {e}")
            scaler = None

    # Initialize Weights & Biases
    wandb_config = cfg.wandb.dict()
    wandb_config["run_name"] = wandb_config["run_name"].replace(
        "{timestamp}",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        name=wandb_config["run_name"],
        tags=wandb_config["tags"],
        notes=wandb_config["notes"],
        group=wandb_config["group"],
        config=cfg.dict(),
        save_code=wandb_config["save_code"],
    )

    logger.info(f"W&B run initialized: {run.name}")

    # Create model
    model = create_model(
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
        device=device,
    )

    logger.info(f"Model has {count_parameters(model):,} trainable parameters")

    # Setup loss and optimizer
    # Class weights will be calculated after dataset split (if auto_class_weights is enabled)
    class_weights = None
    # Use manual class weights if provided
    if cfg.training.class_weights is not None:
        class_weights = torch.tensor(cfg.training.class_weights, device=device)
        logger.info(f"Using manual class weights: {class_weights.tolist()}")

    # Create loss function
    criterion = create_loss_function(
        loss_name=cfg.training.loss_function,
        num_classes=cfg.model.classes,
        device=device,
        class_weights=class_weights.tolist() if class_weights is not None else None,
        loss_alpha=cfg.training.loss_alpha,
        loss_gamma=cfg.training.loss_gamma,
    )
    logger.info(f"Using loss function: {cfg.training.loss_function}")

    # Setup optimizer
    if cfg.training.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.model.learning_rate,
            **{
                k: v
                for k, v in cfg.training.optimizer_params.items()
                if k in ["betas", "weight_decay"]
            },
        )
    elif cfg.training.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.model.learning_rate,
            **{
                k: v
                for k, v in cfg.training.optimizer_params.items()
                if k in ["momentum", "nesterov", "weight_decay"]
            },
        )
    elif cfg.training.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.model.learning_rate,
            **{
                k: v
                for k, v in cfg.training.optimizer_params.items()
                if k in ["betas", "weight_decay"]
            },
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

    # Setup learning rate scheduler
    scheduler = None
    if cfg.training.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.lr_scheduler_params.get("T_max", cfg.model.epochs),
            eta_min=cfg.training.lr_scheduler_params.get("eta_min", 1e-6),
        )
    elif cfg.training.lr_scheduler == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.training.lr_scheduler_params.get("factor", 0.5),
            patience=cfg.training.lr_scheduler_params.get("patience", 5),
            min_lr=cfg.training.lr_scheduler_params.get("min_lr", 1e-6),
        )
    elif cfg.training.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.lr_scheduler_params.get("step_size", 10),
            gamma=cfg.training.lr_scheduler_params.get("gamma", 0.1),
        )

    # Create dataloaders
    if use_mock_data:
        logger.info("Using mock data for training")
        train_loader = create_mock_dataloader(
            batch_size=cfg.model.batch_size,
            num_samples=100,
            image_size=(cfg.preprocessing.patch_size, cfg.preprocessing.patch_size),
            num_classes=cfg.model.classes,
            device=device,
        )
        val_loader = create_mock_dataloader(
            batch_size=cfg.model.batch_size,
            num_samples=20,
            image_size=(cfg.preprocessing.patch_size, cfg.preprocessing.patch_size),
            num_classes=cfg.model.classes,
            device=device,
        )
    else:
        logger.info("Using real data for training")
        # Load real dataset
        img_patch_paths = sorted((cfg.paths.output_dir / "img_patches").glob("*.png"))
        mask_patch_paths = sorted((cfg.paths.output_dir / "mask_patches").glob("*.png"))

        if len(img_patch_paths) == 0:
            raise ValueError(
                f"No image patches found in {cfg.paths.output_dir / 'img_patches'}",
            )
        if len(mask_patch_paths) == 0:
            raise ValueError(
                f"No mask patches found in {cfg.paths.output_dir / 'mask_patches'}",
            )

        logger.info(
            f"Found {len(img_patch_paths)} image patches and {len(mask_patch_paths)} mask patches",
        )

        # Split patches by original image to prevent data leakage
        train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = split_by_image(
            img_patch_paths,
            mask_patch_paths,
            train_ratio=0.8,
            seed=cfg.training.seed,
        )

        # Validate the split
        if not validate_split_by_image(train_img_paths, val_img_paths):
            logger.warning("Data leakage detected in split! Proceeding anyway...")

        logger.info(
            f"Split patches: {len(train_img_paths)} train, {len(val_img_paths)} validation"
        )

        # Create transforms using config-driven helper functions
        train_transforms = create_train_transforms(cfg)
        val_transforms = create_val_transforms(cfg)

        # Create datasets
        train_dataset = SatteliteImgsDataset(
            img_paths=train_img_paths,
            mask_paths=train_mask_paths,
            transforms=train_transforms,
        )
        val_dataset = SatteliteImgsDataset(
            img_paths=val_img_paths,
            mask_paths=val_mask_paths,
            transforms=val_transforms,
        )

        # Calculate class weights from training data if auto_class_weights is enabled
        if cfg.training.auto_class_weights and class_weights is None:
            try:
                # Create a simple dataset without augmentation for weight calculation
                from torchvision.transforms import v2
                weight_transforms = v2.Compose([
                    v2.ToDtype(dtype=torch.float32, scale=True),
                ])
                weight_dataset = SatteliteImgsDataset(
                    img_paths=train_img_paths,
                    mask_paths=train_mask_paths,
                    transforms=weight_transforms,
                )
                calculated_weights = calculate_class_weights(
                    weight_dataset, cfg.model.classes, device,
                )
                class_weights = calculated_weights
                logger.info(f"Auto-calculated class weights from training data: {class_weights.tolist()}")
                
                # Recreate loss function with new class weights
                criterion = create_loss_function(
                    loss_name=cfg.training.loss_function,
                    num_classes=cfg.model.classes,
                    device=device,
                    class_weights=class_weights.tolist() if class_weights is not None else None,
                    loss_alpha=cfg.training.loss_alpha,
                    loss_gamma=cfg.training.loss_gamma,
                )
                logger.info(f"Recreated loss function with auto-calculated class weights")
            except Exception as e:
                logger.warning(f"Failed to auto-calculate class weights: {e}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.model.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.model.batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches",
        )

    # Training loop
    best_loss = float("inf")
    last_save_epoch = 0

    # Create checkpoint directory in W&B run folder
    checkpoint_dir = Path(run.dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    for epoch in range(1, cfg.model.epochs + 1):
        logger.info(f"Starting epoch {epoch}/{cfg.model.epochs}")

        # Train
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            cfg,
            cfg.training.gradient_accumulation_steps,
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, cfg)

        # Update learning rate scheduler
        if scheduler is not None:
            if cfg.training.lr_scheduler == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log validation metrics
        log_data = {
            "val/loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for metric_name, metric_value in val_metrics.items():
            if isinstance(metric_value, dict):
                for class_idx, class_value in metric_value.items():
                    log_data[f"val/{metric_name}_class_{class_idx}"] = class_value
            else:
                log_data[f"val/{metric_name}"] = metric_value
        
        wandb.log(log_data, step=epoch)

        # Check if we should save best model
        should_save_best = False

        if cfg.checkpoints.save_best:
            if cfg.checkpoints.mode == "min":
                # For minimization (e.g., loss)
                improvement = best_loss - val_loss
                if improvement > cfg.checkpoints.min_delta:
                    best_loss = val_loss
                    should_save_best = True
                    logger.info(
                        f"Significant improvement: {improvement:.6f} > {cfg.checkpoints.min_delta}",
                    )
            else:
                # For maximization (e.g., accuracy)
                improvement = (
                    val_loss - best_loss
                )  # Note: val_loss is actually the metric being monitored
                if improvement > cfg.checkpoints.min_delta:
                    best_loss = val_loss
                    should_save_best = True
                    logger.info(
                        f"Significant improvement: {improvement:.6f} > {cfg.checkpoints.min_delta}",
                    )

        # Check if we should save based on frequency
        should_save_frequency = False
        if cfg.checkpoints.save_every_n_epochs is not None:
            if epoch - last_save_epoch >= cfg.checkpoints.save_every_n_epochs:
                should_save_frequency = True
                logger.info(
                    f"Frequency-based save triggered (every {cfg.checkpoints.save_every_n_epochs} epochs)",
                )

        # Save best model if conditions met
        if should_save_best or should_save_frequency:
            checkpoint_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            last_save_epoch = epoch
            logger.info(f"Model saved to: {checkpoint_path} (loss: {val_loss:.4f})")

    # Save last model at the end of training
    if cfg.checkpoints.save_last:
        last_path = checkpoint_dir / "last_model.pt"
        save_checkpoint(model, optimizer, cfg.model.epochs, val_loss, last_path)
        logger.info(f"Last model saved to: {last_path}")

    # Finish W&B run
    wandb.finish()

    end_time = time.perf_counter()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_loss:.4f}")
