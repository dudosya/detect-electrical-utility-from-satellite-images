"""
Model evaluation script using Typer CLI.
Evaluates trained segmentation models on validation/test data.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import typer
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from detect_electrical_utility_from_satellite_images.config import AppConfig
from detect_electrical_utility_from_satellite_images.dataset import SatteliteImgsDataset
from detect_electrical_utility_from_satellite_images.metrics import (
    calculate_all_metrics,
)
from detect_electrical_utility_from_satellite_images.model import (
    create_model,
    load_checkpoint,
)
from detect_electrical_utility_from_satellite_images.utils.logging_config import (
    setup_logger,
)

app = typer.Typer(help="Evaluate trained segmentation models")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> AppConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    try:
        cfg = AppConfig(**cfg_dict)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

    return cfg


def load_model(checkpoint_path: Path, cfg: AppConfig) -> tuple[nn.Module, str]:
    """Load model from checkpoint."""
    device = cfg.model.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    model = create_model(
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
        device=device,
    )

    epoch, loss = load_checkpoint(model, None, checkpoint_path)
    logger.info(f"Loaded model from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")

    model.eval()
    return model, device


def create_dataloader(cfg: AppConfig, split: str = "val") -> DataLoader:
    """Create dataloader for evaluation."""
    # Load image and mask paths
    img_patch_paths = sorted((cfg.paths.output_dir / "img_patches").glob("*.png"))
    mask_patch_paths = sorted((cfg.paths.output_dir / "mask_patches").glob("*.png"))

    if len(img_patch_paths) == 0:
        raise ValueError(
            f"No image patches found in {cfg.paths.output_dir / 'img_patches'}"
        )

    logger.info(f"Found {len(img_patch_paths)} image patches")

    # Create dataset
    from torchvision.transforms import v2

    transforms = v2.Compose(
        [
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]
    )

    dataset = SatteliteImgsDataset(
        img_paths=img_patch_paths,
        mask_paths=mask_patch_paths,
        transforms=transforms,
    )

    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if split == "val":
        # Use validation split (last 20%)
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.model.batch_size,
            shuffle=False,
            num_workers=0,
        )
        logger.info(f"Created validation dataloader with {len(val_dataset)} samples")
    else:
        # Use all data
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.model.batch_size,
            shuffle=False,
            num_workers=0,
        )
        logger.info(f"Created dataloader with {len(dataset)} samples")

    return dataloader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    cfg: AppConfig,
    num_samples: int = 5,
) -> dict[str, Any]:
    """Evaluate model and return metrics."""
    model.eval()

    all_predictions = []
    all_targets = []
    sample_images = []
    sample_masks = []
    sample_preds = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Store for metrics
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())

            # Store samples for visualization
            if batch_idx == 0 and num_samples > 0:
                n = min(num_samples, images.shape[0])
                sample_images.extend(images.cpu()[:n])
                sample_masks.extend(masks.cpu()[:n])
                sample_preds.extend(predictions.cpu()[:n])

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

    # Calculate confusion matrix
    cm = confusion_matrix(
        all_targets.flatten().numpy(),
        all_predictions.flatten().numpy(),
        labels=list(range(cfg.model.classes)),
    )

    results = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(all_targets),
        "sample_data": {
            "images": [img.numpy().tolist() for img in sample_images],
            "masks": [mask.numpy().tolist() for mask in sample_masks],
            "predictions": [pred.numpy().tolist() for pred in sample_preds],
        }
        if sample_images
        else {},
    }

    return results


def save_results(results: dict[str, Any], output_dir: Path):
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)

    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.json"
    with open(cm_path, "w") as f:
        json.dump(results["confusion_matrix"], f, indent=2)

    # Create visualization
    create_visualizations(results, output_dir)

    logger.info(f"Results saved to {output_dir}")


def create_visualizations(results: dict[str, Any], output_dir: Path):
    """Create visualizations from evaluation results."""
    # Plot confusion matrix
    cm = np.array(results["confusion_matrix"])
    if cm.size > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

    # Plot sample predictions
    if results["sample_data"]:
        sample_images = results["sample_data"]["images"]
        sample_masks = results["sample_data"]["masks"]
        sample_preds = results["sample_data"]["predictions"]

        n_samples = len(sample_images)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Original image
            img = np.array(sample_images[i])
            img = np.transpose(img, (1, 2, 0))  # CHW to HWC
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Sample {i + 1}: Image")
            axes[i, 0].axis("off")

            # Ground truth mask
            mask = np.array(sample_masks[i])
            axes[i, 1].imshow(mask, cmap="tab20")
            axes[i, 1].set_title(f"Sample {i + 1}: Ground Truth")
            axes[i, 1].axis("off")

            # Prediction
            pred = np.array(sample_preds[i])
            axes[i, 2].imshow(pred, cmap="tab20")
            axes[i, 2].set_title(f"Sample {i + 1}: Prediction")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "sample_predictions.png", dpi=150)
        plt.close()


@app.command()
def evaluate(
    config: Path = typer.Option(
        Path("config.yaml"), help="Path to config file (default: config.yaml)"
    ),
    checkpoint: Path | None = typer.Option(
        None, help="Direct path to model checkpoint (.pt file)"
    ),
    run_id: str | None = typer.Option(None, help="Specific W&B run ID to evaluate"),
    latest_run: int = typer.Option(
        1, help="N-th latest run to evaluate (1=latest, 2=previous, etc.)"
    ),
    model_type: str = typer.Option(
        "best", help="Model type to evaluate: 'best' or 'last'"
    ),
    wandb_dir: Path = typer.Option(
        Path("wandb"), help="Path to W&B runs directory (default: wandb/)"
    ),
    output_dir: Path | None = typer.Option(
        None, help="Directory to save evaluation results"
    ),
    split: str = typer.Option("val", help="Data split to evaluate on: 'val' or 'all'"),
    num_samples: int = typer.Option(5, help="Number of samples to visualize"),
    log_level: str = typer.Option(
        "info", help="Logging level: debug, info, warning, error"
    ),
):
    """
    Evaluate a trained segmentation model.

    Examples:
        # Minimal command (uses all defaults)
        uv run eval-model evaluate

        # Evaluate latest run's best model (same as above)
        uv run eval-model evaluate --latest-run 1 --model-type best

        # Evaluate previous run's last model
        uv run eval-model evaluate --latest-run 2 --model-type last

        # Evaluate specific run by ID
        uv run eval-model evaluate --run-id run-20251228_181143-vqqmk5eb

        # Direct checkpoint path (backward compatible)
        uv run eval-model evaluate --checkpoint wandb/.../best_model.pt
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Validate input
        if checkpoint is None and run_id is None and latest_run < 1:
            raise ValueError(
                "Either checkpoint, run_id, or latest_run must be provided"
            )

        # Load config
        config_path = (
            config.with_suffix(".yaml") if config.suffix != ".yaml" else config
        )
        logger.info(f"Loading config from {config_path}")
        cfg = load_config(config_path)

        # Resolve checkpoint path
        checkpoint_path = None
        if checkpoint is not None:
            checkpoint_path = checkpoint
            logger.info(f"Using direct checkpoint path: {checkpoint_path}")
        else:
            # Find run directory
            wandb_dir = wandb_dir.resolve()
            if not wandb_dir.exists():
                raise ValueError(f"W&B directory not found: {wandb_dir}")

            runs = list(wandb_dir.glob("run-*"))
            if not runs:
                raise ValueError(f"No runs found in {wandb_dir}")

            # Sort by modification time (newest first)
            runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            if run_id:
                # Find specific run
                run_dir = next((r for r in runs if run_id in r.name), None)
                if run_dir is None:
                    raise ValueError(f"Run with ID '{run_id}' not found")
            else:
                # Use N-th latest run
                if latest_run > len(runs):
                    raise ValueError(
                        f"Only {len(runs)} runs available, cannot get run #{latest_run}"
                    )
                run_dir = runs[latest_run - 1]

            logger.info(f"Using run: {run_dir.name}")

            # Construct checkpoint path
            model_file = f"{model_type}_model.pt"
            checkpoint_dirs = [
                run_dir / "files" / "checkpoints" / model_file,
                run_dir / "checkpoints" / model_file,
            ]

            for cp_path in checkpoint_dirs:
                if cp_path.exists():
                    checkpoint_path = cp_path
                    break

            if checkpoint_path is None:
                raise ValueError(f"No {model_type} model found in run {run_dir.name}")

        # Setup output directory
        if output_dir is None:
            if run_id or latest_run:
                run_name = run_dir.name if "run_dir" in locals() else "direct"
                output_dir = Path("evaluation_results") / run_name / model_type
            else:
                output_dir = (
                    Path("evaluation_results")
                    / checkpoint_path.parent.parent.name
                    / checkpoint_path.stem
                )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        model, device = load_model(checkpoint_path, cfg)

        # Create dataloader
        logger.info(f"Creating dataloader for {split} split")
        dataloader = create_dataloader(cfg, split)

        # Evaluate
        logger.info(f"Evaluating model on {len(dataloader.dataset)} samples")
        results = evaluate_model(model, dataloader, device, cfg, num_samples)

        # Save results
        logger.info(f"Saving results to {output_dir}")
        save_results(results, output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {checkpoint_path}")
        print(f"Config: {config_path}")
        print(f"Samples evaluated: {results['num_samples']}")
        print(f"Results saved to: {output_dir}")

        if "metrics" in results:
            print("\nMetrics:")
            for metric_name, metric_value in results["metrics"].items():
                if isinstance(metric_value, dict):
                    print(f"  {metric_name}:")
                    for class_idx, class_value in metric_value.items():
                        print(f"    Class {class_idx}: {class_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value:.4f}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def list_checkpoints(
    wandb_dir: Path = typer.Option(
        Path("wandb"), help="Path to W&B runs directory (default: wandb/)"
    ),
    run_id: str | None = typer.Option(
        None, help="Specific run ID to list checkpoints for"
    ),
):
    """
    List available checkpoints in W&B runs.

    Example:
        uv run evaluate_model list-checkpoints wandb/
    """
    wandb_dir = wandb_dir.resolve()

    if not wandb_dir.exists():
        logger.error(f"W&B directory not found: {wandb_dir}")
        raise typer.Exit(code=1)

    runs = []
    if run_id:
        # Look for specific run
        run_pattern = f"*{run_id}*"
        run_dirs = list(wandb_dir.glob(run_pattern))
        if not run_dirs:
            logger.error(f"No run found with ID containing '{run_id}'")
            raise typer.Exit(code=1)
        runs = run_dirs
    else:
        # List all runs
        runs = list(wandb_dir.glob("run-*"))

    if not runs:
        logger.error(f"No runs found in {wandb_dir}")
        raise typer.Exit(code=1)

    print(f"Found {len(runs)} runs in {wandb_dir}")
    print("=" * 80)

    for run_dir in sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True):
        # Check both possible locations: files/checkpoints and checkpoints
        checkpoint_dirs = [
            run_dir / "files" / "checkpoints",
            run_dir / "checkpoints",
        ]

        found_checkpoints = []
        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pt"))
                found_checkpoints.extend(checkpoints)

        if found_checkpoints:
            print(f"\nRun: {run_dir.name}")
            print(f"  Path: {run_dir}")
            print("  Checkpoints:")
            for cp in sorted(found_checkpoints):
                cp_time = cp.stat().st_mtime
                from datetime import datetime

                time_str = datetime.fromtimestamp(cp_time).strftime("%Y-%m-%d %H:%M:%S")
                print(f"    • {cp.relative_to(run_dir)} ({time_str})")
        else:
            print(f"\nRun: {run_dir.name} (no checkpoints)")


if __name__ == "__main__":
    app()
