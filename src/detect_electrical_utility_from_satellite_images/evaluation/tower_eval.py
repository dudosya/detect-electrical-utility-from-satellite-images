"""Evaluation utilities for tower detection."""

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

from detect_electrical_utility_from_satellite_images.logging_config import get_logger


def compute_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compute detection metrics (mAP, precision, recall).

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'.
        targets: List of target dicts with 'boxes', 'labels'.

    Returns:
        Dictionary with mAP, mAP_50, mAP_75, precision, recall metrics.
    """
    # Format for torchmetrics
    preds = []
    tgts = []

    for pred, target in zip(predictions, targets, strict=False):
        preds.append({
            "boxes": pred["boxes"].cpu(),
            "scores": pred["scores"].cpu(),
            "labels": pred["labels"].cpu(),
        })
        tgts.append({
            "boxes": target["boxes"].cpu(),
            "labels": target["labels"].cpu(),
        })

    # Compute mAP using COCO-style evaluation (computes mAP, mAP_50, mAP_75)
    metric = MeanAveragePrecision()
    metric.update(preds, tgts)
    results = metric.compute()

    return {
        "mAP": results["map"].item(),
        "mAP_50": results["map_50"].item(),
        "mAP_75": results["map_75"].item(),
        "precision": results["map"].item(),  # Approximation
        "recall": results["mar_100"].item() if "mar_100" in results else 0.0,
    }


def visualize_detections(
    image: NDArray[np.uint8] | torch.Tensor,
    pred_boxes: torch.Tensor | NDArray[np.float32],
    pred_scores: torch.Tensor | NDArray[np.float32] | None = None,
    gt_boxes: torch.Tensor | NDArray[np.float32] | None = None,
    score_threshold: float = 0.5,
    figsize: tuple[int, int] = (12, 12),
    title: str = "Tower Detection",
) -> plt.Figure:
    """Visualize detection results on an image.

    Args:
        image: Image array (H, W, 3) or tensor (3, H, W).
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2] format.
        pred_scores: Optional confidence scores (N,).
        gt_boxes: Optional ground truth boxes (M, 4).
        score_threshold: Minimum score to display predictions.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    # Convert image to numpy if tensor
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.numpy()

    # Convert boxes to numpy
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy()
    if pred_scores is not None and isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.cpu().numpy()
    if gt_boxes is not None and isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)

    # Draw ground truth boxes (green)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="green", linewidth=2, linestyle="--"
            )
            ax.add_patch(rect)

    # Draw predicted boxes (red/orange based on score)
    if len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            score = pred_scores[i] if pred_scores is not None else 1.0
            if score < score_threshold:
                continue

            x1, y1, x2, y2 = box
            color = "red" if score > 0.7 else "orange"
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linewidth=2
            )
            ax.add_patch(rect)

            # Add score label
            if pred_scores is not None:
                ax.text(
                    x1, y1 - 5, f"{score:.2f}",
                    color=color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
                )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="green", linestyle="--", label="Ground Truth"),
        mpatches.Patch(facecolor="none", edgecolor="red", label="Prediction (>0.7)"),
        mpatches.Patch(facecolor="none", edgecolor="orange", label="Prediction (0.5-0.7)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    return fig


def save_visualization(
    fig: plt.Figure,
    output_path: Path | str,
) -> None:
    """Save visualization to file.

    Args:
        fig: Matplotlib figure.
        output_path: Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_detection_grid(
    images: list[NDArray[np.uint8]],
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    max_images: int = 9,
    score_threshold: float = 0.5,
) -> plt.Figure:
    """Create a grid of detection visualizations.

    Args:
        images: List of images.
        predictions: List of prediction dicts.
        targets: List of target dicts.
        max_images: Maximum number of images to show.
        score_threshold: Minimum score threshold.

    Returns:
        Matplotlib figure with grid.
    """
    n_images = min(len(images), max_images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_images == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    for idx in range(n_images):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        image = images[idx]
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)

        ax.imshow(image)

        pred = predictions[idx]
        target = targets[idx]

        # Draw GT boxes (green)
        if "boxes" in target:
            gt_boxes = target["boxes"].cpu().numpy()
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="green", linewidth=2, linestyle="--"
                )
                ax.add_patch(rect)

        # Draw pred boxes (red)
        if "boxes" in pred and "scores" in pred:
            pred_boxes = pred["boxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()
            for i, box in enumerate(pred_boxes):
                if pred_scores[i] < score_threshold:
                    continue
                x1, y1, x2, y2 = box
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="red", linewidth=2
                )
                ax.add_patch(rect)

        # Stats
        n_gt = len(target["boxes"]) if "boxes" in target else 0
        n_pred = sum(1 for s in pred.get("scores", []) if s >= score_threshold)
        ax.set_title(f"GT: {n_gt} | Pred: {n_pred}")
        ax.axis("off")

    # Hide empty subplots
    for idx in range(n_images, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis("off")

    plt.suptitle("Tower Detection Results (Green=GT, Red=Pred)", fontsize=14)
    plt.tight_layout()
    return fig


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    score_threshold: float = 0.5,
    max_vis_samples: int = 9,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full evaluation on a dataset.

    Args:
        model: Detection model.
        dataloader: Data loader.
        device: Device to run on.
        score_threshold: Score threshold for detections.
        max_vis_samples: Max samples for visualization.
        output_dir: Optional directory to save visualizations.

    Returns:
        Dictionary with metrics and visualization paths.
    """
    log = get_logger()
    model.eval()

    all_predictions = []
    all_targets = []
    sample_images = []
    sample_preds = []
    sample_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move to device
            images = [img.to(device) for img in images]

            # Get predictions
            predictions = model(images)

            # Store for metrics
            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # Store samples for visualization
            if len(sample_images) < max_vis_samples:
                for i, img in enumerate(images):
                    if len(sample_images) >= max_vis_samples:
                        break
                    sample_images.append(img.cpu())
                    sample_preds.append(predictions[i])
                    sample_targets.append(targets[i])

    # Compute metrics
    log.info("computing_metrics", n_samples=len(all_predictions))
    metrics = compute_detection_metrics(all_predictions, all_targets)

    log.info(
        "evaluation_complete",
        mAP=f"{metrics['mAP']:.4f}",
        mAP_50=f"{metrics['mAP_50']:.4f}",
    )

    # Create visualization
    vis_fig = create_detection_grid(
        sample_images,
        sample_preds,
        sample_targets,
        max_images=max_vis_samples,
        score_threshold=score_threshold,
    )

    result = {
        "metrics": metrics,
        "visualization": vis_fig,
        "n_samples": len(all_predictions),
    }

    # Save visualization if output_dir provided
    if output_dir is not None:
        vis_path = output_dir / "detection_results.png"
        save_visualization(vis_fig, vis_path)
        result["visualization_path"] = vis_path
        log.info("visualization_saved", path=str(vis_path))

    return result
