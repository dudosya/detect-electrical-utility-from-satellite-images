"""Evaluation utilities for line segmentation (Stage 2)."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

from detect_electrical_utility_from_satellite_images.logging_config import get_logger


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute segmentation metrics (IoU, Dice, Precision, Recall).

    Args:
        predictions: Predicted probabilities (B, 1, H, W) or (B, H, W).
        targets: Ground truth binary masks (B, 1, H, W) or (B, H, W).
        threshold: Threshold for binarizing predictions.

    Returns:
        Dictionary with IoU, Dice, Precision, Recall, F1.
    """
    # Flatten and binarize
    preds = (predictions > threshold).float().view(-1)
    targs = targets.float().view(-1)

    # Compute TP, FP, FN
    tp = (preds * targs).sum().item()
    fp = (preds * (1 - targs)).sum().item()
    fn = ((1 - preds) * targs).sum().item()
    tn = ((1 - preds) * (1 - targs)).sum().item()

    # Avoid division by zero
    eps = 1e-7

    # IoU (Jaccard Index)
    iou = tp / (tp + fp + fn + eps)

    # Dice coefficient
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    # Precision and Recall
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    # F1 Score
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def visualize_segmentation(
    image: NDArray[np.uint8],
    ground_truth: NDArray[np.float32],
    prediction: NDArray[np.float32],
    threshold: float = 0.5,
    alpha: float = 0.5,
) -> Figure:
    """Create visualization of segmentation results.

    Args:
        image: RGB image (H, W, 3) in [0, 255].
        ground_truth: Ground truth mask (H, W) in [0, 1].
        prediction: Predicted probability map (H, W) in [0, 1].
        threshold: Threshold for binarizing prediction.
        alpha: Transparency for overlay.

    Returns:
        Matplotlib figure with 4 subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Normalize image to [0, 1] for display
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0

    # 1. Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    # 2. Ground truth overlay
    gt_overlay = image.copy()
    gt_mask = ground_truth > 0.5
    gt_overlay[gt_mask] = gt_overlay[gt_mask] * (1 - alpha) + np.array([0, 0, 1]) * alpha
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("Ground Truth (blue)")
    axes[0, 1].axis("off")

    # 3. Prediction probability map
    im = axes[1, 0].imshow(prediction, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Prediction (probability)")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Prediction overlay with comparison
    pred_overlay = image.copy()
    pred_mask = prediction > threshold

    # True positives (green), False positives (red), False negatives (blue)
    tp_mask = pred_mask & gt_mask
    fp_mask = pred_mask & ~gt_mask
    fn_mask = ~pred_mask & gt_mask

    pred_overlay[tp_mask] = [0, 1, 0]  # Green - correct
    pred_overlay[fp_mask] = [1, 0, 0]  # Red - false positive
    pred_overlay[fn_mask] = [0, 0, 1]  # Blue - false negative

    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title("Comparison (G=TP, R=FP, B=FN)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def create_segmentation_grid(
    images: list[NDArray[np.uint8]],
    ground_truths: list[NDArray[np.float32]],
    predictions: list[NDArray[np.float32]],
    threshold: float = 0.5,
    max_samples: int = 8,
) -> Figure:
    """Create a grid visualization of multiple segmentation results.

    Args:
        images: List of RGB images.
        ground_truths: List of ground truth masks.
        predictions: List of predicted probability maps.
        threshold: Threshold for binarizing predictions.
        max_samples: Maximum number of samples to show.

    Returns:
        Matplotlib figure with grid of results.
    """
    n_samples = min(len(images), max_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        img = images[i]
        gt = ground_truths[i]
        pred = predictions[i]

        # Normalize image
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0

        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input" if i == 0 else "")
        axes[i, 0].axis("off")

        # Ground truth + prediction overlay
        overlay = img.copy()
        gt_mask = gt > 0.5
        pred_mask = pred > threshold

        # Show GT in blue, prediction boundary in red
        overlay[gt_mask] = overlay[gt_mask] * 0.5 + np.array([0, 0, 1]) * 0.5

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis("off")

        # Prediction probability
        pred_overlay = img.copy()
        pred_overlay[pred_mask] = pred_overlay[pred_mask] * 0.5 + np.array([1, 0.3, 0]) * 0.5

        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title("Prediction" if i == 0 else "")
        axes[i, 2].axis("off")

    plt.tight_layout()
    return fig


@torch.no_grad()
def evaluate_line_model_on_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    max_vis_samples: int = 8,
) -> dict[str, Any]:
    """Evaluate line segmentation model on a dataset.

    Args:
        model: Trained U-Net model.
        dataloader: DataLoader for evaluation.
        device: Device to run inference on.
        threshold: Threshold for binarizing predictions.
        max_vis_samples: Number of samples for visualization.

    Returns:
        Dictionary with metrics and visualization samples.
    """
    model.eval()
    log = get_logger()

    all_preds = []
    all_targets = []
    vis_images = []
    vis_gts = []
    vis_preds = []

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)

        # Forward pass
        logits = model(images)
        probs = torch.sigmoid(logits)

        all_preds.append(probs.cpu())
        all_targets.append(masks.cpu())

        # Collect visualization samples
        if len(vis_images) < max_vis_samples:
            for i in range(min(images.size(0), max_vis_samples - len(vis_images))):
                vis_images.append(
                    (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                )
                vis_gts.append(masks[i, 0].cpu().numpy())
                vis_preds.append(probs[i, 0].cpu().numpy())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_segmentation_metrics(all_preds, all_targets, threshold)

    log.info(
        "line_evaluation_complete",
        IoU=f"{metrics['IoU']:.4f}",
        Dice=f"{metrics['Dice']:.4f}",
        Precision=f"{metrics['Precision']:.4f}",
        Recall=f"{metrics['Recall']:.4f}",
    )

    # Create visualization
    vis_figure = None
    if vis_images:
        vis_figure = create_segmentation_grid(
            vis_images, vis_gts, vis_preds, threshold, max_vis_samples
        )

    return {
        "metrics": metrics,
        "visualization": vis_figure,
        "num_samples": len(all_preds),
    }
