"""Segmentation metrics for electrical utility detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = "macro",
    ignore_index: int | None = None,
) -> float | dict[int, float]:
    """
    Calculate Intersection over Union (IoU/Jaccard Index).

    Args:
        predictions: Predicted class indices [batch_size, height, width] or [height, width]
        targets: Ground truth class indices [batch_size, height, width] or [height, width]
        num_classes: Number of classes
        average: "macro", "micro", "weighted", or "none" for per-class
        ignore_index: Class index to ignore (e.g., background)

    Returns:
        IoU score(s) based on average type
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    if ignore_index is not None:
        mask = targets != ignore_index
        predictions = predictions[mask]
        targets = targets[mask]

    # Calculate confusion matrix
    cm = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=predictions.device,
    )
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = torch.sum((predictions == i) & (targets == j))

    # Calculate IoU per class
    ious = []
    for c in range(num_classes):
        intersection = cm[c, c]
        union = torch.sum(cm[c, :]) + torch.sum(cm[:, c]) - intersection
        if union > 0:
            ious.append(intersection.float() / union.float())
        else:
            ious.append(torch.tensor(0.0, device=predictions.device))

    ious = torch.stack(ious)

    if average == "none":
        return {i: ious[i].item() for i in range(num_classes)}
    if average == "macro":
        return torch.mean(ious).item()
    if average == "micro":
        total_intersection = torch.sum(torch.diag(cm))
        total_union = torch.sum(cm) - total_intersection
        return (
            (total_intersection.float() / total_union.float()).item()
            if total_union > 0
            else 0.0
        )
    if average == "weighted":
        # Weight by class frequency
        class_counts = torch.sum(cm, dim=1)
        weights = class_counts.float() / torch.sum(class_counts).float()
        return torch.sum(ious * weights).item()
    raise ValueError(f"Unknown average type: {average}")


def calculate_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = "macro",
    ignore_index: int | None = None,
) -> float | dict[int, float]:
    """
    Calculate Dice coefficient (F1 score).

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes
        average: "macro", "micro", "weighted", or "none" for per-class
        ignore_index: Class index to ignore

    Returns:
        Dice score(s) based on average type
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    if ignore_index is not None:
        mask = targets != ignore_index
        predictions = predictions[mask]
        targets = targets[mask]

    # Calculate confusion matrix
    cm = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=predictions.device,
    )
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = torch.sum((predictions == i) & (targets == j))

    # Calculate Dice per class
    dice_scores = []
    for c in range(num_classes):
        intersection = cm[c, c]
        total = torch.sum(cm[c, :]) + torch.sum(cm[:, c])
        if total > 0:
            dice_scores.append(2 * intersection.float() / total.float())
        else:
            dice_scores.append(torch.tensor(0.0, device=predictions.device))

    dice_scores = torch.stack(dice_scores)

    if average == "none":
        return {i: dice_scores[i].item() for i in range(num_classes)}
    if average == "macro":
        return torch.mean(dice_scores).item()
    if average == "micro":
        total_intersection = torch.sum(torch.diag(cm))
        total = torch.sum(cm)
        return (
            (2 * total_intersection.float() / total.float()).item()
            if total > 0
            else 0.0
        )
    if average == "weighted":
        class_counts = torch.sum(cm, dim=1)
        weights = class_counts.float() / torch.sum(class_counts).float()
        return torch.sum(dice_scores * weights).item()
    raise ValueError(f"Unknown average type: {average}")


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
) -> float:
    """
    Calculate pixel accuracy.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        ignore_index: Class index to ignore

    Returns:
        Accuracy score
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    if ignore_index is not None:
        mask = targets != ignore_index
        predictions = predictions[mask]
        targets = targets[mask]

    correct = torch.sum(predictions == targets)
    total = len(predictions)

    return (correct.float() / total).item() if total > 0 else 0.0


def calculate_precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = "macro",
    ignore_index: int | None = None,
) -> dict[str, float | dict[int, float]]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes
        average: "macro", "micro", "weighted", or "none" for per-class
        ignore_index: Class index to ignore

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    if ignore_index is not None:
        mask = targets != ignore_index
        predictions = predictions[mask]
        targets = targets[mask]

    # Calculate confusion matrix
    cm = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=predictions.device,
    )
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = torch.sum((predictions == i) & (targets == j))

    # Calculate per-class metrics
    precisions = []
    recalls = []
    f1_scores = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = torch.sum(cm[c, :]) - tp
        fn = torch.sum(cm[:, c]) - tp

        precision = (
            tp.float() / (tp + fp)
            if (tp + fp) > 0
            else torch.tensor(0.0, device=predictions.device)
        )
        recall = (
            tp.float() / (tp + fn)
            if (tp + fn) > 0
            else torch.tensor(0.0, device=predictions.device)
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else torch.tensor(0.0, device=predictions.device)
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    precisions = torch.stack(precisions)
    recalls = torch.stack(recalls)
    f1_scores = torch.stack(f1_scores)

    if average == "none":
        return {
            "precision": {i: precisions[i].item() for i in range(num_classes)},
            "recall": {i: recalls[i].item() for i in range(num_classes)},
            "f1": {i: f1_scores[i].item() for i in range(num_classes)},
        }
    if average == "macro":
        return {
            "precision": torch.mean(precisions).item(),
            "recall": torch.mean(recalls).item(),
            "f1": torch.mean(f1_scores).item(),
        }
    if average == "micro":
        total_tp = torch.sum(torch.diag(cm))
        total_fp = torch.sum(cm) - total_tp - torch.sum(torch.diag(cm.T))
        total_fn = torch.sum(cm.T) - total_tp - torch.sum(torch.diag(cm))

        micro_precision = (
            total_tp.float() / (total_tp + total_fp)
            if (total_tp + total_fp) > 0
            else torch.tensor(0.0, device=predictions.device)
        )
        micro_recall = (
            total_tp.float() / (total_tp + total_fn)
            if (total_tp + total_fn) > 0
            else torch.tensor(0.0, device=predictions.device)
        )
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else torch.tensor(0.0, device=predictions.device)
        )

        return {
            "precision": micro_precision.item(),
            "recall": micro_recall.item(),
            "f1": micro_f1.item(),
        }
    if average == "weighted":
        class_counts = torch.sum(cm, dim=1)
        weights = class_counts.float() / torch.sum(class_counts).float()

        return {
            "precision": torch.sum(precisions * weights).item(),
            "recall": torch.sum(recalls * weights).item(),
            "f1": torch.sum(f1_scores * weights).item(),
        }
    raise ValueError(f"Unknown average type: {average}")


def extract_boundary(
    mask: torch.Tensor,
    dilation_radius: int = 1,
) -> torch.Tensor:
    """
    Extract boundary pixels from a segmentation mask.

    Args:
        mask: Segmentation mask [H, W] or [batch_size, H, W]
        dilation_radius: Radius for boundary dilation

    Returns:
        Binary boundary mask
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # Add batch dimension
    
    # Convert to float for convolution
    mask_float = mask.float()
    
    # Create a simple edge kernel
    kernel = torch.ones(1, 1, 3, 3, device=mask.device) / 9.0
    
    # Apply convolution to smooth the mask
    smoothed = F.conv2d(
        mask_float.unsqueeze(1),  # Add channel dimension
        kernel,
        padding=1,
    )
    
    # Boundary is where the smoothed value is between 0 and 1
    boundary = (smoothed > 0) & (smoothed < 1)
    
    # Remove added dimensions
    boundary = boundary.squeeze(1)
    
    if dilation_radius > 1:
        # Create dilation kernel
        kernel_size = 2 * dilation_radius + 1
        dilation_kernel = torch.ones(
            1, 1, kernel_size, kernel_size, device=mask.device
        )
        boundary = F.conv2d(
            boundary.float().unsqueeze(1),
            dilation_kernel,
            padding=dilation_radius,
        ) > 0
        boundary = boundary.squeeze(1)
    
    return boundary


def calculate_boundary_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    boundary_dilation: int = 2,
    average: str = "macro",
    ignore_index: int | None = None,
) -> float | dict[int, float]:
    """
    Calculate Boundary IoU (Intersection over Union on boundary pixels).

    Args:
        predictions: Predicted class indices [batch_size, height, width] or [height, width]
        targets: Ground truth class indices [batch_size, height, width] or [height, width]
        num_classes: Number of classes
        boundary_dilation: Dilation radius for boundary extraction
        average: "macro", "micro", "weighted", or "none" for per-class
        ignore_index: Class index to ignore (e.g., background)

    Returns:
        Boundary IoU score(s) based on average type
    """
    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(0)
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)
    
    batch_size = predictions.shape[0]
    
    # Initialize accumulators
    total_intersection = torch.zeros(num_classes, device=predictions.device)
    total_union = torch.zeros(num_classes, device=predictions.device)
    
    for batch_idx in range(batch_size):
        pred = predictions[batch_idx]
        target = targets[batch_idx]
        
        if ignore_index is not None:
            # Create mask for valid pixels
            valid_mask = target != ignore_index
            pred = pred[valid_mask]
            target = target[valid_mask]
            if pred.numel() == 0:
                continue
        
        for c in range(num_classes):
            # Create binary masks for class c
            pred_binary = (pred == c)
            target_binary = (target == c)
            
            # Skip if no pixels of this class in ground truth
            if not target_binary.any():
                continue
            
            # Extract boundaries
            pred_boundary = extract_boundary(pred_binary, boundary_dilation)
            target_boundary = extract_boundary(target_binary, boundary_dilation)
            
            # Calculate intersection and union
            intersection = (pred_boundary & target_boundary).sum()
            union = (pred_boundary | target_boundary).sum()
            
            total_intersection[c] += intersection
            total_union[c] += union
    
    # Calculate Boundary IoU per class
    boundary_ious = []
    for c in range(num_classes):
        if total_union[c] > 0:
            boundary_ious.append(total_intersection[c].float() / total_union[c].float())
        else:
            boundary_ious.append(torch.tensor(0.0, device=predictions.device))
    
    boundary_ious = torch.stack(boundary_ious)
    
    if average == "none":
        return {i: boundary_ious[i].item() for i in range(num_classes)}
    if average == "macro":
        # Average over classes that have at least some boundary pixels
        valid_classes = total_union > 0
        if valid_classes.any():
            return torch.mean(boundary_ious[valid_classes]).item()
        return 0.0
    if average == "micro":
        total_inter = total_intersection.sum()
        total_un = total_union.sum()
        return (total_inter.float() / total_un.float()).item() if total_un > 0 else 0.0
    if average == "weighted":
        # Weight by number of boundary pixels in ground truth
        weights = total_union.float() / total_union.sum().float()
        return torch.sum(boundary_ious * weights).item()
    raise ValueError(f"Unknown average type: {average}")


def calculate_boundary_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    boundary_dilation: int = 2,
    average: str = "macro",
    ignore_index: int | None = None,
) -> float | dict[int, float]:
    """
    Calculate Boundary F1 score (Dice coefficient on boundary pixels).

    Args:
        predictions: Predicted class indices [batch_size, height, width] or [height, width]
        targets: Ground truth class indices [batch_size, height, width] or [height, width]
        num_classes: Number of classes
        boundary_dilation: Dilation radius for boundary extraction
        average: "macro", "micro", "weighted", or "none" for per-class
        ignore_index: Class index to ignore (e.g., background)

    Returns:
        Boundary F1 score(s) based on average type
    """
    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(0)
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)
    
    batch_size = predictions.shape[0]
    
    # Initialize accumulators
    total_intersection = torch.zeros(num_classes, device=predictions.device)
    total_pred_boundary = torch.zeros(num_classes, device=predictions.device)
    total_target_boundary = torch.zeros(num_classes, device=predictions.device)
    
    for batch_idx in range(batch_size):
        pred = predictions[batch_idx]
        target = targets[batch_idx]
        
        if ignore_index is not None:
            # Create mask for valid pixels
            valid_mask = target != ignore_index
            pred = pred[valid_mask]
            target = target[valid_mask]
            if pred.numel() == 0:
                continue
        
        for c in range(num_classes):
            # Create binary masks for class c
            pred_binary = (pred == c)
            target_binary = (target == c)
            
            # Skip if no pixels of this class in ground truth
            if not target_binary.any():
                continue
            
            # Extract boundaries
            pred_boundary = extract_boundary(pred_binary, boundary_dilation)
            target_boundary = extract_boundary(target_binary, boundary_dilation)
            
            # Calculate intersection and totals
            intersection = (pred_boundary & target_boundary).sum()
            total_pred_boundary[c] += pred_boundary.sum()
            total_target_boundary[c] += target_boundary.sum()
            total_intersection[c] += intersection
    
    # Calculate Boundary F1 per class
    boundary_f1_scores = []
    for c in range(num_classes):
        if total_pred_boundary[c] + total_target_boundary[c] > 0:
            f1 = 2 * total_intersection[c].float() / (
                total_pred_boundary[c].float() + total_target_boundary[c].float()
            )
            boundary_f1_scores.append(f1)
        else:
            boundary_f1_scores.append(torch.tensor(0.0, device=predictions.device))
    
    boundary_f1_scores = torch.stack(boundary_f1_scores)
    
    if average == "none":
        return {i: boundary_f1_scores[i].item() for i in range(num_classes)}
    if average == "macro":
        # Average over classes that have at least some boundary pixels
        valid_classes = (total_pred_boundary + total_target_boundary) > 0
        if valid_classes.any():
            return torch.mean(boundary_f1_scores[valid_classes]).item()
        return 0.0
    if average == "micro":
        total_inter = total_intersection.sum()
        total_pred = total_pred_boundary.sum()
        total_target = total_target_boundary.sum()
        if total_pred + total_target > 0:
            return (2 * total_inter.float() / (total_pred.float() + total_target.float())).item()
        return 0.0
    if average == "weighted":
        # Weight by number of boundary pixels in ground truth
        weights = total_target_boundary.float() / total_target_boundary.sum().float()
        return torch.sum(boundary_f1_scores * weights).item()
    raise ValueError(f"Unknown average type: {average}")


def calculate_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    metrics_config: dict,
    ignore_index: int | None = None,
) -> dict[str, float | dict[int, float]]:
    """
    Calculate all requested metrics.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes
        metrics_config: Configuration dictionary with 'track' list and average settings
        ignore_index: Class index to ignore

    Returns:
        Dictionary with all calculated metrics
    """
    results = {}

    if "iou" in metrics_config.get("track", []):
        results["iou"] = calculate_iou(
            predictions,
            targets,
            num_classes,
            average=metrics_config.get("iou_average", "macro"),
            ignore_index=ignore_index,
        )

    if "dice" in metrics_config.get("track", []):
        results["dice"] = calculate_dice(
            predictions,
            targets,
            num_classes,
            average=metrics_config.get("dice_average", "macro"),
            ignore_index=ignore_index,
        )

    if "accuracy" in metrics_config.get("track", []):
        results["accuracy"] = calculate_accuracy(predictions, targets, ignore_index)

    if any(m in metrics_config.get("track", []) for m in ["precision", "recall", "f1"]):
        prf_results = calculate_precision_recall_f1(
            predictions,
            targets,
            num_classes,
            average=metrics_config.get("f1_average", "macro"),
            ignore_index=ignore_index,
        )
        results.update(prf_results)

    # Boundary metrics
    if "boundary_iou" in metrics_config.get("track", []):
        results["boundary_iou"] = calculate_boundary_iou(
            predictions,
            targets,
            num_classes,
            boundary_dilation=metrics_config.get("boundary_dilation", 2),
            average=metrics_config.get("boundary_iou_average", "macro"),
            ignore_index=ignore_index,
        )

    if "boundary_f1" in metrics_config.get("track", []):
        results["boundary_f1"] = calculate_boundary_f1(
            predictions,
            targets,
            num_classes,
            boundary_dilation=metrics_config.get("boundary_dilation", 2),
            average=metrics_config.get("boundary_f1_average", "macro"),
            ignore_index=ignore_index,
        )

    return results


def visualize_predictions(
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> dict[str, np.ndarray]:
    """
    Create visualization arrays for logging.

    Args:
        image: Original image [C, H, W]
        prediction: Predicted mask [H, W]
        target: Ground truth mask [H, W]
        num_classes: Number of classes

    Returns:
        Dictionary with visualization arrays
    """
    # Convert to numpy for visualization
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    prediction_np = prediction.cpu().numpy()
    target_np = target.cpu().numpy()

    # Normalize image for display
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    # Create colored masks
    def create_colored_mask(mask, num_classes):
        # Simple colormap for visualization
        colors = np.array(
            [
                [0, 0, 0],  # Background - black
                [255, 0, 0],  # Class 1 - red
                [0, 255, 0],  # Class 2 - green
                [0, 0, 255],  # Class 3 - blue
                [255, 255, 0],  # Class 4 - yellow
                [255, 0, 255],  # Class 5 - magenta
                [0, 255, 255],  # Class 6 - cyan
            ],
        )

        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for c in range(min(num_classes, len(colors))):
            colored[mask == c] = colors[c]
        return colored / 255.0

    pred_colored = create_colored_mask(prediction_np, num_classes)
    target_colored = create_colored_mask(target_np, num_classes)

    # Create overlay (50% image + 50% mask)
    pred_overlay = 0.5 * image_np + 0.5 * pred_colored
    target_overlay = 0.5 * image_np + 0.5 * target_colored

    return {
        "image": image_np,
        "prediction": pred_colored,
        "target": target_colored,
        "prediction_overlay": pred_overlay,
        "target_overlay": target_overlay,
    }
