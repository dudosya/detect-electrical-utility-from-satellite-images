"""
Loss functions for segmentation tasks with class imbalance support.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Cross entropy loss with optional class weights."""

    def __init__(self, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross entropy loss.

        Args:
            input: (N, C, H, W) logits
            target: (N, H, W) ground truth labels

        Returns:
            Scalar loss value
        """
        return F.cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks.

    Dice coefficient measures overlap between prediction and ground truth.
    Dice loss = 1 - Dice coefficient.
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            input: (N, C, H, W) logits
            target: (N, H, W) ground truth labels

        Returns:
            Scalar loss value
        """
        # Convert to probabilities
        probs = F.softmax(input, dim=1)

        # One-hot encode target
        n_classes = probs.shape[1]
        target_one_hot = (
            F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()
        )

        # Compute Dice per class
        intersection = torch.sum(probs * target_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        # Average across classes and batch
        if self.reduction == "mean":
            return dice_loss.mean()
        if self.reduction == "sum":
            return dice_loss.sum()
        return dice_loss.mean(dim=1)  # Return per-batch loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Focal loss reduces the relative loss for well-classified examples
    and focuses more on hard, misclassified examples.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            input: (N, C, H, W) logits
            target: (N, H, W) ground truth labels

        Returns:
            Scalar loss value
        """
        ce_loss = F.cross_entropy(input, target, weight=self.alpha, reduction="none")

        # Get probabilities
        probs = F.softmax(input, dim=1)

        # Gather probabilities for true classes
        target_probs = torch.gather(probs, 1, target.unsqueeze(1)).squeeze(1)

        # Compute modulating factor
        modulating_factor = (1 - target_probs) ** self.gamma

        # Apply modulating factor
        focal_loss = modulating_factor * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """Combination of multiple losses (e.g., CrossEntropy + Dice)."""

    def __init__(self, losses: list[nn.Module], weights: list[float] | None = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            input: (N, C, H, W) logits
            target: (N, H, W) ground truth labels

        Returns:
            Scalar combined loss value
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(input, target)
        return total_loss


def create_loss_function(
    loss_name: str,
    num_classes: int,
    device: str = "cpu",
    class_weights: list[float] | None = None,
    loss_alpha: float = 0.5,
    loss_gamma: float = 2.0,
) -> nn.Module:
    """Factory function to create loss function from configuration.

    Args:
        loss_name: Name of loss function ('cross_entropy', 'dice', 'focal', 'combined')
        num_classes: Number of classes in segmentation task
        device: Device to place tensors on
        class_weights: Optional list of class weights
        loss_alpha: Alpha parameter for combined loss
        loss_gamma: Gamma parameter for focal loss

    Returns:
        Initialized loss function
    """
    # Convert class weights to tensor if provided
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if loss_name == "cross_entropy":
        return WeightedCrossEntropyLoss(weight=weight_tensor)

    if loss_name == "dice":
        return DiceLoss()

    if loss_name == "focal":
        return FocalLoss(alpha=weight_tensor, gamma=loss_gamma)

    if loss_name == "combined":
        # Combined loss: CrossEntropy + Dice
        ce_loss = WeightedCrossEntropyLoss(weight=weight_tensor)
        dice_loss = DiceLoss()
        return CombinedLoss(
            [ce_loss, dice_loss], weights=[loss_alpha, 1.0 - loss_alpha]
        )

    raise ValueError(f"Unknown loss function: {loss_name}")


def calculate_class_weights(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Calculate class weights from dataset for imbalanced classification.

    Uses inverse frequency weighting: weight = total_pixels / (num_classes * class_pixels)

    Args:
        dataset: PyTorch dataset with masks
        num_classes: Number of classes
        device: Device to place tensor on

    Returns:
        Tensor of class weights
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    # Count pixels per class
    for _, mask in dataset:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Flatten and count
        unique, counts = np.unique(mask_np, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < num_classes:
                class_counts[cls] += count

    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)

    # Inverse frequency weighting
    total_pixels = class_counts.sum()
    weights = total_pixels / (num_classes * class_counts)

    # Normalize to sum to num_classes
    weights = weights * (num_classes / weights.sum())

    return weights.to(torch.float32).to(device)
