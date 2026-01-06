"""Tower detection model using Faster R-CNN."""

from typing import Any

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detect_electrical_utility_from_satellite_images.logging_config import get_logger


def create_anchor_generator(
    anchor_areas: list[int],
    anchor_ratios: list[float],
) -> AnchorGenerator:
    """Create a custom anchor generator for tower detection.

    The paper uses custom anchors optimized for small tower footprints:
    - Areas: {10², 25², 50², 100², 200²} pixels
    - Ratios: {0.5, 1.0, 2.0}

    Args:
        anchor_areas: List of anchor areas (e.g., [100, 625, 2500, 10000, 40000]).
        anchor_ratios: List of aspect ratios (e.g., [0.5, 1.0, 2.0]).

    Returns:
        Configured AnchorGenerator.
    """
    # Convert areas to sizes (side length of square)
    sizes = tuple(tuple([int(a**0.5)]) for a in anchor_areas)
    aspect_ratios = tuple([tuple(anchor_ratios)] * len(anchor_areas))

    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


def create_tower_detector(
    num_classes: int = 2,  # background + tower
    backbone_name: str = "resnet50",
    pretrained_backbone: bool = True,
    anchor_areas: list[int] | None = None,
    anchor_ratios: list[float] | None = None,
    trainable_backbone_layers: int = 3,
    min_size: int = 500,
    max_size: int = 500,
    box_score_thresh: float = 0.5,
    box_nms_thresh: float = 0.5,
) -> FasterRCNN:
    """Create a Faster R-CNN model for tower detection.

    Note: The paper uses Inception V2, but we use ResNet50-FPN as it's more
    readily available in torchvision and provides similar performance.
    For exact paper replication, consider using a custom Inception V2 backbone.

    Args:
        num_classes: Number of classes (including background).
        backbone_name: Backbone architecture ('resnet50' or 'resnet101').
        pretrained_backbone: Whether to use pretrained weights.
        anchor_areas: Custom anchor areas. Defaults to paper values.
        anchor_ratios: Custom anchor ratios. Defaults to paper values.
        trainable_backbone_layers: Number of backbone layers to train.
        min_size: Minimum image size for transform.
        max_size: Maximum image size for transform.
        box_score_thresh: Score threshold for inference.
        box_nms_thresh: NMS threshold for inference.

    Returns:
        Configured Faster R-CNN model.
    """
    log = get_logger()

    # Default anchor config from paper
    if anchor_areas is None:
        anchor_areas = [100, 625, 2500, 10000, 40000]  # 10², 25², 50², 100², 200²
    if anchor_ratios is None:
        anchor_ratios = [0.5, 1.0, 2.0]

    log.info(
        "creating_tower_detector",
        backbone=backbone_name,
        num_classes=num_classes,
        anchor_areas=anchor_areas,
        anchor_ratios=anchor_ratios,
    )

    # Create backbone with FPN
    weights = "DEFAULT" if pretrained_backbone else None
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=weights,
        trainable_layers=trainable_backbone_layers,
    )

    # Create custom anchor generator
    # FPN has 5 feature levels, so we need 5 sets of anchors
    anchor_generator = AnchorGenerator(
        sizes=((10,), (25,), (50,), (100,), (200,)),
        aspect_ratios=(anchor_ratios,) * 5,
    )

    # Create model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
    )

    log.info("tower_detector_created", num_params=sum(p.numel() for p in model.parameters()))

    return model


def get_tower_centroids(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.5,
) -> torch.Tensor:
    """Extract tower centroids from detection boxes.

    Args:
        boxes: (N, 4) tensor of [x1, y1, x2, y2] boxes.
        scores: (N,) tensor of confidence scores.
        score_threshold: Minimum score to keep.

    Returns:
        (M, 2) tensor of [x, y] centroids for boxes above threshold.
    """
    mask = scores >= score_threshold
    filtered_boxes = boxes[mask]

    if len(filtered_boxes) == 0:
        return torch.zeros((0, 2), device=boxes.device)

    # Compute centroids
    x_center = (filtered_boxes[:, 0] + filtered_boxes[:, 2]) / 2
    y_center = (filtered_boxes[:, 1] + filtered_boxes[:, 3]) / 2

    return torch.stack([x_center, y_center], dim=1)
