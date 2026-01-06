"""Tests for tower detection dataset."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from detect_electrical_utility_from_satellite_images.datasets.tower_dataset import (
    TowerDetectionDataset,
    collate_fn,
    extract_bounding_boxes_from_mask,
)


def test_extract_bounding_boxes_single_tower() -> None:
    """Test extracting bounding box from a mask with one tower."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 1  # Tower at (30,20) to (50,40)

    boxes, labels = extract_bounding_boxes_from_mask(mask)

    assert boxes.shape == (1, 4)
    assert labels.shape == (1,)
    assert labels[0] == 1

    # Check box coordinates (x1, y1, x2, y2)
    assert boxes[0, 0] == 30  # x1
    assert boxes[0, 1] == 20  # y1
    assert boxes[0, 2] == 49  # x2
    assert boxes[0, 3] == 39  # y2


def test_extract_bounding_boxes_multiple_towers() -> None:
    """Test extracting bounding boxes from a mask with multiple towers."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # Tower 1
    mask[60:80, 60:80] = 1  # Tower 2

    boxes, labels = extract_bounding_boxes_from_mask(mask)

    assert boxes.shape == (2, 4)
    assert labels.shape == (2,)
    assert all(labels == 1)


def test_extract_bounding_boxes_empty_mask() -> None:
    """Test extracting from empty mask returns empty arrays."""
    mask = np.zeros((100, 100), dtype=np.uint8)

    boxes, labels = extract_bounding_boxes_from_mask(mask)

    assert boxes.shape == (0, 4)
    assert labels.shape == (0,)


def test_extract_bounding_boxes_other_tower() -> None:
    """Test that other_tower class (2) is also detected."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 2  # Other tower class

    boxes, labels = extract_bounding_boxes_from_mask(mask)

    assert boxes.shape == (1, 4)
    assert labels[0] == 1  # Both classes get label 1


def test_tower_dataset_initialization(tmp_path: Path) -> None:
    """Test dataset initialization."""
    # Create mock data
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Create a sample image and mask
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 1  # Tower

    Image.fromarray(img).save(images_dir / "sample_000001.png")
    Image.fromarray(mask).save(masks_dir / "sample_000001.png")

    dataset = TowerDetectionDataset(tmp_path, include_background=True)

    assert len(dataset) == 1


def test_tower_dataset_getitem(tmp_path: Path) -> None:
    """Test getting a sample from the dataset."""
    # Create mock data
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 1

    Image.fromarray(img).save(images_dir / "sample_000001.png")
    Image.fromarray(mask).save(masks_dir / "sample_000001.png")

    dataset = TowerDetectionDataset(tmp_path, include_background=True)
    sample = dataset[0]

    assert "image" in sample
    assert "boxes" in sample
    assert "labels" in sample
    assert "image_id" in sample

    assert sample["image"].shape == (3, 100, 100)
    assert sample["boxes"].shape[1] == 4
    assert isinstance(sample["image"], torch.Tensor)


def test_collate_fn() -> None:
    """Test the custom collate function."""
    batch = [
        {
            "image": torch.rand(3, 100, 100),
            "boxes": torch.tensor([[10, 10, 20, 20]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor([0]),
        },
        {
            "image": torch.rand(3, 100, 100),
            "boxes": torch.tensor([[30, 30, 40, 40], [50, 50, 60, 60]]),
            "labels": torch.tensor([1, 1]),
            "image_id": torch.tensor([1]),
        },
    ]

    images, targets = collate_fn(batch)

    assert len(images) == 2
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (2, 4)
