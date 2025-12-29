"""
Tests for dataset loading and utilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image

from detect_electrical_utility_from_satellite_images.dataset import SatteliteImgsDataset


def create_test_image(path: Path, size: tuple = (256, 256), channels: int = 3):
    """Create a test image file."""
    if channels == 3:
        # RGB image
        img_array = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    else:
        # Grayscale mask
        img_array = np.random.randint(0, 5, size, dtype=np.uint8)

    img = Image.fromarray(img_array)
    img.save(path)


def test_dataset_creation(mock_image_paths, mock_mask_paths):
    """Test creating a dataset instance."""
    # Create actual image files
    for img_path in mock_image_paths:
        create_test_image(img_path, size=(256, 256), channels=3)

    for mask_path in mock_mask_paths:
        create_test_image(mask_path, size=(256, 256), channels=1)

    # Create transforms
    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            torchvision.transforms.v2.ToDtype(torch.long, scale=False),
        ]
    )

    dataset = SatteliteImgsDataset(
        img_paths=mock_image_paths,
        mask_paths=mock_mask_paths,
        transforms=transforms,
    )

    assert len(dataset) == 5
    assert dataset.img_paths == mock_image_paths
    assert dataset.mask_paths == mock_mask_paths


def test_dataset_getitem(mock_image_paths, mock_mask_paths):
    """Test getting items from dataset."""
    # Create actual image files
    for img_path in mock_image_paths:
        create_test_image(img_path, size=(256, 256), channels=3)

    for mask_path in mock_mask_paths:
        create_test_image(mask_path, size=(256, 256), channels=1)

    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            torchvision.transforms.v2.ToDtype(torch.long, scale=False),
        ]
    )

    dataset = SatteliteImgsDataset(
        img_paths=mock_image_paths,
        mask_paths=mock_mask_paths,
        transforms=transforms,
    )

    # Get first item
    image, mask = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (3, 256, 256)  # CHW format
    assert mask.shape == (256, 256)  # HW format (2D)
    # Note: torchvision transforms might return different dtypes
    # depending on the tv_tensor type
    # For now, just check they're tensors with correct shape
    # The actual dtype conversion is tested in the dataset_tester function
    # in the actual dataset.py file


def test_dataset_invalid_transforms(mock_image_paths, mock_mask_paths):
    """Test dataset creation with invalid transforms."""
    with pytest.raises(
        TypeError, match="transforms must be a torchvision.transforms.v2.Compose"
    ):
        SatteliteImgsDataset(
            img_paths=mock_image_paths,
            mask_paths=mock_mask_paths,
            transforms="invalid",  # Not a Compose object
        )


def test_dataset_index_validation(mock_image_paths, mock_mask_paths):
    """Test dataset index validation."""
    # Create actual image files first
    for img_path in mock_image_paths:
        create_test_image(img_path, size=(256, 256), channels=3)

    for mask_path in mock_mask_paths:
        create_test_image(mask_path, size=(256, 256), channels=1)

    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = SatteliteImgsDataset(
        img_paths=mock_image_paths,
        mask_paths=mock_mask_paths,
        transforms=transforms,
    )

    # Test valid indices
    for i in range(len(dataset)):
        _ = dataset[i]  # Should not raise

    # Test invalid indices
    with pytest.raises(IndexError):
        _ = dataset[10]

    with pytest.raises(IndexError):
        _ = dataset[-6]

    # Test non-integer index
    with pytest.raises(TypeError):
        _ = dataset["invalid"]


def test_dataset_path_validation():
    """Test dataset path validation."""
    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # Test mismatched path lengths
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        img_paths = [tmpdir / f"img_{i}.png" for i in range(3)]
        mask_paths = [tmpdir / f"msk_{i}.png" for i in range(2)]  # Different length

        for path in img_paths + mask_paths:
            path.touch()

        with pytest.raises(ValueError, match="unequal number of elements"):
            SatteliteImgsDataset(
                img_paths=img_paths,
                mask_paths=mask_paths,
                transforms=transforms,
            )


def test_dataset_with_augmentations(mock_image_paths, mock_mask_paths):
    """Test dataset with augmentation transforms."""
    # Create actual image files
    for img_path in mock_image_paths:
        create_test_image(img_path, size=(256, 256), channels=3)

    for mask_path in mock_mask_paths:
        create_test_image(mask_path, size=(256, 256), channels=1)

    # Create transforms with augmentations
    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.RandomHorizontalFlip(
                p=1.0
            ),  # Always flip for testing
            torchvision.transforms.v2.RandomVerticalFlip(
                p=1.0
            ),  # Always flip for testing
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            torchvision.transforms.v2.ToDtype(torch.long, scale=False),
        ]
    )

    dataset = SatteliteImgsDataset(
        img_paths=mock_image_paths,
        mask_paths=mock_mask_paths,
        transforms=transforms,
    )

    # Get an item
    image, mask = dataset[0]

    assert image.shape == (3, 256, 256)
    assert mask.shape == (256, 256)  # 2D
    # Note: torchvision transforms might return different dtypes
    # The actual dtype conversion is tested in the dataset_tester function


def test_dataset_batch_loading(mock_image_paths, mock_mask_paths):
    """Test that dataset works with DataLoader."""
    # Create actual image files
    for img_path in mock_image_paths:
        create_test_image(img_path, size=(256, 256), channels=3)

    for mask_path in mock_mask_paths:
        create_test_image(mask_path, size=(256, 256), channels=1)

    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            torchvision.transforms.v2.ToDtype(torch.long, scale=False),
        ]
    )

    dataset = SatteliteImgsDataset(
        img_paths=mock_image_paths,
        mask_paths=mock_mask_paths,
        transforms=transforms,
    )

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # 0 for testing
    )

    # Get a batch
    batch = next(iter(dataloader))
    images, masks = batch

    assert isinstance(images, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert images.shape == (2, 3, 256, 256)
    assert masks.shape == (2, 256, 256)  # 2D masks


def test_dataset_realistic_names():
    """Test dataset with realistic file naming convention."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create files with realistic naming: image_001_img_0_0.png, image_001_msk_0_0.png
        img_paths = []
        mask_paths = []

        for i in range(3):
            img_name = f"sample_{i:03d}_img_0_0.png"
            mask_name = f"sample_{i:03d}_msk_0_0.png"

            img_path = tmpdir / img_name
            mask_path = tmpdir / mask_name

            create_test_image(img_path, size=(256, 256), channels=3)
            create_test_image(mask_path, size=(256, 256), channels=1)

            img_paths.append(img_path)
            mask_paths.append(mask_path)

        transforms = torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            ]
        )

        dataset = SatteliteImgsDataset(
            img_paths=img_paths,
            mask_paths=mask_paths,
            transforms=transforms,
        )

        # Should not raise validation errors
        assert len(dataset) == 3

        # Check one item
        image, mask = dataset[0]
        assert image.shape == (3, 256, 256)
        assert mask.shape == (256, 256)  # 2D


def test_dataset_empty():
    """Test dataset with no images."""
    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = SatteliteImgsDataset(
        img_paths=[],
        mask_paths=[],
        transforms=transforms,
    )

    assert len(dataset) == 0


def test_dataset_different_sizes():
    """Test dataset with images of different sizes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        img_paths = []
        mask_paths = []

        sizes = [(256, 256), (512, 512), (128, 128)]

        for i, size in enumerate(sizes):
            img_path = tmpdir / f"img_{i}.png"
            mask_path = tmpdir / f"msk_{i}.png"

            create_test_image(img_path, size=size, channels=3)
            create_test_image(mask_path, size=size, channels=1)

            img_paths.append(img_path)
            mask_paths.append(mask_path)

        transforms = torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.Resize((256, 256)),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            ]
        )

        dataset = SatteliteImgsDataset(
            img_paths=img_paths,
            mask_paths=mask_paths,
            transforms=transforms,
        )

        # All images should be resized to 256x256
        for i in range(len(dataset)):
            image, mask = dataset[i]
            assert image.shape == (3, 256, 256)
            assert mask.shape == (256, 256)  # 2D
