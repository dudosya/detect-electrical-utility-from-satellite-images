"""
Tests for split_utils module.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from detect_electrical_utility_from_satellite_images.utils.split_utils import (
    group_patches_by_image,
    split_by_image,
    validate_split_by_image,
)


def create_mock_patch_files(temp_dir: Path, num_images: int = 3, patches_per_image: int = 4):
    """Create mock patch files for testing.
    
    Creates files with names like:
    - image1_img_0_0.png, image1_msk_0_0.png
    - image1_img_0_256.png, image1_msk_0_256.png
    - image2_img_0_0.png, image2_msk_0_0.png
    etc.
    """
    img_dir = temp_dir / "img_patches"
    mask_dir = temp_dir / "mask_patches"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    img_paths = []
    mask_paths = []
    
    for img_idx in range(num_images):
        for patch_idx in range(patches_per_image):
            y = patch_idx * 256
            x = patch_idx * 256
            img_name = f"image{img_idx}_img_{y}_{x}.png"
            mask_name = f"image{img_idx}_msk_{y}_{x}.png"
            
            img_path = img_dir / img_name
            mask_path = mask_dir / mask_name
            
            img_path.touch()
            mask_path.touch()
            
            img_paths.append(img_path)
            mask_paths.append(mask_path)
    
    return img_paths, mask_paths


def test_group_patches_by_image():
    """Test grouping patches by original image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        img_paths, mask_paths = create_mock_patch_files(temp_dir, num_images=3, patches_per_image=4)
        
        groups = group_patches_by_image(img_paths, mask_paths)
        
        # Should have 3 original images
        assert len(groups) == 3
        
        # Each image should have 4 patches
        for orig_name, (img_list, mask_list) in groups.items():
            assert len(img_list) == 4
            assert len(mask_list) == 4
            # Check that all patches belong to the same original image
            for img_path in img_list:
                assert orig_name in img_path.stem
            for mask_path in mask_list:
                assert orig_name in mask_path.stem


def test_group_patches_by_image_mismatched_lengths():
    """Test that mismatched lengths raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        img_paths, mask_paths = create_mock_patch_files(temp_dir, num_images=2, patches_per_image=3)
        
        # Remove one mask to create mismatch
        mask_paths.pop()
        
        with pytest.raises(ValueError, match="does not match"):
            group_patches_by_image(img_paths, mask_paths)


def test_group_patches_by_image_mismatched_names():
    """Test that patches from different original images raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        img_dir = temp_dir / "img_patches"
        mask_dir = temp_dir / "mask_patches"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mismatched pair
        img_path = img_dir / "image1_img_0_0.png"
        mask_path = mask_dir / "image2_msk_0_0.png"
        img_path.touch()
        mask_path.touch()
        
        with pytest.raises(ValueError, match="different original images"):
            group_patches_by_image([img_path], [mask_path])


def test_split_by_image():
    """Test splitting patches by original image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        img_paths, mask_paths = create_mock_patch_files(
            temp_dir, num_images=10, patches_per_image=5
        )
        
        train_img, train_mask, val_img, val_mask = split_by_image(
            img_paths, mask_paths, train_ratio=0.8, seed=42
        )
        
        # Check total counts
        total_patches = len(img_paths)
        assert len(train_img) + len(val_img) == total_patches
        assert len(train_mask) + len(val_mask) == total_patches
        
        # Check no overlap
        train_set = set(train_img)
        val_set = set(val_img)
        assert len(train_set.intersection(val_set)) == 0
        
        # Check that patches from same image are in same split
        # Group by original image
        def get_original_name(path: Path) -> str:
            stem = path.stem
            if "_img_" in stem:
                return stem.split("_img_")[0]
            else:
                return stem.split("_msk_")[0]
        
        train_originals = {get_original_name(p) for p in train_img}
        val_originals = {get_original_name(p) for p in val_img}
        
        # No original image should be in both splits
        assert len(train_originals.intersection(val_originals)) == 0
        
        # With 10 images and 0.8 ratio, we should have 8 train, 2 val images
        # (but due to rounding it could be 8/2 or 7/3)
        assert len(train_originals) >= 7
        assert len(val_originals) >= 2


def test_split_by_image_reproducibility():
    """Test that split is reproducible with same seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        img_paths, mask_paths = create_mock_patch_files(
            temp_dir, num_images=10, patches_per_image=3
        )
        
        # First split with seed 42
        train_img1, train_mask1, val_img1, val_mask1 = split_by_image(
            img_paths, mask_paths, train_ratio=0.7, seed=42
        )
        
        # Second split with same seed
        train_img2, train_mask2, val_img2, val_mask2 = split_by_image(
            img_paths, mask_paths, train_ratio=0.7, seed=42
        )
        
        # Should be identical
        assert set(train_img1) == set(train_img2)
        assert set(val_img1) == set(val_img2)


def test_split_by_image_different_seeds():
    """Test that different seeds produce different splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        img_paths, mask_paths = create_mock_patch_files(
            temp_dir, num_images=10, patches_per_image=3
        )
        
        # Split with seed 42
        train_img1, train_mask1, val_img1, val_mask1 = split_by_image(
            img_paths, mask_paths, train_ratio=0.7, seed=42
        )
        
        # Split with seed 123
        train_img2, train_mask2, val_img2, val_mask2 = split_by_image(
            img_paths, mask_paths, train_ratio=0.7, seed=123
        )
        
        # With high probability they should be different (not guaranteed but likely)
        # At least check they're valid splits
        assert len(train_img1) == len(train_img2)  # Same ratio
        assert set(train_img1) != set(train_img2) or set(val_img1) != set(val_img2)


def test_validate_split_by_image():
    """Test validation of split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        img_paths, mask_paths = create_mock_patch_files(
            temp_dir, num_images=5, patches_per_image=4
        )
        
        # Create a valid split
        train_img, train_mask, val_img, val_mask = split_by_image(
            img_paths, mask_paths, train_ratio=0.6, seed=42
        )
        
        # Validation should pass
        assert validate_split_by_image(train_img, val_img) is True
        
        # Create an invalid split by adding a patch from train to val
        if val_img:
            invalid_val = val_img + [train_img[0]]
            assert validate_split_by_image(train_img, invalid_val) is False
        
        # Create an invalid split by having same original image in both
        if train_img and val_img:
            # Get original names
            def get_original(path):
                stem = path.stem
                return stem.split("_img_")[0] if "_img_" in stem else stem.split("_msk_")[0]
            
            # Find an original from train
            train_original = get_original(train_img[0])
            # Create a fake val patch with same original
            fake_val_path = Path(str(val_img[0]).replace(get_original(val_img[0]), train_original))
            invalid_val = list(val_img) + [fake_val_path]
            assert validate_split_by_image(train_img, invalid_val) is False


def test_split_by_image_edge_cases():
    """Test edge cases for split_by_image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        
        # Test with single image
        img_paths, mask_paths = create_mock_patch_files(
            temp_dir, num_images=1, patches_per_image=10
        )
        
        # With single image, all patches go to train (train_ratio=1.0)
        train_img, train_mask, val_img, val_mask = split_by_image(
            img_paths, mask_paths, train_ratio=1.0, seed=42
        )
        assert len(train_img) == 10
        assert len(val_img) == 0
        
        # With train_ratio=0.0, all patches go to val
        train_img, train_mask, val_img, val_mask = split_by_image(
            img_paths, mask_paths, train_ratio=0.0, seed=42
        )
        assert len(train_img) == 0
        assert len(val_img) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
