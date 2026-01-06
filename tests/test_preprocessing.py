"""Tests for preprocessing pipeline."""

import numpy as np
import pytest

from detect_electrical_utility_from_satellite_images.data_loading import TileData
from detect_electrical_utility_from_satellite_images.preprocessing import (
    Patch,
    compute_patch_statistics,
    extract_patches,
    filter_patches,
)


@pytest.fixture
def sample_tile() -> TileData:
    """Create a sample tile for testing."""
    # 1000x1000 image with some infrastructure in the center
    image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    mask = np.zeros((1000, 1000), dtype=np.uint8)

    # Add a tower (class 1) in the center
    mask[400:600, 400:600] = 1
    # Add a line (class 3)
    mask[500, 300:700] = 3

    return TileData(
        name="test_tile",
        image=image,
        mask=mask,
        annotations=[],
        geojson=None,
    )


def test_extract_patches_no_overlap(sample_tile: TileData) -> None:
    """Test extracting non-overlapping patches."""
    patches = extract_patches(sample_tile, patch_size=500, stride=500)

    # 1000x1000 image with 500x500 patches should give 4 patches
    assert len(patches) == 4

    # Check patch dimensions
    for patch in patches:
        assert patch.image.shape == (500, 500, 3)
        assert patch.mask.shape == (500, 500)
        assert patch.tile_name == "test_tile"


def test_extract_patches_with_overlap(sample_tile: TileData) -> None:
    """Test extracting overlapping patches."""
    patches = extract_patches(sample_tile, patch_size=500, stride=250)

    # With stride 250, we get more patches
    assert len(patches) > 4


def test_extract_patches_infrastructure_detection(sample_tile: TileData) -> None:
    """Test that infrastructure is correctly detected in patches."""
    patches = extract_patches(sample_tile, patch_size=500, stride=500)

    # At least one patch should have infrastructure (center has tower)
    infra_patches = [p for p in patches if p.has_infrastructure]
    assert len(infra_patches) >= 1


def test_filter_patches_keeps_all_infrastructure() -> None:
    """Test that filtering keeps all infrastructure patches."""
    patches = [
        Patch(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            mask=np.zeros((100, 100), dtype=np.uint8),
            tile_name="test",
            x=0,
            y=0,
            has_infrastructure=True,
        ),
        Patch(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            mask=np.zeros((100, 100), dtype=np.uint8),
            tile_name="test",
            x=100,
            y=0,
            has_infrastructure=False,
        ),
        Patch(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            mask=np.zeros((100, 100), dtype=np.uint8),
            tile_name="test",
            x=200,
            y=0,
            has_infrastructure=False,
        ),
    ]

    # With 0% background, only keep infrastructure
    filtered = filter_patches(patches, background_fraction=0.0)
    assert len(filtered) == 1
    assert all(p.has_infrastructure for p in filtered)

    # With 100% background, keep all
    filtered = filter_patches(patches, background_fraction=1.0)
    assert len(filtered) == 3


def test_filter_patches_reproducibility() -> None:
    """Test that filtering is reproducible with same seed."""
    patches = [
        Patch(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            mask=np.zeros((100, 100), dtype=np.uint8),
            tile_name="test",
            x=i * 100,
            y=0,
            has_infrastructure=False,
        )
        for i in range(100)
    ]

    result1 = filter_patches(patches, background_fraction=0.1, seed=42)
    result2 = filter_patches(patches, background_fraction=0.1, seed=42)

    # Same seed should give same results
    assert len(result1) == len(result2)
    assert [p.x for p in result1] == [p.x for p in result2]


def test_compute_patch_statistics() -> None:
    """Test computing patch statistics."""
    # Create patches with known distributions
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[:50, :] = 1  # Half is tower

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[:, :50] = 3  # Half is line

    patches = [
        Patch(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            mask=mask1,
            tile_name="test",
            x=0,
            y=0,
            has_infrastructure=True,
        ),
        Patch(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            mask=mask2,
            tile_name="test",
            x=100,
            y=0,
            has_infrastructure=True,
        ),
    ]

    stats = compute_patch_statistics(patches)

    assert stats["total_patches"] == 2
    assert stats["patches_with_infrastructure"] == 2
    assert stats["patches_background_only"] == 0
    assert stats["infrastructure_fraction"] == 1.0
    assert "tower" in stats["class_pixel_fractions"]
    assert "line" in stats["class_pixel_fractions"]
