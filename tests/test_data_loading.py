"""Tests for data loading utilities."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detect_electrical_utility_from_satellite_images.data_loading import (
    MASK_CLASSES,
    discover_regions,
    discover_tiles,
    load_annotations_csv,
    load_image,
    load_mask,
)


def test_mask_classes_defined() -> None:
    """Test that all mask classes are defined correctly."""
    assert MASK_CLASSES[0] == "background"
    assert MASK_CLASSES[1] == "tower"
    assert MASK_CLASSES[2] == "other_tower"
    assert MASK_CLASSES[3] == "line"
    assert MASK_CLASSES[4] == "edge_node"
    assert MASK_CLASSES[5] == "substation"


def test_load_image(tmp_path: Path) -> None:
    """Test loading an image file."""
    # Create a test image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    Image.fromarray(img_array).save(img_path)

    loaded = load_image(img_path)

    assert loaded.shape == (100, 100, 3)
    assert loaded.dtype == np.uint8


def test_load_image_missing_file() -> None:
    """Test that missing image raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_image(Path("nonexistent.jpg"))


def test_load_mask_png(tmp_path: Path) -> None:
    """Test loading a PNG mask file."""
    # Create a test mask
    mask_array = np.random.randint(0, 6, (100, 100), dtype=np.uint8)
    mask_path = tmp_path / "test_mask.png"
    Image.fromarray(mask_array).save(mask_path)

    loaded = load_mask(mask_path)

    assert loaded.shape == (100, 100)


def test_load_mask_npz(tmp_path: Path) -> None:
    """Test loading an NPZ mask file."""
    # Create a test mask
    mask_array = np.random.randint(0, 6, (100, 100), dtype=np.uint8)
    mask_path = tmp_path / "test_mask.npz"
    np.savez(mask_path, arr_0=mask_array)

    loaded = load_mask(mask_path)

    assert loaded.shape == (100, 100)
    np.testing.assert_array_equal(loaded, mask_array)


def test_load_annotations_csv(tmp_path: Path) -> None:
    """Test loading CSV annotations."""
    csv_content = """Object ID,X,Y,Type
1,100,200,Tower
1,150,250,Tower
2,300,400,Line
"""
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(csv_content)

    annotations = load_annotations_csv(csv_path)

    assert len(annotations) == 3
    assert annotations[0]["Object ID"] == "1"
    assert annotations[0]["Type"] == "Tower"


def test_load_annotations_csv_missing_file(tmp_path: Path) -> None:
    """Test that missing CSV returns empty list."""
    annotations = load_annotations_csv(tmp_path / "nonexistent.csv")
    assert annotations == []


def test_discover_regions(tmp_path: Path) -> None:
    """Test discovering regions in data directory."""
    # Create test region directories
    (tmp_path / "NZ_Dunedin").mkdir()
    (tmp_path / "US_Tucson").mkdir()
    (tmp_path / ".hidden").mkdir()  # Should be ignored

    regions = discover_regions(tmp_path)

    assert "NZ_Dunedin" in regions
    assert "US_Tucson" in regions
    assert ".hidden" not in regions


def test_discover_tiles(tmp_path: Path) -> None:
    """Test discovering tiles in a region directory."""
    # Create test tile files
    (tmp_path / "Region_1.jpg").touch()
    (tmp_path / "Region_2.jpg").touch()
    (tmp_path / "Region_1.csv").touch()

    tiles = discover_tiles(tmp_path)

    assert "Region_1" in tiles
    assert "Region_2" in tiles
    assert len(tiles) == 2
