"""Data loading utilities for the GridTracer dataset."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from detect_electrical_utility_from_satellite_images.logging_config import get_logger


@dataclass
class TileData:
    """Container for a single dataset tile with all associated data.

    Attributes:
        name: Tile identifier (e.g., 'NZ_Dunedin_1').
        image: RGB satellite image as numpy array (H, W, 3).
        mask: Multiclass segmentation mask (H, W).
        annotations: List of annotation dictionaries from CSV.
        geojson: GeoJSON feature collection if available.
    """

    name: str
    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    annotations: list[dict]
    geojson: dict | None = None


# Mask class values as defined in dataset_description.md
MASK_CLASSES = {
    0: "background",
    1: "tower",
    2: "other_tower",
    3: "line",
    4: "edge_node",
    5: "substation",
}


def load_image(path: Path) -> NDArray[np.uint8]:
    """Load an image file as a numpy array.

    Args:
        path: Path to the image file.

    Returns:
        Image as numpy array with shape (H, W, 3) for RGB.

    Raises:
        FileNotFoundError: If the image file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def load_mask(path: Path) -> NDArray[np.uint8]:
    """Load a segmentation mask as a numpy array.

    Args:
        path: Path to the mask file (.png or .npz).

    Returns:
        Mask as numpy array with shape (H, W).

    Raises:
        FileNotFoundError: If the mask file doesn't exist.
        ValueError: If the file format is not supported.
    """
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path)
        # Assume the mask is stored under 'arr_0' or 'mask' key
        if "arr_0" in data:
            return data["arr_0"].astype(np.uint8)
        elif "mask" in data:
            return data["mask"].astype(np.uint8)
        else:
            raise ValueError(f"Unknown npz keys: {list(data.keys())}")
    elif path.suffix == ".png":
        with Image.open(path) as img:
            return np.array(img)
    else:
        raise ValueError(f"Unsupported mask format: {path.suffix}")


def load_annotations_csv(path: Path) -> list[dict]:
    """Load annotations from a CSV file.

    The CSV format has columns for vertex coordinates grouped by Object ID.

    Args:
        path: Path to the CSV file.

    Returns:
        List of annotation dictionaries.
    """
    if not path.exists():
        return []

    annotations = []
    with path.open("r") as f:
        lines = f.readlines()

    if not lines:
        return []

    # Parse header
    header = lines[0].strip().split(",")

    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.strip().split(",")
        annotation = dict(zip(header, values, strict=False))
        annotations.append(annotation)

    return annotations


def load_geojson(path: Path) -> dict | None:
    """Load a GeoJSON file.

    Args:
        path: Path to the GeoJSON file.

    Returns:
        GeoJSON dictionary or None if file doesn't exist.
    """
    if not path.exists():
        return None

    with path.open("r") as f:
        return json.load(f)


def load_tile(base_path: Path, tile_name: str) -> TileData:
    """Load all data for a single tile.

    Args:
        base_path: Base directory containing the tile files.
        tile_name: Name of the tile (e.g., 'NZ_Dunedin_1').

    Returns:
        TileData instance with all loaded data.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    log = get_logger()

    image_path = base_path / f"{tile_name}.jpg"
    mask_path = base_path / f"{tile_name}_multiclass.png"
    csv_path = base_path / f"{tile_name}.csv"
    geojson_path = base_path / f"{tile_name}.geojson"

    log.debug("loading_tile", tile=tile_name, path=str(base_path))

    image = load_image(image_path)
    mask = load_mask(mask_path)
    annotations = load_annotations_csv(csv_path)
    geojson = load_geojson(geojson_path)

    log.debug(
        "tile_loaded",
        tile=tile_name,
        image_shape=image.shape,
        mask_shape=mask.shape,
        num_annotations=len(annotations),
    )

    return TileData(
        name=tile_name,
        image=image,
        mask=mask,
        annotations=annotations,
        geojson=geojson,
    )


def discover_regions(data_dir: Path) -> list[str]:
    """Discover all available regions in the data directory.

    Args:
        data_dir: Path to the raw_data directory.

    Returns:
        List of region names (e.g., ['NZ_Dunedin', 'US_Tucson']).
    """
    if not data_dir.exists():
        return []

    regions = []
    for item in data_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            regions.append(item.name)

    return sorted(regions)


def discover_tiles(region_dir: Path) -> list[str]:
    """Discover all tiles within a region directory.

    Args:
        region_dir: Path to a region directory.

    Returns:
        List of tile names (e.g., ['NZ_Dunedin_1', 'NZ_Dunedin_2']).
    """
    if not region_dir.exists():
        return []

    tiles = set()
    for jpg in region_dir.glob("*.jpg"):
        # Extract tile name from filename (remove .jpg extension)
        tile_name = jpg.stem
        tiles.add(tile_name)

    return sorted(tiles)
