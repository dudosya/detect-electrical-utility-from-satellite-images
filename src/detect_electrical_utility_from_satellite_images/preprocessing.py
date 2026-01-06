"""Preprocessing pipeline for creating training patches from satellite imagery."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from detect_electrical_utility_from_satellite_images.config import Config
from detect_electrical_utility_from_satellite_images.data_loading import (
    MASK_CLASSES,
    TileData,
    discover_regions,
    discover_tiles,
    load_tile,
)
from detect_electrical_utility_from_satellite_images.logging_config import get_logger


@dataclass
class Patch:
    """A single image patch with its mask.

    Attributes:
        image: RGB image patch (patch_size, patch_size, 3).
        mask: Segmentation mask patch (patch_size, patch_size).
        tile_name: Source tile name.
        x: X coordinate of patch origin in source image.
        y: Y coordinate of patch origin in source image.
        has_infrastructure: Whether this patch contains any infrastructure.
    """

    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    tile_name: str
    x: int
    y: int
    has_infrastructure: bool


def extract_patches(
    tile: TileData,
    patch_size: int,
    stride: int | None = None,
) -> list[Patch]:
    """Extract patches from a tile using a sliding window.

    Args:
        tile: TileData instance to extract patches from.
        patch_size: Size of each square patch.
        stride: Step size between patches. Defaults to patch_size (no overlap).

    Returns:
        List of Patch instances.
    """
    if stride is None:
        stride = patch_size

    h, w = tile.image.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = tile.image[y : y + patch_size, x : x + patch_size]
            mask_patch = tile.mask[y : y + patch_size, x : x + patch_size]

            # Check if patch contains any infrastructure (non-background pixels)
            has_infrastructure = np.any(mask_patch > 0)

            patches.append(
                Patch(
                    image=img_patch,
                    mask=mask_patch,
                    tile_name=tile.name,
                    x=x,
                    y=y,
                    has_infrastructure=has_infrastructure,
                )
            )

    return patches


def filter_patches(
    patches: list[Patch],
    background_fraction: float,
    seed: int = 42,
) -> list[Patch]:
    """Filter patches to balance infrastructure vs background.

    Keeps all patches with infrastructure, and a random subset of
    background-only patches.

    Args:
        patches: List of patches to filter.
        background_fraction: Fraction of background patches to keep (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Filtered list of patches.
    """
    infrastructure_patches = [p for p in patches if p.has_infrastructure]
    background_patches = [p for p in patches if not p.has_infrastructure]

    # Randomly sample background patches
    random.seed(seed)
    num_background = int(len(background_patches) * background_fraction)
    sampled_background = random.sample(
        background_patches, min(num_background, len(background_patches))
    )

    return infrastructure_patches + sampled_background


def save_patch(patch: Patch, output_dir: Path, index: int) -> tuple[Path, Path]:
    """Save a patch to disk.

    Args:
        patch: Patch to save.
        output_dir: Directory to save patches to.
        index: Unique index for the patch filename.

    Returns:
        Tuple of (image_path, mask_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filenames with metadata
    base_name = f"{patch.tile_name}_{index:06d}_x{patch.x}_y{patch.y}"
    image_path = output_dir / "images" / f"{base_name}.png"
    mask_path = output_dir / "masks" / f"{base_name}.png"

    # Ensure subdirectories exist
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image
    Image.fromarray(patch.image).save(image_path)

    # Save mask (as single-channel PNG)
    Image.fromarray(patch.mask).save(mask_path)

    return image_path, mask_path


def compute_patch_statistics(patches: list[Patch]) -> dict:
    """Compute statistics about the patches.

    Args:
        patches: List of patches to analyze.

    Returns:
        Dictionary with statistics.
    """
    total = len(patches)
    with_infra = sum(1 for p in patches if p.has_infrastructure)

    # Count pixels per class across all patches
    class_pixels: dict[int, int] = {k: 0 for k in MASK_CLASSES}
    for patch in patches:
        unique, counts = np.unique(patch.mask, return_counts=True)
        for val, count in zip(unique, counts, strict=False):
            if val in class_pixels:
                class_pixels[val] += count

    total_pixels = sum(class_pixels.values())
    class_fractions = {
        MASK_CLASSES[k]: v / total_pixels if total_pixels > 0 else 0
        for k, v in class_pixels.items()
    }

    return {
        "total_patches": total,
        "patches_with_infrastructure": with_infra,
        "patches_background_only": total - with_infra,
        "infrastructure_fraction": with_infra / total if total > 0 else 0,
        "class_pixel_fractions": class_fractions,
    }


def preprocess_tile(
    tile: TileData,
    config: Config,
    output_dir: Path,
    start_index: int = 0,
) -> tuple[int, dict]:
    """Preprocess a single tile into patches.

    Args:
        tile: TileData to preprocess.
        config: Configuration instance.
        output_dir: Output directory for patches.
        start_index: Starting index for patch numbering.

    Returns:
        Tuple of (next_index, statistics).
    """
    log = get_logger()

    log.info(
        "preprocessing_tile",
        tile=tile.name,
        image_shape=tile.image.shape,
    )

    # Extract patches
    patches = extract_patches(tile, config.preprocessing.patch_size)
    log.debug("patches_extracted", tile=tile.name, count=len(patches))

    # Filter patches
    patches = filter_patches(
        patches,
        config.preprocessing.background_fraction,
        seed=config.training.seed,
    )
    log.debug("patches_filtered", tile=tile.name, count=len(patches))

    # Save patches
    for i, patch in enumerate(patches):
        save_patch(patch, output_dir, start_index + i)

    # Compute statistics
    stats = compute_patch_statistics(patches)
    log.info(
        "tile_preprocessed",
        tile=tile.name,
        total_patches=stats["total_patches"],
        with_infrastructure=stats["patches_with_infrastructure"],
    )

    return start_index + len(patches), stats


def run_preprocessing(config: Config, region: str | None = None) -> None:
    """Run the preprocessing pipeline.

    Args:
        config: Configuration instance.
        region: Specific region to process, or None for all regions.
    """
    log = get_logger()
    data_dir = Path(config.paths.data_dir)
    output_dir = Path(config.paths.output_dir)

    # Discover regions
    if region:
        regions = [region]
    else:
        regions = discover_regions(data_dir)

    if not regions:
        log.warning("no_regions_found", data_dir=str(data_dir))
        return

    log.info("preprocessing_start", regions=regions, output_dir=str(output_dir))

    total_patches = 0
    all_stats: dict[str, dict] = {}

    for region_name in regions:
        region_dir = data_dir / region_name

        if not region_dir.exists():
            log.warning("region_not_found", region=region_name)
            continue

        tiles = discover_tiles(region_dir)
        log.info("processing_region", region=region_name, num_tiles=len(tiles))

        for tile_name in tiles:
            try:
                tile = load_tile(region_dir, tile_name)
                next_idx, stats = preprocess_tile(
                    tile, config, output_dir / region_name, total_patches
                )
                total_patches = next_idx
                all_stats[tile_name] = stats
            except Exception as e:
                log.exception("tile_processing_error", tile=tile_name, error=str(e))

    log.info(
        "preprocessing_complete",
        total_patches=total_patches,
        tiles_processed=len(all_stats),
    )
