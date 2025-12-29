"""
Utilities for splitting datasets by original image to prevent data leakage.
"""
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def group_patches_by_image(
    img_patch_paths: List[Path], mask_patch_paths: List[Path]
) -> Dict[str, Tuple[List[Path], List[Path]]]:
    """Group image and mask patches by their original image name.

    Args:
        img_patch_paths: List of paths to image patch files.
        mask_patch_paths: List of paths to mask patch files.

    Returns:
        Dictionary mapping original image name to tuple of
        (list of image patch paths, list of mask patch paths).
    """
    # Validate that lists have same length
    if len(img_patch_paths) != len(mask_patch_paths):
        raise ValueError(
            f"Number of image patches ({len(img_patch_paths)}) does not match "
            f"number of mask patches ({len(mask_patch_paths)})"
        )

    # Group patches by original image
    image_groups: Dict[str, Tuple[List[Path], List[Path]]] = defaultdict(
        lambda: ([], [])
    )

    for img_path, mask_path in zip(img_patch_paths, mask_patch_paths):
        # Extract original image name from patch filename
        # Format: {original_name}_img_{y}_{x}.png or {original_name}_msk_{y}_{x}.png
        img_stem = img_path.stem
        mask_stem = mask_path.stem

        # Remove the _img_... or _msk_... suffix
        if "_img_" in img_stem:
            original_name = img_stem.split("_img_")[0]
        elif "_msk_" in img_stem:
            original_name = img_stem.split("_msk_")[0]
        else:
            raise ValueError(f"Unexpected image patch filename format: {img_path.name}")

        # Verify mask has same original name
        if "_msk_" in mask_stem:
            mask_original_name = mask_stem.split("_msk_")[0]
        elif "_img_" in mask_stem:
            mask_original_name = mask_stem.split("_img_")[0]
        else:
            raise ValueError(f"Unexpected mask patch filename format: {mask_path.name}")

        if original_name != mask_original_name:
            raise ValueError(
                f"Image and mask patches from different original images: "
                f"{img_path.name} vs {mask_path.name}"
            )

        image_groups[original_name][0].append(img_path)
        image_groups[original_name][1].append(mask_path)

    logger.info(f"Grouped patches into {len(image_groups)} original images")
    for orig_name, (img_paths, mask_paths) in image_groups.items():
        logger.debug(f"  {orig_name}: {len(img_paths)} patches")

    return dict(image_groups)


def split_by_image(
    img_patch_paths: List[Path],
    mask_patch_paths: List[Path],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """Split patches by original image to prevent data leakage.

    Args:
        img_patch_paths: List of paths to image patch files.
        mask_patch_paths: List of paths to mask patch files.
        train_ratio: Proportion of original images to use for training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_img_paths, train_mask_paths, val_img_paths, val_mask_paths)
    """
    # Group patches by original image
    image_groups = group_patches_by_image(img_patch_paths, mask_patch_paths)

    # Get list of original image names
    original_names = list(image_groups.keys())
    num_images = len(original_names)

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Shuffle original image names
    shuffled_names = original_names.copy()
    rng.shuffle(shuffled_names)

    # Split original images
    split_idx = int(train_ratio * num_images)
    train_names = shuffled_names[:split_idx]
    val_names = shuffled_names[split_idx:]

    logger.info(
        f"Split {num_images} original images: {len(train_names)} train, {len(val_names)} validation"
    )

    # Collect patches for each split
    train_img_paths: List[Path] = []
    train_mask_paths: List[Path] = []
    val_img_paths: List[Path] = []
    val_mask_paths: List[Path] = []

    for name in train_names:
        img_paths, mask_paths = image_groups[name]
        train_img_paths.extend(img_paths)
        train_mask_paths.extend(mask_paths)

    for name in val_names:
        img_paths, mask_paths = image_groups[name]
        val_img_paths.extend(img_paths)
        val_mask_paths.extend(mask_paths)

    # Verify no overlap
    train_set = set(train_img_paths)
    val_set = set(val_img_paths)
    if train_set.intersection(val_set):
        raise ValueError("Data leakage detected: Some patches appear in both splits")

    logger.info(
        f"Total patches: {len(train_img_paths)} train, {len(val_img_paths)} validation"
    )

    return train_img_paths, train_mask_paths, val_img_paths, val_mask_paths


def validate_split_by_image(
    train_img_paths: List[Path],
    val_img_paths: List[Path],
) -> bool:
    """Validate that no original image appears in both train and validation sets.

    Args:
        train_img_paths: List of training image patch paths.
        val_img_paths: List of validation image patch paths.

    Returns:
        True if split is valid (no leakage), False otherwise.
    """
    # Extract original image names from all patches
    def extract_original_names(paths: List[Path]) -> set:
        names = set()
        for path in paths:
            stem = path.stem
            if "_img_" in stem:
                names.add(stem.split("_img_")[0])
            elif "_msk_" in stem:
                names.add(stem.split("_msk_")[0])
        return names

    train_originals = extract_original_names(train_img_paths)
    val_originals = extract_original_names(val_img_paths)

    overlap = train_originals.intersection(val_originals)
    if overlap:
        logger.error(f"Data leakage: {len(overlap)} original images in both splits")
        for name in list(overlap)[:5]:  # Show first 5
            logger.error(f"  - {name}")
        return False

    logger.info("Split validation passed: No original images in both splits")
    return True
