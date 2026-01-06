import json
import logging
import math
import random
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from detect_electrical_utility_from_satellite_images.utils.file_utils import (
    drop_jpg_paths_with_no_npz_pair,
)

logger = logging.getLogger(__name__)


def calculate_gsd_from_geojson(geojson_path: Path, image_width: int, image_height: int) -> float:
    """Calculate Ground Sample Distance (GSD) from geojson geocoordinates.

    Uses Haversine formula to compute the ground distance represented by the image,
    then divides by pixel dimensions to get cm/pixel.

    Args:
        geojson_path: Path to the geojson file containing image_geocoordinates.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        GSD in cm/pixel (average of horizontal and vertical).

    Raises:
        FileNotFoundError: If geojson file doesn't exist.
        KeyError: If required geocoordinate fields are missing.
    """
    with open(geojson_path) as f:
        data = json.load(f)

    props = data["features"][0]["properties"]
    ul = props["image_geocoordinates_upper_left"]
    ur = props["image_geocoordinates_upper_right"]
    ll = props["image_geocoordinates_lower_left"]

    def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate great-circle distance between two points in meters."""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate ground distances
    width_m = haversine(ul[0], ul[1], ur[0], ur[1])
    height_m = haversine(ul[0], ul[1], ll[0], ll[1])

    # GSD in cm/pixel (average of horizontal and vertical)
    gsd_h = (width_m / image_width) * 100  # convert m to cm
    gsd_v = (height_m / image_height) * 100
    gsd = (gsd_h + gsd_v) / 2

    return gsd


def resample_to_target_gsd(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    source_gsd: float,
    target_gsd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample image and mask to match target GSD.

    Args:
        image_arr: Source image array (H, W, C).
        mask_arr: Source mask array (H, W).
        source_gsd: Current GSD of the image in cm/pixel.
        target_gsd: Desired GSD in cm/pixel.

    Returns:
        Tuple of (resampled_image, resampled_mask).
        Image uses LANCZOS for quality, mask uses NEAREST to preserve class labels.
    """
    scale_factor = source_gsd / target_gsd
    new_width = int(image_arr.shape[1] * scale_factor)
    new_height = int(image_arr.shape[0] * scale_factor)

    logger.info(
        f"Resampling: {image_arr.shape[:2]} -> ({new_height}, {new_width}) "
        f"(GSD: {source_gsd:.2f} -> {target_gsd:.2f} cm/px, scale: {scale_factor:.3f})"
    )

    # Resample image with high-quality interpolation
    img_pil = Image.fromarray(image_arr)
    img_resampled = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Resample mask with nearest neighbor to preserve class labels
    mask_pil = Image.fromarray(mask_arr)
    mask_resampled = mask_pil.resize((new_width, new_height), Image.Resampling.NEAREST)

    return np.array(img_resampled), np.array(mask_resampled)


def apply_clahe(
    image_arr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast.

    CLAHE is particularly effective for satellite imagery with thin structures like power lines,
    as it enhances local contrast without over-amplifying noise. Applied per RGB channel.

    Args:
        image_arr: Input image array (H, W, C) in uint8 format.
        clip_limit: Threshold for contrast limiting (higher = more contrast). Typical: 2.0-4.0.
        tile_grid_size: Size of grid for histogram equalization. Smaller = more local adaptation.

    Returns:
        Enhanced image array with same shape and dtype as input.

    Raises:
        ImportError: If opencv-python (cv2) is not installed.
    """
    try:
        import cv2
    except ImportError:
        logger.error(
            "opencv-python is required for CLAHE. Install with: uv add opencv-python"
        )
        raise

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply to each channel separately
    channels = [clahe.apply(image_arr[:, :, i]) for i in range(image_arr.shape[2])]

    # Stack channels back together
    enhanced = np.stack(channels, axis=2)

    logger.debug(
        f"Applied CLAHE: clip_limit={clip_limit}, tile_grid_size={tile_grid_size}"
    )

    return enhanced


def remap_classes(
    mask_arr: np.ndarray,
    classes_to_background: list[int] | None = None,
    fixed_class_order: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, int] | None]:
    """Remap specified classes to background (0) and renumber remaining classes.

    Args:
        mask_arr: The mask array with original class indices.
        classes_to_background: List of class indices to treat as background.
            If None or empty, returns original mask unchanged.
        fixed_class_order: Optional ordered list of source class ids to keep (excluding
            background). When provided, mapping follows this order regardless of which
            classes are present in a particular mask, ensuring stable indices across files.

    Returns:
        Tuple of (remapped mask array, mapping dict or None). Mapping uses contiguous
        class indices starting from 1 for non-background classes.

    Example:
        If original classes are [0, 1, 2, 3, 4, 5] and classes_to_background=[1]:
        - Class 3 (LINE) becomes 0 (background)
        - Classes 2,3,4,5 become 1,2,3,4 respectively
        Result: [0, 1, 2, 3, 4] (5 classes instead of 6)
    """
    classes_to_background_set = set(classes_to_background or [])

    if fixed_class_order:
        original_classes = [cls for cls in fixed_class_order if cls not in classes_to_background_set and cls != 0]
        if not original_classes:
            return np.zeros_like(mask_arr), {}
    else:
        original_classes = sorted(set(np.unique(mask_arr)) - classes_to_background_set - {0})
        if not original_classes:
            return mask_arr, None

    remapped = np.zeros_like(mask_arr)

    # Create mapping: old_class -> new_class (contiguous from 1)
    class_mapping = {old_cls: new_idx for new_idx, old_cls in enumerate(original_classes, start=1)}

    # Apply mapping
    for old_cls, new_cls in class_mapping.items():
        remapped[mask_arr == old_cls] = new_cls

    return remapped, class_mapping


def create_patches(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    patch_size: int,
    output_path: str | Path,
    original_name: str,
    background_fraction: float,
    classes_to_background: list[int] | None = None,
    fixed_class_order: list[int] | None = None,
) -> None:
    """Generate and save patches from a large image and its corresponding mask.

    This function takes a large image and mask, pads them to be perfectly divisible
    by the patch size, and then extracts smaller corresponding patches. It filters the patches,
    keeping all patches that contain labeled objects and a random sampling of patches that are
    purely background. The resulting image and mask patches are saved to separate subdirectories.

    Args:
        image_arr (np.ndarray): The source image as a NumPy array, expected in (H,W,C) format.
        mask_arr (np.ndarray): The corresponding mask as a NumPy array, expected with the same
            height and width as the image_arr.
        patch_size (int): The side length of the square patches to generate (e.g., 256 for 256x256).
        output_path (str | Path): The root directory where "img_patches" and "mask_patches"
            subfolders will be created and populated.
        original_name (str): A unique identifier for the source image, used as a prefix for
            the output patch filenames (e.g., "image_01").
        background_fraction (float): Probability from 0.0 to 1.0 of keeping a patch if its
            corresponding mask is empty (i.e., background).
        classes_to_background (list[int] | None): List of class indices to remap to background (0).
            Useful for ignoring certain classes like LINE. If None, no remapping is performed.
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    img_dir = output_path / "img_patches"
    mask_dir = output_path / "mask_patches"
    img_dir.mkdir(parents=False, exist_ok=True)
    mask_dir.mkdir(parents=False, exist_ok=True)

    logger.info(f"Creating patches in directory: {output_path}")
    logger.info(f"Image shape: {image_arr.shape}, Mask shape: {mask_arr.shape}")
    logger.info(f"Patch size: {patch_size}, Background fraction: {background_fraction}")
    if classes_to_background:
        logger.info(f"Classes remapped to background: {classes_to_background}")

    # Remap specified classes to background before processing
    mask_arr, class_mapping = remap_classes(mask_arr, classes_to_background, fixed_class_order)

    # Calculate how many complete patches fit (no padding)
    n_patches_h = image_arr.shape[0] // patch_size
    n_patches_w = image_arr.shape[1] // patch_size
    discarded_h = image_arr.shape[0] % patch_size
    discarded_w = image_arr.shape[1] % patch_size

    logger.info(
        f"Extracting {n_patches_h}x{n_patches_w} complete patches. "
        f"Discarding {discarded_h}px from bottom, {discarded_w}px from right (no padding).",
    )

    # generate the coordinates to slice (only complete patches, no padding)
    for y_start in range(0, n_patches_h * patch_size, patch_size):
        for x_start in range(0, n_patches_w * patch_size, patch_size):
            # define the ends
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            # slice from the original images
            img_patch = image_arr[y_start:y_end, x_start:x_end, :]
            mask_patch = mask_arr[y_start:y_end, x_start:x_end]

            # create Pillow Image objects from np arrays
            img_patch_obj = Image.fromarray(img_patch)
            mask_patch_obj = Image.fromarray(mask_patch, mode="L")

            # define names for the objects
            img_name = f"{original_name}_img_{y_start}_{x_start}.png"
            mask_name = f"{original_name}_msk_{y_start}_{x_start}.png"

            # full path names
            img_fp = img_dir / img_name
            mask_fp = mask_dir / mask_name

            # check if the mask patch is not all zeros
            # or if the mask patch is empty, then randomly save some of them
            if np.any(mask_patch) or random.random() < background_fraction:
                # save the images
                img_patch_obj.save(img_fp)
                mask_patch_obj.save(mask_fp)

                # logger
                logger.info(f"Saved {img_fp}")
                logger.info(f"Saved {mask_fp}")

            else:
                # logger
                logger.debug(f"Discarded {img_fp}")
                logger.debug(f"Discarded {mask_fp}")

    # Log remapping summary at the end so it appears once per run
    if classes_to_background:
        logger.info(
            "Class remap summary | to_background=%s | mapping=%s | remapped_uniques=%s",
            classes_to_background,
            class_mapping,
            np.unique(mask_arr),
        )


def summarize_mask_classes(mask_dir: Path, num_classes: int) -> None:
    """Log pixel counts per class and detect out-of-range labels in mask patches."""

    if num_classes <= 0:
        logger.warning("Invalid num_classes %s; skipping class summary", num_classes)
        return

    mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        logger.warning("No mask patches found in %s; skipping class summary", mask_dir)
        return

    counts = np.zeros(num_classes, dtype=np.int64)
    out_of_range: set[int] = set()

    for mask_path in mask_paths:
        mask_arr = np.array(Image.open(mask_path))
        uniques, freqs = np.unique(mask_arr, return_counts=True)
        for cls_val, freq in zip(uniques, freqs):
            cls_int = int(cls_val)
            if 0 <= cls_int < num_classes:
                counts[cls_int] += int(freq)
            else:
                out_of_range.add(cls_int)

    logger.info(
        "Mask class distribution | dir=%s | counts=%s | total_pixels=%d | out_of_range=%s",
        mask_dir,
        counts.tolist(),
        int(counts.sum()),
        sorted(out_of_range) if out_of_range else None,
    )

    if out_of_range:
        logger.error("Found out-of-range labels %s in %s", sorted(out_of_range), mask_dir)


def pad_to_patch_size(img_arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Pad a numpy array to be divisible by a patch size

    Adds constant padding (zeros) to the bottom and right edges of an array
    until its height and widths are perfectly divisible by the
    given patch size

    Args:
        img_arr (np.ndarray): The input array to pad, expected to have shape
            of (H,W,C)
        patch_size (int): The target patch size. The output array's
            height and width will be a multiple of this value

    Returns:
        np.ndarray: A new, padded array. If not padding was needed,
            a copy of the original array is returned
    """
    # calculate the needed paddings
    pad_h = (patch_size - img_arr.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - img_arr.shape[1] % patch_size) % patch_size

    # apply the paddings
    if img_arr.ndim == 3:
        padded_arr = np.pad(
            img_arr, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode="constant",
        )
    elif img_arr.ndim == 2:
        padded_arr = np.pad(
            img_arr, pad_width=((0, pad_h), (0, pad_w)), mode="constant",
        )
    else:
        raise ValueError(
            "image arr is expected to have two types of dims: (H,W,C) or (H,W)",
        )

    # return the padded arr
    return padded_arr


def jpg_paths_to_patches(cfg):
    """Process all jpg images into patches, with optional GSD resampling.

    Args:
        cfg: Application configuration containing paths, preprocessing settings,
             and optional target_gsd_cm for scale normalization.
    """
    # get all the jpg paths
    jpg_paths = sorted(cfg.paths.data_dir.glob("*/*.jpg"))

    # drop paths that dont have npz pair
    jpg_paths = drop_jpg_paths_with_no_npz_pair(jpg_paths)

    target_gsd = cfg.preprocessing.target_gsd_cm
    gsd_tolerance = cfg.preprocessing.gsd_tolerance

    if target_gsd:
        logger.info(f"GSD normalization enabled: target={target_gsd} cm/px, tolerance={gsd_tolerance*100}%")

    for jpg_path in jpg_paths:
        # get filename
        filename = jpg_path.stem

        # load img arr
        PIL_obj = Image.open(jpg_path)
        img_arr = np.array(PIL_obj)

        # Apply CLAHE enhancement if enabled
        if cfg.preprocessing.apply_clahe:
            img_arr = apply_clahe(
                img_arr,
                clip_limit=cfg.preprocessing.clahe_clip_limit,
                tile_grid_size=cfg.preprocessing.clahe_tile_grid_size,
            )
            logger.info(f"{filename}: Applied CLAHE enhancement")

        # assuming that mask files are all .npz files
        npz_path = jpg_path.with_suffix(".npz")

        # load the mask arr
        npz_obj = np.load(npz_path)
        mask_arr = npz_obj[npz_obj.files[0]]

        # GSD-based resampling if target GSD is specified
        if target_gsd:
            geojson_path = jpg_path.with_suffix(".geojson")
            if geojson_path.exists():
                try:
                    source_gsd = calculate_gsd_from_geojson(
                        geojson_path, img_arr.shape[1], img_arr.shape[0]
                    )
                    gsd_diff = abs(source_gsd - target_gsd) / target_gsd

                    if gsd_diff > gsd_tolerance:
                        img_arr, mask_arr = resample_to_target_gsd(
                            img_arr, mask_arr, source_gsd, target_gsd
                        )
                    else:
                        logger.info(
                            f"{filename}: GSD {source_gsd:.2f} cm/px within tolerance of target "
                            f"{target_gsd} cm/px (diff: {gsd_diff*100:.1f}%), skipping resampling"
                        )
                except (KeyError, json.JSONDecodeError) as e:
                    logger.warning(f"{filename}: Could not calculate GSD from geojson: {e}")
            else:
                logger.warning(f"{filename}: No geojson found, skipping GSD resampling")

        logger.info(f"{filename} is going to be sliced to patches")

        # call create_patch func for a single pair
        create_patches(
            img_arr,
            mask_arr,
            cfg.preprocessing.patch_size,
            cfg.paths.output_dir,
            filename,
            cfg.preprocessing.background_fraction,
            cfg.preprocessing.classes_to_background,
            cfg.preprocessing.fixed_class_order,
        )

    mask_dir = Path(cfg.paths.output_dir) / "mask_patches"
    if mask_dir.exists():
        summarize_mask_classes(mask_dir, cfg.model.classes)
    else:
        logger.warning("Mask directory %s not found; skipping class summary", mask_dir)

    mask_dir = Path(cfg.paths.output_dir) / "mask_patches"
    if mask_dir.exists():
        summarize_mask_classes(mask_dir, cfg.model.classes)
    else:
        logger.warning("Mask directory %s not found; skipping class summary", mask_dir)


if __name__ == "__main__":
    import argparse
    import logging
    import sys
    from pathlib import Path

    import numpy as np
    import pydantic
    import yaml
    from PIL import Image

    from detect_electrical_utility_from_satellite_images.config import AppConfig
    from detect_electrical_utility_from_satellite_images.utils.file_utils import (
        drop_jpg_paths_with_no_npz_pair,
    )
    from detect_electrical_utility_from_satellite_images.utils.logging_config import (
        setup_logger,
    )

    # set up parser
    parser = argparse.ArgumentParser()

    # add config path arg
    parser.add_argument(
        "-cfg",
        "--config",
        required=True,
        help="relative path to the config file. no need to write the extension ie .yaml",
    )

    # parse the args
    args = parser.parse_args()

    # get the relative path
    CFG_PATH = Path.cwd() / f"{args.config}.yaml"

    # load the config dict
    with open(CFG_PATH) as f:
        cfg_dict = yaml.safe_load(f)

    # get the configs
    try:
        cfg = AppConfig(**cfg_dict)
    except pydantic.ValidationError as e:
        print(f"FATAL. CONFIG FAILED: {e}")
        sys.exit(1)

    # logger setup
    setup_logger(
        cfg.paths.logging_dir_name,
        cfg.logging.logger_lvl,
        cfg.logging.console_handler_lvl,
        cfg.logging.file_handler_lvl,
    )

    # get logger
    logger = logging.getLogger(__name__)

    # now we can JUST use the logger like this
    # logger.debug("lvl 1 whaat")
    # logger.warning("there is a warning dude")

    # get all the jpg paths
    jpg_paths = sorted(cfg.paths.data_dir.glob("*/*.jpg"))

    # drop paths that dont have npz pair
    jpg_paths = drop_jpg_paths_with_no_npz_pair(jpg_paths)

    for jpg_path in jpg_paths:
        # get filename
        filename = jpg_path.stem

        # load img arr
        PIL_obj = Image.open(jpg_path)
        img_arr = np.array(PIL_obj)

        # assuming that mask files are all .npz files
        npz_path = jpg_path.with_suffix(".npz")

        # load the mask arr
        npz_obj = np.load(npz_path)
        mask_arr = npz_obj[npz_obj.files[0]]

        logger.info(f"{filename} is going to be sliced to patches")

        # call create_patch func for a single pair
        create_patches(
            img_arr,
            mask_arr,
            cfg.preprocessing.patch_size,
            cfg.paths.output_dir,
            filename,
            cfg.preprocessing.background_fraction,
            cfg.preprocessing.classes_to_background,
            cfg.preprocessing.fixed_class_order,
        )
