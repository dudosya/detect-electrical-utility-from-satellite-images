"""Dataset classes for line segmentation training."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from detect_electrical_utility_from_satellite_images.logging_config import get_logger


def dilate_line_mask(
    mask: NDArray[np.uint8],
    line_class: int = 3,
    target_width: int = 30,
) -> NDArray[np.float32]:
    """Create dilated binary line mask from multiclass mask.

    The paper specifies 30px-wide line segments for training targets.
    This function extracts the line class and dilates to target width.

    Args:
        mask: Multiclass segmentation mask (H, W) with class labels.
        line_class: Class label for power lines (default 3).
        target_width: Target line width in pixels (paper: 30 for training).

    Returns:
        Binary mask (H, W) with values 0.0 or 1.0 for line segmentation.
    """
    from scipy import ndimage

    # Extract line pixels
    line_mask = (mask == line_class).astype(np.uint8)

    if not np.any(line_mask):
        return np.zeros(mask.shape, dtype=np.float32)

    # Dilate to target width using disk structuring element
    # The radius should be half the target width
    radius = target_width // 2
    if radius > 0:
        # Create disk structuring element
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        struct = x * x + y * y <= radius * radius

        # Dilate the line mask
        dilated = ndimage.binary_dilation(line_mask, structure=struct)
        return dilated.astype(np.float32)

    return line_mask.astype(np.float32)


class LineSegmentationDataset(Dataset[dict[str, Any]]):
    """Dataset for line segmentation training.

    Loads preprocessed image patches and creates binary line masks
    with configurable dilation (paper uses 30px width for training).
    """

    def __init__(
        self,
        patches_dir: Path,
        line_width: int = 30,
        transform: Any | None = None,
        include_empty: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            patches_dir: Directory containing 'images/' and 'masks/' subdirectories.
            line_width: Target line width for dilation (paper: 30 for training).
            transform: Optional torchvision transforms to apply.
            include_empty: Whether to include patches with no lines.
        """
        self.patches_dir = Path(patches_dir)
        self.images_dir = self.patches_dir / "images"
        self.masks_dir = self.patches_dir / "masks"
        self.line_width = line_width
        self.transform = transform
        self.include_empty = include_empty

        # Discover all image files
        self.image_files = sorted(self.images_dir.glob("*.png"))

        # Filter to only patches with lines if requested
        if not include_empty:
            self.image_files = self._filter_patches_with_lines()

        self.log = get_logger()
        self.log.info(
            "line_dataset_initialized",
            patches_dir=str(patches_dir),
            line_width=line_width,
            num_samples=len(self.image_files),
        )

    def _filter_patches_with_lines(self) -> list[Path]:
        """Filter to only patches containing line annotations."""
        filtered = []
        for img_path in self.image_files:
            mask_path = self.masks_dir / img_path.name
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                # Check if mask contains lines (class 3)
                if np.any(mask == 3):
                    filtered.append(img_path)
        return filtered

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - image: (3, H, W) tensor normalized to [0, 1]
                - mask: (1, H, W) tensor binary line mask
                - image_path: Path to original image file
        """
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        # Load image
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # Load mask and create dilated binary line mask
        mask_np = np.array(Image.open(mask_path))
        line_mask = dilate_line_mask(
            mask_np,
            line_class=3,
            target_width=self.line_width,
        )

        # Apply transforms if provided (for augmentation)
        if self.transform is not None:
            # Transforms should handle both image and mask
            transformed = self.transform(image=image_np, mask=line_mask)
            image_np = transformed["image"]
            line_mask = transformed["mask"]

        # Convert to tensors
        # Image: normalize to [0, 1] and convert to (C, H, W)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        # Mask: add channel dimension (1, H, W)
        mask_tensor = torch.from_numpy(line_mask).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(img_path),
        }


def line_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Custom collate function for line segmentation.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary with stacked tensors.
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    paths = [item["image_path"] for item in batch]

    return {
        "images": images,
        "masks": masks,
        "image_paths": paths,
    }
