"""Dataset classes for tower detection training."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from detect_electrical_utility_from_satellite_images.logging_config import get_logger


def extract_bounding_boxes_from_mask(
    mask: NDArray[np.uint8],
    tower_class: int = 1,
    other_tower_class: int = 2,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Extract bounding boxes from a segmentation mask.

    Args:
        mask: Segmentation mask (H, W) with class labels.
        tower_class: Class label for towers.
        other_tower_class: Class label for other towers.

    Returns:
        Tuple of (boxes, labels) where:
            - boxes: (N, 4) array of [x1, y1, x2, y2] coordinates
            - labels: (N,) array of class labels (1 for all towers)
    """
    boxes = []
    labels = []

    # Find connected components for both tower classes
    from scipy import ndimage

    for class_id in [tower_class, other_tower_class]:
        binary_mask = (mask == class_id).astype(np.uint8)

        if not np.any(binary_mask):
            continue

        # Label connected components
        labeled, num_features = ndimage.label(binary_mask)

        for i in range(1, num_features + 1):
            component_mask = labeled == i
            rows = np.any(component_mask, axis=1)
            cols = np.any(component_mask, axis=0)

            if not np.any(rows) or not np.any(cols):
                continue

            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]

            y1, y2 = y_indices[0], y_indices[-1]
            x1, x2 = x_indices[0], x_indices[-1]

            # Skip tiny boxes (noise)
            if (x2 - x1) < 3 or (y2 - y1) < 3:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # All towers get label 1

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


class TowerDetectionDataset(Dataset[dict[str, Any]]):
    """Dataset for tower detection training.

    Loads preprocessed image patches and extracts bounding boxes from masks.
    """

    def __init__(
        self,
        patches_dir: Path,
        transform: Any | None = None,
        include_background: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            patches_dir: Directory containing 'images/' and 'masks/' subdirectories.
            transform: Optional torchvision transforms to apply.
            include_background: Whether to include patches with no towers.
        """
        self.patches_dir = Path(patches_dir)
        self.images_dir = self.patches_dir / "images"
        self.masks_dir = self.patches_dir / "masks"
        self.transform = transform
        self.include_background = include_background

        # Discover all image files
        self.image_files = sorted(self.images_dir.glob("*.png"))

        # Filter to only patches with towers if requested
        if not include_background:
            self.image_files = self._filter_patches_with_towers()

        self.log = get_logger()
        self.log.info(
            "dataset_initialized",
            patches_dir=str(patches_dir),
            num_samples=len(self.image_files),
        )

    def _filter_patches_with_towers(self) -> list[Path]:
        """Filter to only patches containing tower annotations."""
        filtered = []
        for img_path in self.image_files:
            mask_path = self.masks_dir / img_path.name
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                # Check if mask contains towers (class 1 or 2)
                if np.any((mask == 1) | (mask == 2)):
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
                - image: (3, H, W) tensor
                - boxes: (N, 4) tensor of bounding boxes
                - labels: (N,) tensor of class labels
                - image_id: tensor with the image index
        """
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        # Load image
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # Load mask and extract boxes
        mask = np.array(Image.open(mask_path))
        boxes, labels = extract_bounding_boxes_from_mask(mask)

        # Apply transforms if any
        if self.transform is not None:
            transformed = self.transform(image=image_np, bboxes=boxes, labels=labels)
            image_np = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32)
            labels = np.array(transformed["labels"], dtype=np.int64)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        # Handle empty boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.from_numpy(boxes)
            labels = torch.from_numpy(labels)

        return {
            "image": image_tensor,
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }


def collate_fn(batch: list[dict[str, Any]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Custom collate function for detection datasets.

    Args:
        batch: List of samples from __getitem__.

    Returns:
        Tuple of (images, targets) where:
            - images: List of (3, H, W) tensors
            - targets: List of dicts with 'boxes', 'labels', 'image_id'
    """
    images = [item["image"] for item in batch]
    targets = [
        {
            "boxes": item["boxes"],
            "labels": item["labels"],
            "image_id": item["image_id"],
        }
        for item in batch
    ]
    return images, targets
