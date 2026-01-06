"""
Preprocessing configuration.
"""

import typing

import pydantic


class PreprocessConfig(pydantic.BaseModel):
    """Configuration for data preprocessing."""

    patch_size: typing.Annotated[int, pydantic.Field(ge=128)]
    """Size of image patches (must be ≥ 128)."""

    background_fraction: typing.Annotated[float, pydantic.Field(le=1.0, ge=0.00001)]
    """Fraction of background pixels to include (between 0.00001 and 1.0)."""

    classes_to_background: list[int] | None = None
    """List of class indices to remap to background (0). E.g., [1] to treat LINE as background."""

    fixed_class_order: list[int] | None = None
    """Optional ordered list of source class ids to keep (excluding background). When set,
    remapping uses this order for consistent class indices across all masks."""

    target_gsd_cm: typing.Annotated[float, pydantic.Field(gt=0)] | None = None
    """Target Ground Sample Distance in cm/pixel. Images will be resampled to match this.
    If None, no resampling is performed. Recommended: 15.0 for this dataset."""

    gsd_tolerance: typing.Annotated[float, pydantic.Field(gt=0, le=0.5)] = 0.05
    """Tolerance for GSD matching (0.05 = 5%). Images within tolerance skip resampling."""

    apply_clahe: bool = False
    """Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
    Recommended for images with thin structures like power lines. Requires opencv-python."""

    clahe_clip_limit: typing.Annotated[float, pydantic.Field(gt=0, le=10)] = 2.0
    """CLAHE contrast limiting threshold. Higher values = more contrast. Typical range: 2.0-4.0."""

    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    """CLAHE tile grid size for local histogram equalization. Smaller = more local adaptation."""
