"""
Data augmentation configuration.
"""

import pydantic


class AugmentationConfig(pydantic.BaseModel):
    """Data augmentation configuration."""

    enabled: bool = True
    """Whether to enable data augmentation."""

    horizontal_flip: float = 0.5
    """Probability of horizontal flip (0.0 to 1.0)."""

    vertical_flip: float = 0.5
    """Probability of vertical flip (0.0 to 1.0)."""

    rotation: int = 30
    """Maximum rotation angle in degrees."""

    brightness: float = 0.2
    """Brightness adjustment factor (0.0 to 1.0)."""

    contrast: float = 0.2
    """Contrast adjustment factor (0.0 to 1.0)."""

    saturation: float = 0.2
    """Saturation adjustment factor (0.0 to 1.0)."""

    gaussian_blur: bool = True
    """Whether to apply Gaussian blur."""

    blur_kernel_size: int = 3
    """Kernel size for Gaussian blur (must be odd)."""
