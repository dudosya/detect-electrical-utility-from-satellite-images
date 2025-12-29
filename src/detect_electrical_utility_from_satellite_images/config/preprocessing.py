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
