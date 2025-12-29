"""
Weights & Biases configuration.
"""

import typing

import pydantic


class WandbConfig(pydantic.BaseModel):
    """Weights & Biases configuration."""

    project: str = "detect-electrical-utility"
    """W&B project name."""

    entity: str | None = None
    """W&B entity (username or team name)."""

    run_name: str = "experiment_{timestamp}"
    """Run name (supports {timestamp} placeholder)."""

    tags: list[str] = pydantic.Field(
        default_factory=lambda: ["unet", "mobilenet_v2", "1024_patches", "baseline"],
    )
    """Tags for the W&B run."""

    log_model: bool = True
    """Whether to log model checkpoints to W&B."""

    save_code: bool = True
    """Whether to save code to W&B."""

    notes: str | None = None
    """Notes for the W&B run."""

    group: str | None = None
    """Group for the W&B run (for grouping related experiments)."""
