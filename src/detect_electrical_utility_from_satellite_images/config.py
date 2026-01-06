"""Configuration management using Pydantic for type-safe config loading."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PreprocessingConfig(BaseModel):
    """Configuration for the preprocessing pipeline."""

    patch_size: int = Field(default=500, description="Size of image patches in pixels")
    background_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of background-only patches to keep",
    )


class PathsConfig(BaseModel):
    """Configuration for file paths."""

    output_dir: Path = Field(
        default=Path("preprocessed_data/patches"),
        description="Directory for preprocessed output",
    )
    data_dir: Path = Field(
        default=Path("raw_data"),
        description="Directory containing raw data",
    )
    logging_dir_name: str = Field(
        default="logs",
        description="Directory name for log files",
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    logger_lvl: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="debug",
        description="Root logger level",
    )
    console_handler_lvl: Literal["debug", "info", "warning", "error", "critical"] = (
        Field(
            default="info",
            description="Console handler level",
        )
    )
    file_handler_lvl: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="debug",
        description="File handler level",
    )


class TowerDetectionConfig(BaseModel):
    """Configuration for Stage 1: Tower Detection (Faster R-CNN)."""

    backbone: str = Field(default="inception_v2", description="Backbone architecture")
    anchor_areas: list[int] = Field(
        default=[100, 625, 2500, 10000, 40000],
        description="Anchor box areas (10², 25², 50², 100², 200²)",
    )
    anchor_ratios: list[float] = Field(
        default=[0.5, 1.0, 2.0],
        description="Anchor box aspect ratios",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for detections",
    )
    nms_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS",
    )


class LineSegmentationConfig(BaseModel):
    """Configuration for Stage 2: Line Segmentation (StackNetMTL)."""

    line_width_train: int = Field(
        default=30,
        description="Line width in pixels during training",
    )
    line_width_inference: int = Field(
        default=9,
        description="Line width in pixels during graph inference",
    )


class GraphInferenceConfig(BaseModel):
    """Configuration for Stage 3: Graph Inference."""

    max_distance_m: float = Field(
        default=600.0,
        description="Maximum distance (meters) between connected towers",
    )
    connectivity_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum connectivity score (gamma) for edge creation",
    )
    resolution_m_per_px: float = Field(
        default=0.3,
        description="Image resolution in meters per pixel",
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    iterations: int = Field(default=50000, description="Total training iterations")
    max_epochs: int = Field(default=50, description="Maximum training epochs")
    batch_size: int = Field(default=5, description="Batch size")
    initial_lr: float = Field(default=3e-3, description="Initial learning rate")
    lr_decay_factor: float = Field(default=0.1, description="LR decay factor")
    lr_decay_steps: int = Field(
        default=10000, description="Steps between LR decay"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    val_split: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Validation split fraction"
    )
    num_workers: int = Field(
        default=0, description="DataLoader workers (0 for Windows compatibility)"
    )
    tensor_core_precision: Literal["highest", "high", "medium"] = Field(
        default="medium",
        description="Float32 matmul precision for Tensor Cores (medium=faster)",
    )


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases experiment tracking."""

    enabled: bool = Field(default=True, description="Enable W&B logging")
    project: str = Field(default="gridtracer", description="W&B project name")
    entity: str | None = Field(default=None, description="W&B team/user entity")
    tags: list[str] = Field(default_factory=list, description="Tags for the run")
    notes: str = Field(default="", description="Notes for the run")
    offline: bool = Field(default=False, description="Run W&B in offline mode")


class Config(BaseSettings):
    """Root configuration for GridTracer pipeline."""

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tower_detection: TowerDetectionConfig = Field(default_factory=TowerDetectionConfig)
    line_segmentation: LineSegmentationConfig = Field(
        default_factory=LineSegmentationConfig
    )
    graph_inference: GraphInferenceConfig = Field(default_factory=GraphInferenceConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Config instance with loaded values.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)


def load_config(config_path: Path | str | None = None) -> Config:
    """Load configuration from default or specified path.

    Args:
        config_path: Optional path to config file. Defaults to 'config.yaml'.

    Returns:
        Loaded Config instance.
    """
    if config_path is None:
        config_path = Path("config.yaml")

    return Config.from_yaml(config_path)
