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
    dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Dropout rate for MC Dropout uncertainty (0 disables)",
    )
    mc_dropout_enabled: bool = Field(
        default=False,
        description="Enable MC Dropout uncertainty during validation",
    )
    mc_dropout_passes: int = Field(
        default=10,
        ge=1,
        description="Number of MC Dropout passes",
    )
    mc_dropout_log_samples: int = Field(
        default=4,
        ge=1,
        description="Number of samples to log to W&B",
    )
    line_class_value: int = Field(
        default=3,
        ge=0,
        description="Class value for line pixels in multiclass masks",
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


class InferenceConfig(BaseModel):
    """Configuration for inference-time tiling."""

    patch_size: int = Field(
        default=500,
        description="Patch size in pixels for tiled inference",
    )
    patch_stride: int = Field(
        default=500,
        description="Stride in pixels for tiled inference",
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

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

    split_level: Literal["patch", "tile"] = Field(
        default="tile",
        description="How to split train/val: patch-level or tile-level (recommended)",
    )
    split_strategy: Literal["within_region", "region_holdout"] = Field(
        default="within_region",
        description="Split strategy for generalization evaluation",
    )
    holdout_regions: list[str] = Field(
        default_factory=list,
        description="Regions reserved for validation when using region_holdout",
    )
    persist_split: bool = Field(
        default=True,
        description="Persist the computed split to disk for reproducible eval/infer",
    )
    split_file_name: str = Field(
        default="split_tiles.json",
        description="Split file name stored under each region patches directory",
    )
    min_val_tiles: int = Field(
        default=1,
        ge=0,
        description="Minimum number of validation tiles (when tile-level splitting)",
    )
    checkpoint_every_n_epochs: int = Field(
        default=1,
        ge=0,
        description="Checkpoint cadence in epochs (0 disables)",
    )
    checkpoint_every_n_train_steps: int = Field(
        default=0,
        ge=0,
        description="Checkpoint cadence in steps (0 disables)",
    )
    resume_checkpoint_tower: Path | None = Field(
        default=None,
        description="Path to resume tower training checkpoint",
    )
    resume_checkpoint_line: Path | None = Field(
        default=None,
        description="Path to resume line training checkpoint",
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
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
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
