import pydantic
import typing
from pathlib import Path
import pydantic_settings

def find_project_root():
    root_dir = Path(__file__).resolve()
    
    while not (root_dir / "pyproject.toml").exists():
        if root_dir != root_dir.parent:
            root_dir = root_dir.parent
        else:
            raise ValueError("Folder containing pyproject.toml is not found")
    # should i put logging here?
    # like logging.debug: we found the root dir. here it is type thing?
    return root_dir

PROJECT_ROOT = find_project_root()
    
class PreprocessConfig(pydantic.BaseModel):
    patch_size: typing.Annotated[int, pydantic.Field(ge=128)]
    background_fraction: typing.Annotated[float, pydantic.Field(le=1.0, ge=0.00001)]
    
class PathsConfig(pydantic.BaseModel):
    # these are relative folder names of the dirs
    output_dir: Path
    data_dir: Path
    logging_dir_name: Path
    
    @pydantic.field_validator('*',mode='after')
    @classmethod
    def convert_to_absolute_path(cls, v: typing.Any | Path) -> Path:
        if Path(v).is_absolute():
            return v
        else:
            # these are meant to be at the project root level right?
            # not in the src/project_name/v or something?
            return (PROJECT_ROOT / v).resolve()
    
    
    
class ModelConfig(pydantic.BaseModel):
    architecture: typing.Literal['unet', 'deeplabv3', 'fpn']
    encoder_name: typing.Literal['resnet18', 'resnet34', 'resnet50', 'efficientnet-b0', 'mobilenet_v2']
    encoder_weights: typing.Literal['imagenet', None]
    in_channels: typing.Annotated[int, pydantic.Field(ge=3, le=4)]
    classes: typing.Annotated[int, pydantic.Field(ge=2)]
    learning_rate: typing.Annotated[float, pydantic.Field(gt=0.0)]
    batch_size: typing.Annotated[int, pydantic.Field(ge=1)]
    epochs: typing.Annotated[int, pydantic.Field(ge=1)]
    device: typing.Literal['cpu', 'cuda']

class WandbConfig(pydantic.BaseModel):
    """Weights & Biases configuration."""
    project: str = "detect-electrical-utility"
    entity: typing.Optional[str] = None
    run_name: str = "experiment_{timestamp}"
    tags: typing.List[str] = pydantic.Field(default_factory=lambda: ["unet", "mobilenet_v2", "1024_patches", "baseline"])
    log_model: bool = True
    save_code: bool = True
    notes: typing.Optional[str] = None
    group: typing.Optional[str] = None

class MetricsConfig(pydantic.BaseModel):
    """Metrics tracking configuration."""
    track: typing.List[typing.Literal["iou", "dice", "accuracy", "precision", "recall", "f1"]] = pydantic.Field(
        default_factory=lambda: ["iou", "dice", "accuracy", "precision", "recall", "f1"]
    )
    iou_average: typing.Literal["macro", "micro", "weighted"] = "macro"
    dice_average: typing.Literal["macro", "micro", "weighted"] = "macro"
    f1_average: typing.Literal["macro", "micro", "weighted"] = "macro"
    log_samples: int = 4
    sample_indices: typing.List[int] = pydantic.Field(default_factory=lambda: [0, 1, 2, 3])

class TrainingConfig(pydantic.BaseModel):
    """Enhanced training configuration."""
    experiment_name: str = "baseline"  # Used for folder naming and W&B grouping
    seed: int = 42
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    early_stopping_patience: int = 10
    lr_scheduler: typing.Literal["cosine", "reduce_on_plateau", "step", "none"] = "cosine"
    lr_scheduler_params: typing.Dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {
            "T_max": 50,
            "eta_min": 1e-6,
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
            "step_size": 10,
            "gamma": 0.1,
        }
    )
    optimizer: typing.Literal["adam", "sgd", "adamw"] = "adam"
    optimizer_params: typing.Dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
        }
    )
    class_weights: typing.Optional[typing.List[float]] = None

class AugmentationConfig(pydantic.BaseModel):
    """Data augmentation configuration."""
    enabled: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    rotation: int = 30
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    gaussian_blur: bool = True
    blur_kernel_size: int = 3

class CheckpointsConfig(pydantic.BaseModel):
    """Checkpoint configuration."""
    save_best: bool = True
    save_last: bool = True
    metric_to_monitor: str = "val_loss"
    mode: typing.Literal["max", "min"] = "min"

class LoggingConfig(pydantic.BaseModel):
    logger_lvl: typing.Literal['debug','info','warning','error','critical']
    console_handler_lvl: typing.Literal['debug','info','warning','error','critical']
    file_handler_lvl: typing.Literal['debug','info','warning','error','critical']

class AppConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        frozen=False  # Changed from True to allow path formatting
    )
    preprocessing: PreprocessConfig
    paths: PathsConfig
    logging: LoggingConfig
    model: ModelConfig
    wandb: WandbConfig = pydantic.Field(default_factory=WandbConfig)
    metrics: MetricsConfig = pydantic.Field(default_factory=MetricsConfig)
    training: TrainingConfig = pydantic.Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = pydantic.Field(default_factory=AugmentationConfig)
    checkpoints: CheckpointsConfig = pydantic.Field(default_factory=CheckpointsConfig)
    
    @pydantic.model_validator(mode='after')
    def format_paths(self) -> "AppConfig":
        """Format path placeholders with actual values after validation."""
        # Format output_dir with patch_size and experiment_name
        output_dir_str = str(self.paths.output_dir)
        if '{patch_size}' in output_dir_str or '{experiment_name}' in output_dir_str:
            formatted_output_dir = output_dir_str.format(
                patch_size=self.preprocessing.patch_size,
                experiment_name=self.training.experiment_name
            )
            # Update the output_dir path
            self.paths.output_dir = Path(formatted_output_dir)
        
        # Format logging_dir_name with experiment_name
        logging_dir_str = str(self.paths.logging_dir_name)
        if '{experiment_name}' in logging_dir_str:
            formatted_logging_dir = logging_dir_str.format(
                experiment_name=self.training.experiment_name
            )
            # Update the logging_dir_name path
            self.paths.logging_dir_name = Path(formatted_logging_dir)
        
        # Re-run the field validator to convert to absolute paths
        # Create a new PathsConfig instance with updated values
        updated_paths = PathsConfig(
            output_dir=self.paths.output_dir,
            data_dir=self.paths.data_dir,
            logging_dir_name=self.paths.logging_dir_name
        )
        
        # Use object.__setattr__ to bypass frozen instance restriction
        object.__setattr__(self, 'paths', updated_paths)
        
        return self
