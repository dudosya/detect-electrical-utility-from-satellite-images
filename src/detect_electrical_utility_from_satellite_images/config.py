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

class LoggingConfig(pydantic.BaseModel):
    logger_lvl: typing.Literal['debug','info','warning','error','critical']
    console_handler_lvl: typing.Literal['debug','info','warning','error','critical']
    file_handler_lvl: typing.Literal['debug','info','warning','error','critical']

class AppConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        frozen=True # we dont want config changes during runtime
    )
    preprocessing: PreprocessConfig
    paths: PathsConfig
    logging: LoggingConfig
    model: ModelConfig
