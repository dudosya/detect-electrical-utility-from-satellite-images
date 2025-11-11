import pydantic
import typing
from pathlib import Path
    
class PreprocessConfig(pydantic.BaseModel):
    patch_size: typing.Annotated[int, pydantic.Field(ge=128)]
    background_fraction: typing.Annotated[float, pydantic.Field(le=1.0, ge=0.00001)]
    
class PathsConfig(pydantic.BaseModel):
    # these are relative folder names of the dirs
    output_dir: Path
    data_dir: Path
    logging_dir_name: Path
    
    # i dont really understand this part. 
    # look into it later maybe
    @pydantic.field_validator('*',mode='before')
    @classmethod
    def convert_to_absolute_path(cls, v: typing.Any) -> Path:
        if isinstance(v,str):
            return (Path.cwd() / v).resolve()
        return v
    
    
    
class LoggingConfig(pydantic.BaseModel):
    logger_lvl: typing.Literal['debug','info','warning','error','critical']
    console_handler_lvl: typing.Literal['debug','info','warning','error','critical']
    file_handler_lvl: typing.Literal['debug','info','warning','error','critical']

class AppConfig(pydantic.BaseModel):
    preprocessing: PreprocessConfig
    paths: PathsConfig
    logging: LoggingConfig