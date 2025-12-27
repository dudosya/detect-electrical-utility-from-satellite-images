from __future__ import annotations
import time

start_import = time.perf_counter()

import typer
from typing import Annotated, TYPE_CHECKING
from dataclasses import dataclass




end_import = time.perf_counter()

# init typer app
app = typer.Typer()

if TYPE_CHECKING:
    from detect_electrical_utility_from_satellite_images.config import AppConfig
    from detect_electrical_utility_from_satellite_images.utils.logging_config import setup_logger
    import logging

@dataclass
class State:
    app_config: AppConfig
    start_time: float
    
    

@app.callback()
def manage_internal_state(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(help="relative path to the config file. no need to write the extension ie .yaml")] = "config"
):
    from detect_electrical_utility_from_satellite_images.utils.logging_config import setup_logger

    # start timer
    start_time = time.perf_counter()
    cfg = get_cfg(config)
    
    # logger setup
    setup_logger(cfg.paths.logging_dir_name, cfg.logging.logger_lvl, cfg.logging.console_handler_lvl, cfg.logging.file_handler_lvl)
    
    state_instance = State(app_config=cfg, start_time=start_time)
    
    ctx.obj = state_instance


def get_cfg(config: str):
    import yaml
    from pathlib import Path
    from detect_electrical_utility_from_satellite_images.config import AppConfig
    import pydantic
    import sys
    
    CFG_PATH = Path(config).with_suffix(".yaml")
    
    #load the config dict
    with open(CFG_PATH,'r') as f:
        cfg_dict = yaml.safe_load(f)
        
    #get the configs
    try:
        cfg = AppConfig(**cfg_dict)
    except pydantic.ValidationError as e:
        print(f"FATAL. CONFIG FAILED: {e}")
        sys.exit(1)
        
    return cfg

@app.command()
def preprocess(
    ctx: typer.Context
):
    from detect_electrical_utility_from_satellite_images.preprocess import jpg_paths_to_patches
    import logging
    
    state: State = ctx.obj
    cfg = state.app_config
    start_time = state.start_time
    
    # the bulk of the logic goes on here
    jpg_paths_to_patches(cfg)
    
    # end timer
    end_time = time.perf_counter()
    
    # get logger
    logger = logging.getLogger(__name__)
    
    # final log
    logger.info(f"Done")
    logger.info(f"Import Time: {end_import-start_import} sec")
    logger.info(f"Running Time: {end_time-start_time} sec")

    
@app.command()
def test_dataset(
    ctx: typer.Context,
):
    from detect_electrical_utility_from_satellite_images.dataset import dataset_tester
    import logging
    
    state: State = ctx.obj
    cfg = state.app_config
    start_time = state.start_time
    # dataset tester program
    dataset_tester(cfg)
    
    # get logger
    logger = logging.getLogger(__name__)
    
    # end timer
    end_time = time.perf_counter()
    
    # final log
    logger.info(f"Done")
    logger.info(f"Import Time: {end_import-start_import} sec")
    logger.info(f"Running Time: {end_time-start_time} sec")

@app.command()
def train(
    ctx: typer.Context,
    use_real_data: Annotated[
        bool,
        typer.Option("--real-data", help="Use real data instead of mock data")
    ] = False
):
    from detect_electrical_utility_from_satellite_images.train import train_model
    import logging
    
    state: State = ctx.obj
    cfg = state.app_config
    start_time = state.start_time
    
    # training program
    train_model(cfg, use_mock_data=not use_real_data)
    
    # get logger
    logger = logging.getLogger(__name__)
    
    # end timer
    end_time = time.perf_counter()
    
    # final log
    logger.info(f"Done")
    logger.info(f"Import Time: {end_import-start_import} sec")
    logger.info(f"Running Time: {end_time-start_time} sec")

if __name__ == "__main__":
    app()
