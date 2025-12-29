from __future__ import annotations

import time

start_import = time.perf_counter()

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

import typer

end_import = time.perf_counter()

# init typer app
app = typer.Typer()

if TYPE_CHECKING:
    import logging

    from detect_electrical_utility_from_satellite_images.config import AppConfig
    from detect_electrical_utility_from_satellite_images.utils.logging_config import (
        setup_logger,
    )


@dataclass
class State:
    app_config: AppConfig
    start_time: float


@app.callback()
def manage_internal_state(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            help="relative path to the config file. no need to write the extension ie .yaml"
        ),
    ] = "config",
):
    from detect_electrical_utility_from_satellite_images.utils.logging_config import (
        setup_logger,
    )

    # start timer
    start_time = time.perf_counter()
    cfg = get_cfg(config)

    # logger setup
    setup_logger(
        cfg.paths.logging_dir_name,
        cfg.logging.logger_lvl,
        cfg.logging.console_handler_lvl,
        cfg.logging.file_handler_lvl,
    )

    state_instance = State(app_config=cfg, start_time=start_time)

    ctx.obj = state_instance


def get_cfg(config: str):
    import sys
    from pathlib import Path

    import pydantic
    import yaml

    from detect_electrical_utility_from_satellite_images.config import AppConfig

    CFG_PATH = Path(config).with_suffix(".yaml")

    # load the config dict
    with open(CFG_PATH) as f:
        cfg_dict = yaml.safe_load(f)

    # get the configs
    try:
        cfg = AppConfig(**cfg_dict)
    except pydantic.ValidationError as e:
        print(f"FATAL. CONFIG FAILED: {e}")
        sys.exit(1)

    return cfg


@app.command()
def preprocess(
    ctx: typer.Context,
):
    import logging

    from detect_electrical_utility_from_satellite_images.preprocess import (
        jpg_paths_to_patches,
    )

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
    logger.info("Done")
    logger.info(f"Import Time: {end_import - start_import} sec")
    logger.info(f"Running Time: {end_time - start_time} sec")


@app.command()
def test_dataset(
    ctx: typer.Context,
):
    import logging

    from detect_electrical_utility_from_satellite_images.dataset import dataset_tester

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
    logger.info("Done")
    logger.info(f"Import Time: {end_import - start_import} sec")
    logger.info(f"Running Time: {end_time - start_time} sec")


@app.command()
def train(
    ctx: typer.Context,
    use_real_data: Annotated[
        bool,
        typer.Option("--real-data", help="Use real data instead of mock data"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level: debug, info, warning, error"),
    ] = "info",
):
    """Train a segmentation model using the specified configuration.

    Examples:
        # Train with default config and mock data
        uv run main train

        # Train with real data
        uv run main train --real-data

        # Train with real data and debug logging
        uv run main train --real-data --log-level debug
    """
    import logging

    from detect_electrical_utility_from_satellite_images.train import train_model

    state: State = ctx.obj
    cfg = state.app_config
    start_time = state.start_time

    # Override logging level if specified
    if log_level != "info":
        import logging

        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        for handler in logging.getLogger().handlers:
            handler.setLevel(getattr(logging, log_level.upper()))

    # training program
    train_model(cfg, use_mock_data=not use_real_data)

    # get logger
    logger = logging.getLogger(__name__)

    # end timer
    end_time = time.perf_counter()

    # final log
    logger.info("Training completed successfully")
    logger.info(f"Import Time: {end_import - start_import:.2f} sec")
    logger.info(f"Running Time: {end_time - start_time:.2f} sec")


# Standalone train command for direct invocation (used by pyproject.toml train script)
def train_standalone(
    config: str = typer.Option(
        "config", help="Relative path to config file (without .yaml extension)"
    ),
    real_data: bool = typer.Option(
        False, "--real-data", help="Use real data instead of mock"
    ),
    log_level: str = typer.Option(
        "info", "--log-level", help="Logging level: debug, info, warning, error"
    ),
):
    """Standalone train command for direct invocation.

    This is used by the 'train' script entry point in pyproject.toml.
    For more advanced usage, use 'uv run main train' instead.

    Examples:
        # Train with default config and mock data
        uv run train

        # Train with real data
        uv run train --real-data

        # Train with custom config
        uv run train --config config_experiment
    """
    import logging
    import sys
    from pathlib import Path

    import pydantic
    import yaml

    from detect_electrical_utility_from_satellite_images.config import AppConfig
    from detect_electrical_utility_from_satellite_images.train import train_model
    from detect_electrical_utility_from_satellite_images.utils.logging_config import (
        setup_logger,
    )

    # Setup basic logging first
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    try:
        # Load config
        config_path = Path(config).with_suffix(".yaml")
        logger.info(f"Loading config from {config_path}")

        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)

        try:
            cfg = AppConfig(**cfg_dict)
        except pydantic.ValidationError as e:
            logger.error(f"Config validation failed: {e}")
            raise typer.Exit(code=1)

        # Setup project logging
        setup_logger(
            cfg.paths.logging_dir_name,
            cfg.logging.logger_lvl,
            cfg.logging.console_handler_lvl,
            cfg.logging.file_handler_lvl,
        )

        logger.info(f"Starting training with config: {config_path}")
        logger.info(f"Using {'real' if real_data else 'mock'} data")

        # Train
        train_model(cfg, use_mock_data=not real_data)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(code=1)


# Create standalone app for train command
train_app = typer.Typer()
train_app.command()(train_standalone)

if __name__ == "__main__":
    app()
