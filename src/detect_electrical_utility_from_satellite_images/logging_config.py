"""Structured logging configuration using structlog."""

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(
    logger_level: str = "debug",
    console_level: str = "info",
    file_level: str = "debug",
    log_dir: Path | None = None,
    log_filename: str = "gridtracer.log",
) -> structlog.stdlib.BoundLogger:
    """Configure structured logging for the application.

    Sets up both console (human-readable) and file (JSON) logging outputs.

    Args:
        logger_level: Root logger level.
        console_level: Console handler level.
        file_level: File handler level.
        log_dir: Directory for log files. If None, file logging is disabled.
        log_filename: Name of the log file.

    Returns:
        Configured structlog logger instance.
    """
    # Convert string levels to logging constants
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    root_level = level_map.get(logger_level.lower(), logging.DEBUG)
    console_lvl = level_map.get(console_level.lower(), logging.INFO)
    file_lvl = level_map.get(file_level.lower(), logging.DEBUG)

    # Shared processors for structlog
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create formatters
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )

    json_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    # Set up our application logger (not root logger to avoid capturing third-party logs)
    app_logger = logging.getLogger("gridtracer")
    app_logger.setLevel(root_level)
    app_logger.handlers.clear()
    app_logger.propagate = False

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_lvl)
    console_handler.setFormatter(console_formatter)
    app_logger.addHandler(console_handler)

    # File handler (JSON) - only if log_dir is provided
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_filename

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(file_lvl)
        file_handler.setFormatter(json_formatter)
        app_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ["torch", "matplotlib", "PIL", "wandb"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Return a structlog logger
    return structlog.get_logger("gridtracer")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name. If None, returns the root gridtracer logger.

    Returns:
        Structlog logger instance.
    """
    if name is None:
        return structlog.get_logger("gridtracer")
    return structlog.get_logger(name)
