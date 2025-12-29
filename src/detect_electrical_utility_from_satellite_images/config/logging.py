"""
Logging configuration.
"""

import typing

import pydantic


class LoggingConfig(pydantic.BaseModel):
    """Logging configuration."""

    logger_lvl: typing.Literal["debug", "info", "warning", "error", "critical"]
    """Logging level for the root logger."""

    console_handler_lvl: typing.Literal["debug", "info", "warning", "error", "critical"]
    """Logging level for console handler."""

    file_handler_lvl: typing.Literal["debug", "info", "warning", "error", "critical"]
    """Logging level for file handler."""
