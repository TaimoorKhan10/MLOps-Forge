"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger

from mlops_forge.config.settings import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward loguru."""

    def emit(self, record):
        """Intercept log record and redirect it to loguru."""
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(log_file: Optional[str] = None, level: str = None):
    """Set up logging configuration.
    
    Args:
        log_file: Path to log file. If None, logs to stdout only.
        level: Logging level. If None, uses the level from settings.
    """
    # Get log level from settings if not provided
    log_level = level or settings.LOGGING.LEVEL
    
    # Remove default loguru handler
    loguru_logger.remove()
    
    # Add stdout handler
    loguru_logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # Add file handler if log file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            str(log_path),
            level=log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # Intercept third-party logs
    for logger_name in ("uvicorn", "uvicorn.access", "fastapi"):
        logging.getLogger(logger_name).handlers = [InterceptHandler()]


def get_logger(name: str):
    """Get a logger instance.
    
    Args:
        name: Logger name.
    
    Returns:
        Logger instance.
    """
    return loguru_logger.bind(name=name)


# Set up logging on module import
setup_logging()
