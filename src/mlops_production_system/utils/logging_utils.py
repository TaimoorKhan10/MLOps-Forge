"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger

from mlops_production_system.config.settings import settings


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
    # Get settings
    log_level = level or settings.LOGGING.LEVEL
    log_file = log_file or settings.LOGGING.FILE_PATH
    
    # Remove default handlers
    loguru_logger.remove()
    
    # Add stdout handler
    loguru_logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        level=log_level,
        colorize=True,
    )
    
    # Add file handler if log file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        loguru_logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            compression="zip",
            retention="7 days",
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Replace standard library logging handlers
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


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
