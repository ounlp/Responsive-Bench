"""
Logging configuration for evaluation pipeline.

Provides structured logging with configurable levels, console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Any


def setup_eval_logging(config: dict[str, Any], eval_dir: Path) -> logging.Logger:
    """
    Set up logging for evaluation pipeline.

    Args:
        config: Evaluation config dict (may contain 'logging' section)
        eval_dir: Directory where eval outputs are saved (for log file)

    Returns:
        Configured logger instance
    """
    # Get logging config with defaults
    logging_config = config.get("logging", {})
    log_level_str = logging_config.get("level", "INFO").upper()
    console_enabled = logging_config.get("console", True)
    file_enabled = logging_config.get("file", True)
    file_path = logging_config.get("file_path")

    # Convert log level string to logging constant
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Create logger
    logger = logging.getLogger("responsive.eval")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_enabled:
        if file_path:
            log_file_path = Path(file_path)
        else:
            log_file_path = eval_dir / "eval.log"

        # Ensure directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "responsive.eval") -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (e.g., "responsive.eval.orchestrator", "responsive.eval.metrics")

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
