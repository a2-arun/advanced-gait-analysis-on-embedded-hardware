"""Shared logger setup, configured from config.yaml's `logging` section."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.config import load_config, resolve_path

_configured_loggers = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _configured_loggers:
        return logger

    config = load_config().get("logging", {})
    level = getattr(logging, config.get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    if config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    log_file = config.get("file")
    if log_file:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.get("max_size_mb", 100) * 1024 * 1024,
            backupCount=config.get("backup_count", 5),
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _configured_loggers.add(name)
    return logger
