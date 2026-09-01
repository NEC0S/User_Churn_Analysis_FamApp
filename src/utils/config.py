"""
Small helper for loading config.yaml from anywhere in the project and
resolving paths relative to the project root (so it doesn't matter whether
you run a script from the repo root, from src/, or from inside a container).
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

# The project root is two levels up from this file: src/utils/config.py -> src/ -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    """
    Load the YAML config once and return it as a plain dict.

    Using a plain dict (instead of scattering config values across function
    defaults) means every threshold / path / hyperparameter has exactly one
    source of truth, and you can point at a different config.yaml per
    environment (dev/staging/prod) without touching any code.
    """
    path = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml (which is written relative to project root)."""
    return PROJECT_ROOT / relative_path


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    """
    Return a configured logger. Production code should never rely on bare
    print() statements (as the original notebooks did) -- logs need levels,
    timestamps, and a destination that survives after the process exits.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured, avoid duplicate handlers
        return logger

    level = logging.INFO
    log_file = None
    if cfg is not None:
        level = getattr(logging, cfg.get("logging", {}).get("level", "INFO"))
        log_file = cfg.get("logging", {}).get("log_file")

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
