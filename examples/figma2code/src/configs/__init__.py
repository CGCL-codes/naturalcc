"""
Configuration management for Figma2Code.

This module provides:
- Path constants (paths.py)
- LLM model configurations (models.py)
- Environment settings (settings.py)
"""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    DATA_TEST_DIR,
    DATA_REST_DIR,
    OUTPUT_DIR,
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    enter_project_root,
    ensure_output_dirs,
)
from .models import MODELS, get_model_config
from .settings import Settings, get_settings, settings

__all__ = [
    # Path constants
    "PROJECT_ROOT",
    "DATA_DIR",
    "DATA_TEST_DIR",
    "DATA_REST_DIR",
    "OUTPUT_DIR",
    "EXPERIMENTS_DIR",
    "RESULTS_DIR",
    "LOGS_DIR",
    "enter_project_root",
    "ensure_output_dirs",
    # Model configurations
    "MODELS",
    "get_model_config",
    # Settings
    "Settings",
    "get_settings",
    "settings",
]
