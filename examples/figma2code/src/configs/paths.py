"""
Path constants for Figma2Code.

All paths are relative to the project root directory.
"""
import os
from pathlib import Path

# Project root directory (where main2.py is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_TEST_DIR = DATA_DIR / "data_test"
DATA_REST_DIR = DATA_DIR / "data_rest"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
DATA_PROCESSING_DIR = OUTPUT_DIR / "data_processing"

def enter_project_root():
    """Enter the project root directory."""
    os.chdir(str(PROJECT_ROOT))

# Ensure output directories exist
def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, EXPERIMENTS_DIR, RESULTS_DIR, LOGS_DIR, DATA_PROCESSING_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)