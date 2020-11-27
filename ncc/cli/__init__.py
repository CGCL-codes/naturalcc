import os
from dataset import DEFAULT_DIR

DEMO_MODEL_DIR = os.path.join(DEFAULT_DIR, 'demo')
os.makedirs(DEMO_MODEL_DIR, exist_ok=True)

__all__ = (
    DEMO_MODEL_DIR,
)
