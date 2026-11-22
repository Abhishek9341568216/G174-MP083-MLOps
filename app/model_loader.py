import os

from src.mp083.config import COLUMNS, TARGET
from src.mp083.registry import load_version


def load_model():
    pipeline, metadata = load_version(
        os.environ.get("MODEL_DIR", "models"), os.environ.get("MODEL_VERSION") or None
    )
    if metadata.get("features") != COLUMNS or metadata.get("target") != TARGET:
        raise ValueError("Model does not match the prediction schema")
    return pipeline, metadata
