# src/sentiment_analysis/persistence.py
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_model(pipeline: Pipeline, path: Path) -> None:
    """Save a trained scikit-learn pipeline with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    """Load a trained scikit-learn pipeline."""
    return joblib.load(path)


def save_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Save model metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        **metadata,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_dataframe_csv(df, path: Path) -> None:
    """Save a pandas DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
