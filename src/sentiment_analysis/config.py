# src/sentiment_analysis/config.py
from pathlib import Path

SEED = 1

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

TRAINING_DATA_PATH = RAW_DATA_DIR / "twitter_training.csv"
VALIDATION_DATA_PATH = RAW_DATA_DIR / "twitter_validation.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

FINAL_MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
FINAL_METADATA_PATH = MODEL_DIR / "sentiment_pipeline_metadata.json"

EXPERIMENT_SUMMARY_PATH = REPORTS_DIR / "experiment_summary.csv"
MISCLASSIFIED_EXAMPLES_PATH = REPORTS_DIR / "misclassified_examples.csv"

COLUMNS = ["Tweet_ID", "entity", "sentiment", "Tweet_content"]

MAPPING_SENTIMENT = {
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2,
    "Irrelevant": 3,
}

LABEL_NAMES = {
    0: "Negative",
    1: "Positive",
    2: "Neutral",
    3: "Irrelevant",
}
