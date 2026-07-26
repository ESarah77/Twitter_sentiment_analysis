# src/sentiment_analysis/data.py
from pathlib import Path

import pandas as pd

from sentiment_analysis.config import COLUMNS, MAPPING_SENTIMENT


def load_twitter_dataset(path: Path) -> pd.DataFrame:
    """Load a Twitter sentiment CSV file and encode sentiment labels."""
    df = pd.read_csv(path, names=COLUMNS)

    unknown_mask = ~df["sentiment"].isin(MAPPING_SENTIMENT.keys())
    if unknown_mask.any():
        unknown_labels = df.loc[unknown_mask, "sentiment"].unique()
        raise ValueError(f"Unknown sentiment labels found: {unknown_labels}")

    df["sentiment"] = df["sentiment"].map(MAPPING_SENTIMENT).astype(int)

    return df



def drop_missing_texts(
    df: pd.DataFrame,
    text_col: str = "Tweet_content",
) -> pd.DataFrame:
    """Remove rows with missing tweet text."""
    return df.dropna(subset=[text_col]).copy()


def fill_missing_texts(
    df: pd.DataFrame,
    text_col: str = "Tweet_content",
    fill_value: str = "",
) -> pd.DataFrame:
    """Fill missing tweet text values."""
    df = df.copy()
    df[text_col] = df[text_col].fillna(fill_value)
    return df


def get_features_and_target(
    df: pd.DataFrame,
    text_col: str = "Tweet_content",
    target_col: str = "sentiment",
) -> tuple[pd.Series, pd.Series]:
    """Return text features and target labels."""
    return df[text_col], df[target_col]
