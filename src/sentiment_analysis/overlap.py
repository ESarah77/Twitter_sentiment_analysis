# src/sentiment_analysis/overlap.py
import pandas as pd


def normalize_text_for_duplicate_check(text: str) -> str:
    """Normalize text for duplicate and overlap checks."""
    return str(text).strip().lower()


def add_normalized_text_column(
    df: pd.DataFrame,
    text_col: str = "Tweet_content",
    normalized_col: str = "text_normalized",
) -> pd.DataFrame:
    """Add a normalized text column used for duplicate analysis."""
    df = df.copy()
    df[normalized_col] = df[text_col].map(normalize_text_for_duplicate_check)
    return df


def analyze_text_overlap_detailed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "Tweet_content",
    label_col: str = "sentiment",
) -> tuple[dict, pd.DataFrame]:
    """Analyze exact normalized text overlap between train and validation sets."""
    train_tmp = add_normalized_text_column(train_df, text_col=text_col)
    val_tmp = add_normalized_text_column(val_df, text_col=text_col)

    train_text_set = set(train_tmp["text_normalized"])

    n_train_duplicates = train_tmp["text_normalized"].duplicated().sum()
    n_val_duplicates = val_tmp["text_normalized"].duplicated().sum()

    val_overlap_mask = val_tmp["text_normalized"].isin(train_text_set)
    n_overlap = val_overlap_mask.sum()
    overlap_pct = 100 * n_overlap / len(val_tmp)

    train_labels_by_text = (
        train_tmp
        .groupby("text_normalized")[label_col]
        .agg(lambda labels: sorted(set(labels)))
        .to_dict()
    )

    overlap_val = val_tmp[val_overlap_mask].copy()
    overlap_val["train_labels_for_same_text"] = overlap_val["text_normalized"].map(
        train_labels_by_text
    )

    overlap_val["label_seen_with_same_text_in_train"] = overlap_val.apply(
        lambda row: row[label_col] in row["train_labels_for_same_text"],
        axis=1,
    )

    n_same_label_overlap = overlap_val["label_seen_with_same_text_in_train"].sum()
    n_conflicting_label_overlap = n_overlap - n_same_label_overlap

    results = {
        "n_train_rows": int(len(train_tmp)),
        "n_val_rows": int(len(val_tmp)),
        "n_duplicates_text_in_train": int(n_train_duplicates),
        "n_duplicates_text_in_val": int(n_val_duplicates),
        "n_val_texts_present_in_train": int(n_overlap),
        "val_overlap_percentage": float(overlap_pct),
        "n_overlap_same_label": int(n_same_label_overlap),
        "n_overlap_conflicting_label": int(n_conflicting_label_overlap),
        "overlap_same_label_percentage": (
            float(100 * n_same_label_overlap / n_overlap)
            if n_overlap > 0
            else 0.0
        ),
    }

    return results, overlap_val


def remove_validation_overlap_with_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "Tweet_content",
) -> pd.DataFrame:
    """Remove validation rows whose normalized text is present in train."""
    train_texts_normalized = train_df[text_col].map(normalize_text_for_duplicate_check)
    val_texts_normalized = val_df[text_col].map(normalize_text_for_duplicate_check)

    train_text_set = set(train_texts_normalized)
    keep_mask = ~val_texts_normalized.isin(train_text_set)

    return val_df.loc[keep_mask].copy()


def compare_class_distributions(
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    label_col: str,
    label_names: dict[int, str],
) -> pd.DataFrame:
    """Compare class distributions before and after filtering."""
    labels = sorted(label_names.keys())

    original_counts = original_df[label_col].value_counts().sort_index()
    filtered_counts = filtered_df[label_col].value_counts().sort_index()

    original_pct = original_df[label_col].value_counts(normalize=True).sort_index() * 100
    filtered_pct = filtered_df[label_col].value_counts(normalize=True).sort_index() * 100

    return pd.DataFrame({
        "label": [label_names[label] for label in labels],
        "original_count": original_counts.reindex(labels, fill_value=0).values,
        "original_percentage": original_pct.reindex(labels, fill_value=0).round(2).values,
        "no_overlap_count": filtered_counts.reindex(labels, fill_value=0).values,
        "no_overlap_percentage": filtered_pct.reindex(labels, fill_value=0).round(2).values,
    })
