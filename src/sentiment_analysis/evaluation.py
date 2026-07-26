# src/sentiment_analysis/evaluation.py
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def cross_validate_pipeline(
    exp_name: str,
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 1,
) -> dict:
    """Evaluate a pipeline with stratified cross-validation using macro F1."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    scores = []
    fold_times = []
    cv_start_time = perf_counter()

    for train_idx, val_idx in cv.split(X, y):
        fold_start_time = perf_counter()

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        y_pred = fold_pipeline.predict(X_val)
        score = f1_score(y_val, y_pred, average="macro")

        scores.append(score)
        fold_times.append(perf_counter() - fold_start_time)

    total_cv_time = perf_counter() - cv_start_time

    return {
        "exp_name": exp_name,
        "cv_mean_f1_macro": float(np.mean(scores)),
        "cv_std_f1_macro": float(np.std(scores)),
        "fold_scores": [float(score) for score in scores],
        "cv_total_time_sec": float(total_cv_time),
        "cv_mean_time_per_fold_sec": float(np.mean(fold_times)),
        "fold_times_sec": [float(fold_time) for fold_time in fold_times],
    }


def run_experiments(
    experiments: list[dict],
    exp_group: str,
    n_splits: int = 5,
    seed: int = 1,
) -> list[dict]:
    """Run cross-validation for a list of experiment configurations."""
    results = []

    for experiment in experiments:
        cv_result = cross_validate_pipeline(
            exp_name=experiment["name"],
            pipeline=experiment["pipeline"],
            X=experiment["X_train"],
            y=experiment["y_train"],
            n_splits=n_splits,
            seed=seed,
        )

        results.append({
            "exp_group": exp_group,
            **experiment,
            **cv_result,
        })

    return results


def summarize_experiment_results(results: list[dict]) -> pd.DataFrame:
    """Create a compact DataFrame from experiment results."""
    return pd.DataFrame([
        {
            "experiment_group": result.get("exp_group"),
            "experiment_name": result["exp_name"],
            "cv_mean_f1_macro": result["cv_mean_f1_macro"],
            "cv_std_f1_macro": result["cv_std_f1_macro"],
            "cv_mean_time_per_fold_sec": result["cv_mean_time_per_fold_sec"],
            "fold_scores": result["fold_scores"],
        }
        for result in results
    ])


def train_and_evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
) -> dict:
    """Train a pipeline on the full training set and evaluate it on validation."""
    start_train_time = perf_counter()

    trained_pipeline = clone(pipeline)
    trained_pipeline.fit(X_train, y_train)

    train_time_sec = perf_counter() - start_train_time

    y_pred_train = trained_pipeline.predict(X_train)
    train_f1_macro = f1_score(y_train, y_pred_train, average="macro")

    start_eval_time = perf_counter()

    y_pred_val = trained_pipeline.predict(X_val)

    eval_time_sec = perf_counter() - start_eval_time
    val_f1_macro = f1_score(y_val, y_pred_val, average="macro")

    return {
        "trained_pipeline": trained_pipeline,
        "y_val_pred": y_pred_val,
        "val_f1_macro": float(val_f1_macro),
        "train_f1_macro": float(train_f1_macro),
        "train_time_sec": float(train_time_sec),
        "eval_time_sec": float(eval_time_sec),
    }


def build_classification_report_dict(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[int],
    target_names: list[str],
) -> dict:
    """Return a classification report as a dictionary."""
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
    )


def get_misclassified_examples(
    val_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    label_names: dict[int, str],
) -> pd.DataFrame:
    """Return validation examples where predicted label differs from true label."""
    errors_df = val_df.copy()

    errors_df["true_label_id"] = y_true.values
    errors_df["pred_label_id"] = y_pred

    errors_df["sentiment"] = errors_df["true_label_id"].map(label_names)
    errors_df["pred_sentiment"] = errors_df["pred_label_id"].map(label_names)

    misclassified_df = errors_df[
        errors_df["true_label_id"] != errors_df["pred_label_id"]
    ].copy()

    columns_to_display = [
        "Tweet_ID",
        "entity",
        "Tweet_content",
        "sentiment",
        "pred_sentiment",
    ]

    return misclassified_df[columns_to_display]
