import pandas as pd
from sklearn.metrics import classification_report, f1_score

from sentiment_analysis.config import (
    EXPERIMENT_SUMMARY_PATH,
    FINAL_METADATA_PATH,
    FINAL_MODEL_PATH,
    LABEL_NAMES,
    MISCLASSIFIED_EXAMPLES_PATH,
    SEED,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from sentiment_analysis.data import (
    drop_missing_texts,
    get_features_and_target,
    load_twitter_dataset,
)
from sentiment_analysis.evaluation import (
    cross_validate_pipeline,
    get_misclassified_examples,
    train_and_evaluate_pipeline,
)
from sentiment_analysis.models import build_linear_svc_pipeline
from sentiment_analysis.overlap import (
    analyze_text_overlap_detailed,
    remove_validation_overlap_with_train,
)
from sentiment_analysis.persistence import (
    save_dataframe_csv,
    save_metadata,
    save_model,
)
from sentiment_analysis.utils import set_seed


def main() -> None:
    set_seed(SEED)

    train_df = load_twitter_dataset(TRAINING_DATA_PATH)
    val_df = load_twitter_dataset(VALIDATION_DATA_PATH)

    train_df = drop_missing_texts(train_df)

    X_train, y_train = get_features_and_target(train_df)
    X_val, y_val = get_features_and_target(val_df)

    pipeline = build_linear_svc_pipeline(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        ngram_range=(1, 2),
    )

    cv_result = cross_validate_pipeline(
        exp_name="Linear SVM (+TF-IDF)",
        pipeline=pipeline,
        X=X_train,
        y=y_train,
        n_splits=5,
        seed=SEED,
    )

    final_eval = train_and_evaluate_pipeline(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    trained_pipeline = final_eval["trained_pipeline"]

    val_no_overlap_df = remove_validation_overlap_with_train(
        train_df=train_df,
        val_df=val_df,
    )

    X_val_no_overlap, y_val_no_overlap = get_features_and_target(val_no_overlap_df)
    y_pred_val_no_overlap = trained_pipeline.predict(X_val_no_overlap)

    val_no_overlap_f1_macro = f1_score(
        y_val_no_overlap,
        y_pred_val_no_overlap,
        average="macro",
    )

    overlap_results, _ = analyze_text_overlap_detailed(
        train_df=train_df,
        val_df=val_df,
    )

    labels = sorted(LABEL_NAMES.keys())
    target_names = [LABEL_NAMES[label] for label in labels]

    print("Final model: Linear SVM (+TF-IDF)")
    print(f"CV macro F1: {cv_result['cv_mean_f1_macro']:.4f}")
    print(f"Validation macro F1: {final_eval['val_f1_macro']:.4f}")
    print(f"Validation no-overlap macro F1: {val_no_overlap_f1_macro:.4f}")
    print()
    print(classification_report(
        y_val,
        final_eval["y_val_pred"],
        labels=labels,
        target_names=target_names,
    ))

    save_model(trained_pipeline, FINAL_MODEL_PATH)

    metadata = {
        "model_name": "sentiment_pipeline",
        "model_type": type(trained_pipeline.named_steps["clf"]).__name__,
        "vectorizer": type(trained_pipeline.named_steps["tfidf"]).__name__,
        "experiment_name": "Linear SVM (+TF-IDF)",
        "cv_mean_f1_macro": cv_result["cv_mean_f1_macro"],
        "cv_std_f1_macro": cv_result["cv_std_f1_macro"],
        "train_f1_macro": final_eval["train_f1_macro"],
        "validation_f1_macro": final_eval["val_f1_macro"],
        "validation_no_overlap_f1_macro": float(val_no_overlap_f1_macro),
        "validation_overlap_percentage": overlap_results["val_overlap_percentage"],
        "validation_no_overlap_size": int(len(val_no_overlap_df)),
        "train_time_sec": final_eval["train_time_sec"],
        "eval_time_sec": final_eval["eval_time_sec"],
        "seed": SEED,
        "labels": LABEL_NAMES,
    }

    save_metadata(metadata, FINAL_METADATA_PATH)

    experiment_summary = pd.DataFrame([{
        "experiment_group": "Final model",
        "experiment_name": "Linear SVM (+TF-IDF)",
        "cv_mean_f1_macro": cv_result["cv_mean_f1_macro"],
        "cv_std_f1_macro": cv_result["cv_std_f1_macro"],
        "train_f1_macro": final_eval["train_f1_macro"],
        "validation_f1_macro": final_eval["val_f1_macro"],
        "validation_no_overlap_f1_macro": float(val_no_overlap_f1_macro),
        "train_time_sec": final_eval["train_time_sec"],
        "eval_time_sec": final_eval["eval_time_sec"],
    }])

    save_dataframe_csv(experiment_summary, EXPERIMENT_SUMMARY_PATH)

    misclassified_examples = get_misclassified_examples(
        val_df=val_df,
        y_true=y_val,
        y_pred=final_eval["y_val_pred"],
        label_names=LABEL_NAMES,
    )

    save_dataframe_csv(misclassified_examples, MISCLASSIFIED_EXAMPLES_PATH)

    print(f"Saved model to: {FINAL_MODEL_PATH}")
    print(f"Saved metadata to: {FINAL_METADATA_PATH}")
    print(f"Saved experiment summary to: {EXPERIMENT_SUMMARY_PATH}")
    print(f"Saved misclassified examples to: {MISCLASSIFIED_EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
