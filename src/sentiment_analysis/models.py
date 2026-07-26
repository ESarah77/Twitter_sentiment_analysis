# src/sentiment_analysis/models.py
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sentiment_analysis.features import build_tfidf_vectorizer


def build_logistic_regression_pipeline(
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    random_state: int | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: str | None = None,
) -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        (
            "tfidf",
            build_tfidf_vectorizer(
                ngram_range=ngram_range,
                stop_words=stop_words,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
            ),
        ),
    ])


def build_linear_svc_pipeline(
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    random_state: int | None = None,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = None,
) -> Pipeline:
    """Build a TF-IDF + Linear SVM pipeline."""
    return Pipeline([
        (
            "tfidf",
            build_tfidf_vectorizer(
                ngram_range=ngram_range,
                stop_words=stop_words,
            ),
        ),
        (
            "clf",
            LinearSVC(
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
            ),
        ),
    ])
