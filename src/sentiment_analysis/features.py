# src/sentiment_analysis/features.py
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: str | None = None,
) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer for tweet text classification."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
