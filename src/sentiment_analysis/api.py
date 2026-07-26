"""FastAPI application exposing the trained sentiment analysis pipeline."""

from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment_analysis.config import FINAL_MODEL_PATH, LABEL_NAMES
from sentiment_analysis.persistence import load_model


# ----- Schémas Pydantic (contrats d'API) -----

class PredictRequest(BaseModel):
    """Input schema for /predict: a single text or a list of texts."""

    texts: Union[str, list[str]] = Field(..., description="Tweet text(s) to classify")


class PredictionItem(BaseModel):
    """One prediction result, aligned with the input order."""

    text: str
    sentiment: str
    label_id: int


class PredictResponse(BaseModel):
    """Output schema for /predict."""

    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    """Output schema for /health."""

    status: str
    model_loaded: bool


# ----- État partagé (modèle chargé une fois au démarrage) -----

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, clear state on shutdown.

    Using lifespan is the modern FastAPI way (replaces on_event("startup")).
    """
    if not FINAL_MODEL_PATH.exists():
        raise RuntimeError(
            f"Trained model not found at {FINAL_MODEL_PATH}. "
            f"Run 'uv run python scripts/train_final_model.py' first."
        )
    _state["pipeline"] = load_model(FINAL_MODEL_PATH)
    yield
    _state.clear()


app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="REST API serving a TF-IDF + LinearSVC sentiment classifier.",
    version="1.0.0",
    lifespan=lifespan,
)


# ----- Routes -----

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check that the service is up and the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded="pipeline" in _state,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for the given tweet text(s).

    - Input: {"texts": "I love it"} or {"texts": ["I love it", "Awful"]}.
    - Output: predictions in the same order as the input.
    """
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    texts = [request.texts] if isinstance(request.texts, str) else list(request.texts)
    if not texts:
        raise HTTPException(status_code=400, detail="No text provided")

    predictions = pipeline.predict(texts)

    items = [
        PredictionItem(
            text=text,
            sentiment=LABEL_NAMES[int(label)],
            label_id=int(label),
        )
        for text, label in zip(texts, predictions)
    ]
    return PredictResponse(predictions=items)
