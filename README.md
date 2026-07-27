# Twitter Sentiment Analysis

An end-to-end Machine Learning engineering project: from rigorous experimentation on tweet sentiment classification to a reproducible training pipeline and a packaged REST API served with FastAPI and Docker.

This repository demonstrates a complete, production-minded ML workflow rather than a single notebook prototype:
- **clean, modular Python code** structured as an installable package;
- **rigorous experimentation** with stratified cross-validation, explicit train/validation discipline, and a data-leakage check;
- **reproducible results** through pinned dependencies and a fixed seed;
- **model persistence and deployment** through a documented FastAPI API and a Docker image.

The final model — **TF-IDF (unigrams + bigrams) + LinearSVC** — is intentionally a strong, simple, explainable baseline. It was selected after comparing several preprocessing and modeling configurations, and its performance is close to the practical ceiling on this task for linear TF-IDF representations.

---

## Table of contents

- [Twitter Sentiment Analysis](#twitter-sentiment-analysis)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Project structure](#project-structure)
  - [Key results](#key-results)
  - [Quickstart](#quickstart)
    - [Requirements](#requirements)
    - [Install the environment](#install-the-environment)
    - [Download the dataset](#download-the-dataset)
  - [Usage](#usage)
    - [Run the experiments notebook](#run-the-experiments-notebook)
    - [Train the final model](#train-the-final-model)
    - [Run predictions from the CLI](#run-predictions-from-the-cli)
    - [Serve the model with FastAPI](#serve-the-model-with-fastapi)
    - [Run with Docker](#run-with-docker)
  - [Methodology](#methodology)
    - [Problem framing](#problem-framing)
    - [Evaluation protocol](#evaluation-protocol)
    - [Experiments](#experiments)
    - [Final model](#final-model)
  - [Known limitations and future work](#known-limitations-and-future-work)
  - [Reproducibility](#reproducibility)
  - [License](#license)

---

## Overview

This project builds a **multi-class sentiment classifier** on tweets — `Positive`, `Negative`, `Neutral`, `Irrelevant` — using **TF-IDF features** and a **linear model** (`LinearSVC`), selected through stratified cross-validation.

It is structured as a small but complete ML system, not just a notebook:

- a modular `sentiment_analysis` Python package, reusable across the notebook, training script, CLI, and API;
- an experimentation notebook that documents every modeling decision (missing-value strategy, class weighting, preprocessing, model choice) with a consistent evaluation protocol;
- a saved, versioned pipeline serialized with `joblib` and served through a **FastAPI** application exposing `/predict` and `/health`, with an interactive Swagger UI;
- a **Docker** setup for reproducible deployment.

Only the tweet text is used as model input. The `entity` column is kept for analysis and error inspection but is **not** used as a feature, to keep the model focused on text-based sentiment classification and the design choices easy to justify.

---

## Dataset

The dataset is the **[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)** dataset, available on Kaggle.

It contains tweet texts, an associated entity, and a sentiment label.

| Column          | Description                              |
|-----------------|------------------------------------------|
| `Tweet_ID`      | Unique tweet identifier                   |
| `entity`        | Entity mentioned in the tweet            |
| `sentiment`     | Label: Positive / Negative / Neutral / Irrelevant |
| `Tweet_content` | Raw tweet text                           |

The dataset is **not** redistributed in this repository (see [License](#license)). It must be downloaded separately via the provided script.

---

## Project structure

```
twitter_sentiment_analysis/
├── data/
│   ├── raw/                     # Downloaded CSVs 
│   └── download.py              # Dataset download / extraction script
├── models/
│   ├── sentiment_pipeline.joblib
│   └── sentiment_pipeline_metadata.json
├── notebook/
│   └── experiments.ipynb        # Experimentation notebook
├── reports/
│   ├── experiment_summary.csv
│   └── misclassified_examples.csv
├── scripts/
│   ├── predict.py               # CLI inference
│   └── train_final_model.py     # Final model training
├── src/
│   └── sentiment_analysis/
│       ├── api.py                # FastAPI application
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       ├── models.py
│       ├── overlap.py
│       ├── persistence.py
│       └── utils.py
├── .dockerignore
├── .gitignore
├── .python-version
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Key results

The final model is **TF-IDF (unigrams + bigrams) + LinearSVC**, selected from cross-validation on the training set.

| Metric                                         | Value   |
|------------------------------------------------|---------|
| Cross-validation macro F1 (training set)        | 0.9393  |
| Cross-validation std                            | 0.0008  |
| Training macro F1                               | 0.9787  |
| Validation macro F1 (full validation set)       | 0.9878  |
| Validation macro F1 (no train overlap subset)   | 0.9797  |
| Validation / training overlap (%)               | 51.7    |
| Training time (s)                              | 5.0     |

**How to read these numbers:**

- Cross-validation macro F1 on the training set (≈ **0.94**) is the metric used for model selection; it is reported with its standard deviation across folds to show stability.
- Validation macro F1 on the held-out validation set (≈ **0.99**) is reported as an external check, *not* used to tune the model.
- Because the provided validation set partially overlaps with the training set (≈ 52%), the model is also evaluated on a **filtered subset with no train overlap** (≈ **0.98** macro F1). The score staying high on this conservative subset indicates that performance is not primarily driven by exact-duplicate memorization.

Full per-experiment results and error analysis are available in `notebook/experiments.ipynb` and in `reports/`.

---

## Quickstart

### Requirements

- Python 3.12 (managed by `uv`)
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- A Kaggle account and a configured `kaggle.json` API token (see [this page](https://www.kaggle.com/docs/api))

### Install the environment

```bash
uv sync
```

All dependency versions are pinned in `uv.lock`; there is no need to install packages manually.

### Download the dataset

```bash
uv run python data/download.py
```

This downloads and extracts the CSV files into `data/raw/`.

---

## Usage

### Run the experiments notebook

```bash
uv run jupyter notebook notebook/experiments.ipynb
```

The notebook reproduces the full experimentation workflow:

1. data loading and inspection;
2. missing value analysis;
3. class distribution analysis;
4. train/validation overlap analysis;
5. experiments on NaN handling, class weighting, preprocessing, and model choice;
6. final model selection and persistence;
7. error analysis and export of results.

### Train the final model

```bash
uv run python scripts/train_final_model.py
```

This retrains the selected pipeline on the full training set and saves:

- `models/sentiment_pipeline.joblib`
- `models/sentiment_pipeline_metadata.json`

### Run predictions from the CLI

```bash
uv run python scripts/predict.py "I love this game, it is amazing!"
```

### Serve the model with FastAPI

Development (with auto-reload):

```bash
uv run uvicorn sentiment_analysis.api:app --reload --app-dir src
```

Production:

```bash
uv run uvicorn sentiment_analysis.api:app --host 0.0.0.0 --port 8000
```

Then open:

- [Swagger UI](http://localhost:8000/docs)
- [Health check](http://localhost:8000/health)

Example request:

```bash
curl -X POST "[localhost](http://localhost:8000/predict)" \
  -H "Content-Type: application/json" \
  -d '{"texts": "I absolutely love the new design"}'
```

Example response:

```json
{
  "predictions": [
    {
      "text": "I absolutely love the new design",
      "sentiment": "Positive",
      "label_id": 1
    }
  ]
}
```

The `/predict` endpoint accepts either a single string or a list of strings. Predictions are returned in the same order as the input.

### Run with Docker

Build the image:

```bash
docker build -t twitter-sentiment-api:latest .
```

Run the container:

```bash
docker run --rm -p 8000:8000 --name sentiment_api twitter-sentiment-api:latest
```

Or with `docker-compose`:

```bash
docker compose up --build
```


> The model file is expected to be present in `models/` inside the image.

---

## Methodology

### Problem framing

Multi-class text classification with four labels: `Positive`, `Negative`, `Neutral`, `Irrelevant`. The main metric is **macro F1-score**, chosen because the classes are not perfectly balanced and we want the model to perform reasonably on all of them rather than only on the majority class.

### Evaluation protocol

- **Cross-validation** (stratified, 5 folds) on the **training set only** is used to compare configurations and select the best one.
- The **validation set** is kept untouched during model selection and used only as an external final check.
- The selected configuration is then retrained on the full training set and evaluated on the validation set.
- An additional evaluation is performed on a **filtered validation subset** that removes tweets already present in the training set, to check that performance is not driven by exact duplicate memorization.

### Experiments

The notebook compares several configurations in a controlled way:

1. **Missing value strategy**: dropping rows with missing text vs. filling with an empty string.
2. **Class weighting**: with vs. without `class_weight="balanced"`.
3. **Text preprocessing**: TF-IDF with unigrams, stopword removal, bigrams, and their combinations.
4. **Model choice**: Logistic Regression vs. LinearSVC.

Each experiment reuses shared utility functions for cross-validation and evaluation, so configurations are compared on the same protocol.

### Final model

`LinearSVC` with TF-IDF (unigrams + bigrams) and `class_weight="balanced"` was selected because it gave the best cross-validation macro F1 while remaining fast and explainable.

Although `LinearSVC` does not output calibrated probabilities (unlike Logistic Regression), it is significantly stronger on this task and was therefore kept as the production model.

---

## Known limitations and future work

- **Train/validation overlap.** A significant portion of the validation set also appears in the training set. The filtered (no-overlap) evaluation mitigates this, but a cleaner split built from raw data would be more rigorous.
- **Text-only features.** The `entity` column is not used as input. Incorporating it could improve performance on entity-specific tweets.
- **Linear model.** Performance is close to the ceiling achievable on this task with simple TF-IDF representations. Embedding-based approaches (GloVe, fine-tuned BERT) were explored during development but dropped because the marginal expected gain (~0.01 macro F1) did not justify the added complexity for this project.
- **No calibrated probabilities.** `LinearSVC` does not natively output probabilities. If confidence scores are needed (e.g. for downstream business logic), wrapping it with `CalibratedClassifierCV` would be a natural extension.
- **Single language.** The dataset is English; the model is not expected to generalize to other languages without additional data.

---

## Reproducibility

- A fixed seed (`SEED = 1`) is set for Python's `random`, NumPy, and scikit-learn components exposing `random_state`.
- The cross-validation split is stratified and seeded.
- All dependency versions are pinned in `uv.lock`; running `uv sync` reproduces the exact environment.
- Re-running the notebook or `scripts/train_final_model.py` with the same seed and dataset produces the same model and the same metrics.
- The only runtime values that can vary between runs are execution timestamps and elapsed times, which do not affect the model or the scores.

---

## License

The **source code** in this repository is licensed under the **MIT License** — see the `LICENSE` file.

The **dataset** (`twitter_training.csv`, `twitter_validation.csv`) is **not** part of this repository and is **not** redistributed here. It is provided by Kaggle under its own license terms; please review the dataset page on Kaggle before using it. By downloading the dataset through `data/download.py`, you agree to the Kaggle terms of use.
