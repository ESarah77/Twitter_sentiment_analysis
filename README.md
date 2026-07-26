# Installation de l'environnement

### Initialiser le projet (crée le projet Python et le pyproject.toml)
    uv init

### Créer l'environnement
    uv venv

### Activer l'environnement
    source .venv/bin/activate


### Installer les dépendances
    uv add numpy pandas scikit-learn matplotlib jupyter ipykernel kaggle 
    accelerate datasets
(à revoir et installer un par un car peut-être que je n'ai plus besoin d'autant de packages à installer)

    uv add fastapi "uvicorn[standard]"
(dépendances pour FastAPI)

### Commandes pour lancer FastAPI
Développement (avec rechargement automatique quand tu modifies api.py)
    uv run uvicorn sentiment_analysis.api:app --reload --app-dir src

Production (sans rechargement) :
    uv run uvicorn sentiment_analysis.api:app --host 0.0.0.0 --port 8000
Puis 
    http://localhost:8000/docs --> interface Swagger interactive pour tester /predict directement dans le navigateur
    http://localhost:8000/health --> vérifier que le modèle est chargé

Exemple d'appel curl :
    curl -X POST 'http://localhost:8000/predict' \
  -H "Content-Type: application/json" \
  -d '{"texts": "I absolutely love the new design"}'

### Commandes pour lancer Docker
#### Construire l'image (build)
    sudo docker build -t twitter-sentiment-api:latest .

#### Lancer l'image/le conteneur (run)
    sudo docker run --rm -p 8000:8000 --name sentiment_api twitter-sentiment-api:latest

#### Pour construire et lancer le conteneur (grâce au docker-compose.yml)
    sudo docker compose up --build

#### Vérifier que le conteneur est inactif, sinon supprimer le processus
    sudo docker ps -a
    sudo docker rm -f container

### POINT A VERIFIER : EXECUTION DES SCRIPTS PYTHON
En fait, j'avais essayé d'exécuter les scripts .py mais ça ne marchait pas, du coup j'ai ajouté ça à la fin du pyproject.toml :
    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

    [tool.hatch.build.targets.wheel]
    packages = ["src/sentiment_analysis"]

# Setup de l'environnement
    uv sync


# Setup les données

- Télécharger kaggle.json et le mettre dans credentials/
- Lancer le téléchargement du dataset (.zip + décompression) avec la commande :
uv run data/download.py

# Arborescence du projet

## Arborescence actuelle
twitter-sentiment-analysis/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── twitter_training.csv
│   │   └── twitter_validation.csv
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── notebooks/
│   └── my_notebook_of_experiments.ipynb
├── src/
│   └── sentiment_analysis/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── evaluation.py
│       ├── overlap.py
│       ├── persistence.py
│       └── utils.py
└── scripts/
    ├── train_final_model.py
    └── predict.py

----------------------

## Arborescence possible (à revoir) avec Docker et FastAPI
    twitter-sentiment-analysis/
    │
    ├── app/
    ├── notebook/
    │   └── experiments.ipynb
    ├── saved_model/
    ├── pyproject.toml
    ├── uv.lock
    ├── README.md
    └── Dockerfile

## Dernière version d'arborescence possible
sentiment-analysis-ml-project/
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── model_loader.py
├── data/
│   ├── raw/
│   │   ├── twitter_training.csv
│   │   └── twitter_validation.csv
│   └── README.md
├── models/
│   ├── sentiment_pipeline.joblib
│   └── sentiment_pipeline_metadata.json
├── notebooks/
│   └── sentiment_analysis_experiments.ipynb
├── reports/
│   └── experiment_summary.csv
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── training.py
│   └── utils.py
├── tests/
│   └── test_api.py
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .gitignore
└── README.md

