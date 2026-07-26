# Image de base : Python officiel, version slim (plus légère) basée sur Debian.
# On choisit 3.11 car c'est une version stable et largement compatible scikit-learn / FastAPI.
FROM python:3.12-slim

# Répertoire de travail à l'intérieur du conteneur.
# Toutes les instructions COPY/RUN suivantes s'exécuteront relativement à /app.
WORKDIR /app

# Variables d'environnement recommandées pour un usage production :
# - PYTHONUNBUFFERED=1 : les logs Python sortent en direct (pas de buffer), utile pour docker logs.
# - PYTHONDONTWRITEBYTECODE=1 : empêche la génération de fichiers .pyc (gain de place).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Installation de uv via pip. On utilise --no-cache-dir pour ne pas grossir l'image.
RUN pip install --no-cache-dir uv

# Étape clé pour le cache Docker :
# on copie d'abord les fichiers de dépendances seuls, avant le code.
# Comme pyproject.toml et uv.lock changent rarement, cette couche restera en cache
# tant qu'on ne modifie pas les dépendances => rebuilds beaucoup plus rapides.
COPY pyproject.toml uv.lock README.md ./

# Installation des dépendances (mode "gel" : respect strict de uv.lock, pas de résolution).
# --no-dev : on n'installe pas les dépendances de dev (ruff, pytest...) car l'image sert l'API.
RUN uv sync --frozen --no-dev

# Copie du code source et du modèle entraîné.
# L'ordre a son importance : le code change plus souvent que le modèle,
# mais pour rester simple on copie les deux ici.
COPY src/ ./src/
COPY models/ ./models/

# Indique à Docker (et au lecteur) que le conteneur expose ce port.
# NOTE : EXPOSE ne publie pas le port vers l'hôte — il faut aussi -p au run.
EXPOSE 8000

# Commande par défaut exécutée quand le conteneur démarre.
# --host 0.0.0.0 est INDISPENSABLE : par défaut uvicorn écoute 127.0.0.1,
# qui à l'intérieur du conteneur correspond au conteneur lui-même, et l'hôte
# ne pourrait pas joindre l'API même avec -p 8000:8000.
CMD ["uv", "run", "uvicorn", "sentiment_analysis.api:app", "--host", "0.0.0.0", "--port", "8000"]
