# Étape 1 : Récupérer le binaire uv depuis l'image officielle
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# Étape 2 : Image finale Python
FROM python:3.12-slim-bookworm

# Copier le binaire uv
COPY --from=uv_bin /uv /uvx /bin/

# Définir le répertoire de travail
WORKDIR /app

# Variables d'environnement pour optimiser Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copier le fichier de configuration des dépendances
COPY pyproject.toml ./

# Installer les dépendances globales dans le conteneur via uv
RUN uv pip install --system -r pyproject.toml

# Copier le reste du code de l'application
COPY . .

# Commande pour exécuter votre script
CMD ["python", "evaluate_article.py"]