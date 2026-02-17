# Dockerfile pour l'application de prédiction de diabète

# https://hub.docker.com/_/python
FROM python:3.9-slim

# Permettre l'affichage immédiat des instructions et messages de log dans les journaux de Knative
ENV PYTHONUNBUFFERED True

# Copier le code local dans l'image du conteneur.
# Définir le répertoire de travail dans le conteneur à /app
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Installer les dépendances de production.
# Exécute pip install pour les packages spécifiés dans requirements.txt
RUN pip install -r requirements.txt

# Exécuter le service web au démarrage du conteneur. Ici, nous utilisons le serveur web gunicorn,
# avec un processus worker et 8 threads.
# Le timeout est réglé sur 0 pour désactiver les timeouts des workers pour permettre à Cloud Run de gérer l'échelonnement des instances.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app