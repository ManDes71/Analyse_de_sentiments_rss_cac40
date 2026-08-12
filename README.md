# Analyse de sentiments appliquée à des flux rss boursiers

**`Analyse_de_sentiments_rss_cac40`** est un pipeline d'IA bout en bout conçu pour automatiser la veille financière et la notation de sentiment d'entreprises à partir de flux de presse (RSS).

Face aux limites des approches génératives globales (coûts API, latence, faux positifs), le projet met en œuvre une **architecture hybride et ciblée** :

1. **Extraction dynamique d'alias :** Identification automatique du jargon et des surnoms journalistiques via Google Gemini 2.5 Flash et schémas Pydantic.


2. **Filtrage & Prétraitement :** Qualification des articles par expressions régulières et résumés TF-IDF avec pondération contextuelle (boost d'entité).


3. **Scoring NLP dual :** Évaluation fine du sentiment financier par un modèle CamemBERT spécialisé, comparant une approche par fenêtres glissantes GPU et une agrégation par blocs.


4. **Industrialisation MLOps :** Déploiement conteneurisé sous Docker, gestionnaire de paquets `uv`, et persistance PostgreSQL.



---

### Architecture


* **Conception d'un pipeline d'IA hybride :** Traitement de flux RSS financiers combinant règles Regex, résumés TF-IDF ciblés et Transformers (CamemBERT Finance).


* **Intégration LLM & Structured Outputs :** Automatisation de la création d'un dictionnaire d'alias d'entreprises avec Gemini 2.5 et Pydantic.


* **Optimisation de l'inférence :** Mise en place d'une évaluation duale (mémoire/tokens via `pipeline` vs fenêtres glissantes matricielles sous PyTorch/GPU).


* **Industrialisation MLOps :** Stack conteneurisée sous Docker Compose, migration vers `uv` (Python 3.12) et stockage structuré PostgreSQL.


