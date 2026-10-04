# Analyse de sentiment CAC 40 par LLM

**Une note de sentiment fiable, entreprise par entreprise, pour les articles de presse financière — construite en mesurant ce que valent les modèles plutôt qu'en leur faisant confiance.**

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)
![Statut](https://img.shields.io/badge/statut-en%20cours-orange)

> Série d'articles associée : *Market-RSS-Sentiment* sur [aventuresdata.com](https://aventuresdata.com). Ce dépôt documente le code ; le blog raconte la démarche.

---

## Contexte

Noter le sentiment d'un article sur une entreprise précise paraît simple, jusqu'à ce qu'on essaie de le mesurer. Un article sur la séance du CAC 40 cite dix valeurs et ne dit rien de chacune d'elles ; un modèle de langage lui attribue pourtant une note. Deux modèles cloud de premier plan ne s'accordent que dans 80 % des cas, et aucun n'est « la vérité ».

Ce projet répond à une question de produit avant d'être une question de modèle : **comment obtenir une note de référence dont on connaît la fiabilité, au meilleur coût ?** Il part de flux RSS, décide quels couples (article, entreprise) méritent d'être notés, les fait noter par plusieurs LLM, et mesure le résultat contre 191 annotations humaines.

## Ce que fait le projet

- **Délimite le périmètre par genre d'article**, pas par comptage d'occurrences : embeddings multilingues, clustering, puis règles ordonnées (`perimetre.py`). Un article de chronique de marché sort ; un article dédié à l'entreprise entre.
- **Fait noter chaque couple par cinq modèles** — Gemini 2.5 Flash et Claude Haiku (API distantes), Llama 3.1 8B, Mistral Nemo et Qwen 2.5 7B (locaux via Ollama) — avec des prompts versionnés et une sortie validée par Pydantic.
- **Exécute en mode synchrone ou batch** (−50 % de coût, jusqu'à 24 h de délai), avec reprise automatique des runs interrompus pour ne jamais refacturer ce qui est déjà fait.
- **Historise toutes les notes** par (couple, modèle, version de prompt) pour comparer deux versions d'un prompt par une simple requête SQL.
- **Calcule une note de consensus** : médiane de trois modèles, avec un indicateur d'unanimité qui dit, couple par couple, s'il faut s'y fier.
- **Mesure tout contre une vérité humaine** : 191 couples annotés à la main avec un outil HTML maison, jamais utilisés pour l'entraînement.
- **Prépare et teste l'entraînement d'un transformer** (CamemBERT) à partir des étiquettes du consensus. Résultat à ce jour : un échec instructif, documenté plus bas.

## Architecture et stack

```
flux RSS (collecte et rattachement par entreprise : projet voisin fluxRss)
        │
        ▼
PostgreSQL  (articles, entreprises + alias, couples article/entreprise, notes)
        │
        ├─ cluster_genres.py ─► genres d'article (E5 multilingue + HDBSCAN, pgvector)
        │
        ├─ perimetre.py ─────► quels couples méritent d'être notés
        │
        ▼
evaluate_article.py ── providers/ (gemini · haiku · ollama) ── llm_common.py (prompts, validation, retry)
        │
        ├─ historique_notes.py ─► table article_company_notes (une ligne par couple / modèle / version)
        │
        ▼
note_consensus.py ─► note_consensus + consensus_unanime
        │
        ├─ benchmark_classification.py ─► comparaison aux annotations humaines
        │
        └─ preparer_corpus.py ─► entrainer_transformer.py (CamemBERT, test sur les 191 annotations)
```

| Domaine | Choix |
|---|---|
| Langage, dépendances | Python 3.13+, gestion par `uv` (`pyproject.toml`, `uv.lock`) |
| LLM distants | Gemini 2.5 Flash, Claude Haiku 4.5 (appels REST directs, API batch) |
| LLM locaux | Llama 3.1 8B, Mistral Nemo, Qwen 2.5 7B via Ollama (RTX 3060 12 Go) |
| Données | PostgreSQL, extension pgvector pour les embeddings de genre |
| Validation | Pydantic (schéma de sortie, plages de valeurs, énumérations) |
| ML classique | scikit-learn, pandas, matplotlib, seaborn |
| Transformers | PyTorch, Hugging Face `transformers`, `camembert-base` |
| Traitement du texte | NLTK, spaCy (modèle français), `intfloat/multilingual-e5-base` |

## Installation

Prérequis : Python 3.13 ou plus, [`uv`](https://docs.astral.sh/uv/), une base PostgreSQL alimentée par le projet de collecte, et pour les modèles locaux un serveur [Ollama](https://ollama.com) avec `llama3.1:8b`, `mistral-nemo` et `qwen2.5:7b`.

```bash
uv sync                      # crée .venv à partir de pyproject.toml / uv.lock
source .venv/bin/activate
cp .env.example .env         # puis renseigner la base et les clés API
```

Les clés et mots de passe ne doivent jamais être versionnés : `.env` est dans `.gitignore`. Variables principales :

```
DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
GOOGLE_API_KEY      # Gemini
ANTHROPIC_API_KEY   # Haiku
OLLAMA_URL=http://127.0.0.1:11434/api/generate
```

La migration SQL initiale (colonnes `statut_*`, `prompt_version_*`, suppression des `DEFAULT 0` sur les colonnes `note_*`) est décrite en tête de `evaluate_article.py`. Le `DEFAULT 0` est un piège : sans lui, une ligne jamais évaluée naît avec la note 0, c'est-à-dire négative, indiscernable d'un vrai jugement négatif.

## Utilisation

Le pipeline se lance dans cet ordre.

```bash
# 1. Évaluer les couples du périmètre (interactif : modèle, version de prompt, mode)
python evaluate_article.py
PROMPT_VERSION=v7 python evaluate_article.py          # run scripté

# 2. Mode batch (Gemini, Haiku) : soumission, puis récupération plus tard
python evaluate_article.py --collect output/batches_gemini_v7_<date>.json

# 3. Historiser les notes, puis calculer le consensus
python historique_notes.py --creer
python historique_notes.py --amorcer --appliquer
python note_consensus.py --creer-colonnes --appliquer
python note_consensus.py                               # test à blanc

# 4. Comparer les modèles entre eux
python benchmark_classification.py

# 5. Corpus et entraînement du transformer (expérimental)
python preparer_corpus.py
python entrainer_transformer.py
```

Variables d'environnement utiles :

| Variable | Défaut | Rôle |
|---|---|---|
| `PROMPT_VERSION` | *(demandé)* | Version de prompt à utiliser |
| `REPRENDRE` | `1` | `0` pour tout réévaluer, y compris ce qui est déjà fait |
| `LIMITE_PAR_ENTREPRISE` | *(aucune)* | Plafond d'articles par entreprise, pour les tests |
| `BATCH_TAILLE_MAX` | `1000` | Requêtes par batch soumis |
| `RETRY_MAX_TENTATIVES` | `4` | Tentatives avant abandon sur erreur transitoire |
| `GEMINI_MODEL` / `HAIKU_MODEL` | `gemini-2.5-flash` / `claude-haiku-4-5` | Modèles distants |
| `SEUIL_VARIATION_PCT` | `5.0` | Seuil de la règle « mouvement de cours seul » |

Principes de fonctionnement à connaître :

- **Les prompts sont immuables.** Toute évolution crée une nouvelle version dans le registre `PROMPTS` de `llm_common.py`. Le mécanisme de reprise compare la version stockée à celle demandée : modifier un prompt en place ferait croire que tout est à jour.
- **Le LLM constate, le code décide.** À partir de la v6, le modèle extrait des faits (variation de cours, fait concret, recommandation d'analyste) et `note_finale()` applique en Python les règles déterministes, sans nouvel appel d'API.
- **Un échec n'est jamais une note.** Le statut d'une évaluation suit `not_evaluated` → `submitted` → `ok` ou `failed` ; un `failed` est repris au run suivant.
- **Les scripts utilitaires** (récupération de batchs orphelins, diagnostic de quota Gemini, suivi des batchs Haiku, rechargement depuis les CSV, construction des alias) sont décrits dans l'en-tête de chaque fichier.

## Résultats clés

Mesurés contre 191 annotations humaines. Le test est volontairement biaisé vers les cas difficiles (131 désaccords Gemini/Haiku, 60 accords tirés au hasard) : il faut comparer les chiffres bruts entre eux, et lire les estimations repondérées comme un ordre de grandeur (marge d'erreur de 7 à 10 points).

| Mesure | Résultat |
|---|---|
| Accord Gemini / Haiku (prompt v7) | 79,9 %, kappa de Cohen 0,64 |
| Meilleur modèle seul, estimé sur tout le périmètre | Haiku 78,8 % — Gemini 77,9 %, Llama 71,6 %, Qwen 71,0 %, Mistral 50,9 % |
| **Consensus** (médiane Gemini + Haiku + Llama, v7) | **81,1 %** estimé sur le périmètre ; 67,5 % brut sur les 191 (Llama seul : 63,4 %) |
| Consensus quand les trois modèles sont **unanimes** (61 % des couples) | **91,3 %** de justesse, contre 60,0 % sinon |
| Transformer (CamemBERT) entraîné sur le consensus | 54,5 % sur le test humain, kappa 0,24 : non retenu |

Ce que ces chiffres disent : la combinaison de trois modèles bat n'importe quel modèle seul et n'importe quelle variation de prompt, et le désaccord entre modèles est le meilleur détecteur de difficulté disponible, gratuitement. Deux prompts plus détaillés (v8, v9) n'ont rien apporté de significatif.

## Pour aller plus loin

- Épisode 2 — [Construire sa vérité terrain : quel LLM local pour noter l'actualité boursière](https://aventuresdata.com/construire-sa-verite-terrain-quel-llm-local-pour-noter-lactualite-boursiere/)
- Épisode 3 — *Entraîner un Transformer sur sa vérité terrain : ce que trois échecs nous apprennent* (à venir sur le blog)
- Le README documente le code ; le blog documente la démarche, y compris les hypothèses réfutées en route.

## Limites connues et feuille de route

**Limites assumées**

- **Le transformer n'est pas prêt.** Avec 830 exemples d'entraînement jugés sûrs, CamemBERT imite le consensus sur les cas faciles (82,6 %) et le perd sur les cas litigieux (45,5 %). Le verrou est le volume d'étiquettes, pas le code : le projet attend plus d'annotations ou plusieurs mois de consensus sur les 68 sociétés suivies.
- **Le consensus reste faible sur les cas litigieux** : 60 % de justesse quand les modèles divergent. Il faut relire en priorité les couples non unanimes.
- **Les estimations sur le périmètre entier reposent sur 191 annotations**, d'où la marge d'erreur de 7 à 10 points.
- **Le TF-IDF a été retiré de la chaîne quotidienne** le 20 septembre : son pouvoir de détection de difficulté est deux fois plus faible que celui de l'unanimité des modèles (écart de 17,4 points contre 31,3). Le code reste archivé.
- **Pas de CI ni de suite de tests automatisés** sur le pipeline principal à ce jour.
- **Docker partiel** : le `Dockerfile` et le `docker-compose.yml` ne couvrent que l'évaluation par Gemini et ne sont pas alignés sur la version de Python du projet (3.12 dans l'image, 3.13 requis par `pyproject.toml`).

**Pistes**

- Élargir le jeu d'étiquettes dans le périmètre : annoter davantage, ou laisser s'accumuler le consensus quotidien.
- Réentraîner le transformer une fois le volume suffisant, en conservant le test humain intact.
- Porter côté production le rattachement des 34 sociétés ajoutées en local.
- Aligner le rattachement des entreprises entre la production (recherche plein texte PostgreSQL) et le local (expressions régulières).

## Auteur

Manuel — data scientist / MLOps engineer, en transition vers un rôle de Data Product Manager. Blog : [aventuresdata.com](https://aventuresdata.com) · GitHub : [ManDes71](https://github.com/ManDes71)

Licence : à définir (aucun fichier `LICENSE` dans le dépôt à ce jour).
