# Analyse de sentiment CAC 40 par LLM

**Une note de sentiment fiable, entreprise par entreprise, pour les articles de presse financière — construite en mesurant ce que valent les modèles plutôt qu'en leur faisant confiance.**

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)
![Statut](https://img.shields.io/badge/statut-en%20cours-orange)

> Série d'articles associée : *Market-RSS-Sentiment* sur [aventuresdata.com](https://aventuresdata.com). Ce dépôt documente le code ; le blog raconte la démarche.

---

## Sommaire

1. [Contexte](#contexte)
2. [Ce que fait le projet](#ce-que-fait-le-projet)
3. [Architecture et stack](#architecture-et-stack)
4. [Contenu du dépôt](#contenu-du-dépôt)
5. [Installation](#installation)
6. [Utilisation](#utilisation)
7. [Référence de `evaluate_article.py`](#référence-de-evaluate_articlepy)
8. [Programmes par famille](#programmes-par-famille)
9. [Résultats clés](#résultats-clés)
10. [Dépannage](#dépannage)
11. [Pour aller plus loin](#pour-aller-plus-loin)
12. [Limites connues et feuille de route](#limites-connues-et-feuille-de-route)

---

## Contexte

Noter le sentiment d'un article sur une entreprise précise paraît simple, jusqu'à ce qu'on essaie de le mesurer. Un article sur la séance du CAC 40 cite dix valeurs et ne dit rien de chacune d'elles ; un modèle de langage lui attribue pourtant une note. Deux modèles cloud de premier plan ne s'accordent que dans 80 % des cas, et aucun n'est « la vérité ».

Ce projet répond à une question de produit avant d'être une question de modèle : **comment obtenir une note de référence dont on connaît la fiabilité, au meilleur coût ?** Il part de flux RSS, décide quels couples (article, entreprise) méritent d'être notés, les fait noter par plusieurs LLM, et mesure le résultat contre 191 annotations humaines.

## Ce que fait le projet

- **Délimite le périmètre par genre d'article**, pas par comptage d'occurrences : embeddings multilingues, clustering, puis règles ordonnées (`perimetre.py`). Un article de chronique de marché sort ; un article dédié à l'entreprise entre.
- **Fait noter chaque couple par cinq modèles** — Gemini 2.5 Flash et Claude Haiku (API distantes), Llama 3.1 8B, Mistral Nemo et Qwen 2.5 7B (locaux via Ollama) — avec des prompts versionnés et une sortie validée par Pydantic.
- **Exécute en mode synchrone ou batch** (−50 % de coût, jusqu'à 24 h de délai), avec reprise automatique des runs interrompus pour ne jamais refacturer ce qui est déjà fait.
- **Sépare le constat de la décision** : le LLM extrait des faits, une fonction Python applique les règles. Changer un seuil devient une requête SQL, pas 1 300 appels d'API.
- **Historise toutes les notes** par (couple, modèle, version de prompt) pour comparer deux versions d'un prompt par une simple requête SQL.
- **Calcule une note de consensus** : médiane de trois modèles, avec un indicateur d'unanimité qui dit, couple par couple, s'il faut s'y fier.
- **Mesure tout contre une vérité humaine** : 191 couples annotés à la main avec un outil HTML maison, jamais utilisés pour l'entraînement.
- **Compare l'approche LLM à une alternative classique** (TF-IDF + modèle de sentiment finance pré-entraîné), puis l'écarte chiffres à l'appui.
- **Prépare et teste l'entraînement d'un transformer** (CamemBERT) à partir des étiquettes du consensus. Résultat à ce jour : un échec instructif, documenté plus bas.
- **Fournit les outils d'exploitation** d'un pipeline qui paie ses appels : diagnostic de quota, suivi de batchs, récupération de batchs orphelins, reprise des requêtes non soumises.

## Architecture et stack

```
flux RSS (collecte et rattachement par entreprise : projet voisin fluxRss)
        │
        ▼
PostgreSQL  (articles, entreprises + alias, couples article/entreprise, notes)
        │
        ├─ embed_articles.py ─► vecteurs E5 multilingues (pgvector)
        ├─ cluster_genres.py ─► genres d'article (HDBSCAN) ; rattacher_genres.py au quotidien
        │
        ├─ perimetre.py ─────► quels couples méritent d'être notés
        │
        ▼
evaluate_article.py ── providers/ (gemini · haiku · ollama) ── llm_common.py (prompts, validation, retry)
        │                                                        └─ note_finale() : règles déterministes
        │
        ├─ historique_notes.py ─► table article_company_notes (une ligne par couple / modèle / version)
        │
        ▼
note_consensus.py ─► note_consensus + consensus_unanime
        │
        ├─ benchmark_classification*.py ─► comparaison aux annotations humaines
        │
        └─ preparer_corpus.py ─► entrainer_transformer.py (CamemBERT, test sur les 191 annotations)
```

Le cœur est **`evaluate_article.py`**, l'orchestrateur (CLI, base de données, CSV, modes sync et batch). Il délègue à `llm_common.py` tout ce qui est partagé, et aux **providers** tout ce qui dépend d'un fournisseur :

```
evaluate_article.py
    ├─ llm_common.py        prompts v1…v7, validation Pydantic, retry/backoff,
    │                       note_finale(), construction des custom_id de batch
    ├─ perimetre.py         filtre de périmètre (raison_exclusion)
    ├─ entreprises.py       reconnaissance d'une entreprise dans un texte
    ├─ historique_notes.py  historisation des notes
    └─ providers/
        ├─ base.py          interfaces abstraites LLMProvider et BatchLLMProvider
        ├─ ollama.py        modèles locaux (Llama, Mistral, Qwen) : sync uniquement
        ├─ gemini.py        Google Gemini : sync + batch (−50 %)
        └─ haiku.py         Claude Haiku : sync + batch (−50 %)
```

| Domaine | Choix |
|---|---|
| Langage, dépendances | Python 3.12+, gestion par `uv` (`pyproject.toml`) |
| LLM distants | Gemini 2.5 Flash, Claude Haiku 4.5 (appels REST directs, API batch) |
| LLM locaux | Llama 3.1 8B, Mistral Nemo, Qwen 2.5 7B via Ollama (RTX 3060 12 Go) |
| Données | PostgreSQL, extension pgvector pour les embeddings de genre |
| Validation | Pydantic (schéma de sortie, plages de valeurs, énumérations) |
| ML classique | scikit-learn, pandas, matplotlib, seaborn |
| Transformers | PyTorch, Hugging Face `transformers`, `camembert-base`, `bardsai/finance-sentiment-fr-base` |
| Traitement du texte | NLTK, spaCy (modèle français), `intfloat/multilingual-e5-base` |

## Contenu du dépôt

| Dossier / fichier | Contenu |
|---|---|
| `*.py` (racine) | Scripts du pipeline, de mesure et d'exploitation — détaillés plus bas |
| `providers/` | Un module par fournisseur de LLM, derrière une interface commune |
| `schema.sql` | Schéma de base : articles, entreprises, alias, couples, secteurs, vues |
| `Backups/` | Exports SQL des tables (articles, entreprises, couples, alias, secteurs, flux RSS) |
| `_output/` | **Résultats archivés des runs de référence** : CSV de benchmark, notes par modèle et par version, journaux, graphiques, export de l'outil d'annotation |
| `outil_annotation.html` | Outil HTML autonome d'annotation humaine des désaccords |
| `*_a_annoter.csv / .xlsx` | Échantillons à annoter (zone grise, contrôle du consensus) |
| `Dockerfile`, `docker-compose.yml` | Conteneur d'évaluation (voir [Docker](#docker-partiel)) |
| `pyproject.toml` | Dépendances et configuration `uv` |

> **`output/` ou `_output/` ?** Les scripts lisent et écrivent dans `output/` (créé à la volée par `evaluate_article.py`). Le dépôt versionne les archives des runs dans `_output/` : pour rejouer les scripts de mesure sur ces données, copiez ou renommez ce dossier en `output/`.

## Installation

Prérequis : Python 3.12 ou plus, [`uv`](https://docs.astral.sh/uv/), une base PostgreSQL alimentée par le projet de collecte, et pour les modèles locaux un serveur [Ollama](https://ollama.com) avec `llama3.1:8b`, `mistral-nemo` et `qwen2.5:7b`.

```bash
uv sync                      # crée .venv à partir de pyproject.toml
source .venv/bin/activate
```

### Configuration locale

**Les identifiants et les clés d'API ne doivent jamais être versionnés.** Le dépôt ne fournit pas de modèle de configuration : créez un fichier `.env` à la racine (il est dans `.gitignore`) ; `load_dotenv()` le lit à l'exécution.

```
DB_HOST=localhost
DB_PORT=5432
DB_USER=...
DB_PASSWORD=...
DB_NAME=...

GOOGLE_API_KEY=...        # pour Gemini
ANTHROPIC_API_KEY=...     # pour Haiku
OLLAMA_URL=http://127.0.0.1:11434/api/generate   # pour les modèles locaux
```

Lancé depuis un autre dossier que celui du `.env`, un script se rabat sur la socket PostgreSQL locale avec un message d'erreur trompeur : lancez-le depuis la racine du dépôt.

### Base de données

`schema.sql` crée les tables de base (`articles_rss`, `companies`, `company_aliases`, `article_companies`, `sectors`, `rss_feeds`, …) et les vues. Deux points à connaître :

- **Migration à passer une seule fois** (décrite en tête de `evaluate_article.py`) : colonnes `statut_*` et `prompt_version_*`, suppression des contraintes `NOT NULL` et des `DEFAULT 0` sur les colonnes `note_*`. Le `DEFAULT 0` est un piège : sans cette migration, une ligne jamais évaluée naît avec la note 0, c'est-à-dire négative, indiscernable d'un vrai jugement négatif. `schema.sql` contient déjà les colonnes `statut_*` et `prompt_version_*`, mais **conserve les `NOT NULL DEFAULT 0` sur les `note_*`** : cette partie-là reste à passer.
- **Les tables du chantier « genre »** (`article_embeddings`, `article_genres`, extension pgvector) ne sont pas créées par ce dépôt : `embed_articles.py`, `cluster_genres.py` et `rattacher_genres.py` supposent qu'elles existent. La table d'historique `article_company_notes`, elle, se crée avec `historique_notes.py --creer`.

### Docker (partiel)

`Dockerfile` (Python 3.12 + `uv`) et `docker-compose.yml` exécutent **uniquement l'évaluation par Gemini** : le service `evaluator` lit `DB_*` et `GOOGLE_API_KEY` dans l'environnement, joint un PostgreSQL de l'hôte via `host.docker.internal` et monte `./output`. Il se rattache à un réseau Docker externe nommé `docker_postgresql_webproxy`, à créer ou à adapter. Haiku, Ollama, le consensus et les benchmarks ne sont pas conteneurisés.

## Utilisation

Toutes les commandes se lancent **depuis la racine du dépôt**, environnement virtuel activé (ou préfixées par `uv run`).

### Le pipeline, dans l'ordre

```bash
# 0. Genres et périmètre (pgvector requis) — à refaire à chaque arrivée d'articles
python embed_articles.py --dry-run          # combien d'articles à vectoriser ?
python embed_articles.py
python rattacher_genres.py                  # à blanc + CSV de relecture
python rattacher_genres.py --appliquer

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

`evaluate_article.py` n'a pas d'option `--modele` : il pose la question au démarrage. Pour scripter un run, on lui répond par l'entrée standard :

```bash
printf 'gemini\n' | PROMPT_VERSION=v7 MODE=batch python evaluate_article.py
printf 'haiku\n'  | PROMPT_VERSION=v7 MODE=batch python evaluate_article.py
printf 'lama\n'   | PROMPT_VERSION=v7 MODE=sync  python evaluate_article.py   # Llama, local
```

Les noms de modèles acceptés sont `gemini`, `haiku`, `lama`, `mistral` et `queen` (Qwen).

### Mise à jour périodique de la base

Quand de nouveaux articles arrivent (collecte par le projet voisin fluxRss), chaque étape suivante est rejouable sans risque : elle ne traite que ce qui manque.

1. **Sauvegarder** la base juste avant d'écrire.
2. **Vectoriser** les nouveaux articles (`embed_articles.py`), puis **rattacher leur genre** (`rattacher_genres.py`, à blanc d'abord).
3. **Lire l'alerte de `rattacher_genres.py`** avant d'écrire : si la part d'« articles dédiés » du jour dépasse 60 %, un nouveau genre est peut-être apparu. Dans ce cas, ne pas écrire : voir la docstring du script pour la marche à suivre (relancer HDBSCAN sans `--ecrire` et chercher des clusters formés surtout d'articles récents).
4. **Lancer Gemini et Haiku en batch**, puis **Llama en local** pendant que les batchs tournent.
5. **Récupérer les batchs** (`--collect`, non bloquant : à relancer tant qu'ils sont en cours).
6. **Calculer le consensus** une fois les *trois* notes présentes (`note_consensus.py`, à blanc puis `--appliquer`).

Deux contrôles évitent les mauvaises surprises :

- **Avant l'étape 4, vérifier que Ollama tourne sur le GPU** (voir [Dépannage](#llama-se-fige-ou-tourne-50-fois-plus-lentement)). Le calcul des vecteurs occupe le GPU : si Ollama charge Llama juste après, il peut se rabattre sur le CPU.
- **Lire le contrôle affiché par `note_consensus.py`** : à chaque exécution, même à blanc, il mesure le consensus et chaque modèle sur les couples annotés à la main. Le consensus doit rester à **67,5 %** (brut, sur les 191 annotations) tant qu'aucune annotation n'est ajoutée. S'il bouge, une note historisée a changé : en chercher la cause avant `--appliquer`.

## Référence de `evaluate_article.py`

### Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `PROMPT_VERSION` | *(demandé)* | Version de prompt : `v1` à `v7` (référence : `v7`) |
| `MODE` | *(demandé)* | `sync` ou `batch` (batch : Gemini et Haiku seulement) |
| `REPRENDRE` | `1` | `0` pour tout réévaluer, y compris ce qui est déjà fait |
| `LIMITE_PAR_ENTREPRISE` | *(aucune)* | Plafond d'articles par entreprise, pour les tests |
| `BATCH_TAILLE_MAX` | `1000` | Requêtes par batch soumis |
| `BATCH_POLL_INTERVAL` / `BATCH_TIMEOUT` | `30` s / `24` h | Cadence et délai maximal de l'attente d'un batch |
| `RETRY_MAX_TENTATIVES` | `4` | Tentatives avant abandon sur erreur transitoire (429, 5xx) |
| `RETRY_DELAI_BASE` / `RETRY_DELAI_MAX` | `2` s / `60` s | Backoff exponentiel entre tentatives |
| `GEMINI_MODEL` / `HAIKU_MODEL` | `gemini-2.5-flash` / `claude-haiku-4-5` | Modèles distants |
| `SEUIL_VARIATION_PCT` | `5.0` | Seuil de la règle « mouvement de cours seul » (`note_finale`) |
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | — | Connexion PostgreSQL |
| `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_URL` | — | Accès aux fournisseurs |

### Versions de prompt

Les prompts sont **communs à tous les modèles** et vivent dans le registre `PROMPTS` de `llm_common.py`.

| Version | Contenu | Enseignement |
|---|---|---|
| `v1` | Règles courtes, NEUTRAL par défaut en cas de doute | Baseline simple, biais vers NEUTRAL |
| `v2` | NEUTRAL défini restrictivement, isolation des phrases visant l'entreprise, few-shot, règle anti-repli-neutre | Optimum mesuré pour les modèles locaux |
| `v3` | v2 + règles sur les objectifs de cours et recommandations de brokers | Dégrade les modèles locaux (kappa de Llama en zone de consensus : 0,707 → 0,549) |
| `v4` | v3 corrigé : étape de lecture factuelle explicite et grille complète 3 × 4 (sens de l'objectif × évolution de la recommandation) | Kappa de Llama à 0,490. Modifiée en place par erreur : origine de la règle d'immutabilité |
| `v5` | v2 + trois règles tirées de 101 cas annotés : mouvement de cours intraday, palmarès de valeurs, seuil de matérialité | La règle intraday, en langage naturel, a produit l'effet inverse sur Llama |
| `v6` | v2 + **extraction structurée** (variation de cours, autre fait concret, sens de la recommandation, niveau), sans règle de décision dans le prompt | « Le LLM constate, le code décide » : `note_finale()` applique les règles |
| `v7` | v6 + définition corrigée du champ « autre fait concret » (interdiction de compter la variation de cours, test de suppression des pourcentages, contre-exemples réels) | ✅ **Référence** du consensus |

Deux versions ultérieures ont été testées sans gain significatif et ne figurent pas dans le code de ce dépôt : une `v8` (v7 + étape « entreprise visée, entreprise source, matérialité » : match nul, Gemini plus sévère) et une `v9` (v7 + règle de matérialité seule : trop de neutres chez Haiku).

**Règle absolue : ne jamais modifier un prompt existant en place.** Toute évolution crée une nouvelle version. Le mécanisme de reprise compare la version stockée en base à celle demandée ; modifier `v4` sans le renommer ferait croire au script que tout est à jour alors que les notes proviennent d'un prompt différent.

### Modes d'exécution

**`sync`** — appel immédiat, article par article. Seul mode des modèles locaux. Les résultats sont écrits au fil de l'eau.

```
evaluate_article.py
    ├─ Récupère les articles (filtre de reprise appliqué par défaut)
    ├─ Pour chaque couple :
    │   ├─ Filtre de périmètre (perimetre.raison_exclusion)
    │   ├─ provider.evaluer_un()
    │   │    ├─ construit le prompt (llm_common.PROMPTS)
    │   │    ├─ appelle le LLM (ollama / gemini / haiku), avec retry/backoff
    │   │    ├─ valide la réponse (Pydantic : note ∈ {0, 1, 2})
    │   │    └─ retourne (note, statut)
    │   ├─ UPDATE article_companies + historisation
    │   └─ écrit en CSV et dans le journal
    └─ terminé immédiatement
```

**`batch`** — réservé à Gemini et Haiku. Soumission par lot à **−50 % de coût**, délai jusqu'à 24 h (en pratique de quelques minutes à 4 h). Deux temps :

```
1. Soumission (retour immédiat)
   evaluate_article.py
    ├─ récupère les couples à évaluer
    ├─ provider.soumettre_batch()  →  BATCH_TAILLE_MAX requêtes par lot, un batch_id par lot
    └─ écrit output/batches_{modele}_{version}_{date}.json

2. Récupération (non bloquante, à relancer tant que des batchs sont en cours)
   evaluate_article.py --collect output/batches_{modele}_{version}_{date}.json
    ├─ interroge chaque batch (provider.statut_batch)
    ├─ récupère les résultats des batchs terminés (provider.recuperer_resultats)
    └─ écrit en base et en CSV
```

Le batch continue de tourner côté fournisseur même machine éteinte. La récupération est idempotente : les batchs déjà collectés sont ignorés.

### Reprise automatique des runs interrompus

Activée par défaut (`REPRENDRE=1`). Le script ne traite que les couples qui ne sont **pas** déjà en `statut = 'ok'` ou `'submitted'` **pour la version de prompt demandée**. Relancer après une coupure ne retraite que le reliquat ; changer de version réévalue tout, puisque les notes ne sont plus comparables.

```
not_evaluated  ──►  submitted  ──►  ok
                                └►  failed  ──► (repris au run suivant)
```

`submitted` est écrit dès la soumission d'un batch : il empêche un second run, lancé pendant que le batch est en vol, de resoumettre — et refacturer — les mêmes articles. Pour forcer une réévaluation complète (après correction d'un bug de parsing, par exemple) : `REPRENDRE=0 PROMPT_VERSION=v7 python evaluate_article.py`.

### Sorties produites

| Fichier | Contenu |
|---|---|
| `output/resultats_{modele}_{version}_{date}.csv` | Une ligne par évaluation : note, justification, statut |
| `output/log_articles_{modele}_{version}_{date}.txt` | Journal détaillé, y compris les articles filtrés |
| `output/batches_{modele}_{version}_{date}.json` | Métadonnées de batch (mode batch uniquement) |

Les colonnes `note_*` d'`article_companies` ne conservent que le **dernier** run. L'historique complet, par version de prompt, est dans `article_company_notes` (`historique_notes.py`) : comparer deux versions est une requête SQL.

### Suivi et diagnostic

```sql
-- Avancement par modèle
SELECT statut_gemini, prompt_version_gemini, COUNT(*)
FROM public.article_companies GROUP BY 1,2 ORDER BY 3 DESC;

-- Reste à traiter pour une version donnée
SELECT COUNT(*)
FROM public.article_companies ac
JOIN public.articles_rss a ON a.id = ac.article_id
WHERE a.contenu IS NOT NULL
  AND (ac.statut_gemini IS DISTINCT FROM 'ok'
       OR ac.prompt_version_gemini IS DISTINCT FROM 'v7');
```

Le nombre annoncé en fin de collecte (« Collecte terminée : N article(s) à évaluer ») correspond aux couples du périmètre sans note pour la version demandée : il devrait être le même pour Gemini et Haiku.

### Principes à connaître

- **Le périmètre est décidé par `perimetre.py`** : genre de l'article (chroniques, palmarès… exclus), puis nom de l'entreprise dans le titre. `nbocc` n'est plus un filtre, seulement une information.
- **Le LLM constate, le code décide.** À partir de la v6, le modèle extrait des faits et `note_finale()` applique en Python les règles déterministes, sans nouvel appel d'API. `recalculer_notes.py` rejoue ces règles avec un autre seuil.
- **Un échec n'est jamais une note.** Un `failed` est repris au run suivant.
- **Aucune limite par entreprise** : tout le corpus est traité (`LIMITE_PAR_ENTREPRISE` sert aux tests).

## Programmes par famille

### Évaluation et consensus

| Programme | Rôle |
|---|---|
| `evaluate_article.py` | Orchestrateur : CLI interactive, accès BDD, CSV, modes sync et batch, reprise |
| `llm_common.py` | Prompts versionnés, validation Pydantic, retry/backoff, `custom_id` de batch, `note_finale()` |
| `providers/base.py` | Interfaces abstraites `LLMProvider` et `BatchLLMProvider` |
| `providers/ollama.py` | Modèles locaux gratuits, sync uniquement |
| `providers/gemini.py` | Google Gemini : sync + batch par fichier JSONL (−50 %) |
| `providers/haiku.py` | Claude Haiku : sync + Message Batches API (−50 %) |
| `historique_notes.py` | Table `article_company_notes` : une ligne par (couple, modèle, version). Options `--creer`, `--amorcer`, `--etat`, `--restaurer VERSION`, `--appliquer` |
| `note_consensus.py` | Médiane de trois modèles (Gemini, Haiku, Llama, en v7 par défaut) et indicateur `consensus_unanime`. Options `--creer-colonnes`, `--version`, `--modeles`, `--appliquer` |
| `recalculer_notes.py` | Recalcule les notes depuis les faits déjà extraits, sans appel d'API : `python recalculer_notes.py gemini --seuil 10 --dry-run`, ou `--comparer 3 5 8 10 15` pour comparer plusieurs seuils |

Pourquoi la **médiane de trois** et pas la moyenne : sur une échelle ordinale (0 négatif, 1 neutre, 2 positif), la médiane renvoie toujours une note valide et revient à un vote majoritaire dès que deux modèles s'accordent ; la moyenne inventerait des valeurs intermédiaires sans sens. Les notes sont lues dans `article_company_notes`, jamais dans `article_companies` : l'historique permet de fixer la version de référence indépendamment du dernier run en date.

### Périmètre : genres d'articles et entreprises

| Programme | Rôle |
|---|---|
| `embed_articles.py` | Vectorise les articles (`intfloat/multilingual-e5-base`, 768 dimensions, normalisation L2) dans `article_embeddings`. Options `--dry-run`, `--limit`, `--force`, `--batch-size` |
| `cluster_genres.py` | Regroupe les articles par genre (HDBSCAN, ou zero-shot) ; sort des clusters, qu'un humain étiquette. Options `--explorer`, `--detailler`, `--ecrire` |
| `rattacher_genres.py` | **À lancer chaque jour** : rattache les nouveaux articles au genre du centre de cluster le plus proche, sans relancer HDBSCAN (qui renumérote les clusters et change le genre d'articles déjà relus). Alerte si la part d'articles dédiés du jour dépasse 60 % |
| `perimetre.py` | Décide, couple par couple, s'il est admis, en franchissant dans l'ordre : genre exclu, titre de chronique, titre de liste, corps absent, entreprise absente du titre. Remplace le seuil `nbocc > 2`, qui se trompait dans les deux sens |
| `entreprises.py` | Reconnaissance du nom d'une entreprise dans un texte (suffixes de cotation, formes juridiques, accents, alias), module léger partagé par plusieurs scripts |
| `societes_suivies.py` | Ajoute des sociétés à l'univers suivi, avec leurs alias (`--appliquer`, `--isin fichier.csv`) |
| `creer_couples.py` | Crée les couples (article, entreprise) manquants pour les sociétés qui n'en ont pas (`--societes`, `--appliquer`) |
| `build_alias_db.py` | Demande à Gemini les alias et périphrases de chaque entreprise à partir d'articles échantillons, et les écrit dans `alias.json`. Payant |
| `realigner_sur_prod.py` | Réaligne une base locale sur les identifiants de la base de production, à partir d'exports CSV, et y importe les articles et couples nouveaux (`--appliquer`) |
| `set_company_occurs.py` | ⚪ Archive : calcule `nbocc`, désormais purement informatif |

### Exploitation des batchs

| Programme | Rôle |
|---|---|
| `verif_limit_batch.py` | Liste les jobs batch Gemini et leur état ; `--annuler` annule ceux qui traînent |
| `verif_batch_haiku.py` | Statut des batchs Haiku, lus depuis les fichiers `output/batches_haiku_*.json` ou un `batch_id` direct |
| `recuperer_batch_orphelin.py` | Reconstruit le fichier de métadonnées de batchs Gemini soumis mais jamais reliés aux articles |
| `reprendre_non_soumis.py` | Remet à `not_evaluated` les requêtes qu'un chunk échoué n'a pas soumises, pour que la reprise les retraite |
| `reload_article_companies_notes.py` | Recharge les notes en base depuis les CSV `output/resultats_*.csv`, sans rappeler les LLM (`--dry-run` pour simuler) |

**Le problème du quota fantôme.** Une erreur `429 RESOURCE_EXHAUSTED — Token quota exceeded` à la soumission d'un batch Gemini alors qu'on ne lance rien : des jobs antérieurs occupent encore la file. Gemini compte les jetons « mis en file d'attente » (3 M en Niveau 1 pour Gemini 2.5 Flash) tant que le job n'a pas atteint un état terminal (`SUCCEEDED`, `FAILED`, `CANCELLED`, `EXPIRED`). `verif_limit_batch.py` les liste, `--annuler` les libère (sans confirmation : à relire avant).

**Le batch orphelin.** Si la soumission échoue au milieu de la boucle de chunks (un 429 au chunk 3, par exemple), les chunks déjà soumis — et facturés — tournent chez le fournisseur sans qu'aucun fichier local ne relie leur `batch_id` aux articles : `--collect` devient impossible. `recuperer_batch_orphelin.py` rejoue la liaison à partir des `custom_id` du fichier de résultats :

```bash
python recuperer_batch_orphelin.py v7 "<batch_id>" ["<batch_id>" …] [--debug]
python evaluate_article.py --collect output/batches_gemini_v7_<date>_RECUPERE.json
```

Pour Haiku, qui n'offre pas de listing global des batchs, le suivi se fait depuis les fichiers de métadonnées locaux :

```bash
python verif_batch_haiku.py                                   # tous les batchs locaux
python verif_batch_haiku.py output/batches_haiku_v7_*.json    # fichiers précis
python verif_batch_haiku.py msgbatch_XXXXXXXXXX               # batch_id direct
```

### Mesure, annotation et benchmark

| Programme | Rôle |
|---|---|
| `benchmark_classification.py` | Compare les modèles entre eux : accuracy, kappa de Cohen, matrices de confusion, rapport de classification, biais, zone de consensus. Génère CSV, graphiques et rapport HTML. Référence principale : Gemini ; seconde : Haiku |
| `benchmark_classification_gemini.py`, `benchmark_classification_haiku.py` | Variantes du benchmark avec Gemini, ou Haiku, comme référence principale |
| `benchmark_classification_tfidf.py` | Compare les notes TF-IDF aux LLM (voir [ci-dessous](#alternative-tf-idf-et-pourquoi-elle-a-été-écartée)) |
| `annoter_desaccords.py` | Prépare l'annotation humaine : `--desaccords` (défaut : couples où Gemini et Haiku divergent), `--accords [N]` (tirage au hasard parmi les accords), puis `--importer FICHIER --appliquer` pour réimporter les notes relues dans `note_humaine` |
| `outil_annotation.html` | Outil autonome d'annotation, ouvert dans un navigateur |
| `preparer_controle_consensus.py` | Tire un échantillon de contrôle dans la zone de consensus, stratifié sur les cas litigieux |
| `calcul_fiabilite_regle_vs_modele.py` | Compare la justesse des notes imposées par la règle déterministe à celle des notes tranchées par le modèle seul |
| `tableau_accord_references.py`, `tableau_controle_consensus.py`, `tableau_qualite_etiquette.py`, `tableau_regle_intraday.py`, `tableau_renversement.py` | Reproduisent, chiffre par chiffre, les tableaux des articles du blog à partir des CSV et annotations archivés |

### Alternative TF-IDF, et pourquoi elle a été écartée

`classifier.py` note chaque couple sans appel d'API : un résumé TF-IDF (`note_tfidf`) et le modèle de sentiment finance pré-entraîné `bardsai/finance-sentiment-fr-base` appliqué au texte complet (`note_full`), toutes deux normalisées sur l'échelle {0, 1, 2}.

```bash
python classifier.py                  # toutes les entreprises
python classifier.py 15               # une seule entreprise (ex. TotalEnergies)

# Restreindre exactement au périmètre d'un benchmark LLM : même ensemble d'articles,
# donc aucun biais d'échantillon, et pas de filtre sur nbocc (qui change avec le temps)
python classifier.py --subset-csv output/benchmark_classification_V7.csv
python classifier.py --subset-csv output/benchmark_classification_V7.csv 15

# Puis comparer aux LLM
python benchmark_classification_tfidf.py
```

Sorties : `output/comparaison_sentiment_{...}_{date}.csv`, `output/benchmark_comparatif_tfidf_vs_llm.csv` et le rapport `output/benchmark_rapport_tfidf_vs_llm.html`. Métriques : accuracy, kappa, accuracy en zone de consensus, couverture, biais net, erreurs polaires (inversions complètes).

**Verdict** : les LLM surpassent clairement le TF-IDF (Qwen à 72,5 % contre 52,6 % pour le TF-IDF sur texte complet, par rapport à la baseline Gemini). Le TF-IDF a donc été retiré de la chaîne quotidienne le 20 septembre 2026, et sa variante « ciblée » (`note_targeted`) abandonnée ; le code reste archivé.

### Transformer (expérimental)

| Programme | Rôle |
|---|---|
| `preparer_corpus.py` | Construit les trois jeux : entraînement (couples du périmètre notés à l'unanimité, étiquette = consensus), validation (tirage stable par md5) et test (les 191 annotations humaines, jamais utilisées pour l'entraînement) |
| `entrainer_transformer.py` | Fine-tuning de CamemBERT par une boucle PyTorch explicite, pondération des classes, mesure contre les annotations humaines. Options `--epochs`, `--batch`, `--lr`, `--fraction`, `--graine`, `--sans-sauvegarde` |

L'entrée du modèle est le point délicat : 57 % des articles dépassent la fenêtre de 512 tokens, et dans 7 % des cas l'essentiel de l'argument est au-delà des 1 500 premiers caractères. Plutôt que de tronquer le début, le modèle reçoit *entreprise — titre — phrases qui citent l'entreprise et leurs voisines*.

## Résultats clés

Mesurés contre 191 annotations humaines. Le test est volontairement biaisé vers les cas difficiles (131 désaccords Gemini/Haiku, 60 accords tirés au hasard) : il faut comparer les chiffres bruts entre eux, et lire les estimations repondérées comme un ordre de grandeur (marge d'erreur de 7 à 10 points).

| Mesure | Résultat |
|---|---|
| Accord Gemini / Haiku (prompt v7) | 79,9 %, kappa de Cohen 0,64 |
| Meilleur modèle seul, estimé sur tout le périmètre | Haiku 78,8 % — Gemini 77,9 %, Llama 71,6 %, Qwen 71,0 %, Mistral 50,9 % |
| **Consensus** (médiane Gemini + Haiku + Llama, v7) | **81,1 %** estimé sur le périmètre ; 67,5 % brut sur les 191 (Llama seul : 63,4 %) |
| Consensus quand les trois modèles sont **unanimes** (61 % des couples) | **91,3 %** de justesse, contre 60,0 % sinon |
| TF-IDF + sentiment finance pré-entraîné, texte complet | 52,6 % (contre 72,5 % pour Qwen) : écarté |
| Transformer (CamemBERT) entraîné sur le consensus | 54,5 % sur le test humain, kappa 0,24 : non retenu |

Ce que ces chiffres disent : la combinaison de trois modèles bat n'importe quel modèle seul et n'importe quelle variation de prompt, et le désaccord entre modèles est le meilleur détecteur de difficulté disponible, gratuitement. Deux prompts plus détaillés (v8, v9) n'ont rien apporté de significatif. Autre leçon de méthode : le classement des modèles locaux *contre Haiku* sur tout le corpus ne coïncide pas avec leur classement *contre la vérité humaine* sur les cas litigieux (Mistral Nemo, dernier sur le premier, premier sur le second) — c'est ce qui a conduit à ajouter un échantillon de contrôle dans la zone de consensus.

## Dépannage

### Llama se fige ou tourne 50 fois plus lentement

Symptômes : le journal `output/log_articles_lama_*.txt` n'avance plus, ou le modèle est très lent. Vérifier où il tourne :

```bash
curl -s http://127.0.0.1:11434/api/ps      # size_vram doit être > 0
```

`size_vram: 0` signifie que le modèle est chargé sur le **CPU**. Les appels à Ollama n'ont pas de délai d'expiration : le script attend indéfiniment. Arrêter le run (Ctrl+C), réinitialiser Ollama dans l'environnement où il tourne, revérifier `size_vram`, relancer : seuls les couples restants sont repris. Le compte de notes du journal peut être faux après un arrêt (tampon non écrit) : la base fait foi.

```bash
sudo systemctl stop ollama
sudo pkill -f ollama
sudo systemctl start ollama
```

### Erreur 429 à la soumission d'un batch Gemini

Voir [le problème du quota fantôme](#exploitation-des-batchs) : `python verif_limit_batch.py`, puis `--annuler`, puis attendre environ une minute.

### Un run batch a échoué au milieu de la soumission

`recuperer_batch_orphelin.py` pour relier les chunks déjà soumis, puis `reprendre_non_soumis.py <fichier_metadonnees>` pour que la reprise retraite exactement les requêtes restantes. Sans ce second script, il n'y a que deux issues, toutes deux mauvaises : relancer normalement (0 article à évaluer, on croit à tort avoir fini) ou relancer avec `REPRENDRE=0` (tout est resoumis, y compris ce qui vient d'être payé).

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
- **La mise à jour de la base est manuelle.** Elle n'est pas automatisée : on la lance à la main, de temps en temps, étape par étape (voir [Mise à jour périodique](#mise-à-jour-périodique-de-la-base)).
- **Pas de CI ni de suite de tests automatisés** sur le pipeline principal à ce jour.
- **Docker partiel** : le `Dockerfile` et le `docker-compose.yml` ne couvrent que l'évaluation par Gemini.
- **Dépôt non autonome** : les tables pgvector (`article_embeddings`, `article_genres`) et les données de la base ne sont pas fournies, et aucun fichier `.env.example` n'est versionné.

**Pistes**

- Élargir le jeu d'étiquettes dans le périmètre : annoter davantage, ou laisser s'accumuler le consensus quotidien.
- Réentraîner le transformer une fois le volume suffisant, en conservant le test humain intact.
- Porter côté production le rattachement des 34 sociétés ajoutées en local.
- Aligner le rattachement des entreprises entre la production (recherche plein texte PostgreSQL) et le local (expressions régulières).
- Versionner un `.env.example` et le DDL des tables pgvector pour rendre le dépôt rejouable de bout en bout.

## Auteur

Manuel — data scientist / MLOps engineer, en transition vers un rôle de Data Product Manager. Blog : [aventuresdata.com](https://aventuresdata.com) · GitHub : [ManDes71](https://github.com/ManDes71)

Licence : à définir (aucun fichier `LICENSE` dans le dépôt à ce jour).
