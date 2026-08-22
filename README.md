# Analyse de sentiment CAC 40 par LLM

Pipeline d'evaluation du sentiment financier d'articles RSS par cinq modeles de
langage (deux API distantes, trois modeles locaux), avec comparaison de leurs
performances.

> Identifiants de connexion et notes d'environnement : voir `NOTES-LOCALES.md`
> (non versionne, present uniquement sur la machine de developpement).

## 📊 STRUCTURE GÉNÉRALE

```
evaluate_article.py (orchestrateur : CLI, BDD, CSV, modes sync/batch)
    ├─ llm_common.py (module partagé)
    │  ├─ PROMPTS (v1 a v4, communs a tous les modeles)
    │  ├─ Validation (pydantic BaseModel)
    │  ├─ Retry/Backoff
    │  └─ Construction custom_id pour batchs
    │
    └─ providers/ (abstraction des LLM)
       ├─ base.py (classe abstraite Provider)
       ├─ ollama.py (Lama, Mistral, Queen locaux)
       ├─ gemini.py (Google Gemini API distante)
       └─ haiku.py (Claude Haiku API distante)
```

---


## 🔄 FLUX D'EXÉCUTION

### Mode SYNC (temps réel)

```
evaluate_article.py
    ├─ Récupère articles (filtre de reprise applique par defaut)
    ├─ Pour chaque article :
    │  ├─ Filtre nbocc > 1
    │  ├─ Appelle provider.evaluer_un()
    │  │  ├─ Construit prompt (llm_common.PROMPTS)
    │  │  ├─ Appelle LLM (ollama / gemini / haiku)
    │  │  ├─ Valide réponse (llm_common.EvaluationSentiment)
    │  │  └─ Retourne (note, statut)
    │  ├─ Sauvegarde en BDD (UPDATE article_companies)
    │  ├─ Écrit en CSV
    │  └─ Log article
    └─ Terminé immédiatement
```

### Mode BATCH (différé, -50% coût)

```
evaluate_article.py
    ├─ Récupère articles (filtre de reprise applique par defaut)
    ├─ Appelle provider.soumettre_batch()
    │  ├─ Construit les requetes JSON (BATCH_TAILLE_MAX par lot)
    │  ├─ Soumet batch à API (Gemini / Haiku)
    │  └─ Reçoit batch_id
    ├─ Sauvegarde métadonnées (batches_XXX.json)
    └─ Terminé (batch tourne côté serveur)

PLUS TARD :
evaluate_article.py --collect batches_XXX.json
    ├─ Récupère batch_id
    ├─ Appelle provider.recuperer_resultats()
    ├─ Récupère résultats (une fois SUCCEEDED)
    ├─ Sauvegarde en BDD
    └─ Écrit en CSV
```

---

## 📋 RÉSUMÉ INTERACTIONS

```
evaluate_article.py (orchestre)
        ↓
    llm_common.py (config partagée)
        ├─ PROMPTS (v1 a v4)
        ├─ EvaluationSentiment (validation)
        ├─ Retry/Backoff
        └─ Batch construction
        ↓
    providers/ (abstraction LLM)
        ├─ ollama.py (local, gratuit, sync seul)
        ├─ gemini.py (cloud, sync+batch)
        └─ haiku.py (cloud, sync+batch)
```

---

## 🎯 POINTS CLÉS

1. **nbocc > 1 obligatoire** : l'entreprise doit etre citee plus d'une fois, pour ecarter les faux positifs de jointure (mention en passant dans une revue de marche)
2. **Aucune limite par entreprise** : tout le corpus est traite (`LIMITE_PAR_ENTREPRISE` pour restreindre en phase de test)
3. **Quatre versions de prompts** : v1 basique -> v4 (regles brokers + grille complete)
4. **Deux modes** : SYNC (immédiat) vs BATCH (-50% coût, +24h délai)
5. **Validation stricte** : pydantic vérifie note ∈ {0,1,2}
6. **Retry/Backoff** : gère automatiquement les erreurs API (429, 5xx)
7. **Logging complet** : chaque article tracé pour diagnostiquer
8. **Runs reprenables** : un run interrompu se relance sans retraiter (ni refacturer) ce qui est deja fait — voir `REPRENDRE`

---

## 📊 TABLEAU RÉCAPITULATIF

| Composant | Rôle |
|-----------|------|
| `evaluate_article.py` | Orchestre : CLI, acces BDD, export CSV, modes sync et batch, reprise |
| `llm_common.py` | Prompts versionnes, validation pydantic, retry/backoff, custom_id |
| `providers/base.py` | Interfaces abstraites `LLMProvider` et `BatchLLMProvider` |
| `providers/ollama.py` | Modeles locaux gratuits (sync uniquement) |
| `providers/gemini.py` | Google Gemini (sync + batch a -50%) |
| `providers/haiku.py` | Claude Haiku (sync + batch a -50%) |

---
#### Exemple :
```bash
REMOVED@DESKTOP-FULACPK:~/projets/classification_textes$ python3 evaluate_article.py

Modèles disponibles :
  gemini  -> Google Gemini 2.5 Flash (API distante)
  haiku   -> Claude Haiku (API distante Anthropic)
  lama    -> LLaMA 3 via Ollama (local)
  mistral -> Mistral Nemo via Ollama (local)
  queen   -> Qwen 2.5 7B via Ollama (local)

Choisissez le modèle [gemini/haiku/lama/mistral/queen] : gemini

Versions de prompt disponibles :
  v1 -> Prompt original (règles courtes, NEUTRAL par défaut)
  v2 -> Prompt enrichi (isolation des phrases pertinentes, few-shot, anti-repli-neutre)
  v3 -> v2 + règles dédiées objectifs de cours / recommandations de brokers

Choisissez la version de prompt [v1/v2/v3] : v2

Modes disponibles :
  sync  -> appel immédiat article par article (comme avant)
  batch -> soumission par lot (-50% de coût, jusqu'à 24h de délai)

Choisissez le mode [sync/batch] : batch

Modèle sélectionné : gemini (mode batch — soumission uniquement)
Connexion à la BDD...

--- Phase 1/2 : collecte des articles éligibles ---
Collecte terminée : 1281 article(s) à évaluer, 1595 filtré(s).

--- Phase 2/2 : soumission des batchs ---
Soumission du chunk 1/2 (1000 requête(s))...
  -> batch_id=batches/5t1r7kh780i0m2wxbsixqchmcecxtlbukhx6
Soumission du chunk 2/2 (281 requête(s))...
  -> batch_id=batches/a8818d0aic680bag4hrzxiu3r9a99elyg2n2

1281 article(s) soumis dans 2 batch(s).
Métadonnées : output/batches_gemini_v2_20260813_212410.json
Le batch continue de tourner côté fournisseur même si cette machine s'éteint. Pour récupérer les résultats (à relancer tant que des batchs sont en cours, ex. via une tâche planifiée) :
  python evaluate_article.py --collect output/batches_gemini_v2_20260813_212410.json
```


resultats du batch :

```bash
# important pour commencer
source .venv/bin/activate
# Pour Gemini (listing centralisé des batchs)
python3 verif__limit_batch.py              # liste les jobs batch Gemini et leur état
python3 verif__limit_batch.py --annuler    # annule les jobs qui traînent (libère quota)

# Pour Haiku (lecture depuis fichiers JSON locaux)
python3 verif_batch_haiku.py                          # statut de tous les batchs Haiku
python3 verif_batch_haiku.py output/batches_haiku_v3_*.json  # fichiers spécifiques
python3 verif_batch_haiku.py msgbatch_XXXXXXXXXX     # batch_id direct

# Récupérer batchs orphelins
python3 recuperer_batch_orphelin.py v4 batches/xxxx   # reconstruit les métadonnées d'un batch perdu
```


---

## 7️⃣ PROGRAMMES UTILITAIRES

### ✅ set_company_occurs.py

**Rôle** : Calcule `nbocc` (nombre d'occurrences) pour chaque entreprise dans chaque article

**Importance** : 🔴 **CRITIQUE** - Ce programme doit tourner AVANT `evaluate_article.py`

#### Flux :

```python
1. Charger alias.json (alias par entreprise)
2. Pour chaque entreprise :
   a. Compiler regex avec \b (word boundaries)
   b. SELECT tous les articles
   c. Compter occurrences du nom + tous ses alias
   d. UPDATE article_companies.nbocc
   e. UPDATE article_companies.date_estim = TODAY
```

#### Configuration :

```python
QUERY_ARTICLES = SELECT articles pour une entreprise
nbocc = nombre total (nom + aliases)
note_tfidf = 0 (pas touché)
note_full = 0 (pas touché)
```

#### Usage :

```bash
# Une entreprise (id=15)
python3 set_company_occurs.py 15

# Toutes les entreprises
python3 set_company_occurs.py
```

#### Points clés :

- 🔴 **DOIT tourner avant evaluate_article.py**
- Filtre articles avec nbocc > 1 au moment de l'évaluation

---

### ✅ reload_article_companies_notes.py

**Rôle** : Récharge les notes LLM en BDD à partir de fichiers CSV exportés précédemment

**Use-case** : 
- Vous aviez sauvegardé un CSV avec des notes, mais elles ne sont plus en BDD
- Vous voulez ré-importer des résultats depuis des runs antérieurs
- Récupérer des résultats qui ont été accidentellement supprimés

#### Flux :

```python
1. Scanner fichiers CSV (avec glob patterns)
2. Détecter le modèle depuis le nom de fichier
   → "resultats_mistral_XXX.csv" → "mistral"
   → "resultats_haiku_XXX.csv" → "haiku"
3. Pour chaque ligne du CSV :
   a. Normaliser nom entreprise + article_id
   b. Chercher en BDD via (article_id, company_id)
   c. Importer note_llm et justification
   d. UPDATE article_companies avec ces colonnes
4. Commit en fin de processus
```

#### Colonnes mappées :

```python
MODEL_TO_COLUMNS = {
    "gemini": ("note_gemini", "justification_gemini"),
    "haiku": ("note_haiku", "justification_haiku"),
    "mistral": ("note_mistral", "justification_mistral"),
    "queen": ("note_queen", "justification_queen"),
    "lama": ("note_llama3", "justification_lama"),
}
```

#### Usage :

```bash
# Tous les CSV de sortie/
python3 reload_article_companies_notes.py

# Un CSV spécifique
python3 reload_article_companies_notes.py output/resultats_mistral_20260803_193015.csv

# Plusieurs CSV (glob)
python3 reload_article_companies_notes.py output/resultats_mistral_*.csv output/resultats_queen_*.csv

# Mode dry-run (vérification sans écriture)
python3 reload_article_companies_notes.py --dry-run
```

#### Points clés :

- ✅ Récupère notes depuis CSV (pas depuis API)
- ✅ Supporte glob patterns pour plusieurs fichiers
- ✅ Normalise les noms (majuscules, accents, espaces)
- ✅ Mode --dry-run pour vérifier avant de committer
- ⚠️ Nécessite que article_id + company_id existent en BDD

---

### ✅ benchmark_classification.py

**Rôle** : Compare la qualité des classifications entre différents modèles LLM

**Objectif** : Mesurer qui est le meilleur (Haiku? Gemini? Ollama local?)

#### Configuration de référence :

```python
BASELINE = "note_haiku"               # modèle de référence principal
REFERENCE_SECONDAIRE = "note_gemini"  # 2e référence pour isoler les divergences

MODELES_LOCAUX = ["note_llama3", "note_mistral", "note_queen"]
```

#### Métriques calculées :

```python
1. Accuracy (% de prédictions correctes vs baseline)
2. Cohen's Kappa (accord inter-annotateurs)
3. Confusion Matrix (quels cas sont mal classés?)
4. Classification Report (precision, recall, F1-score)
5. Heatmaps (visualisation des erreurs)
```

#### Flux :

```python
1. Charger DataFrame depuis BDD (article_companies)
2. Filtrer par version de prompt (v1 a v4) depuis nom de fichier
3. Exclure lignes avec statut != "ok" (failed ne compte pas)
4. Comparer chaque modèle vs BASELINE
   ├─ Haiku vs Gemini
   ├─ Ollama local vs Haiku
   └─ Ollama local vs Gemini
5. Générer rapports + graphiques
```

#### Output :

```
📊 Graphiques (PNG/SVG)
├─ Confusion matrices (heatmaps)
├─ Accuracy comparatives
└─ F1-score par classe

📋 Rapports texte
├─ Classification report (precision, recall)
├─ Kappa scores (accord inter-raters)
└─ Zones de divergence (où les modèles ne sont pas d'accord)
```

#### Usage :

```bash
python3 benchmark_classification.py
```

#### Points clés :

- ✅ Compare LOCAL vs CLOUD automatiquement
- ✅ Isole version de prompt (v1 a v4)
- ✅ Génère graphiques professionnels (matplotlib/seaborn)
- ✅ Utilise statut pour filtrer les vraies notes (pas les "failed")
- ⚠️ Nécessite colonnes de statut (statut_lama, statut_haiku, etc.)

---

### ✅ build_alias_db.py

**Rôle** : Construit `alias.json` à partir des articles en BDD using Gemini LLM

**Objectif** : Détecter TOUS les aliases, périphrases, surnoms que les journalistes utilisent pour parler de chaque entreprise

#### Exemple :

```
Entreprise : "Air Liquide"
Aliases trouvés : [
  "AL",
  "le leader du gaz",
  "le géant français",
  "le spécialiste des gaz industriels",
  "l'équipementier historique",
  ...
]
```

#### Flux :

```python
1. Pour chaque entreprise :
   a. SELECT 10 articles mentionnant cette entreprise
   b. Préparer prompt Gemini avec ces articles
   c. Appeler Gemini en JSON mode
   d. Parser réponse → liste d'aliases
2. Construire alias.json :
   {
     "Air Liquide": ["AL", "le leader du gaz", ...],
     "Renault": ["Renault Group", "la marque au losange", ...],
     ...
   }
3. Sauvegarder dans output/alias.json
```

#### Configuration :

```python
GOOGLE_API_KEY = charger depuis .env
GEMINI_MODEL = "gemini-flash-latest"

class ListeAlias(BaseModel):
    alias: list[str]
```

#### Usage :

```bash
python3 build_alias_db.py
```

#### Points clés :

- 💰 Payant (utilise API Gemini)
- ✅ Automatise la détection d'aliases
- ✅ Supprime le besoin de saisie REMOVEDelle
- ⚠️ Qualité dépend du sample d'articles fournis (10 articles)
- ✅ Schema Pydantic garantit format JSON correct

---

### ✅ recuperer_batch_orphelin.py

**Rôle** : Récupère les résultats de batches "orphelins" (soumis mais métadonnées perdues)

**Contexte du problème** :
- Vous lancez `--mode batch` pour Gemini
- Pendant la soumission, une erreur 429 survient au milieu (ex: chunk 3/5)
- Les chunks 1-2 étaient déjà soumis ET facturés
- L'ancien code sortait AVANT d'écrire `batches_gemini_XXX.json`
- Résultat : des batch_id tournent côté Google, mais aucun fichier local ne les relie aux articles
- `--collect` devient impossible

#### Solution :

```python
Flux de récupération :
1. Lire batch_id depuis CLI (ex: "batch_123abc")
2. Appeler Google API : client.batches.get(batch_id)
3. Vérifier état = JOB_STATE_SUCCEEDED (sinon ignore et liste pour reprise)
4. Lire fichier de résultats depuis GCS (job.dest.file_name)
5. Extraire chaque custom_id depuis la colonne "key"
6. Parser custom_id → (article_id, company_id)
7. SELECT correspondant en BDD pour récupérer texte + métadonnées
8. Reconstruire fichier métadonnées compatible :
   {
     "modele": "gemini",
     "prompt_version": "v3",
     "batches": [...],
     "requetes": {...},  # article_id → texte
   }
9. Écrire output/batches_gemini_v3_YYYYMMDD_RECUPERE.json
10. Proposer de lancer : python evaluate_article.py --collect <fichier>
```

#### Usage :

```bash
# Récupérer 1 batch
python3 recuperer_batch_orphelin.py v3 "batch_XXXXXX"

# Récupérer plusieurs batchs
python3 recuperer_batch_orphelin.py v3 "batch_AAA" "batch_BBB" "batch_CCC"

# Mode debug (inspecter structure du job)
python3 recuperer_batch_orphelin.py v3 "batch_XXXXXX" --debug
```

#### Filtrage des batchs :

```python
ETATS_TERMINAUX = {
    "JOB_STATE_SUCCEEDED",    # ✅ récupérable
    "JOB_STATE_FAILED",       # ❌ pas de résultats
    "JOB_STATE_CANCELLED",    # ❌ pas de résultats
    "JOB_STATE_EXPIRED",      # ❌ pas de résultats
}

Les batchs NON terminaux sont listés pour reprise ultérieure
```

#### Points clés :

- ✅ Récupère batches perdus
- ✅ Supporte multiples batch_id en une commande
- ✅ Mode debug pour diagnostiquer
- ⚠️ Nécessite que batch soit en JOB_STATE_SUCCEEDED
- ⚠️ Les articles doivent toujours exister en BDD
- ✅ Output compatible avec `evaluate_article.py --collect`

---

### ✅ verif__limit_batch.py

**Rôle** : Diagnostic des jobs batch Gemini qui occupent la file d'attente

**Problème résolu** :

Vous recevez l'erreur :
```
429 RESOURCE_EXHAUSTED - "Token quota exceeded: 3M tokens queued"
```

Mais vous ne lancez rien en ce moment... pourquoi?

**Réponse** : Des jobs batch antérieurs occupent ENCORE la file, même s'ils sont terminés!

Gemini Batch API compte les tokens "mis en file d'attente" jusqu'à ce que le job atteigne un **état terminal**.

#### États et quotas :

```python
ETATS_TERMINAUX = {
    "JOB_STATE_SUCCEEDED",    # ✅ n'occupe PLUS la file
    "JOB_STATE_FAILED",       # ✅ n'occupe PLUS la file
    "JOB_STATE_CANCELLED",    # ✅ n'occupe PLUS la file
    "JOB_STATE_EXPIRED",      # ✅ n'occupe PLUS la file
}

États NON terminaux (occupent la file) :
    "JOB_STATE_UNSPECIFIED"
    "JOB_STATE_QUEUED"
    "JOB_STATE_RUNNING"
    "JOB_STATE_PAUSED"
```

#### Flux :

```python
1. Charger GOOGLE_API_KEY depuis .env
2. Appeler client.batches.list()
3. Pour chaque batch :
   a. Vérifier job.state
   b. Afficher :
      "->" si NON terminal (⚠️ occupe la file)
      "  " si terminal (✅ libre)
4. Compter combien de jobs NON-terminaux
5. Si --annuler passé en argument :
   a. Pour chaque job NON-terminal :
      - Appeler client.batches.cancel(job.name)
      - Afficher "Annulé : {job.name}"
   b. Afficher "Attendez 1 minute pour la propagation"
```

#### Usage :

```bash
# Lister seulement
python3 verif__limit_batch.py

# Annuler les jobs non-terminaux
python3 verif__limit_batch.py --annuler
```

#### Output exemple :

```
Clé API chargée : GOOG...XXXX (47 caractères)

2 job(s) trouvé(s) :

  JOB_STATE_SUCCEEDED  batch_ABC123  2026-08-14T10:00:00Z
-> JOB_STATE_RUNNING   batch_DEF456  2026-08-13T15:00:00Z

1 job(s) NON terminal(aux) occupent encore votre quota de file d'attente.
Relancez avec --annuler pour les annuler et libérer la file :
    python verif__limit_batch.py --annuler
```

#### Points clés :

- ✅ Diagnostic rapide des quotas
- ✅ Lire sans toucher (mode par défaut)
- ✅ Annuler les jobs qui traînent
- ✅ Affiche créateur + timestamp de chaque job
- ⚠️ Attention : pas de confirmation avant annulation
- ✅ Masque la clé API pour la sécurité

---

### ✅ verif_batch_haiku.py

**Rôle** : Vérifier le statut des batchs Haiku (Anthropic)

**Différence avec Gemini** :

Contrairement à Gemini qui expose une liste centralisée des batchs via `client.batches.list()`,
Anthropic n'offre pas de listing global. On récupère donc les batch_id depuis les fichiers 
JSON de métadonnées sauvegardés localement (`output/batches_haiku_*.json`).

#### Statuts Haiku :

```python
in_progress  → batch tourne encore côté Anthropic
ended        → batch terminé, résultats disponibles
unknown      → erreur de communication avec l'API
```

#### Flux :

```python
1. Scanner fichiers batches_haiku_*.json (ou patterns spécifiques)
2. Extraire batch_id de chaque fichier
3. Appeler API Anthropic pour chaque batch : GET /messages/batches/{batch_id}
4. Afficher statut (in_progress / ended)
5. Si ended → afficher request_counts (succès / erreurs)
6. Proposer --collect si résultats disponibles
```

#### Usage :

```bash
# Tous les batchs Haiku locaux (par défaut)
python3 verif_batch_haiku.py

# Fichiers spécifiques (glob patterns)
python3 verif_batch_haiku.py output/batches_haiku_v3_*.json

# Plusieurs patterns
python3 verif_batch_haiku.py output/batches_haiku_v3_*.json output/batches_haiku_v2_*.json

# Batch_id direct
python3 verif_batch_haiku.py msgbatch_XXXXXXXXXXXXXXXXXXXXXXXXX
```

#### Output exemple :

```
Clé API chargée : sk-ant-...XXXX (48 caractères)

2 batch(s) trouvé(s) :

→ in_progress    msgbatch_ABC123...
   Créé : 2026-08-14T14:30:22Z
   Source : output/batches_haiku_v3_20260814_143022.json
   Requêtes : 500 succès, 0 erreurs

  ended          msgbatch_DEF456...
   Créé : 2026-08-14T12:00:00Z
   Source : output/batches_haiku_v3_20260814_120000.json
   Requêtes : 1000 succès, 5 erreurs
   Results URL : https://...

============================================================
1 batch(s) ENCORE EN COURS (in_progress) :

  • msgbatch_ABC123... (source: output/batches_haiku_v3_20260814_143022.json)
    Progression: 500 succès | 0 en cours | 0 erreurs

Attendez quelques heures ou relancez ce script plus tard.

============================================================
1 batch(s) TERMINÉ(S) :

  • msgbatch_DEF456... (ended)
    Résultats: 1000 succès | 5 erreurs
    Results URL : https://...

Pour récupérer les résultats, lancez :
  python evaluate_article.py --collect output/batches_haiku_v3_20260814_120000.json
```

#### Workflow complet Haiku batch :

```bash
# 1. Soumettre batch
python3 evaluate_article.py --modele haiku --prompt-version v3 --mode batch
# Output : output/batches_haiku_v3_20260814_143022.json

# 2. Vérifier statut (répétez toutes les heures)
python3 verif_batch_haiku.py
# Output : "in_progress" → attendez

# 3. Quand "ended" s'affiche → récupérer résultats
python3 evaluate_article.py --collect output/batches_haiku_v3_20260814_143022.json
```

#### Points clés :

- ✅ Lit batch_id depuis fichiers JSON locaux
- ✅ Pas besoin de listing centralisé (comme Gemini)
- ✅ Support glob patterns
- ✅ Support batch_id direct
- ✅ Affiche progression (requêtes succès/erreurs)
- ⚠️ Nécessite fichiers JSON de métadonnées (output/batches_haiku_*.json)
- ✅ Propose automatiquement --collect quand résultats disponibles
- ✅ Haiku batchs : généralement 2-24h (vs Gemini : jusqu'à 24h)

---

## 📊 TABLEAU RÉCAPITULATIF UTILITAIRES

| Programme | Rôle | Critique? | Pour |
|-----------|------|-----------|------|
| set_company_occurs.py | Calcul nbocc (word boundaries) | 🔴 OUI | Avant evaluate_article |
| reload_article_companies_notes.py | Ré-import depuis CSV | 🟡 Optionnel | Récupérer anciennes notes |
| benchmark_classification.py | Comparer qualité modèles | 🟡 Optionnel | Analyser résultats |
| build_alias_db.py | Générer alias.json auto (Gemini) | 🟡 Optionnel | Détection aliases |
| recuperer_batch_orphelin.py | Récupérer batchs perdus | 🟡 Urgence si besoin | Récupérer batchs Gemini orphelins |
| verif__limit_batch.py | Diagnostic quota Gemini | 🟡 Urgence si 429 | Vérifier jobs Gemini |
| verif_batch_haiku.py | Vérifier statut batchs Haiku | 🟡 Usage régulier | Vérifier jobs Haiku |

---



### Lancement


Recharger tous les resultats trouves dans `output/`:

```bash
uv run reload_article_companies_notes.py
```

Ce script recharge les notes dans `public.article_companies` a partir des fichiers deja exportes:
- `output/resultats_gemini_*.csv`
- `output/resultats_haiku_*.csv`
- `output/resultats_mistral_*.csv`
- `output/resultats_queen_*.csv`
- `output/resultats_lama_*.csv` 

Il evite de relancer les appels LLM: il relit les CSV puis met a jour la base directement.

Recharger seulement certains lots:

```bash
uv run reload_article_companies_notes.py output/resultats_mistral_*.csv output/resultats_queen_*.csv output/resultats_gemini_*.csv
```

exemple :


```bash
uv run reload_article_companies_notes.py  --dry-run  output/resultats_gemini_20260709_204325.csv
```

Simulation sans ecriture en base:

```bash
uv run reload_article_companies_notes.py --dry-run
```



#### Prérequis

Fichier `.env` à la racine :

```
DB_HOST=localhost
DB_PORT=5432
DB_USER=REMOVED
DB_PASSWORD=...
DB_NAME=REMOVED

GOOGLE_API_KEY=...        # pour gemini
ANTHROPIC_API_KEY=...     # pour haiku
OLLAMA_URL=http://127.0.0.1:11434/api/generate   # pour les modeles locaux
```

Migration SQL à passer **une seule fois** (détaillée en tête de `evaluate_article.py`) :
colonnes `statut_*` et `prompt_version_*`, suppression des contraintes `NOT NULL`
et des `DEFAULT 0` sur les colonnes `note_*`.

> Le `DEFAULT 0` est piégeux : sans lui, une ligne jamais évaluée naît avec la note 0,
> c'est-à-dire NEGATIVE, indiscernable d'un vrai jugement négatif.

### Lancement

Le script est interactif : il demande le modèle, la version de prompt, puis le mode.

```bash
python evaluate_article.py
```

Tout peut être piloté par variables d'environnement pour scripter les runs :

```bash
PROMPT_VERSION=v4 python evaluate_article.py
BATCH_TAILLE_MAX=500 PROMPT_VERSION=v4 python evaluate_article.py
```

### Variables d'environnement

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `PROMPT_VERSION` | *(demandé)* | Version de prompt : `v1` à `v4` |
| `REPRENDRE` | `1` | `0` pour tout réévaluer, y compris ce qui est déjà fait |
| `LIMITE_PAR_ENTREPRISE` | *(aucune)* | Limite le nombre d'articles par entreprise (tests) |
| `BATCH_TAILLE_MAX` | `1000` | Nombre de requêtes par batch soumis |
| `RETRY_MAX_TENTATIVES` | `4` | Tentatives avant abandon sur erreur transitoire |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Modèle Gemini |
| `HAIKU_MODEL` | `claude-haiku-4-5` | Modèle Anthropic |

### Versions de prompt

| Version | Contenu |
| --- | --- |
| `v1` | Règles courtes, NEUTRAL par défaut en cas de doute |
| `v2` | Isolation des phrases pertinentes, few-shot, règle anti-repli-neutre |
| `v3` | v2 + règles sur les objectifs de cours et recommandations de brokers |
| `v4` | v3 corrigé : étape de lecture factuelle explicite + grille complète 3×4 |

**Règle absolue : ne jamais modifier un prompt existant en place.** Toute évolution
crée une nouvelle version. Le mécanisme de reprise compare la version stockée en base
à celle demandée ; modifier `v4` sans le renommer ferait croire au script que tout est
à jour alors que les notes proviennent d'un prompt différent.

### Modes d'exécution

**`sync`** — appel immédiat, article par article. Seul mode disponible pour les modèles
locaux. Les résultats sont écrits au fil de l'eau.

**`batch`** — réservé à Gemini et Haiku. Soumission par lot à **-50 % de coût**, avec un
délai pouvant aller jusqu'à 24 h (en pratique 2 à 4 h). Se déroule en deux temps :

```bash
# 1. Soumission (retour immédiat)
PROMPT_VERSION=v4 python evaluate_article.py
#    -> Métadonnées : output/batches_gemini_v4_20260814_084228.json

# 2. Récupération (à relancer tant que des batchs sont en cours)
python evaluate_article.py --collect output/batches_gemini_v4_20260814_084228.json
```

Le batch continue de tourner côté fournisseur même machine éteinte. La récupération
est idempotente : les batchs déjà collectés sont ignorés.

### Reprise automatique des runs interrompus

Activée par défaut (`REPRENDRE=1`). Le script ne traite que les couples
(article, entreprise) qui ne sont **pas** déjà en `statut = 'ok'` ou `'submitted'`
**pour la version de prompt demandée**.

Concrètement, relancer après une coupure ne retraite que le reliquat. Changer de
version de prompt réévalue tout, puisque les notes ne sont plus comparables.

Cycle de vie du statut :

```
not_evaluated  ──►  submitted  ──►  ok
                                └►  failed  ──► (repris au run suivant)
```

`submitted` est écrit dès la soumission d'un batch : il empêche qu'un second run
lancé pendant que le batch est en vol resoumette — et refacture — les mêmes articles.

Pour forcer une réévaluation complète (ex. après correction d'un bug de parsing) :

```bash
REPRENDRE=0 PROMPT_VERSION=v4 python evaluate_article.py
```

### Sorties produites

| Fichier | Contenu |
| --- | --- |
| `output/resultats_{modele}_{version}_{date}.csv` | Une ligne par évaluation : note, justification, statut, longueurs |
| `output/log_articles_{modele}_{version}_{date}.txt` | Journal détaillé, y compris les articles filtrés |
| `output/batches_{modele}_{version}_{date}.json` | Métadonnées de batch (mode batch uniquement) |

La base ne conserve que le **dernier** état de chaque colonne. Pour comparer deux
versions de prompt, il faut donc s'appuyer sur les CSV horodatés, pas sur la base.

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
       OR ac.prompt_version_gemini IS DISTINCT FROM 'v4');
```




### reinit d'ollama

sudo systemctl stop ollama
sudo pkill -f ollama
sudo systemctl start ollama


