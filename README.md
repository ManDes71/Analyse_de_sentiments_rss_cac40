# Analyse de sentiments appliquée à des flux rss boursiers

**`Analyse_de_sentiments_rss_cac40`** est un pipeline d'IA bout en bout conçu pour automatiser la veille financière et la notation de sentiment d'entreprises à partir de flux de presse (RSS).

# Construire sa Vérité Terrain : Quel LLM Local pour Noter l'Actualité Boursière ?

> **Série Market-RSS-Sentiment — Épisode 2**
> Dans [l'épisode 1](https://aventuresdata.com/avant-lanalyse-de-sentiment-boursier-letape-cruciale-de-lextraction-dalias-avec-un-llm/), nous avons réglé le problème des coréférences journalistiques en générant un dictionnaire d'alias par LLM. Aujourd'hui, nous attaquons une étape charnière de tout projet NLP : la constitution d'une vérité terrain (*ground truth*). L'objectif : noter le sentiment boursier de chaque article, et trouver une alternative locale, gratuite, à une API propriétaire.

---

## 1. Le besoin d'une vérité terrain

Pour le troisième épisode de cette série, nous allons évaluer un Transformer spécialisé en finance (`bardsai/finance-sentiment-fr-base`) et éventuellement le ré-entraîner. Impossible de faire ça sans un jeu d'articles annotés de confiance.

Notre grille de notation reste volontairement simple, un entier par couple (article, entreprise) :

* **0 (Négative)** : fait concret défavorable — baisse de résultats, amende, dégradation par un broker.
* **1 (Neutre)** : mention factuelle, mouvement de marché global, ou signaux contradictoires qui s'annulent.
* **2 (Positive)** : fait concret favorable — hausse de résultats, contrat gagné, relèvement d'objectif de cours.

Pour produire ces notes, **Claude Haiku** (Anthropic) est notre référence : bon rapport qualité/prix, suit les instructions à la lettre, sort du JSON propre. Mais interroger une API distante pour chaque article, à chaque itération de prompt, à chaque relance de pipeline, ça a un coût — en argent et en dépendance réseau. La question MLOps est directe : **un modèle open-weights, hébergé en local sur une carte grand public, peut-il produire des notes suffisamment proches de Haiku pour servir de vérité terrain ?**

C'est ce que mesure le script `evaluate_article.py`.

---

## 2. Faire tourner Ollama sur une RTX 3060 12 Go

Avant de comparer des modèles, il faut pouvoir les faire tourner. La RTX 3060 12 Go est une carte grand public très répandue, avec une contrainte claire : 12 Go de VRAM, pas un octet de plus. Voici comment j'ai configuré le poste de benchmark.

### Installation

```bash
# Linux / WSL
curl -fsSL https://ollama.com/install.sh | sh

# Windows : installeur .exe sur ollama.com/download
# Le service ollama serve démarre automatiquement et écoute sur le port 11434
```

Ollama détecte le driver NVIDIA et bascule l'inférence sur GPU sans configuration supplémentaire. Pour vérifier que le GPU est bien utilisé pendant une requête :

```bash
nvidia-smi          # doit afficher le process "ollama" et la VRAM occupée
ollama ps           # colonne PROCESSOR : "100% GPU" = tout tient en VRAM
```

Si `ollama ps` affiche un pourcentage CPU, une partie du modèle déborde en RAM système : l'inférence sera nettement plus lente. C'est le premier signal à surveiller sur une carte à VRAM limitée.

### Choisir les bons modèles pour 12 Go

Chaque modèle Ollama est distribué en quantification 4 bits (`Q4_K_M`) par défaut, ce qui divise environ par 4 l'empreinte mémoire d'un modèle FP16. Voici l'empreinte VRAM des trois candidats retenus pour ce benchmark :

| Modèle | Taille (poids Q4) | Marge restante sur 12 Go* |
| --- | --- | --- |
| **Llama 3.1 (8B)** — `llama3.1:8b` | ~4,9 Go | ~7 Go |
| **Qwen 2.5 (7B)** — `qwen2.5:7b` | ~4,7 Go | ~7,3 Go |
| **Mistral NeMo (12B)** — `mistral-nemo` | ~7,1 Go | ~4,9 Go |

*\*Marge indicative avant prise en compte du contexte (KV-cache) et des autres processus GPU (affichage, navigateur…).*

```bash
ollama pull llama3.1:8b
ollama pull mistral-nemo
ollama pull qwen2.5:7b
```

Deux réglages comptent autant que la taille du modèle sur une carte contrainte :

* **`num_ctx`** (taille du contexte) : plus la fenêtre est grande, plus le KV-cache consomme de VRAM en plus des poids. `evaluate_article.py` utilise `num_ctx=8192`, suffisant pour nos dépêches financières (rarement plus de 2000 mots) sans faire déborder Mistral NeMo hors du GPU.
* **`OLLAMA_KEEP_ALIVE`** : par défaut, Ollama garde un modèle chargé 5 minutes après le dernier appel. Comme le benchmark teste un modèle à la fois (jamais deux en parallèle — impossible de faire tenir Llama et Mistral simultanément dans 12 Go), il vaut mieux le laisser à sa valeur par défaut plutôt que le forcer à décharger entre chaque article, ce qui ajouterait une latence de rechargement à chaque requête.

Appel HTTP local utilisé dans le script (extrait) :

```python
payload = {
    "model": MODELES_OLLAMA[modele],   # ex: "llama3.1:8b"
    "prompt": prompt,
    "stream": False,
    "format": "json",
    "options": {
        "num_ctx": 8192,
        "num_predict": 3000,
        "temperature": 0.1
    }
}
response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
```

`format: "json"` force Ollama à contraindre le sampling pour produire du JSON syntaxiquement valide — indispensable, mais on verra plus bas que ça ne garantit pas des valeurs *sémantiquement* valides.

---

## 3. Le protocole : 3 candidats, 3 versions de prompt

Trois modèles open-weights ont été confrontés à Claude Haiku sur le même corpus d'articles :

* **Llama 3.1 (8B)** — le standard Meta.
* **Mistral NeMo (12B)** — l'architecture franco-américaine réputée pour son français.
* **Qwen 2.5 (7B)** — la référence d'Alibaba sur les tâches de raisonnement.

Le premier run (prompt **v1**, des règles courtes et sans exemple) a fait ressortir un défaut classique des modèles de taille modeste : le **repli vers le Neutre**. Face à une phrase financière ambiguë, un petit modèle a tendance à jouer la sécurité et à sortir la note 1, ce qui écrase le signal utile.

Le prompt a donc évolué en deux temps :

* **v2** — approche *few-shot* : le modèle doit d'abord isoler les phrases qui concernent spécifiquement l'entreprise, puis appliquer une définition restrictive du Neutre (« pas une valeur par défaut »), ancrée par 3 exemples (positif / neutre / négatif).
* **v3** — v2 + un bloc de règles dédié aux notes de brokers/analystes (relèvement/abaissement d'objectif de cours, changement de recommandation), ajouté après avoir observé que ces cas concentraient la majorité des désaccords. La règle retenue : c'est le *niveau* de la recommandation qui prime sur le seul sens de l'objectif de cours — un objectif abaissé reste neutre si le broker maintient « acheter », mais devient négatif s'il n'est qu'à « conserver ».

```python
# Extrait du prompt v3 (règles brokers)
"""
- POSITIVE (2) : relèvement de la recommandation, OU rehaussement de l'objectif de
  cours avec maintien de la recommandation à "acheter" ou "surpondérer".
- NEUTRAL (1) : objectif de cours abaissé MAIS recommandation maintenue à l'achat.
- NEGATIVE (0) : abaissement de l'objectif de cours avec maintien de la
  recommandation à "conserver" ou "surperformance".
"""
```

Chaque run est tracé dans un CSV horodaté (`resultats_{modele}_{version}_{date}.csv`) avec le statut par article (`ok` / `failed`) et la version de prompt utilisée, pour ne jamais comparer par erreur une note v1 à une note v3.

---

## 4. Les résultats, sans enjolivement

Les runs v1 et v2 portent exactement sur le même corpus (**1 421 couples article/entreprise**). Le run v3 a été relancé un jour plus tard : comme la requête SQL ne conserve que les 100 derniers articles par entreprise, le corpus a naturellement dérivé (**954 couples**, dont 951 en commun avec v1/v2). Les chiffres v3 restent donc indicatifs plutôt que strictement comparables aux deux premiers — je le précise pour ne pas laisser croire à un A/B test parfaitement contrôlé.

### A. Accord exact avec Claude Haiku

| Modèle | v1 | v2 | v3 |
| --- | --- | --- | --- |
| **Llama 3.1 (8B)** | **74,10 %** | **74,92 %** | 65,72 % |
| Qwen 2.5 (7B) | 71,42 % | 74,08 % | 70,23 % |
| Mistral NeMo (12B) | 57,73 % | 61,27 % | 59,54 % |

*Étalon utile pour lire ce tableau : sur ce même corpus, Claude Haiku et Gemini 2.5 Flash — deux API distantes de premier plan — ne sont d'accord entre elles qu'à 82,15 %. Un modèle de 8B qui atteint 75 % d'accord avec Haiku n'est donc pas loin du plafond « désaccord naturel entre deux experts ».*

### B. Robustesse du format JSON

| Modèle | v1 : valeurs hors {0,1,2} | v2 : valeurs hors {0,1,2} | v3 |
| --- | --- | --- | --- |
| **Llama 3.1 (8B)** | 0 | 0 | 0 |
| Mistral NeMo (12B) | 0 | 0 | 0 |
| Qwen 2.5 (7B) | 2 (`-1`) | **24** (`-9, -5, -1, 3, 4`) | 0 (mais 12 échecs `failed`) |

`format: "json"` garantit une syntaxe JSON valide, mais Qwen a produit à plusieurs reprises un champ `note_llm` syntaxiquement correct (c'est un entier) tout en étant hors du domaine métier {0,1,2} — une valeur comme `-5` ou `4` casse silencieusement toute analyse en aval si elle n'est pas validée explicitement. C'est exactement le rôle du `valider_evaluation()` de `evaluate_article.py`, qui rejette ces réponses et les repasse en `failed` plutôt que de les insérer telles quelles.

### C. La bataille du Neutre — distribution des notes (v2)

| Modèle | 0 (Négatif) | 1 (Neutre) | 2 (Positif) |
| --- | --- | --- | --- |
| **Haiku (référence)** | 21,9 % | 30,7 % | 47,5 % |
| **Llama 3.1 (8B)** | 15,7 % | 30,5 % | 53,8 % |
| Qwen 2.5 (7B) | 13,1 % | 40,8 % | 44,2 % |
| Mistral NeMo (12B) | 5,7 % | 64,1 % | 30,2 % |

Le signal est net : plus un modèle est petit ou prudent, plus il gonfle la classe Neutre. Mistral NeMo classe **64,1 % des articles en Neutre**, contre 30,7 % pour Haiku — malgré une taille de modèle supérieure (12B) et une consommation VRAM plus élevée. Le prompt v2/v3, pourtant conçu spécifiquement pour contrer ce biais, ne suffit pas à corriger l'inertie de ce modèle.

---

## 5. Verdict : Llama 3.1 (8B), meilleur compromis local

Trois constats se dégagent des chiffres, pas d'une impression :

1. **Mistral NeMo (12B) est trop prudent.** Malgré une empreinte VRAM plus élevée (~7 Go) et un français de bonne qualité, il lisse le signal en sur-notant le Neutre. Résultat : le pire taux d'accord des trois candidats (58–61 %), sur les trois versions de prompt.
2. **Qwen 2.5 (7B) est précis mais fragile en production.** Deuxième meilleur taux d'accord, mais des sorties hors-plage récurrentes (jusqu'à 24 valeurs aberrantes sur un run) qui obligent à des rejets et des relances — un vrai coût opérationnel sur un pipeline automatisé.
3. **Llama 3.1 (8B) l'emporte** sur les deux versants : le meilleur accord avec Haiku (74–75 % sur le corpus v1/v2), zéro erreur de format sur les trois runs, et l'empreinte VRAM la plus légère (~4,9 Go) des trois — de la marge pour augmenter `num_ctx` si besoin.

Une réserve honnête s'impose : 75 % d'accord exact n'est **pas** une équivalence à Haiku. C'est un bon proxy local, utilisable pour construire un volume de vérité terrain à moindre coût, mais l'écart de 25 % restant — concentré sur les cas ambigus (recommandations de brokers, signaux contradictoires) — justifie de garder Haiku (ou un contrôle humain) sur un échantillon d'audit plutôt que de lui faire une confiance aveugle sur l'intégralité du corpus.

---

## 6. Industrialisation : traçabilité et reprise sur erreur

`evaluate_article.py` n'est pas qu'un script de benchmark, il est pensé pour tourner en production :

* **Backoff exponentiel** sur les erreurs réseau ou 429/5xx, aussi bien pour les API distantes que pour Ollama.
* **Statut explicite par article** (`ok` / `failed` / `not_evaluated`) : un `failed` laisse la note à `NULL` en base plutôt que de la confondre avec un vrai 0, et permet de reprendre uniquement les articles en échec sans tout ré-évaluer.
* **Traçabilité de la version de prompt** (`prompt_version_*`) directement dans PostgreSQL, pour ne jamais comparer par erreur deux runs générés avec des règles différentes.

```python
update_query = f'''
    UPDATE public.article_companies
    SET {colonne_note} = %s, {colonne_justif} = %s, {colonne_statut} = %s,
        {colonne_prompt_version} = %s
    WHERE article_id = %s AND company_id = %s
'''
cur.execute(update_query, (note, justif, statut, version_prompt, article_id, company_id))
```

---

## Conclusion et prochaine étape

Llama 3.1 (8B) via Ollama devient notre générateur local de vérité terrain : gratuit, tournant confortablement sur une RTX 3060 12 Go, et suffisamment aligné avec Claude Haiku pour servir de base d'entraînement et d'évaluation.

Reste une question qu'un LLM généraliste de 8 milliards de paramètres ne résout pas : la vitesse. Dans le **troisième et dernier épisode**, nous testerons si un Transformer ultra-spécialisé (`bardsai/finance-sentiment-fr-base`) peut approcher ces résultats à une fraction du coût de calcul — avec ou sans résumé préalable par TF-IDF, et en évaluant si un ré-entraînement (tête de classification figée sur le corps pré-entraîné) est nécessaire pour combler l'écart.

---

*Retrouvez le code source de cette implémentation sur le dépôt GitHub du projet.*
