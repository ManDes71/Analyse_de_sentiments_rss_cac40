import argparse
import os
import json
import time
import re
import csv
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

##{nom llm;nom modele;type d'API;nom fichier CSV;colonne tableau de résultats}
#{gemini;gemini-2.5-flash;API Distante;output/resultats_gemini_{date_jour}.csv; note_gemini}
#{haiku;claude-haiku-4-5;API Distante;output/resultats_haiku_{date_jour}.csv; note_haiku}
#{lama;llama3;API locale Ollama;output/resultats_lama_{date_jour}.csv; note_llama3}
#{mistral;mistral-nemo;API locale Ollama;output/resultats_mistral_{date_jour}.csv; note_mistral}
#{queen;qwen2.5:7b;API locale Ollama;output/resultats_queen_{date_jour}.csv; note_queen}
#
# ==========================================
# MIGRATION BDD REQUISE (à exécuter une seule fois avant d'utiliser ce script) :
#
#   ALTER TABLE public.article_companies ADD COLUMN statut_gemini  varchar(20) DEFAULT 'not_evaluated';
#   ALTER TABLE public.article_companies ADD COLUMN statut_haiku   varchar(20) DEFAULT 'not_evaluated';
#   ALTER TABLE public.article_companies ADD COLUMN statut_lama    varchar(20) DEFAULT 'not_evaluated';
#   ALTER TABLE public.article_companies ADD COLUMN statut_mistral varchar(20) DEFAULT 'not_evaluated';
#   ALTER TABLE public.article_companies ADD COLUMN statut_queen   varchar(20) DEFAULT 'not_evaluated';
#
#   ALTER TABLE public.article_companies ADD COLUMN prompt_version_gemini  varchar(10);
#   ALTER TABLE public.article_companies ADD COLUMN prompt_version_haiku   varchar(10);
#   ALTER TABLE public.article_companies ADD COLUMN prompt_version_lama    varchar(10);
#   ALTER TABLE public.article_companies ADD COLUMN prompt_version_mistral varchar(10);
#   ALTER TABLE public.article_companies ADD COLUMN prompt_version_queen   varchar(10);
#
#   -- v6 : faits bruts extraits par le LLM, pour recalculer une note sans
#   -- relancer d'appel API (cf. llm_common.note_finale et recalculer_notes.py).
#   ALTER TABLE public.article_companies ADD COLUMN extraction_gemini  jsonb;
#   ALTER TABLE public.article_companies ADD COLUMN extraction_haiku   jsonb;
#   ALTER TABLE public.article_companies ADD COLUMN extraction_lama    jsonb;
#   ALTER TABLE public.article_companies ADD COLUMN extraction_mistral jsonb;
#   ALTER TABLE public.article_companies ADD COLUMN extraction_queen   jsonb;
#
#   -- INDISPENSABLE : les colonnes note_* doivent accepter NULL.
#   -- Une contrainte NOT NULL oblige a ecrire 0 quand une evaluation echoue,
#   -- ce qui cree une fausse note NEGATIVE indiscernable d'une vraie et fausse
#   -- toute analyse en aval. NULL + statut='failed' est la representation
#   -- correcte : "tente, sans resultat".
#   ALTER TABLE public.article_companies ALTER COLUMN note_gemini  DROP NOT NULL;
#   ALTER TABLE public.article_companies ALTER COLUMN note_haiku   DROP NOT NULL;
#   ALTER TABLE public.article_companies ALTER COLUMN note_llama3  DROP NOT NULL;
#   ALTER TABLE public.article_companies ALTER COLUMN note_mistral DROP NOT NULL;
#   ALTER TABLE public.article_companies ALTER COLUMN note_queen   DROP NOT NULL;
#
#   -- Verifier aussi l'absence de DEFAULT 0 sur ces memes colonnes :
#   --   SELECT column_name, is_nullable, column_default
#   --   FROM information_schema.columns
#   --   WHERE table_name='article_companies' AND column_name LIKE 'note_%';
#   -- puis, le cas echeant : ALTER COLUMN note_xxx DROP DEFAULT;
#
# Valeurs possibles pour statut_* :
#   'not_evaluated' : jamais soumis (ex. filtré par nb_occ <= 1)
#   'submitted'     : soumis en batch, résultat pas encore collecté. État
#                     transitoire : empêche un autre run de resoumettre (donc de
#                     refacturer) les mêmes articles pendant que le batch est en vol.
#   'ok'            : note/justification fiables
#   'failed'        : appel tenté, jamais abouti après retries -> note et
#                     justification remises à NULL, à exclure de toute analyse de
#                     qualité et à reprendre lors d'un futur run.
#
# prompt_version_* indique quelle version de PROMPTS (v1/v2/...) a produit la
# note actuellement stockée. Comme les colonnes note_*/statut_*/prompt_version_*
# sont écrasées à chaque run, comparer deux versions de prompt sur l'historique
# complet se fait via les fichiers CSV horodatés (output/resultats_{modele}_{version}_{date}.csv),
# pas via l'état courant de la BDD qui ne reflète que le dernier run.
# ==========================================


# Charger les variables d'environnement
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
DB_HOST    = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME     = os.getenv("DB_NAME", "finance_db")

# ==========================================
# PROVIDERS LLM
# ==========================================
# Chaque famille de modèles a désormais son propre module dans providers/,
# avec sa config colocalisée (clé API, nom de modèle, URL...) :
#   - providers/ollama.py -> modèles locaux gratuits (lama, mistral, queen)
#   - providers/gemini.py -> Google Gemini (API distante payante)
#   - providers/haiku.py  -> Claude Haiku (API distante payante)
# llm_common.py regroupe ce qui est partagé entre tous (prompts, schéma de
# validation, retry/backoff).
from llm_common import (
    CHAMPS_EXTRACTION,
    PROMPTS,
    BATCH_TAILLE_MAX,
    FatalLLMError,
    construire_custom_id,
    note_finale,
)
from providers.ollama import MODELES_OLLAMA, OllamaProvider
from providers.gemini import GeminiProvider, construire_client_gemini
from providers.haiku import HaikuProvider

# Modèles distants payants pour lesquels le mode batch (soumission par lot,
# -50% de coût, jusqu'à 24h de délai) est disponible en plus du mode sync.
MODELES_BATCH_DISPONIBLE = {
    "gemini": lambda: GeminiProvider(),
    "haiku": lambda: HaikuProvider(),
}

# Colonne BDD cible par modèle
COLONNE_NOTE = {
    "gemini":  "note_gemini",
    "haiku":   "note_haiku",
    "lama":    "note_llama3",
    "mistral": "note_mistral",
    "queen":   "note_queen",
}

COLONNE_JUSTIF = {
    "gemini":  "justification_gemini",
    "haiku":   "justification_haiku",
    "lama":    "justification_lama",
    "mistral": "justification_mistral",
    "queen":   "justification_queen",
}

# Colonne de statut d'évaluation par modèle : 'ok' | 'failed' | 'not_evaluated'
# 'not_evaluated' est la valeur par défaut en base (article jamais soumis au LLM,
# ex: filtré par nb_occ <= 1). 'failed' signifie que l'appel a été tenté mais n'a
# jamais abouti (après retries), donc note/justification restent NULL et ne
# doivent PAS être interprétées comme une note 0/NEGATIVE.
COLONNE_STATUT = {
    "gemini":  "statut_gemini",
    "haiku":   "statut_haiku",
    "lama":    "statut_lama",
    "mistral": "statut_mistral",
    "queen":   "statut_queen",
}

# Trace quelle version de prompt (v1/v2/...) a produit la note actuellement en
# base pour ce modèle. Nécessaire pour ne pas comparer par erreur une note v1 à
# une note v2 lors d'une analyse ultérieure.
COLONNE_PROMPT_VERSION = {
    "gemini":  "prompt_version_gemini",
    "haiku":   "prompt_version_haiku",
    "lama":    "prompt_version_lama",
    "mistral": "prompt_version_mistral",
    "queen":   "prompt_version_queen",
}

# Faits bruts extraits par le prompt v6, stockés en JSONB. Une seule colonne
# plutôt que quatre par modèle : le format peut évoluer sans migration, et la
# recherche reste possible (ex: extraction_gemini->>'variation_cours_pct').
#
# C'est ce qui permet de recalculer une note avec un autre seuil sans relancer
# le moindre appel API (cf. llm_common.note_finale et recalculer_notes.py).
COLONNE_EXTRACTION = {
    "gemini":  "extraction_gemini",
    "haiku":   "extraction_haiku",
    "lama":    "extraction_lama",
    "mistral": "extraction_mistral",
    "queen":   "extraction_queen",
}



# ==========================================
# GESTION DES ALIAS ET REGEX (FILTRAGE)
# ==========================================
def charger_dictionnaire_alias(chemin_fichier="output/alias.json"):
    if not os.path.exists(chemin_fichier):
        print(f"Attention : le fichier {chemin_fichier} est introuvable. On continue sans alias.")
        return {}
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        return json.load(f)

def compiler_regex_entreprise(nom_entreprise, liste_alias):
    cibles = [nom_entreprise.lower()] + [a.lower() for a in liste_alias]
    cibles_echappees = [re.escape(cible) for cible in set(cibles) if cible]
    motif = r'\b(' + '|'.join(cibles_echappees) + r')\b'
    return re.compile(motif, re.IGNORECASE)

def compter_occurrences(texte, regex_compilee):
    if not texte: return 0
    return len(regex_compilee.findall(texte))

# ==========================================
# ACCÈS AUX ARTICLES (partagé entre mode sync et mode batch)
# ==========================================
# longueur_caracteres et longueur_mots sont calculées directement en SQL pour éviter
# de recharger tout le texte en Python juste pour un comptage, et pour que la métrique
# soit disponible même si on veut l'utiliser comme critère de filtrage SQL plus tard.
# Limite du nombre d'articles remontés PAR ENTREPRISE.
#
# Historiquement figée à 100 ("limite pour tester"), elle est désormais désactivée
# par défaut. Raison : combinée au filtre de reprise, une limite fixe rend le
# corpus NON DÉTERMINISTE. Une fois les 100 articles les plus récents marqués
# 'ok', le run suivant remonte les 100 précédents — un ensemble différent. Deux
# runs successifs ne portent donc pas sur le même corpus, ce qui interdit toute
# comparaison entre versions de prompt.
#
# Mettre LIMITE_PAR_ENTREPRISE=100 pour retrouver l'ancien comportement (tests).
_LIMITE = os.getenv("LIMITE_PAR_ENTREPRISE", "").strip()
CLAUSE_LIMITE = f"LIMIT {int(_LIMITE)}" if _LIMITE.isdigit() else ""

QUERY_ARTICLES = f"""
    SELECT
        a.id,
        a.contenu,
        LENGTH(a.contenu) AS longueur_caracteres,
        array_length(regexp_split_to_array(trim(a.contenu), '\\s+'), 1) AS longueur_mots
    FROM public.articles_rss a
    JOIN public.article_companies ac ON ac.article_id = a.id
    WHERE ac.company_id = %s
    AND a.contenu IS NOT NULL
    ORDER BY a.id DESC
    {CLAUSE_LIMITE}
"""

# Variante reprenable : exclut les couples (article, entreprise) déjà évalués
# avec succès PAR CE MODÈLE et POUR CETTE VERSION DE PROMPT.
#
# Sans ce filtre, relancer après une interruption (quota batch atteint, coupure
# réseau, arrêt REMOVEDel) resoumet tout depuis le début — donc repaie les articles
# déjà traités et sature à nouveau le quota.
#
# Les lignes en statut 'failed' sont volontairement RECONSERVÉES : un échec doit
# être réessayé au run suivant. Idem pour un changement de version de prompt, qui
# doit tout réévaluer puisque les notes ne sont plus comparables.
#
# 'submitted' est le troisième état, indispensable en mode batch : une ligne
# soumise mais pas encore collectée n'est ni 'ok' ni disponible. Sans lui, un
# second run lancé pendant qu'un batch est en vol resoumet (et refacture) les
# mêmes articles, puisque leur statut ne passera à 'ok' qu'au --collect.
#
# IS DISTINCT FROM (et non <> / NOT ... =) est indispensable ici : en SQL,
# NULL <> 'ok' vaut NULL, pas TRUE. Avec un simple NOT(statut='ok' AND version=%s),
# toutes les lignes jamais évaluées (statut NULL) seraient silencieusement
# exclues — soit précisément celles qu'on veut traiter. IS DISTINCT FROM traite
# NULL comme une valeur ordinaire et renvoie bien TRUE.
QUERY_ARTICLES_REPRENABLE = f"""
    SELECT
        a.id,
        a.contenu,
        LENGTH(a.contenu) AS longueur_caracteres,
        array_length(regexp_split_to_array(trim(a.contenu), '\\s+'), 1) AS longueur_mots
    FROM public.articles_rss a
    JOIN public.article_companies ac ON ac.article_id = a.id
    WHERE ac.company_id = %s
    AND a.contenu IS NOT NULL
    AND (
        (ac.{{colonne_statut}} IS DISTINCT FROM 'ok'
         AND ac.{{colonne_statut}} IS DISTINCT FROM 'submitted')
        OR ac.{{colonne_prompt_version}} IS DISTINCT FROM %s
    )
    ORDER BY a.id DESC
    {CLAUSE_LIMITE}
"""


def recuperer_articles_entreprise(cur, company_id, modele=None, version_prompt=None):
    """
    Récupère les articles d'une entreprise.

    Si modele ET version_prompt sont fournis, applique le filtre de reprise :
    seuls les articles pas encore évalués avec succès pour cette combinaison
    modèle/prompt sont renvoyés. Sinon (rétrocompatibilité), renvoie tout.

    Le filtre est désactivable via REPRENDRE=0, pour forcer une réévaluation
    complète (ex: pour rejouer un run entier après correction d'un bug).
    """
    reprendre = os.getenv("REPRENDRE", "1").strip() not in ("0", "false", "False")

    if modele and version_prompt and reprendre:
        # Les noms de colonnes viennent de dictionnaires internes (jamais d'une
        # entrée utilisateur), leur interpolation en f-string est donc sûre ;
        # les valeurs restent passées en paramètres liés.
        requete = QUERY_ARTICLES_REPRENABLE.format(
            colonne_statut=COLONNE_STATUT[modele],
            colonne_prompt_version=COLONNE_PROMPT_VERSION[modele],
        )
        cur.execute(requete, (company_id, version_prompt))
        return cur.fetchall()

    cur.execute(QUERY_ARTICLES, (company_id,))
    return cur.fetchall()

# ==========================================
# ÉVALUATION LLM
# ==========================================
def evaluer_sentiment_llm(texte_article, nom_entreprise, modele, gemini_client=None, prompt_version="v1"):
    """
    Retourne un tuple (resultat, statut).
    - statut == "ok"     -> resultat est un dict {"note_llm": int, "justification": str}
    - statut == "failed" -> resultat est None (échec définitif après retries, ou erreur
                             non retryable). Ne JAMAIS interpréter ça comme une note 0.
    Peut lever FatalLLMError pour les erreurs de config/auth (400/401/403/404), qui
    doivent stopper le run entier plutôt que d'être comptées comme un échec par article.

    prompt_version : clé de PROMPTS ("v1" par défaut, "v2" = version enrichie
    few-shot + règle anti-repli-neutre). Permet l'A/B testing sans dupliquer le script.

    Simple dispatcher vers le provider adapté au modèle demandé (voir providers/).
    Comportement strictement identique à l'ancienne implémentation inline, seule
    l'organisation du code change :
      - lama/mistral/queen -> OllamaProvider (local, gratuit)
      - gemini             -> GeminiProvider (distant, payant)
      - haiku              -> HaikuProvider  (distant, payant)
    """
    if modele in MODELES_OLLAMA:
        return OllamaProvider(modele).evaluer_un(
            texte_article, nom_entreprise, prompt_version=prompt_version
        )
    if modele == "gemini":
        return GeminiProvider(client=gemini_client).evaluer_un(
            texte_article, nom_entreprise, prompt_version=prompt_version
        )
    if modele == "haiku":
        return HaikuProvider().evaluer_un(
            texte_article, nom_entreprise, prompt_version=prompt_version
        )

    raise ValueError(f"Modèle inconnu: {modele!r}")

def choisir_modele():
    choix_valides = ["gemini", "haiku", "lama", "mistral", "queen"]
    print("\nModèles disponibles :")
    print("  gemini  -> Google Gemini 2.5 Flash (API distante)")
    print("  haiku   -> Claude Haiku (API distante Anthropic)")
    print("  lama    -> LLaMA 3 via Ollama (local)")
    print("  mistral -> Mistral Nemo via Ollama (local)")
    print("  queen   -> Qwen 2.5 7B via Ollama (local)")
    while True:
        choix = input("\nChoisissez le modèle [gemini/haiku/lama/mistral/queen] : ").strip().lower()
        if choix in choix_valides:
            return choix
        print(f"Choix invalide. Entrez l'un de : {', '.join(choix_valides)}")

def choisir_version_prompt():
    """
    Sélectionne la version de prompt à utiliser (A/B testing).
    Priorité à la variable d'environnement PROMPT_VERSION pour permettre de
    lancer des runs scriptés (ex: deux appels successifs du script, un par
    version, sans interaction REMOVEDelle) :
        PROMPT_VERSION=v2 python evaluate_article.py
    """
    depuis_env = os.getenv("PROMPT_VERSION")
    if depuis_env:
        depuis_env = depuis_env.strip().lower()
        if depuis_env in PROMPTS:
            print(f"\nVersion de prompt (via PROMPT_VERSION) : {depuis_env}")
            return depuis_env
        print(f"PROMPT_VERSION={depuis_env!r} invalide, ignoré. Versions dispo: {', '.join(PROMPTS)}")

    print("\nVersions de prompt disponibles :")
    print("  v1 -> Prompt original (règles courtes, NEUTRAL par défaut)")
    print("  v2 -> Prompt enrichi (isolation des phrases pertinentes, few-shot, anti-repli-neutre)")
    print("  v3 -> v2 + règles dédiées objectifs de cours / recommandations de brokers")
    print("  v4 -> v3 corrigé (lecture factuelle explicite + grille complète 3x4)")
    print("  v5 -> v2 + règles issues de l'annotation REMOVEDelle (intraday, palmarès, matérialité)")
    print("  v6 -> v2 + extraction structurée : le LLM constate, le code décide")
    print("  v7 -> v6 avec autre_fait_concret corrigé (la variation de cours n'est plus un fait)")
    while True:
        choix = input(f"\nChoisissez la version de prompt [{'/'.join(PROMPTS)}] : ").strip().lower()
        if choix in PROMPTS:
            return choix
        print(f"Choix invalide. Entrez l'un de : {', '.join(PROMPTS)}")

def choisir_mode(modele):
    """
    Sélectionne le mode d'exécution :
      - 'sync'  : appel immédiat article par article (comportement historique).
      - 'batch' : soumission par lot via l'API Batch du fournisseur (-50% de
                  coût, jusqu'à 24h de délai avant d'avoir les résultats).
    Le mode batch n'existe que pour les modèles distants payants (gemini,
    haiku) ; les modèles locaux Ollama restent toujours en mode sync (pas de
    batching serveur chez Ollama).

    Priorité à la variable d'environnement MODE, sur le même principe que
    PROMPT_VERSION, pour permettre des runs scriptés sans interaction :
        MODE=batch python evaluate_article.py
    """
    if modele not in MODELES_BATCH_DISPONIBLE:
        print(f"\nMode d'exécution : sync (seul mode disponible pour {modele!r})")
        return "sync"

    depuis_env = os.getenv("MODE")
    if depuis_env:
        depuis_env = depuis_env.strip().lower()
        if depuis_env in ("sync", "batch"):
            print(f"\nMode d'exécution (via MODE) : {depuis_env}")
            return depuis_env
        print(f"MODE={depuis_env!r} invalide, ignoré. Valeurs possibles: sync, batch")

    print("\nModes disponibles :")
    print("  sync  -> appel immédiat article par article (comme avant)")
    print("  batch -> soumission par lot (-50% de coût, jusqu'à 24h de délai)")
    while True:
        choix = input("\nChoisissez le mode [sync/batch] : ").strip().lower()
        if choix in ("sync", "batch"):
            return choix
        print("Choix invalide. Entrez 'sync' ou 'batch'.")

# ==========================================
# FONCTION PRINCIPALE
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Évaluation de sentiment d'articles par LLM (sync ou batch)."
    )
    parser.add_argument(
        "--collect",
        metavar="FICHIER_BATCH",
        help=(
            "Ne fait que vérifier/récupérer les résultats d'un batch déjà soumis "
            "(fichier output/batches_*.json généré par un run en mode batch). "
            "Ne bloque pas : à relancer plus tard (REMOVEDellement ou via cron) tant "
            "que des batchs sont encore en cours."
        ),
    )
    args = parser.parse_args()

    if args.collect:
        recuperer_batch_run(args.collect)
        return

    modele = choisir_modele()
    version_prompt = choisir_version_prompt()
    mode = choisir_mode(modele)

    if mode == "batch":
        soumettre_batch_run(modele, version_prompt)
    else:
        main_sync(modele, version_prompt)


def main_sync(modele, version_prompt):
    print(f"\nModèle sélectionné : {modele} (mode sync)")

    gemini_client = None
    if modele == "gemini":
        gemini_client = construire_client_gemini()

    print("Connexion à la BDD...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Charger les alias
    dict_alias = charger_dictionnaire_alias("output/alias.json")

    # Récupérer les entreprises
    cur.execute("SELECT id, name FROM public.companies ORDER BY id")
    companies = cur.fetchall()

    total_evalues = 0
    total_echecs = 0

    date_jour = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    nom_fichier_csv = f"output/resultats_{modele}_{version_prompt}_{date_jour}.csv"
    nom_fichier_log = f"output/log_articles_{modele}_{version_prompt}_{date_jour}.txt"

    colonnes_csv = ['article_id', 'entreprise', 'prompt_version', 'statut', 'note_llm', 'justification',
                    'longueur_caracteres', 'longueur_mots', 'texte_extrait']

    # On ouvre le fichier en mode "append" (ajout) pour écrire ligne par ligne
    # C'est plus sûr en cas de plantage du script au milieu du traitement.
    with open(nom_fichier_csv, mode='w', newline='', encoding='utf-8') as fichier_csv, \
         open(nom_fichier_log, mode='w', encoding='utf-8') as fichier_log:
        writer = csv.DictWriter(fichier_csv, fieldnames=colonnes_csv)
        writer.writeheader()
        print(f"Fichier de sauvegarde créé : {nom_fichier_csv}")
        print(f"Fichier de log créé : {nom_fichier_log}\n")

        # Fonction pour logger
        def log_article(article_id, company_name, nbocc, raison, resultat="", statut=""):
            message = f"[{article_id:5d}] {company_name:30} | nbocc={nbocc} | {raison:40} | {resultat:20} | {statut}"
            fichier_log.write(message + "\n")
            fichier_log.flush()

        # En-tête du log
        fichier_log.write("="*120 + "\n")
        fichier_log.write(f"LOG DE TRAITEMENT - Modèle: {modele}, Version prompt: {version_prompt}, Date: {date_jour}\n")
        fichier_log.write("="*120 + "\n")
        fichier_log.write(f"{'Article':>6} | {'Company':30} | nbocc | {'Raison':40} | {'Résultat':20} | Statut\n")
        fichier_log.write("-"*120 + "\n")
        fichier_log.flush()

        for company in companies:
            company_id = company['id']
            company_name = company['name']

            # Préparer le filtre pour cette entreprise
            alias_liste = dict_alias.get(company_name, [])
            regex_entreprise = compiler_regex_entreprise(company_name, alias_liste)

            print(f"\n--- Traitement de [{company_id}] {company_name} ---")
            fichier_log.write(f"\n>>> ENTREPRISE: {company_name} (ID={company_id})\n")

            # Récupérer les articles pas encore évalués avec succès pour ce
            # couple (modèle, version de prompt). REPRENDRE=0 pour tout rejouer.
            articles = recuperer_articles_entreprise(cur, company_id, modele, version_prompt)

            for article in articles:
                texte = article['contenu']
                article_id = article['id']
                longueur_caracteres = article['longueur_caracteres']
                longueur_mots = article['longueur_mots']

                # 1. Filtrage par occurrence (> 1)
                nb_occ = compter_occurrences(texte, regex_entreprise)

                if nb_occ > 1:
                    print(f"Article {article_id} retenu ({nb_occ} occurrences). Évaluation LLM en cours...")
                    log_article(article_id, company_name, nb_occ, "Sélectionné pour évaluation", "", "")

                    # 2. Évaluation LLM
                    try:
                        resultat, statut = evaluer_sentiment_llm(
                            texte, company_name, modele, gemini_client,
                            prompt_version=version_prompt,
                        )
                    except FatalLLMError as e:
                        print(f"Erreur bloquante API distante: {e}")
                        print("Arrêt du traitement pour éviter de répéter la même erreur.")
                        cur.close()
                        conn.close()
                        return

                    colonne_note = COLONNE_NOTE[modele]
                    colonne_justif = COLONNE_JUSTIF[modele]
                    colonne_statut = COLONNE_STATUT[modele]
                    colonne_prompt_version = COLONNE_PROMPT_VERSION[modele]
                    colonne_extraction = COLONNE_EXTRACTION[modele]

                    if statut == "ok" and resultat:
                        note = resultat.get("note_llm")
                        justif = resultat.get("justification", "")

                        print(f"  -> Note: {note} | Justif: {justif[:80]}...")
                        log_article(article_id, company_name, nb_occ, "Évaluation réussie", f"Note={note}", "ok")

                        # 3. Sauvegarder en BDD : note, justification, statut, version de
                        # prompt ET faits extraits (v6). Attention : le run le plus récent
                        # écrase le précédent en base (colonnes non-cumulatives). Pour
                        # comparer deux versions sans perte, se référer aux fichiers CSV
                        # horodatés, qui gardent l'historique complet de chaque run.
                        #
                        # note_finale() applique les règles déterministes aux faits
                        # extraits : la note stockée peut donc différer de celle du
                        # modèle. origine_note trace laquelle des deux a été retenue.
                        note, origine_note = note_finale(resultat)
                        extraction = _extraire_faits(resultat)
                        update_query = f'''
                            UPDATE public.article_companies
                            SET {colonne_note} = %s, {colonne_justif} = %s, {colonne_statut} = %s,
                                {colonne_prompt_version} = %s, {colonne_extraction} = %s
                            WHERE article_id = %s AND company_id = %s
                        '''
                        cur.execute(update_query, (note, justif, statut, version_prompt,
                                                   extraction, article_id, company_id))
                        conn.commit()

                        writer.writerow({
                            'article_id': article_id,
                            'entreprise': company_name,
                            'prompt_version': version_prompt,
                            'statut': statut,
                            'note_llm': note,
                            'justification': justif,
                            'longueur_caracteres': longueur_caracteres,
                            'longueur_mots': longueur_mots,
                            # On garde les 200 premiers caractères du texte pour s'y retrouver
                            'texte_extrait': texte[:200].replace('\n', ' ') + '...'
                        })

                        # Pour forcer l'écriture sur le disque immédiatement (optionnel mais sécurisant)
                        fichier_csv.flush()

                        total_evalues += 1
                    else:
                        # Échec définitif après retries : on remet explicitement note et
                        # justification à NULL, et on trace le statut 'failed'.
                        #
                        # Le passage à NULL est indispensable : sans lui, la note d'un run
                        # PRÉCÉDENT (autre version de prompt, voire autre modèle) resterait en
                        # base tout en étant réétiquetée avec la version courante. On obtenait
                        # alors des lignes 'failed' portant une note orpheline, comptées à tort
                        # dans les analyses de qualité (cas constaté sur 19 lignes du run v3).
                        print(f"  -> Échec de l'évaluation pour l'article {article_id} (statut=failed).")
                        log_article(article_id, company_name, nb_occ, "Évaluation échouée", "Statut=failed", "failed")

                        update_query = f'''
                            UPDATE public.article_companies
                            SET {colonne_note} = NULL, {colonne_justif} = NULL,
                                {colonne_statut} = %s, {colonne_prompt_version} = %s
                            WHERE article_id = %s AND company_id = %s
                        '''
                        cur.execute(update_query, (statut, version_prompt, article_id, company_id))
                        conn.commit()

                        writer.writerow({
                            'article_id': article_id,
                            'entreprise': company_name,
                            'prompt_version': version_prompt,
                            'statut': statut,
                            'note_llm': '',
                            'justification': '',
                            'longueur_caracteres': longueur_caracteres,
                            'longueur_mots': longueur_mots,
                            'texte_extrait': texte[:200].replace('\n', ' ') + '...'
                        })
                        fichier_csv.flush()

                        total_echecs += 1

                    # API distantes limitees en debit, pause plus longue
                    time.sleep(4 if modele in ("gemini", "haiku") else 1)
                else:
                    # L'article mentionne l'entreprise 1 fois ou 0 fois (faux positif de jointure)
                    if nb_occ == 1:
                        log_article(article_id, company_name, nb_occ, "Filtré: nbocc=1", "SKIPPED", "not_evaluated")
                    elif nb_occ == 0:
                        log_article(article_id, company_name, nb_occ, "Filtré: nbocc=0 (faux positif)", "SKIPPED", "not_evaluated")
                    else:
                        log_article(article_id, company_name, nb_occ, f"Filtré: nbocc={nb_occ}", "SKIPPED", "not_evaluated")

        # Fermer le fichier de log avec un résumé final
        fichier_log.write("\n" + "="*120 + "\n")
        fichier_log.write("RÉSUMÉ FINAL\n")
        fichier_log.write("="*120 + "\n")
        fichier_log.write(f"Modèle: {modele}\n")
        fichier_log.write(f"Version de prompt: {version_prompt}\n")
        fichier_log.write(f"Articles évalués avec succès (ok): {total_evalues}\n")
        fichier_log.write(f"Articles échoués (failed): {total_echecs}\n")
        fichier_log.write(f"Total: {total_evalues + total_echecs}\n")

    cur.close()
    conn.close()
    print(f"\nTerminé ! {total_evalues} articles évalués avec succès, {total_echecs} échecs (statut='failed', à revoir/relancer).")
    print(f"📋 Consulte le log: {nom_fichier_log}")


def construire_provider_batch(modele):
    fabrique = MODELES_BATCH_DISPONIBLE.get(modele)
    if fabrique is None:
        raise ValueError(f"Le mode batch n'est pas supporté pour le modèle {modele!r}.")
    return fabrique()


def _sauver_meta_batch(chemin, modele, version_prompt, batches_meta, requetes_par_custom_id,
                       nom_fichier_csv, nom_fichier_log, non_soumis=None):
    """
    Écrit le fichier de métadonnées d'un run batch.

    Extrait dans sa propre fonction pour pouvoir être appelé AUSSI en cas d'échec
    partiel : sans ça, un chunk en erreur au milieu de la boucle faisait sortir la
    fonction avant l'écriture, et les batchs déjà soumis (donc déjà facturés)
    devenaient irrécupérables faute de fichier reliant batch_id et articles.

    non_soumis : liste des custom_id collectés mais jamais soumis (quota atteint).
    Ils sont conservés pour permettre une reprise ultérieure.
    """
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({
            "modele": modele,
            "prompt_version": version_prompt,
            "batches": batches_meta,
            "requetes": requetes_par_custom_id,
            "non_soumis": non_soumis or [],
            "csv": nom_fichier_csv,
            "log": nom_fichier_log,
        }, f, ensure_ascii=False)


def soumettre_batch_run(modele, version_prompt):
    """
    Phase 1+2 seulement : collecte des articles éligibles (mêmes règles qu'en
    mode sync, nb_occ > 1) puis soumission en batch(s) de BATCH_TAILLE_MAX
    requêtes. Retour immédiat (pas d'attente des résultats) : les métadonnées
    (batch_id + requêtes + emplacement des fichiers de sortie) sont écrites
    dans output/batches_{modele}_{version}_{date}.json.

    La récupération des résultats se fait ensuite, indépendamment et autant
    de fois que nécessaire, via recuperer_batch_run() / `--collect` (voir
    plus bas) : le batch continue de tourner côté fournisseur même si cette
    machine est éteinte entre les deux.
    """
    print(f"\nModèle sélectionné : {modele} (mode batch — soumission uniquement)")
    provider = construire_provider_batch(modele)

    # Colonnes BDD propres au modèle, utilisées pour marquer les lignes 'submitted'
    # dès la soumission (voir la boucle de soumission plus bas).
    colonne_statut = COLONNE_STATUT[modele]
    colonne_prompt_version = COLONNE_PROMPT_VERSION[modele]
    colonne_extraction = COLONNE_EXTRACTION[modele]

    print("Connexion à la BDD...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    dict_alias = charger_dictionnaire_alias("output/alias.json")
    cur.execute("SELECT id, name FROM public.companies ORDER BY id")
    companies = cur.fetchall()

    date_jour = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    nom_fichier_csv = f"output/resultats_{modele}_{version_prompt}_{date_jour}.csv"
    nom_fichier_log = f"output/log_articles_{modele}_{version_prompt}_{date_jour}.txt"
    nom_fichier_batches = f"output/batches_{modele}_{version_prompt}_{date_jour}.json"

    with open(nom_fichier_log, mode='w', encoding='utf-8') as fichier_log:

        def log_article(article_id, company_name, nbocc, raison, resultat="", statut=""):
            message = f"[{article_id:5d}] {company_name:30} | nbocc={nbocc} | {raison:40} | {resultat:20} | {statut}"
            fichier_log.write(message + "\n")
            fichier_log.flush()

        fichier_log.write("="*120 + "\n")
        fichier_log.write(f"LOG DE TRAITEMENT (BATCH) - Modèle: {modele}, Version prompt: {version_prompt}, Date: {date_jour}\n")
        fichier_log.write("="*120 + "\n")
        fichier_log.flush()

        # ---------- PHASE 1 : COLLECTE ----------
        print("\n--- Phase 1/2 : collecte des articles éligibles ---")
        requetes = []
        total_filtres = 0

        for company in companies:
            company_id = company['id']
            company_name = company['name']
            alias_liste = dict_alias.get(company_name, [])
            regex_entreprise = compiler_regex_entreprise(company_name, alias_liste)

            articles = recuperer_articles_entreprise(cur, company_id, modele, version_prompt)

            for article in articles:
                texte = article['contenu']
                article_id = article['id']
                nb_occ = compter_occurrences(texte, regex_entreprise)

                if nb_occ > 1:
                    requetes.append({
                        "custom_id": construire_custom_id(article_id, company_id),
                        "article_id": article_id,
                        "company_id": company_id,
                        "company_name": company_name,
                        "texte": texte,
                        "longueur_caracteres": article['longueur_caracteres'],
                        "longueur_mots": article['longueur_mots'],
                    })
                    log_article(article_id, company_name, nb_occ, "Sélectionné pour le batch", "", "")
                else:
                    total_filtres += 1
                    log_article(article_id, company_name, nb_occ, f"Filtré: nbocc={nb_occ}", "SKIPPED", "not_evaluated")

        print(f"Collecte terminée : {len(requetes)} article(s) à évaluer, {total_filtres} filtré(s).")

        if not requetes:
            print("Aucun article éligible pour ce run, arrêt.")
            fichier_log.write("\nAucun article éligible pour le batch.\n")
            cur.close()
            conn.close()
            return None

        # ---------- PHASE 2 : SOUMISSION ----------
        print("\n--- Phase 2/2 : soumission des batchs ---")
        chunks = [requetes[i:i + BATCH_TAILLE_MAX] for i in range(0, len(requetes), BATCH_TAILLE_MAX)]
        batches_meta = []

        for i, chunk in enumerate(chunks, start=1):
            print(f"Soumission du chunk {i}/{len(chunks)} ({len(chunk)} requête(s))...")
            requetes_provider = [
                {"custom_id": r["custom_id"], "texte_article": r["texte"], "nom_entreprise": r["company_name"]}
                for r in chunk
            ]
            try:
                batch_id = provider.soumettre_batch(requetes_provider, prompt_version=version_prompt)
            except FatalLLMError as e:
                print(f"Erreur bloquante lors de la soumission du batch: {e}")
                fichier_log.write(f"\nErreur bloquante à la soumission (chunk {i}/{len(chunks)}): {e}\n")

                # IMPORTANT : les chunks déjà soumis tournent (et sont facturés) côté
                # fournisseur. On sauvegarde donc les métadonnées AVANT de sortir, sinon
                # ces batchs deviennent irrécupérables (aucun fichier ne relie les
                # batch_id aux articles). Les requêtes non soumises sont conservées à
                # part pour pouvoir être reprises plus tard.
                custom_ids_soumis = {cid for b in batches_meta for cid in b["custom_ids"]}
                requetes_non_soumises = [r["custom_id"] for r in requetes
                                         if r["custom_id"] not in custom_ids_soumis]

                if batches_meta:
                    _sauver_meta_batch(
                        nom_fichier_batches, modele, version_prompt, batches_meta,
                        {r["custom_id"]: r for r in requetes},
                        nom_fichier_csv, nom_fichier_log,
                        non_soumis=requetes_non_soumises,
                    )
                    fichier_log.write(f"\nMétadonnées partielles sauvegardées: {nom_fichier_batches}\n")
                    print(f"\n{len(custom_ids_soumis)} requête(s) déjà soumise(s) dans "
                          f"{len(batches_meta)} batch(s) : métadonnées sauvegardées malgré l'erreur.")
                    print(f"Métadonnées : {nom_fichier_batches}")
                    print(f"{len(requetes_non_soumises)} requête(s) NON soumise(s), à relancer plus tard.")
                    print("Récupérez d'abord les batchs en cours (ce qui libère aussi le quota "
                          "de jetons en file d'attente) :")
                    print(f"  python evaluate_article.py --collect {nom_fichier_batches}")
                else:
                    print("Aucun batch n'a été soumis : rien à récupérer.")

                cur.close()
                conn.close()
                return nom_fichier_batches if batches_meta else None
            print(f"  -> batch_id={batch_id}")
            fichier_log.write(f"\nBatch soumis: {batch_id} ({len(chunk)} requête(s))\n")

            # Marque immédiatement les lignes comme 'submitted' : elles sont en vol,
            # ni évaluées ni disponibles. Sans ce marquage, un run lancé avant le
            # --collect les considérerait comme "à faire" et les resoumettrait.
            # Le statut passera à 'ok'/'failed' à la collecte.
            for r in chunk:
                cur.execute(
                    f'''
                        UPDATE public.article_companies
                        SET {colonne_statut} = 'submitted', {colonne_prompt_version} = %s
                        WHERE article_id = %s AND company_id = %s
                    ''',
                    (version_prompt, r["article_id"], r["company_id"]),
                )
            conn.commit()

            # statut par batch : 'soumis' -> 'recupere' (ok) ou 'echec' (batch en
            # échec définitif). Permet à recuperer_batch_run() de ne retraiter
            # que ce qui reste réellement en attente, même sur plusieurs runs.
            batches_meta.append({
                "batch_id": batch_id,
                "custom_ids": [r["custom_id"] for r in chunk],
                "statut": "soumis",
            })

        requetes_par_custom_id = {r["custom_id"]: r for r in requetes}
        _sauver_meta_batch(
            nom_fichier_batches, modele, version_prompt, batches_meta,
            requetes_par_custom_id, nom_fichier_csv, nom_fichier_log,
        )
        fichier_log.write(f"\nMétadonnées sauvegardées: {nom_fichier_batches}\n")

    cur.close()
    conn.close()

    print(f"\n{len(requetes)} article(s) soumis dans {len(batches_meta)} batch(s).")
    print(f"Métadonnées : {nom_fichier_batches}")
    print("Le batch continue de tourner côté fournisseur même si cette machine "
          "s'éteint. Pour récupérer les résultats (à relancer tant que des "
          "batchs sont en cours, ex. via une tâche planifiée) :")
    print(f"  python evaluate_article.py --collect {nom_fichier_batches}")
    return nom_fichier_batches


def _extraire_faits(resultat):
    """
    Sérialise les faits bruts extraits par le prompt v6 pour stockage en JSONB.

    Retourne None si aucun champ d'extraction n'est présent (versions v1 a v5),
    ce qui laisse la colonne NULL plutot que d'y ecrire un objet vide.
    """
    faits = {k: resultat.get(k) for k in CHAMPS_EXTRACTION}
    if all(v is None for v in faits.values()):
        return None
    return json.dumps(faits, ensure_ascii=False)


def _ecrire_resultat_batch(cur, writer, log_article, req, resultat, statut, version_prompt,
                            colonne_note, colonne_justif, colonne_statut, colonne_prompt_version,
                            colonne_extraction):
    """Écrit un résultat de batch (BDD + ligne CSV + log), même contrat que le mode sync."""
    article_id = req["article_id"]
    company_id = req["company_id"]
    company_name = req["company_name"]
    texte = req["texte"]

    if statut == "ok" and resultat:
        note, origine_note = note_finale(resultat)
        justif = resultat.get("justification", "")
        extraction = _extraire_faits(resultat)
        log_article(article_id, company_name, "?", "Évaluation réussie",
                    f"Note={note} ({origine_note})", "ok")
        cur.execute(
            f'''
                UPDATE public.article_companies
                SET {colonne_note} = %s, {colonne_justif} = %s, {colonne_statut} = %s,
                    {colonne_prompt_version} = %s, {colonne_extraction} = %s
                WHERE article_id = %s AND company_id = %s
            ''',
            (note, justif, statut, version_prompt, extraction, article_id, company_id),
        )
        writer.writerow({
            'article_id': article_id,
            'entreprise': company_name,
            'prompt_version': version_prompt,
            'statut': statut,
            'note_llm': note,
            'justification': justif,
            'longueur_caracteres': req["longueur_caracteres"],
            'longueur_mots': req["longueur_mots"],
            'texte_extrait': texte[:200].replace('\n', ' ') + '...',
        })
        return True

    # Échec : on remet note et justification à NULL (même raison qu'en mode sync —
    # sinon la note d'un run précédent survit en base sous la version courante).
    log_article(article_id, company_name, "?", "Évaluation échouée", "Statut=failed", "failed")
    cur.execute(
        f'''
            UPDATE public.article_companies
            SET {colonne_note} = NULL, {colonne_justif} = NULL,
                {colonne_statut} = %s, {colonne_prompt_version} = %s
            WHERE article_id = %s AND company_id = %s
        ''',
        (statut, version_prompt, article_id, company_id),
    )
    writer.writerow({
        'article_id': article_id,
        'entreprise': company_name,
        'prompt_version': version_prompt,
        'statut': statut,
        'note_llm': '',
        'justification': '',
        'longueur_caracteres': req["longueur_caracteres"],
        'longueur_mots': req["longueur_mots"],
        'texte_extrait': texte[:200].replace('\n', ' ') + '...',
    })
    return False


def recuperer_batch_run(chemin_meta):
    """
    Vérifie l'état des batchs référencés par chemin_meta (fichier produit par
    soumettre_batch_run()) et écrit les résultats de ceux qui sont terminés.

    Ne bloque JAMAIS : une seule vérification par batch, puis retour. À
    relancer plus tard (REMOVEDellement ou via une tâche planifiée) tant qu'il
    reste des batchs en cours — c'est ce découpage qui permet d'éteindre la
    machine entre la soumission et la récupération, le batch tournant
    entièrement côté fournisseur.

    Idempotent : les batchs déjà récupérés (statut 'recupere'/'echec' dans le
    fichier de métadonnées) ne sont pas retraités ; les fichiers CSV/log sont
    ouverts en ajout pour accumuler les résultats de plusieurs exécutions.
    """
    with open(chemin_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    modele = meta["modele"]
    version_prompt = meta["prompt_version"]
    provider = construire_provider_batch(modele)
    requetes_par_custom_id = meta["requetes"]
    nom_fichier_csv = meta["csv"]
    nom_fichier_log = meta["log"]

    batches_a_verifier = [b for b in meta["batches"] if b.get("statut") == "soumis"]
    if not batches_a_verifier:
        print("Tous les batchs de ce fichier ont déjà été récupérés (ou sont en échec définitif).")
        return

    colonne_note = COLONNE_NOTE[modele]
    colonne_justif = COLONNE_JUSTIF[modele]
    colonne_statut = COLONNE_STATUT[modele]
    colonne_prompt_version = COLONNE_PROMPT_VERSION[modele]
    colonne_extraction = COLONNE_EXTRACTION[modele]

    colonnes_csv = ['article_id', 'entreprise', 'prompt_version', 'statut', 'note_llm', 'justification',
                    'longueur_caracteres', 'longueur_mots', 'texte_extrait']
    csv_existe_deja = os.path.exists(nom_fichier_csv)

    conn = None
    cur = None
    total_evalues = 0
    total_echecs = 0

    with open(nom_fichier_log, mode='a', encoding='utf-8') as fichier_log, \
         open(nom_fichier_csv, mode='a', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.DictWriter(fichier_csv, fieldnames=colonnes_csv)
        if not csv_existe_deja:
            writer.writeheader()

        def log_article(article_id, company_name, nbocc, raison, resultat="", statut=""):
            message = f"[{article_id:5d}] {company_name:30} | nbocc={nbocc} | {raison:40} | {resultat:20} | {statut}"
            fichier_log.write(message + "\n")
            fichier_log.flush()

        fichier_log.write(f"\n--- Récupération lancée le {datetime.now().isoformat()} ---\n")

        for b in batches_a_verifier:
            batch_id = b["batch_id"]
            print(f"Vérification du batch {batch_id}...")
            try:
                statut_batch = provider.statut_batch(batch_id)
            except FatalLLMError as e:
                print(f"  -> échec définitif: {e}")
                fichier_log.write(f"\nBatch {batch_id} en échec définitif: {e}\n")
                if conn is None:
                    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
                                             password=DB_PASSWORD, dbname=DB_NAME)
                    cur = conn.cursor(cursor_factory=RealDictCursor)
                for custom_id in b["custom_ids"]:
                    req = requetes_par_custom_id[custom_id]
                    _ecrire_resultat_batch(cur, writer, log_article, req, None, "failed", version_prompt,
                                            colonne_note, colonne_justif, colonne_statut, colonne_prompt_version,
                                            colonne_extraction)
                    total_echecs += 1
                conn.commit()
                b["statut"] = "echec"
                continue

            if statut_batch == "in_progress":
                print("  -> encore en cours.")
                continue

            print("  -> terminé, récupération des résultats...")
            if conn is None:
                conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
                                         password=DB_PASSWORD, dbname=DB_NAME)
                cur = conn.cursor(cursor_factory=RealDictCursor)

            try:
                resultats = provider.recuperer_resultats(batch_id)
            except FatalLLMError as e:
                print(f"  -> erreur lors de la récupération: {e}")
                fichier_log.write(f"\nErreur récupération batch {batch_id}: {e}\n")
                continue  # on retentera au prochain --collect (statut reste 'soumis')

            for custom_id in b["custom_ids"]:
                req = requetes_par_custom_id[custom_id]
                resultat, statut_res = resultats.get(custom_id, (None, "failed"))
                ok = _ecrire_resultat_batch(cur, writer, log_article, req, resultat, statut_res, version_prompt,
                                             colonne_note, colonne_justif, colonne_statut, colonne_prompt_version,
                                        colonne_extraction)
                if ok:
                    total_evalues += 1
                else:
                    total_echecs += 1
            conn.commit()
            b["statut"] = "recupere"

        fichier_csv.flush()

    if conn is not None:
        cur.close()
        conn.close()

    # Persiste les statuts mis à jour, pour que les prochains --collect ne
    # retraitent pas ce qui vient d'être fait.
    with open(chemin_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    restants = [b for b in meta["batches"] if b.get("statut") == "soumis"]
    print(f"\n{total_evalues} évalué(s), {total_echecs} échoué(s) écrit(s) lors de cette récupération.")
    if restants:
        print(f"{len(restants)} batch(s) encore en cours. Relancer plus tard :")
        print(f"  python evaluate_article.py --collect {chemin_meta}")
    else:
        print("Tous les batchs de ce fichier ont été traités.")


if __name__ == "__main__":
    main()
