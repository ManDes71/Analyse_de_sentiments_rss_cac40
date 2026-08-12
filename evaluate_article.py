import os
import json
import time
import re
import csv
import random
import requests
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

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
# Valeurs possibles pour statut_* : 'not_evaluated' (jamais soumis, ex. filtré par
# nb_occ <= 1), 'ok' (note/justification fiables), 'failed' (appel tenté, jamais
# abouti après retries -> note/justification restent NULL, à exclure de toute
# analyse de qualité et à reprendre lors d'un futur run).
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
# CONFIGURATION GOOGLE GEMINI
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ==========================================
# CONFIGURATION ANTHROPIC HAIKU
# ==========================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Identifiant valide sur ce endpoint: alias "claude-haiku-4-5" (resolu cote API).
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-haiku-4-5")
ANTHROPIC_API_URL = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

MODELES_OLLAMA = {
    "lama":    "llama3.1:8b",
    "mistral": "mistral-nemo",
    "queen":   "qwen2.5:7b",
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

# ==========================================
# CONFIGURATION RETRY / BACKOFF (429, 5xx)
# ==========================================
RETRY_MAX_TENTATIVES = int(os.getenv("RETRY_MAX_TENTATIVES", 4))
RETRY_DELAI_BASE      = float(os.getenv("RETRY_DELAI_BASE", 2))   # secondes
RETRY_DELAI_MAX       = float(os.getenv("RETRY_DELAI_MAX", 60))   # secondes
STATUS_CODES_RETRYABLES = {429, 500, 502, 503, 504}



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
# RETRY / BACKOFF
# ==========================================
def attendre_avec_backoff(tentative):
    """Backoff exponentiel avec jitter. tentative commence à 0."""
    delai = min(RETRY_DELAI_BASE * (2 ** tentative), RETRY_DELAI_MAX)
    delai += random.uniform(0, 1)
    print(f"    Nouvelle tentative dans {delai:.1f}s (tentative {tentative + 1}/{RETRY_MAX_TENTATIVES})...")
    time.sleep(delai)

def extraire_status_code_gemini(exception):
    """
    La lib google-genai n'expose pas toujours un status_code de façon uniforme
    selon les versions. On essaie plusieurs attributs connus, puis on retombe
    sur une recherche regex dans le message d'erreur.
    """
    for attribut in ("status_code", "code"):
        valeur = getattr(exception, attribut, None)
        if isinstance(valeur, int):
            return valeur

    reponse = getattr(exception, "response", None)
    if reponse is not None:
        valeur = getattr(reponse, "status_code", None)
        if isinstance(valeur, int):
            return valeur

    match = re.search(r"\b(429|500|502|503|504)\b", str(exception))
    if match:
        return int(match.group(1))
    return None

# ==========================================
# ÉVALUATION LLM
# ==========================================
PROMPT_TEMPLATE = """Tu es un analyste financier expert.
Lis l'article suivant et évalue le sentiment STRICTEMENT par rapport à l'entreprise : {entreprise}.

Règles de notation :
- 2 (POSITIVE) : L'article annonce une bonne nouvelle spécifique pour l'entreprise (hausse, bon résultat, contrat gagné).
- 1 (NEUTRAL) : Simple mention factuelle, mouvement de marché global sans impact spécifique, ou informations contradictoires.
- 0 (NEGATIVE) : Mauvaise nouvelle pour l'entreprise (baisse, perte, amende, dégradation par un broker).

Fais bien la différence entre le sentiment du marché global et le sentiment lié à {entreprise}.

Réponds UNIQUEMENT avec un objet JSON valide contenant les champs "note_llm" (entier 0, 1 ou 2) et "justification" (string).

Article :
---
{article}
---"""

# v2 : définit NEUTRAL de façon restrictive (pas une classe "par défaut"), impose
# d'isoler d'abord les phrases spécifiques à l'entreprise, et ancre les 3 classes
# avec un exemple few-shot chacune. Objectif : réduire le repli excessif vers NEUTRAL
# et améliorer le rappel NEG/POS observés faibles chez les modèles locaux.
PROMPT_TEMPLATE_V2 = """Tu es un analyste financier expert.

ÉTAPE 1 : Identifie les phrases de l'article qui mentionnent ou concernent directement {entreprise}. Ignore le reste du contexte de marché.

ÉTAPE 2 : Évalue le sentiment de CES PHRASES SPÉCIFIQUEMENT par rapport à {entreprise}, selon ces règles strictes :

- POSITIVE (2) : au moins un fait concret et spécifique à {entreprise} qui est une bonne nouvelle (hausse de résultat, contrat gagné, relèvement d'objectif, note d'analyste améliorée). Un ton globalement positif SANS fait concret ne suffit pas.
- NEGATIVE (0) : au moins un fait concret et spécifique à {entreprise} qui est une mauvaise nouvelle (baisse, perte, amende, dégradation, avertissement sur résultats, procès). Un ton globalement inquiet SANS fait concret ne suffit pas.
- NEUTRAL (1) : UNIQUEMENT si aucun fait concret positif OU négatif n'est identifiable pour {entreprise} — simple citation, mouvement de marché général non spécifique à l'entreprise, ou signaux positifs et négatifs qui s'annulent exactement.

RÈGLE IMPORTANTE : NEUTRAL n'est pas une valeur "par défaut en cas de doute". Si tu hésites entre NEUTRAL et POSITIVE/NEGATIVE, cherche à nouveau un fait concret dans le texte avant de trancher pour NEUTRAL.

Voici 3 exemples de référence :

Exemple A (POSITIVE) : "Renault a annoncé une hausse de 15% de son bénéfice net au T3, dépassant les attentes des analystes." → note_llm=2 (fait concret positif spécifique à Renault)

Exemple B (NEUTRAL) : "Parmi les valeurs du CAC40 en légère baisse ce matin figurent TotalEnergies, Renault et Sanofi, dans un marché globalement attentiste avant les chiffres de l'inflation." → note_llm=1 (mention factuelle sans information spécifique à l'entreprise, mouvement de marché global)

Exemple C (NEGATIVE) : "L'Autorité des marchés financiers a infligé une amende de 2M€ à Renault pour manquement à ses obligations d'information." → note_llm=0 (fait concret négatif spécifique à Renault)

Réponds UNIQUEMENT avec un objet JSON valide contenant "note_llm" (entier 0, 1 ou 2) et "justification" (string, citant le fait concret identifié ou expliquant l'absence de fait concret).

Article :
---
{article}
---"""

# v3 = v2 + un bloc de règles dédié aux notes de brokers/analystes (relèvement ou
# abaissement d'objectif de cours, changement de recommandation). Ces cas étaient
# une source d'ambiguïté récurrente : les règles ci-dessous viennent de cas réels
# annotés REMOVEDellement (cf. modif_prompt.txt).
#
# Logique métier retenue : c'est le NIVEAU DE RECOMMANDATION qui prime sur le
# mouvement de l'objectif de cours. Un objectif abaissé reste NEUTRAL si le broker
# maintient "acheter"/"surpondérer" (il reste au meilleur palier), mais devient
# NEGATIVE s'il n'est qu'à "conserver"/"surperformance" (palier inférieur).
PROMPT_TEMPLATE_V3 = """Tu es un analyste financier expert.

ÉTAPE 1 : Identifie les phrases de l'article qui mentionnent ou concernent directement {entreprise}. Ignore le reste du contexte de marché.

ÉTAPE 2 : Évalue le sentiment de CES PHRASES SPÉCIFIQUEMENT par rapport à {entreprise}, selon ces règles strictes :

- POSITIVE (2) : au moins un fait concret et spécifique à {entreprise} qui est une bonne nouvelle (hausse de résultat, contrat gagné, relèvement d'objectif, note d'analyste améliorée). Un ton globalement positif SANS fait concret ne suffit pas.
- NEGATIVE (0) : au moins un fait concret et spécifique à {entreprise} qui est une mauvaise nouvelle (baisse, perte, amende, dégradation, avertissement sur résultats, procès). Un ton globalement inquiet SANS fait concret ne suffit pas.
- NEUTRAL (1) : UNIQUEMENT si aucun fait concret positif OU négatif n'est identifiable pour {entreprise} — simple citation, mouvement de marché général non spécifique à l'entreprise, ou signaux positifs et négatifs qui s'annulent exactement.

RÈGLE IMPORTANTE : NEUTRAL n'est pas une valeur "par défaut en cas de doute". Si tu hésites entre NEUTRAL et POSITIVE/NEGATIVE, cherche à nouveau un fait concret dans le texte avant de trancher pour NEUTRAL.

RÈGLES SPÉCIFIQUES — objectifs de cours et recommandations de brokers/analystes.
Ces règles s'appliquent EN L'ABSENCE D'AUTRE NOUVELLE sur {entreprise} : si l'article contient par ailleurs un fait concret (résultats, contrat, amende...), applique d'abord les règles générales ci-dessus.

Principe général : note 2 quand l'objectif de cours est relevé ET la recommandation s'améliore ; note 0 quand l'objectif de cours est abaissé ET la recommandation se détériore.
Point clé : c'est le NIVEAU de la recommandation maintenue qui tranche les cas mixtes, pas le seul sens de l'objectif de cours.

- POSITIVE (2) : relèvement de la recommandation, OU rehaussement de l'objectif de cours avec maintien de la recommandation à "acheter" ou "surpondérer".
  Ex: "EssilorLuxottica a fini dans le vert, soutenu par HSBC qui a relevé sa recommandation à Acheter sur le titre." → note_llm=2
  Ex: "Jefferies rehausse sa cible de prix de 141,1 à 144,5 euros et maintient sa recommandation à l'achat." → note_llm=2
  Ex: "Barclays relève significativement son objectif de cours pour TotalEnergies de 78 euros à 94 euros et maintient sa recommandation 'surpondérer'." → note_llm=2

- NEUTRAL (1) : maintien de recommandation ambigu, notamment un objectif de cours abaissé MAIS une recommandation maintenue à l'achat, ou des signaux contradictoires entre conseil et objectif.
  Ex: "Citi abaisse légèrement sa cible de cours pour Carrefour (de 19 à 18 euros) mais maintient sa recommandation à l'achat." → note_llm=1
  Ex: "Berenberg abaisse son conseil à 'conserver' contre 'acheter' et relève son objectif de cours à 24,5 euros contre 17,5 euros." → note_llm=1

- NEGATIVE (0) : abaissement de l'objectif de cours avec maintien de la recommandation à "conserver" ou "surperformance".
  Ex: "Jefferies ajuste son objectif de cours pour Nexans de 140 euros à 136 euros mais maintient sa recommandation à 'conserver'." → note_llm=0
  Ex: "Bernstein reste à surperformance mais abaisse son objectif de cours de 199 euros à 190 euros." → note_llm=0

- NEUTRAL (1) : simple mention factuelle d'un mouvement de marché sectoriel, sans annonce spécifique positive ou négative propre à {entreprise}.
  Ex: "Dans un contexte de regain des tensions géopolitiques, le secteur de l'armement est en forte hausse en Europe, avec 6% pour BAE Systems, 2% pour Rheinmetall ou encore 1,4% pour Dassault Aviation." → note_llm=1

Réponds UNIQUEMENT avec un objet JSON valide contenant "note_llm" (entier 0, 1 ou 2) et "justification" (string, citant le fait concret identifié — y compris l'objectif de cours et le niveau de recommandation le cas échéant — ou expliquant l'absence de fait concret).

Article :
---
{article}
---"""

# Registre des prompts disponibles pour l'A/B testing. Ajouter une entrée ici
# suffit pour qu'une nouvelle version soit sélectionnable via --prompt-version.
PROMPTS = {
    "v1": PROMPT_TEMPLATE,
    "v2": PROMPT_TEMPLATE_V2,
    "v3": PROMPT_TEMPLATE_V3,
}

class EvaluationSentiment(BaseModel):
    note_llm: int
    justification: str


NOTES_VALIDES = {0, 1, 2}


def valider_evaluation(parsed):
    """
    Validation stricte d'une réponse LLM brute (dict issu du JSON) :
    1. Schéma Pydantic (types corrects, champs présents) — lève ValidationError sinon.
    2. Plage métier note_llm ∈ {0,1,2} — lève ValueError sinon.

    Sans cette étape, une réponse malformée (ex: le modèle recopie un pourcentage
    du texte dans note_llm au lieu de trancher 0/1/2) serait insérée telle quelle
    en base et fausserait silencieusement toute analyse de qualité en aval.

    Retourne un EvaluationSentiment validé, ou lève une exception (ValidationError
    ou ValueError) à charge de l'appelant de la traiter comme un échec.
    """
    val = EvaluationSentiment(**parsed)
    if val.note_llm not in NOTES_VALIDES:
        raise ValueError(
            f"note_llm hors plage attendue {sorted(NOTES_VALIDES)}: {val.note_llm!r}"
        )
    return val


class FatalLLMError(Exception):
    """Erreur bloquante de configuration/API distante."""


def extraire_json_depuis_texte(texte):
    if not texte:
        return None

    brut = texte.strip()
    if brut.startswith("```"):
        brut = re.sub(r"^```(?:json)?", "", brut, flags=re.IGNORECASE).strip()
        brut = re.sub(r"```$", "", brut).strip()

    try:
        return json.loads(brut)
    except Exception:
        debut = brut.find("{")
        fin = brut.rfind("}")
        if debut == -1 or fin == -1 or fin <= debut:
            return None
        try:
            return json.loads(brut[debut:fin + 1])
        except Exception:
            return None

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
    """
    template = PROMPTS.get(prompt_version)
    if template is None:
        raise ValueError(
            f"Version de prompt inconnue: {prompt_version!r}. "
            f"Versions disponibles: {', '.join(PROMPTS)}"
        )
    prompt = template.format(entreprise=nom_entreprise, article=texte_article)

    if modele == "gemini":
        for tentative in range(RETRY_MAX_TENTATIVES):
            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=EvaluationSentiment,
                        temperature=0.0,
                    ),
                )
                r = response.parsed
                if r is None:
                    print("Réponse Gemini non exploitable (parsing du schéma a échoué).")
                    return None, "failed"
                if r.note_llm not in NOTES_VALIDES:
                    print(f"Réponse Gemini hors plage attendue: note_llm={r.note_llm!r}")
                    return None, "failed"
                return {"note_llm": r.note_llm, "justification": r.justification}, "ok"
            except Exception as e:
                status_code = extraire_status_code_gemini(e)

                if status_code in (400, 401, 403, 404):
                    raise FatalLLMError(
                        f"Erreur Gemini de configuration/authentification (HTTP {status_code}): {e}"
                    )

                if status_code in STATUS_CODES_RETRYABLES and tentative < RETRY_MAX_TENTATIVES - 1:
                    print(f"Erreur Gemini HTTP {status_code} (transitoire): {e}")
                    attendre_avec_backoff(tentative)
                    continue

                print(f"Erreur Gemini lors de l'évaluation (abandon après {tentative + 1} tentative(s)): {e}")
                return None, "failed"

        return None, "failed"

    elif modele == "haiku":
        if not ANTHROPIC_API_KEY:
            print("ANTHROPIC_API_KEY manquante. Impossible d'appeler Haiku.")
            return None, "failed"

        payload = {
            "model": HAIKU_MODEL,
            "max_tokens": 1024,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "system": "Retourne uniquement un JSON valide avec note_llm (0,1,2) et justification.",
        }
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        for tentative in range(RETRY_MAX_TENTATIVES):
            try:
                response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=60)

                if response.status_code >= 400:
                    detail = ""
                    try:
                        err = response.json()
                        detail = err.get("error", {}).get("message", "") or json.dumps(err)
                    except Exception:
                        detail = response.text[:500]

                    message = (
                        f"HTTP {response.status_code} Anthropic: {detail} "
                        f"(model={HAIKU_MODEL})"
                    )

                    # Erreurs de configuration/authentification/modèle: on stoppe tout de suite.
                    if response.status_code in (400, 401, 403, 404):
                        raise FatalLLMError(message)

                    if response.status_code in STATUS_CODES_RETRYABLES and tentative < RETRY_MAX_TENTATIVES - 1:
                        print(f"Erreur Haiku {message} (transitoire)")
                        attendre_avec_backoff(tentative)
                        continue

                    print(f"Erreur Haiku lors de l'évaluation (abandon après {tentative + 1} tentative(s)): {message}")
                    return None, "failed"

                data = response.json()
                contenu = "".join(
                    bloc.get("text", "")
                    for bloc in data.get("content", [])
                    if bloc.get("type") == "text"
                )
                parsed = extraire_json_depuis_texte(contenu)
                if not parsed:
                    print("Réponse Haiku non exploitable (JSON absent).")
                    return None, "failed"
                try:
                    val = valider_evaluation(parsed)
                except (ValidationError, ValueError) as e:
                    print(f"Réponse Haiku hors schéma: {e}")
                    return None, "failed"
                return {"note_llm": val.note_llm, "justification": val.justification}, "ok"

            except FatalLLMError:
                raise
            except (requests.ConnectionError, requests.Timeout) as e:
                # Réseau instable: retryable comme un 5xx.
                if tentative < RETRY_MAX_TENTATIVES - 1:
                    print(f"Erreur réseau Haiku (transitoire): {e}")
                    attendre_avec_backoff(tentative)
                    continue
                print(f"Erreur réseau Haiku (abandon après {tentative + 1} tentative(s)): {e}")
                return None, "failed"
            except Exception as e:
                print(f"Erreur Haiku lors de l'évaluation: {e}")
                return None, "failed"

        return None, "failed"

    else:
        payload = {
            "model": MODELES_OLLAMA[modele],
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": 8192,
                "num_predict": 3000,
                "temperature": 0.1
            }
        }

        for tentative in range(RETRY_MAX_TENTATIVES):
            try:
                response = requests.post(OLLAMA_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                parsed = json.loads(data["response"])
                val = valider_evaluation(parsed)
                return {"note_llm": val.note_llm, "justification": val.justification}, "ok"

            except (requests.ConnectionError, requests.Timeout) as e:
                # Ollama local temporairement indisponible (ex: modèle en cours de
                # chargement) : retryable.
                if tentative < RETRY_MAX_TENTATIVES - 1:
                    print(f"Erreur réseau Ollama (transitoire): {e}")
                    attendre_avec_backoff(tentative)
                    continue
                print(f"Erreur réseau Ollama (abandon après {tentative + 1} tentative(s)): {e}")
                return None, "failed"

            except (KeyError, json.JSONDecodeError, ValidationError, ValueError) as e:
                # Sortie du modèle malformée ou hors schéma (JSON invalide, champ
                # manquant, note_llm hors {0,1,2}...). Comme temperature > 0, une
                # nouvelle tentative peut suffire à obtenir une sortie conforme.
                if tentative < RETRY_MAX_TENTATIVES - 1:
                    print(f"Réponse Ollama invalide (transitoire, tentative {tentative + 1}): {e}")
                    attendre_avec_backoff(tentative)
                    continue
                print(f"Réponse Ollama invalide après {tentative + 1} tentative(s), abandon: {e}")
                return None, "failed"

            except Exception as e:
                print(f"Erreur Ollama lors de l'évaluation: {e}")
                return None, "failed"

        return None, "failed"

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
    while True:
        choix = input(f"\nChoisissez la version de prompt [{'/'.join(PROMPTS)}] : ").strip().lower()
        if choix in PROMPTS:
            return choix
        print(f"Choix invalide. Entrez l'un de : {', '.join(PROMPTS)}")

# ==========================================
# FONCTION PRINCIPALE
# ==========================================
def main():
    modele = choisir_modele()
    version_prompt = choisir_version_prompt()
    print(f"\nModèle sélectionné : {modele}")

    gemini_client = None
    if modele == "gemini":
        gemini_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(timeout=30000)
        )

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
            
            # Récupérer les articles non encore évalués par le LLM (à adapter selon votre schéma)
            # On suppose qu'il y a une colonne "note_llm" ou qu'on écrit dans une table de résultats
            # longueur_caracteres et longueur_mots sont calculées directement en SQL pour éviter
            # de recharger tout le texte en Python juste pour un comptage, et pour que la métrique
            # soit disponible même si on veut l'utiliser comme critère de filtrage SQL plus tard.
            query_articles = """
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
                LIMIT 100 -- Limite pour tester
            """
            cur.execute(query_articles, (company_id,))
            articles = cur.fetchall()
            
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

                    if statut == "ok" and resultat:
                        note = resultat.get("note_llm")
                        justif = resultat.get("justification", "")

                        print(f"  -> Note: {note} | Justif: {justif[:80]}...")
                        log_article(article_id, company_name, nb_occ, "Évaluation réussie", f"Note={note}", "ok")

                        # 3. Sauvegarder en BDD : note, justification, statut ET version de
                        # prompt utilisée. Attention : le run le plus récent écrase le
                        # précédent en base (colonnes non-cumulatives). Pour comparer v1 vs v2
                        # sans perte, se référer aux fichiers CSV horodatés (un par run) qui,
                        # eux, gardent l'historique complet de chaque version testée.
                        update_query = f'''
                            UPDATE public.article_companies
                            SET {colonne_note} = %s, {colonne_justif} = %s, {colonne_statut} = %s,
                                {colonne_prompt_version} = %s
                            WHERE article_id = %s AND company_id = %s
                        '''
                        cur.execute(update_query, (note, justif, statut, version_prompt, article_id, company_id))
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
                        # Échec définitif après retries : on n'écrit PAS de note (reste NULL en
                        # base), mais on trace le statut 'failed' pour que ces articles soient
                        # exclus de toute analyse de qualité en aval, et repris lors d'un futur run.
                        print(f"  -> Échec de l'évaluation pour l'article {article_id} (statut=failed).")
                        log_article(article_id, company_name, nb_occ, "Évaluation échouée", "Statut=failed", "failed")

                        update_query = f'''
                            UPDATE public.article_companies
                            SET {colonne_statut} = %s, {colonne_prompt_version} = %s
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

if __name__ == "__main__":
    main()