"""
Provider pour Google Gemini (API distante payante).

evaluer_un() : logique strictement identique à l'ancienne branche
`if modele == "gemini"` de evaluer_sentiment_llm() dans evaluate_article.py :
mêmes retries, mêmes règles d'erreur fatale (400/401/403/404), même
construction du client.

soumettre_batch()/statut_batch()/recuperer_resultats() : implémentent le mode
batch via la Batch API de Gemini (-50% de coût, SLO 24h), en mode fichier
(recommandé par Google pour les batchs de taille non-triviale, cf. doc) :
  1. écrit les requêtes en JSONL avec une clé ("key") = notre custom_id
  2. client.files.upload() le fichier
  3. client.batches.create(src=fichier.name) crée le job
  4. client.batches.get(name=...) pour le statut (JOB_STATE_*)
  5. client.files.download() sur le fichier de résultats une fois terminé
Référence : https://ai.google.dev/gemini-api/docs/batch-mode

Note : le format exact des lignes de résultat (clé "key" + "response"/"candidates"
ou "status"/erreur) est documenté par Google mais n'a pas pu être vérifié ici sur
un vrai run ; le parsing ci-dessous est défensif (plusieurs formes acceptées) —
à valider avec une clé API réelle avant mise en prod.
"""
import json
import os
import tempfile
import uuid

from google import genai
from google.genai import types
from pydantic import ValidationError

from llm_common import (
    EvaluationSentiment,
    FatalLLMError,
    NOTES_VALIDES,
    PROMPTS,
    RETRY_MAX_TENTATIVES,
    STATUS_CODES_RETRYABLES,
    attendre_avec_backoff,
    extraire_json_depuis_texte,
    extraire_status_code_gemini,
    resultat_depuis_evaluation,
    valider_evaluation,
)
from providers.base import BatchLLMProvider

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_ETATS_ECHEC_JOB = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

# Schéma de sortie contraint, en mode BATCH.
#
# En mode sync, on passe directement le modèle Pydantic
# (response_schema=EvaluationSentiment) : le SDK le convertit lui-même. Le mode
# batch écrit du JSONL envoyé tel quel à l'API REST, qui attend le schéma au
# format OpenAPI de Google (types en MAJUSCULES). On ne peut donc pas réutiliser
# EvaluationSentiment.model_json_schema() : celui-ci produit des types en
# minuscules et des clés "title" non reconnues.
#
# Sans ce schéma, le batch n'imposait que "réponds en JSON valide", alors que le
# mode sync impose en plus la structure exacte : les deux modes n'avaient pas les
# mêmes garanties de format, ce qui fausse toute comparaison entre un run sync et
# un run batch. Doit rester synchronisé avec EvaluationSentiment (llm_common.py).
# Les champs d'extraction (v6) sont declares mais NON requis : les versions v1 a
# v5 ne les demandent pas, et le meme schema sert a toutes. "nullable" autorise
# le null explicite attendu quand l'article ne contient pas l'information.
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "note_llm": {"type": "INTEGER"},
        "justification": {"type": "STRING"},
        "variation_cours_pct": {"type": "NUMBER", "nullable": True},
        "autre_fait_concret": {"type": "BOOLEAN", "nullable": True},
        "reco_sens": {"type": "STRING", "nullable": True},
        "reco_niveau": {"type": "STRING", "nullable": True},
    },
    "required": ["note_llm", "justification"],
    "propertyOrdering": [
        "note_llm", "justification", "variation_cours_pct",
        "autre_fait_concret", "reco_sens", "reco_niveau",
    ],
}


def construire_client_gemini():
    """Construit le client Gemini partagé (créé une seule fois par run dans main())."""
    return genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options=types.HttpOptions(timeout=30000),
    )


class GeminiProvider(BatchLLMProvider):
    """Google Gemini (API distante payante), appel synchrone article par article."""

    def __init__(self, client=None):
        # Un client peut être injecté (recommandé : un seul client réutilisé pour
        # tout le run, cf. main()) ; sinon on en construit un nouveau à la volée.
        self.client = client or construire_client_gemini()

    def evaluer_un(self, texte_article, nom_entreprise, prompt_version="v1"):
        template = PROMPTS.get(prompt_version)
        if template is None:
            raise ValueError(
                f"Version de prompt inconnue: {prompt_version!r}. "
                f"Versions disponibles: {', '.join(PROMPTS)}"
            )
        prompt = template.format(entreprise=nom_entreprise, article=texte_article)

        for tentative in range(RETRY_MAX_TENTATIVES):
            try:
                response = self.client.models.generate_content(
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
                return resultat_depuis_evaluation(r), "ok"
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

    # ==========================================
    # MODE BATCH (Batch API, fichier JSONL)
    # ==========================================
    def soumettre_batch(self, requetes, prompt_version="v1"):
        template = PROMPTS.get(prompt_version)
        if template is None:
            raise ValueError(
                f"Version de prompt inconnue: {prompt_version!r}. "
                f"Versions disponibles: {', '.join(PROMPTS)}"
            )

        chemin_jsonl = os.path.join(
            tempfile.gettempdir(), f"batch_gemini_{uuid.uuid4().hex}.jsonl"
        )
        try:
            with open(chemin_jsonl, "w", encoding="utf-8") as f:
                for req in requetes:
                    prompt = template.format(entreprise=req["nom_entreprise"], article=req["texte_article"])
                    ligne = {
                        "key": req["custom_id"],
                        "request": {
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generation_config": {
                                "temperature": 0.0,
                                "response_mime_type": "application/json",
                                # Aligne le batch sur le mode sync : structure imposée,
                                # pas seulement "du JSON valide".
                                "response_schema": GEMINI_RESPONSE_SCHEMA,
                            },
                        },
                    }
                    f.write(json.dumps(ligne) + "\n")

            derniere_erreur = None
            for tentative in range(RETRY_MAX_TENTATIVES):
                try:
                    fichier_televerse = self.client.files.upload(
                        file=chemin_jsonl,
                        config=types.UploadFileConfig(
                            display_name=f"evaluate-article-batch-{uuid.uuid4().hex[:8]}",
                            mime_type="jsonl",
                        ),
                    )
                    batch_job = self.client.batches.create(
                        model=GEMINI_MODEL,
                        src=fichier_televerse.name,
                        config={"display_name": f"evaluate-article-batch-{uuid.uuid4().hex[:8]}"},
                    )
                    break
                except Exception as e:
                    status_code = extraire_status_code_gemini(e)
                    derniere_erreur = e

                    if status_code in (400, 401, 403, 404):
                        raise FatalLLMError(
                            f"Erreur Gemini de configuration/authentification lors de la "
                            f"soumission du batch (HTTP {status_code}): {e}"
                        )

                    # 429 (quota/débit) et 5xx sont transitoires : on retente avec backoff,
                    # comme le fait déjà evaluer_un(). Sans ça, un simple pic de débit fait
                    # échouer tout le run alors qu'aucun batch n'a été créé.
                    if status_code in STATUS_CODES_RETRYABLES and tentative < RETRY_MAX_TENTATIVES - 1:
                        print(f"Erreur Gemini HTTP {status_code} à la soumission du batch (transitoire): {e}")
                        attendre_avec_backoff(tentative)
                        continue

                    if status_code == 429:
                        raise FatalLLMError(
                            f"Quota Gemini épuisé après {tentative + 1} tentative(s) (HTTP 429). "
                            f"S'il s'agit d'un quota JOURNALIER, réessayer plus tard ne suffira pas : "
                            f"vérifier le plan et la facturation sur https://ai.dev/rate-limit — "
                            f"détail: {e}"
                        )
                    raise FatalLLMError(f"Erreur lors de la soumission du batch Gemini: {e}")
            else:
                raise FatalLLMError(
                    f"Soumission du batch Gemini abandonnée après {RETRY_MAX_TENTATIVES} "
                    f"tentative(s): {derniere_erreur}"
                )
        finally:
            if os.path.exists(chemin_jsonl):
                os.remove(chemin_jsonl)

        return batch_job.name

    def statut_batch(self, batch_id):
        try:
            job = self.client.batches.get(name=batch_id)
        except Exception as e:
            print(f"Erreur réseau/transitoire lors de la vérification du batch Gemini {batch_id}: {e}")
            return "in_progress"

        etat = job.state.name
        if etat == "JOB_STATE_SUCCEEDED":
            return "ended"
        if etat in _ETATS_ECHEC_JOB:
            raise FatalLLMError(
                f"Batch Gemini {batch_id} terminé en échec (état={etat}): {getattr(job, 'error', None)}"
            )
        return "in_progress"

    def recuperer_resultats(self, batch_id):
        job = self.client.batches.get(name=batch_id)
        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise FatalLLMError(
                f"recuperer_resultats appelé alors que le batch Gemini {batch_id} "
                f"n'est pas terminé avec succès (état={job.state.name})."
            )

        lignes = self._lire_lignes_resultats(job)

        resultats = {}
        for ligne in lignes:
            if not ligne.strip():
                continue
            entree = json.loads(ligne)
            custom_id = entree.get("key")
            resultats[custom_id] = self._parser_ligne_resultat(entree)
        return resultats

    def _lire_lignes_resultats(self, job):
        dest = getattr(job, "dest", None)
        if dest is not None and getattr(dest, "file_name", None):
            contenu = self.client.files.download(file=dest.file_name)
            if isinstance(contenu, bytes):
                contenu = contenu.decode("utf-8")
            return contenu.splitlines()

        if dest is not None and getattr(dest, "inlined_responses", None) is not None:
            raise FatalLLMError(
                f"Batch Gemini {job.name} soumis en mode fichier mais réponse en mode "
                f"inline inattendue : format de résultat non géré par ce provider."
            )

        raise FatalLLMError(f"Batch Gemini {job.name} terminé mais sans résultats exploitables (dest vide).")

    @staticmethod
    def _parser_ligne_resultat(entree):
        """
        Parsing défensif d'une ligne de résultat (voir note en tête de fichier :
        format non vérifié sur un run réel). Reconnaît une erreur explicite
        ('status'/'error') ou une réponse standard avec 'candidates'.

        La validation passe par valider_evaluation() — la même fonction que les
        modes sync de tous les providers — et non par un contrôle manuel. Un
        contrôle du type `note not in NOTES_VALIDES` laisserait passer un flottant
        (2.0 in {0,1,2} vaut True en Python) et l'insérerait tel quel en base.
        """
        if entree.get("status") or entree.get("error"):
            return None, "failed"

        reponse = entree.get("response", entree)
        candidats = reponse.get("candidates") or []
        if not candidats:
            return None, "failed"

        parties = (candidats[0].get("content") or {}).get("parts") or []
        contenu = "".join(p.get("text", "") for p in parties)
        parsed = extraire_json_depuis_texte(contenu)
        if not parsed:
            return None, "failed"

        try:
            val = valider_evaluation(parsed)
        except (ValidationError, ValueError, TypeError, AttributeError):
            return None, "failed"
        return resultat_depuis_evaluation(val), "ok"
