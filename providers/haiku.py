"""
Provider pour Claude Haiku via l'API Anthropic (API distante payante).

evaluer_un() : logique strictement identique à l'ancienne branche
`elif modele == "haiku"` de evaluer_sentiment_llm() dans evaluate_article.py :
même appel REST direct (pas de SDK), mêmes retries, mêmes règles d'erreur
fatale (400/401/403/404).

soumettre_batch()/statut_batch()/recuperer_resultats() : implémentent le mode
batch via la Message Batches API d'Anthropic (-50% de coût, SLO 24h) :
  - POST   {ANTHROPIC_BATCHES_API_URL}          -> crée le batch
  - GET    {ANTHROPIC_BATCHES_API_URL}/{id}      -> statut + results_url une fois "ended"
  - GET    {results_url}                         -> résultats en JSON Lines
Référence : https://docs.claude.com/en/docs/build-with-claude/batch-processing
"""
import json
import os

import requests
from pydantic import ValidationError

from llm_common import (
    FatalLLMError,
    PROMPTS,
    RETRY_MAX_TENTATIVES,
    STATUS_CODES_RETRYABLES,
    attendre_avec_backoff,
    extraire_json_depuis_texte,
    resultat_depuis_evaluation,
    valider_evaluation,
)
from providers.base import BatchLLMProvider

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Identifiant valide sur ce endpoint: alias "claude-haiku-4-5" (resolu cote API).
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-haiku-4-5")
ANTHROPIC_API_URL = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_BATCHES_API_URL = os.getenv(
    "ANTHROPIC_BATCHES_API_URL", "https://api.anthropic.com/v1/messages/batches"
)
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

# custom_id imposé par l'API Anthropic : ^[a-zA-Z0-9_-]{1,64}$
# (voir llm_common.construire_custom_id, qui produit un format compatible)


def _headers():
    return {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _construire_params_message(prompt):
    return {
        "model": HAIKU_MODEL,
        "max_tokens": 1024,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "system": "Retourne uniquement un JSON valide avec note_llm (0,1,2) et justification.",
    }


class HaikuProvider(BatchLLMProvider):
    """Claude Haiku (API distante payante Anthropic), appel synchrone article par article."""

    def evaluer_un(self, texte_article, nom_entreprise, prompt_version="v1"):
        template = PROMPTS.get(prompt_version)
        if template is None:
            raise ValueError(
                f"Version de prompt inconnue: {prompt_version!r}. "
                f"Versions disponibles: {', '.join(PROMPTS)}"
            )
        prompt = template.format(entreprise=nom_entreprise, article=texte_article)

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
                return resultat_depuis_evaluation(val), "ok"

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

    # ==========================================
    # MODE BATCH (Message Batches API)
    # ==========================================
    def soumettre_batch(self, requetes, prompt_version="v1"):
        if not ANTHROPIC_API_KEY:
            raise FatalLLMError("ANTHROPIC_API_KEY manquante. Impossible de soumettre un batch Haiku.")

        template = PROMPTS.get(prompt_version)
        if template is None:
            raise ValueError(
                f"Version de prompt inconnue: {prompt_version!r}. "
                f"Versions disponibles: {', '.join(PROMPTS)}"
            )

        corps_requetes = [
            {
                "custom_id": req["custom_id"],
                "params": _construire_params_message(
                    template.format(entreprise=req["nom_entreprise"], article=req["texte_article"])
                ),
            }
            for req in requetes
        ]

        derniere_erreur = None
        for tentative in range(RETRY_MAX_TENTATIVES):
            try:
                response = requests.post(
                    ANTHROPIC_BATCHES_API_URL,
                    headers=_headers(),
                    json={"requests": corps_requetes},
                    timeout=60,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                derniere_erreur = e
                if tentative < RETRY_MAX_TENTATIVES - 1:
                    print(f"Erreur réseau à la soumission du batch Haiku (transitoire): {e}")
                    attendre_avec_backoff(tentative)
                    continue
                raise FatalLLMError(f"Erreur réseau lors de la soumission du batch Haiku: {e}")

            if response.status_code < 400:
                break

            derniere_erreur = f"HTTP {response.status_code}: {response.text[:500]}"

            # Config/auth : inutile de retenter.
            if response.status_code in (400, 401, 403, 404):
                raise FatalLLMError(
                    f"Erreur lors de la soumission du batch Haiku (HTTP {response.status_code}): "
                    f"{response.text[:500]}"
                )

            # 429 (débit) et 5xx : transitoires, on retente avec backoff.
            if response.status_code in STATUS_CODES_RETRYABLES and tentative < RETRY_MAX_TENTATIVES - 1:
                print(
                    f"Erreur HTTP {response.status_code} à la soumission du batch Haiku "
                    f"(transitoire), nouvelle tentative..."
                )
                attendre_avec_backoff(tentative)
                continue

            raise FatalLLMError(
                f"Erreur lors de la soumission du batch Haiku après {tentative + 1} "
                f"tentative(s) (HTTP {response.status_code}): {response.text[:500]}"
            )
        else:
            raise FatalLLMError(
                f"Soumission du batch Haiku abandonnée après {RETRY_MAX_TENTATIVES} "
                f"tentative(s): {derniere_erreur}"
            )

        batch_id = response.json().get("id")
        if not batch_id:
            raise FatalLLMError(f"Réponse de soumission du batch Haiku sans champ 'id': {response.text[:500]}")
        return batch_id

    def statut_batch(self, batch_id):
        try:
            response = requests.get(f"{ANTHROPIC_BATCHES_API_URL}/{batch_id}", headers=_headers(), timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            # Transitoire par nature (le batch, lui, continue de tourner côté Anthropic) :
            # on ne lève pas FatalLLMError, l'appelant réessaiera au prochain tour de polling.
            print(f"Erreur réseau lors de la vérification du batch Haiku {batch_id} (transitoire): {e}")
            return "in_progress"

        if response.status_code >= 400:
            raise FatalLLMError(
                f"Erreur lors de la vérification du batch Haiku {batch_id} "
                f"(HTTP {response.status_code}): {response.text[:500]}"
            )

        data = response.json()
        # Petit cache pour éviter un second appel identique dans recuperer_resultats().
        self._dernier_batch_info = data
        return "ended" if data.get("processing_status") == "ended" else "in_progress"

    def recuperer_resultats(self, batch_id):
        info = getattr(self, "_dernier_batch_info", None)
        if not info or info.get("id") != batch_id:
            response = requests.get(f"{ANTHROPIC_BATCHES_API_URL}/{batch_id}", headers=_headers(), timeout=30)
            if response.status_code >= 400:
                raise FatalLLMError(
                    f"Erreur lors de la récupération du batch Haiku {batch_id} "
                    f"(HTTP {response.status_code}): {response.text[:500]}"
                )
            info = response.json()

        results_url = info.get("results_url")
        if not results_url:
            raise FatalLLMError(
                f"Batch Haiku {batch_id} marqué 'ended' mais sans results_url "
                f"(processing_status={info.get('processing_status')!r})."
            )

        resultats = {}
        with requests.get(results_url, headers=_headers(), stream=True, timeout=120) as r:
            r.raise_for_status()
            for ligne in r.iter_lines():
                if not ligne:
                    continue
                entree = json.loads(ligne)
                custom_id = entree.get("custom_id")
                result = entree.get("result", {}) or {}
                type_resultat = result.get("type")

                if type_resultat != "succeeded":
                    # errored / canceled / expired : même contrat qu'un échec en mode sync.
                    resultats[custom_id] = (None, "failed")
                    continue

                message = result.get("message", {}) or {}
                contenu = "".join(
                    bloc.get("text", "")
                    for bloc in message.get("content", [])
                    if bloc.get("type") == "text"
                )
                parsed = extraire_json_depuis_texte(contenu)
                if not parsed:
                    resultats[custom_id] = (None, "failed")
                    continue
                try:
                    val = valider_evaluation(parsed)
                    resultats[custom_id] = (resultat_depuis_evaluation(val), "ok")
                except (ValidationError, ValueError):
                    resultats[custom_id] = (None, "failed")

        return resultats
