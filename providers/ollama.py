"""
Provider pour les modèles locaux gratuits servis par Ollama (lama, mistral, queen).

Logique strictement identique à l'ancien bloc `else` de evaluer_sentiment_llm()
dans evaluate_article.py : même payload, mêmes retries, même gestion d'erreurs.
Extrait ici pour séparer clairement les modèles locaux/gratuits des modèles
distants/payants (Gemini, Haiku), qui vivront dans leurs propres modules.
"""
import json
import os

import requests
from pydantic import ValidationError

from llm_common import (
    PROMPTS,
    RETRY_MAX_TENTATIVES,
    attendre_avec_backoff,
    resultat_depuis_evaluation,
    valider_evaluation,
)
from providers.base import LLMProvider

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

# Modèles locaux disponibles : clé interne (utilisée partout dans le script,
# colonnes BDD, CSV, CLI) -> nom du modèle tel qu'attendu par Ollama.
MODELES_OLLAMA = {
    "lama":    "llama3.1:8b",
    "mistral": "mistral-nemo",
    "queen":   "qwen2.5:7b",
}


class OllamaProvider(LLMProvider):
    """Modèle local gratuit servi par Ollama (lama, mistral ou queen)."""

    def __init__(self, modele):
        if modele not in MODELES_OLLAMA:
            raise ValueError(
                f"Modèle Ollama inconnu: {modele!r}. Disponibles: {', '.join(MODELES_OLLAMA)}"
            )
        self.modele = modele
        self.nom_modele_ollama = MODELES_OLLAMA[modele]

    def evaluer_un(self, texte_article, nom_entreprise, prompt_version="v1"):
        template = PROMPTS.get(prompt_version)
        if template is None:
            raise ValueError(
                f"Version de prompt inconnue: {prompt_version!r}. "
                f"Versions disponibles: {', '.join(PROMPTS)}"
            )
        prompt = template.format(entreprise=nom_entreprise, article=texte_article)

        payload = {
            "model": self.nom_modele_ollama,
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
                return resultat_depuis_evaluation(val), "ok"

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
