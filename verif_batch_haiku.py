#!/usr/bin/env python3
"""
Vérifier le statut des batchs Haiku (Anthropic).

Contrairement à Gemini qui expose une liste des batchs, Anthropic n'offre pas
de listing centralisé. On récupère donc les batch_id depuis les fichiers
JSON de métadonnées sauvegardés localement.

Statuts Haiku :
  - in_progress : batch tourne encore côté Anthropic
  - ended       : batch terminé, résultats disponibles

Usage :
    python verif_batch_haiku.py                    # statut de tous les batchs locaux
    python verif_batch_haiku.py batches_haiku_v3_*.json  # statut de fichiers spécifiques
    python verif_batch_haiku.py batch_ABC123       # statut d'un batch_id direct
"""

import glob
import json
import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BATCHES_API_URL = os.getenv(
    "ANTHROPIC_BATCHES_API_URL", "https://api.anthropic.com/v1/messages/batches"
)
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")


def _headers():
    return {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def verifier_batch(batch_id):
    """
    Appelle l'API Anthropic pour obtenir le statut d'un batch.
    Retourne (statut, info) où info est le JSON de réponse.
    """
    try:
        response = requests.get(
            f"{ANTHROPIC_BATCHES_API_URL}/{batch_id}",
            headers=_headers(),
            timeout=30,
        )
    except requests.ConnectionError as e:
        return "unknown", {"error": f"Connexion échouée: {e}"}
    except requests.Timeout as e:
        return "unknown", {"error": f"Timeout: {e}"}

    if response.status_code >= 400:
        return "unknown", {
            "error": f"HTTP {response.status_code}: {response.text[:200]}"
        }

    info = response.json()
    statut = info.get("processing_status", "unknown")
    return statut, info


def charger_batch_ids_depuis_fichiers(patterns):
    """
    Scanne les fichiers JSON de métadonnées et extrait les batch_id.
    """
    batch_ids = []
    fichiers = []

    for pattern in patterns:
        fichiers.extend(glob.glob(pattern))

    if not fichiers:
        return []

    for fichier in fichiers:
        try:
            with open(fichier, "r", encoding="utf-8") as f:
                data = json.load(f)
            for batch_meta in data.get("batches", []):
                if batch_meta.get("statut") == "soumis":
                    batch_id = batch_meta.get("batch_id")
                    if batch_id:
                        batch_ids.append((batch_id, fichier))
        except Exception as e:
            print(f"Erreur lecture {fichier}: {e}")

    return batch_ids


def main():
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY introuvable. Vérifiez le fichier .env")
        print("(même variable que celle utilisée par evaluate_article.py).")
        return 1

    print(
        f"Clé API chargée : {ANTHROPIC_API_KEY[:6]}...{ANTHROPIC_API_KEY[-4:]} "
        f"({len(ANTHROPIC_API_KEY)} caractères)\n"
    )

    # Déterminer quels batch_id vérifier
    batch_ids_et_sources = []

    if len(sys.argv) > 1:
        # Modes :
        # 1. python verif_batch_haiku.py batch_ABC123  -> batch_id direct
        # 2. python verif_batch_haiku.py batches_haiku_v3_*.json -> patterns fichiers
        args = sys.argv[1:]

        for arg in args:
            if arg.startswith("batch_"):
                # Batch ID direct
                batch_ids_et_sources.append((arg, "CLI argument"))
            else:
                # Pattern fichier (peut inclure *)
                batch_ids_et_sources.extend(charger_batch_ids_depuis_fichiers([arg]))
    else:
        # Défaut : scanner tous les fichiers batches_haiku_*.json
        batch_ids_et_sources.extend(
            charger_batch_ids_depuis_fichiers(["output/batches_haiku_*.json"])
        )

    if not batch_ids_et_sources:
        print("Aucun batch trouvé.")
        print("\nUtilisation :")
        print("  python verif_batch_haiku.py                        # tous les batchs locaux")
        print("  python verif_batch_haiku.py batches_haiku_v3_*.json  # fichiers spécifiques")
        print("  python verif_batch_haiku.py batch_ABC123           # batch_id direct")
        return 0

    print(f"{len(batch_ids_et_sources)} batch(s) trouvé(s) :\n")

    actifs = []
    termines = []

    for batch_id, source in batch_ids_et_sources:
        statut, info = verifier_batch(batch_id)

        # Affichage
        if statut == "in_progress":
            marqueur = "→ "
            actifs.append((batch_id, source, info))
        else:
            marqueur = "  "
            termines.append((batch_id, source, info, statut))

        # Infos supplémentaires
        created_at = info.get("created_at", "")
        processing_status = info.get("processing_status", "unknown")
        request_counts = info.get("request_counts", {})
        succeeded = request_counts.get("succeeded", 0)
        errored = request_counts.get("errored", 0)

        print(f"{marqueur}{processing_status:<12} {batch_id}")
        print(f"   Créé : {created_at}")
        print(f"   Source : {source}")

        if succeeded > 0 or errored > 0:
            print(f"   Requêtes : {succeeded} succès, {errored} erreurs")

        if info.get("error"):
            print(f"   ⚠️ Erreur API: {info['error']}")

        print()

    # Résumé
    if actifs:
        print(f"{'=' * 60}")
        print(f"{len(actifs)} batch(s) ENCORE EN COURS (in_progress) :")
        print()
        for batch_id, source, info in actifs:
            request_counts = info.get("request_counts", {})
            succeeded = request_counts.get("succeeded", 0)
            processing = request_counts.get("processing", 0)
            errored = request_counts.get("errored", 0)
            print(
                f"  • {batch_id} (source: {source})"
            )
            print(
                f"    Progression: {succeeded} succès | {processing} en cours | {errored} erreurs"
            )
        print()
        print("Attendez quelques heures ou relancez ce script plus tard.")

    if termines:
        print(f"{'=' * 60}")
        print(f"{len(termines)} batch(s) TERMINÉ(S) :")
        print()
        for batch_id, source, info, statut in termines:
            request_counts = info.get("request_counts", {})
            succeeded = request_counts.get("succeeded", 0)
            errored = request_counts.get("errored", 0)
            print(
                f"  • {batch_id} ({statut})"
            )
            print(
                f"    Résultats: {succeeded} succès | {errored} erreurs"
            )
            if succeeded > 0:
                results_url = info.get("results_url")
                if results_url:
                    print(f"    Results URL : {results_url}")

        print()
        print("Pour récupérer les résultats, lancez :")
        print("  python evaluate_article.py --collect output/batches_haiku_v3_YYYYMMDD_HHMMSS.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
