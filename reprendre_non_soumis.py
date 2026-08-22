"""
Remet a 'not_evaluated' les requetes enregistrees comme NON SOUMISES dans un
fichier de metadonnees de batch.

Contexte : quand un chunk echoue (quota 429), les chunks deja soumis sont
sauvegardes et les requetes restantes sont listees dans le champ "non_soumis"
du fichier de metadonnees. Mais en base, ces lignes portent encore le statut
d'un run ANTERIEUR (souvent 'ok' avec la meme version de prompt), donc le filtre
de reprise les considere comme deja faites.

Sans ce script, il n'y a que deux issues, toutes deux mauvaises :
  - relancer normalement  -> 0 article a evaluer, on croit a tort avoir fini ;
  - relancer REPRENDRE=0  -> tout est resoumis, y compris ce qui vient d'etre paye.

Ce script remet uniquement les lignes non soumises a 'not_evaluated', pour qu'un
run normal (REPRENDRE=1) reprenne exactement celles-la.

Usage :
    python reprendre_non_soumis.py output/batches_gemini_v4_20260814_213748.json
    python reprendre_non_soumis.py <fichier.json> --dry-run
"""
import json
import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

from llm_common import parser_custom_id

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "finance_db")

# Doit rester aligne sur COLONNE_STATUT / COLONNE_PROMPT_VERSION d'evaluate_article.py
COLONNE_STATUT = {
    "gemini": "statut_gemini",
    "haiku": "statut_haiku",
    "lama": "statut_lama",
    "mistral": "statut_mistral",
    "queen": "statut_queen",
}
COLONNE_NOTE = {
    "gemini": "note_gemini",
    "haiku": "note_haiku",
    "lama": "note_llama3",
    "mistral": "note_mistral",
    "queen": "note_queen",
}
COLONNE_JUSTIF = {
    "gemini": "justification_gemini",
    "haiku": "justification_haiku",
    "lama": "justification_lama",
    "mistral": "justification_mistral",
    "queen": "justification_queen",
}
COLONNE_PROMPT_VERSION = {
    "gemini": "prompt_version_gemini",
    "haiku": "prompt_version_haiku",
    "lama": "prompt_version_lama",
    "mistral": "prompt_version_mistral",
    "queen": "prompt_version_queen",
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    chemin_meta = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    if not os.path.exists(chemin_meta):
        print(f"Fichier introuvable : {chemin_meta}")
        return 1

    with open(chemin_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    modele = meta.get("modele")
    version = meta.get("prompt_version")
    non_soumis = meta.get("non_soumis") or []

    if modele not in COLONNE_STATUT:
        print(f"Modele inconnu dans les metadonnees : {modele!r}")
        return 1

    print(f"Modele : {modele} | version de prompt : {version}")
    print(f"Requetes non soumises enregistrees : {len(non_soumis)}")

    if not non_soumis:
        print("\nAucune requete non soumise : rien a faire.")
        print("(Si le run a ete interrompu avec une version anterieure du script,")
        print(" le champ 'non_soumis' peut etre absent du fichier.)")
        return 0

    if dry_run:
        print("\n[--dry-run] Aucune modification effectuee. Exemples :")
        for cid in non_soumis[:5]:
            article_id, company_id = parser_custom_id(cid)
            print(f"  {cid} -> article_id={article_id}, company_id={company_id}")
        return 0

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    colonne_statut = COLONNE_STATUT[modele]
    colonne_note = COLONNE_NOTE[modele]
    colonne_justif = COLONNE_JUSTIF[modele]
    colonne_version = COLONNE_PROMPT_VERSION[modele]

    # On remet note, justification et version a NULL en plus du statut : ces
    # valeurs proviennent d'un run anterieur et ne correspondent pas au prompt
    # courant. Les laisser en place recreerait exactement le probleme de notes
    # orphelines mal etiquetees.
    requete = f"""
        UPDATE public.article_companies
        SET {colonne_statut} = 'not_evaluated',
            {colonne_note} = NULL,
            {colonne_justif} = NULL,
            {colonne_version} = NULL
        WHERE article_id = %s AND company_id = %s
    """

    modifiees = 0
    for cid in non_soumis:
        try:
            article_id, company_id = parser_custom_id(cid)
        except ValueError as e:
            print(f"  custom_id ignore ({e})")
            continue
        cur.execute(requete, (article_id, company_id))
        modifiees += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n{modifiees} ligne(s) remise(s) a 'not_evaluated'.")
    print("\nRelancez maintenant un run NORMAL (sans REPRENDRE=0) :")
    print(f"  PROMPT_VERSION={version} python evaluate_article.py")
    print(f"Il devrait annoncer environ {len(non_soumis)} article(s) a evaluer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())