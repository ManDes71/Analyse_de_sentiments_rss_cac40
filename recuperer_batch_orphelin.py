"""
Reconstruction d'un fichier de métadonnées pour des batchs orphelins.

Contexte : si la soumission échoue au milieu de la boucle de chunks (ex: quota
429 au chunk 3), l'ancienne version d'evaluate_article.py sortait AVANT d'écrire
le fichier de métadonnées. Les chunks déjà soumis — et facturés — tournaient
alors côté fournisseur sans qu'aucun fichier ne relie leurs batch_id aux
articles, rendant `--collect` impossible.

Ce script rejoue cette liaison :
  1. lit les custom_id depuis le fichier de RESULTATS de chaque batch
     (job.dest.file_name), chaque ligne portant sa cle "key" ;
  2. decode chaque custom_id en (article_id, company_id) ;
  3. relit les articles correspondants en base pour reconstituer les champs
     attendus par recuperer_batch_run() ;
  4. ecrit un fichier de metadonnees exploitable par :
         python evaluate_article.py --collect <fichier>

Un batch doit etre en JOB_STATE_SUCCEEDED pour etre recuperable : les batchs
encore en cours sont ignores et listes pour une reprise ulterieure.

Usage :
    python recuperer_batch_orphelin.py v4 batches/xxxx batches/yyyy
    python recuperer_batch_orphelin.py v4 batches/xxxx --debug
"""
import json
import os
import sys
from datetime import datetime

import psycopg2
from dotenv import load_dotenv
from google import genai
from psycopg2.extras import RealDictCursor

from llm_common import parser_custom_id

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "finance_db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def lire_custom_ids_du_batch(client, batch_id, debug=False):
    """
    Récupère la liste des custom_id d'un batch.

    Le JSONL d'entrée local a été supprimé par le finally de soumettre_batch, et
    l'attribut `src` du job n'expose pas le fichier source dans toutes les
    versions du SDK. On lit donc les custom_id depuis le fichier de RÉSULTATS
    (job.dest.file_name), où chaque ligne porte sa clé "key".

    Conséquence : un batch doit être TERMINÉ (JOB_STATE_SUCCEEDED) pour être
    récupérable par ce script. Un batch encore en cours devra être repris plus tard.
    """
    job = client.batches.get(name=batch_id)
    etat = job.state.name

    if debug:
        print(f"  [debug] attributs du job : "
              f"{[a for a in dir(job) if not a.startswith('_')]}")
        print(f"  [debug] src={getattr(job, 'src', None)!r}")
        print(f"  [debug] dest={getattr(job, 'dest', None)!r}")

    if etat != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Batch encore en état {etat} : les résultats ne sont pas disponibles. "
            f"Relancez ce script quand il sera passé à JOB_STATE_SUCCEEDED."
        )

    dest = getattr(job, "dest", None)
    nom_fichier = getattr(dest, "file_name", None) if dest is not None else None
    if not nom_fichier:
        raise RuntimeError(
            f"Batch {batch_id} terminé mais sans fichier de résultats exploitable "
            f"(dest={dest!r}). Relancez avec --debug pour inspecter le job."
        )

    contenu = client.files.download(file=nom_fichier)
    if isinstance(contenu, bytes):
        contenu = contenu.decode("utf-8")

    custom_ids = []
    for ligne in contenu.splitlines():
        if not ligne.strip():
            continue
        cle = json.loads(ligne).get("key")
        if cle:
            custom_ids.append(cle)
    return custom_ids, etat


def reconstruire_requetes(cur, custom_ids):
    """
    Reconstitue, pour chaque custom_id, la structure attendue par
    recuperer_batch_run() : mêmes clés que celles produites en phase 1.
    """
    requetes = {}
    manquants = []
    for custom_id in custom_ids:
        article_id, company_id = parser_custom_id(custom_id)
        cur.execute(
            """
            SELECT a.id,
                   a.contenu,
                   LENGTH(a.contenu) AS longueur_caracteres,
                   array_length(regexp_split_to_array(trim(a.contenu), '\\s+'), 1) AS longueur_mots,
                   c.name AS company_name
            FROM public.articles_rss a
            JOIN public.companies c ON c.id = %s
            WHERE a.id = %s
            """,
            (company_id, article_id),
        )
        row = cur.fetchone()
        if not row:
            manquants.append(custom_id)
            continue
        requetes[custom_id] = {
            "custom_id": custom_id,
            "article_id": article_id,
            "company_id": company_id,
            "company_name": row["company_name"],
            "texte": row["contenu"],
            "longueur_caracteres": row["longueur_caracteres"],
            "longueur_mots": row["longueur_mots"],
        }
    return requetes, manquants


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("Erreur : indiquez la version de prompt puis au moins un batch_id.")
        return 1

    version_prompt = sys.argv[1]
    debug = "--debug" in sys.argv
    batch_ids = [a for a in sys.argv[2:] if not a.startswith("--")]

    if not batch_ids:
        print("Erreur : aucun batch_id fourni.")
        return 1

    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY introuvable (vérifiez .env).")
        return 1

    client = genai.Client(api_key=GOOGLE_API_KEY)
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    batches_meta = []
    toutes_requetes = {}
    a_reprendre = []

    for batch_id in batch_ids:
        print(f"\nLecture du batch {batch_id}...")
        try:
            custom_ids, etat = lire_custom_ids_du_batch(client, batch_id, debug=debug)
        except RuntimeError as e:
            print(f"  IGNORÉ : {e}")
            a_reprendre.append(batch_id)
            continue
        except Exception as e:
            print(f"  ÉCHEC : {e}")
            a_reprendre.append(batch_id)
            continue

        print(f"  état={etat}, {len(custom_ids)} requête(s)")
        requetes, manquants = reconstruire_requetes(cur, custom_ids)
        if manquants:
            print(f"  ATTENTION : {len(manquants)} custom_id sans article correspondant en base "
                  f"(ignorés) : {manquants[:5]}...")
        toutes_requetes.update(requetes)
        batches_meta.append({
            "batch_id": batch_id,
            "custom_ids": custom_ids,
            "statut": "soumis",
        })

    cur.close()
    conn.close()

    if a_reprendre:
        print(f"\n{len(a_reprendre)} batch(s) non récupérable(s) pour l'instant :")
        for bid in a_reprendre:
            print(f"  {bid}")
        print("Relancez ce script sur ces batch_id une fois qu'ils seront terminés.")

    if not batches_meta:
        print("\nAucun batch exploitable, rien à écrire.")
        return 1

    date_jour = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    chemin = f"output/batches_gemini_{version_prompt}_{date_jour}_RECUPERE.json"
    nom_csv = f"output/resultats_gemini_{version_prompt}_{date_jour}_RECUPERE.csv"
    nom_log = f"output/log_gemini_{version_prompt}_{date_jour}_RECUPERE.txt"

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({
            "modele": "gemini",
            "prompt_version": version_prompt,
            "batches": batches_meta,
            "requetes": toutes_requetes,
            "non_soumis": [],
            "csv": nom_csv,
            "log": nom_log,
        }, f, ensure_ascii=False)

    print(f"\nMétadonnées reconstruites : {chemin}")
    print(f"{len(toutes_requetes)} requête(s) sur {len(batches_meta)} batch(s).")
    print("\nRécupérez maintenant les résultats avec :")
    print(f"  python evaluate_article.py --collect {chemin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())