"""
Recalcule les notes v6 a partir des faits deja extraits en base, sans aucun
appel API.

C'est l'interet principal de la v6 : le LLM a constate des faits (variation de
cours, presence d'un autre fait concret, recommandation), stockes en JSONB dans
extraction_*. Les regles de decision vivent dans le code, pas dans le prompt.
Changer un seuil devient donc un UPDATE, pas 1281 appels API.

Usage :
    python recalculer_notes.py gemini --seuil 10 --dry-run
    python recalculer_notes.py gemini --seuil 10
    python recalculer_notes.py gemini --comparer 3 5 8 10 15
"""
import argparse
import json
import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

from llm_common import note_finale

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "finance_db")

COLONNE_NOTE = {
    "gemini": "note_gemini", "haiku": "note_haiku", "lama": "note_llama3",
    "mistral": "note_mistral", "queen": "note_queen",
}
COLONNE_EXTRACTION = {
    "gemini": "extraction_gemini", "haiku": "extraction_haiku", "lama": "extraction_lama",
    "mistral": "extraction_mistral", "queen": "extraction_queen",
}
COLONNE_STATUT = {
    "gemini": "statut_gemini", "haiku": "statut_haiku", "lama": "statut_lama",
    "mistral": "statut_mistral", "queen": "statut_queen",
}


def charger(cur, modele):
    """Lit les lignes disposant de faits extraits (donc evaluees en v6)."""
    cur.execute(
        f"""
        SELECT article_id, company_id,
               {COLONNE_NOTE[modele]} AS note_actuelle,
               {COLONNE_EXTRACTION[modele]} AS extraction
        FROM public.article_companies
        WHERE {COLONNE_EXTRACTION[modele]} IS NOT NULL
          AND {COLONNE_STATUT[modele]} = 'ok'
        """
    )
    return cur.fetchall()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modele", choices=sorted(COLONNE_NOTE))
    p.add_argument("--seuil", type=float, default=5.0,
                   help="seuil de variation de cours en %% (defaut 5)")
    p.add_argument("--dry-run", action="store_true",
                   help="affiche l'impact sans rien ecrire en base")
    p.add_argument("--comparer", type=float, nargs="+", metavar="SEUIL",
                   help="compare plusieurs seuils sans rien ecrire")
    args = p.parse_args()

    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
                            password=DB_PASSWORD, dbname=DB_NAME)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    lignes = charger(cur, args.modele)

    if not lignes:
        print(f"Aucune ligne avec extraction pour '{args.modele}'.")
        print("Le modele a-t-il ete evalue avec le prompt v6 ?")
        cur.close(); conn.close()
        return 1

    print(f"{len(lignes)} ligne(s) avec faits extraits pour '{args.modele}'.\n")

    # Mode comparaison : balayage de seuils, aucune ecriture.
    if args.comparer:
        print(f"{'Seuil':>7}{'Notes 0':>10}{'Notes 1':>10}{'Notes 2':>10}{'Changees':>10}")
        for seuil in args.comparer:
            distrib = {0: 0, 1: 0, 2: 0}
            changees = 0
            for r in lignes:
                extraction = r["extraction"]
                if isinstance(extraction, str):
                    extraction = json.loads(extraction)
                res = dict(extraction or {})
                res["note_llm"] = r["note_actuelle"]
                note, _ = note_finale(res, seuil_pct=seuil)
                distrib[note] = distrib.get(note, 0) + 1
                if note != r["note_actuelle"]:
                    changees += 1
            print(f"{seuil:>7.1f}{distrib[0]:>10}{distrib[1]:>10}{distrib[2]:>10}{changees:>10}")
        cur.close(); conn.close()
        return 0

    # Mode application
    maj = []
    par_origine = {"regle": 0, "modele": 0}
    for r in lignes:
        extraction = r["extraction"]
        if isinstance(extraction, str):
            extraction = json.loads(extraction)
        res = dict(extraction or {})
        res["note_llm"] = r["note_actuelle"]
        note, origine = note_finale(res, seuil_pct=args.seuil)
        par_origine[origine] += 1
        if note != r["note_actuelle"]:
            maj.append((note, r["article_id"], r["company_id"]))

    print(f"Seuil applique : {args.seuil}%")
    print(f"  notes imposees par la regle : {par_origine['regle']}")
    print(f"  notes laissees au modele    : {par_origine['modele']}")
    print(f"  notes qui CHANGENT          : {len(maj)}")

    if args.dry_run:
        print("\n[--dry-run] Aucune ecriture effectuee.")
    elif maj:
        cur.executemany(
            f"""UPDATE public.article_companies SET {COLONNE_NOTE[args.modele]} = %s
                WHERE article_id = %s AND company_id = %s""",
            maj,
        )
        conn.commit()
        print(f"\n{len(maj)} note(s) mise(s) a jour en base.")
    else:
        print("\nAucun changement a appliquer.")

    cur.close(); conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
