#!/usr/bin/env python3
"""
Historise les notes des LLM : une ligne par (couple, modèle, version de prompt).

Pourquoi : `article_companies` n'a qu'une colonne de note par modèle, et
`evaluate_article.py` l'écrase à chaque run. `prompt_version_*` ne garde pas
l'historique — elle dit seulement quelle version a produit la note présente.
Lancer le v8 le 20/09/2026 a donc effacé les notes v7 des 663 couples du
périmètre, qui n'ont survécu que dans les CSV horodatés. Comparer deux versions
supposait alors de relire des fichiers ; avec cette table, c'est une requête.

`article_companies` garde la note COURANTE (rien ne change pour le reste du
code) ; cette table garde TOUT. Clé : (article_id, company_id, modele,
prompt_version) — relancer la même version écrase sa propre ligne, ce qui borne
la croissance en production tout en gardant une ligne par version.

Usage :
    cd tfidf
    .venv/bin/python historique_notes.py --creer                  # crée la table
    .venv/bin/python historique_notes.py --amorcer                # test à blanc
    .venv/bin/python historique_notes.py --amorcer --appliquer    # remplit
    .venv/bin/python historique_notes.py --etat                   # ce qu'elle contient

L'amorçage lit deux sources :
  - l'état courant d'`article_companies` (source 'bdd') ;
  - tous les CSV output/resultats_*.csv (source 'csv'), qui portent les runs
    passés, avec la date réelle du run tirée du nom de fichier.
En cas de doublon, la ligne la plus récente gagne.
"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

from psycopg2.extras import execute_values

# `connexion` vit dans cluster_genres, qui tire scikit-learn et hdbscan :
# import tardif, pour qu'evaluate_article.py ne paie pas ce chargement à chaque
# run alors qu'il n'a besoin que d'enregistrer().

SQL_TABLE = """
CREATE TABLE IF NOT EXISTS public.article_company_notes (
    article_id     integer     NOT NULL,
    company_id     integer     NOT NULL,
    modele         text        NOT NULL,
    prompt_version text        NOT NULL,
    note           smallint,
    justification  text,
    statut         text        NOT NULL,
    extraction     jsonb,
    source         text        NOT NULL DEFAULT 'run',
    evalue_le      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, company_id, modele, prompt_version),
    FOREIGN KEY (article_id, company_id)
        REFERENCES public.article_companies (article_id, company_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_acn_modele_version
    ON public.article_company_notes (modele, prompt_version);
"""

# Colonnes d'article_companies, par modèle : note, justification, statut,
# version de prompt, extraction.
MODELES = {
    "gemini":  ("note_gemini",  "justification_gemini",  "statut_gemini",  "prompt_version_gemini",  "extraction_gemini"),
    "haiku":   ("note_haiku",   "justification_haiku",   "statut_haiku",   "prompt_version_haiku",   "extraction_haiku"),
    "lama":    ("note_llama3",  "justification_lama",    "statut_lama",    "prompt_version_lama",    "extraction_lama"),
    "mistral": ("note_mistral", "justification_mistral", "statut_mistral", "prompt_version_mistral", "extraction_mistral"),
    "queen":   ("note_queen",   "justification_queen",   "statut_queen",   "prompt_version_queen",   "extraction_queen"),
}

# La ligne la plus récente gagne : un amorçage ne doit jamais écraser un run
# plus récent, et rejouer l'amorçage doit être sans effet.
SQL_INSERT = """
INSERT INTO public.article_company_notes
    (article_id, company_id, modele, prompt_version, note, justification, statut,
     extraction, source, evalue_le)
VALUES %s
ON CONFLICT (article_id, company_id, modele, prompt_version) DO UPDATE
   SET note          = EXCLUDED.note,
       justification = EXCLUDED.justification,
       statut        = EXCLUDED.statut,
       extraction    = EXCLUDED.extraction,
       source        = EXCLUDED.source,
       evalue_le     = EXCLUDED.evalue_le
 WHERE EXCLUDED.evalue_le >= public.article_company_notes.evalue_le
"""


def enregistrer(cur, modele, prompt_version, article_id, company_id,
                note, justification, statut, extraction) -> None:
    """Ajoute une évaluation à l'historique. Appelé par evaluate_article.py.

    Silencieux si la table n'existe pas encore : l'historisation ne doit jamais
    faire échouer une évaluation déjà payée.
    """
    try:
        cur.execute(
            SQL_INSERT.replace("VALUES %s", "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())"),
            (article_id, company_id, modele, prompt_version, note, justification,
             statut, extraction, "run"),
        )
    except Exception as e:                                    # pragma: no cover
        print(f"  (historisation ignorée : {e})")


def _horodatage(chemin: str) -> datetime | None:
    # Le suffixe facultatif couvre les fichiers _RECUPERE.csv des batchs orphelins.
    m = re.search(r"_(\d{8}_\d{6})(?:_[A-Z]+)?\.csv$", chemin)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


def _dedoublonner(lignes: list[tuple]) -> list[tuple]:
    """Une seule ligne par clé, la plus récente.

    PostgreSQL refuse qu'un même INSERT ... ON CONFLICT touche deux fois la même
    ligne : il faut donc trancher ici. Le cas est fréquent, une même version de
    prompt ayant souvent été rejouée dans plusieurs runs.
    """
    par_cle = {}
    for l in lignes:
        cle = l[:4]                       # article_id, company_id, modele, version
        if cle not in par_cle or l[-1] >= par_cle[cle][-1]:
            par_cle[cle] = l
    return list(par_cle.values())


def _modele_du_fichier(chemin: str) -> str | None:
    m = re.match(r"resultats_([a-z0-9]+)_", os.path.basename(chemin))
    return m.group(1) if m and m.group(1) in MODELES else None


def lignes_bdd(cur) -> list[tuple]:
    """État courant d'article_companies, pour ne rien perdre de ce qui est en base."""
    lignes = []
    for modele, (note, justif, statut, version, extraction) in MODELES.items():
        cur.execute(f"""
            SELECT article_id, company_id, {note} AS note, {justif} AS justification,
                   {statut} AS statut, {version} AS version, {extraction} AS extraction
              FROM public.article_companies
             WHERE {version} IS NOT NULL AND {statut} IN ('ok', 'failed')
        """)
        for r in cur.fetchall():
            lignes.append((r["article_id"], r["company_id"], modele, r["version"], r["note"],
                           r["justification"], r["statut"],
                           r["extraction"] if r["extraction"] is None else __import__("json").dumps(r["extraction"]),
                           "bdd", datetime.now()))
    return lignes


def lignes_csv(cur) -> tuple[list[tuple], list[str]]:
    """Runs passés, lus dans output/resultats_*.csv. L'extraction n'y figure pas."""
    cur.execute("SELECT id, name FROM public.companies")
    par_nom = {r["name"].strip().upper(): r["id"] for r in cur.fetchall()}
    cur.execute("SELECT article_id, company_id FROM public.article_companies")
    couples = {(r["article_id"], r["company_id"]) for r in cur.fetchall()}

    lignes, ignores = [], []
    for chemin in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "output", "resultats_*.csv"))):
        modele, quand = _modele_du_fichier(chemin), _horodatage(chemin)
        if not modele or not quand:
            ignores.append(f"{os.path.basename(chemin)} (modèle ou date illisible)")
            continue
        with open(chemin, encoding="utf-8", newline="") as f:
            for l in csv.DictReader(f):
                version = (l.get("prompt_version") or "").strip()
                company_id = par_nom.get((l.get("entreprise") or "").strip().upper())
                statut = (l.get("statut") or "").strip()
                if not version or company_id is None or statut not in ("ok", "failed"):
                    continue
                try:
                    article_id = int(l["article_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (article_id, company_id) not in couples:
                    continue          # couple disparu depuis (réalignement des id du 12/09)
                note = (l.get("note_llm") or "").strip()
                lignes.append((article_id, company_id, modele, version,
                               int(note) if note.isdigit() else None,
                               l.get("justification") or None, statut, None, "csv", quand))
    return lignes, ignores


def amorcer(cur, appliquer: bool) -> None:
    bdd = lignes_bdd(cur)
    csvs, ignores = lignes_csv(cur)
    print(f"état courant d'article_companies : {len(bdd)} évaluations")
    print(f"CSV de runs passés               : {len(csvs)} évaluations")
    for i in ignores:
        print(f"  fichier ignoré : {i}")
    # Les CSV d'abord, la base ensuite : à égalité de clé, c'est l'état courant
    # (plus récent) qui doit l'emporter.
    lignes = _dedoublonner(csvs + bdd)
    print(f"après dédoublonnage              : {len(lignes)} évaluations")
    execute_values(cur, SQL_INSERT, lignes, page_size=1000)
    cur.execute("SELECT count(*) AS n FROM public.article_company_notes")
    print(f"→ {cur.fetchone()['n']} lignes dans l'historique" + ("" if appliquer else " (test à blanc)"))


def restaurer(cur, version: str, modeles: list[str]) -> None:
    """Recopie les notes d'une version de l'historique vers article_companies.

    Sert avant de relancer une évaluation : le filtre de reprise
    d'evaluate_article.py compare la version STOCKÉE dans article_companies à
    celle demandée. Si la base porte du v9 et qu'on relance en v7, il refait
    tout — y compris ce que l'historique contient déjà, et le refait payer.
    Restaurer la version cible d'abord, c'est n'évaluer que ce qui manque.
    """
    for modele in modeles:
        if modele not in MODELES:
            sys.exit(f"Modèle inconnu : {modele}. Connus : {', '.join(MODELES)}")
        note, justif, statut, col_version, extraction = MODELES[modele]
        cur.execute(f"""
            UPDATE public.article_companies ac
               SET {note} = n.note, {justif} = n.justification, {statut} = n.statut,
                   {col_version} = n.prompt_version, {extraction} = n.extraction
              FROM public.article_company_notes n
             WHERE n.article_id = ac.article_id AND n.company_id = ac.company_id
               AND n.modele = %s AND n.prompt_version = %s
        """, (modele, version))
        print(f"  {modele:<8} {cur.rowcount:>5} couples restaurés en {version}")


def etat(cur) -> None:
    cur.execute("""
        SELECT modele, prompt_version, source, count(*) AS n,
               count(note) AS avec_note, min(evalue_le)::date AS debut, max(evalue_le)::date AS fin
          FROM public.article_company_notes
         GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """)
    for r in cur.fetchall():
        print(f"  {r['modele']:<8} {r['prompt_version']:<5} {r['source']:<4} "
              f"{r['n']:>6} lignes, {r['avec_note']:>6} notées, {r['debut']} → {r['fin']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--creer", action="store_true", help="crée la table d'historique")
    parser.add_argument("--amorcer", action="store_true",
                        help="remplit l'historique depuis la base et les CSV passés")
    parser.add_argument("--etat", action="store_true", help="résume le contenu de la table")
    parser.add_argument("--restaurer", metavar="VERSION",
                        help="recopie les notes de cette version vers article_companies "
                             "(à faire avant de relancer une évaluation dans cette version)")
    parser.add_argument("--modeles", default=",".join(MODELES),
                        help="avec --restaurer : modèles concernés, séparés par des virgules")
    parser.add_argument("--appliquer", action="store_true",
                        help="avec --creer ou --amorcer : valide (sinon test à blanc)")
    args = parser.parse_args()
    if not (args.creer or args.amorcer or args.etat or args.restaurer):
        parser.error("choisir --creer, --amorcer, --restaurer ou --etat")

    from cluster_genres import connexion
    conn = connexion()
    try:
        with conn.cursor() as cur:
            if args.creer:
                cur.execute(SQL_TABLE)
                print("table article_company_notes créée" + ("" if args.appliquer else " (test à blanc)"))
            if args.amorcer:
                amorcer(cur, args.appliquer)
            if args.restaurer:
                restaurer(cur, args.restaurer, [m.strip() for m in args.modeles.split(",") if m.strip()])
            if args.etat:
                etat(cur)
            conn.commit() if args.appliquer else conn.rollback()
        if not args.appliquer and (args.creer or args.amorcer or args.restaurer):
            print("\nTEST À BLANC : rien n'a été validé. Relancer avec --appliquer.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
