#!/usr/bin/env python3
"""
Réaligne la base locale sur les id de la prod (EC2) et y importe les articles
et les couples nouveaux, à partir de trois exports CSV de la prod :
    output/articles_rss_EC2.csv, output/article_companies_ec2.csv,
    output/companies_ec2.csv

La base locale est une copie de la prod (fin mars 2026), alimentée ensuite à
part : des articles communs y portent un autre id, et des id locaux désignent
en prod d'autres articles. Un import par id est donc impossible.

Rapprochement par unique_hash SEULEMENT : le lien n'est pas fiable, la prod
republie des articles sous un même lien (calendriers des jours fériés…).

1. Articles communs dont l'id diffère : renumérotés à l'id de la prod, avec
   leurs lignes filles (couples, embeddings, genres, secteurs).
2. Articles locaux absents de la prod : id négatif (-id), hors de portée de
   toute séquence, pour ne jamais heurter un futur id de la prod.
3. Articles de prod absents du local : insérés avec l'id de la prod.
   Le texte des articles communs n'est PAS remplacé par celui de la prod :
   les notes doivent rester cohérentes avec le texte noté.
4. Couples de la prod manquants : insérés avec les valeurs par défaut
   (nbocc = 0, statuts not_evaluated).
5. Séquence recalée sur max(id).

Contrôle d'intégrité : avant et après, une empreinte de chaque table fille,
indexée par unique_hash et non par id, doit rester identique — autrement dit,
chaque note, vecteur et genre reste attaché au même article.

Par défaut : tout s'exécute dans une transaction, puis est ANNULÉ (test à blanc).
    python3 realigner_sur_prod.py              # test à blanc
    python3 realigner_sur_prod.py --appliquer  # valide ; écrit la correspondance
                                               # output/correspondance_ids_{date}.csv
⚠ Sauvegarder la base avant --appliquer (pg_dump -Fc).
"""

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ICI = Path(__file__).resolve().parent
load_dotenv(ICI / ".env")

SORTIE = ICI / "output"
CSV_ARTICLES = SORTIE / "articles_rss_EC2.csv"
CSV_COUPLES = SORTIE / "article_companies_ec2.csv"
CSV_ENTREPRISES = SORTIE / "companies_ec2.csv"

TABLES_FILLES = ("article_companies", "article_embeddings", "article_genres", "article_sectors")
# Décalage temporaire de la renumérotation : hors de portée de tout id réel,
# pour que les clés primaires ne se heurtent jamais en cours de route.
DECALAGE = 100_000_000
COLONNES_INSEREES = ("id", "titre", "lien", "date", "source", "contenu", "feed_id",
                     "published_at", "companies_tagged_at", "tagging_rules_version")


def verifier(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Contrôle échoué : {message}")


def un(cur, requete: str, params=None):
    cur.execute(requete, params)
    return cur.fetchone()[0]


def copier_csv(cur, table: str, chemin: Path) -> int:
    """Charge un CSV d'export dans une table temporaire, colonnes nommées par l'en-tête."""
    with open(chemin, encoding="utf-8", newline="") as f:
        entete = next(csv.reader(f))
        f.seek(0)
        cur.copy_expert(
            f"COPY {table} ({', '.join(entete)}) FROM STDIN WITH (FORMAT csv, HEADER true)", f)
    return un(cur, f"SELECT count(*) FROM {table}")


def empreintes(cur) -> dict:
    """(nombre de lignes, md5) de chaque table fille, indexée par unique_hash."""
    resultat = {}
    for table in TABLES_FILLES:
        cur.execute(f"""
            SELECT count(*), md5(string_agg(e, '|' ORDER BY e))
              FROM (SELECT md5(a.unique_hash || (to_jsonb(t) - 'article_id')::text) AS e
                      FROM {table} t JOIN articles_rss a ON a.id = t.article_id) s
        """)
        resultat[table] = cur.fetchone()
    return resultat


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--appliquer", action="store_true",
                        help="valider la transaction (sinon test à blanc, tout est annulé)")
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"), port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", ""), user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
    )
    cur = conn.cursor()
    try:
        cur.execute("SET LOCAL lock_timeout = '5s'")
        # Bloque toute insertion concurrente (scheduler fluxRss) jusqu'à la fin.
        cur.execute("LOCK TABLE articles_rss IN EXCLUSIVE MODE")

        # ── Chargement des exports de la prod ────────────────────────────────
        cur.execute("CREATE TEMP TABLE imp_articles (LIKE articles_rss) ON COMMIT DROP")
        cur.execute("CREATE TEMP TABLE imp_couples (article_id int, company_id int) ON COMMIT DROP")
        cur.execute("""CREATE TEMP TABLE imp_entreprises (id int, name text, isin text, ticker text,
                       country text, sector_id int) ON COMMIT DROP""")
        n_prod = copier_csv(cur, "imp_articles", CSV_ARTICLES)
        n_couples_prod = copier_csv(cur, "imp_couples", CSV_COUPLES)
        copier_csv(cur, "imp_entreprises", CSV_ENTREPRISES)
        n_local = un(cur, "SELECT count(*) FROM articles_rss")
        print(f"Prod : {n_prod} articles, {n_couples_prod} couples | local : {n_local} articles")

        # ── Contrôles préalables ─────────────────────────────────────────────
        verifier(un(cur, """SELECT count(*) FROM imp_entreprises i FULL JOIN companies c ON c.id = i.id
                            WHERE c.id IS NULL OR i.id IS NULL OR c.name IS DISTINCT FROM i.name""") == 0,
                 "les entreprises diffèrent entre prod et local")
        verifier(un(cur, "SELECT count(*) - count(DISTINCT unique_hash) FROM imp_articles") == 0,
                 "hash en double dans l'export de la prod")
        verifier(un(cur, "SELECT count(*) - count(DISTINCT id) FROM imp_articles") == 0,
                 "id en double dans l'export de la prod")
        verifier(un(cur, """SELECT count(*) FROM imp_couples c
                            WHERE NOT EXISTS (SELECT 1 FROM imp_articles i WHERE i.id = c.article_id)""") == 0,
                 "des couples de la prod pointent vers des articles absents de l'export")
        verifier(un(cur, "SELECT count(*) FROM articles_rss WHERE id <= 0") == 0,
                 "des id locaux sont déjà négatifs (script déjà appliqué ?)")

        avant = empreintes(cur)

        # ── Correspondance des id ────────────────────────────────────────────
        cur.execute("""
            CREATE TEMP TABLE corresp ON COMMIT DROP AS
            SELECT a.id AS ancien, i.id AS nouveau
              FROM articles_rss a JOIN imp_articles i USING (unique_hash)
             WHERE a.id <> i.id
            UNION ALL
            SELECT a.id, -a.id
              FROM articles_rss a
             WHERE NOT EXISTS (SELECT 1 FROM imp_articles i WHERE i.unique_hash = a.unique_hash)
        """)
        cur.execute("""
            CREATE TEMP TABLE nouveaux ON COMMIT DROP AS
            SELECT i.id FROM imp_articles i
             WHERE NOT EXISTS (SELECT 1 FROM articles_rss a WHERE a.unique_hash = i.unique_hash)
        """)
        n_renum = un(cur, "SELECT count(*) FROM corresp WHERE nouveau > 0")
        n_local_seul = un(cur, "SELECT count(*) FROM corresp WHERE nouveau < 0")
        n_nouveaux = un(cur, "SELECT count(*) FROM nouveaux")
        n_collisions = un(cur, """SELECT count(*) FROM nouveaux n JOIN articles_rss a ON a.id = n.id""")
        print(f"Renumérotés : {n_renum} | locaux absents de la prod (→ id négatif) : {n_local_seul}"
              f" | nouveaux : {n_nouveaux} (dont {n_collisions} sur un id local occupé)")
        verifier(n_local + n_nouveaux - n_local_seul == n_prod,
                 "les comptes ne bouclent pas (local + nouveaux - locaux seuls ≠ prod)")
        verifier(un(cur, """
            SELECT count(*) - count(DISTINCT id) FROM (
                SELECT id FROM articles_rss WHERE id NOT IN (SELECT ancien FROM corresp)
                UNION ALL SELECT nouveau FROM corresp
                UNION ALL SELECT id FROM nouveaux) f""") == 0,
                 "deux articles recevraient le même id")

        # ── Renumérotation ───────────────────────────────────────────────────
        # Les clés étrangères sont en ON UPDATE NO ACTION et non différables :
        # on les retire le temps de renuméroter, puis on les recrée à l'identique
        # (leur recréation revalide toutes les lignes).
        cur.execute("""SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid)
                         FROM pg_constraint WHERE contype = 'f' AND confrelid = 'articles_rss'::regclass""")
        cles = cur.fetchall()
        verifier(sorted(t for t, _, _ in cles) == sorted(TABLES_FILLES),
                 f"tables dépendantes inattendues : {sorted(t for t, _, _ in cles)}")
        for table, nom, _ in cles:
            cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT {nom}")

        # Deux passes : d'abord vers id + DECALAGE, zone libre, puis vers l'id
        # final. En une seule passe, un id cible encore occupé par un article
        # pas encore déplacé ferait échouer la clé primaire.
        for table, colonne in [(t, "article_id") for t in TABLES_FILLES] + [("articles_rss", "id")]:
            cur.execute(f"""UPDATE {table} t SET {colonne} = c.nouveau + %s
                              FROM corresp c WHERE t.{colonne} = c.ancien""", (DECALAGE,))
            cur.execute(f"UPDATE {table} SET {colonne} = {colonne} - %s WHERE {colonne} > %s",
                        (DECALAGE, DECALAGE // 2))

        for table, nom, definition in cles:
            cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {nom} {definition}")

        # ── Insertion des articles nouveaux ──────────────────────────────────
        colonnes = ", ".join(COLONNES_INSEREES)
        cur.execute(f"""INSERT INTO articles_rss ({colonnes})
                        SELECT {', '.join('i.' + c for c in COLONNES_INSEREES)}
                          FROM imp_articles i JOIN nouveaux n ON n.id = i.id""")
        verifier(cur.rowcount == n_nouveaux, f"{cur.rowcount} articles insérés au lieu de {n_nouveaux}")

        # Chaque article de la prod doit maintenant exister sous son id, avec
        # son hash (recalculé par le trigger pour les nouveaux).
        verifier(un(cur, """SELECT count(*) FROM imp_articles i
                            LEFT JOIN articles_rss a ON a.id = i.id AND a.unique_hash = i.unique_hash
                            WHERE a.id IS NULL""") == 0,
                 "des articles de la prod manquent ou ont un autre hash sous leur id")

        # Les articles nouveaux n'ont encore aucune ligne fille : l'empreinte
        # doit être strictement identique.
        apres = empreintes(cur)
        for table in TABLES_FILLES:
            print(f"  {table:<20} {avant[table][0]:>6} lignes  "
                  f"{'identique' if avant[table] == apres[table] else 'DIFFÉRENT'}")
            verifier(avant[table] == apres[table], f"{table} a changé pendant la renumérotation")

        # ── Couples de la prod ───────────────────────────────────────────────
        n_couples_communs = un(cur, """
            SELECT count(*) FROM imp_couples c
             WHERE c.article_id NOT IN (SELECT id FROM nouveaux)
               AND NOT EXISTS (SELECT 1 FROM article_companies ac
                                WHERE ac.article_id = c.article_id AND ac.company_id = c.company_id)""")
        cur.execute("""INSERT INTO article_companies (article_id, company_id)
                       SELECT article_id, company_id FROM imp_couples
                       ON CONFLICT (article_id, company_id) DO NOTHING""")
        print(f"Couples insérés : {cur.rowcount} (dont {n_couples_communs} sur des articles communs)")

        # setval échappe à la transaction (un ROLLBACK ne l'annule pas) :
        # en test à blanc, on se contente d'afficher la valeur prévue.
        id_max = un(cur, "SELECT max(id) FROM articles_rss")
        if args.appliquer:
            cur.execute("SELECT setval('articles_rss_id_seq', %s)", (id_max,))
        print(f"Séquence recalée à {id_max}" + ("" if args.appliquer else " (prévu, non appliqué)"))
        print(f"Articles en base : {un(cur, 'SELECT count(*) FROM articles_rss')}"
              f" | couples : {un(cur, 'SELECT count(*) FROM article_companies')}")

        cur.execute("SELECT ancien, nouveau FROM corresp ORDER BY ancien")
        correspondance = cur.fetchall()

        if not args.appliquer:
            conn.rollback()
            print("\nTEST À BLANC : transaction annulée, la base n'a pas changé."
                  " Relancer avec --appliquer pour valider.")
            return

        conn.commit()
        chemin = SORTIE / f"correspondance_ids_{date.today()}.csv"
        with open(chemin, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ancien_id_local", "nouvel_id"])
            w.writerows(correspondance)
        print(f"\nAPPLIQUÉ. Correspondance des {len(correspondance)} id modifiés : {chemin}")
    except Exception as e:
        conn.rollback()
        print(f"\nÉCHEC, transaction annulée : {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
