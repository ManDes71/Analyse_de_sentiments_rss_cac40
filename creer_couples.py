#!/usr/bin/env python3
"""
Crée les couples (article, entreprise) manquants pour les sociétés qui n'en ont
pas encore — étape 2 de l'élargissement de l'univers suivi.

En production, ces couples viennent du tagging de fluxRss. Les 34 sociétés
ajoutées le 20/09/2026 (étape 1) n'ont, elles, jamais été taguées : ce script
rattrape le retard sur la base locale, sur les articles DÉJÀ importés.

Règle retenue (option B, décision de l'utilisateur) : un couple est créé dès
que l'entreprise est citée, dans le TITRE ou dans le CORPS, sous l'une de ses
graphies connues (`entreprises.variantes_entreprise`). C'est la règle de
fluxRss, et elle garde la base comparable d'une société à l'autre.

Pourquoi ne pas se limiter au titre, alors que le périmètre l'exige ? Parce que
ce sont deux questions différentes. La base doit rester une image fidèle de
« qui est mentionné où » ; c'est `perimetre.raison_exclusion()`, et lui seul,
qui décide de ce qui mérite une évaluation. Se limiter au titre rendrait les
statistiques des nouvelles sociétés incomparables à celles des anciennes.

`nbocc` est renseigné comme le fait set_company_occurs.py (titre + contenu,
alias compris). Il reste informatif : il ne filtre plus rien.

Usage :
    cd tfidf
    .venv/bin/python creer_couples.py                    # test à blanc
    .venv/bin/python creer_couples.py --appliquer
    .venv/bin/python creer_couples.py --societes 35,36   # sociétés choisies
"""

import argparse
import sys
from collections import Counter

from psycopg2.extras import execute_values

from entreprises import charger_alias, citee_dans, compter_citations, variantes_entreprise
from perimetre import raison_exclusion


def societes_a_traiter(cur, demandees: list[int] | None) -> list[tuple[int, str]]:
    """Par défaut : les sociétés qui n'ont aucun couple (jamais taguées)."""
    if demandees:
        cur.execute("SELECT id, name FROM public.companies WHERE id = ANY(%s) ORDER BY id", (demandees,))
    else:
        cur.execute("""
            SELECT c.id, c.name FROM public.companies c
             WHERE NOT EXISTS (SELECT 1 FROM public.article_companies ac WHERE ac.company_id = c.id)
             ORDER BY c.id
        """)
    return [(r["id"], r["name"]) for r in cur.fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--societes", help="ids séparés par des virgules (défaut : celles sans couple)")
    parser.add_argument("--appliquer", action="store_true", help="valide (sinon test à blanc)")
    args = parser.parse_args()
    demandees = [int(x) for x in args.societes.split(",")] if args.societes else None

    from cluster_genres import connexion
    conn = connexion()
    try:
        with conn.cursor() as cur:
            charger_alias(cur)
            societes = societes_a_traiter(cur, demandees)
            if not societes:
                print("Aucune société à traiter : toutes ont déjà des couples.")
                return 0
            print(f"{len(societes)} sociétés à rattacher : "
                  f"{', '.join(n for _, n in societes[:6])}{'…' if len(societes) > 6 else ''}")

            cur.execute("""
                SELECT a.id, a.titre, a.contenu, g.label AS genre
                  FROM public.articles_rss a
                  LEFT JOIN public.article_genres g ON g.article_id = a.id AND g.actif
                 WHERE a.contenu IS NOT NULL
            """)
            articles = cur.fetchall()
            print(f"{len(articles)} articles à balayer…")

            couples, par_societe, admis = [], Counter(), Counter()
            for company_id, nom in societes:
                motif = variantes_entreprise(company_id, nom)
                for a in articles:
                    titre, contenu = a["titre"] or "", a["contenu"] or ""
                    if not (citee_dans(motif, titre) or citee_dans(motif, contenu)):
                        continue
                    couples.append((a["id"], company_id, compter_citations(motif, f"{titre} {contenu}")))
                    par_societe[nom] += 1
                    if not raison_exclusion(a["genre"], titre, contenu, motif):
                        admis[nom] += 1

            print(f"\n{len(couples)} couples à créer, dont {sum(admis.values())} dans le périmètre\n")
            print(f"{'société':<28} {'couples':>8} {'dont périmètre':>15}")
            for nom, n in par_societe.most_common():
                print(f"{nom:<28} {n:>8} {admis[nom]:>15}")

            execute_values(cur, """
                INSERT INTO public.article_companies (article_id, company_id, nbocc)
                VALUES %s ON CONFLICT (article_id, company_id) DO NOTHING
            """, couples, page_size=1000)
            cur.execute("SELECT count(*) AS n FROM public.article_companies")
            print(f"\n→ {cur.fetchone()['n']} couples en base")

            if args.appliquer:
                conn.commit()
                print("APPLIQUÉ.")
            else:
                conn.rollback()
                print("TEST À BLANC : rien n'a été écrit. Relancer avec --appliquer.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
