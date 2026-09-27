#!/usr/bin/env python3
"""
Ajoute des sociétés à l'univers suivi (table companies) avec leurs alias.

Pourquoi : au 20/09/2026, seules 34 sociétés étaient suivies, et 2975 articles
de genre « article dédié », pourtant pourvus d'un corps, n'avaient AUCUN couple
— ils parlaient d'entreprises absentes de la table. Les noms qui revenaient le
plus sont tous du CAC 40 ou du SBF 120. Ajouter ces 34 sociétés fait entrer
environ 540 articles dédiés de plus dans le périmètre, sans scraper une seule
page : ils sont déjà en base. C'est le meilleur levier pour grossir le corpus
d'étiquettes qui servira à entraîner un transformer.

Les alias comptent autant que le nom : `entreprises.variantes_entreprise()` les
lit dans company_aliases, et c'est par eux que la plupart des titres sont
reconnus (« Hermès » pour Hermès International, « Ubisoft », « LVMH »).

isin et ticker sont laissés VIDES : ils acceptent NULL, ne servent pas dans la
chaîne, et des valeurs de seconde main y seraient invérifiables à l'œil.
`--isin fichier.csv` (colonnes name, isin, ticker) les renseigne plus tard.

⚠ Ce script ne crée PAS les couples article-entreprise : c'est l'étape 2
(rattachement par le titre). Et il ne vaut que pour la base locale — en
production, les couples viennent du tagging de fluxRss, qui doit connaître les
mêmes sociétés, sans quoi la chaîne quotidienne ne les rattachera jamais.

Usage :
    cd tfidf
    .venv/bin/python societes_suivies.py                  # test à blanc
    .venv/bin/python societes_suivies.py --appliquer
    .venv/bin/python societes_suivies.py --isin isin.csv --appliquer
"""

import argparse
import csv
import sys
from pathlib import Path

# (nom canonique, alias supplémentaires, secteur_id)
# Le secteur vient de la table sectors ; il est indicatif et modifiable.
SOCIETES = [
    ("LVMH",                      ["LVMH Moët Hennessy Louis Vuitton", "Moët Hennessy", "Louis Vuitton"], 11),
    ("Stellantis",                [], 11),
    ("Kering",                    [], 11),
    ("Renault",                   ["Groupe Renault"], 11),
    ("Vinci",                     ["Vinci SA"], 2),
    ("Hermès International",      ["Hermès"], 11),
    ("Ubisoft Entertainment",     ["Ubisoft"], 11),
    ("Air France-KLM",            ["Air France", "AF-KLM"], 2),
    ("Michelin",                  ["Compagnie Générale des Établissements Michelin"], 11),
    ("Pernod Ricard",             [], 9),
    ("Orange",                    ["Orange SA"], 10),
    ("Danone",                    ["Groupe Danone"], 9),
    ("Edenred",                   [], 13),
    ("Worldline",                 [], 3),
    ("Eramet",                    [], 1),
    ("L'Oréal",                   ["LOreal", "L Oreal"], 9),
    ("Accor",                     ["AccorHotels"], 11),
    ("Vivendi",                   [], 10),
    ("Crédit Agricole",           ["Crédit Agricole SA", "Crédit Agricole S.A."], 4),
    ("Rexel",                     [], 2),
    ("Société Générale",          ["SocGen", "Societe Generale"], 4),
    ("Bouygues",                  ["Groupe Bouygues"], 2),
    ("Legrand",                   [], 2),
    ("Dassault Systèmes",         ["Dassault Systemes"], 3),
    ("Eurofins Scientific",       ["Eurofins"], 7),
    ("Unibail-Rodamco-Westfield", ["Unibail", "URW"], 13),
    ("ArcelorMittal",             ["Arcelor Mittal"], 1),
    ("Rubis",                     [], 6),
    ("Elis",                      [], 2),
    ("Getlink",                   ["Eurotunnel"], 2),
    ("Sodexo",                    [], 9),
    ("Amundi",                    [], 4),
    ("Klepierre",                 ["Klépierre"], 13),
    ("Icade",                     [], 13),
]
PAYS = "France"


def inserer(cur, appliquer: bool) -> None:
    # Les 34 sociétés d'origine viennent de la prod, insérées avec des id
    # explicites : les séquences sont restées à 1 et un INSERT sans id échoue
    # sur la clé primaire. On pose donc les id à la main, puis on recale les
    # séquences (setval échappe à la transaction : seulement si --appliquer).
    cur.execute("SELECT coalesce(max(id), 0) AS n FROM public.companies")
    prochain_id = cur.fetchone()["n"] + 1
    cur.execute("SELECT coalesce(max(id), 0) AS n FROM public.company_aliases")
    prochain_alias = cur.fetchone()["n"] + 1

    cur.execute("SELECT name FROM public.companies")
    existantes = {r["name"].strip().upper() for r in cur.fetchall()}
    creees, ignorees, alias_ajoutes = [], [], 0
    for nom, alias, secteur in SOCIETES:
        if nom.strip().upper() in existantes:
            ignorees.append(nom)
            continue
        company_id = prochain_id
        prochain_id += 1
        cur.execute(
            """INSERT INTO public.companies (id, name, country, sector_id) VALUES (%s, %s, %s, %s)
               ON CONFLICT (name) DO NOTHING RETURNING id""",
            (company_id, nom, PAYS, secteur),
        )
        if not cur.fetchone():
            ignorees.append(nom)
            continue
        creees.append((company_id, nom))
        # Le nom lui-même est un alias : c'est la convention des 34 sociétés
        # déjà présentes, et variantes_entreprise() s'appuie dessus.
        for a in [nom] + alias:
            cur.execute(
                """INSERT INTO public.company_aliases (id, company_id, alias) VALUES (%s, %s, %s)
                   ON CONFLICT (company_id, alias) DO NOTHING""",
                (prochain_alias, company_id, a),
            )
            alias_ajoutes += cur.rowcount
            prochain_alias += cur.rowcount
    if appliquer:
        cur.execute("SELECT setval('companies_id_seq', (SELECT max(id) FROM public.companies))")
        cur.execute("SELECT setval('company_aliases_id_seq', (SELECT max(id) FROM public.company_aliases))")
        print("séquences companies_id_seq et company_aliases_id_seq recalées")
    print(f"{len(creees)} sociétés créées, {alias_ajoutes} alias posés"
          + (f", {len(ignorees)} déjà présentes : {', '.join(ignorees)}" if ignorees else ""))
    for company_id, nom in creees:
        print(f"    [{company_id:>3}] {nom}")


def charger_isin(cur, chemin: Path) -> None:
    """Renseigne isin et ticker depuis un CSV (colonnes name, isin, ticker)."""
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        lignes = list(csv.DictReader(f))
    manquantes, maj = [], 0
    for l in lignes:
        cur.execute(
            """UPDATE public.companies SET isin = NULLIF(%s, ''), ticker = NULLIF(%s, '')
                WHERE upper(name) = upper(%s)""",
            (l.get("isin", ""), l.get("ticker", ""), l.get("name", "")),
        )
        maj += cur.rowcount or 0
        if not cur.rowcount:
            manquantes.append(l.get("name"))
    print(f"{maj} sociétés mises à jour depuis {chemin.name}"
          + (f" | noms introuvables : {manquantes}" if manquantes else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--isin", metavar="FICHIER",
                        help="CSV (name, isin, ticker) pour renseigner ces colonnes")
    parser.add_argument("--appliquer", action="store_true", help="valide (sinon test à blanc)")
    args = parser.parse_args()

    from cluster_genres import connexion
    conn = connexion()
    try:
        with conn.cursor() as cur:
            if args.isin:
                charger_isin(cur, Path(args.isin))
            else:
                inserer(cur, args.appliquer)
            cur.execute("SELECT count(*) AS n FROM public.companies")
            print(f"→ {cur.fetchone()['n']} sociétés en base")
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
