#!/usr/bin/env python3
"""
Met à jour article_companies.nbocc avec le nombre d'occurrences du nom
(et de ses alias) de chaque entreprise dans le contenu+titre de l'article.
note_tfidf et note_full restent à 0.
date_estim est mis à la date du jour.

Usage : python set_company_occurs.py [company_id]
Sans argument : traite TOUTES les entreprises.

uv run set_company_occurs.py 15   # une seule entreprise
uv run set_company_occurs.py      # toutes les entreprises
"""

import json
import os
import re
import sys
from datetime import date

import psycopg2
from psycopg2.extras import RealDictCursor

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "REMOVED")
DB_USER     = os.getenv("DB_USER",     "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mdp")

TODAY = date.today()


def charger_dictionnaire_alias(chemin_fichier="output/alias.json"):
    """Charge les alias depuis le fichier JSON s'il existe."""
    candidates = [chemin_fichier]
    if not os.path.isabs(chemin_fichier):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(base_dir, chemin_fichier))
        candidates.append(os.path.join(os.getcwd(), chemin_fichier))

    for chemin in candidates:
        if os.path.exists(chemin):
            with open(chemin, "r", encoding="utf-8") as f:
                raw_aliases = json.load(f)

            if not isinstance(raw_aliases, dict):
                return {}

            return {
                str(nom).strip().upper(): aliases
                for nom, aliases in raw_aliases.items()
                if isinstance(aliases, list)
            }

    print(f"Attention : le fichier {chemin_fichier} est introuvable. On continue sans alias.")
    return {}


def compter_occurrences(texte: str, patterns: list[re.Pattern]) -> int:
    """Compte le nombre total d'occurrences de tous les alias dans le texte."""
    total = 0
    for pat in patterns:
        total += len(pat.findall(texte))
    return total


def build_patterns(termes: list[str]) -> list[re.Pattern]:
    """Construit des patterns regex insensibles à la casse pour chaque terme.

    Utilise les word boundaries \b pour éviter les faux positifs
    (ex: "air" ne matchera pas "airplane").
    """
    return [
        re.compile(r'\b' + re.escape(terme) + r'\b', re.IGNORECASE)
        for terme in termes
        if terme
    ]


def traiter_entreprise(cur, company_id: int, company_name: str, alias_dict: dict) -> int:
    """Traite tous les articles d'une entreprise et retourne le nb de lignes mises à jour."""

    # Récupère tous les alias de l'entreprise + son nom officiel
    cur.execute(
        "SELECT alias FROM public.company_aliases WHERE company_id = %s",
        (company_id,),
    )
    termes = [row["alias"] for row in cur.fetchall()]

    # Ajoute les alias du fichier JSON associés à cette entreprise
    alias_json = alias_dict.get(company_name.upper(), [])
    if isinstance(alias_json, list):
        termes.extend(alias_json)

    # Ajoute aussi une version normalisée du nom de l'entreprise si elle n'est pas déjà présente
    nom_normalise = company_name.strip().upper()
    if nom_normalise not in alias_dict:
        alias_dict[nom_normalise] = []

    termes.append(company_name)
    patterns = build_patterns(list(set(termes)))

    # Récupère les articles liés à cette entreprise
    cur.execute(
        """
        SELECT ac.article_id, a.titre, a.contenu
        FROM public.article_companies ac
        JOIN public.articles_rss a ON a.id = ac.article_id
        WHERE ac.company_id = %s
        """,
        (company_id,),
    )
    articles = cur.fetchall()

    nb_maj = 0
    for art in articles:
        texte = f"{art['titre'] or ''} {art['contenu'] or ''}"
        nbocc = compter_occurrences(texte, patterns)

        cur.execute(
            """
            UPDATE public.article_companies
               SET nbocc      = %s,
                   note_tfidf = 0,
                   note_full  = 0,
                   date_estim = %s
             WHERE article_id = %s
               AND company_id = %s
            """,
            (nbocc, TODAY, art["article_id"], company_id),
        )
        nb_maj += 1

    return nb_maj


def main():
    company_id_filtre = int(sys.argv[1]) if len(sys.argv) > 1 else None
    alias_dict = charger_dictionnaire_alias()

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )

    with conn:
        with conn.cursor() as cur:

            # Sélectionne les entreprises à traiter
            if company_id_filtre:
                cur.execute(
                    "SELECT id, name FROM public.companies WHERE id = %s",
                    (company_id_filtre,),
                )
            else:
                cur.execute("SELECT id, name FROM public.companies ORDER BY id")

            companies = cur.fetchall()

            if not companies:
                print("Aucune entreprise trouvée.")
                return

            total_maj = 0
            for company in companies:
                nb = traiter_entreprise(cur, company["id"], company["name"], alias_dict)
                total_maj += nb
                print(f"  [{company['id']:>4}] {company['name']:<40} → {nb} articles mis à jour")

    conn.close()
    print(f"\nTerminé. {total_maj} lignes mises à jour dans article_companies (date_estim={TODAY}).")


if __name__ == "__main__":
    main()
