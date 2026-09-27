#!/usr/bin/env python3
"""
Rattache à un genre les articles arrivés après le clustering de référence.

À lancer CHAQUE JOUR (en prod, une fois les articles du jour importés) :
    fluxRss (articles, couples) → embed_articles.py → rattacher_genres.py --appliquer
    → evaluate_article.py / classifier.py

Pourquoi ne pas relancer cluster_genres.py : un nouvel HDBSCAN renumérote les
clusters, fait retomber des centaines de chroniques dans le bruit et change le
genre d'articles déjà relus. Test du 2026-09-12 sur 8892 vecteurs : 839 anciens
articles changeaient de genre, 344 couples basculaient dans le périmètre, et les
nouveaux clusters trouvés étaient des sujets (SpaceX, Stellantis, Airbus), pas
des genres.

Méthode : centre de chaque cluster du run de référence (moyenne des vecteurs de
ses articles, renormalisée) ; chaque article sans genre actif rejoint le cluster
dont le centre est le plus proche, si la similarité cosinus atteint SEUIL.
Sinon il est « article dédié » (cluster_id -1), comme le bruit d'HDBSCAN.
Testé sur les 5980 articles de référence : 88 % de genres exacts, 94 % de
décisions exclu/admis identiques au clustering. Sans seuil, la décision tombait
à 90 % : les articles dédiés partaient dans des genres exclus.

Les centres sont TOUJOURS calculés sur le run de référence seul, jamais sur les
articles déjà rattachés : sinon les genres dériveraient d'un jour à l'autre.
Rejouable : seuls les articles sans genre actif sont traités, sous un run_id
stable (RUN_RATTACHEMENT) ; created_at date chaque ajout.

⚠ Si cluster_genres.py --ecrire produit un nouveau run de référence, mettre à
  jour RUN_REFERENCE : le script refuse de tourner si ce run n'est plus actif.
⚠ Si la part d'articles dédiés du jour dépasse ALERTE_PART_DEDIES, un nouveau
  genre est peut-être apparu : relancer HDBSCAN sur tout le corpus (sans
  --ecrire) et chercher des clusters formés surtout d'articles récents qui ne
  soient pas de simples sujets.

Usage :
    cd tfidf
    .venv/bin/python rattacher_genres.py              # test à blanc + CSV de relecture
    .venv/bin/python rattacher_genres.py --appliquer  # écrit dans article_genres
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from psycopg2.extras import execute_values

from cluster_genres import (ETIQUETTE_BRUIT, ETIQUETTES, MODELE_DEFAUT,
                            TAILLES_ATTENDUES, connexion)

RUN_REFERENCE = "e5base-hdbscan-pca50-mcs30-ms5-20260910"
SEUIL = 0.92
RUN_RATTACHEMENT = f"centres-s{SEUIL:.2f}-{RUN_REFERENCE}"
# Référence : 49 % de bruit dans le clustering, 43 % de dédiés parmi les 2912
# articles rattachés le 2026-09-12.
ALERTE_PART_DEDIES = 0.60
TAILLE_ECHANTILLON = 30
SORTIE = Path(__file__).resolve().parent / "output"


def normer(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def charger_centres(cur) -> tuple[np.ndarray, np.ndarray]:
    """(cluster_ids, centres normés) du run de référence, bruit exclu."""
    cur.execute(
        """
        SELECT g.cluster_id, e.embedding
          FROM public.article_genres g
          JOIN public.article_embeddings e ON e.article_id = g.article_id AND e.modele = %s
         WHERE g.run_id = %s AND g.actif AND g.cluster_id <> -1
        """,
        (MODELE_DEFAUT, RUN_REFERENCE),
    )
    lignes = cur.fetchall()
    if not lignes:
        sys.exit(f"Le run de référence {RUN_REFERENCE} n'a plus aucune ligne active "
                 "(nouveau clustering ?) : mettre à jour RUN_REFERENCE.")
    clusters = np.array([l["cluster_id"] for l in lignes])
    if set(clusters) != set(TAILLES_ATTENDUES):
        sys.exit(f"Les clusters du run de référence {sorted(set(clusters))} ne sont pas "
                 f"ceux qui ont été nommés {sorted(TAILLES_ATTENDUES)}.")
    vecteurs = np.array([l["embedding"].to_numpy() for l in lignes], dtype=np.float32)
    ids = np.array(sorted(set(clusters)))
    centres = normer(np.array([vecteurs[clusters == c].mean(0) for c in ids]))
    return ids, centres


def charger_a_rattacher(cur) -> list[dict]:
    """Articles vectorisés sans genre actif."""
    cur.execute(
        """
        SELECT e.article_id, e.embedding, a.titre, a.source, left(a.contenu, 300) AS extrait,
               trim(coalesce(a.contenu, '')) = trim(coalesce(a.titre, '')) AS sans_corps
          FROM public.article_embeddings e
          JOIN public.articles_rss a ON a.id = e.article_id
         WHERE e.modele = %s
           AND NOT EXISTS (SELECT 1 FROM public.article_genres g
                            WHERE g.article_id = e.article_id AND g.actif)
         ORDER BY e.article_id
        """,
        (MODELE_DEFAUT,),
    )
    return cur.fetchall()


def _cle(article_id) -> str:
    return hashlib.md5(str(article_id).encode()).hexdigest()


def echantillon(articles: list[dict]) -> list[dict]:
    """Tour de rôle entre les genres, tirage stable par md5(id)."""
    par_genre = {}
    for a in sorted(articles, key=lambda a: _cle(a["article_id"])):
        par_genre.setdefault(a["label"], []).append(a)
    choisis, rang = [], 0
    while len(choisis) < TAILLE_ECHANTILLON and any(rang < len(v) for v in par_genre.values()):
        for genre in sorted(par_genre):
            if rang < len(par_genre[genre]) and len(choisis) < TAILLE_ECHANTILLON:
                choisis.append(par_genre[genre][rang])
        rang += 1
    return choisis


def ecrire_relecture(articles: list[dict]) -> None:
    chemin = SORTIE / f"relecture_rattachement_{date.today()}.csv"
    if chemin.exists():
        print(f"{chemin.name} existe déjà : non réécrit, pour ne pas écraser une relecture.")
        return
    with open(chemin, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["article_id", "genre", "cluster_id", "similarite", "source",
                    "sans_corps", "titre", "extrait", "verdict", "genre_corrige"])
        for a in echantillon(articles):
            w.writerow([a["article_id"], a["label"], a["cluster_id"], f"{a['similarite']:.3f}",
                        a["source"], a["sans_corps"], a["titre"],
                        " ".join((a["extrait"] or "").split()), "", ""])
    print(f"Relecture : {chemin}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--appliquer", action="store_true",
                        help="écrire les genres dans article_genres (sinon test à blanc)")
    args = parser.parse_args()

    conn = connexion()
    with conn.cursor() as cur:
        clusters, centres = charger_centres(cur)
        articles = charger_a_rattacher(cur)
    print(f"Référence : {RUN_REFERENCE} ({len(clusters)} clusters) | seuil {SEUIL}")
    if not articles:
        print("Aucun article vectorisé sans genre actif : rien à faire.")
        return 0

    vecteurs = normer(np.array([a["embedding"].to_numpy() for a in articles], dtype=np.float32))
    similarites = vecteurs @ centres.T
    plus_proche, sim_max = similarites.argmax(1), similarites.max(1)
    for a, k, s in zip(articles, plus_proche, sim_max):
        a["similarite"] = float(s)
        a["cluster_id"] = int(clusters[k]) if s >= SEUIL else -1
        a["label"] = ETIQUETTES.get(a["cluster_id"], ETIQUETTE_BRUIT)

    comptes = Counter(a["label"] for a in articles)
    print(f"\n{len(articles)} articles à rattacher :")
    for genre, n in comptes.most_common():
        print(f"  {genre:<36} {n:>6}  ({n / len(articles):.0%})")
    part_dedies = comptes[ETIQUETTE_BRUIT] / len(articles)
    if part_dedies > ALERTE_PART_DEDIES:
        print(f"\n⚠ ALERTE : {part_dedies:.0%} d'articles dédiés (seuil d'alerte "
              f"{ALERTE_PART_DEDIES:.0%}). Un nouveau genre est peut-être apparu : "
              "voir la docstring avant de continuer.")

    if not args.appliquer:
        ecrire_relecture(articles)
        print("\nTEST À BLANC : rien n'a été écrit. Relancer avec --appliquer pour valider.")
        return 0

    with conn, conn.cursor() as cur:
        avant = _compter(cur)
        # DO NOTHING couvre aussi l'index « un seul genre actif par article » :
        # un article rattaché entre-temps par un autre run est laissé tel quel.
        execute_values(
            cur,
            """INSERT INTO public.article_genres (article_id, run_id, cluster_id, label, actif)
               VALUES %s ON CONFLICT DO NOTHING""",
            [(a["article_id"], RUN_RATTACHEMENT, a["cluster_id"], a["label"], True) for a in articles],
            page_size=1000,
        )
        ecrits = _compter(cur) - avant
    conn.close()
    print(f"\n{ecrits} genres écrits sous run_id={RUN_RATTACHEMENT}")
    return 0


def _compter(cur) -> int:
    cur.execute("SELECT count(*) AS n FROM public.article_genres WHERE run_id = %s", (RUN_RATTACHEMENT,))
    return cur.fetchone()["n"]


if __name__ == "__main__":
    sys.exit(main())
