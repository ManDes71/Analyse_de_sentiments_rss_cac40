#!/usr/bin/env python3
"""
Étape 3 du chantier pgvector : regrouper les articles par GENRE.

Le genre — chronique de marché, article dédié, note d'analyste, palmarès — a une
signature de surface massive (vocabulaire, structure, listes de pourcentages) que
les embeddings capturent bien. C'est ce qui doit remplacer le filtre nbocc > 2,
lequel se trompe dans les deux sens : 275 articles dédiés sur 775 exclus à tort,
385 couples non dédiés admis à tort.

⚠️ Le clustering REGROUPE, il ne CLASSE pas. Il sort cluster_0, cluster_1… et du
bruit. Aucun paquet ne porte de nom : c'est à un humain de lire quelques titres
par cluster et d'écrire l'étiquette. Ce script prépare ce travail, il ne l'évite pas.

Deux approches, comparées sur le même corpus comme le demande le plan :

  HDBSCAN     découvre les genres réellement présents dans le corpus
  zero-shot   impose des genres décrits à la main, chaque article rejoignant
              la description la plus proche

Juge de paix : les 761 articles que la regex sur les titres identifie comme
chroniques. C'est un jeu étiqueté gratuit, imparfait (elle sur-capture), mais
sur la tâche exacte. Un bon cluster en concentre l'essentiel ET en attrape que
la regex avait ratés — c'est tout l'intérêt de l'opération.

Usage :
    cd tfidf
    .venv/bin/python cluster_genres.py --explorer     # balaie les réglages
    .venv/bin/python cluster_genres.py --detailler 50 # titres par cluster, à lire
"""

import os
import re
import sys
import argparse
from collections import Counter
from datetime import date

import warnings

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pgvector.psycopg2 import register_vector
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from dotenv import load_dotenv

from perimetre import REGEX_CHRONIQUE

load_dotenv()

# sklearn 1.8 annonce un changement de défaut de `copy` dans HDBSCAN : bruit inutile.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_USER     = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME     = os.getenv("DB_NAME", "")

MODELE_DEFAUT = "intfloat/multilingual-e5-base"

# REGEX_CHRONIQUE (importée de perimetre.py) : repère approximatif, repris du
# journal de prospection. Il sur-capture, et c'est précisément pourquoi on
# attend mieux du clustering. Ici il sert de contrôle, jamais de vérité.

# Genres décrits à la main pour l'approche zero-shot. e5 est asymétrique :
# les documents portent "passage: ", les requêtes "query: ".
DESCRIPTIONS_GENRES = {
    "chronique de marché":
        "revue quotidienne de l'indice CAC 40 citant une liste de valeurs et leurs variations",
    "résultats d'entreprise":
        "article consacré aux résultats financiers publiés par une entreprise",
    "note d'analyste":
        "note d'analyste relevant ou abaissant un objectif de cours sur une action",
    "palmarès":
        "palmarès des plus fortes hausses et des plus fortes baisses de la séance",
    "analyse technique":
        "analyse graphique du cours d'une action, supports, résistances et tendance",
    "actualité d'entreprise":
        "annonce d'un contrat, d'une acquisition ou d'un événement dans la vie d'une société",
}



# ---------------------------------------------------------------------------
# Nommage des clusters — l'étape humaine du plan, faite le 2026-09-10
#
# Le clustering REGROUPE, il ne CLASSE pas : ces étiquettes viennent de la
# lecture des titres, cluster par cluster, pas d'un calcul.
#
# ⚠️ Ces identifiants ne valent QUE pour la configuration ci-dessous
# (e5-base, PCA 50, min_cluster_size 30, min_samples 5) sur ce corpus. Changer
# un réglage renumérote les clusters. Le garde-fou TAILLES_ATTENDUES refuse
# d'écrire si les tailles ne correspondent plus.
# ---------------------------------------------------------------------------

CONFIG_NOMMEE = {"pca": 50, "min_cluster_size": 30, "min_samples": 5}

ETIQUETTES = {
    13: "chronique de marché",
     3: "indice étranger quotidien",
     1: "indice étranger quotidien",
     4: "indice étranger quotidien",
     2: "palmarès et statistiques de séance",
     5: "palmarès et statistiques de séance",
     7: "liste de recommandations",
    11: "note d'analyste",
    12: "note d'analyste",
     9: "résultats d'entreprise",
     0: "actualité industrielle",
    14: "analyse technique",
     6: "franchissement de seuil",
     8: "crypto",
    # Le cluster 10 regroupe LVMH, Hermès, Kering : un SECTEUR, pas un genre.
    # Dissous — ses articles rejoignent les dédiés, mais on conserve son
    # cluster_id en base pour garder trace de ce qu'HDBSCAN avait trouvé.
    10: "article dédié",
}

ETIQUETTE_BRUIT = "article dédié"

# Garde-fou : tailles observées lors du nommage.
TAILLES_ATTENDUES = {13: 1850, 9: 269, 2: 185, 0: 135, 7: 81, 3: 78, 1: 64,
                     8: 63, 4: 62, 14: 60, 11: 57, 5: 56, 6: 45, 10: 32, 12: 32}


def verifier_correspondance(etiquettes) -> None:
    """Refuse d'écrire si les clusters ne sont plus ceux qui ont été nommés."""
    tailles = Counter(c for c in etiquettes if c != -1)
    ecarts = []
    for cluster, attendue in TAILLES_ATTENDUES.items():
        obtenue = tailles.get(cluster, 0)
        if obtenue != attendue:
            ecarts.append(f"cluster {cluster} : {obtenue} articles, {attendue} attendus")
    inconnus = set(tailles) - set(TAILLES_ATTENDUES)
    if inconnus:
        ecarts.append(f"clusters inattendus : {sorted(inconnus)}")
    if ecarts:
        sys.exit("Les clusters ne correspondent plus au nommage :\n  "
                 + "\n  ".join(ecarts)
                 + "\n\nRelancer --detailler, relire les titres et mettre à jour "
                   "ETIQUETTES et TAILLES_ATTENDUES.")


def ecrire(conn, ids, etiquettes, run_id: str) -> None:
    """Enregistre le genre de chaque article et active ce run.

    Un index unique partiel garantit un seul genre actif par article : il faut
    donc désactiver le run précédent AVANT d'insérer, dans la même transaction.
    """
    lignes = [
        (int(article_id), run_id, int(cluster),
         ETIQUETTES.get(int(cluster), ETIQUETTE_BRUIT) if cluster != -1 else ETIQUETTE_BRUIT,
         True)
        for article_id, cluster in zip(ids, etiquettes)
    ]
    with conn.cursor() as cur:
        cur.execute("UPDATE public.article_genres SET actif = false WHERE actif")
        desactives = cur.rowcount
        execute_values(
            cur,
            """
            INSERT INTO public.article_genres (article_id, run_id, cluster_id, label, actif)
            VALUES %s
            ON CONFLICT (article_id, run_id) DO UPDATE
               SET cluster_id = EXCLUDED.cluster_id,
                   label      = EXCLUDED.label,
                   actif      = EXCLUDED.actif
            """,
            lignes, page_size=1000,
        )
    print(f"{desactives} lignes désactivées, {len(lignes)} écrites sous run_id={run_id}")


def connexion():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, cursor_factory=RealDictCursor,
    )
    register_vector(conn)
    return conn


def charger(cur, modele: str):
    """Vecteurs, titres et sources, dans un ordre stable."""
    cur.execute(
        """
        SELECT e.article_id, e.embedding, a.titre, a.source
        FROM public.article_embeddings e
        JOIN public.articles_rss a ON a.id = e.article_id
        WHERE e.modele = %s
        ORDER BY e.article_id
        """,
        (modele,),
    )
    lignes = cur.fetchall()
    if not lignes:
        sys.exit(f"Aucun embedding pour le modèle {modele}. Lancer embed_articles.py d'abord.")
    # pgvector 0.5 renvoie un objet Vector, pas un tableau numpy.
    vecteurs = np.array([l["embedding"].to_numpy() for l in lignes], dtype=np.float32)
    ids = np.array([l["article_id"] for l in lignes])
    titres = [l["titre"] or "" for l in lignes]
    sources = [l["source"] or "?" for l in lignes]
    return ids, vecteurs, titres, sources


def qualite(etiquettes, est_chronique) -> dict:
    """Le cluster dominant des chroniques les concentre-t-il vraiment ?

    rappel    : part des chroniques-regex tombées dans ce cluster
    purete    : part de ce cluster qui est chronique-regex
    surplus   : articles du cluster que la regex avait ratés — le gain espéré
    """
    interessants = [c for c in set(etiquettes) if c != -1]
    if not interessants:
        return {"cluster": None, "rappel": 0.0, "purete": 0.0, "surplus": 0}
    total_chroniques = int(est_chronique.sum())
    meilleur, score = None, -1
    for c in interessants:
        masque = etiquettes == c
        pris = int((masque & est_chronique).sum())
        if pris > score:
            meilleur, score = c, pris
    masque = etiquettes == meilleur
    return {
        "cluster": meilleur,
        "rappel": score / total_chroniques if total_chroniques else 0.0,
        "purete": score / int(masque.sum()),
        "surplus": int(masque.sum()) - score,
    }


def lancer_hdbscan(vecteurs, min_cluster_size, dims_pca=None, min_samples=5):
    """Les vecteurs étant normalisés L2, la distance euclidienne ordonne comme
    le cosinus : pas besoin de métrique spéciale.

    min_samples est LE levier sur le bruit, et il faut le découpler de
    min_cluster_size : laissé à sa valeur par défaut (= min_cluster_size),
    HDBSCAN classait 61 à 82 % du corpus en bruit, inexploitable. À 5, le bruit
    tombe autour de 49 % pour 15 clusters.
    """
    X = vecteurs
    if dims_pca:
        X = PCA(n_components=dims_pca, random_state=0).fit_transform(X)
    modele = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                     metric="euclidean", n_jobs=-1)
    return modele.fit_predict(X)


def explorer(vecteurs, titres, est_chronique):
    print("=== HDBSCAN : balayage des réglages ===\n")
    print(f"{'PCA':>6} {'min_size':>9} {'clusters':>9} {'bruit':>8} "
          f"{'rappel':>8} {'pureté':>8} {'surplus':>8}")
    resultats = []
    for dims in (None, 50):
        for taille in (30, 50, 100):
            etiquettes = lancer_hdbscan(vecteurs, taille, dims)  # min_samples=5
            nb = len(set(etiquettes)) - (1 if -1 in etiquettes else 0)
            bruit = float((etiquettes == -1).mean())
            q = qualite(etiquettes, est_chronique)
            print(f"{str(dims or '—'):>6} {taille:>9} {nb:>9} {bruit:>7.1%} "
                  f"{q['rappel']:>7.1%} {q['purete']:>7.1%} {q['surplus']:>8}")
            resultats.append((dims, taille, etiquettes, nb, bruit, q))
    return resultats


def zero_shot(vecteurs, est_chronique, modele_nom):
    """Chaque article rejoint la description de genre la plus proche."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(modele_nom, device=device)
    noms = list(DESCRIPTIONS_GENRES)
    descriptions = [f"query: {DESCRIPTIONS_GENRES[n]}" for n in noms]
    ancres = st.encode(descriptions, normalize_embeddings=True)

    # Centrage sur la moyenne du corpus AVANT de comparer. Les espaces
    # d'embedding sont anisotropes : une direction commune à tous les vecteurs
    # crée des « hubs », des ancres qui attirent tout. Sans centrage, « palmarès »
    # raflait 2979 articles sur 5980 et « actualité d'entreprise » 3, pour un
    # rappel de 26 % sur les chroniques. Centré : répartition équilibrée et
    # rappel de 62 %.
    centre = vecteurs.mean(axis=0)
    V = vecteurs - centre
    A = ancres - centre
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    affectation = (V @ A.T).argmax(axis=1)

    print("\n=== Zero-shot par description ===\n")
    print(f"{'genre décrit':<26}{'articles':>9}{'dont chroniques-regex':>24}")
    for i, nom in enumerate(noms):
        masque = affectation == i
        n = int(masque.sum())
        chron = int((masque & est_chronique).sum())
        part = f"{chron}  ({chron/n:.0%})" if n else "—"
        print(f"{nom:<26}{n:>9}{part:>24}")

    i_chronique = noms.index("chronique de marché")
    masque = affectation == i_chronique
    pris = int((masque & est_chronique).sum())
    total = int(est_chronique.sum())
    print(f"\nsur le genre « chronique de marché » : "
          f"rappel {pris/total:.1%}, pureté {pris/max(int(masque.sum()),1):.1%}, "
          f"surplus {int(masque.sum()) - pris}")
    return affectation, noms


def detailler(etiquettes, vecteurs, titres, sources, est_chronique, nb_titres=10):
    """Le matériel de nommage : ce qu'un humain doit lire pour étiqueter.

    Les titres sont triés par distance au centroïde du cluster. Les plus proches
    montrent le cœur du genre, les deux derniers montrent sa frontière — c'est
    là qu'on voit si le cluster tient ou s'il mélange deux choses.
    """
    tailles = Counter(c for c in etiquettes if c != -1)
    print(f"\n=== {len(tailles)} clusters, {nb_titres} titres chacun ===")
    for cluster, taille in tailles.most_common():
        masque = etiquettes == cluster
        indices = np.flatnonzero(masque)
        centroide = vecteurs[indices].mean(axis=0)
        distances = np.linalg.norm(vecteurs[indices] - centroide, axis=1)
        ordre = indices[np.argsort(distances)]

        chron = int((masque & est_chronique).sum())
        top_sources = Counter(sources[i] for i in indices).most_common(3)
        print(f"\n━━━ cluster {cluster} · {taille} articles · "
              f"{chron} chroniques-regex ({chron/taille:.0%})")
        print(f"    sources : {', '.join(f'{s} ({n})' for s, n in top_sources)}")
        for i in ordre[:nb_titres]:
            print(f"      {titres[i][:100]}")
        if taille > nb_titres + 2:
            print(f"    ── en marge du cluster ──")
            for i in ordre[-2:]:
                print(f"      {titres[i][:100]}")

    bruit = int((etiquettes == -1).sum())
    print(f"\n━━━ bruit · {bruit} articles non classés "
          f"({int(est_chronique[etiquettes == -1].sum())} chroniques-regex)")
    indices = np.flatnonzero(etiquettes == -1)
    for i in indices[:: max(len(indices) // nb_titres, 1)][:nb_titres]:
        print(f"      {titres[i][:100]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clustering de genre sur les embeddings.")
    parser.add_argument("--modele", default=MODELE_DEFAUT)
    parser.add_argument("--explorer", action="store_true",
                        help="balayer les réglages HDBSCAN et comparer au zero-shot")
    parser.add_argument("--detailler", type=int, metavar="MIN_SIZE", default=None,
                        help="afficher les titres par cluster pour ce min_cluster_size")
    parser.add_argument("--pca", type=int, default=50,
                        help="dimensions PCA avant HDBSCAN (0 = pas de PCA)")
    parser.add_argument("--ecrire", action="store_true",
                        help="écrire les genres nommés dans article_genres")
    parser.add_argument("--titres", type=int, default=10,
                        help="nombre de titres affichés par cluster")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="densité minimale HDBSCAN : le levier sur le bruit")
    args = parser.parse_args()

    conn = connexion()
    with conn:
        with conn.cursor() as cur:
            ids, vecteurs, titres, sources = charger(cur, args.modele)
    conn.close()

    est_chronique = np.array([bool(REGEX_CHRONIQUE.match(t.strip())) for t in titres])
    print(f"{len(ids)} articles · {vecteurs.shape[1]} dimensions · "
          f"{int(est_chronique.sum())} chroniques repérées par la regex\n")

    if args.ecrire:
        c = CONFIG_NOMMEE
        etiquettes = lancer_hdbscan(vecteurs, c["min_cluster_size"], c["pca"],
                                    c["min_samples"])
        verifier_correspondance(etiquettes)
        run_id = (f"e5base-hdbscan-pca{c['pca']}-mcs{c['min_cluster_size']}"
                  f"-ms{c['min_samples']}-{date.today():%Y%m%d}")
        repartition = Counter(
            ETIQUETTES.get(int(x), ETIQUETTE_BRUIT) if x != -1 else ETIQUETTE_BRUIT
            for x in etiquettes)
        print("Répartition par genre :")
        for genre, n in repartition.most_common():
            print(f"  {genre:<38}{n:>6}  ({n/len(etiquettes):.1%})")
        conn = connexion()
        with conn:
            ecrire(conn, ids, etiquettes, run_id)
        conn.close()
    elif args.explorer:
        explorer(vecteurs, titres, est_chronique)
        zero_shot(vecteurs, est_chronique, args.modele)
    elif args.detailler is not None:
        etiquettes = lancer_hdbscan(vecteurs, args.detailler, args.pca or None,
                                    args.min_samples)
        detailler(etiquettes, vecteurs, titres, sources, est_chronique, args.titres)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
