#!/usr/bin/env python3
"""
Calcule les embeddings des articles et les stocke dans public.article_embeddings.

Première brique du chantier « genre de l'article » : le clustering de l'étape
suivante consommera ces vecteurs pour distinguer chronique de marché, article
dédié et analyse — et remplacer à terme le filtre nbocc > 2, qui se trompe dans
les deux sens (275 articles dédiés sur 775 exclus à tort, 385 couples non dédiés
admis à tort).

Contrat, repris de prospection_pour_pgvector.md (étape 2) :

  modèle          intfloat/multilingual-e5-base, 768 dimensions
  texte embarqué  "passage: " + titre + " " + contenu[:1500]
  troncature      512 tokens — tronquer, PAS chunker (cf. plus bas)
  normalisation   L2
  source          v_articles_embeddables
  clé             (article_id, modele)

Pourquoi tronquer plutôt que chunker : le genre d'un article se lit dans son
titre et son chapô. La fin des dépêches est du boilerplate (mentions légales,
avertissements, pied de page de la source) qui rapprocherait artificiellement
tous les articles d'une même source — l'inverse de ce qu'on cherche.

Pourquoi le préfixe "passage: " : les modèles e5 sont entraînés avec les
préfixes "query: " / "passage: ". Les omettre dégrade nettement la qualité des
vecteurs. Les articles sont des documents, donc "passage: ".

Reprise incrémentale : texte_hash est le sha256 du texte réellement embarqué,
préfixe compris. Un article n'est recalculé que si son texte a changé, donc un
run interrompu reprend sans rien refaire.

Usage :
    cd tfidf                       # le .env vit ici, pas à la racine
    .venv/bin/python embed_articles.py --dry-run     # compte, sans charger le modèle
    .venv/bin/python embed_articles.py --limit 50    # essai
    .venv/bin/python embed_articles.py               # les 5980 articles
"""

import os
import sys
import time
import hashlib
import argparse

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Le .env vit dans tfidf/, jamais à la racine : lancé d'ailleurs, DB_HOST vaut
# None et psycopg2 se rabat sur la socket Unix locale, avec un message d'erreur
# trompeur qui fait croire que PostgreSQL est éteint.
load_dotenv()

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_USER     = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME     = os.getenv("DB_NAME", "")

MODELE_DEFAUT = "intfloat/multilingual-e5-base"
DIMENSION_ATTENDUE = 768          # doit correspondre à la colonne vector(768)
PREFIXE_E5 = "passage: "
MAX_CARACTERES_CONTENU = 1500     # ~512 tokens une fois le titre ajouté
TAILLE_LOT_DEFAUT = 64            # confortable sur 12 Go à 512 tokens
TAILLE_LOT_INSERT = 500


def construire_texte(titre: str | None, contenu: str | None) -> str:
    """Texte soumis au modèle, préfixe e5 compris."""
    titre = (titre or "").strip()
    contenu = (contenu or "").strip()[:MAX_CARACTERES_CONTENU]
    return f"{PREFIXE_E5}{titre} {contenu}".strip()


def hacher(texte: str) -> str:
    """sha256 du texte réellement embarqué — clé de la reprise incrémentale."""
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def lister_travail(cur, modele: str, force: bool, limite: int | None) -> list[dict]:
    """Articles à (ré)embarquer : nouveaux, ou dont le texte a changé.

    Avec --force, tout est recalculé, y compris les vecteurs à jour.
    """
    cur.execute("SELECT id, titre, contenu FROM public.v_articles_embeddables ORDER BY id")
    candidats = cur.fetchall()

    cur.execute(
        "SELECT article_id, texte_hash FROM public.article_embeddings WHERE modele = %s",
        (modele,),
    )
    connus = {r["article_id"]: r["texte_hash"] for r in cur.fetchall()}

    travail = []
    for art in candidats:
        texte = construire_texte(art["titre"], art["contenu"])
        empreinte = hacher(texte)
        if not force and connus.get(art["id"]) == empreinte:
            continue
        travail.append({"article_id": art["id"], "texte": texte, "hash": empreinte})

    if limite is not None:
        travail = travail[:limite]
    return travail, len(candidats), len(connus)


def enregistrer(cur, modele: str, lot: list[tuple]) -> None:
    """Insère ou met à jour un paquet de vecteurs."""
    execute_values(
        cur,
        """
        INSERT INTO public.article_embeddings (article_id, modele, embedding, texte_hash)
        VALUES %s
        ON CONFLICT (article_id, modele) DO UPDATE
           SET embedding  = EXCLUDED.embedding,
               texte_hash = EXCLUDED.texte_hash,
               created_at = now()
        """,
        [(article_id, modele, vecteur, empreinte) for article_id, vecteur, empreinte in lot],
        page_size=TAILLE_LOT_INSERT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calcule et stocke les embeddings des articles (pgvector).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  .venv/bin/python embed_articles.py --dry-run   # ce qui serait fait, sans modèle
  .venv/bin/python embed_articles.py --limit 50  # essai sur 50 articles
  .venv/bin/python embed_articles.py             # tout ce qui manque
  .venv/bin/python embed_articles.py --force     # tout recalculer
        """,
    )
    parser.add_argument("--modele", default=MODELE_DEFAUT,
                        help=f"modèle SentenceTransformer (défaut : {MODELE_DEFAUT})")
    parser.add_argument("--batch-size", type=int, default=TAILLE_LOT_DEFAUT,
                        help=f"taille de lot GPU (défaut : {TAILLE_LOT_DEFAUT})")
    parser.add_argument("--limit", type=int, default=None,
                        help="ne traiter que les N premiers articles à faire")
    parser.add_argument("--force", action="store_true",
                        help="recalculer même les vecteurs à jour")
    parser.add_argument("--dry-run", action="store_true",
                        help="compter et hacher sans charger le modèle ni écrire")
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )
    register_vector(conn)

    with conn:
        with conn.cursor() as cur:
            travail, nb_candidats, nb_connus = lister_travail(
                cur, args.modele, args.force, args.limit
            )

            print(f"Modèle              : {args.modele}")
            print(f"Articles embeddables: {nb_candidats}")
            print(f"Déjà en base        : {nb_connus}")
            print(f"À (ré)embarquer     : {len(travail)}"
                  + ("  [--force]" if args.force else "")
                  + (f"  [--limit {args.limit}]" if args.limit else ""))

            if not travail:
                print("Rien à faire.")
                return 0

            if args.dry_run:
                print("\n--dry-run : aucun modèle chargé, aucune écriture.")
                for t in travail[:3]:
                    print(f"  article {t['article_id']} · {len(t['texte'])} car. "
                          f"· {t['hash'][:12]}… · {t['texte'][:90]!r}")
                return 0

            # Import tardif : --dry-run ne doit pas payer le chargement de torch.
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\nChargement du modèle sur {device}…")
            modele = SentenceTransformer(args.modele, device=device)
            print(f"Fenêtre du modèle   : {modele.max_seq_length} tokens")

            debut = time.time()
            nb_ecrits = 0
            for depart in range(0, len(travail), args.batch_size):
                paquet = travail[depart:depart + args.batch_size]
                vecteurs = modele.encode(
                    [t["texte"] for t in paquet],
                    batch_size=args.batch_size,
                    normalize_embeddings=True,   # norme L2, exigée par le contrat
                    show_progress_bar=False,
                )
                if vecteurs.shape[1] != DIMENSION_ATTENDUE:
                    print(f"\n❌ Le modèle produit {vecteurs.shape[1]} dimensions, "
                          f"la colonne en attend {DIMENSION_ATTENDUE}. Abandon.",
                          file=sys.stderr)
                    return 1

                enregistrer(cur, args.modele,
                            [(t["article_id"], v, t["hash"])
                             for t, v in zip(paquet, vecteurs)])
                nb_ecrits += len(paquet)

                ecoule = time.time() - debut
                debit = nb_ecrits / ecoule if ecoule else 0
                restant = (len(travail) - nb_ecrits) / debit if debit else 0
                print(f"  {nb_ecrits}/{len(travail)}  "
                      f"({debit:.1f} art/s, ~{restant:.0f}s restantes)", end="\r")

            print(f"\n{nb_ecrits} vecteurs écrits en {time.time() - debut:.1f}s.")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
