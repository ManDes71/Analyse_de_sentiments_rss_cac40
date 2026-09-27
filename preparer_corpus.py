#!/usr/bin/env python3
"""
Prépare le corpus d'entraînement du transformer, à partir des étiquettes déjà
en base — étape préalable à entrainer_transformer.py.

Trois jeux, trois rôles :

  ENTRAÎNEMENT : couples du PÉRIMÈTRE, unanimes (les 3 modèles du consensus
      d'accord), NON annotés à la main. Étiquette = note_consensus, fiable à
      91,3 %. ~830 couples.
  VALIDATION   : une part tirée de l'entraînement, pour surveiller le
      sur-apprentissage. Tirage stable par md5, donc reproductible.
  TEST         : les 191 couples annotés à la main. Étiquette = note_humaine.
      C'est la SEULE vérité, et elle ne doit jamais servir à entraîner.

Pourquoi pas les 1438 unanimes : 562 sont hors périmètre, sans aucune
vérification humaine, et leurs étiquettes enseignent la mauvaise tâche — « Le
CAC 40 finit en hausse » y est noté POSITIF pour Schneider Electric. C'est
précisément l'erreur (le ton du marché pris pour l'effet sur l'entreprise) que
l'annotation humaine reprochait aux LLM.

L'ENTRÉE DU MODÈLE est le point délicat. 57 % des articles dépassent la fenêtre
de 512 tokens, et dans 7 % des cas l'essentiel de l'argument se trouve au-delà
des 1500 premiers caractères (observation de l'annotateur, vérifiée). Tronquer
le début perdrait donc de l'information. On donne à la place :

    <nom de l'entreprise> </s> <titre> <phrases citant l'entreprise, ± voisines>

Le contexte ciblé vient de extraire_contexte_cible(), reprise telle quelle de
classifier.py (retirée le 20/09 comme note, mais pertinente comme entrée) :
elle garde ce qui parle de l'entreprise, où que ce soit dans l'article.

Usage :
    cd tfidf
    .venv/bin/python preparer_corpus.py            # écrit output/corpus_*.jsonl
    .venv/bin/python preparer_corpus.py --exemples # + affiche quelques entrées
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import nltk

from entreprises import charger_alias, citee_dans, variantes_entreprise
from perimetre import raison_exclusion

nltk.download("punkt_tab", quiet=True)

SORTIE = Path(__file__).resolve().parent / "output"
PART_VALIDATION = 0.15
SEPARATEUR = " </s> "        # séparateur de segments de CamemBERT


def extraire_contexte_cible(texte: str, motif) -> str | None:
    """Phrases citant l'entreprise, avec la précédente et la suivante.

    Reprise de classifier.py (commit d0e5e47^). `motif` vient de
    variantes_entreprise() : la recherche porte sur toutes les graphies.
    Retourne None si l'entreprise est introuvable dans le corps.
    """
    phrases = nltk.sent_tokenize(texte, language="french")
    index_trouves = [i for i, p in enumerate(phrases) if citee_dans(motif, p)]
    if not index_trouves:
        return None
    garder = set()
    for idx in index_trouves:
        garder.update({max(0, idx - 1), idx, min(len(phrases) - 1, idx + 1)})
    return " ".join(phrases[i] for i in sorted(garder))


def construire_entree(entreprise: str, titre: str, contenu: str, motif) -> str:
    """L'entrée du modèle : qui on évalue, puis ce qui le concerne.

    Les blancs sont écrasés : le scraping laisse des blocs de dizaines de
    retours à la ligne, qui ne coûteraient que des tokens.
    """
    contexte = extraire_contexte_cible(contenu or "", motif)
    # Faute de contexte (entreprise absente du corps), le titre porte déjà le
    # nom — c'est la règle du périmètre — et suffit donc à poser la question.
    texte = f"{titre or ''} {contexte or ''}"
    return f"{entreprise}{SEPARATEUR}{' '.join(texte.split())}".strip()


def _cle(article_id: int, company_id: int) -> float:
    """Tirage stable dans [0, 1[ : deux exécutions donnent le même partage."""
    return int(hashlib.md5(f"{article_id}-{company_id}".encode()).hexdigest()[:8], 16) / 16**8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exemples", action="store_true", help="affiche quelques entrées construites")
    parser.add_argument("--tous", action="store_true",
                        help="inclut aussi les couples NON unanimes (étiquettes moins sûres, "
                             "mais distribution plus proche du réel : c'est là que vivent les neutres)")
    parser.add_argument("--suffixe", default="", help="suffixe des fichiers produits (ex : _tous)")
    parser.add_argument("--doux", action="store_true",
                        help="ajoute la DISTRIBUTION des votes des 3 modèles du consensus "
                             "(étiquettes douces) au lieu de la seule médiane")
    args = parser.parse_args()

    from cluster_genres import connexion
    conn = connexion()
    votes = {}
    with conn.cursor() as cur:
        charger_alias(cur)
        if args.doux:
            # Distribution des votes des 3 modèles du consensus. Mistral est
            # écarté : il note « neutre » 84 % du temps quand l'humain dit
            # neutre, mais aussi 72 % du temps sinon — il ne discrimine pas.
            # Qwen n'améliore aucune combinaison. Mesuré sur les 191 annotés.
            cur.execute("""
                SELECT article_id, company_id, note, modele
                  FROM public.article_company_notes
                 WHERE prompt_version = 'v7' AND statut = 'ok' AND note IS NOT NULL
                   AND modele IN ('gemini', 'haiku', 'lama')
            """)
            for r in cur.fetchall():
                votes.setdefault((r["article_id"], r["company_id"]), []).append(r["note"])
        cur.execute("""
            SELECT ac.article_id, ac.company_id, cp.name AS entreprise, a.titre, a.contenu,
                   g.label AS genre, ac.note_consensus, ac.consensus_unanime, ac.note_humaine
              FROM public.article_companies ac
              JOIN public.companies cp ON cp.id = ac.company_id
              JOIN public.articles_rss a ON a.id = ac.article_id
              LEFT JOIN public.article_genres g ON g.article_id = a.id AND g.actif
             WHERE ac.note_consensus IS NOT NULL
        """)
        lignes = cur.fetchall()
    conn.close()

    jeux = {"entrainement": [], "validation": [], "test": []}
    hors_perimetre = sans_contexte = 0
    for r in lignes:
        motif = variantes_entreprise(r["company_id"], r["entreprise"])
        if raison_exclusion(r["genre"], r["titre"], r["contenu"], motif):
            hors_perimetre += 1
            continue
        entree = construire_entree(r["entreprise"], r["titre"], r["contenu"], motif)
        if not extraire_contexte_cible(r["contenu"] or "", motif):
            sans_contexte += 1
        exemple = {"article_id": r["article_id"], "company_id": r["company_id"],
                   "entreprise": r["entreprise"], "texte": entree,
                   "titre": r["titre"], "longueur": len(entree)}
        if r["note_humaine"] is not None:
            jeux["test"].append({**exemple, "label": int(r["note_humaine"]), "source": "humain"})
        elif r["consensus_unanime"] or args.tous:
            cible = "validation" if _cle(r["article_id"], r["company_id"]) < PART_VALIDATION else "entrainement"
            ligne = {**exemple, "label": int(r["note_consensus"]),
                     "source": "consensus", "unanime": bool(r["consensus_unanime"])}
            if args.doux:
                v = votes.get((r["article_id"], r["company_id"]), [])
                if len(v) != 3:
                    continue          # sans les 3 votes, pas de distribution fiable
                ligne["distribution"] = [v.count(c) / 3 for c in (0, 1, 2)]
            jeux[cible].append(ligne)

    for nom, ex in jeux.items():
        chemin = SORTIE / f"corpus_{nom}{args.suffixe}_{date.today()}.jsonl"
        with open(chemin, "w", encoding="utf-8") as f:
            for e in ex:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        classes = {c: sum(1 for e in ex if e["label"] == c) for c in (0, 1, 2)}
        longueurs = sorted(e["longueur"] for e in ex)
        print(f"{nom:<13} {len(ex):>4} exemples | classes {classes} | "
              f"longueur médiane {longueurs[len(longueurs)//2] if ex else 0} car. → {chemin.name}")
    print(f"\n{hors_perimetre} couples hors périmètre écartés | "
          f"{sans_contexte} sans contexte dans le corps (titre seul)")

    if args.exemples:
        print("\nExemples d'entrées :")
        for e in jeux["entrainement"][:3]:
            print(f"\n  [label {e['label']}] {e['texte'][:400]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
