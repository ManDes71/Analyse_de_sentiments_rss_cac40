#!/usr/bin/env python3
"""
Calcule la note de consensus : la médiane de trois modèles, sur une version de
prompt donnée, écrite dans article_companies.note_consensus.

Pourquoi : mesurée sur les 191 annotations humaines, la médiane de Gemini,
Haiku et llama en v7 atteint 81,1 %, contre 79,6 % pour le meilleur modèle
seul (Haiku v8) et 77,9 % pour Gemini v7. Le gain est gratuit — llama tourne en
local — et la règle tient en une ligne. Trois versions de prompt (v7, v8, v9)
n'avaient, elles, rien donné : c'est la combinaison qui paie, pas la consigne.

Pourquoi la médiane de TROIS et pas la moyenne : sur une échelle ordinale
(0 négatif, 1 neutre, 2 positif), la médiane renvoie toujours une note valide
et revient à un vote majoritaire dès que deux modèles s'accordent. La moyenne
inventerait des valeurs intermédiaires qui ne veulent rien dire.

`consensus_unanime` marque les couples où les trois modèles disent la MÊME
chose. C'est le meilleur indicateur de fiabilité mesuré : sur les annotations
humaines, le consensus est juste à 91,3 % sur ces couples, contre 60,0 % sur
les autres (Fisher p < 0,001). Le TF-IDF, qu'on avait envisagé pour ce rôle,
sépare deux fois moins bien (74,6 % contre 57,1 %) et coûte un passage GPU.

À quoi cela sert : ces étiquettes sont la vérité de substitution qui servira à
mesurer, puis à entraîner un transformer — l'annotation humaine (191 couples)
restant le jeu de test. Les couples unanimes forment le sous-ensemble
d'entraînement le plus propre.

Les notes sont lues dans article_company_notes, jamais dans article_companies :
cette dernière ne garde que le dernier run, tous modèles et versions mêlés
(au 20/09/2026 : Gemini et Haiku en v9, les locaux en v8). L'historique permet
de fixer la version de référence, indépendamment du dernier run en date.

Usage :
    cd tfidf
    .venv/bin/python note_consensus.py --creer-colonnes --appliquer
    .venv/bin/python note_consensus.py                    # test à blanc
    .venv/bin/python note_consensus.py --appliquer
    .venv/bin/python note_consensus.py --version v8 --modeles gemini,haiku,lama
"""

import argparse
import sys
from collections import Counter

from psycopg2.extras import execute_values

VERSION_DEFAUT = "v7"
MODELES_DEFAUT = ("gemini", "haiku", "lama")

SQL_COLONNES = """
ALTER TABLE public.article_companies
    ADD COLUMN IF NOT EXISTS note_consensus    smallint,
    ADD COLUMN IF NOT EXISTS consensus_le      date,
    ADD COLUMN IF NOT EXISTS consensus_regle   text,
    ADD COLUMN IF NOT EXISTS consensus_unanime boolean;
"""

# Une ligne par couple, avec les notes des modèles demandés. array_agg garde
# l'ordre imposé par le ORDER BY, mais on ne s'en sert pas : la médiane s'en
# moque, et exiger les trois modèles suffit à garantir la comparabilité.
SQL_NOTES = """
SELECT article_id, company_id, array_agg(note ORDER BY modele) AS notes
  FROM public.article_company_notes
 WHERE prompt_version = %s AND modele = ANY(%s) AND statut = 'ok' AND note IS NOT NULL
 GROUP BY article_id, company_id
HAVING count(*) = %s
"""


def mediane(notes: list[int]) -> int:
    """Médiane d'un nombre impair de notes ordinales."""
    return sorted(notes)[len(notes) // 2]


def calculer(cur, version: str, modeles: tuple[str, ...]) -> list[tuple]:
    cur.execute(SQL_NOTES, (version, list(modeles), len(modeles)))
    lignes = cur.fetchall()
    regle = f"médiane {'+'.join(modeles)} {version}"
    return [(mediane(r["notes"]), len(set(r["notes"])) == 1, regle,
             r["article_id"], r["company_id"]) for r in lignes]


def controler(cur, version: str, modeles: tuple[str, ...]) -> None:
    """Exactitude du consensus et de chaque modèle sur les couples annotés."""
    cur.execute("""
        SELECT n.modele, n.article_id, n.company_id, n.note, ac.note_humaine AS h
          FROM public.article_company_notes n
          JOIN public.article_companies ac USING (article_id, company_id)
         WHERE ac.note_humaine IS NOT NULL AND n.prompt_version = %s
           AND n.modele = ANY(%s) AND n.statut = 'ok'
    """, (version, list(modeles)))
    par_couple = {}
    for r in cur.fetchall():
        par_couple.setdefault((r["article_id"], r["company_id"], r["h"]), {})[r["modele"]] = r["note"]
    complets = {k: v for k, v in par_couple.items() if len(v) == len(modeles)}
    if not complets:
        print("  (aucun couple annoté avec les trois modèles : pas de contrôle possible)")
        return
    print(f"  sur {len(complets)} couples annotés (échantillon biaisé vers les cas "
          "difficiles, à ne pas lire comme un taux global) :")
    for m in modeles:
        print(f"    {m:<10} {sum(v[m] == k[2] for k, v in complets.items())/len(complets):>6.1%}")
    justes = sum(mediane(list(v.values())) == k[2] for k, v in complets.items())
    print(f"    {'CONSENSUS':<10} {justes/len(complets):>6.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", default=VERSION_DEFAUT, help=f"version de prompt (défaut {VERSION_DEFAUT})")
    parser.add_argument("--modeles", default=",".join(MODELES_DEFAUT),
                        help="modèles à combiner, séparés par des virgules (nombre IMPAIR)")
    parser.add_argument("--creer-colonnes", action="store_true",
                        help="ajoute note_consensus, consensus_le et consensus_regle")
    parser.add_argument("--appliquer", action="store_true", help="valide (sinon test à blanc)")
    args = parser.parse_args()

    modeles = tuple(m.strip() for m in args.modeles.split(",") if m.strip())
    if len(modeles) % 2 == 0:
        sys.exit(f"Il faut un nombre IMPAIR de modèles pour une médiane sans ambiguïté : {modeles}")

    from cluster_genres import connexion
    conn = connexion()
    try:
        with conn.cursor() as cur:
            if args.creer_colonnes:
                cur.execute(SQL_COLONNES)
                print("colonnes note_consensus / consensus_le / consensus_regle ajoutées")
            lignes = calculer(cur, args.version, modeles)
            if not lignes:
                sys.exit(f"Aucun couple n'a les {len(modeles)} modèles en {args.version} : "
                         "vérifier --version et --modeles (cf. historique_notes.py --etat).")
            print(f"{len(lignes)} couples ont les {len(modeles)} modèles en {args.version}")
            print("  répartition du consensus :",
                  dict(sorted(Counter(l[0] for l in lignes).items())))
            unanimes = [l for l in lignes if l[1]]
            print(f"  dont unanimes : {len(unanimes)} ({len(unanimes)/len(lignes):.0%}) — "
                  f"étiquettes les plus sûres, 91 % de justesse mesurée : "
                  f"{dict(sorted(Counter(l[0] for l in unanimes).items()))}")
            execute_values(cur, """
                UPDATE public.article_companies ac
                   SET note_consensus = v.note, consensus_unanime = v.unanime,
                       consensus_regle = v.regle, consensus_le = current_date
                  FROM (VALUES %s) AS v (note, unanime, regle, article_id, company_id)
                 WHERE ac.article_id = v.article_id AND ac.company_id = v.company_id
            """, lignes, page_size=1000)
            # cur.rowcount ne rend compte que de la DERNIÈRE page d'execute_values :
            # on recompte pour de bon.
            cur.execute("""SELECT count(*) AS n FROM public.article_companies
                            WHERE consensus_le = current_date AND note_consensus IS NOT NULL""")
            print(f"  {cur.fetchone()['n']} lignes mises à jour")
            controler(cur, args.version, modeles)
            if args.appliquer:
                conn.commit()
                print("\nAPPLIQUÉ.")
            else:
                conn.rollback()
                print("\nTEST À BLANC : rien n'a été écrit. Relancer avec --appliquer.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
