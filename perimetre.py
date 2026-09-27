#!/usr/bin/env python3
"""
Étape 4 du chantier pgvector : le périmètre d'évaluation, décidé par le genre.

Remplace le seuil `nbocc > 2`, qui se trompe dans les deux sens : un article bien
écrit répète peu le nom de son sujet (« le groupe », « la société »), une liste de
valeurs le répète mécaniquement. Et nbocc lui-même est faussé par les mots collés
(« l'actionTOTALENERGIESinvitera » : 6 citations, nbocc = 2) et par les corps
dupliqués des chroniques Reuters/Zonebourse (compte ×2).

Un couple (article, entreprise) est ADMIS s'il franchit, dans l'ordre :

  1. genre exclu      chronique, palmarès, indice étranger, crypto, liste de
                      recommandations, actualité industrielle (= articles sans corps)
  2. titre de chronique   REGEX_CHRONIQUE, en OR : rattrape les chroniques que le
                          clustering a laissées dans le bruit « article dédié »
  2b. titre de liste  « SBF 120 - Les actions à suivre… », « Point recommandations
                      des analystes… » : un paragraphe par valeur. Même logique que
                      l'exclusion des listes de recommandations, dont ces titres
                      sont soit la variante Agefi, soit les fuites dans le bruit
  3. sans corps      contenu = titre : 27 de ces articles sont hors cluster 0
  4. nom absent du titre  l'entreprise, sous l'une de ses graphies, doit figurer
                          dans le titre. Sans ce critère, le reste est fait de
                          chroniques non repérées et de mentions incidentes (BNP
                          conseil d'une OPAS, Publicis rattaché à tort à 206
                          articles qui ne le citent pas).

nbocc n'intervient plus.

Usage :
    cd tfidf
    .venv/bin/python perimetre.py      # compare l'ancien et le nouveau périmètre

Écrit deux CSV dans output/ :
    comparaison_perimetre_{date}.csv   tous les couples, ancien / nouveau / raison
    relecture_perimetre_{date}.csv     30 entrants + 30 sortants, colonne verdict
"""

import csv
import hashlib
import re
from collections import Counter
from datetime import date
from itertools import zip_longest

from entreprises import charger_alias, citee_dans, variantes_entreprise

# Titres de chroniques et de palmarès. Repère historique du journal de
# prospection, qui sur-capture ; cluster_genres.py l'importe d'ici pour contrôler
# le clustering. Comme règle, il ne joue que sur les genres non exclus, où il
# ne rattrape que de vraies chroniques (vérifié à la lecture le 2026-09-12).
REGEX_CHRONIQUE = re.compile(r"^(March|CAC ?40|Bourse|Wall Street|Palmar|Indices)",
                             re.IGNORECASE)

GENRES_EXCLUS = frozenset({
    "chronique de marché",
    "palmarès et statistiques de séance",
    "indice étranger quotidien",
    "crypto",
    "liste de recommandations",
    "actualité industrielle",
})

REGEX_TITRE_LISTE = re.compile(
    r"^((SBF ?120 - )?Les actions à suivre|Point recommandations)", re.IGNORECASE)

ANCIEN_NBOCC_MIN = 2
TAILLE_ECHANTILLON = 30


def raison_exclusion(genre: str, titre: str, contenu: str, motif_entreprise) -> str:
    """Pourquoi ce couple sort du périmètre ; chaîne vide s'il est admis.

    Suppose charger_alias() déjà appelé, comme variantes_entreprise().
    """
    titre, contenu = titre or "", contenu or ""
    if genre in GENRES_EXCLUS:
        return "genre exclu"
    if REGEX_CHRONIQUE.match(titre):
        return "titre de chronique"
    if REGEX_TITRE_LISTE.match(titre):
        return "titre de liste"
    if contenu.strip() == titre.strip():
        return "sans corps"
    if not citee_dans(motif_entreprise, titre):
        return "nom absent du titre"
    return ""


def classer(cur) -> list[dict]:
    cur.execute(
        """
        SELECT ac.article_id, ac.company_id, co.name AS entreprise, ac.nbocc,
               coalesce(g.label, '(sans genre)') AS genre, g.cluster_id,
               a.titre, a.contenu
        FROM public.article_companies ac
        JOIN public.articles_rss a ON a.id = ac.article_id
        JOIN public.companies co   ON co.id = ac.company_id
        LEFT JOIN public.article_genres g ON g.article_id = a.id AND g.actif
        ORDER BY ac.article_id, ac.company_id
        """
    )
    lignes = []
    for r in cur.fetchall():
        motif = variantes_entreprise(r["company_id"], r["entreprise"])
        raison = raison_exclusion(r["genre"], r["titre"], r["contenu"], motif)
        ancien = r["nbocc"] > ANCIEN_NBOCC_MIN
        nouveau = not raison
        lignes.append({
            "article_id": r["article_id"],
            "company_id": r["company_id"],
            "entreprise": r["entreprise"],
            "genre": r["genre"],
            "cluster_id": r["cluster_id"],
            "nbocc": r["nbocc"],
            "ancien_admis": ancien,
            "nouveau_admis": nouveau,
            "mouvement": {(False, True): "entre", (True, False): "sort",
                          (True, True): "reste admis", (False, False): "reste exclu"}[(ancien, nouveau)],
            "raison": raison,
            "titre": r["titre"],
            "extrait": (r["contenu"] or "")[:600].replace("\n", " "),
        })
    return lignes


def _cle(ligne: dict) -> str:
    """Tirage reproductible, sans lien avec l'ordre d'insertion."""
    return hashlib.md5(f"{ligne['article_id']}-{ligne['company_id']}".encode()).hexdigest()


def echantillon_sortants(sortants: list[dict], n: int) -> list[dict]:
    """Tour de rôle entre les raisons : chaque règle est relue, pas seulement la
    plus massive (les chroniques rempliraient seules un tirage uniforme)."""
    par_raison: dict[str, list[dict]] = {}
    for ligne in sorted(sortants, key=_cle):
        par_raison.setdefault(ligne["raison"], []).append(ligne)
    melange = [l for tour in zip_longest(*par_raison.values()) for l in tour if l]
    return melange[:n]


def main() -> int:
    # Import tardif : cluster_genres charge sklearn et importe ce module.
    from cluster_genres import connexion

    with connexion() as conn, conn.cursor() as cur:
        charger_alias(cur)
        lignes = classer(cur)

    mouvements = Counter(l["mouvement"] for l in lignes)
    print(f"{len(lignes)} couples")
    print(f"  ancien périmètre (nbocc > {ANCIEN_NBOCC_MIN}) : {sum(l['ancien_admis'] for l in lignes)}")
    print(f"  nouveau périmètre             : {sum(l['nouveau_admis'] for l in lignes)}")
    for m in ("reste admis", "entre", "sort", "reste exclu"):
        print(f"    {m:12} {mouvements[m]}")
    print("  sortants par raison :",
          dict(Counter(l["raison"] for l in lignes if l["mouvement"] == "sort")))
    print("  entrants par genre  :",
          dict(Counter(l["genre"] for l in lignes if l["mouvement"] == "entre")))

    jour = date.today().isoformat()
    chemin_complet = f"output/comparaison_perimetre_{jour}.csv"
    with open(chemin_complet, "w", newline="", encoding="utf-8") as f:
        champs = [k for k in lignes[0] if k != "extrait"]
        w = csv.DictWriter(f, fieldnames=champs, extrasaction="ignore")
        w.writeheader()
        w.writerows(lignes)

    entrants = sorted((l for l in lignes if l["mouvement"] == "entre"), key=_cle)[:TAILLE_ECHANTILLON]
    sortants = echantillon_sortants([l for l in lignes if l["mouvement"] == "sort"], TAILLE_ECHANTILLON)
    chemin_relecture = f"output/relecture_perimetre_{jour}.csv"
    with open(chemin_relecture, "w", newline="", encoding="utf-8") as f:
        champs = ["mouvement", "raison", "article_id", "company_id", "entreprise", "genre",
                  "nbocc", "titre", "extrait", "verdict", "commentaire"]
        w = csv.DictWriter(f, fieldnames=champs, extrasaction="ignore")
        w.writeheader()
        w.writerows(entrants + sortants)

    print(f"\n{chemin_complet}\n{chemin_relecture}  "
          f"({len(entrants)} entrants + {len(sortants)} sortants à relire)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
