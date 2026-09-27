#!/usr/bin/env python3
"""
Prépare l'annotation humaine des couples du périmètre, puis réimporte les notes
relues dans article_companies.note_humaine.

Deux échantillons, deux usages :

--desaccords (défaut) : les couples que Gemini et Haiku notent DIFFÉREMMENT.
    C'est là qu'une note humaine tranche vraiment quelque chose : au moins un
    des deux modèles s'y trompe. Fait le 2026-09-15 (131 cas annotés sur 133).

--accords [N] : un tirage AU HASARD parmi les couples que les deux modèles
    notent PAREIL. Sans lui, on ne sait rien des ~530 couples où ils
    s'accordent, et donc rien de leur exactitude réelle : les désaccords seuls
    donnent une image faussement sombre. Tirage uniforme (et non stratifié par
    note) pour que la part de bonnes réponses s'estime sans correction.

Dans les deux cas le script écrit :
  - output/annotation_{type}_{date}.csv, colonnes ordonnées pour limiter
    l'ancrage : l'article d'abord, votre note ensuite, les verdicts des modèles
    en dernier ;
  - output/outil_annotation_{type}_{date}.html, copie de outil_annotation.html
    alimentée avec ces cas (annotation au clavier, avancement conservé dans le
    navigateur, export d'un CSV au même format). La clé de stockage est propre
    à chaque échantillon, pour que deux séances ne se mélangent pas.

À remplir : `note_humaine` (0 négatif, 1 neutre, 2 positif). Laisser vide un
couple qu'on ne sait pas trancher : l'import l'ignore.

Usage :
    cd tfidf
    .venv/bin/python annoter_desaccords.py                 # les désaccords
    .venv/bin/python annoter_desaccords.py --accords 60    # 60 cas d'accord
    .venv/bin/python annoter_desaccords.py --importer output/..._annote.csv
    .venv/bin/python annoter_desaccords.py --importer ... --appliquer

Les fichiers sont datés et ne sont jamais écrasés : une annotation en cours est
en sécurité.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from cluster_genres import connexion
from entreprises import charger_alias, variantes_entreprise
from perimetre import raison_exclusion

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "output"
MODELE_HTML = ICI / "outil_annotation.html"
LIGNE_DONNEES = 280            # index 0 de la ligne `const DONNEES = [...]`
LONGUEUR_TEXTE = 3000          # au-delà, Excel devient pénible
ECHANTILLON_ACCORDS = 60
SEPARATEUR = ";"               # Excel francophone
COLONNES = ["article_id", "company_id", "entreprise", "source", "published_at", "genre",
            "titre", "texte", "note_humaine", "commentaire",
            "note_gemini", "justification_gemini", "note_haiku", "justification_haiku"]


def _cle(r: dict) -> str:
    return hashlib.md5(f"{r['article_id']}-{r['company_id']}".encode()).hexdigest()


def couples(cur, accord: bool) -> list[dict]:
    """Couples du périmètre notés en v7 par les deux modèles cloud."""
    charger_alias(cur)
    cur.execute(
        f"""
        SELECT ac.article_id, ac.company_id, c.name AS entreprise, a.source, a.published_at,
               a.titre, a.contenu, g.label AS genre,
               ac.note_gemini, ac.justification_gemini, ac.note_haiku, ac.justification_haiku,
               ac.note_humaine
          FROM public.article_companies ac
          JOIN public.companies c ON c.id = ac.company_id
          JOIN public.articles_rss a ON a.id = ac.article_id
          LEFT JOIN public.article_genres g ON g.article_id = a.id AND g.actif
         WHERE ac.statut_gemini = 'ok' AND ac.prompt_version_gemini = 'v7'
           AND ac.statut_haiku  = 'ok' AND ac.prompt_version_haiku  = 'v7'
           AND ac.note_gemini IS {'NOT DISTINCT' if accord else 'DISTINCT'} FROM ac.note_haiku
        """
    )
    retenus = [r for r in cur.fetchall()
               if not raison_exclusion(r["genre"], r["titre"], r["contenu"],
                                       variantes_entreprise(r["company_id"], r["entreprise"]))]
    if accord:
        # Tirage stable par md5 : deux exports successifs donnent le même
        # échantillon, et l'ordre ne dépend pas de la note.
        return sorted(retenus, key=_cle)
    # Les écarts de 2 points d'abord : ce sont les cas les plus instructifs.
    return sorted(retenus, key=lambda r: (-abs(r["note_gemini"] - r["note_haiku"]), _cle(r)))


def _lignes_csv(cas: list[dict]) -> list[dict]:
    return [{**r,
             "published_at": str(r["published_at"]),
             "texte": " ".join((r["contenu"] or "").split())[:LONGUEUR_TEXTE],
             "commentaire": "",
             "note_humaine": r["note_humaine"] if r["note_humaine"] is not None else ""}
            for r in cas]


def ecrire_outil(lignes: list[dict], type_echantillon: str, chemin: Path) -> None:
    """Copie outil_annotation.html en y remplaçant les données et la clé de stockage."""
    if not MODELE_HTML.exists():
        print(f"{MODELE_HTML.name} est introuvable : seul le CSV a été écrit.")
        return
    modele = MODELE_HTML.read_text(encoding="utf-8").split("\n")
    donnees = [{c: l[c] for c in COLONNES} for l in lignes]
    modele[LIGNE_DONNEES] = "const DONNEES = " + json.dumps(donnees, ensure_ascii=False) + ";"
    texte = "\n".join(modele).replace(
        '"annotation_desaccords_v1"', f'"annotation_{type_echantillon}_{date.today()}"')
    chemin.write_text(texte, encoding="utf-8")
    print(f"Outil  : {chemin}")


def exporter(cas: list[dict], type_echantillon: str) -> None:
    csv_chemin = SORTIE / f"annotation_{type_echantillon}_{date.today()}.csv"
    html_chemin = SORTIE / f"outil_annotation_{type_echantillon}_{date.today()}.html"
    for chemin in (csv_chemin, html_chemin):
        if chemin.exists():
            sys.exit(f"{chemin} existe déjà : le renommer avant d'en régénérer un, "
                     "pour ne pas écraser une annotation en cours.")
    lignes = _lignes_csv(cas)
    with open(csv_chemin, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES, delimiter=SEPARATEUR, extrasaction="ignore")
        w.writeheader()
        w.writerows(lignes)
    print(f"{len(lignes)} couples ({type_echantillon}).")
    print(f"CSV    : {csv_chemin}")
    ecrire_outil(lignes, type_echantillon, html_chemin)


def importer(chemin: Path, appliquer: bool) -> None:
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        lignes = list(csv.DictReader(f, delimiter=SEPARATEUR))
    notes, ignorees, invalides = [], 0, []
    for l in lignes:
        brut = (l.get("note_humaine") or "").strip()
        if not brut:
            ignorees += 1
            continue
        if brut not in ("0", "1", "2"):
            invalides.append((l.get("article_id"), l.get("company_id"), brut))
            continue
        notes.append((int(brut), int(l["article_id"]), int(l["company_id"])))
    print(f"{len(lignes)} lignes | {len(notes)} annotées | {ignorees} laissées vides "
          f"| {len(invalides)} invalides")
    if invalides:
        sys.exit(f"Notes hors de 0/1/2, rien n'a été importé : {invalides[:10]}")
    if not notes:
        return
    conn = connexion()
    with conn, conn.cursor() as cur:
        cur.executemany(
            """UPDATE public.article_companies
                  SET note_humaine = %s, annotee_le = current_date
                WHERE article_id = %s AND company_id = %s""",
            notes,
        )
        cur.execute("SELECT count(*) AS n FROM public.article_companies WHERE note_humaine IS NOT NULL")
        total = cur.fetchone()["n"]
        if not appliquer:
            conn.rollback()
            print(f"TEST À BLANC : {len(notes)} notes seraient écrites ({total} au total). "
                  "Relancer avec --appliquer.")
            return
    print(f"{len(notes)} notes humaines écrites ({total} au total en base).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accords", nargs="?", type=int, const=ECHANTILLON_ACCORDS, metavar="N",
                        help=f"tire N couples (défaut {ECHANTILLON_ACCORDS}) parmi ceux où "
                             "Gemini et Haiku donnent la même note")
    parser.add_argument("--importer", metavar="FICHIER",
                        help="réimporte les note_humaine d'un CSV d'annotation rempli")
    parser.add_argument("--appliquer", action="store_true",
                        help="avec --importer : valide l'écriture (sinon test à blanc)")
    args = parser.parse_args()

    if args.importer:
        importer(Path(args.importer), args.appliquer)
        return 0
    with connexion().cursor() as cur:
        if args.accords:
            tous = couples(cur, accord=True)
            print(f"{len(tous)} couples d'accord au total ; tirage de {args.accords}.")
            exporter(tous[:args.accords], "accords")
        else:
            exporter(couples(cur, accord=False), "desaccords")
    return 0


if __name__ == "__main__":
    sys.exit(main())
