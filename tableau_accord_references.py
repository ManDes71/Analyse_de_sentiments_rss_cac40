"""
Reproduit le tableau de l'article :
"le taux d'accord entre mes deux modèles de référence — Claude Haiku et
Gemini 2.5 Flash — calculé sur l'ensemble du corpus, à chaque version de prompt."

C'est LA métrique de la section 3 : elle ne mesure aucun des modèles évalués
(Llama, Mistral, Qwen n'entrent pas dans le calcul), seulement le degré
d'accord entre les deux références. Elle ne varie que si le prompt varie.

Entrées attendues :
  1. Par défaut : cherche dans output/
     - output/benchmark_classification_V*.csv
     - output/resultats_*.csv (depuis evaluate_article.py)
  2. Sinon : cherche dans le répertoire courant
     - benchmark_classification_V*.csv

(v6 est absente : ce run n'a porté que sur Llama, sans Haiku/Gemini en v6.)

Usage :
    python tableau_accord_references.py
    python tableau_accord_references.py --fichiers output/resultats_haiku_v*.csv output/resultats_gemini_v*.csv
"""
import argparse
import glob
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

BASELINE = "note_haiku"
REFERENCE_SECONDAIRE = "note_gemini"

# Ordre d'affichage voulu pour l'article (v6 exclue : pas de run Haiku/Gemini
# sous cette version, cf. docstring).
VERSIONS_ARTICLE = ["v1", "v2", "v3", "v4", "v5", "v7"]


def extraire_version(nom_fichier):
    """Déduit la version de prompt du nom de fichier (ex: '..._V4.csv' -> 'v4')."""
    m = re.search(r"[_-][Vv](\d+)", nom_fichier)
    return f"v{m.group(1)}" if m else None


def mesurer_accord(chemin, version):
    """
    Charge un export, le restreint à sa version de prompt et aux lignes fiables
    des deux références, puis calcule accord/kappa/taille de la zone grise.
    """
    df = pd.read_csv(chemin)

    if "prompt_version_lama" in df.columns:
        df = df[df["prompt_version_lama"] == version]

    # Ne garder que les lignes où les deux références ont effectivement répondu
    # (statut 'ok'), pour ne pas compter un échec silencieux comme un désaccord.
    for col in ("statut_haiku", "statut_gemini"):
        if col in df.columns:
            df = df[df[col] == "ok"]

    df = df[df[BASELINE].isin([0, 1, 2]) & df[REFERENCE_SECONDAIRE].isin([0, 1, 2])]
    if df.empty:
        return None

    accord = accuracy_score(df[BASELINE], df[REFERENCE_SECONDAIRE])
    kappa = cohen_kappa_score(df[BASELINE], df[REFERENCE_SECONDAIRE])
    zone_grise = int((df[BASELINE] != df[REFERENCE_SECONDAIRE]).sum())

    return {
        "version": version,
        "n": len(df),
        "accord_pct": round(100 * accord, 1),
        "kappa": round(kappa, 3),
        "zone_grise": zone_grise,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fichiers", nargs="+", default=None,
                    help="Fichiers CSV à traiter (par défaut : benchmark_classification_V*.csv)")
    args = p.parse_args()

    if args.fichiers:
        fichiers = args.fichiers
    else:
        # Chercher d'abord dans output/, puis dans le répertoire courant
        fichiers = sorted(glob.glob("output/benchmark_classification_V*.csv"))
        if not fichiers:
            fichiers = sorted(glob.glob("output/resultats_*.csv"))
        if not fichiers:
            fichiers = sorted(glob.glob("benchmark_classification_V*.csv"))

    if not fichiers:
        print("❌ Aucun fichier trouvé.")
        print("Cherche dans :")
        print("  - output/benchmark_classification_V*.csv")
        print("  - output/resultats_*.csv")
        print("  - benchmark_classification_V*.csv (répertoire courant)")
        print("\nUsage :")
        print("  python tableau_accord_references.py")
        print("  python tableau_accord_references.py --fichiers output/resultats_*.csv")
        return 1

    lignes = []
    for chemin in fichiers:
        version = extraire_version(os.path.basename(chemin))
        if version is None:
            print(f"Version non détectée pour {chemin}, fichier ignoré.")
            continue
        resultat = mesurer_accord(chemin, version)
        if resultat is None:
            print(f"{chemin} ({version}) : aucune ligne exploitable, ignoré.")
            continue
        lignes.append(resultat)
        print(f"{version} : {resultat['n']} lignes exploitables "
              f"(statut Haiku et Gemini = 'ok')")

    if not lignes:
        print("\nAucun résultat calculé.")
        return 1

    tableau = pd.DataFrame(lignes).set_index("version")

    # Réordonne selon VERSIONS_ARTICLE quand ces versions sont présentes ;
    # ajoute en fin les versions supplémentaires trouvées (ex: v6, si on
    # décide un jour de la réintégrer avec un run Haiku/Gemini dédié).
    ordre = [v for v in VERSIONS_ARTICLE if v in tableau.index]
    ordre += [v for v in tableau.index if v not in ordre]
    tableau = tableau.loc[ordre]

    print("\n=== Taux d'accord Haiku <-> Gemini par version de prompt ===\n")
    affichage = tableau.rename(columns={
        "n": "n exploitable",
        "accord_pct": "Accord Haiku <-> Gemini",
        "kappa": "Kappa",
        "zone_grise": "Zone grise",
    })
    print(affichage[["n exploitable", "Accord Haiku <-> Gemini", "Kappa", "Zone grise"]]
          .to_string())

    print("\n--- Format Markdown pour l'article ---")
    print("| Version | Accord Haiku ↔ Gemini | Kappa | Zone grise |")
    print("|---|---|---|---|")
    for version, ligne in tableau.iterrows():
        print(f"| {version} | {ligne['accord_pct']:.1f} % | {ligne['kappa']:.3f} | "
              f"{ligne['zone_grise']} cas |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())