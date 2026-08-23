"""
Trouve les deux chiffres cités dans l'article :
"sur les cas où mes prompts imposent une note par une règle déterministe
plutôt que par un jugement du modèle, l'étiquette est juste à 93% -- contre
71% quand c'est le modèle qui tranche seul."

Principe : sur les 94 cas annotés qui tombent dans la zone de consensus
(Haiku = Gemini), on regarde pour chacun si la note de Haiku a été PRODUITE
PAR LA RÈGLE DÉTERMINISTE de note_finale() -- c'est-à-dire que l'extraction
v6/v7 donne autre_fait_concret=false ET reco_sens='aucune' -- ou si c'est le
JUGEMENT BRUT du modèle qui a été conservé. On compare ensuite chaque groupe
à la vérité humaine.

Entrées attendues :
  1. CSV corpus (cherche d'abord dans output/)
     - output/benchmark_classification_VX.csv
     - output/resultats_*.csv
  2. Excel zone grise (cherche dans output/)
     - output/zone_grise_a_annoter.xlsx
  3. Excel contrôle (cherche dans output/)
     - output/controle_consensus_a_annoter.xlsx

Usage :
    python calcul_fiabilite_regle_vs_modele.py --version v7
    python calcul_fiabilite_regle_vs_modele.py --version v7 \
      --corpus output/resultats_haiku_v7.csv \
      --zone-grise output/zone_grise_a_annoter.xlsx \
      --controle output/controle_consensus_a_annoter.xlsx
"""
import argparse
import json

import pandas as pd

BASELINE = "note_haiku"
REFERENCE_SECONDAIRE = "note_gemini"


def charger_corpus(chemin, version):
    df = pd.read_csv(chemin)
    if "prompt_version_lama" in df.columns:
        df = df[df["prompt_version_lama"] == version]
    for col in ("statut_haiku", "statut_gemini"):
        if col in df.columns:
            df = df[df[col] == "ok"]
    return df


def charger_annotations(chemin):
    ann = pd.read_excel(chemin)
    ann = ann.dropna(subset=["note_humaine"])
    ann = ann[ann["note_humaine"] != 1.5].copy()
    ann["note_humaine"] = ann["note_humaine"].astype(int)
    return ann[["article_id", "company_id", "note_humaine"]]


def origine_regle_ou_modele(extraction_json):
    """
    Reproduit exactement la condition de llm_common.note_finale() : la note
    est imposée par la règle uniquement si aucun autre fait concret n'est
    identifié ET qu'aucun mouvement de recommandation n'est mentionné.
    Retourne 'regle', 'modele', ou None si l'extraction est absente
    (versions v1 à v5, sans champs structurés).
    """
    if pd.isna(extraction_json):
        return None
    try:
        f = json.loads(extraction_json)
    except (TypeError, json.JSONDecodeError):
        return None

    autre_fait = f.get("autre_fait_concret")
    reco_sens = f.get("reco_sens")

    if autre_fait is None:
        return None  # extraction incomplète, on ne sait pas trancher

    regle_applicable = (autre_fait is False) and (reco_sens in (None, "aucune"))
    return "regle" if regle_applicable else "modele"


def main():
    import os
    import glob

    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v7")
    p.add_argument("--corpus", default=None, help="CSV corpus (défaut: cherche dans output/)")
    p.add_argument("--zone-grise", default=None, help="Excel zone grise (défaut: output/zone_grise_a_annoter.xlsx)")
    p.add_argument("--controle", default=None, help="Excel contrôle (défaut: output/controle_consensus_a_annoter.xlsx)")
    args = p.parse_args()

    # Chercher le fichier corpus
    if args.corpus:
        chemin_corpus = args.corpus
    else:
        candidates = [
            f"output/benchmark_classification_{args.version.upper()}.csv",
            f"output/resultats_haiku_{args.version}.csv",
            f"output/resultats_gemini_{args.version}.csv",
            f"benchmark_classification_{args.version.upper()}.csv",
        ]
        chemin_corpus = None
        for cand in candidates:
            if os.path.exists(cand):
                chemin_corpus = cand
                break
        if not chemin_corpus:
            found = glob.glob(f"output/*{args.version.upper()}.csv") + glob.glob(f"output/*{args.version}.csv")
            if found:
                chemin_corpus = found[0]
            else:
                chemin_corpus = f"benchmark_classification_{args.version.upper()}.csv"

    # Chercher zone grise
    if args.zone_grise:
        chemin_zone_grise = args.zone_grise
    else:
        candidates = ["output/zone_grise_a_annoter.xlsx", "zone_grise_a_annoter.xlsx"]
        chemin_zone_grise = None
        for cand in candidates:
            if os.path.exists(cand):
                chemin_zone_grise = cand
                break
        if not chemin_zone_grise:
            chemin_zone_grise = "zone_grise_a_annoter.xlsx"

    # Chercher contrôle
    if args.controle:
        chemin_controle = args.controle
    else:
        candidates = ["output/controle_consensus_a_annoter.xlsx", "controle_consensus_a_annoter.xlsx"]
        chemin_controle = None
        for cand in candidates:
            if os.path.exists(cand):
                chemin_controle = cand
                break
        if not chemin_controle:
            chemin_controle = "controle_consensus_a_annoter.xlsx"

    try:
        corpus = charger_corpus(chemin_corpus, args.version)
    except FileNotFoundError as e:
        print(f"❌ Erreur corpus : {e}")
        print(f"Cherche : {chemin_corpus}")
        return 1

    if "extraction_haiku" not in corpus.columns:
        print("Colonne 'extraction_haiku' absente : cette version ne fournit pas "
              "l'extraction structurée (v6/v7 uniquement). Impossible de continuer.")
        return 1

    try:
        grise = charger_annotations(chemin_zone_grise)
        controle = charger_annotations(chemin_controle)
    except FileNotFoundError as e:
        print(f"❌ Erreur annotations : {e}")
        print(f"Cherche :")
        print(f"  Zone grise : {chemin_zone_grise}")
        print(f"  Contrôle : {chemin_controle}")
        return 1
    toutes = pd.concat([grise, controle]).drop_duplicates(subset=["article_id", "company_id"])
    print(f"Annotations disponibles : {len(toutes)} (après déduplication)\n")

    # Même périmètre que le tableau de qualité d'étiquette : zone de consensus
    # (Haiku = Gemini) uniquement, puisque c'est là que l'étiquette sert
    # potentiellement à l'entraînement.
    consensus = corpus[corpus[BASELINE] == corpus[REFERENCE_SECONDAIRE]].copy()
    fusion = consensus.merge(toutes, on=["article_id", "company_id"], how="inner")
    print(f"Cas annotés dans la zone de consensus : {len(fusion)}\n")

    fusion["origine"] = fusion["extraction_haiku"].apply(origine_regle_ou_modele)

    sans_extraction = fusion["origine"].isna().sum()
    if sans_extraction:
        print(f"({sans_extraction} cas sans extraction exploitable, exclus du calcul)\n")
    fusion = fusion.dropna(subset=["origine"])

    print("=== Répartition ===")
    print(fusion["origine"].value_counts().to_string())
    print()

    print("=== Justesse par origine de la note ===\n")
    for origine, libelle in [("regle", "Note imposée par la RÈGLE"),
                              ("modele", "Note issue du JUGEMENT du modèle")]:
        sous = fusion[fusion["origine"] == origine]
        if sous.empty:
            print(f"  {libelle:<38} : aucun cas")
            continue
        justesse = 100 * (sous[BASELINE] == sous["note_humaine"]).mean()
        print(f"  {libelle:<38} : {justesse:.0f}% (n={len(sous)})")

    print("\n--- Valeurs à reporter dans FIABILITE_REGLE / FIABILITE_MODELE ---")
    for origine, var in [("regle", "FIABILITE_REGLE"), ("modele", "FIABILITE_MODELE")]:
        sous = fusion[fusion["origine"] == origine]
        if not sous.empty:
            justesse = (sous[BASELINE] == sous["note_humaine"]).mean()
            print(f"{var} = {justesse:.2f}")


if __name__ == "__main__":
    main()