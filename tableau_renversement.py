"""
Reproduit le tableau de l'article, section "Le renversement" :
modèles locaux mesurés d'abord contre Haiku sur l'ensemble du corpus (kappa),
puis contre l'annotation REMOVEDelle sur les 101 cas de la zone grise (justesse).

C'est le tableau qui montre l'inversion : le classement "vs Haiku" (métrique
statistique, sur tout le corpus) et le classement "vs vérité terrain" (annotation
REMOVEDelle, sur les seuls cas où les deux références se contredisaient) ne sont
pas d'accord. Mistral Nemo est dernier sur le premier, premier sur le second.

Entrées attendues :
  1. CSV corpus (cherche d'abord dans output/)
     - output/benchmark_classification_VX.csv
     - output/resultats_*.csv
  2. Excel annotations (cherche dans output/)
     - output/zone_grise_a_annoter.xlsx
     - zone_grise_a_annoter.xlsx (répertoire courant)

Usage :
    python tableau_renversement.py --version v7
    python tableau_renversement.py --version v7 --corpus output/resultats_haiku_v7.csv --annotations output/zone_grise_a_annoter.xlsx
"""
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

BASELINE = "note_haiku"

MODELES = {
    "note_mistral": "Mistral Nemo (12B)",
    "note_queen":   "Qwen 2.5 (7B)",
    "note_llama3":  "Llama 3.1 (8B)",
}

COLONNE_STATUT = {
    "note_mistral": "statut_mistral",
    "note_queen":   "statut_queen",
    "note_llama3":  "statut_lama",
}


def charger_corpus(chemin, version):
    """Charge le corpus complet, restreint à la version de prompt et aux
    lignes où Haiku a répondu correctement."""
    df = pd.read_csv(chemin)
    if "prompt_version_lama" in df.columns:
        df = df[df["prompt_version_lama"] == version]
    if "statut_haiku" in df.columns:
        df = df[df["statut_haiku"] == "ok"]
    return df


def charger_verite_terrain(chemin):
    """
    Charge l'annotation REMOVEDelle de la zone grise. Exclut les notes non
    entières (ex: 1.5, laissée pour un cas jugé ambigu par l'annotateur).
    """
    ann = pd.read_excel(chemin)
    ann = ann.dropna(subset=["note_humaine"])
    ann = ann[ann["note_humaine"] != 1.5].copy()
    ann["note_humaine"] = ann["note_humaine"].astype(int)
    return ann[["article_id", "company_id", "note_humaine"]]


def kappa_vs_haiku(corpus, colonne_note, colonne_statut):
    """Kappa entre un modèle local et Haiku, sur tout le corpus fourni."""
    df = corpus
    if colonne_statut in df.columns:
        df = df[df[colonne_statut] == "ok"]
    df = df[df[colonne_note].isin([0, 1, 2]) & df[BASELINE].isin([0, 1, 2])]
    if df.empty:
        return None, 0
    return round(cohen_kappa_score(df[BASELINE], df[colonne_note]), 3), len(df)


def justesse_vs_annotation(corpus, verite, colonne_note):
    """Taux d'accord d'un modèle local avec la vérité terrain, sur les cas
    de la zone grise retrouvés dans le corpus."""
    fusion = corpus.merge(verite, on=["article_id", "company_id"], how="inner")
    fusion = fusion[fusion[colonne_note].isin([0, 1, 2])]
    if fusion.empty:
        return None, 0
    justesse = 100 * (fusion[colonne_note] == fusion["note_humaine"]).mean()
    return round(justesse, 0), len(fusion)


def main():
    import os
    import glob

    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v7", help="version de prompt à étudier")
    p.add_argument("--corpus", default=None,
                    help="chemin du CSV corpus (défaut: cherche dans output/)")
    p.add_argument("--annotations", default=None,
                    help="chemin du fichier annotations (défaut: cherche dans output/)")
    args = p.parse_args()

    # Chercher le fichier corpus
    if args.corpus:
        chemin_corpus = args.corpus
    else:
        # Chercher d'abord dans output/
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
            # Chercher avec glob
            found = glob.glob(f"output/*{args.version.upper()}.csv") + glob.glob(f"output/*{args.version}.csv")
            if found:
                chemin_corpus = found[0]
            else:
                chemin_corpus = f"benchmark_classification_{args.version.upper()}.csv"

    # Chercher le fichier annotations
    if args.annotations:
        chemin_annotations = args.annotations
    else:
        # Chercher d'abord dans output/
        candidates = [
            "output/zone_grise_a_annoter.xlsx",
            "zone_grise_a_annoter.xlsx",
        ]
        chemin_annotations = None
        for cand in candidates:
            if os.path.exists(cand):
                chemin_annotations = cand
                break

        if not chemin_annotations:
            chemin_annotations = "zone_grise_a_annoter.xlsx"

    try:
        corpus = charger_corpus(chemin_corpus, args.version)
        verite = charger_verite_terrain(chemin_annotations)
    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
        print(f"\nCherche :")
        print(f"  Corpus : {chemin_corpus}")
        print(f"  Annotations : {chemin_annotations}")
        return 1

    print(f"Version étudiée : {args.version}")
    print(f"Corpus complet  : {len(corpus)} lignes exploitables (statut Haiku = 'ok')")
    print(f"Vérité terrain  : {len(verite)} cas annotés\n")

    lignes = []
    for colonne, nom in MODELES.items():
        kappa, n_corpus = kappa_vs_haiku(corpus, colonne, COLONNE_STATUT[colonne])
        justesse, n_annot = justesse_vs_annotation(corpus, verite, colonne)
        lignes.append({
            "Modèle": nom,
            "Kappa vs Haiku": kappa,
            "n (corpus)": n_corpus,
            "Justesse vs annotation (%)": justesse,
            "n (annotés)": n_annot,
        })

    tableau = pd.DataFrame(lignes).set_index("Modèle")

    print("=== Tableau complet ===\n")
    print(tableau.to_string())

    # Classements, pour repérer l'inversion comme dans l'article.
    par_kappa = tableau["Kappa vs Haiku"].sort_values(ascending=False)
    par_justesse = tableau["Justesse vs annotation (%)"].sort_values(ascending=False)
    print("\nClassement par kappa (vs Haiku, tout le corpus) :")
    for rang, (nom, val) in enumerate(par_kappa.items(), 1):
        print(f"  {rang}. {nom} ({val})")
    print("\nClassement par justesse (vs annotation, zone grise) :")
    for rang, (nom, val) in enumerate(par_justesse.items(), 1):
        print(f"  {rang}. {nom} ({val} %)")

    print("\n--- Format Markdown pour l'article ---")
    print("| Modèle | Kappa vs Haiku | Justesse vs annotation |")
    print("|---|---|---|")
    rangs_kappa = {nom: r for r, nom in enumerate(par_kappa.index, 1)}
    rangs_justesse = {nom: r for r, nom in enumerate(par_justesse.index, 1)}
    for nom, ligne in tableau.iterrows():
        etiquette_kappa = f"{ligne['Kappa vs Haiku']}"
        if rangs_kappa[nom] == len(tableau):
            etiquette_kappa += " (dernier)"
        etiquette_justesse = f"{ligne['Justesse vs annotation (%)']:.0f} %"
        if rangs_justesse[nom] == 1:
            etiquette_justesse += " (premier)"
        print(f"| {nom} | {etiquette_kappa} | {etiquette_justesse} |")


if __name__ == "__main__":
    main()