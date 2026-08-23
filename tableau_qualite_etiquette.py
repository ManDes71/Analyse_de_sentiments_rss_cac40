"""
Reproduit le tableau de l'article, section "La suite : étalonner un transformer" :
fiabilité de l'étiquette "accord Haiku-Gemini" par classe, mesurée sur les 94
cas annotés (parmi les 140 au total) qui tombent dans la zone de consensus.

Point important, différent du calcul du "Deuxième échantillon : le contrôle" :
ici on réunit les DEUX lots d'annotations (les 101 de la zone grise et les 40
du contrôle), on ne garde que ceux qui tombent dans la zone de consensus du
corpus étudié, et on mesure la justesse GLOBALE ainsi que PAR CLASSE de
l'étiquette. C'est ce qui fait passer le chiffre de 88% (mesuré sur les 40 cas
du seul échantillon de contrôle) à 82% (mesuré sur les 94 cas disponibles),
et surtout ce qui révèle que la fiabilité n'est pas uniforme selon la classe.

Entrées attendues :
  1. CSV corpus (cherche d'abord dans output/)
     - output/benchmark_classification_VX.csv
     - output/resultats_*.csv
  2. Excel zone grise (cherche dans output/)
     - output/zone_grise_a_annoter.xlsx
  3. Excel contrôle (cherche dans output/)
     - output/controle_consensus_a_annoter.xlsx

Usage :
    python tableau_qualite_etiquette.py --version v7
    python tableau_qualite_etiquette.py --version v7 \
      --corpus output/resultats_haiku_v7.csv \
      --zone-grise output/zone_grise_a_annoter.xlsx \
      --controle output/controle_consensus_a_annoter.xlsx
"""
import argparse

import pandas as pd

BASELINE = "note_haiku"
REFERENCE_SECONDAIRE = "note_gemini"

NOMS_CLASSES = {0: "NÉGATIF", 1: "NEUTRE", 2: "POSITIF"}


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

    # Réunion des deux lots d'annotations. drop_duplicates protège contre un
    # éventuel recouvrement entre les deux fichiers (aucun n'est attendu ici,
    # les échantillons ayant été tirés de zones disjointes, mais mieux vaut
    # ne pas compter un même cas deux fois si les lots se recoupaient).
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
    print(f"Annotations disponibles : {len(grise)} (zone grise) + {len(controle)} (contrôle) "
          f"= {len(grise) + len(controle)}, {len(toutes)} après déduplication\n")

    # Zone de consensus du corpus étudié : Haiku et Gemini d'accord.
    consensus = corpus[corpus[BASELINE] == corpus[REFERENCE_SECONDAIRE]].copy()
    print(f"Zone de consensus du corpus {args.version} : {len(consensus)} lignes")

    # Cas annotés qui tombent dans cette zone.
    fusion = consensus.merge(toutes, on=["article_id", "company_id"], how="inner")
    print(f"Cas annotés retrouvés dans la zone de consensus : {len(fusion)}\n")

    justesse_globale = 100 * (fusion[BASELINE] == fusion["note_humaine"]).mean()
    print(f"=== Justesse GLOBALE de l'étiquette 'accord Haiku-Gemini' : "
          f"{justesse_globale:.0f}% (n={len(fusion)}) ===\n")

    print("=== Justesse PAR CLASSE ===\n")
    lignes = []
    for classe, nom in NOMS_CLASSES.items():
        sous = fusion[fusion[BASELINE] == classe]
        if sous.empty:
            continue
        justesse = 100 * (sous[BASELINE] == sous["note_humaine"]).mean()
        lignes.append({"Étiquette": nom, "Justesse": round(justesse), "n": len(sous)})

    tableau = pd.DataFrame(lignes).sort_values("n", ascending=False).set_index("Étiquette")
    print(tableau.to_string())

    print("\nMatrice de confusion (étiquette 'accord Haiku-Gemini' -> vérité humaine), "
          "proportions par ligne :")
    print(pd.crosstab(fusion[BASELINE], fusion["note_humaine"], normalize="index").round(2))

    print("\n--- Format Markdown pour l'article ---")
    print("| Étiquette (accord Haiku–Gemini) | Justesse | n |")
    print("|---|---|---|")
    for nom, ligne in tableau.iterrows():
        print(f"| {nom} | {ligne['Justesse']:.0f} % | {ligne['n']} |")


if __name__ == "__main__":
    main()