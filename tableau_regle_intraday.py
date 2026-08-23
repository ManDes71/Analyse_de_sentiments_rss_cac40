"""
Reproduit le tableau de l'article :
"Résultat sur les 38 cas de mouvement intraday annotés à la main"
(comparaison v3 -- sans la règle de mouvement de cours -- vs v5 -- avec la règle).

Entrées attendues dans le répertoire courant :
    benchmark_classification_V3.csv
    benchmark_classification_V5.csv
    zone_grise_a_annoter.xlsx   (colonne 'note_humaine' remplie, colonne
                                  'Ma conclusion' contenant le mot-clé "intraday"
                                  sur les cas concernés)

Usage :
    python tableau_regle_intraday.py
"""
import pandas as pd

FICHIER_V3 = "output/benchmark_classification_V3.csv"
FICHIER_V5 = "output/benchmark_classification_V5.csv"
FICHIER_ANNOTATIONS = "output/zone_grise_a_annoter.xlsx"

MODELES = {
    "note_haiku":   "Claude Haiku",
    "note_gemini":  "Gemini 2.5 Flash",
    "note_llama3":  "Llama 3.1 (8B)",
    "note_queen":   "Qwen 2.5 (7B)",
}


def charger_verite_terrain(chemin):
    """
    Charge les annotations REMOVEDelles et isole les cas de mouvement de cours
    intraday, identifiés par le mot-clé "intraday" dans le commentaire libre.
    Exclut les notes non entières (ex: 1.5, laissée pour un cas jugé ambigu).
    """
    ann = pd.read_excel(chemin)
    ann = ann.dropna(subset=["note_humaine"])
    ann = ann[ann["note_humaine"] != 1.5].copy()
    ann["note_humaine"] = ann["note_humaine"].astype(int)
    ann["intraday"] = ann["Ma conclusion"].fillna("").str.lower().str.contains("intraday")
    return ann[ann["intraday"]][["article_id", "company_id", "note_humaine"]]


def charger_run(chemin, version):
    """Charge un export CSV et le restreint à la version de prompt demandée."""
    df = pd.read_csv(chemin)
    if "prompt_version_lama" in df.columns:
        df = df[df["prompt_version_lama"] == version]
    for col in ("statut_lama", "statut_mistral", "statut_queen",
                "statut_gemini", "statut_haiku"):
        if col in df.columns:
            df = df[df[col] == "ok"]
    return df


def justesse_par_modele(df, verite):
    """Calcule, pour chaque modèle, le taux d'accord avec la vérité terrain."""
    fusion = df.merge(verite, on=["article_id", "company_id"], how="inner")
    resultats = {}
    for colonne, nom in MODELES.items():
        sous = fusion[fusion[colonne].isin([0, 1, 2])]
        if len(sous) == 0:
            resultats[nom] = None
            continue
        resultats[nom] = round(100 * (sous[colonne] == sous["note_humaine"]).mean(), 0)
    return resultats, len(fusion)


def main():
    verite = charger_verite_terrain(FICHIER_ANNOTATIONS)
    print(f"Cas de mouvement intraday annotés : {len(verite)}")
    print(f"Répartition de la vérité humaine : "
          f"{verite['note_humaine'].value_counts().sort_index().to_dict()}\n")

    df_v3 = charger_run(FICHIER_V3, "v3")
    df_v5 = charger_run(FICHIER_V5, "v5")

    res_v3, n_v3 = justesse_par_modele(df_v3, verite)
    res_v5, n_v5 = justesse_par_modele(df_v5, verite)

    print(f"Cas retrouvés dans le run v3 : {n_v3}")
    print(f"Cas retrouvés dans le run v5 : {n_v5}\n")

    tableau = pd.DataFrame({
        "v3 (sans la règle)": res_v3,
        "v5 (avec la règle)": res_v5,
    })
    tableau.index.name = "Modèle"
    print(tableau.to_string())

    print("\n--- Format Markdown pour l'article ---")
    print("| Modèle | v3 (sans la règle) | v5 (avec la règle) |")
    print("|---|---|---|")
    for nom in tableau.index:
        v3 = tableau.loc[nom, "v3 (sans la règle)"]
        v5 = tableau.loc[nom, "v5 (avec la règle)"]
        print(f"| {nom} | {v3:.0f} % | {v5:.0f} % |")


if __name__ == "__main__":
    main()