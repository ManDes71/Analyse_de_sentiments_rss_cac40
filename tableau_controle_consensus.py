"""
Reproduit l'analyse de l'article, section "Deuxième échantillon : le contrôle" :

1. le constat de départ (Mistral à 82% de NEUTRE contre 39,8% pour Haiku,
   et 38% de mouvements intraday dans l'échantillon de la zone grise) ;
2. le verdict sur les 24 cas litigieux de l'échantillon de contrôle
   (l'humain donne raison aux références ou à Mistral) ;
3. le tableau final : justesse de chaque modèle local en zone grise vs en
   zone de consensus, plus l'accord Haiku-Gemini sur ce second échantillon.

Entrées attendues :
  1. CSV corpus (cherche d'abord dans output/)
     - output/benchmark_classification_VX.csv
     - output/resultats_*.csv
  2. Excel zone grise (cherche dans output/)
     - output/zone_grise_a_annoter.xlsx
  3. Excel contrôle (cherche dans output/)
     - output/controle_consensus_a_annoter.xlsx

Usage :
    python tableau_controle_consensus.py --version v7
    python tableau_controle_consensus.py --version v7 \
      --corpus output/resultats_haiku_v7.csv \
      --zone-grise output/zone_grise_a_annoter.xlsx \
      --controle output/controle_consensus_a_annoter.xlsx
"""
import argparse

import pandas as pd

MODELES = {
    "note_mistral": "Mistral Nemo",
    "note_queen":   "Qwen 2.5",
    "note_llama3":  "Llama 3.1",
}
COLONNE_STATUT = {
    "note_mistral": "statut_mistral",
    "note_queen":   "statut_queen",
    "note_llama3":  "statut_lama",
}


def charger_corpus(chemin, version):
    df = pd.read_csv(chemin)
    if "prompt_version_lama" in df.columns:
        df = df[df["prompt_version_lama"] == version]
    return df


def charger_annotations(chemin):
    ann = pd.read_excel(chemin)
    ann = ann.dropna(subset=["note_humaine"])
    ann = ann[ann["note_humaine"] != 1.5].copy()
    ann["note_humaine"] = ann["note_humaine"].astype(int)
    return ann


def etape_1_constat(corpus):
    """Distribution des notes : le chiffre qui a motivé l'échantillon de contrôle."""
    print("=== 1. LE CONSTAT DE DÉPART ===\n")
    for col, nom in [("note_haiku", "Claude Haiku"), ("note_mistral", "Mistral Nemo")]:
        df = corpus
        statut = "statut_" + col.split("_")[1]
        if statut in df.columns:
            df = df[df[statut] == "ok"]
        d = df[col].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0) * 100
        print(f"  {nom:<14} NEG={d[0]:5.1f}%  NEU={d[1]:5.1f}%  POS={d[2]:5.1f}%")
    print()


def etape_1bis_poids_intraday(zone_grise):
    """Part des cas de mouvement intraday dans l'échantillon de la zone grise."""
    com = zone_grise["Ma conclusion"].fillna("").str.lower()
    intraday = com.str.contains("intraday")
    print(f"  Cas de mouvement intraday dans l'échantillon zone grise : "
          f"{intraday.sum()}/{len(zone_grise)} ({100 * intraday.mean():.0f}%)\n")


def etape_2_verdict(controle):
    """Verdict sur les cas litigieux : la règle/Mistral contredit-elle à raison
    ou à tort le consensus Haiku-Gemini ?"""
    print("=== 2. VERDICT SUR LES CAS LITIGIEUX ===\n")
    if "strate" not in controle.columns:
        print("  Colonne 'strate' absente : impossible d'isoler les cas litigieux.")
        return
    lit = controle[controle["strate"] == "regle_contre_consensus"]
    raison_mistral = (lit["note_humaine"] == lit["note_mistral"]).sum()
    raison_consensus = (lit["note_humaine"] == lit["note_haiku"]).sum()
    print(f"  Cas litigieux (n={len(lit)}) : Mistral contredit le consensus Haiku-Gemini.")
    print(f"    humain donne raison aux références : {raison_consensus}")
    print(f"    humain donne raison à Mistral       : {raison_mistral}\n")


def justesse(fusion, colonne):
    sous = fusion[fusion[colonne].isin([0, 1, 2])]
    if sous.empty:
        return None
    return round(100 * (sous[colonne] == sous["note_humaine"]).mean(), 0)


def etape_3_tableau(corpus, zone_grise, controle):
    """Le tableau final : justesse par modèle, zone grise vs zone de consensus,
    plus l'accord Haiku-Gemini sur le second échantillon."""
    print("=== 3. TABLEAU FINAL ===\n")

    fusion_grise = corpus.merge(
        zone_grise[["article_id", "company_id", "note_humaine"]],
        on=["article_id", "company_id"], how="inner")
    fusion_controle = corpus.merge(
        controle[["article_id", "company_id", "note_humaine"]],
        on=["article_id", "company_id"], how="inner")

    lignes = []
    for col, nom in MODELES.items():
        lignes.append({
            "Modèle": nom,
            f"Zone grise ({len(fusion_grise)} cas)": justesse(fusion_grise, col),
            f"Zone de consensus ({len(fusion_controle)} cas)": justesse(fusion_controle, col),
        })

    # Accord Haiku-Gemini : la zone de consensus est PAR CONSTRUCTION celle où
    # Haiku et Gemini sont d'accord entre eux (accord mutuel = 100%, sans intérêt
    # ici). Le chiffre qui compte est leur JUSTESSE FACE À L'ANNOTATION HUMAINE
    # sur cet échantillon : comme note_haiku == note_gemini par construction,
    # justesse(fusion_controle, "note_haiku") == justesse(..., "note_gemini").
    accord_controle = justesse(fusion_controle, "note_haiku")
    lignes.append({
        "Modèle": "Accord Haiku–Gemini",
        f"Zone grise ({len(fusion_grise)} cas)": None,
        f"Zone de consensus ({len(fusion_controle)} cas)": accord_controle,
    })

    tableau = pd.DataFrame(lignes).set_index("Modèle")
    print(tableau.to_string(na_rep="—"))

    print("\n--- Format Markdown pour l'article ---")
    cols = list(tableau.columns)
    print(f"| Modèle | Zone grise ({len(fusion_grise)} cas) | "
          f"Zone de consensus ({len(fusion_controle)} cas) |")
    print("|---|---|---|")
    for nom, ligne in tableau.iterrows():
        v1 = "—" if pd.isna(ligne[cols[0]]) else f"{ligne[cols[0]]:.0f} %"
        v2 = "—" if pd.isna(ligne[cols[1]]) else f"{ligne[cols[1]]:.0f} %"
        print(f"| {nom} | {v1} | {v2} |")


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
        zone_grise = charger_annotations(chemin_zone_grise)
        controle = charger_annotations(chemin_controle)
    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
        print(f"\nCherche :")
        print(f"  Corpus : {chemin_corpus}")
        print(f"  Zone grise : {chemin_zone_grise}")
        print(f"  Contrôle : {chemin_controle}")
        return 1

    print(f"Version étudiée : {args.version} | corpus : {len(corpus)} lignes\n")

    etape_1_constat(corpus)
    etape_1bis_poids_intraday(zone_grise)
    etape_2_verdict(controle)
    etape_3_tableau(corpus, zone_grise, controle)


if __name__ == "__main__":
    main()