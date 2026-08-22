"""
Prepare un echantillon de controle a annoter, tire de la ZONE DE CONSENSUS.

Pourquoi : les 100 premiers cas annotes venaient de la zone GRISE v2, ou les
mouvements intraday representaient 38% des cas. Mistral y obtient 79% de
justesse (meilleur score tous modeles confondus), mais il produit 82% de notes
NEUTRE sur l'ensemble du corpus et s'ecarte du consensus Haiku/Gemini sur 47%
des lignes -- dont 88% imposees par la regle deterministe de note_finale().

Deux lectures possibles, que seule une annotation peut departager :
  (a) la regle a raison, et les deux references sur-notent les mouvements de
      cours -> Mistral est reellement le meilleur ;
  (b) la regle se declenche a tort hors zone grise -> Mistral sur-neutralise et
      son score sur la zone grise n'etait qu'un artefact d'echantillonnage.

L'echantillon est donc stratifie sur le CAS LITIGIEUX, pas au hasard : on tire
en priorite les lignes ou la regle contredit le consensus, puisque ce sont elles
qui portent l'information. Un tirage uniforme aurait dilue le signal.

Usage :
    python preparer_controle_consensus.py benchmark_classification_V7.csv
    python preparer_controle_consensus.py benchmark_classification_V7.csv --taille 40
"""
import argparse
import json
import os

import pandas as pd

MODELES = {
    "note_llama3": "extraction_lama",
    "note_mistral": "extraction_mistral",
    "note_queen": "extraction_queen",
}


def charger(csv_path, version=None):
    df = pd.read_csv(csv_path)
    if version is None:
        base = os.path.basename(csv_path)
        if "_V" in base:
            version = "v" + base.split("_V")[1].split(".")[0]
    if version and "prompt_version_lama" in df.columns:
        df = df[df["prompt_version_lama"] == version]
        print(f"Version retenue : {version} ({len(df)} lignes)")

    for col in ("statut_lama", "statut_mistral", "statut_queen",
                "statut_gemini", "statut_haiku"):
        if col in df.columns:
            df = df[df[col] == "ok"]
    return df


def extraire(df, colonne):
    """Deplie la colonne JSONB d'extraction en colonnes plates."""
    faits = df[colonne].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    return pd.DataFrame(list(faits), index=df.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="benchmark_classification_VX.csv")
    p.add_argument("--taille", type=int, default=40)
    p.add_argument("--sortie", default="controle_consensus_a_annoter.csv")
    p.add_argument("--version", default=None)
    args = p.parse_args()

    df = charger(args.csv, args.version)

    consensus = df[df["note_haiku"] == df["note_gemini"]].copy()
    print(f"Zone de consensus : {len(consensus)} lignes "
          f"({100 * len(consensus) / len(df):.0f}% du corpus)")

    ex = extraire(consensus, "extraction_mistral")
    consensus["mistral_autre_fait"] = ex.get("autre_fait_concret")
    consensus["mistral_variation"] = ex.get("variation_cours_pct")
    consensus["mistral_reco"] = ex.get("reco_sens")

    # La regle deterministe s'est-elle appliquee chez Mistral ?
    regle = (consensus["mistral_autre_fait"] == False) & (
        consensus["mistral_reco"].fillna("aucune") == "aucune")
    desaccord = consensus["note_mistral"] != consensus["note_haiku"]

    # Trois strates, par valeur informative decroissante.
    strates = {
        # LE cas litigieux : la regle impose une note contre le consensus.
        "regle_contre_consensus": consensus[regle & desaccord],
        # Controle negatif : la regle s'applique ET confirme le consensus.
        "regle_avec_consensus": consensus[regle & ~desaccord],
        # Temoin : la regle ne joue pas, Mistral juge seul et diverge.
        "modele_contre_consensus": consensus[~regle & desaccord],
    }

    print("\nStrates disponibles :")
    for nom, sous in strates.items():
        print(f"  {nom:<26} {len(sous):>5} lignes")

    # Repartition : on charge l'echantillon sur le cas litigieux.
    quotas = {
        "regle_contre_consensus": int(args.taille * 0.60),
        "regle_avec_consensus": int(args.taille * 0.20),
        "modele_contre_consensus": args.taille - int(args.taille * 0.60) - int(args.taille * 0.20),
    }

    morceaux = []
    for nom, sous in strates.items():
        n = min(quotas[nom], len(sous))
        if n == 0:
            continue
        tire = sous.sample(n, random_state=42).copy()
        tire["strate"] = nom
        morceaux.append(tire)

    echantillon = pd.concat(morceaux).sample(frac=1, random_state=42)  # melange

    colonnes = ["article_id", "company_id", "strate"]
    if "nbocc" in echantillon.columns:
        colonnes.append("nbocc")
    colonnes += ["note_haiku", "note_gemini", "note_llama3", "note_mistral", "note_queen",
                 "mistral_variation", "mistral_autre_fait", "mistral_reco"]
    for c in ("justification_haiku", "justification_mistral"):
        if c in echantillon.columns:
            colonnes.append(c)

    export = echantillon[colonnes].copy()
    export["note_humaine"] = ""
    export["commentaire"] = ""
    export.to_csv(args.sortie, index=False, encoding="utf-8")

    print(f"\n{len(export)} cas exportes vers '{args.sortie}'.")
    print("Repartition :")
    print(export["strate"].value_counts().to_string())
    print("\nNote de consensus (Haiku=Gemini) :",
          export["note_haiku"].value_counts().sort_index().to_dict())
    print("Note de Mistral                 :",
          export["note_mistral"].value_counts().sort_index().to_dict())
    print("\nQuestion a trancher pour chaque ligne : la note de Mistral (souvent 1)")
    print("est-elle plus juste que celle du consensus Haiku/Gemini ?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
