#!/usr/bin/env python3
"""
Benchmark : comparer TF-IDF + sentiment finance pré-entraîné vs LLMs

Charge le dernier benchmark LLM (V7), fusionne avec les résultats de classifier.py
(mode subset V7), et compares les 3 candidats TF-IDF (note_targeted, note_full, note_tfidf)
à la baseline consensus Gemini+Haiku, en réutilisant les métriques existantes.

Usage :
  python benchmark_classification_tfidf.py
    → Cherche output/benchmark_classification_V7.csv (benchmark LLM)
      et output/comparaison_sentiment_subset_v7_*.csv (dernier run classifier.py)
"""

import os
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
import benchmark_classification as bc


def charger_subset_tfidf():
    """Localise et charge le CSV de comparaison le plus récent."""
    pattern = "output/comparaison_sentiment_subset_v7_*.csv"
    fichiers = sorted(glob.glob(pattern), reverse=True)
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier trouvé correspondant à '{pattern}'. "
            "Avez-vous lancé classifier.py --subset-csv output/benchmark_classification_V7.csv ?"
        )
    fichier = fichiers[0]
    print(f"[benchmark_tfidf] CSV TF-IDF chargé : {fichier}")
    df = pd.read_csv(fichier)
    print(f"  → {len(df)} lignes")
    return df


def assembler_tableau_comparatif(resultats_candidats, resultats_llm_baseline):
    """
    Assemble un tableau récapitulatif unique comparant :
    - Les 3 LLM locaux (sur V7 complet)
    - Les 3 candidats TF-IDF (chacun sur son périmètre valide)

    Colonnes : nom, accuracy, kappa, accuracy_consensus, couverture %, biais_net, % erreurs polaires
    """
    lignes = []

    # 1. Ajouter les 3 LLM locaux d'abord (résultats pré-calculés de benchmark V7)
    for modele, nom in [
        ("note_llama3", "Llama 3.1 (8B)"),
        ("note_mistral", "Mistral Nemo (12B)"),
        ("note_queen", "Qwen 2.5 (7B)"),
    ]:
        if modele in resultats_llm_baseline:
            res = resultats_llm_baseline[modele]
            lignes.append({
                "Modèle": nom,
                "Type": "LLM local",
                "Accuracy (%)": res["accuracy"],
                "Kappa": res["kappa"],
                "Accuracy consensus (%)": res.get("accuracy_consensus", "–"),
                "Couverture (%)": "100.0",
                "Biais net (%)": res.get("biais_net", "–"),
                "Erreurs polaires (%)": res.get("erreurs_polaires", "–"),
            })

    # 2. Ajouter les 3 candidats TF-IDF (résultats du benchmark TF-IDF)
    for candidat, nom in [
        ("note_targeted", "TF-IDF ciblé (bardsai)"),
        ("note_full", "TF-IDF texte complet (bardsai)"),
        ("note_tfidf", "TF-IDF résumé (bardsai)"),
    ]:
        if candidat in resultats_candidats:
            res = resultats_candidats[candidat]
            lignes.append({
                "Modèle": nom,
                "Type": "TF-IDF + HF",
                "Accuracy (%)": res["accuracy"],
                "Kappa": res["kappa"],
                "Accuracy consensus (%)": res.get("accuracy_consensus", "–"),
                "Couverture (%)": res["couverture"],
                "Biais net (%)": res.get("biais_net", "–"),
                "Erreurs polaires (%)": res.get("erreurs_polaires", "–"),
            })

    return pd.DataFrame(lignes)


def main():
    print("=" * 80)
    print("BENCHMARK : TF-IDF + HuggingFace finance vs LLMs")
    print("=" * 80)

    # 1. Charger le benchmark V7 de base (LLM)
    print("\n[1] Chargement du benchmark LLM (V7)...")
    df_v7 = pd.read_csv("output/benchmark_classification_V7.csv")
    print(f"  → {len(df_v7)} lignes")

    # 2. Charger le subset TF-IDF
    print("\n[2] Chargement des résultats TF-IDF (subset V7)...")
    df_subset = charger_subset_tfidf()

    # 3. Merger V7 + subset TF-IDF
    print("\n[3] Fusion V7 + TF-IDF...")
    df_merged = df_v7.merge(
        df_subset[["article_id", "company_id", "note_tfidf", "note_targeted", "note_full", "type"]],
        on=["article_id", "company_id"],
        how="left"
    )
    nb_sans_tfidf = df_merged["note_full"].isna().sum()
    if nb_sans_tfidf > 0:
        print(f"  ⚠ {nb_sans_tfidf} ligne(s) de V7 sans correspondance TF-IDF (texte vide, article absent, etc.)")
    print(f"  → {len(df_merged)} lignes fusionnées")

    # 4. Analyser les données TF-IDF en parallèle (3 candidats, chacun séparément)
    print("\n[4] Analyse des candidats TF-IDF...")
    resultats_candidats = {}
    CANDIDATS = {
        "note_targeted": "TF-IDF ciblé (bardsai finance-fr)",
        "note_full": "TF-IDF texte complet (bardsai finance-fr)",
        "note_tfidf": "TF-IDF résumé (bardsai finance-fr)",
    }

    for col_note, nom_candidat in CANDIDATS.items():
        print(f"\n--- {nom_candidat} ---")
        df_candidat = df_merged.copy()

        # Appliquer les filtres du benchmark classique (status, validité des notes)
        df_candidat = bc.filtrer_statuts_ok(df_candidat)
        if df_candidat.empty:
            print(f"  Aucune ligne valide après filtrage statut.")
            continue

        df_candidat = bc.filtrer_notes_valides(df_candidat, [col_note, "note_gemini"])
        if df_candidat.empty:
            print(f"  Aucune ligne valide après filtrage validité notes.")
            continue

        # Calculer les métriques
        accuracy = accuracy_score(df_candidat["note_gemini"], df_candidat[col_note]) * 100
        kappa = cohen_kappa_score(df_candidat["note_gemini"], df_candidat[col_note])
        couverture = (df_candidat[col_note].notna().sum() / len(df_merged)) * 100

        # Accuracy en zone de consensus (Gemini = Haiku)
        consensus = df_candidat[df_candidat["note_gemini"] == df_candidat["note_haiku"]]
        if not consensus.empty:
            acc_consensus = accuracy_score(consensus["note_gemini"], consensus[col_note]) * 100
        else:
            acc_consensus = None

        # Calcul du biais (sur/sous-notation)
        diff = df_candidat[col_note] - df_candidat["note_gemini"]
        biais_net = diff.mean() * 100  # en points de note approximatifs
        erreurs_polaires = (diff.abs() == 2).sum() / len(df_candidat) * 100

        resultats_candidats[col_note] = {
            "accuracy": round(accuracy, 1),
            "kappa": round(kappa, 3),
            "accuracy_consensus": round(acc_consensus, 1) if acc_consensus is not None else None,
            "couverture": round(couverture, 1),
            "biais_net": round(biais_net, 1),
            "erreurs_polaires": round(erreurs_polaires, 1),
            "n_traite": df_candidat[col_note].notna().sum(),
            "n_total": len(df_merged),
        }

        print(f"  Accuracy: {accuracy:.1f}%, Kappa: {kappa:.3f}")
        print(f"  Couverture: {couverture:.1f}% ({resultats_candidats[col_note]['n_traite']}/{resultats_candidats[col_note]['n_total']})")
        if acc_consensus is not None:
            print(f"  Accuracy en zone consensus: {acc_consensus:.1f}%")

    # 5. Relancer le benchmark classique sur V7 pour avoir les résultats LLM
    print("\n[5] Benchmark LLM baseline (V7)...")
    resultats_llm_baseline = {}

    # Appliquer les filtres classiques
    df_llm = df_merged.copy()
    df_llm = bc.filtrer_par_version(df_llm, "benchmark_classification_V7.csv")
    df_llm = bc.filtrer_statuts_ok(df_llm)
    df_llm = bc.filtrer_notes_valides(df_llm, ["note_llama3", "note_mistral", "note_queen", "note_gemini", "note_haiku"])

    if not df_llm.empty:
        for mod, nom in [("note_llama3", "Llama 3.1"), ("note_mistral", "Mistral"), ("note_queen", "Qwen")]:
            acc = accuracy_score(df_llm["note_gemini"], df_llm[mod]) * 100
            kappa = cohen_kappa_score(df_llm["note_gemini"], df_llm[mod])

            consensus = df_llm[df_llm["note_gemini"] == df_llm["note_haiku"]]
            if not consensus.empty:
                acc_consensus = accuracy_score(consensus["note_gemini"], consensus[mod]) * 100
            else:
                acc_consensus = None

            diff = df_llm[mod] - df_llm["note_gemini"]
            biais_net = diff.mean() * 100
            erreurs_polaires = (diff.abs() == 2).sum() / len(df_llm) * 100

            resultats_llm_baseline[mod] = {
                "accuracy": round(acc, 1),
                "kappa": round(kappa, 3),
                "accuracy_consensus": round(acc_consensus, 1) if acc_consensus is not None else None,
                "biais_net": round(biais_net, 1),
                "erreurs_polaires": round(erreurs_polaires, 1),
            }
            print(f"  {nom}: accuracy={acc:.1f}%, kappa={kappa:.3f}")

    # 6. Assembler le tableau comparatif
    print("\n[6] Tableau comparatif...")
    tableau = assembler_tableau_comparatif(resultats_candidats, resultats_llm_baseline)
    print("\n" + tableau.to_string(index=False))

    # 7. Exporter le tableau
    csv_out = "output/benchmark_comparatif_tfidf_vs_llm.csv"
    tableau.to_csv(csv_out, index=False)
    print(f"\nTableau exporté → {csv_out}")

    # 8. Génération HTML (minimaliste pour l'instant)
    html_out = "output/benchmark_rapport_tfidf_vs_llm.html"
    generer_rapport_html(tableau, html_out)
    print(f"Rapport HTML → {html_out}")


def generer_rapport_html(tableau, chemin):
    """Génère un rapport HTML minimaliste avec le tableau comparatif."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Benchmark TF-IDF vs LLM</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            .llm-local { background: #e8f4f8; }
            .tfidf { background: #e8f8e8; }
            table {
                border-collapse: collapse;
                width: 100%;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th, td {
                border: 1px solid #ccc;
                padding: 10px;
                text-align: left;
            }
            th { background: #333; color: white; }
            tr.llm-local td:first-child { background: #e8f4f8; font-weight: bold; }
            tr.tfidf td:first-child { background: #e8f8e8; font-weight: bold; }
            .note { font-size: 0.9em; color: #666; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Benchmark : TF-IDF vs LLMs</h1>
        <p>Comparaison sur le jeu V7 (1751 lignes, consensus Gemini+Haiku comme baseline).</p>

        <h2>Tableau comparatif</h2>
    """

    for _, row in tableau.iterrows():
        css_class = "llm-local" if row["Type"] == "LLM local" else "tfidf"
        html += f"        <tr class='{css_class}'>\n"
        for col in tableau.columns:
            html += f"            <td>{row[col]}</td>\n"
        html += "        </tr>\n"

    html += """
        </table>

        <div class="note">
            <h3>Notation</h3>
            <ul>
                <li><strong>Accuracy</strong> : % d'exactitude vs Gemini (baseline).</li>
                <li><strong>Kappa</strong> : accord corrigé du hasard (Cohen's kappa).</li>
                <li><strong>Accuracy consensus</strong> : accuracy uniquement sur la zone de consensus (Gemini = Haiku), supposée fiable.</li>
                <li><strong>Couverture (%)</strong> : % de lignes avec une note valide (important pour TF-IDF, pénalisé par note_targeted=9).</li>
                <li><strong>Biais net</strong> : tendance systématique de sur/sous-notation.</li>
                <li><strong>Erreurs polaires (%)</strong> : % de prédictions inversées (0 ↔ 2).</li>
            </ul>
        </div>
    </body>
    </html>
    """

    with open(chemin, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
