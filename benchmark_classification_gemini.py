#C:\Users\DESPLANCHES\Documents\CODES\WINDEV\REMOVED\post blog
#mon_env\Scripts\activate
#python -m pip install scikit-learn
#python benchmark_classification.py

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

# ==========================================
# CONFIGURATION
# ==========================================
NOTES_VALIDES = [0, 1, 2]

BASELINE = "note_gemini"              # référence principale
REFERENCE_SECONDAIRE = "note_haiku"   # 2e "master", pour isoler la zone grise

MODELES_LOCAUX = ["note_llama3", "note_mistral", "note_queen"]

NOMS_PROPRES = {
    "note_llama3": "Llama 3.1 (8B)",
    "note_mistral": "Mistral Nemo (12B)",
    "note_queen": "Qwen 2.5 (7B)",
    "note_haiku": "Claude Haiku",
    "note_gemini": "Gemini 2.5 Flash",
}

# Colonne de statut associée à chaque colonne de note. Indispensable : une ligne
# 'failed' peut conserver une note périmée issue d'un run précédent, qui serait
# alors comptée à tort dans les métriques de la version analysée.
COLONNE_STATUT = {
    "note_llama3": "statut_lama",
    "note_mistral": "statut_mistral",
    "note_queen": "statut_queen",
    "note_haiku": "statut_haiku",
    "note_gemini": "statut_gemini",
}


# ==========================================
# CHARGEMENT ET NETTOYAGE
# ==========================================
def filtrer_par_version(df, nom_fichier):
    """
    Filtre le DataFrame sur la version de prompt déduite du nom de fichier.

    La détection se fait par expression régulière (et non sur une liste figée) :
    une liste codée en dur devient silencieusement obsolète dès qu'une nouvelle
    version de prompt est créée, et l'analyse porte alors sur un mélange de
    versions sans le signaler autrement que par une ligne de log discrète.
    """
    correspondance = re.search(r"[_-][Vv](\d+)", nom_fichier)
    if not correspondance:
        print("ATTENTION : aucune version détectée dans le nom du fichier.")
        print("  L'analyse va porter sur TOUTES les versions de prompt présentes,")
        print("  donc mélanger des notes non comparables. Nommez le fichier")
        print("  'benchmark_classification_V4.csv' pour filtrer sur la v4.")
        return df

    cible = f"v{correspondance.group(1)}"
    if "prompt_version_lama" not in df.columns:
        print(f"Version '{cible}' détectée mais colonne 'prompt_version_lama' absente : pas de filtrage.")
        return df

    avant = len(df)
    df = df[df["prompt_version_lama"] == cible]
    print(f"Filtrage appliqué : prompt_version_lama = '{cible}' "
          f"({avant} -> {len(df)} lignes).")

    # Contrôle de cohérence : tous les modèles doivent avoir été évalués avec la
    # MÊME version, sinon on compare des notes produites par des prompts différents.
    # Les valeurs NULL comptent comme une incohérence : une colonne de version vide
    # signifie que le modèle n'a pas été relancé depuis l'introduction du suivi de
    # version, donc que ses notes viennent d'un prompt inconnu.
    colonnes_version = [c for c in df.columns if c.startswith("prompt_version_")]
    incoherences = {}
    for col in colonnes_version:
        autres = sorted(str(v) for v in df[col].unique() if str(v) != cible and pd.notna(v))
        nb_nuls = int(df[col].isna().sum())
        if nb_nuls:
            autres.append(f"{nb_nuls} valeur(s) NULL")
        if autres:
            incoherences[col] = autres
    if incoherences:
        print("\nATTENTION : versions de prompt hétérogènes entre modèles !")
        for col, versions in incoherences.items():
            modele = col.replace("prompt_version_", "")
            print(f"  {modele} : contient aussi {versions} au lieu de '{cible}' uniquement")
        print("  Les modèles n'ont pas tous été relancés dans cette version.")
        print("  Comparer leurs notes revient à comparer des prompts différents.\n")
    else:
        print(f"Cohérence vérifiée : les {len(colonnes_version)} modèles sont tous en '{cible}'.")

    return df


def filtrer_statuts_ok(df):
    """
    Ne conserve que les lignes où TOUS les modèles comparés ont un statut 'ok'.

    Sans ce filtre, les lignes 'failed' qui ont gardé une note d'un run antérieur
    polluent silencieusement l'analyse : la note appartient à une autre version de
    prompt (voire à un autre modèle) que celle qu'on croit mesurer.
    """
    avant = len(df)
    colonnes_presentes = [
        COLONNE_STATUT[m]
        for m in MODELES_LOCAUX + [BASELINE, REFERENCE_SECONDAIRE]
        if COLONNE_STATUT.get(m) in df.columns
    ]
    if not colonnes_presentes:
        print("Aucune colonne de statut trouvée : filtrage des statuts ignoré.")
        return df

    masque = pd.Series(True, index=df.index)
    for col in colonnes_presentes:
        masque &= df[col] == "ok"

    df_ok = df[masque]
    exclus = avant - len(df_ok)
    if exclus:
        print(
            f"Filtrage des statuts : {exclus} ligne(s) exclue(s) "
            f"(statut != 'ok', note potentiellement périmée). Reste {len(df_ok)}."
        )
    else:
        print(f"Filtrage des statuts : aucune ligne exclue ({len(df_ok)} lignes).")
    return df_ok


def filtrer_notes_valides(df, colonnes):
    """Exclut les lignes dont une des notes est hors de {0,1,2} ou manquante."""
    avant = len(df)
    masque = pd.Series(True, index=df.index)
    for col in colonnes:
        masque &= df[col].isin(NOTES_VALIDES)
    df_valide = df[masque]
    exclus = avant - len(df_valide)
    if exclus:
        print(f"Notes hors plage/manquantes : {exclus} ligne(s) exclue(s). Reste {len(df_valide)}.")
    return df_valide


# ==========================================
# ANALYSES
# ==========================================
def analyser_robustesse(df):
    print("\n=== ANALYSE DE LA ROBUSTESSE (JSON) ===")
    for mod in MODELES_LOCAUX:
        hors_limite = df[~df[mod].isin(NOTES_VALIDES)][mod].count()
        manquantes = df[mod].isna().sum()
        print(
            f"{NOMS_PROPRES[mod]} : {hors_limite} valeur(s) aberrante(s), "
            f"{manquantes} manquante(s)."
        )


def analyser_accord(df):
    """
    Accuracy ET kappa. L'accuracy seule surestime l'accord quand les classes sont
    déséquilibrées ; kappa corrige la part d'accord due au hasard.
    """
    print(f"\n=== ACCORD vs {NOMS_PROPRES[BASELINE]} (accuracy + kappa) ===")
    scores = {}
    for mod in MODELES_LOCAUX:
        acc = accuracy_score(df[BASELINE], df[mod]) * 100
        kappa = cohen_kappa_score(df[BASELINE], df[mod])
        scores[NOMS_PROPRES[mod]] = acc
        print(f"{NOMS_PROPRES[mod]:<22} accuracy={acc:>5.2f}%  kappa={kappa:.3f}")
    return scores


def analyser_gravite_erreurs(df):
    """
    Toutes les erreurs ne se valent pas : confondre NEUTRE et POSITIF est bénin,
    confondre NÉGATIF et POSITIF est une erreur de signe, bien plus grave.
    """
    print("\n=== GRAVITÉ DES ERREURS (vs baseline) ===")
    print(f"{'Modèle':<22} {'Exact':>8} {'Adjacent':>10} {'Polaire':>9}")
    for mod in MODELES_LOCAUX:
        ecart = (df[mod] - df[BASELINE]).abs()
        exact = 100 * (ecart == 0).mean()
        adjacent = 100 * (ecart == 1).mean()
        polaire = 100 * (ecart == 2).mean()
        print(f"{NOMS_PROPRES[mod]:<22} {exact:>7.1f}% {adjacent:>9.1f}% {polaire:>8.1f}%")
    print("  (Polaire = 0<->2 : le modèle inverse le signe du sentiment.)")


def analyser_sens_biais(df):
    print("\n=== SENS DU BIAIS ===")
    for mod in MODELES_LOCAUX:
        diff = df[mod] - df[BASELINE]
        print(
            f"{NOMS_PROPRES[mod]:<22} sous-note={100 * (diff < 0).mean():>5.1f}%  "
            f"sur-note={100 * (diff > 0).mean():>5.1f}%"
        )


def analyser_par_classe(df):
    print("\n=== PRÉCISION / RAPPEL / F1 PAR CLASSE ===")
    for mod in MODELES_LOCAUX:
        print(f"\n--- {NOMS_PROPRES[mod]} ---")
        print(
            classification_report(
                df[BASELINE],
                df[mod],
                labels=NOTES_VALIDES,
                target_names=["NEG", "NEU", "POS"],
                zero_division=0,
                digits=3,
            )
        )
        matrice = confusion_matrix(df[BASELINE], df[mod], labels=NOTES_VALIDES)
        print("Matrice de confusion (lignes = baseline, colonnes = modèle) :")
        print(
            pd.DataFrame(
                matrice,
                index=["ref:NEG", "ref:NEU", "ref:POS"],
                columns=["mod:NEG", "mod:NEU", "mod:POS"],
            )
        )


def analyser_distribution(df):
    print("\n=== BIAIS DE PRUDENCE (distribution des classes) ===")
    colonnes = [BASELINE] + MODELES_LOCAUX
    distrib = df[colonnes].apply(lambda x: x.value_counts(normalize=True)).T * 100
    for val in NOTES_VALIDES:
        if val not in distrib.columns:
            distrib[val] = 0.0
    distrib = distrib[NOTES_VALIDES]
    distrib.columns = ["0 (Négatif)", "1 (Neutre)", "2 (Positif)"]
    distrib.index = [NOMS_PROPRES[c] for c in colonnes]
    print(distrib.round(1))
    return distrib


def analyser_zones_reference(df):
    """
    AXE CLÉ : la baseline n'est fiable que là où les deux modèles "master"
    (Haiku et Gemini) sont d'accord. Ailleurs, l'accord d'un modèle local avec
    Haiku peut n'être qu'un artefact du choix de baseline.
    """
    if REFERENCE_SECONDAIRE not in df.columns:
        print(f"\n[Zone grise] {REFERENCE_SECONDAIRE} absent : analyse ignorée.")
        return None

    print("\n=== ZONES DE CONFIANCE (double référence) ===")
    accord_ref = accuracy_score(df[BASELINE], df[REFERENCE_SECONDAIRE]) * 100
    kappa_ref = cohen_kappa_score(df[BASELINE], df[REFERENCE_SECONDAIRE])
    print(
        f"Accord entre les 2 références ({NOMS_PROPRES[BASELINE]} vs "
        f"{NOMS_PROPRES[REFERENCE_SECONDAIRE]}) : {accord_ref:.1f}% (kappa={kappa_ref:.3f})"
    )

    consensus = df[df[BASELINE] == df[REFERENCE_SECONDAIRE]]
    zone_grise = df[df[BASELINE] != df[REFERENCE_SECONDAIRE]]

    print(f"\n--- Zone de CONSENSUS (n={len(consensus)}, {100 * len(consensus) / len(df):.1f}%) ---")
    print("Métriques fiables : les deux références s'accordent.")
    for mod in MODELES_LOCAUX:
        acc = accuracy_score(consensus[BASELINE], consensus[mod]) * 100
        kappa = cohen_kappa_score(consensus[BASELINE], consensus[mod])
        print(f"{NOMS_PROPRES[mod]:<22} accuracy={acc:>5.1f}%  kappa={kappa:.3f}")

    print(f"\n--- ZONE GRISE (n={len(zone_grise)}) ---")
    print("Cas où les références se contredisent : de quel côté penche chaque modèle ?")
    for mod in MODELES_LOCAUX:
        suit_baseline = 100 * (zone_grise[mod] == zone_grise[BASELINE]).mean()
        suit_secondaire = 100 * (zone_grise[mod] == zone_grise[REFERENCE_SECONDAIRE]).mean()
        print(
            f"{NOMS_PROPRES[mod]:<22} suit {NOMS_PROPRES[BASELINE]}={suit_baseline:>5.1f}%  "
            f"suit {NOMS_PROPRES[REFERENCE_SECONDAIRE]}={suit_secondaire:>5.1f}%"
        )
    print(
        "  Note : un score 'vs baseline' élevé peut refléter une proximité de style\n"
        "  avec la baseline plutôt qu'une meilleure justesse. Seule une annotation\n"
        "  REMOVEDelle de la zone grise permet de trancher (cf. export CSV ci-dessous)."
    )
    return zone_grise


def analyser_par_tranche(df, colonne, libelle, quantiles=4):
    """Croise la performance avec une variable continue (nbocc, longueur...)."""
    if colonne not in df.columns:
        print(f"\n[{libelle}] Colonne '{colonne}' absente du fichier : analyse ignorée.")
        return

    sous_df = df[df[colonne].notna()].copy()
    if sous_df.empty:
        print(f"\n[{libelle}] Aucune donnée exploitable.")
        return

    print(f"\n=== PERFORMANCE PAR TRANCHE DE {libelle.upper()} ===")
    try:
        sous_df["_tranche"] = pd.qcut(sous_df[colonne], q=quantiles, duplicates="drop")
    except ValueError:
        print(f"Impossible de découper '{colonne}' en tranches (valeurs trop concentrées).")
        return

    lignes = []
    for tranche, groupe in sous_df.groupby("_tranche", observed=True):
        ligne = {"tranche": str(tranche), "n": len(groupe)}
        for mod in MODELES_LOCAUX:
            ligne[NOMS_PROPRES[mod]] = round(
                accuracy_score(groupe[BASELINE], groupe[mod]) * 100, 1
            )
        lignes.append(ligne)
    print(pd.DataFrame(lignes).to_string(index=False))

    # Corrélation : la performance dépend-elle vraiment de cette variable ?
    print(f"\nCorrélation entre '{colonne}' et l'accord avec la baseline :")
    for mod in MODELES_LOCAUX:
        correct = (sous_df[mod] == sous_df[BASELINE]).astype(int)
        if correct.nunique() < 2:
            print(f"{NOMS_PROPRES[mod]:<22} (pas de variance, corrélation non calculable)")
            continue
        correlation = correct.corr(sous_df[colonne])
        print(f"{NOMS_PROPRES[mod]:<22} r={correlation:+.3f}")
    print("  (|r| < 0.1 : effet négligeable, quelle que soit l'apparence du tableau.)")


def analyser_par_entreprise(df, seuil_mini=15, top_n=10):
    """
    Si les erreurs se concentrent sur quelques entreprises, le problème vient
    probablement du filtrage/des alias (mauvaise détection de l'entité) plutôt
    que de la capacité du modèle à juger un sentiment.
    """
    if "company_id" not in df.columns:
        print("\n[Par entreprise] Colonne 'company_id' absente : analyse ignorée.")
        return

    print(f"\n=== ENTREPRISES LES PLUS PROBLÉMATIQUES (>= {seuil_mini} articles) ===")
    lignes = []
    for company_id, groupe in df.groupby("company_id"):
        if len(groupe) < seuil_mini:
            continue
        ligne = {"company_id": company_id, "n": len(groupe)}
        accuracies = []
        for mod in MODELES_LOCAUX:
            acc = accuracy_score(groupe[BASELINE], groupe[mod]) * 100
            ligne[NOMS_PROPRES[mod]] = round(acc, 1)
            accuracies.append(acc)
        ligne["moyenne"] = round(sum(accuracies) / len(accuracies), 1)
        lignes.append(ligne)

    if not lignes:
        print(f"Aucune entreprise avec au moins {seuil_mini} articles.")
        return

    resultat = pd.DataFrame(lignes).sort_values("moyenne")
    print(resultat.head(top_n).to_string(index=False))
    print("\n  Une entreprise nettement sous la moyenne générale signale plutôt un")
    print("  problème d'alias/de détection d'entité qu'une faiblesse du modèle.")


def exporter_zone_grise(zone_grise, chemin="zone_grise_gemini_a_annoter.csv", taille=100):
    """
    Exporte un échantillon stratifié de la zone grise pour annotation REMOVEDelle.
    C'est la seule façon de transformer ce benchmark d'un simple "accord
    inter-modèles" en une véritable mesure de justesse.
    """
    if zone_grise is None or zone_grise.empty:
        print("\n[Export] Zone grise vide : pas d'export.")
        return

    taille = min(taille, len(zone_grise))
    # Échantillon stratifié sur la note de la baseline, pour couvrir les 3 classes.
    # On collecte les index plutôt que d'utiliser groupby().apply(), qui consomme
    # la colonne de groupage et la ferait disparaître de l'export.
    fraction = taille / len(zone_grise)
    index_retenus = []
    for _, groupe in zone_grise.groupby(BASELINE):
        n_tirage = min(len(groupe), max(1, round(len(groupe) * fraction)))
        index_retenus.extend(groupe.sample(n_tirage, random_state=42).index)
    echantillon = zone_grise.loc[index_retenus]

    colonnes_export = ["article_id", "company_id"]
    if "nbocc" in echantillon.columns:
        colonnes_export.append("nbocc")
    colonnes_export += [BASELINE, REFERENCE_SECONDAIRE] + MODELES_LOCAUX
    for col in ("justification_haiku", "justification_gemini"):
        if col in echantillon.columns:
            colonnes_export.append(col)

    export = echantillon[colonnes_export].copy()
    export["note_humaine"] = ""  # colonne à remplir à la main (0, 1 ou 2)
    export.to_csv(chemin, index=False, encoding="utf-8")
    print(f"\n[Export] {len(export)} cas ambigus exportés vers '{chemin}'.")
    print("  Remplir la colonne 'note_humaine' permettra de déterminer laquelle des")
    print("  deux références est la plus juste, et de corriger le classement final.")


# ==========================================
# VISUALISATIONS
# ==========================================
def tracer_graphiques(acc_scores, distrib):
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=list(acc_scores.keys()),
        y=list(acc_scores.values()),
        hue=list(acc_scores.keys()),
        palette="viridis",
        legend=False,
    )
    plt.title(f"Taux d'accord exact avec l'API {NOMS_PROPRES[BASELINE]}", fontsize=14)
    plt.ylabel("Précision (%)")
    plt.ylim(0, 100)
    for i, v in enumerate(acc_scores.values()):
        plt.text(i, v + 2, f"{v:.1f}%", color="black", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("taux_accord_modeles.png")
    plt.close()

    distrib.plot(
        kind="bar",
        stacked=True,
        color=["#e63946", "#f1faee", "#457b9d"],
        edgecolor="black",
        figsize=(10, 6),
    )
    plt.title("Distribution des Notes : Le Biais de Prudence des Modèles Locaux", fontsize=14)
    plt.ylabel("Pourcentage (%)")
    plt.xticks(rotation=0)
    plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("distribution_biais.png")
    plt.close()

    print("\nGraphiques sauvegardés : 'taux_accord_modeles.png' et 'distribution_biais.png'")


# ==========================================
# POINT D'ENTRÉE
# ==========================================
def analyser_benchmark_llm(csv_path="benchmark_classification_V2.csv"):
    if not os.path.exists(csv_path):
        print(f"Fichier {csv_path} introuvable.")
        return

    print(csv_path)
    df = pd.read_csv(csv_path)

    df = filtrer_par_version(df, os.path.basename(csv_path))
    if df.empty:
        print("Le DataFrame est vide après le filtrage. Vérifiez vos données.")
        return

    # Robustesse mesurée AVANT nettoyage : on veut justement compter les anomalies.
    analyser_robustesse(df)

    # Nettoyage : statuts d'abord (notes périmées), puis plages de valeurs.
    df = filtrer_statuts_ok(df)
    colonnes_notes = MODELES_LOCAUX + [BASELINE]
    if REFERENCE_SECONDAIRE in df.columns:
        colonnes_notes.append(REFERENCE_SECONDAIRE)
    df = filtrer_notes_valides(df, colonnes_notes)

    if df.empty:
        print("Plus aucune ligne exploitable après nettoyage.")
        return
    print(f"\nÉchantillon d'analyse final : {len(df)} lignes.")

    acc_scores = analyser_accord(df)
    analyser_gravite_erreurs(df)
    analyser_sens_biais(df)
    analyser_par_classe(df)
    distrib = analyser_distribution(df)

    zone_grise = analyser_zones_reference(df)

    analyser_par_tranche(df, "nbocc", "nombre d'occurrences")
    analyser_par_tranche(df, "longueur_mots", "longueur de l'article (mots)")

    analyser_par_entreprise(df)
    exporter_zone_grise(zone_grise)

    tracer_graphiques(acc_scores, distrib)


if __name__ == "__main__":
    analyser_benchmark_llm()