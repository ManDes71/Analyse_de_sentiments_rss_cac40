#cd C:\Users\DESPLANCHES\Documents\CODES\WINDEV
#mon_env\Scripts\activate
#python -m pip install scikit-learn
#python benchmark_classification.py



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import os

def analyser_benchmark_llm(csv_path='benchmark_classification_V3.csv'):
    if not os.path.exists(csv_path):
        print(f"Fichier {csv_path} introuvable.")
        return

    df = pd.read_csv(csv_path)
    
    # === FILTRAGE SELON LE NOM DU FICHIER ===
    nom_fichier = os.path.basename(csv_path)
    if 'V1' in nom_fichier:
        df = df[df['prompt_version_lama'] == 'v1']
        print("Filtrage appliqué : prompt_version_lama = 'v1'")
    elif 'V2' in nom_fichier:
        df = df[df['prompt_version_lama'] == 'v2']
        print("Filtrage appliqué : prompt_version_lama = 'v2'")
    elif 'V3' in nom_fichier:
        df = df[df['prompt_version_lama'] == 'v3']
        print("Filtrage appliqué : prompt_version_lama = 'v3'")
        
    if df.empty:
        print("Le DataFrame est vide après le filtrage. Vérifiez vos données.")
        return
    
    # === DÉFINITION DES COLONNES (Baseline : Haiku) ===
    baseline = 'note_haiku'
    modeles_locaux = ['note_llama3', 'note_mistral', 'note_queen']
    noms_propres = {
        'note_llama3': 'Llama 3.1 (8B)', 
        'note_mistral': 'Mistral Nemo (12B)', 
        'note_queen': 'Qwen 2.5 (7B)'
    }

    print("\n=== ANALYSE DE LA ROBUSTESSE (JSON) ===")
    for mod in modeles_locaux:
        hors_limite = df[~df[mod].isin([0, 1, 2])][mod].count()
        print(f"{noms_propres[mod]} : {hors_limite} erreurs de format/valeurs aberrantes.")

    print("\n=== TAUX D'ACCORD EXACT (vs Claude Haiku) ===")
    df_clean = df.dropna(subset=[baseline])
    acc_scores = {}
    
    for mod in modeles_locaux:
        sub_df = df_clean.dropna(subset=[mod])
        # Filtrer les erreurs de parsing pour le calcul d'accuracy
        sub_df = sub_df[sub_df[mod].isin([0, 1, 2])] 
        if not sub_df.empty:
            acc = accuracy_score(sub_df[baseline], sub_df[mod]) * 100
            acc_scores[noms_propres[mod]] = acc
            print(f"{noms_propres[mod]} : {acc:.2f}%")
        else:
            acc_scores[noms_propres[mod]] = 0.0
            print(f"{noms_propres[mod]} : N/A (Aucune donnée valide)")

    print("\n=== BIAIS DE PRUDENCE (% de classe Neutre) ===")
    distrib = df[[baseline] + modeles_locaux].apply(lambda x: x.value_counts(normalize=True)).T * 100
    
    # Sécurité si une classe (0, 1 ou 2) est totalement absente après filtrage
    for val in [0, 1, 2]:
        if val not in distrib.columns:
            distrib[val] = 0.0
            
    distrib = distrib[[0, 1, 2]] # On ne garde que les notes valides
    distrib.columns = ['0 (Négatif)', '1 (Neutre)', '2 (Positif)']
    distrib.index = ['Haiku (Référence)', 'Llama 3.1', 'Mistral Nemo', 'Qwen 2.5']
    print(distrib.round(1))

    # --- VISUALISATION 1 : Taux d'accord ---
    plt.figure(figsize=(10, 5))
    # Correction du warning Seaborn : utilisation de 'hue' et 'legend=False'
    sns.barplot(
        x=list(acc_scores.keys()), 
        y=list(acc_scores.values()), 
        hue=list(acc_scores.keys()), 
        palette='viridis', 
        legend=False
    )
    plt.title("Taux d'accord exact avec l'API Claude Haiku", fontsize=14)
    plt.ylabel("Précision (%)")
    plt.ylim(0, 100)
    for i, v in enumerate(acc_scores.values()):
        plt.text(i, v + 2, f"{v:.1f}%", color='black', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('taux_accord_modeles.png')

    # --- VISUALISATION 2 : Distribution et biais de prudence ---
    plt.figure(figsize=(10, 6))
    distrib.plot(kind='bar', stacked=True, color=['#e63946', '#f1faee', '#457b9d'], edgecolor='black', figsize=(10,6))
    plt.title('Distribution des Notes : Le Biais de Prudence des Modèles Locaux', fontsize=14)
    plt.ylabel('Pourcentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('distribution_biais.png')
    
    print("\nGraphiques sauvegardés : 'taux_accord_modeles.png' et 'distribution_biais.png'")

if __name__ == "__main__":
    analyser_benchmark_llm()