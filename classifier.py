#!/usr/bin/env python3
"""
Analyse le sentiment financier des articles liés à une entreprise (ou toutes),
uniquement pour les articles dont nbocc > 2.

- note_tfidf : sentiment calculé sur le résumé TF-IDF
- note_full  : sentiment calculé sur le texte brut
- date_estim : mis à la date du jour

Usage : python classifier.py [company_id]
Sans argument : traite TOUTES les entreprises.

uv run classifier.py 15   # TotalEnergies uniquement
uv run classifier.py      # toutes les entreprises


si problème de permission : sudo mkdir -p /run/user/1000 && sudo chown $USER /run/user/1000

Les deux notes sont complémentaires :

Une forte divergence (ex: tfidf=2 / full=0) signale un article ambigu — positif sur le fond mais négatif dans le contexte ou l'accroche
Une cohérence (tfidf=2 / full=2) indique un signal fort et fiable
Vous pourriez envisager une note composite :
note_composite = round((note_tfidf + note_full) / 2)

Ou utiliser la divergence comme indicateur d'incertitude du modèle.

source .venv/bin/activate
python3 classifier.py --subset-csv output/benchmark_classification_V7.csv 15
python3 classifier.py --subset-csv output/benchmark_classification_V7.csv
python3 benchmark_classification_tfidf.py
"""

import os
import csv
import argparse
from datetime import date

import nltk
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import torch

nltk.download("punkt_tab", quiet=True)

# Charger les variables d'environnement depuis .env
load_dotenv()

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME     = os.getenv("DB_NAME", "finance_db")


TODAY = date.today()
NBOCC_MIN = 2


def resumer_texte_tfidf(texte, nom_entreprise, n_phrases=3, facteur_boost=2.0):
    """Génère un résumé extractif basé sur les scores TF-IDF."""
    phrases = nltk.sent_tokenize(texte, language="french")
    if len(phrases) <= n_phrases:
        return texte

    # 1. Calcul classique des scores TF-IDF par phrase
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(phrases)

    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # 2. Application du boost pour le nom de l'entreprise
    for i, phrase in enumerate(phrases):
        if nom_entreprise.lower() in phrase.lower():
            scores[i] *= facteur_boost  # On multiplie le score de la phrase par le boost

    # 3. Sélection et tri des meilleures phrases
    indices_cles = np.argsort(scores)[-n_phrases:]
    indices_cles.sort()

    return " ".join([phrases[i] for i in indices_cles])


def analyser_sentiment_finance(texte, sentiment_pipeline):
    """
    Analyse le sentiment financier sur la TOTALITÉ du texte en le découpant
    par blocs (chunks) de tokens si nécessaire, puis agrège les résultats.
    
    Retourne : (label_final, note_finale, confiance_moyenne)
    """
    if not texte or not texte.strip():
        return "NEUTRAL", 1, 1.0

    # 1. Accéder au tokenizer du pipeline
    tokenizer = sentiment_pipeline.tokenizer
    max_tokens_modele = 512
    # On garde une marge de sécurité pour les tokens spéciaux (<s>, </s>)
    taille_bloc = max_tokens_modele - 12 

    # 2. Convertir le texte complet en listes d'IDs de tokens
    tokens_ids = tokenizer.encode(texte, add_special_tokens=False)

    # 3. Découper en blocs (chunks) si le texte dépasse la capacité
    if len(tokens_ids) <= taille_bloc:
        # Cas simple : tout rentre en une seule fois
        resultat = sentiment_pipeline(texte, truncation=True)[0]
        label = resultat["label"].upper()
        confiance = resultat["score"]
        mapping_score = {"POSITIVE": 2, "NEUTRAL": 1, "NEGATIVE": 0}
        note = mapping_score.get(label, 1)
        return label, note, confiance

    # Cas complexe : Le texte est trop long, on applique le chunking
    chunks_ids = [tokens_ids[i:i + taille_bloc] for i in range(0, len(tokens_ids), taille_bloc)]
    
    # Reconvertir les blocs d'IDs en texte pour le pipeline
    text_chunks = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks_ids]
    
    # Envoyer tous les blocs d'un coup au pipeline (batch processing)
    resultats_chunks = sentiment_pipeline(text_chunks, truncation=True)

    # 4. Agrégation des scores par Moyenne Pondérée par la Confiance
    mapping_score = {"POSITIVE": 2, "NEUTRAL": 1, "NEGATIVE": 0}
    inverse_mapping = {2: "POSITIVE", 1: "NEUTRAL", 0: "NEGATIVE"}
    
    somme_notes_ponderees = 0.0
    somme_confiances = 0.0

    for res in resultats_chunks:
        lbl = res["label"].upper()
        conf = res["score"]
        nt = mapping_score.get(lbl, 1)
        
        somme_notes_ponderees += nt * conf
        somme_confiances += conf

    # Calcul des métriques finales consolidées
    note_continue = somme_notes_ponderees / somme_confiances
    note_finale = round(note_continue)  # Donne 0, 1 ou 2 pour rester compatible avec votre BDD
    label_final = inverse_mapping.get(note_finale, "NEUTRAL")
    confiance_moyenne = somme_confiances / len(resultats_chunks)

    return label_final, note_finale, confiance_moyenne

import torch


def analyser_sentiment_finance_gpu(texte: str, tokenizer, model, max_length: int = 512, overlap: int = 50):
    if not texte or not texte.strip():
        return None, None

    # Encodage complet du texte sans troncature pour mesurer la masse réelle de tokens
    inputs = tokenizer(texte, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    total_tokens = len(input_ids)

    # Si le texte rentre dans une seule fenêtre, exécution directe de l'inférence
    if total_tokens <= max_length:
        with torch.no_grad():
            outputs = model(**tokenizer(texte, return_tensors="pt", truncation=True, max_length=max_length))
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        return probs, [float(torch.max(torch.nn.functional.softmax(outputs.logits, dim=-1)))]

    # Découpage par fenêtres glissantes avec chevauchement (overlap)
    chunks = []
    stride = max_length - overlap
    for i in range(0, total_tokens, stride):
        chunk = input_ids[i : i + max_length]
        if len(chunk) > 0:
            chunks.append(chunk)

    # Reconstitution des tenseurs et conversion en batch pour le GPU/CPU
    batch_input_ids = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask)
        batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

    # Extraction des probabilités et des scores de confiance de chaque segment
    confiances = [float(np.max(p)) for p in batch_probs]
    
    # Agrégation mathématique par moyenne pondérée par la confiance (détaillée ci-dessous)
    aggregated_probs = np.average(batch_probs, axis=0, weights=confiances)
    return aggregated_probs, confiances

def extraire_contexte_cible(texte, nom_entreprise):
    """
    Extrait la phrase contenant le nom de l'entreprise ainsi que 
    la phrase immédiatement avant et la phrase immédiatement après.
    """
    phrases = nltk.sent_tokenize(texte, language="french")
    nb_phrases = len(phrases)
    
    # Trouver tous les index des phrases qui mentionnent l'entreprise (insensible à la casse)
    index_trouves = [
        i for i, phrase in enumerate(phrases) 
        if nom_entreprise.lower() in phrase.lower()
    ]
    
    if not index_trouves:
        # En cas de repli si le nom exact n'est pas matché (ex: alias différent)
        return " ".join(phrases[:3])
    
    # On rassemble les index uniques (contexte élargi)
    index_a_garder = set()
    for idx in index_trouves:
        # On ajoute la phrase précédente, l'actuelle et la suivante
        index_a_garder.add(max(0, idx - 1))
        index_a_garder.add(idx)
        index_a_garder.add(min(nb_phrases - 1, idx + 1))
        
    # Recomposer le texte ciblé dans l'ordre chronologique de l'article
    phrases_cibles = [phrases[i] for i in sorted(index_a_garder)]
    return " ".join(phrases_cibles)

def traiter_entreprise(cur, company_id: int, company_name: str, sentiment_pipeline, tokenizer_gpu=None, model_gpu=None, mapping_gpu=None, csv_rows: list | None = None, article_ids_filtres: set[int] | None = None, compteur_ignores: dict | None = None) -> int:
    """Analyse et met à jour les articles éligibles (nbocc > NBOCC_MIN, ou article_ids_filtres si fourni).

    Args:
        article_ids_filtres: si fourni, ne traiter que ces article_id (pour le mode subset).
                            Ne PAS filtrer sur nbocc dans ce cas (cf. plan, constat 3).
        compteur_ignores: dict pour tracker le nombre d'articles ignorés (texte vide, etc.)
    """

    if article_ids_filtres is not None:
        # Mode subset : filtrer sur les paires exactes (article_id, company_id) du CSV V7
        # SANS condition sur nbocc (le nbocc courant en base peut différer du nbocc au moment
        # de l'évaluation LLM, donc on suit strictement le périmètre V7)
        cur.execute(
            """
            SELECT ac.article_id, a.titre, a.contenu
            FROM public.article_companies ac
            JOIN public.articles_rss a ON a.id = ac.article_id
            WHERE ac.company_id = %s
              AND ac.article_id = ANY(%s)
            ORDER BY a.published_at DESC NULLS LAST
            """,
            (company_id, list(article_ids_filtres)),
        )
    else:
        # Mode standard : filtrer sur nbocc > NBOCC_MIN
        cur.execute(
            """
            SELECT ac.article_id, a.titre, a.contenu
            FROM public.article_companies ac
            JOIN public.articles_rss a ON a.id = ac.article_id
            WHERE ac.company_id = %s
              AND ac.nbocc > %s
            ORDER BY a.published_at DESC NULLS LAST
            """,
            (company_id, NBOCC_MIN),
        )
    articles = cur.fetchall()

    nb_maj = 0
    for art in articles:
        titre = art["titre"] or ""
        contenu = art["contenu"] or ""
        texte_brut = f"{art['titre'] or ''} {art['contenu'] or ''}".strip()
        if not texte_brut:
            if compteur_ignores is not None:
                compteur_ignores["texte_vide"] += 1
            continue

        # Vérification de la nature de l'article
        est_dedie = company_name.lower() in titre.lower()
        nb_occ_contenu = contenu.lower().count(company_name.lower()) if contenu else 0
        # Article dédié mais cité une seule fois → traitement ciblé sur la phrase
        est_dedie_unique = est_dedie and nb_occ_contenu <= 1

        if est_dedie and not est_dedie_unique:
            # OPTION A : Article Dédié -> Sentiment sur le texte COMPLET (via chunking de tokens)
            # Analyse sur le résumé TF-IDF
            resume = resumer_texte_tfidf(texte_brut, company_name, n_phrases=3)
            _, note_tfidf, _ = analyser_sentiment_finance(resume, sentiment_pipeline)

            # Analyse sur le texte brut
            _, note_full, _ = analyser_sentiment_finance(texte_brut, sentiment_pipeline)
            # Note: note_targeted = 9 est une SENTINELLE (pas une note valide {0,1,2})
            # pour les articles dédiés, le sentiment ciblé n'est pas calculé.
            # Cette valeur doit être filtrée en aval (ex. dans les scripts de benchmark).
            note_targeted = 9

            # Analyse GPU sur le texte brut (comparaison uniquement, non utilisée pour la BDD)
            if tokenizer_gpu is not None and model_gpu is not None:
                probs_gpu, _ = analyser_sentiment_finance_gpu(texte_brut, tokenizer_gpu, model_gpu)
                if probs_gpu is not None:
                    idx_gpu = int(np.argmax(probs_gpu))
                    note_gpu = mapping_gpu[idx_gpu] if mapping_gpu else idx_gpu
                    print(
                        f"      [GPU] article {art['article_id']:>6} | "
                        f"note_full={note_full} | note_gpu={note_gpu} | "
                        f"probs={[round(float(p), 3) for p in probs_gpu]}"
                    )
                else:
                    note_gpu = None
            else:
                note_gpu = None
            note_targeted_gpu = None
            note_full_gpu = None
        else:
            # OPTION B : Article Généraliste -> Sentiment CIBLÉ sur l'environnement du mot
            contexte_cible = extraire_contexte_cible(texte_brut, company_name)
            _, note_targeted, _ = analyser_sentiment_finance(contexte_cible, sentiment_pipeline)
            
             # Analyse sur le résumé TF-IDF
            resume = resumer_texte_tfidf(texte_brut,  company_name, n_phrases=3)
            _, note_tfidf, _ = analyser_sentiment_finance(resume, sentiment_pipeline)

            # Analyse sur le texte brut
            _, note_full, _ = analyser_sentiment_finance(texte_brut, sentiment_pipeline)

            # Analyses GPU (comparaison uniquement, non utilisées pour la BDD)
            if tokenizer_gpu is not None and model_gpu is not None:
                probs_targeted_gpu, _ = analyser_sentiment_finance_gpu(contexte_cible, tokenizer_gpu, model_gpu)
                probs_full_gpu, _ = analyser_sentiment_finance_gpu(texte_brut, tokenizer_gpu, model_gpu)
                if probs_targeted_gpu is not None and probs_full_gpu is not None:
                    note_targeted_gpu = mapping_gpu[int(np.argmax(probs_targeted_gpu))] if mapping_gpu else int(np.argmax(probs_targeted_gpu))
                    note_full_gpu = mapping_gpu[int(np.argmax(probs_full_gpu))] if mapping_gpu else int(np.argmax(probs_full_gpu))
                    print(
                        f"      [GPU] article {art['article_id']:>6} | "
                        f"note_targeted={note_targeted} → note_targeted_gpu={note_targeted_gpu} | "
                        f"note_full={note_full} → note_full_gpu={note_full_gpu}"
                    )
                else:
                    note_targeted_gpu = None
                    note_full_gpu = None
            else:
                note_targeted_gpu = None
                note_full_gpu = None
            note_gpu = None
        cur.execute(
            """
            UPDATE public.article_companies
               SET note_tfidf = %s,
                   note_targeted = %s,
                   note_full  = %s,
                   date_estim = %s
             WHERE article_id = %s
               AND company_id = %s
            """,
            (note_tfidf, note_targeted, note_full, TODAY, art["article_id"], company_id),
        )
        nb_maj += 1
        print(
            f"    article {art['article_id']:>6} | "
            f"note_tfidf={note_tfidf} | note_targeted={note_targeted} | note_full={note_full}"
        )

        if csv_rows is not None:
            csv_rows.append({
                "article_id":       art["article_id"],
                "company_id":       company_id,
                "company_name":     company_name,
                "type":             "dedie" if (est_dedie and not est_dedie_unique) else ("dedie_unique" if est_dedie_unique else "generaliste"),
                "titre":            titre,
                "note_tfidf":       note_tfidf,
                "note_targeted":    note_targeted,
                "note_full":        note_full,
                "note_gpu":         note_gpu,
                "note_targeted_gpu": note_targeted_gpu,
                "note_full_gpu":    note_full_gpu,
                "divergence_full":  abs(note_full - note_gpu) if note_gpu is not None else "",
                "resume":           resume[:300].replace("\n", " "),
                "texte_brut":       texte_brut[:500].replace("\n", " "),
            })

    return nb_maj


def charger_subset_csv(csv_path):
    """
    Charge un CSV de benchmark (ex. benchmark_classification_V7.csv)
    et retourne un dict company_id -> set[article_id] pour filtrer le traitement.
    """
    subset_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_id = int(row["company_id"])
            article_id = int(row["article_id"])
            if company_id not in subset_dict:
                subset_dict[company_id] = set()
            subset_dict[company_id].add(article_id)
    return subset_dict


def main():
    parser = argparse.ArgumentParser(
        description="Analyse le sentiment TF-IDF+finance des articles financiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python classifier.py                    # Traite toutes les entreprises, nbocc > 2
  python classifier.py 15                 # Traite TotalEnergies (id=15) uniquement, nbocc > 2
  python classifier.py --subset-csv output/benchmark_classification_V7.csv
                                          # Traite seulement les couples (article, company) de V7
  python classifier.py --subset-csv output/benchmark_classification_V7.csv 15
                                          # Traite seulement TotalEnergies du subset V7
        """
    )
    parser.add_argument("company_id", nargs="?", type=int, default=None,
                        help="Optionnel : ID de l'entreprise à traiter seule")
    parser.add_argument("--subset-csv", type=str, default=None,
                        help="Optionnel : chemin du CSV de benchmark pour filtrer le périmètre")

    args = parser.parse_args()
    company_id_filtre = args.company_id
    subset_csv = args.subset_csv

    # Charger le subset si fourni
    subset_dict = None
    if subset_csv:
        print(f"Chargement du subset depuis {subset_csv}...")
        subset_dict = charger_subset_csv(subset_csv)
        print(f"  → {len(subset_dict)} entreprise(s) trouvée(s) dans le subset.")

    MODEL_NAME = "bardsai/finance-sentiment-fr-base"

    print("Chargement du modèle de sentiment…")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
    )
    print("Modèle pipeline prêt.")

    print("Chargement du tokenizer/modèle GPU…")
    tokenizer_gpu = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_gpu = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model_gpu.eval()
    # Construire le mapping index → note (0=NEG, 1=NEUTRAL, 2=POS) depuis id2label du modèle
    _label_to_note = {"POSITIVE": 2, "NEUTRAL": 1, "NEGATIVE": 0}
    mapping_gpu = {
        idx: _label_to_note.get(lbl.upper(), 1)
        for idx, lbl in model_gpu.config.id2label.items()
    }
    print(f"Mapping GPU labels : {model_gpu.config.id2label}")
    print("Modèle GPU prêt.\n")

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )

    with conn:
        with conn.cursor() as cur:

            # Déterminer quelles entreprises traiter
            if subset_dict is not None:
                # Mode subset : itérer sur les entreprises du subset
                if company_id_filtre and company_id_filtre not in subset_dict:
                    print(f"Entreprise {company_id_filtre} absente du subset.")
                    return

                company_ids_a_traiter = []
                if company_id_filtre:
                    # Filtrer sur une seule entreprise du subset
                    company_ids_a_traiter = [company_id_filtre]
                else:
                    # Toutes les entreprises du subset
                    company_ids_a_traiter = sorted(subset_dict.keys())

                # Charger les noms des entreprises
                placeholders = ",".join(["%s"] * len(company_ids_a_traiter))
                cur.execute(
                    f"SELECT id, name FROM public.companies WHERE id IN ({placeholders}) ORDER BY id",
                    company_ids_a_traiter
                )
                companies = cur.fetchall()
            else:
                # Mode standard : toutes les entreprises en base
                if company_id_filtre:
                    cur.execute(
                        "SELECT id, name FROM public.companies WHERE id = %s",
                        (company_id_filtre,),
                    )
                else:
                    cur.execute("SELECT id, name FROM public.companies ORDER BY id")

                companies = cur.fetchall()

            if not companies:
                print("Aucune entreprise trouvée.")
                return

            total_maj = 0
            csv_rows = []
            compteur_ignores = {"texte_vide": 0}

            for company in companies:
                print(f"[{company['id']:>4}] {company['name']}")

                article_ids_filtres = None
                if subset_dict is not None:
                    article_ids_filtres = subset_dict[company["id"]]
                    print(f"         Subset : {len(article_ids_filtres)} article(s)")

                nb = traiter_entreprise(
                    cur, company["id"], company["name"],
                    sentiment_pipeline,
                    tokenizer_gpu, model_gpu, mapping_gpu,
                    csv_rows,
                    article_ids_filtres=article_ids_filtres,
                    compteur_ignores=compteur_ignores
                )
                total_maj += nb
                print(f"         → {nb} articles mis à jour\n")

    conn.close()
    print(f"Terminé. {total_maj} lignes mises à jour (date_estim={TODAY}).")
    if compteur_ignores["texte_vide"] > 0:
        print(f"Articles ignorés (texte vide) : {compteur_ignores['texte_vide']}")

    if csv_rows:
        if subset_dict is not None:
            # Mode subset : nommer le CSV différemment pour éviter la confusion
            suffix = "_subset_v7"
        else:
            suffix = f"_{company_id_filtre}" if company_id_filtre else "_all"
        csv_path = os.path.join(os.path.dirname(__file__), f"output/comparaison_sentiment{suffix}_{TODAY}.csv")
        fieldnames = [
            "article_id", "company_id", "company_name", "type", "titre",
            "note_tfidf", "note_targeted", "note_full",
            "note_gpu", "note_targeted_gpu", "note_full_gpu",
            "divergence_full", "resume", "texte_brut",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV exporté → {csv_path}")


if __name__ == "__main__":
    main()