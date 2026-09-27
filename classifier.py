#!/usr/bin/env python3
"""
Analyse le sentiment financier des articles liés à une entreprise (ou toutes),
sur le périmètre décidé par perimetre.raison_exclusion() : genre de l'article,
titre de chronique ou de liste, corps présent, nom de l'entreprise dans le titre.

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
import re
import argparse
from datetime import date

import nltk
import numpy as np
import psycopg2
import torch
from psycopg2.extras import RealDictCursor
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from dotenv import load_dotenv

# Reconnaissance du nom de l'entreprise (suffixes, alias, accents) : module
# léger, partagé avec perimetre.py et evaluate_article.py.
from entreprises import (
    charger_alias, citee_dans, compter_citations, normaliser, variantes_entreprise,
)
from perimetre import raison_exclusion

nltk.download("punkt_tab", quiet=True)

# Charger les variables d'environnement depuis .env
load_dotenv()

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_USER     = os.getenv("DB_USER", "")     # Défini dans .env
DB_PASSWORD = os.getenv("DB_PASSWORD", "") # Défini dans .env
DB_NAME     = os.getenv("DB_NAME", "")     # Défini dans .env


TODAY = date.today()

# Batching inter-articles pour optimiser l'utilisation du GPU/CPU
# Les pipelines HuggingFace traitent naturellement les listes de textes en batch
CLASSIFIER_BATCH_SIZE = int(os.getenv("CLASSIFIER_BATCH_SIZE", 32))

# Fenêtre du modèle de sentiment. Les textes plus longs sont découpés en blocs
# plutôt que tronqués (57% des articles éligibles dépassent cette limite).
MAX_TOKENS_MODELE = 512
# Marge de sécurité pour les tokens spéciaux (<s>, </s>)
TAILLE_BLOC = MAX_TOKENS_MODELE - 12

# note_targeted a été ABANDONNÉE le 20/09/2026, et la colonne est remise à NULL.
#
# Elle notait le sentiment sur le seul voisinage de la mention de l'entreprise,
# avec deux sentinelles : 9 pour un article dédié (nom dans le titre, donc rien
# à isoler) et 8 pour un nom introuvable. Or la dernière règle du périmètre
# EXIGE le nom dans le titre : tout couple évalué est dédié par construction.
# Le recalcul du 20/09 l'a vérifié — 663 couples, 663 fois la valeur 9, plus
# aucun 8. La colonne ne portait donc plus aucune information.


def resumer_texte_tfidf(texte, motif, n_phrases=3, facteur_boost=2.0):
    """Génère un résumé extractif basé sur les scores TF-IDF.

    `motif` vient de variantes_entreprise() : les phrases citant l'entreprise
    sous n'importe laquelle de ses graphies voient leur score multiplié.
    """
    phrases = nltk.sent_tokenize(texte, language="french")
    if len(phrases) <= n_phrases:
        return texte

    # 1. Calcul classique des scores TF-IDF par phrase
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(phrases)

    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # 2. Application du boost pour le nom de l'entreprise
    for i, phrase in enumerate(phrases):
        if citee_dans(motif, phrase):
            scores[i] *= facteur_boost  # On multiplie le score de la phrase par le boost

    # 3. Sélection et tri des meilleures phrases
    indices_cles = np.argsort(scores)[-n_phrases:]
    indices_cles.sort()

    return " ".join([phrases[i] for i in indices_cles])


def analyser_sentiment_finance(texte, sentiment_pipeline):
    """
    Analyse le sentiment financier sur la TOTALITÉ du texte en le découpant
    par blocs (chunks) de tokens si nécessaire, puis agrège les résultats.
    sentiment_pipeline : modèle Hugging Face  sentiment-analysis "bardsai/finance-sentiment-fr-base" 
    
    Retourne : (label_final, note_finale, confiance_moyenne)
    """
    if not texte or not texte.strip():
        return "NEUTRAL", 1, 1.0

    # 1. Accéder au tokenizer du pipeline
    tokenizer = sentiment_pipeline.tokenizer

    # 2. Convertir le texte complet en listes d'IDs de tokens
    # fait correspondre chaque token obtenu à son identifiant numérique unique (un entier) défini dans le vocabulaire du modèle
    tokens_ids = tokenizer.encode(texte, add_special_tokens=False)

    # 3. Découper en blocs (chunks) si le texte dépasse la capacité
    if len(tokens_ids) <= TAILLE_BLOC:
        # Cas simple : tout rentre en une seule fois
        resultat = sentiment_pipeline(texte, truncation=True)[0]
        label = resultat["label"].upper()
        confiance = resultat["score"]
        mapping_score = {"POSITIVE": 2, "NEUTRAL": 1, "NEGATIVE": 0}
        note = mapping_score.get(label, 1)
        return label, note, confiance

    # Cas complexe : Le texte est trop long, on applique le chunking
    chunks_ids = [tokens_ids[i:i + TAILLE_BLOC] for i in range(0, len(tokens_ids), TAILLE_BLOC)]
    
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

def process_batch_articles(cur, batch_articles: list, company_id: int, company_name: str, sentiment_pipeline, csv_rows: list | None = None, compteur_ignores: dict | None = None) -> int:
    """
    Traite un batch d'articles en parallèle (batching inter-articles).

    Version SIMPLIFIÉE : juste batching inter-articles via HF sentiment_pipeline.
    Pas de GPU batching custom (trop complexe et bugué).

    Retourne le nombre d'articles traités.
    """
    if not batch_articles:
        return 0

    nb_maj = 0
    articles_info = []
    resumes_a_analyser = []
    full_texts_a_analyser = []

    # Motif de reconnaissance de l'entreprise, calculé une fois pour le batch
    # (mémoïsé, donc une fois pour toute l'entreprise).
    motif = variantes_entreprise(company_id, company_name)

    # Phase 1 : Préparer tous les textes pour ce batch
    for art in batch_articles:
        titre = art["titre"] or ""
        contenu = art["contenu"] or ""
        texte_brut = f"{titre} {contenu}".strip()

        if not texte_brut:
            if compteur_ignores is not None:
                compteur_ignores["texte_vide"] += 1
            continue

        # Déterminer le type d'article
        #E L'article est-il "dédié" à l'entreprise ? (c'est-à-dire si le nom de l'entreprise figure dans le titre).
        est_dedie = citee_dans(motif, titre)
        nb_occ_contenu = compter_citations(motif, contenu) if contenu else 0
        est_dedie_unique = est_dedie and nb_occ_contenu <= 1

        # Préparer les textes
        resume = resumer_texte_tfidf(texte_brut, motif, n_phrases=3)

        articles_info.append({
            "article": art,
            "titre": titre,
            "resume": resume,
            "texte_brut": texte_brut,
            "est_dedie": est_dedie,
            "est_dedie_unique": est_dedie_unique,
        })

        """
        resumes_a_analyser : Tous les résumés extractifs générés par l'algorithme TF-IDF (qui met en valeur les phrases contenant le nom de l'entreprise) pour les articles du lot.
        full_texts_a_analyser : Elle regroupe les textes bruts complets de chaque article (c'est-à-dire la fusion du titre et du contenu). Le modèle analyse cette liste
            en une seule passe pour calculer le sentiment global, qui sera enregistré sous le nom de note_full.
        """


        resumes_a_analyser.append(resume)
        full_texts_a_analyser.append(texte_brut)

    if not articles_info:
        return 0



    """
    Voici ce que représentent ces variables dans le script. 
    Elles peuvent être séparées en deux catégories : 
        - les notes finales attribuées à un article individuel
        - les variables de stockage temporaire utilisées lors du traitement en lot (batch).

    A) Les notes individuelles (par article)
    
    Ces variables contiennent le score final (généralement 0 pour négatif, 1 pour neutre, 2 pour positif) qui sera enregistré dans la base de données pour un article précis.
    
    * `note_full` : C'est le score de sentiment calculé par le modèle d'IA sur la totalité du texte brut de l'article, c'est-à-dire le titre fusionné avec le contenu.
    
    
    * `note_tfidf` : C'est le score de sentiment calculé uniquement sur un résumé automatique de l'article. Ce résumé est généré via la méthode TF-IDF qui extrait les 3 phrases jugées les plus pertinentes.
    
    
    (`note_targeted` a été abandonnée le 20/09/2026 : voir l'en-tête du fichier.
    La colonne est désormais remise à NULL.)



    B) Les variables de traitement en lot (les "batches")
    
    Pour des raisons de performances, le script n'analyse pas les articles un par un, mais par paquets de 32 (par défaut). Ces variables stockent les résultats globaux renvoyés par l'IA pour tout le paquet d'un seul coup.
    
    * `notes_full_batch` : C'est une liste qui contient l'ensemble des scores "full" (textes complets) pour tous les articles du lot en cours d'analyse.
    
    
    * `notes_tfidf_batch` : C'est une liste qui contient l'ensemble des scores "tfidf" (résumés) pour ce même lot d'articles.
    """

    # Phase 2 : Analyser TOUS les textes en batch via HF pipeline
    print(f"    [BATCH] Analyse de {len(articles_info)} articles...")

    notes_tfidf_batch = analyser_sentiment_finance_batch(resumes_a_analyser, sentiment_pipeline)
    notes_full_batch = analyser_sentiment_finance_batch(full_texts_a_analyser, sentiment_pipeline)

    # Phase 3 : Mettre à jour BDD pour chaque article
    for i, info in enumerate(articles_info):
        art = info["article"]
        note_tfidf = notes_tfidf_batch[i]
        note_full = notes_full_batch[i]

        # note_targeted est remise à NULL : abandonnée (cf. en-tête du fichier).
        cur.execute(
            """
            UPDATE public.article_companies
               SET note_tfidf = %s,
                   note_targeted = NULL,
                   note_full = %s,
                   date_estim = %s
             WHERE article_id = %s AND company_id = %s
            """,
            (note_tfidf, note_full, TODAY, art["article_id"], company_id),
        )
        nb_maj += 1

        # Log simple
        print(f"    article {art['article_id']:>6} | tfidf={note_tfidf} full={note_full}")

        # CSV export (sans colonnes GPU)
        if csv_rows is not None:
            csv_rows.append({
                "article_id": art["article_id"],
                "company_id": company_id,
                "company_name": company_name,
                "type": "dedie" if (info["est_dedie"] and not info["est_dedie_unique"]) else ("dedie_unique" if info["est_dedie_unique"] else "generaliste"),
                "titre": info["titre"],
                "note_tfidf": note_tfidf,
                "note_full": note_full,
                "resume": info["resume"][:300].replace("\n", " "),
                "texte_brut": info["texte_brut"][:500].replace("\n", " "),
            })

    return nb_maj


def analyser_sentiment_finance_batch(textes: list, sentiment_pipeline) -> list:
    """
    Analyse un batch de textes en une seule fournée, en découpant les textes
    trop longs en blocs plutôt qu'en les tronquant.

    Les blocs de TOUS les textes sont aplatis en une seule liste et envoyés au
    pipeline en une passe (une seule fournée GPU), puis regroupés par texte
    d'origine et agrégés par moyenne pondérée par la confiance — même logique
    que analyser_sentiment_finance(), mais sur un batch.

    Retourne une liste de notes (0, 1, 2) correspondant aux textes.
    """
    if not textes:
        return []

    tokenizer = sentiment_pipeline.tokenizer

    # Phase 1 : découper chaque texte en blocs, en gardant la trace de son origine
    chunks_a_plat = []
    index_source = []  # index_source[i] = position du texte dont provient le bloc i

    for i, texte in enumerate(textes):
        if not texte or not texte.strip():
            continue  # texte vide : note neutre par défaut (cf. phase 3)

        tokens_ids = tokenizer.encode(texte, add_special_tokens=False)

        if len(tokens_ids) <= TAILLE_BLOC:
            chunks_a_plat.append(texte)
            index_source.append(i)
            continue

        # Texte trop long : découpage en blocs, reconvertis en texte pour le pipeline
        for debut in range(0, len(tokens_ids), TAILLE_BLOC):
            bloc = tokenizer.decode(tokens_ids[debut:debut + TAILLE_BLOC], skip_special_tokens=True)
            if bloc.strip():
                chunks_a_plat.append(bloc)
                index_source.append(i)

    if not chunks_a_plat:
        return [1] * len(textes)

    # Phase 2 : une seule fournée pour TOUS les blocs de TOUS les textes
    resultats = sentiment_pipeline(chunks_a_plat, truncation=True, batch_size=32)

    # Phase 3 : regrouper par texte d'origine, agréger par moyenne pondérée
    mapping_score = {"POSITIVE": 2, "NEUTRAL": 1, "NEGATIVE": 0}
    sommes_ponderees = [0.0] * len(textes)
    sommes_confiances = [0.0] * len(textes)

    for res, i in zip(resultats, index_source):
        note = mapping_score.get(res["label"].upper(), 1)
        confiance = res["score"]
        sommes_ponderees[i] += note * confiance
        sommes_confiances[i] += confiance

    # Textes vides ou sans bloc exploitable : note neutre (1)
    return [
        round(pond / conf) if conf > 0 else 1
        for pond, conf in zip(sommes_ponderees, sommes_confiances)
    ]


def traiter_entreprise(cur, company_id: int, company_name: str, sentiment_pipeline, csv_rows: list | None = None, article_ids_filtres: set[int] | None = None, compteur_ignores: dict | None = None) -> int:
    """Analyse et met à jour les articles éligibles (périmètre de perimetre.py, ou article_ids_filtres si fourni).

    Args:
        article_ids_filtres: si fourni, ne traiter que ces article_id (pour le mode subset).
                            Ne PAS appliquer le périmètre dans ce cas (cf. plan, constat 3).
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
        # Mode standard : le périmètre de perimetre.py, qui a besoin du genre.
        cur.execute(
            """
            SELECT ac.article_id, a.titre, a.contenu, g.label AS genre
            FROM public.article_companies ac
            JOIN public.articles_rss a ON a.id = ac.article_id
            LEFT JOIN public.article_genres g ON g.article_id = a.id AND g.actif
            WHERE ac.company_id = %s
            ORDER BY a.published_at DESC NULLS LAST
            """,
            (company_id,),
        )
    articles = cur.fetchall()

    if article_ids_filtres is None:
        motif = variantes_entreprise(company_id, company_name)
        articles = [a for a in articles
                    if not raison_exclusion(a["genre"], a["titre"], a["contenu"], motif)]

    nb_maj = 0
    articles_batch = []  # Buffer pour le batching

    # Traiter les articles par batch
    for art in articles:
        articles_batch.append(art)

        # Quand on atteint la taille du batch, traiter et vider le buffer
        if len(articles_batch) >= CLASSIFIER_BATCH_SIZE:
            nb_maj += process_batch_articles(
                cur, articles_batch, company_id, company_name,
                sentiment_pipeline, csv_rows, compteur_ignores
            )
            articles_batch = []

    # Traiter les articles restants (dernier batch incomplet)
    if articles_batch:
        nb_maj += process_batch_articles(
            cur, articles_batch, company_id, company_name,
            sentiment_pipeline, csv_rows, compteur_ignores
        )

    return nb_maj


def suffixe_subset(subset_csv: str) -> str:
    """
    Suffixe du CSV de résultats, dérivé du nom du CSV de subset.

    output/benchmark_classification_V7.csv
        -> _subset_benchmark_classification_v7

    ⚠ Règle dupliquée dans benchmark_classification_tfidf.py, qui reconstruit
    ce préfixe pour retrouver les résultats. Importer classifier depuis un
    script de reporting déclencherait ses effets de bord d'import (nltk.download,
    chargement de torch), d'où la duplication assumée : toute modification ici
    doit être répercutée là-bas.
    """
    return "_subset_" + os.path.splitext(os.path.basename(subset_csv))[0].lower()


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
  python classifier.py                    # Traite toutes les entreprises, périmètre de perimetre.py
  python classifier.py 15                 # Traite TotalEnergies (id=15) uniquement, même périmètre
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
    # Déterminer le device (GPU si disponible, sinon CPU)
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=device,
    )
    print(f"Modèle pipeline prêt (device={device} - {'GPU' if device >= 0 else 'CPU'}).")

    # GPU loading removed - using HF pipeline device=0 instead (simpler, more robust)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )

    with conn:
        with conn.cursor() as cur:

            # Alias chargés avant tout traitement : variantes_entreprise() les lit.
            charger_alias(cur)

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
                    sentiment_pipeline, csv_rows,
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
            # Le suffixe dérive du CSV de subset : deux subsets différents ne
            # doivent pas écraser le même fichier de résultats. Un suffixe fixe
            # ferait passer un subset pour un autre auprès des scripts de
            # reporting qui relisent ces CSV par glob.
            suffix = suffixe_subset(subset_csv)
        else:
            suffix = f"_{company_id_filtre}" if company_id_filtre else "_all"
        csv_path = os.path.join(os.path.dirname(__file__), f"output/comparaison_sentiment{suffix}_{TODAY}.csv")
        fieldnames = [
            "article_id", "company_id", "company_name", "type", "titre",
            "note_tfidf", "note_full",
            "resume", "texte_brut",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV exporté → {csv_path}")


if __name__ == "__main__":
    main()