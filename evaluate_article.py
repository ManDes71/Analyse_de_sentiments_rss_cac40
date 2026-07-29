import os
import json
import time
import re
import csv
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME     = os.getenv("DB_NAME", "finance_db")

# ==========================================
# CONFIGURATION GOOGLE GEMINI
# ==========================================
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")



# ==========================================
# GESTION DES ALIAS ET REGEX (FILTRAGE)
# ==========================================
def charger_dictionnaire_alias(chemin_fichier="alias.json"):
    if not os.path.exists(chemin_fichier):
        print(f"Attention : le fichier {chemin_fichier} est introuvable. On continue sans alias.")
        return {}
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        return json.load(f)

def compiler_regex_entreprise(nom_entreprise, liste_alias):
    cibles = [nom_entreprise.lower()] + [a.lower() for a in liste_alias]
    cibles_echappees = [re.escape(cible) for cible in set(cibles) if cible]
    motif = r'\b(' + '|'.join(cibles_echappees) + r')\b'
    return re.compile(motif, re.IGNORECASE)

def compter_occurrences(texte, regex_compilee):
    if not texte: return 0
    return len(regex_compilee.findall(texte))

# ==========================================
# DÉFINITION DU SCHÉMA LLM
# ==========================================
# On force le LLM à renvoyer un objet JSON strict
class EvaluationSentiment(BaseModel):
    note_llm: int
    justification: str

def evaluer_sentiment_llm(client, texte_article, nom_entreprise):
    prompt = f"""
    Tu es un analyste financier expert. 
    Lis l'article suivant et évalue le sentiment STRICTEMENT par rapport à l'entreprise : {nom_entreprise}.
    
    Règles de notation :
    - 2 (POSITIVE) : L'article annonce une bonne nouvelle spécifique pour l'entreprise (hausse, bon résultat, contrat gagné).
    - 1 (NEUTRAL) : Simple mention factuelle, mouvement de marché global sans impact spécifique, ou informations contradictoires.
    - 0 (NEGATIVE) : Mauvaise nouvelle pour l'entreprise (baisse, perte, amende, dégradation par un broker).
    
    Fais bien la différence entre le sentiment du marché global et le sentiment lié à {nom_entreprise}.
    
    Article :
    ---
    {texte_article}
    ---
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvaluationSentiment,
                temperature=0.0,
            ),
        )
        return response.parsed
    except Exception as e:
        print(f"Erreur API lors de l'évaluation: {e}")
        return None

# ==========================================
# FONCTION PRINCIPALE
# ==========================================
def main():
    print("Connexion à la BDD...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    # Initialisation du client Google Gemini
    client = genai.Client(api_key=GOOGLE_API_KEY, http_options=types.HttpOptions(timeout=30000))
    
    # Charger les alias
    dict_alias = charger_dictionnaire_alias("alias.json")
    
    # Récupérer les entreprises
    cur.execute("SELECT id, name FROM public.companies ORDER BY id")
    companies = cur.fetchall()
    
    total_evalues = 0

    date_jour = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("/app/output", exist_ok=True)
    nom_fichier_csv = f"/app/output/resultats_llm_{date_jour}.csv"
    
    colonnes_csv = ['article_id', 'entreprise', 'note_llm', 'justification', 'texte_extrait']
    
    # On ouvre le fichier en mode "append" (ajout) pour écrire ligne par ligne
    # C'est plus sûr en cas de plantage du script au milieu du traitement.
    with open(nom_fichier_csv, mode='w', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.DictWriter(fichier_csv, fieldnames=colonnes_csv)
        writer.writeheader()
        print(f"Fichier de sauvegarde créé : {nom_fichier_csv}\n")

        for company in companies:
            company_id = company['id']
            company_name = company['name']
            
            # Préparer le filtre pour cette entreprise
            alias_liste = dict_alias.get(company_name, [])
            regex_entreprise = compiler_regex_entreprise(company_name, alias_liste)
            
            print(f"\n--- Traitement de [{company_id}] {company_name} ---")
            
            # Récupérer les articles non encore évalués par le LLM (à adapter selon votre schéma)
            # On suppose qu'il y a une colonne "note_llm" ou qu'on écrit dans une table de résultats
            query_articles = """
                SELECT a.id, a.contenu 
                FROM public.articles_rss a
                JOIN public.article_companies ac ON ac.article_id = a.id
                WHERE ac.company_id = %s 
                AND a.contenu IS NOT NULL
                ORDER BY a.id DESC
                LIMIT 100 -- Limite pour tester
            """
            cur.execute(query_articles, (company_id,))
            articles = cur.fetchall()
            
            for article in articles:
                texte = article['contenu']
                article_id = article['id']
                
                # 1. Filtrage par occurrence (> 1)
                nb_occ = compter_occurrences(texte, regex_entreprise)
                
                if nb_occ > 1:
                    print(f"Article {article_id} retenu ({nb_occ} occurrences). Évaluation LLM en cours...")
                    
                    # 2. Évaluation par Gemini
                    resultat = evaluer_sentiment_llm(client, texte, company_name)
                    
                    if resultat:
                        note = resultat.note_llm
                        justif = resultat.justification
                        
                        print(f"  -> Note: {note} | Justif: {justif[:80]}...")
                        
                        # 3. Sauvegarder en BDD (A DÉCOMMENTER ET ADAPTER SELON VOTRE SCHÉMA)
                        """
                        update_query = '''
                            UPDATE public.article_companies 
                            SET note_llm = %s, justification_llm = %s 
                            WHERE article_id = %s AND company_id = %s
                        '''
                        cur.execute(update_query, (note, justif, article_id, company_id))
                        conn.commit()
                        """
                        writer.writerow({
                            'article_id': article_id,
                            'entreprise': company_name,
                            'note_llm': note,
                            'justification': justif,
                            # On garde les 200 premiers caractères du texte pour s'y retrouver
                            'texte_extrait': texte[:200].replace('\n', ' ') + '...' 
                        })

                        # Pour forcer l'écriture sur le disque immédiatement (optionnel mais sécurisant)
                        fichier_csv.flush()
                        
                        total_evalues += 1
                        
                        # 4. Respecter le Rate Limit API (15 req/min sur le tier gratuit)
                        time.sleep(4)
                else:
                    # L'article mentionne l'entreprise 1 fois ou 0 fois (faux positif de jointure)
                    pass 
                
    cur.close()
    conn.close()
    print(f"\nTerminé ! {total_evalues} articles évalués et mis à jour.")

if __name__ == "__main__":
    main()