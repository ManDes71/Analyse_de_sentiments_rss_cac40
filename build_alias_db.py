import os
import json
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from google import genai
from google.genai import types
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# ==========================================
# CONFIGURATION
# ==========================================
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME     = os.getenv("DB_NAME", "finance_db")


# ==========================================
# DEFINITION DU SCHEMA LLM
# ==========================================
class ListeAlias(BaseModel):
    alias: list[str]

def extraire_alias_avec_gemini(client, echantillon_textes, nom_entreprise):
    if not echantillon_textes:
        return []
        
    texte_concatene = "\n---\n".join(echantillon_textes)
    
    prompt = f"""
    Tu es un expert en analyse de la presse financière française.
    Voici une série d'extraits d'articles. Ton but est de repérer TOUS les termes, 
    périphrases, surnoms ou expressions utilisés par les journalistes pour désigner 
    l'entreprise "{nom_entreprise}".
    
    Exemples d'alias possibles : "le groupe", "la major", "le géant pétrolier", "la marque au losange", etc.
    Ne retiens que les expressions exactes trouvées dans le texte qui désignent cette entreprise.
    
    Textes à analyser :
    {texte_concatene}
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ListeAlias,
                temperature=0.1, 
            ),
        )
        # Nettoyage : on met tout en minuscules pour faciliter la correspondance plus tard
        return [a.lower().strip() for a in response.parsed.alias]
        
    except Exception as e:
        print(f"Erreur API pour {nom_entreprise}: {e}")
        return []

# ==========================================
# FONCTION PRINCIPALE
# ==========================================
def main():
    print("Connexion à la base de données...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
        )
    except Exception as e:
        print(f"Erreur de connexion à la BDD: {e}")
        return

    cur = conn.cursor(cursor_factory=RealDictCursor)
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    dictionnaire_alias = {}
    
    # 1. Récupérer toutes les entreprises
    cur.execute("SELECT id, name FROM public.companies ORDER BY id")
    companies = cur.fetchall()
    
    print(f"{len(companies)} entreprises trouvées. Début de l'extraction...\n")
    
    for company in companies:
        company_id = company['id']
        company_name = company['name']
        print(f"Traitement de [{company_id}] {company_name}...")
        
        # 2. Récupérer 50 articles au hasard pour cette entreprise
        # (Adaptez le nom de la table 'articles' et 'texte_brut' selon votre schéma exact)
        query_articles = """
            SELECT contenu 
            FROM public.articles_rss a
            JOIN public.article_companies ac ON ac.article_id = a.id
            WHERE ac.company_id = %s 
              AND a.contenu IS NOT NULL 
              AND LENGTH(a.contenu) > 100
            ORDER BY RANDOM() 
            LIMIT 50
        """
        cur.execute(query_articles, (company_id,))
        articles = [row['contenu'] for row in cur.fetchall()]
        
        if not articles:
            print(f"  -> Aucun article trouvé pour {company_name}. Ignoré.")
            continue
            
        # 3. Interroger l'API
        alias_trouves = extraire_alias_avec_gemini(client, articles, company_name)
        
        # Filtrer les alias vides et dédoublonner
        alias_propres = list(set([a for a in alias_trouves if a]))
        
        dictionnaire_alias[company_name] = alias_propres
        print(f"  -> Alias trouvés : {alias_propres}")
        
        # 4. Respecter le Rate Limit de l'API gratuite (15 requêtes/min = 1 req toutes les 4 sec)
        time.sleep(4)
        
    # 5. Sauvegarder dans un fichier JSON
    os.makedirs("/app/output", exist_ok=True)
    output_path = "/app/output/alias.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dictionnaire_alias, f, ensure_ascii=False, indent=4)
        
    print(f"\nTerminé ! Dictionnaire sauvegardé dans : {output_path}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()