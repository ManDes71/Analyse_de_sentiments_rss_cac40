# market-rss-sentiment

**`market-rss-sentiment`** est un pipeline d'IA bout en bout conçu pour automatiser la veille financière et la notation de sentiment d'entreprises à partir de flux de presse (RSS).

Face aux limites des approches génératives globales (coûts API, latence, faux positifs), le projet met en œuvre une **architecture hybride et ciblée** :

1. **Extraction dynamique d'alias :** Identification automatique du jargon et des surnoms journalistiques via Google Gemini 2.5 Flash et schémas Pydantic.


2. **Filtrage & Prétraitement :** Qualification des articles par expressions régulières et résumés TF-IDF avec pondération contextuelle (boost d'entité).


3. **Scoring NLP dual :** Évaluation fine du sentiment financier par un modèle CamemBERT spécialisé, comparant une approche par fenêtres glissantes GPU et une agrégation par blocs.


4. **Industrialisation MLOps :** Déploiement conteneurisé sous Docker, gestionnaire de paquets `uv`, et persistance PostgreSQL.



---

### 3. Bullets points pour un CV ou un Portfolio

> *Parfait pour illustrer tes compétences en MLOps, NLP et Data Engineering.*

* **Conception d'un pipeline d'IA hybride :** Traitement de flux RSS financiers combinant règles Regex, résumés TF-IDF ciblés et Transformers (CamemBERT Finance).


* **Intégration LLM & Structured Outputs :** Automatisation de la création d'un dictionnaire d'alias d'entreprises avec Gemini 2.5 et Pydantic.


* **Optimisation de l'inférence :** Mise en place d'une évaluation duale (mémoire/tokens via `pipeline` vs fenêtres glissantes matricielles sous PyTorch/GPU).


* **Industrialisation MLOps :** Stack conteneurisée sous Docker Compose, migration vers `uv` (Python 3.12) et stockage structuré PostgreSQL.



---

### 4. Post de lancement LinkedIn

> *Pour valoriser ton projet auprès de ton réseau professionnel.*

🚀 **Nouveau projet Open-Source : market-rss-sentiment**

Comment analyser efficacement le sentiment financier d'une entreprise dans la presse sans exploser ses coûts d'API ni subir le bruit des articles généralistes ?

Pour répondre à ce besoin, j'ai développé **`market-rss-sentiment`**, un pipeline MLOps complet qui combine le meilleur du NLP classique et des LLMs actuels :

1️⃣ **Compréhension du contexte :** Extraction automatique des alias journalistiques (ex: "la marque au losange", "le géant pétrolier") via Google Gemini & Pydantic.

2️⃣ **Focus sur l'entité :** Résumé extractif TF-IDF avec boost contextuel sur l'entreprise ciblée pour éliminer le bruit.

3️⃣ **Classification fine :** Double scoring via CamemBERT Finance (approche globale vs fenêtre glissante sur GPU).

4️⃣ **Stack MLOps :** Docker, `uv`, Python 3.12 et PostgreSQL.

Le projet est entièrement documenté et disponible sur GitHub !

🔗 [Lien vers ton repo GitHub]
