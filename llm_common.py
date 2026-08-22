"""
Éléments partagés entre tous les providers LLM (locaux gratuits et distants
payants) : prompts A/B, schéma de validation Pydantic, gestion des erreurs et
retry/backoff.

Centralisé ici (plutôt que dans evaluate_article.py) pour que le package
providers/ puisse importer ces éléments sans créer de dépendance circulaire
avec le script principal.
"""
import json
import os
import random
import re
import time

from typing import Optional

from pydantic import BaseModel, ValidationError

# ==========================================
# PROMPTS (A/B testing)
# ==========================================
PROMPT_TEMPLATE = """Tu es un analyste financier expert.
Lis l'article suivant et évalue le sentiment STRICTEMENT par rapport à l'entreprise : {entreprise}.

Règles de notation :
- 2 (POSITIVE) : L'article annonce une bonne nouvelle spécifique pour l'entreprise (hausse, bon résultat, contrat gagné).
- 1 (NEUTRAL) : Simple mention factuelle, mouvement de marché global sans impact spécifique, ou informations contradictoires.
- 0 (NEGATIVE) : Mauvaise nouvelle pour l'entreprise (baisse, perte, amende, dégradation par un broker).

Fais bien la différence entre le sentiment du marché global et le sentiment lié à {entreprise}.

Réponds UNIQUEMENT avec un objet JSON valide contenant les champs "note_llm" (entier 0, 1 ou 2) et "justification" (string).

Article :
---
{article}
---"""

# v2 : définit NEUTRAL de façon restrictive (pas une classe "par défaut"), impose
# d'isoler d'abord les phrases spécifiques à l'entreprise, et ancre les 3 classes
# avec un exemple few-shot chacune. Objectif : réduire le repli excessif vers NEUTRAL
# et améliorer le rappel NEG/POS observés faibles chez les modèles locaux.
PROMPT_TEMPLATE_V2 = """Tu es un analyste financier expert.

ÉTAPE 1 : Identifie les phrases de l'article qui mentionnent ou concernent directement {entreprise}. Ignore le reste du contexte de marché.

ÉTAPE 2 : Évalue le sentiment de CES PHRASES SPÉCIFIQUEMENT par rapport à {entreprise}, selon ces règles strictes :

- POSITIVE (2) : au moins un fait concret et spécifique à {entreprise} qui est une bonne nouvelle (hausse de résultat, contrat gagné, relèvement d'objectif, note d'analyste améliorée). Un ton globalement positif SANS fait concret ne suffit pas.
- NEGATIVE (0) : au moins un fait concret et spécifique à {entreprise} qui est une mauvaise nouvelle (baisse, perte, amende, dégradation, avertissement sur résultats, procès). Un ton globalement inquiet SANS fait concret ne suffit pas.
- NEUTRAL (1) : UNIQUEMENT si aucun fait concret positif OU négatif n'est identifiable pour {entreprise} — simple citation, mouvement de marché général non spécifique à l'entreprise, ou signaux positifs et négatifs qui s'annulent exactement.

RÈGLE IMPORTANTE : NEUTRAL n'est pas une valeur "par défaut en cas de doute". Si tu hésites entre NEUTRAL et POSITIVE/NEGATIVE, cherche à nouveau un fait concret dans le texte avant de trancher pour NEUTRAL.

Voici 3 exemples de référence :

Exemple A (POSITIVE) : "Renault a annoncé une hausse de 15% de son bénéfice net au T3, dépassant les attentes des analystes." → note_llm=2 (fait concret positif spécifique à Renault)

Exemple B (NEUTRAL) : "Parmi les valeurs du CAC40 en légère baisse ce matin figurent TotalEnergies, Renault et Sanofi, dans un marché globalement attentiste avant les chiffres de l'inflation." → note_llm=1 (mention factuelle sans information spécifique à l'entreprise, mouvement de marché global)

Exemple C (NEGATIVE) : "L'Autorité des marchés financiers a infligé une amende de 2M€ à Renault pour manquement à ses obligations d'information." → note_llm=0 (fait concret négatif spécifique à Renault)

Réponds UNIQUEMENT avec un objet JSON valide contenant "note_llm" (entier 0, 1 ou 2) et "justification" (string, citant le fait concret identifié ou expliquant l'absence de fait concret).

Article :
---
{article}
---"""

# v3 = v2 + un bloc de règles dédié aux notes de brokers/analystes (relèvement ou
# abaissement d'objectif de cours, changement de recommandation). Ces cas étaient
# une source d'ambiguïté récurrente : les règles ci-dessous viennent de cas réels
# annotés REMOVEDellement (cf. modif_prompt.txt).
#
# Logique métier retenue : c'est le NIVEAU DE RECOMMANDATION qui prime sur le
# mouvement de l'objectif de cours. Un objectif abaissé reste NEUTRAL si le broker
# maintient "acheter"/"surpondérer" (il reste au meilleur palier), mais devient
# NEGATIVE s'il n'est qu'à "conserver"/"surperformance" (palier inférieur).
PROMPT_TEMPLATE_V3 = """Tu es un analyste financier expert.

ÉTAPE 1 : Identifie les phrases de l'article qui mentionnent ou concernent directement {entreprise}. Ignore le reste du contexte de marché.

ÉTAPE 2 : Évalue le sentiment de CES PHRASES SPÉCIFIQUEMENT par rapport à {entreprise}, selon ces règles strictes :

- POSITIVE (2) : au moins un fait concret et spécifique à {entreprise} qui est une bonne nouvelle (hausse de résultat, contrat gagné, relèvement d'objectif, note d'analyste améliorée). Un ton globalement positif SANS fait concret ne suffit pas.
- NEGATIVE (0) : au moins un fait concret et spécifique à {entreprise} qui est une mauvaise nouvelle (baisse, perte, amende, dégradation, avertissement sur résultats, procès). Un ton globalement inquiet SANS fait concret ne suffit pas.
- NEUTRAL (1) : UNIQUEMENT si aucun fait concret positif OU négatif n'est identifiable pour {entreprise} — simple citation, mouvement de marché général non spécifique à l'entreprise, ou signaux positifs et négatifs qui s'annulent exactement.

RÈGLE IMPORTANTE : NEUTRAL n'est pas une valeur "par défaut en cas de doute". Si tu hésites entre NEUTRAL et POSITIVE/NEGATIVE, cherche à nouveau un fait concret dans le texte avant de trancher pour NEUTRAL.

RÈGLES SPÉCIFIQUES — objectifs de cours et recommandations de brokers/analystes.
Ces règles s'appliquent EN L'ABSENCE D'AUTRE NOUVELLE sur {entreprise} : si l'article contient par ailleurs un fait concret (résultats, contrat, amende...), applique d'abord les règles générales ci-dessus.

Principe général : note 2 quand l'objectif de cours est relevé ET la recommandation s'améliore ; note 0 quand l'objectif de cours est abaissé ET la recommandation se détériore.
Point clé : c'est le NIVEAU de la recommandation maintenue qui tranche les cas mixtes, pas le seul sens de l'objectif de cours.

- POSITIVE (2) : relèvement de la recommandation, OU rehaussement de l'objectif de cours avec maintien de la recommandation à "acheter" ou "surpondérer".
  Ex: "EssilorLuxottica a fini dans le vert, soutenu par HSBC qui a relevé sa recommandation à Acheter sur le titre." → note_llm=2
  Ex: "Jefferies rehausse sa cible de prix de 141,1 à 144,5 euros et maintient sa recommandation à l'achat." → note_llm=2
  Ex: "Barclays relève significativement son objectif de cours pour TotalEnergies de 78 euros à 94 euros et maintient sa recommandation 'surpondérer'." → note_llm=2

- NEUTRAL (1) : maintien de recommandation ambigu, notamment un objectif de cours abaissé MAIS une recommandation maintenue à l'achat, ou des signaux contradictoires entre conseil et objectif.
  Ex: "Citi abaisse légèrement sa cible de cours pour Carrefour (de 19 à 18 euros) mais maintient sa recommandation à l'achat." → note_llm=1
  Ex: "Berenberg abaisse son conseil à 'conserver' contre 'acheter' et relève son objectif de cours à 24,5 euros contre 17,5 euros." → note_llm=1

- NEGATIVE (0) : abaissement de l'objectif de cours avec maintien de la recommandation à "conserver" ou "surperformance".
  Ex: "Jefferies ajuste son objectif de cours pour Nexans de 140 euros à 136 euros mais maintient sa recommandation à 'conserver'." → note_llm=0
  Ex: "Bernstein reste à surperformance mais abaisse son objectif de cours de 199 euros à 190 euros." → note_llm=0

- NEUTRAL (1) : simple mention factuelle d'un mouvement de marché sectoriel, sans annonce spécifique positive ou négative propre à {entreprise}.
  Ex: "Dans un contexte de regain des tensions géopolitiques, le secteur de l'armement est en forte hausse en Europe, avec 6% pour BAE Systems, 2% pour Rheinmetall ou encore 1,4% pour Dassault Aviation." → note_llm=1

Réponds UNIQUEMENT avec un objet JSON valide contenant "note_llm" (entier 0, 1 ou 2) et "justification" (string, citant le fait concret identifié — y compris l'objectif de cours et le niveau de recommandation le cas échéant — ou expliquant l'absence de fait concret).

Article :
---
{article}
---"""

# v4 = correction de deux défauts de la v3, découverts en annotant REMOVEDellement
# la zone grise (cas où Haiku et Gemini divergeaient) :
#
# 1. CONTRESENS FACTUEL INDUIT PAR LE PROMPT. La v3 contenait la consigne
#    "c'est le NIVEAU de la recommandation maintenue qui tranche les cas mixtes",
#    ajoutée pour aider à généraliser. Elle poussait le modèle à réécrire les
#    faits pour les faire coller à la règle. Cas réel (Exosens) : le texte dit
#    "relève sa cible à 49 euros contre 45", Haiku a justifié par "abaisse son
#    objectif de cours de 45 euros à 49 euros", s'est contredit deux lignes plus
#    loin ("Bien que l'objectif soit relevé"), puis a noté 0.
#    -> Consigne supprimée, remplacée par une étape de lecture factuelle
#       explicite (2a) exécutée AVANT le classement.
#
# 2. TROU DANS LE JEU DE RÈGLES. Le cas "objectif RELEVÉ + maintien à conserver"
#    n'était couvert par aucune règle ; face au vide, le modèle extrapolait vers 0.
#    -> Grille complète couvrant les 12 combinaisons (3 sens d'objectif x
#       4 évolutions de recommandation).
#
# Logique de la grille (RÉVISÉE après annotation REMOVEDelle de la zone grise) :
# c'est le NIVEAU et l'ÉVOLUTION de la recommandation qui portent le signal. Un
# objectif de cours relevé peut rehausser la note, mais un objectif abaissé seul
# ne dégrade plus. Les lignes ABAISSÉ et INCHANGÉ sont donc identiques.
# Les exemples 7 et 8 sont passés de 0 à 1 pour rester cohérents avec la grille.
# Ajout d'une règle sur les mouvements de cours intraday sans cause annoncée.
#
# ATTENTION : ce prompt a été modifié EN PLACE, sans changer de numéro de version.
# Les notes 'v4' produites AVANT cette révision ne sont donc pas comparables à
# celles produites APRÈS, et le filtre de reprise ne peut pas les distinguer
# (il compare la chaîne 'v4', pas le contenu du prompt).
# Toute réévaluation doit se faire avec REPRENDRE=0.
PROMPT_TEMPLATE_V4 = """Tu es un analyste financier expert.

ÉTAPE 1 : Identifie les phrases de l'article qui mentionnent ou concernent directement {entreprise}. Ignore le reste du contexte de marché.

ÉTAPE 2 : Évalue le sentiment de CES PHRASES SPÉCIFIQUEMENT par rapport à {entreprise}, selon ces règles strictes :

- POSITIVE (2) : au moins un fait concret et spécifique à {entreprise} qui est une bonne nouvelle (hausse de résultat, contrat gagné, relèvement d'objectif, note d'analyste améliorée). Un ton globalement positif SANS fait concret ne suffit pas.
- NEGATIVE (0) : au moins un fait concret et spécifique à {entreprise} qui est une mauvaise nouvelle (baisse, perte, amende, dégradation, avertissement sur résultats, procès). Un ton globalement inquiet SANS fait concret ne suffit pas.
- NEUTRAL (1) : UNIQUEMENT si aucun fait concret positif OU négatif n'est identifiable pour {entreprise} — simple citation, mouvement de marché général non spécifique à l'entreprise, ou signaux positifs et négatifs qui s'annulent.

RÈGLE IMPORTANTE : NEUTRAL n'est pas une valeur "par défaut en cas de doute". Si tu hésites entre NEUTRAL et POSITIVE/NEGATIVE, cherche à nouveau un fait concret dans le texte avant de trancher pour NEUTRAL.

═══════════════════════════════════════════════════════════════
RÈGLES SPÉCIFIQUES — objectifs de cours et recommandations de brokers
═══════════════════════════════════════════════════════════════

Ces règles s'appliquent EN L'ABSENCE D'AUTRE NOUVELLE sur {entreprise}. Si l'article contient par ailleurs un fait concret (résultats, contrat, amende, avertissement...), applique d'abord les règles générales ci-dessus.

--- ÉTAPE 2a : LECTURE FACTUELLE (obligatoire avant de classer) ---

Relève littéralement dans le texte, sans interpréter :

(a) LE SENS DE L'OBJECTIF DE COURS :
    - RELEVÉ  : "relève", "rehausse", "augmente", "porte à", "remonte", ou une valeur d'arrivée SUPÉRIEURE à la valeur de départ
    - ABAISSÉ : "abaisse", "réduit", "ramène à", "ajuste à la baisse", ou une valeur d'arrivée INFÉRIEURE à la valeur de départ
    - INCHANGÉ : aucun objectif de cours mentionné, ou explicitement maintenu

    ATTENTION : si le texte dit "relève", l'objectif EST relevé. Si le texte donne
    deux montants, compare-les numériquement ("à 49 euros contre 45 euros" = relevé,
    de 45 vers 49). N'inverse JAMAIS ce sens pour le faire correspondre à une règle.

(b) LE NIVEAU DE LA RECOMMANDATION, et son évolution :
    - Niveau HAUT   : acheter, achat, surpondérer, renforcer
    - Niveau MOYEN  : conserver, neutre, surperformance, performance en ligne
    - Niveau BAS    : vendre, alléger, sous-pondérer
    - Évolution : AMÉLIORÉE (passe à un niveau supérieur), MAINTENUE (inchangée), DÉGRADÉE (passe à un niveau inférieur)

--- ÉTAPE 2b : CLASSEMENT selon la grille ---

Croise les deux éléments relevés à l'étape 2a :

| Objectif \\ Recommandation | AMÉLIORÉE | MAINTENUE au niveau HAUT | MAINTENUE au niveau MOYEN | DÉGRADÉE |
|---------------------------|-----------|--------------------------|---------------------------|----------|
| RELEVÉ                    |     2     |            2             |             1             |    1     |
| INCHANGÉ                  |     2     |            1             |             1             |    0     |
| ABAISSÉ                   |     2     |            1             |             1             |    0     |

Logique : c'est le NIVEAU et l'ÉVOLUTION de la recommandation qui portent l'essentiel du signal. Un objectif de cours relevé peut rehausser la note ; un objectif abaissé, seul, ne suffit pas à la dégrader.
Une recommandation MAINTENUE au niveau BAS (vendre) donne toujours 0.

--- EXEMPLES DE RÉFÉRENCE ---

Ex 1 : "EssilorLuxottica a fini dans le vert, soutenu par HSBC qui a relevé sa recommandation à Acheter sur le titre."
→ objectif INCHANGÉ, recommandation AMÉLIORÉE → note_llm=2

Ex 2 : "Jefferies rehausse sa cible de prix de 141,1 à 144,5 euros et maintient sa recommandation à l'achat."
→ objectif RELEVÉ, recommandation MAINTENUE au niveau HAUT → note_llm=2

Ex 3 : "Barclays relève significativement son objectif de cours pour TotalEnergies de 78 euros à 94 euros et maintient sa recommandation 'surpondérer'."
→ objectif RELEVÉ, recommandation MAINTENUE au niveau HAUT → note_llm=2

Ex 4 : "Deutsche Bank relève sa cible à 49 euros contre 45 euros et reste à 'conserver'."
→ objectif RELEVÉ (45 vers 49), recommandation MAINTENUE au niveau MOYEN → note_llm=1

Ex 5 : "Citi abaisse légèrement sa cible de cours pour Carrefour (de 19 à 18 euros) mais maintient sa recommandation à l'achat."
→ objectif ABAISSÉ, recommandation MAINTENUE au niveau HAUT → note_llm=1

Ex 6 : "Berenberg abaisse son conseil à 'conserver' contre 'acheter' et relève son objectif de cours à 24,5 euros contre 17,5 euros."
→ objectif RELEVÉ, recommandation DÉGRADÉE → note_llm=1

Ex 7 : "Jefferies ajuste son objectif de cours pour Nexans de 140 euros à 136 euros mais maintient sa recommandation à 'conserver'."
→ objectif ABAISSÉ, recommandation MAINTENUE au niveau MOYEN → note_llm=1

Ex 8 : "Bernstein reste à surperformance mais abaisse son objectif de cours de 199 euros à 190 euros."
→ objectif ABAISSÉ, recommandation MAINTENUE au niveau MOYEN → note_llm=1

--- AUTRE CAS ---

Simple mention factuelle d'un mouvement de marché sectoriel, sans annonce propre à {entreprise} → note_llm=1
Ex : "Dans un contexte de regain des tensions géopolitiques, le secteur de l'armement est en forte hausse en Europe, avec 6% pour BAE Systems, 2% pour Rheinmetall ou encore 1,4% pour Dassault Aviation."

Un simple mouvement de cours intraday de {entreprise}, sans recommandation ni cause annoncée → note_llm=1
Ex : "La Bourse de Paris gagne 0,3% ce matin, autour des 8120 points, soutenue par Legrand (+1,9%) et Safran (+1,5%)."
Contre-exemple : "Airbus progresse de 2,18% après avoir fait état d'un chiffre d'affaires en hausse" → note_llm=2 (le mouvement a une cause annoncée, qui est un fait concret positif).

═══════════════════════════════════════════════════════════════

Réponds UNIQUEMENT avec un objet JSON valide contenant :
- "note_llm" : entier 0, 1 ou 2
- "justification" : string. En cas de note de broker, indique explicitement le sens relevé de l'objectif de cours ET le niveau de la recommandation, en citant les termes exacts du texte.

Article :
---
{article}
---"""

# v5 = v2 + trois règles issues de l'annotation REMOVEDelle de 101 cas de la zone
# grise (cas où Haiku et Gemini divergeaient). L'arbitrage humain a montré que
# Haiku avait raison dans 76% de ces cas, ce qui valide la baseline.
#
# Base v2 et NON v4 : le benchmark montre que les règles brokers introduites en
# v3/v4 dégradent les modèles locaux (kappa Llama en zone de consensus :
# 0.707 en v2 -> 0.549 en v3 -> 0.490 en v4). La v2 reste l'optimum mesuré.
#
# Les trois ajouts, par ordre d'impact mesuré :
#
# 1. MOUVEMENT DE COURS INTRADAY. 38 des 101 cas annotés, notés 1 par l'humain
#    dans 36 cas. Gemini n'y répondait juste que 5% du temps (il lit "-1,3%"
#    comme un fait concret négatif) : à lui seul, ce motif explique 46% de ses
#    erreurs. Côté modèles locaux, c'est aussi le premier poste d'erreur de
#    Llama en v2 (NEU->POS = 51% de ses erreurs).
#    Seuil de 10% fixé par l'annotateur : au-delà, l'ampleur devient un signal.
#
# 2. TABLEAUX / PALMARÈS DE VALEURS (5 cas annotés) : listes de type "plus hauts
#    sur 1 an", sans information propre à l'entreprise.
#
# 3. SEUIL DE MATÉRIALITÉ. Plusieurs annotations convergentes ("annonce trop
#    vague", "pas une nouvelle liée à la stratégie ou aux résultats") : la v2
#    disait "fait concret" sans préciser qu'il doit être financier ou
#    opérationnel, ce qui laissait passer des annonces d'image.
PROMPT_TEMPLATE_V5 = """Tu es un analyste financier expert.

ÉTAPE 1 : Identifie les phrases de l'article qui mentionnent ou concernent directement {entreprise}. Ignore le reste du contexte de marché.

ÉTAPE 2 : Évalue le sentiment de CES PHRASES SPÉCIFIQUEMENT par rapport à {entreprise}, selon ces règles strictes :

- POSITIVE (2) : au moins un fait concret et spécifique à {entreprise} qui est une bonne nouvelle (hausse de résultat, contrat gagné, relèvement d'objectif, note d'analyste améliorée). Un ton globalement positif SANS fait concret ne suffit pas.
- NEGATIVE (0) : au moins un fait concret et spécifique à {entreprise} qui est une mauvaise nouvelle (baisse, perte, amende, dégradation, avertissement sur résultats, procès). Un ton globalement inquiet SANS fait concret ne suffit pas.
- NEUTRAL (1) : UNIQUEMENT si aucun fait concret positif OU négatif n'est identifiable pour {entreprise} — simple citation, mouvement de marché général non spécifique à l'entreprise, ou signaux positifs et négatifs qui s'annulent exactement.

RÈGLE IMPORTANTE : NEUTRAL n'est pas une valeur "par défaut en cas de doute". Si tu hésites entre NEUTRAL et POSITIVE/NEGATIVE, cherche à nouveau un fait concret dans le texte avant de trancher pour NEUTRAL.

QU'EST-CE QU'UN FAIT CONCRET ? Un fait est concret s'il est financier ou opérationnel et vérifiable : résultats, chiffre d'affaires, marge, contrat signé, commande, acquisition, cession, amende, procès, changement de recommandation ou d'objectif de cours d'un analyste. Ne comptent PAS comme faits concrets : une annonce d'intention vague, une initiative d'image ou de communication, un partenariat sans montant ni portée précisée, une déclaration de dirigeant sans chiffre.

RÈGLE DU MOUVEMENT DE COURS : la variation du cours de l'action de {entreprise} sur la séance (par exemple "recule de 1,3%", "gagne 2,18%", "-4,4% à Paris") n'est PAS en elle-même un fait concret. Elle reflète la réaction du marché, pas une nouvelle.
- IGNORE cette variation et juge l'article sur les AUTRES faits présents.
- S'il n'y a aucun autre fait concret, réponds note_llm=1, quel que soit le sens de la variation.
- EXCEPTION : une variation d'au moins 10% (à la hausse comme à la baisse) est en soi suffisamment significative pour être traitée comme un fait concret (2 si hausse, 0 si baisse).

Ex : "Parmi les lanternes rouges, on trouve Edenred (-3,4%) et Thales (-2,1%), suivies de Sanofi (-1,3%)." → note_llm=1 pour Sanofi (mouvement de cours seul, sous 10%)
Ex : "La Bourse de Paris gagne 0,3% ce matin, soutenue par Legrand (+1,9%) et Safran (+1,5%)." → note_llm=1 pour Legrand (mouvement de cours seul)
Ex : "Airbus progresse de 2,18% après avoir fait état d'un chiffre d'affaires en hausse." → note_llm=2 (on ignore le +2,18% mais le chiffre d'affaires en hausse est un fait concret positif)
Ex : "Le titre décroche de 12% après l'abandon du programme." → note_llm=0 (variation d'au moins 10%)

RÈGLE DES PALMARÈS : un tableau ou une liste de valeurs (plus hauts sur 1 an, plus fortes hausses/baisses du jour, palmarès sectoriel) où {entreprise} n'apparaît que comme une ligne parmi d'autres, sans information propre, → note_llm=1.

Voici 3 exemples de référence :

Exemple A (POSITIVE) : "Renault a annoncé une hausse de 15% de son bénéfice net au T3, dépassant les attentes des analystes." → note_llm=2 (fait concret positif spécifique à Renault)

Exemple B (NEUTRAL) : "Parmi les valeurs du CAC40 en légère baisse ce matin figurent TotalEnergies, Renault et Sanofi, dans un marché globalement attentiste avant les chiffres de l'inflation." → note_llm=1 (mention factuelle sans information spécifique à l'entreprise, mouvement de marché global)

Exemple C (NEGATIVE) : "L'Autorité des marchés financiers a infligé une amende de 2M€ à Renault pour manquement à ses obligations d'information." → note_llm=0 (fait concret négatif spécifique à Renault)

Réponds UNIQUEMENT avec un objet JSON valide contenant "note_llm" (entier 0, 1 ou 2) et "justification" (string, citant le fait concret identifié ou expliquant l'absence de fait concret).

Article :
---
{article}
---"""

# v6 = v2 + EXTRACTION STRUCTURÉE, sans règle de décision en langage naturel.
#
# Constat après cinq versions : ajouter des règles de décision dans le prompt
# dégrade les modèles locaux, et parfois inverse l'effet visé. La règle intraday
# de la v5 devait pousser Llama vers NEUTRAL ; elle l'a poussé vers POSITIVE
# (39,5% de justesse sur les cas intraday annotés, contre 68% en v3 SANS la règle).
# L'explication tenue pour la plus probable : la consigne "ignore la variation
# puis cherche ailleurs" demande deux opérations, et un 8B n'en retient que la
# première moitié.
#
# La v6 change de stratégie : le LLM CONSTATE des faits, le code DÉCIDE.
# - extraire "la variation vaut -1,3%" est une tâche de lecture, faisable par un 7B ;
# - appliquer "si mouvement seul et |variation| < seuil alors NEUTRAL" est une
#   tâche de règle, exacte par construction quand elle est écrite en Python
#   (cf. note_finale()).
#
# Bénéfice secondaire : les seuils deviennent modifiables sans réévaluer. Passer
# de 5% à 10% est une requête SQL sur la colonne extraction_*, pas 1281 appels API.
#
# La variation est demandée en NOMBRE SIGNÉ et non en booléen "> 5%" : figer le
# seuil dans le prompt obligerait à tout relancer pour le changer, ce qui est
# précisément ce qu'on cherche à éviter.
PROMPT_TEMPLATE_V6 = """Tu es un analyste financier expert.

ÉTAPE 1 : Identifie les phrases de l'article qui mentionnent ou concernent directement {entreprise}. Ignore le reste du contexte de marché.

ÉTAPE 2 : Évalue le sentiment de CES PHRASES SPÉCIFIQUEMENT par rapport à {entreprise}, selon ces règles strictes :

- POSITIVE (2) : au moins un fait concret et spécifique à {entreprise} qui est une bonne nouvelle (hausse de résultat, contrat gagné, relèvement d'objectif, note d'analyste améliorée). Un ton globalement positif SANS fait concret ne suffit pas.
- NEGATIVE (0) : au moins un fait concret et spécifique à {entreprise} qui est une mauvaise nouvelle (baisse, perte, amende, dégradation, avertissement sur résultats, procès). Un ton globalement inquiet SANS fait concret ne suffit pas.
- NEUTRAL (1) : UNIQUEMENT si aucun fait concret positif OU négatif n'est identifiable pour {entreprise} — simple citation, mouvement de marché général non spécifique à l'entreprise, ou signaux positifs et négatifs qui s'annulent exactement.

RÈGLE IMPORTANTE : NEUTRAL n'est pas une valeur "par défaut en cas de doute". Si tu hésites entre NEUTRAL et POSITIVE/NEGATIVE, cherche à nouveau un fait concret dans le texte avant de trancher pour NEUTRAL.

Voici 3 exemples de référence :

Exemple A (POSITIVE) : "Renault a annoncé une hausse de 15% de son bénéfice net au T3, dépassant les attentes des analystes." → note_llm=2 (fait concret positif spécifique à Renault)

Exemple B (NEUTRAL) : "Parmi les valeurs du CAC40 en légère baisse ce matin figurent TotalEnergies, Renault et Sanofi, dans un marché globalement attentiste avant les chiffres de l'inflation." → note_llm=1 (mention factuelle sans information spécifique à l'entreprise, mouvement de marché global)

Exemple C (NEGATIVE) : "L'Autorité des marchés financiers a infligé une amende de 2M€ à Renault pour manquement à ses obligations d'information." → note_llm=0 (fait concret négatif spécifique à Renault)

ÉTAPE 3 : Relève ensuite quatre éléments FACTUELS concernant {entreprise}. Ne juge pas, constate seulement ce qui est écrit dans l'article.

1. "variation_cours_pct" : la variation du cours de l'action de {entreprise} sur la séance, en pourcentage, sous forme de NOMBRE SIGNÉ.
   Ex : "recule de 1,3%" → -1.3 | "gagne 2,18%" → 2.18 | "(+1,9%)" → 1.9 | "décroche de 12%" → -12.0
   Mets null si l'article ne donne aucune variation de cours pour {entreprise}.
   ATTENTION : ne confonds pas avec un cours en euros, un objectif de cours, un volume, ou la variation d'un indice ou d'une AUTRE société.

2. "autre_fait_concret" : true s'il existe, EN DEHORS de la variation de cours, au moins un fait concret et spécifique à {entreprise} (résultats, chiffre d'affaires, contrat, commande, acquisition, cession, amende, procès, avertissement, note d'analyste). false s'il n'y a que la variation de cours, une simple citation, ou un commentaire de marché général.
   Ex : "Sanofi (-1,3%)" dans une liste → false
   Ex : "Airbus progresse de 2,18% après avoir fait état d'un chiffre d'affaires en hausse" → true

3. "reco_sens" : évolution de la recommandation d'un analyste sur {entreprise}. Exactement une valeur parmi :
   "amelioree"  (le conseil passe à un niveau supérieur, ex : conserver → acheter)
   "maintenue"  (le conseil est confirmé, même si l'objectif de cours change)
   "degradee"   (le conseil passe à un niveau inférieur, ex : acheter → conserver)
   "aucune"     (aucune recommandation d'analyste n'est mentionnée)

4. "reco_niveau" : niveau de la recommandation à l'issue de l'article. Exactement une valeur parmi :
   "haut"   (acheter, achat, surpondérer, renforcer)
   "moyen"  (conserver, neutre, surperformance, performance en ligne)
   "bas"    (vendre, alléger, sous-pondérer)
   null     (aucune recommandation mentionnée)

Réponds UNIQUEMENT avec un objet JSON valide contenant EXACTEMENT ces six champs :
{{"note_llm": <0, 1 ou 2>, "justification": "<string>", "variation_cours_pct": <nombre ou null>, "autre_fait_concret": <true ou false>, "reco_sens": "<amelioree|maintenue|degradee|aucune>", "reco_niveau": "<haut|moyen|bas ou null>"}}

Article :
---
{article}
---"""

# v7 = v6 avec UN SEUL changement : la définition de "autre_fait_concret".
#
# Diagnostic mesuré sur le run v6 complet (1280 lignes, 100 cas annotés) :
# le champ n'était correct que dans 34% des cas de mouvement intraday. Llama
# comptait la VARIATION DE COURS elle-même comme "autre fait concret", alors
# que le champ est défini comme "en dehors de la variation de cours". Ses
# justifications le montrent sans ambiguïté :
#   "Saint-Gobain a connu une hausse de 2,8%"        -> true (faux)
#   "Capgemini complète le podium du jour avec 4,2%" -> true (faux)
#   "Safran (+2,3%) soutient l'indice CAC40"         -> true (faux)
#   "AXA bénéficie d'une hausse de +1,4%"            -> true (faux)
#
# Conséquence : la règle déterministe de note_finale() ne se déclenchait jamais
# sur les cas où elle aurait été utile, et le modèle reprenait la main là où il
# est le plus faible. Llama v6 = 56% sur la vérité terrain, contre 72% en v4.
#
# Cause probable : l'unique exemple attaché au champ en v6 ("Airbus progresse de
# 2,18% APRÈS avoir fait état d'un chiffre d'affaires en hausse" -> true) montrait
# un cas où variation ET fait coexistent, sans jamais montrer le cas symétrique.
# Le modèle a retenu l'association "il y a un pourcentage -> true".
#
# Correctif : interdiction explicite, test mécanique de suppression des
# pourcentages, et les quatre CAS D'ÉCHEC RÉELS comme contre-exemples — plus
# efficaces que des exemples inventés puisqu'ils viennent du corpus.
PROMPT_TEMPLATE_V7 = """Tu es un analyste financier expert.

ÉTAPE 1 : Identifie les phrases de l'article qui mentionnent ou concernent directement {entreprise}. Ignore le reste du contexte de marché.

ÉTAPE 2 : Évalue le sentiment de CES PHRASES SPÉCIFIQUEMENT par rapport à {entreprise}, selon ces règles strictes :

- POSITIVE (2) : au moins un fait concret et spécifique à {entreprise} qui est une bonne nouvelle (hausse de résultat, contrat gagné, relèvement d'objectif, note d'analyste améliorée). Un ton globalement positif SANS fait concret ne suffit pas.
- NEGATIVE (0) : au moins un fait concret et spécifique à {entreprise} qui est une mauvaise nouvelle (baisse, perte, amende, dégradation, avertissement sur résultats, procès). Un ton globalement inquiet SANS fait concret ne suffit pas.
- NEUTRAL (1) : UNIQUEMENT si aucun fait concret positif OU négatif n'est identifiable pour {entreprise} — simple citation, mouvement de marché général non spécifique à l'entreprise, ou signaux positifs et négatifs qui s'annulent exactement.

RÈGLE IMPORTANTE : NEUTRAL n'est pas une valeur "par défaut en cas de doute". Si tu hésites entre NEUTRAL et POSITIVE/NEGATIVE, cherche à nouveau un fait concret dans le texte avant de trancher pour NEUTRAL.

Voici 3 exemples de référence :

Exemple A (POSITIVE) : "Renault a annoncé une hausse de 15% de son bénéfice net au T3, dépassant les attentes des analystes." → note_llm=2 (fait concret positif spécifique à Renault)

Exemple B (NEUTRAL) : "Parmi les valeurs du CAC40 en légère baisse ce matin figurent TotalEnergies, Renault et Sanofi, dans un marché globalement attentiste avant les chiffres de l'inflation." → note_llm=1 (mention factuelle sans information spécifique à l'entreprise, mouvement de marché global)

Exemple C (NEGATIVE) : "L'Autorité des marchés financiers a infligé une amende de 2M€ à Renault pour manquement à ses obligations d'information." → note_llm=0 (fait concret négatif spécifique à Renault)

ÉTAPE 3 : Relève ensuite quatre éléments FACTUELS concernant {entreprise}. Ne juge pas, constate seulement ce qui est écrit dans l'article.

1. "variation_cours_pct" : la variation du cours de l'action de {entreprise} sur la séance, en pourcentage, sous forme de NOMBRE SIGNÉ.
   Ex : "recule de 1,3%" → -1.3 | "gagne 2,18%" → 2.18 | "(+1,9%)" → 1.9 | "décroche de 12%" → -12.0
   Mets null si l'article ne donne aucune variation de cours pour {entreprise}.
   ATTENTION : ne confonds pas avec un cours en euros, un objectif de cours, un volume, ou la variation d'un indice ou d'une AUTRE société.

2. "autre_fait_concret" : LA VARIATION DE COURS NE COMPTE JAMAIS COMME UN FAIT CONCRET. Elle a déjà été relevée au point 1, ne la compte pas une seconde fois ici.
   Réponds true UNIQUEMENT si l'article contient, EN PLUS du pourcentage, une information sur l'ACTIVITÉ de {entreprise} : résultats, chiffre d'affaires, marge, contrat, commande, acquisition, cession, amende, procès, avertissement sur résultats, ou note d'analyste.

   TEST À APPLIQUER : supprime mentalement TOUS les pourcentages et toutes les mentions de hausse ou de baisse du cours. Reste-t-il une information sur {entreprise} ? Si non, réponds false.

   Ex : "Saint-Gobain a connu une hausse de 2,8%." → false (il ne reste rien après suppression)
   Ex : "Capgemini complète le podium du jour avec 4,2%." → false (il ne reste rien)
   Ex : "Safran (+2,3%) soutient l'indice CAC40." → false (il ne reste rien)
   Ex : "AXA bénéficie d'une hausse de +1,4%." → false (il ne reste rien)
   Ex : "Parmi les baisses : Edenred (-3,4%) et Sanofi (-1,3%)." → false (il ne reste rien)
   Ex : "Airbus progresse de 2,18% après avoir fait état d'un chiffre d'affaires en hausse." → true (il reste "chiffre d'affaires en hausse")
   Ex : "Sanofi recule de 0,7% et a annoncé le rachat du groupe Dynavax." → true (il reste "rachat de Dynavax")

3. "reco_sens" : évolution de la recommandation d'un analyste sur {entreprise}. Exactement une valeur parmi :
   "amelioree"  (le conseil passe à un niveau supérieur, ex : conserver → acheter)
   "maintenue"  (le conseil est confirmé, même si l'objectif de cours change)
   "degradee"   (le conseil passe à un niveau inférieur, ex : acheter → conserver)
   "aucune"     (aucune recommandation d'analyste n'est mentionnée)

4. "reco_niveau" : niveau de la recommandation à l'issue de l'article. Exactement une valeur parmi :
   "haut"   (acheter, achat, surpondérer, renforcer)
   "moyen"  (conserver, neutre, surperformance, performance en ligne)
   "bas"    (vendre, alléger, sous-pondérer)
   null     (aucune recommandation mentionnée)

Réponds UNIQUEMENT avec un objet JSON valide contenant EXACTEMENT ces six champs :
{{"note_llm": <0, 1 ou 2>, "justification": "<string>", "variation_cours_pct": <nombre ou null>, "autre_fait_concret": <true ou false>, "reco_sens": "<amelioree|maintenue|degradee|aucune>", "reco_niveau": "<haut|moyen|bas ou null>"}}

Article :
---
{article}
---"""

# Registre des prompts disponibles pour l'A/B testing. Ajouter une entrée ici
# suffit pour qu'une nouvelle version soit sélectionnable via --prompt-version.
PROMPTS = {
    "v1": PROMPT_TEMPLATE,
    "v2": PROMPT_TEMPLATE_V2,
    "v3": PROMPT_TEMPLATE_V3,
    "v4": PROMPT_TEMPLATE_V4,
    "v5": PROMPT_TEMPLATE_V5,
    "v6": PROMPT_TEMPLATE_V6,
    "v7": PROMPT_TEMPLATE_V7,
}


class EvaluationSentiment(BaseModel):
    """
    Réponse d'un LLM pour un couple (article, entreprise).

    note_llm / justification : présents dans TOUTES les versions de prompt.

    Les quatre champs suivants ne sont demandés qu'à partir de la v6, qui sépare
    l'EXTRACTION de faits (tâche simple, à la portée d'un modèle 7B) de la
    DÉCISION (application de règles, où les petits modèles s'effondrent — mesuré
    sur v3/v4/v5). Ils sont donc optionnels, pour que les versions v1 à v5
    restent valides sans modification.

    Ils permettent de recalculer une note en Python via note_finale(), sans
    relancer un seul appel API : changer un seuil devient une requête SQL.
    """
    note_llm: int
    justification: str
    # Variation du cours de l'action SUR LA SÉANCE, en pourcentage signé
    # (ex: -1.3, 2.18). None si l'article n'en mentionne aucune.
    variation_cours_pct: Optional[float] = None
    # Existe-t-il un fait concret AUTRE que la variation de cours ?
    # C'est ce champ qui distingue "Sanofi -1,3%" de "Sanofi -1,3% après l'échec
    # de son essai clinique".
    autre_fait_concret: Optional[bool] = None
    # Évolution de la recommandation d'analyste, et niveau atteint.
    reco_sens: Optional[str] = None      # amelioree | maintenue | degradee | aucune
    reco_niveau: Optional[str] = None    # haut | moyen | bas | None


NOTES_VALIDES = {0, 1, 2}
RECO_SENS_VALIDES = {"amelioree", "maintenue", "degradee", "aucune"}
RECO_NIVEAUX_VALIDES = {"haut", "moyen", "bas"}

# Bornes de sanité sur la variation de cours INTRADAY. Resserrées à +/-30% après
# le run v6 complet, qui a produit des valeurs de -87% et +170% : au-delà de 30%
# sur une séance, c'est presque toujours que le modèle a recopié autre chose
# (un cours en euros, une performance annuelle, un volume, une capitalisation).
# Une variation hors bornes est ramenée à None, ce qui fait retomber le cas sur
# la règle "pas de variation exploitable" plutôt que de produire une note
# automatique fausse.
VARIATION_MIN, VARIATION_MAX = -30.0, 30.0


def valider_evaluation(parsed):
    """
    Validation stricte d'une réponse LLM brute (dict issu du JSON) :
    1. Schéma Pydantic (types corrects, champs présents) — lève ValidationError sinon.
    2. Plage métier note_llm ∈ {0,1,2} — lève ValueError sinon.
    3. Champs d'extraction v6, quand ils sont présents : énumérations et bornes.

    Sans cette étape, une réponse malformée (ex: le modèle recopie un pourcentage
    du texte dans note_llm au lieu de trancher 0/1/2) serait insérée telle quelle
    en base et fausserait silencieusement toute analyse de qualité en aval.

    Les champs d'extraction sont validés de façon TOLÉRANTE : une valeur
    d'énumération inconnue est ramenée à None plutôt que de faire échouer toute
    l'évaluation. La note reste exploitable même si l'extraction est imparfaite.

    Retourne un EvaluationSentiment validé, ou lève une exception (ValidationError
    ou ValueError) à charge de l'appelant de la traiter comme un échec.
    """
    val = EvaluationSentiment(**parsed)
    if val.note_llm not in NOTES_VALIDES:
        raise ValueError(
            f"note_llm hors plage attendue {sorted(NOTES_VALIDES)}: {val.note_llm!r}"
        )

    if val.reco_sens is not None:
        sens = str(val.reco_sens).strip().lower()
        val.reco_sens = sens if sens in RECO_SENS_VALIDES else None

    if val.reco_niveau is not None:
        niveau = str(val.reco_niveau).strip().lower()
        val.reco_niveau = niveau if niveau in RECO_NIVEAUX_VALIDES else None

    if val.variation_cours_pct is not None:
        if not (VARIATION_MIN <= val.variation_cours_pct <= VARIATION_MAX):
            val.variation_cours_pct = None

    return val


def resultat_depuis_evaluation(val):
    """
    Convertit un EvaluationSentiment validé en dict de résultat, contrat commun
    à tous les providers. Centralisé ici pour qu'ajouter un champ d'extraction
    ne demande pas de modifier les quatre modules de providers.
    """
    return {
        "note_llm": val.note_llm,
        "justification": val.justification,
        "variation_cours_pct": val.variation_cours_pct,
        "autre_fait_concret": val.autre_fait_concret,
        "reco_sens": val.reco_sens,
        "reco_niveau": val.reco_niveau,
    }


# Champs d'extraction stockés en base (colonne JSONB extraction_*), à recalculer
# via note_finale() sans réévaluation.
CHAMPS_EXTRACTION = ("variation_cours_pct", "autre_fait_concret", "reco_sens", "reco_niveau")

SEUIL_VARIATION_DEFAUT = float(os.getenv("SEUIL_VARIATION_PCT", 5.0))


def note_finale(resultat, seuil_pct=None):
    """
    Applique les règles déterministes aux faits extraits, et corrige la note du
    modèle uniquement là où la règle est certaine.

    Principe : le LLM constate, le code décide. Les cas de mouvement de cours
    intraday sans autre information sont la première source d'erreur des modèles
    locaux (51% des erreurs de Llama en v2, et la règle en langage naturel de la
    v5 a produit l'effet INVERSE de celui visé). Ici la règle est appliquée en
    Python, donc exacte par construction.

    Retourne (note, origine) où origine vaut 'regle' si la note a été imposée
    par le code, 'modele' si le jugement du LLM a été conservé.

    Hors du cas traité, on ne touche à rien : un post-traitement trop ambitieux
    reproduirait le problème des règles de v3/v4 en le déplaçant dans le code.
    """
    seuil = SEUIL_VARIATION_DEFAUT if seuil_pct is None else float(seuil_pct)

    autre_fait = resultat.get("autre_fait_concret")
    reco_sens = resultat.get("reco_sens")
    variation = resultat.get("variation_cours_pct")

    # Extraction absente (versions v1 à v5) : rien à appliquer.
    if autre_fait is None:
        return resultat.get("note_llm"), "modele"

    # Un fait concret ou un mouvement de recommandation existe : c'est au modèle
    # de juger, on ne s'en mêle pas.
    if autre_fait or (reco_sens not in (None, "aucune")):
        return resultat.get("note_llm"), "modele"

    # À partir d'ici : l'article ne contient qu'un mouvement de cours, ou rien.
    if variation is None or abs(variation) < seuil:
        return 1, "regle"
    return (2, "regle") if variation > 0 else (0, "regle")


class FatalLLMError(Exception):
    """Erreur bloquante de configuration/API distante."""


def extraire_json_depuis_texte(texte):
    if not texte:
        return None

    brut = texte.strip()
    if brut.startswith("```"):
        brut = re.sub(r"^```(?:json)?", "", brut, flags=re.IGNORECASE).strip()
        brut = re.sub(r"```$", "", brut).strip()

    try:
        return json.loads(brut)
    except Exception:
        debut = brut.find("{")
        fin = brut.rfind("}")
        if debut == -1 or fin == -1 or fin <= debut:
            return None
        try:
            return json.loads(brut[debut:fin + 1])
        except Exception:
            return None


# ==========================================
# CONFIGURATION RETRY / BACKOFF (429, 5xx) — commune à tous les providers
# ==========================================
RETRY_MAX_TENTATIVES = int(os.getenv("RETRY_MAX_TENTATIVES", 4))
RETRY_DELAI_BASE = float(os.getenv("RETRY_DELAI_BASE", 2))   # secondes
RETRY_DELAI_MAX = float(os.getenv("RETRY_DELAI_MAX", 60))    # secondes
STATUS_CODES_RETRYABLES = {429, 500, 502, 503, 504}


def attendre_avec_backoff(tentative):
    """Backoff exponentiel avec jitter. tentative commence à 0."""
    delai = min(RETRY_DELAI_BASE * (2 ** tentative), RETRY_DELAI_MAX)
    delai += random.uniform(0, 1)
    print(f"    Nouvelle tentative dans {delai:.1f}s (tentative {tentative + 1}/{RETRY_MAX_TENTATIVES})...")
    time.sleep(delai)


# ==========================================
# CONFIGURATION BATCH (soumission par lot, modèles distants payants uniquement)
# ==========================================
# BATCH_TAILLE_MAX : nombre max de requêtes par batch soumis. Les fournisseurs
# acceptent des batchs bien plus gros (100k chez Anthropic), mais on découpe en
# chunks plus petits pour limiter le "blast radius" d'un batch malformé et
# obtenir des résultats intermédiaires plus tôt.
BATCH_TAILLE_MAX = int(os.getenv("BATCH_TAILLE_MAX", 1000))
# Intervalle entre deux vérifications de statut d'un batch en cours.
BATCH_POLL_INTERVAL = float(os.getenv("BATCH_POLL_INTERVAL", 30))
# Délai max d'attente avant d'abandonner le polling (par défaut 24h, le SLO
# annoncé par Anthropic et Google pour leurs API batch).
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 24 * 3600))


def construire_custom_id(article_id, company_id):
    """
    Identifiant unique d'une requête au sein d'un batch, permettant de relier
    chaque résultat à l'article/entreprise d'origine (les résultats d'un batch
    ne reviennent pas forcément dans l'ordre de soumission).
    Format compatible avec les contraintes des deux fournisseurs (Anthropic:
    ^[a-zA-Z0-9_-]{1,64}$).
    """
    return f"a{article_id}-c{company_id}"


def parser_custom_id(custom_id):
    """Opération inverse de construire_custom_id(). Lève ValueError si le format est inattendu."""
    match = re.match(r"^a(\d+)-c(\d+)$", custom_id or "")
    if not match:
        raise ValueError(f"custom_id inattendu: {custom_id!r}")
    return int(match.group(1)), int(match.group(2))


def extraire_status_code_gemini(exception):
    """
    La lib google-genai n'expose pas toujours un status_code de façon uniforme
    selon les versions. On essaie plusieurs attributs connus, puis on retombe
    sur une recherche regex dans le message d'erreur.
    """
    for attribut in ("status_code", "code"):
        valeur = getattr(exception, attribut, None)
        if isinstance(valeur, int):
            return valeur

    reponse = getattr(exception, "response", None)
    if reponse is not None:
        valeur = getattr(reponse, "status_code", None)
        if isinstance(valeur, int):
            return valeur

    match = re.search(r"\b(429|500|502|503|504)\b", str(exception))
    if match:
        return int(match.group(1))
    return None
