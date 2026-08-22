"""
Diagnostic des jobs batch Gemini.

Objectif : savoir si des jobs batch antérieurs occupent encore la file d'attente.
Tant qu'un job n'est pas dans un état terminal, ses jetons restent comptabilisés
dans le plafond "jetons mis en file d'attente par lot" (3 M pour Gemini 2.5 Flash
en Niveau 1), ce qui provoque un 429 RESOURCE_EXHAUSTED à la soumission suivante
alors qu'on croit n'avoir rien en cours.

Usage :
    python verif_limit_batch.py            # liste seulement
    python verif_limit_batch.py --annuler  # annule les jobs non terminaux
"""
import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# États terminaux : le job ne consomme plus de quota de file d'attente.
ETATS_TERMINAUX = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def main():
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY introuvable. Vérifiez le fichier .env "
              "(même variable que celle utilisée par evaluate_article.py).")
        return 1

    # Ne jamais afficher la clé en clair : juste de quoi vérifier qu'on a la bonne.
    print(f"Clé API chargée : {GOOGLE_API_KEY[:6]}...{GOOGLE_API_KEY[-4:]} "
          f"({len(GOOGLE_API_KEY)} caractères)")

    annuler = "--annuler" in sys.argv
    client = genai.Client(api_key=GOOGLE_API_KEY)

    try:
        jobs = list(client.batches.list())
    except Exception as e:
        print(f"\nÉchec de la récupération des jobs : {e}")
        return 1

    if not jobs:
        print("\nAucun job batch trouvé.")
        print("=> La file d'attente est vide : le 429 ne vient pas de jobs résiduels.")
        return 0

    print(f"\n{len(jobs)} job(s) trouvé(s) :\n")
    actifs = []
    for job in jobs:
        etat = job.state.name
        marqueur = "  " if etat in ETATS_TERMINAUX else "->"
        date_creation = getattr(job, "create_time", "")
        print(f"{marqueur} {etat:<22} {job.name}  {date_creation}")
        if etat not in ETATS_TERMINAUX:
            actifs.append(job)

    if not actifs:
        print("\nTous les jobs sont dans un état terminal : ils n'occupent plus la file.")
        print("=> Le 429 vient donc du volume de la soumission elle-même, pas de jobs résiduels.")
        return 0

    print(f"\n{len(actifs)} job(s) NON terminal(aux) occupent encore votre quota de file d'attente.")

    if not annuler:
        print("Relancez avec --annuler pour les annuler et libérer la file :")
        print("    python verif_limit_batch.py --annuler")
        return 0

    for job in actifs:
        try:
            client.batches.cancel(name=job.name)
            print(f"  Annulé : {job.name}")
        except Exception as e:
            print(f"  Échec de l'annulation de {job.name} : {e}")

    print("\nAnnulations demandées. Attendez ~1 minute que les états se propagent, "
          "puis relancez ce script sans --annuler pour vérifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())