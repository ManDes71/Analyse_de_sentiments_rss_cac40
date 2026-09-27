#!/usr/bin/env python3
"""
Entraîne un transformer à reproduire la note de sentiment, et le mesure contre
les annotations humaines.

Le but n'est PAS de battre les LLM : avec 740 exemples étiquetés par le
consensus (juste à 91,3 %), le modèle ne dépassera pas son maître. Le but est
d'obtenir un noteur GRATUIT et instantané qui l'imite honorablement, là où le
consensus demande trois appels de LLM dont deux payants.

Jeux (construits par preparer_corpus.py) :
  entraînement 740 — unanimes du périmètre, étiquette = note_consensus
  validation    90 — même source, pour l'arrêt anticipé
  test         191 — annotations HUMAINES, la seule vérité

⚠ Le jeu de test est volontairement biaisé vers les cas difficiles : il vient
des désaccords Gemini/Haiku (131) plus 60 cas d'accord tirés au hasard. Sur ce
même échantillon, le consensus lui-même ne fait que 67,5 %, llama 63,4 %,
Haiku 59,7 %. Ce sont EUX les points de comparaison, pas les 81 % du périmètre
entier — comparer à 81 % serait tricher.

Déséquilibre des classes (172 / 118 / 450) : la perte est pondérée par
l'inverse de la fréquence, sans quoi le modèle apprend à toujours répondre
« positif », ce qui lui donnerait déjà 61 % sur l'entraînement.

Boucle PyTorch classique plutôt que le Trainer de HuggingFace : accelerate
n'est pas installé, et une boucle explicite se lit mieux qu'une configuration.

Usage :
    cd tfidf
    .venv/bin/python entrainer_transformer.py
    .venv/bin/python entrainer_transformer.py --epochs 8 --modele camembert-base
    # courbe d'apprentissage : n'entraîner que sur 25 % des exemples
    .venv/bin/python entrainer_transformer.py --fraction 0.25 --graine 14 --sans-sauvegarde
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "output"
MODELE_DEFAUT = "camembert-base"
MAX_TOKENS = 512
GRAINE = 13


class CorpusNotes(Dataset):
    def __init__(self, exemples: list[dict], tokenizer):
        self.exemples = exemples
        self.encodages = tokenizer([e["texte"] for e in exemples], truncation=True,
                                   max_length=MAX_TOKENS, padding="max_length", return_tensors="pt")

    def __len__(self):
        return len(self.exemples)

    def __getitem__(self, i):
        e = self.exemples[i]
        # La cible d'apprentissage est la DISTRIBUTION des votes quand elle
        # existe (étiquettes douces), sinon l'étiquette dure en one-hot.
        # L'évaluation, elle, se fait toujours sur l'étiquette dure.
        cible = torch.tensor(e["distribution"], dtype=torch.float32) if "distribution" in e \
            else torch.nn.functional.one_hot(torch.tensor(e["label"]), 3).float()
        return ({k: v[i] for k, v in self.encodages.items()},
                torch.tensor(e["label"], dtype=torch.long), cible)


def charger(nom: str, jour: str, suffixe: str = "") -> list[dict]:
    chemin = SORTIE / f"corpus_{nom}{suffixe}_{jour}.jsonl"
    if not chemin.exists():
        sys.exit(f"{chemin} est introuvable. Lancer d'abord preparer_corpus.py.")
    return [json.loads(l) for l in open(chemin, encoding="utf-8")]


def sous_echantillon(exemples: list[dict], fraction: float, graine: int) -> list[dict]:
    """Garde `fraction` des exemples de chaque classe (proportions intactes).

    Les sous-ensembles sont EMBOÎTÉS pour une graine donnée : les 25 % sont
    inclus dans les 50 %, eux-mêmes dans les 75 %. La courbe d'apprentissage
    mesure ainsi l'effet d'exemples AJOUTÉS, pas d'un autre tirage.
    """
    if fraction >= 1:
        return exemples
    rng = np.random.default_rng(graine)
    garde = []
    for c in (0, 1, 2):
        classe = [e for e in exemples if e["label"] == c]
        ordre = rng.permutation(len(classe))
        garde += [classe[i] for i in ordre[:round(len(classe) * fraction)]]
    return garde


def evaluer(modele, chargeur, appareil) -> tuple[np.ndarray, np.ndarray]:
    modele.eval()
    predictions, verites = [], []
    with torch.no_grad():
        for lot, etiquettes, _ in chargeur:
            lot = {k: v.to(appareil) for k, v in lot.items()}
            logits = modele(**lot).logits
            predictions.append(logits.argmax(-1).cpu().numpy())
            verites.append(etiquettes.numpy())
    return np.concatenate(predictions), np.concatenate(verites)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modele", default=MODELE_DEFAUT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--jour", default=str(date.today()), help="date des fichiers corpus_*")
    parser.add_argument("--suffixe", default="", help="suffixe des fichiers corpus (ex : _tous)")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="part du jeu d'entraînement gardée (courbe d'apprentissage)")
    parser.add_argument("--graine", type=int, default=GRAINE)
    parser.add_argument("--sans-sauvegarde", action="store_true", help="n'enregistre pas le modèle")
    args = parser.parse_args()

    torch.manual_seed(args.graine)
    np.random.seed(args.graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.modele)

    jeux = {n: charger(n, args.jour, args.suffixe) for n in ("entrainement", "validation", "test")}
    jeux["entrainement"] = sous_echantillon(jeux["entrainement"], args.fraction, args.graine)
    for nom, ex in jeux.items():
        print(f"{nom:<13} {len(ex):>4} exemples | classes "
              f"{ {c: sum(1 for e in ex if e['label'] == c) for c in (0, 1, 2)} }")

    chargeurs = {n: DataLoader(CorpusNotes(ex, tokenizer), batch_size=args.batch,
                               shuffle=(n == "entrainement"))
                 for n, ex in jeux.items()}

    modele = AutoModelForSequenceClassification.from_pretrained(args.modele, num_labels=3).to(appareil)
    # Poids inversement proportionnels à la fréquence : sans eux, le modèle
    # répond « positif » partout (61 % de l'entraînement).
    effectifs = np.array([sum(1 for e in jeux["entrainement"] if e["label"] == c) for c in (0, 1, 2)])
    poids = torch.tensor(effectifs.sum() / (3 * effectifs), dtype=torch.float32, device=appareil)
    print(f"\npoids des classes (négatif, neutre, positif) : {poids.cpu().numpy().round(2)}")
    perte = torch.nn.CrossEntropyLoss(weight=poids)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=args.lr)

    meilleur, meilleur_etat = -1.0, None
    for epoque in range(1, args.epochs + 1):
        modele.train()
        total = 0.0
        for lot, _, cibles in chargeurs["entrainement"]:
            lot = {k: v.to(appareil) for k, v in lot.items()}
            cibles = cibles.to(appareil)
            optimiseur.zero_grad()
            sortie = perte(modele(**lot).logits, cibles)
            sortie.backward()
            optimiseur.step()
            total += sortie.item()
        pred, vrai = evaluer(modele, chargeurs["validation"], appareil)
        justesse = float((pred == vrai).mean())
        kappa = cohen_kappa_score(pred, vrai)
        marque = ""
        if kappa > meilleur:                       # kappa, pas justesse : les classes sont déséquilibrées
            meilleur, marque = kappa, "  ← meilleur"
            meilleur_etat = {k: v.detach().cpu().clone() for k, v in modele.state_dict().items()}
        print(f"époque {epoque}/{args.epochs} | perte {total/len(chargeurs['entrainement']):.3f} "
              f"| validation : justesse {justesse:.1%}, kappa {kappa:.2f}{marque}")

    if meilleur_etat:
        modele.load_state_dict(meilleur_etat)

    print("\n=== TEST, contre les 191 annotations humaines ===")
    pred, vrai = evaluer(modele, chargeurs["test"], appareil)
    print(f"justesse {float((pred == vrai).mean()):.1%} | kappa {cohen_kappa_score(pred, vrai):.2f}")
    # ligne unique, facile à relire quand plusieurs entraînements s'enchaînent
    print(f"RESULTAT fraction={args.fraction} graine={args.graine} n={len(jeux['entrainement'])} "
          f"justesse={float((pred == vrai).mean()):.4f} kappa={cohen_kappa_score(pred, vrai):.4f} "
          f"kappa_validation={meilleur:.4f}")
    print("\nrappel sur CE MÊME échantillon : consensus 67,5 %, llama 63,4 %, "
          "Haiku 59,7 %, Gemini 56,5 %")
    print("\n", classification_report(vrai, pred, target_names=["négatif", "neutre", "positif"],
                                      digits=3, zero_division=0))
    print("matrice de confusion (lignes = humain, colonnes = modèle) :")
    print(confusion_matrix(vrai, pred))

    if args.sans_sauvegarde:
        return 0
    dossier = SORTIE / f"transformer{args.suffixe}_{date.today()}"
    modele.save_pretrained(dossier)
    tokenizer.save_pretrained(dossier)
    print(f"\nmodèle enregistré → {dossier}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
