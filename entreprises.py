"""
Reconnaissance du nom d'une entreprise dans un texte.

Module léger (ni torch ni sklearn), partagé par classifier.py, perimetre.py et
evaluate_article.py : il vivait dans classifier.py, dont l'import charge torch
et transformers — prohibitif pour un simple filtre de périmètre.

La raison sociale en base ne s'écrit pas comme dans la presse : suffixe de
cotation (« BNP PARIBAS ACT.A »), forme juridique (« Teleperformance SE »),
troncature de l'export boursier (« VEOLIA ENVIRON. » pour Veolia
Environnement). Chercher le nom exact laissait 315 des 608 lignes en
sentinelle 8 alors que l'entreprise était bel et bien citée.

Les règles ci-dessous traitent le régulier (suffixes, accents, ponctuation) ;
les alias traitent l'idiosyncrasique, qu'aucune règle ne devinerait.
"""

import json
import re
import unicodedata
from functools import lru_cache

CHEMIN_ALIAS_JSON = "output/alias.json"

# Suffixes retirés en fin de raison sociale. Liste FERMÉE, calibrée sur les 33
# entreprises réelles. Surtout pas de règle « garde le premier mot » :
# « AIR LIQUIDE » deviendrait « AIR », qui matche « Air France », « airbag »,
# « Airbus » — le bug de bornes de mots que CLAUDE.md signale comme récurrent.
_SUFFIXES_SOCIETE = r"(?:ACT\.?\s*[A-Z]?|SAS|SCA|SA|SE|GROUPES?|GROUP|RG|BS|DR|PROMESSES)"

# Périphrases d'alias.json trop génériques pour ancrer un contexte : « le groupe »
# ou « le groupe français » peuvent désigner n'importe quelle entreprise et
# apparaissent dans n'importe quel article.
#
# La règle : une périphrase est rejetée si, déterminants retirés, il ne lui reste
# que des mots passe-partout. « le groupe français » → {groupe, francais}, rejetée.
# « le géant français des gaz » → contient « gaz », gardée. « l'avionneur » →
# contient « avionneur », gardée malgré sa brièveté. La longueur serait un mauvais
# critère : les meilleures périphrases sont souvent les plus courtes.
_MOTS_VIDES = {
    "le", "la", "l", "les", "un", "une", "de", "des", "du", "d", "aux", "au", "a",
}
_MOTS_GENERIQUES = {
    "groupe", "groupes", "geant", "geants", "societe", "societes", "entreprise",
    "entreprises", "firme", "boite", "compagnie",
    "francais", "francaise", "allemand", "allemande", "europeen", "europeenne",
    "italien", "italienne", "americain", "americaine", "britannique", "espagnol",
    "espagnole", "neerlandais", "suisse", "belge", "mondial", "mondiale",
    "international", "internationale", "national", "nationale",
}


def _est_periphrase_generique(alias: str) -> bool:
    """L'alias se réduit-il à des mots passe-partout ?"""
    mots = [m for m in normaliser(alias).split() if m not in _MOTS_VIDES]
    return bool(mots) and all(m in _MOTS_GENERIQUES for m in mots)

# Remplis une fois par charger_alias(), lu par variantes_entreprise().
_ALIAS_BDD: dict[int, tuple[str, ...]] = {}
_ALIAS_JSON: dict[str, tuple[str, ...]] = {}


def normaliser(texte: str) -> str:
    """Minuscules, sans accents, ponctuation réduite à des espaces.

    Appliquée à l'identique au motif et au texte cherché, pour que
    « Séché Environnement » retrouve « seche environnement ».
    """
    texte = unicodedata.normalize("NFD", texte or "")
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", texte.lower())).strip()


def charger_alias(cur, chemin_json: str = CHEMIN_ALIAS_JSON) -> None:
    """Charge les deux sources d'alias en mémoire, une fois pour toutes.

    - table company_aliases : variantes orthographiques (« GTT »)
    - output/alias.json     : fichier MIXTE, périphrases descriptives ET vraies
                              variantes de nom (« veolia » pour VEOLIA ENVIRON.)

    Vide le cache de variantes_entreprise(), qui lit ces dictionnaires.
    """
    _ALIAS_BDD.clear()
    _ALIAS_JSON.clear()

    cur.execute("SELECT company_id, alias FROM public.company_aliases")
    par_id: dict[int, list[str]] = {}
    for row in cur.fetchall():
        cid = row["company_id"] if isinstance(row, dict) else row[0]
        alias = row["alias"] if isinstance(row, dict) else row[1]
        par_id.setdefault(cid, []).append(alias)
    _ALIAS_BDD.update({cid: tuple(v) for cid, v in par_id.items()})

    try:
        with open(chemin_json, encoding="utf-8") as f:
            brut = json.load(f)
        _ALIAS_JSON.update({
            str(nom).strip().upper(): tuple(alias)
            for nom, alias in brut.items() if isinstance(alias, list)
        })
    except FileNotFoundError:
        print(f"Attention : {chemin_json} introuvable. On continue sans ses alias.")

    variantes_entreprise.cache_clear()
    print(f"Alias chargés : {len(_ALIAS_BDD)} entreprises en base, "
          f"{len(_ALIAS_JSON)} dans {chemin_json}.")


@lru_cache(maxsize=None)
def variantes_entreprise(company_id: int, company_name: str) -> re.Pattern:
    """Motif reconnaissant toutes les graphies sous lesquelles l'entreprise
    peut être citée.

    Trois sources fusionnées et dédoublonnées :
      1. la raison sociale, telle quelle et débarrassée de son suffixe
      2. la table company_aliases
      3. output/alias.json, moins les périphrases génériques

    Le motif est borné par \\b et ordonne les variantes de la plus longue à la
    plus courte : `re` retient la première alternative qui matche, pas la plus
    longue, et « bnp paribas easy stoxx » doit gagner sur « bnp paribas ».

    Mémoïsée (33 entreprises pour ~3000 articles) ; le cache lit les
    dictionnaires de module, donc charger_alias() le vide.
    """
    variantes = {company_name}

    sans_suffixe = re.sub(rf"\s+{_SUFFIXES_SOCIETE}\s*$", "", company_name,
                          flags=re.IGNORECASE).strip(" .-")
    # Garde-fou : ne jamais réduire à un fragment trop court ou à un seul mot
    # issu d'un nom composé.
    if len(sans_suffixe) >= 4 and " " in company_name:
        variantes.add(sans_suffixe)

    variantes.update(_ALIAS_BDD.get(company_id, ()))
    variantes.update(
        a for a in _ALIAS_JSON.get(company_name.upper(), ())
        if not _est_periphrase_generique(a)
    )

    formes = sorted({normaliser(v) for v in variantes if normaliser(v)},
                    key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(f) for f in formes) + r")\b")


def citee_dans(motif: re.Pattern, texte: str) -> bool:
    """L'entreprise est-elle citée dans ce texte ?"""
    return bool(motif.search(normaliser(texte)))


def compter_citations(motif: re.Pattern, texte: str) -> int:
    """Nombre de mentions de l'entreprise dans ce texte."""
    return len(motif.findall(normaliser(texte)))
