"""Outils de calcul JuriRH : préavis et indemnité légale de licenciement.

Référence : Code du travail, L1234-1 (préavis), L1234-9 et R1234-2 (indemnité),
R1234-4 (salaire de référence). Résultat indicatif : une convention collective
peut prévoir des montants plus favorables.
"""

SEUIL_ANCIENNETE_INDEMNITE = 8  # mois (L1234-9)


def salaire_reference(salaires_mensuels: list[float], prime_annuelle: float = 0.0) -> float:
    """Salaire de référence (R1234-4) : le plus favorable entre
    - la moyenne des 12 derniers mois,
    - la moyenne des 3 derniers mois (prime annuelle prise au prorata de 3/12).

    salaires_mensuels : salaires bruts mensuels HORS prime annuelle, du plus ancien
    au plus récent (jusqu'à 12 valeurs ; moins si l'ancienneté est inférieure à 12 mois).
    prime_annuelle : montant d'une prime annuelle ou exceptionnelle versée sur la période.
    """
    if not salaires_mensuels:
        raise ValueError("Il faut au moins un salaire mensuel.")
    s12 = salaires_mensuels[-12:]
    s3 = s12[-3:]
    moyenne_12 = (sum(s12) + prime_annuelle) / len(s12)
    moyenne_3 = (sum(s3) + prime_annuelle * len(s3) / 12) / len(s3)
    return round(max(moyenne_12, moyenne_3), 2)


def preavis_licenciement(anciennete_mois: int) -> dict:
    """Durée minimale légale du préavis de licenciement (L1234-1)."""
    if anciennete_mois < 0:
        raise ValueError("L'ancienneté ne peut pas être négative.")
    if anciennete_mois < 6:
        return {"mois": None, "article": "L1234-1",
                "detail": "Moins de 6 mois d'ancienneté : durée fixée par la convention "
                          "collective ou les usages (pas de minimum légal)."}
    if anciennete_mois < 24:
        return {"mois": 1, "article": "L1234-1",
                "detail": "Entre 6 mois et 2 ans d'ancienneté : 1 mois."}
    return {"mois": 2, "article": "L1234-1",
            "detail": "2 ans d'ancienneté ou plus : 2 mois."}


def preavis_demission() -> dict:
    """Pas de durée légale générale : elle dépend de la convention collective ou des usages (L1237-1)."""
    return {"mois": None, "article": "L1237-1",
            "detail": "Durée fixée par la loi, la convention ou l'accord collectif, "
                      "ou à défaut par les usages. Consultez votre convention collective."}


def indemnite_licenciement(anciennete_mois: int, salaire_ref: float,
                           inaptitude_professionnelle: bool = False) -> dict:
    """Indemnité légale minimale de licenciement (R1234-2).

    1/4 de mois par année jusqu'à 10 ans, 1/3 de mois par année au-delà,
    années incomplètes au prorata des mois. Doublée en cas d'inaptitude
    d'origine professionnelle avec impossibilité de reclassement (L1226-14).
    Non applicable en cas de faute grave ou lourde (non géré ici).
    """
    if anciennete_mois < 0 or salaire_ref < 0:
        raise ValueError("Ancienneté et salaire doivent être positifs.")
    if anciennete_mois < SEUIL_ANCIENNETE_INDEMNITE:
        return {"montant": 0.0, "articles": ["L1234-9"],
                "detail": f"Moins de {SEUIL_ANCIENNETE_INDEMNITE} mois d'ancienneté : "
                          "pas d'indemnité légale de licenciement."}

    annees = anciennete_mois / 12
    tranche_1 = min(annees, 10)
    tranche_2 = max(annees - 10, 0)
    mois_de_salaire = tranche_1 / 4 + tranche_2 / 3
    montant = mois_de_salaire * salaire_ref
    if inaptitude_professionnelle:
        montant *= 2

    return {
        "montant": round(montant, 2),
        "mois_de_salaire": round(mois_de_salaire, 4),
        "articles": ["L1234-9", "R1234-2"] + (["L1226-14"] if inaptitude_professionnelle else []),
        "detail": (f"{tranche_1:.2f} an(s) x 1/4 + {tranche_2:.2f} an(s) x 1/3 "
                   f"= {mois_de_salaire:.2f} mois x {salaire_ref} EUR"
                   + (" (doublé : inaptitude professionnelle)" if inaptitude_professionnelle else "")),
    }