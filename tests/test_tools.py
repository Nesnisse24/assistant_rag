"""Tests des outils de calcul : python -m pytest tests -v"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import (  # noqa: E402
    indemnite_licenciement, preavis_demission, preavis_licenciement, salaire_reference,
)


#  préavis de licenciement (L1234-1)
@pytest.mark.parametrize("mois, attendu", [
    (0, None), (5, None),      # < 6 mois : pas de minimum légal
    (6, 1), (23, 1),           # 6 mois à moins de 2 ans : 1 mois
    (24, 2), (120, 2),         # 2 ans et plus : 2 mois
])
def test_preavis_licenciement(mois, attendu):
    r = preavis_licenciement(mois)
    assert r["mois"] == attendu
    assert r["article"] == "L1234-1"


def test_preavis_licenciement_negatif():
    with pytest.raises(ValueError):
        preavis_licenciement(-1)


def test_preavis_demission():
    r = preavis_demission()
    assert r["mois"] is None and r["article"] == "L1237-1"


#  indemnité de licenciement (L1234-9, R1234-2) 
@pytest.mark.parametrize("mois", [0, 7])
def test_indemnite_sous_seuil_8_mois(mois):
    assert indemnite_licenciement(mois, 2000)["montant"] == 0.0


def test_indemnite_seuil_exactement_8_mois():
    # 8 mois = 2/3 d'année x 1/4 = 1/6 de mois de salaire
    assert indemnite_licenciement(8, 2000)["montant"] == pytest.approx(333.33, abs=0.01)


@pytest.mark.parametrize("mois, salaire, attendu", [
    (60, 2000, 2500.00),        # 5 ans : 5 x 1/4 = 1,25 mois
    (54, 2400, 2700.00),        # 4 ans 6 mois : prorata des mois
    (120, 1800, 4500.00),       # exactement 10 ans : 2,5 mois, pas de 2e tranche
    (144, 3000, 9500.00),       # 12 ans : 10 x 1/4 + 2 x 1/3
    (180, 2500, 10416.67),      # 15 ans : 10 x 1/4 + 5 x 1/3
])
def test_indemnite_montants(mois, salaire, attendu):
    assert indemnite_licenciement(mois, salaire)["montant"] == pytest.approx(attendu, abs=0.01)


def test_indemnite_inaptitude_professionnelle_doublee():
    normal = indemnite_licenciement(120, 2000)
    double = indemnite_licenciement(120, 2000, inaptitude_professionnelle=True)
    assert double["montant"] == pytest.approx(2 * normal["montant"])
    assert "L1226-14" in double["articles"]
    assert "L1226-14" not in normal["articles"]


def test_indemnite_articles_cites():
    assert indemnite_licenciement(60, 2000)["articles"][:2] == ["L1234-9", "R1234-2"]


@pytest.mark.parametrize("mois, salaire", [(-1, 2000), (60, -100)])
def test_indemnite_valeurs_negatives(mois, salaire):
    with pytest.raises(ValueError):
        indemnite_licenciement(mois, salaire)


# salaire de référence (R1234-4) 
def test_salaire_reference_constant():
    assert salaire_reference([2000.0] * 12) == 2000.0


def test_salaire_reference_prend_le_plus_favorable():
    # 12 mois : (9 x 2000 + 3 x 3000) / 12 = 2250 ; 3 derniers mois : 3000
    assert salaire_reference([2000.0] * 9 + [3000.0] * 3) == 3000.0
    # baisse récente : la moyenne sur 12 mois est plus favorable
    assert salaire_reference([3000.0] * 9 + [2000.0] * 3) == 2750.0


def test_salaire_reference_prime_annuelle():
    # 12 x 2000 + 1200 de prime = 25200 / 12 = 2100 ; 3 mois : (6000 + 300) / 3 = 2100
    assert salaire_reference([2000.0] * 12, prime_annuelle=1200) == 2100.0


def test_salaire_reference_vide():
    with pytest.raises(ValueError):
        salaire_reference([])