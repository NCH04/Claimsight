"""Tests de l'agrégation — logique métier pure, sans torch ni dataset."""

import pytest

from claimsight.domain.aggregation import (
    aggregate_damage,
    aggregate_severity,
    build_findings,
    global_confidence,
    suspect_total_loss,
    worst_finding,
)
from claimsight.domain.types import Prediction as P


def findings(*triples):
    """(damage, d_conf, severity, s_conf) -> liste de Finding."""
    names = [f"img{i}.jpg" for i in range(len(triples))]
    dmg = [P(d, dc) for d, dc, _, _ in triples]
    sev = [P(s, sc) for _, _, s, sc in triples]
    return build_findings(names, dmg, sev)


# --------------------------------------------------------------- le bug historique
def test_confident_none_does_not_beat_less_confident_damage():
    """Régression: c'était LE faux négatif critique de la V1.

    `max(preds, key=confidence)` faisait gagner un `none` à 0.99 contre un
    `dent` à 0.62 — un véhicule accidenté ressortait « aucun dégât ».
    """
    f = findings(("none", 0.99, "none", 0.99), ("dent", 0.62, "moderate", 0.60))
    assert aggregate_damage(f).label == "dent"
    assert aggregate_severity(f).label == "moderate"


def test_single_damaged_photo_outweighs_many_clean_ones():
    f = findings(
        *[("none", 0.97, "none", 0.95)] * 5,
        ("crack", 0.55, "severe", 0.51),
    )
    assert aggregate_damage(f).label == "crack"
    assert aggregate_severity(f).label == "severe"


def test_severity_is_the_worst_not_the_most_confident():
    f = findings(("scratch", 0.9, "minor", 0.95), ("dent", 0.7, "severe", 0.55))
    assert aggregate_severity(f).label == "severe"


# --------------------------------------------------------------- cohérence du couple
def test_damage_and_severity_come_from_the_same_image():
    """L'ancienne version agrégeait les deux indépendamment: le rapport
    pouvait mélanger le dégât d'une photo et la gravité d'une autre."""
    f = findings(("scratch", 0.95, "minor", 0.9), ("dent", 0.60, "severe", 0.58))
    worst = worst_finding(f)
    assert worst.damage.label == "dent"
    assert worst.severity.label == "severe"
    assert worst.filename == "img1.jpg"


def test_finding_confidence_is_bounded_by_weakest_link():
    f = findings(("dent", 0.9, "moderate", 0.4))
    assert worst_finding(f).confidence() == pytest.approx(0.4)


def test_unknown_components_are_ignored_not_counted_as_zero():
    # Modèle severity absent: la confiance ne doit pas être écrasée à 0.
    f = findings(("dent", 0.88, "unknown", 0.0))
    assert worst_finding(f).confidence() == pytest.approx(0.88)


# --------------------------------------------------------------- cas dégénérés
def test_all_clean_returns_none():
    f = findings(("none", 0.9, "none", 0.8), ("none", 0.7, "none", 0.6))
    assert aggregate_damage(f).label == "none"
    assert aggregate_severity(f).label == "none"


def test_all_unknown_returns_unknown():
    f = findings(("unknown", 0.0, "unknown", 0.0))
    assert aggregate_damage(f).label == "unknown"
    assert aggregate_severity(f).label == "unknown"


def test_empty_folder():
    assert aggregate_damage([]).label == "unknown"
    assert aggregate_severity([]).label == "unknown"
    assert worst_finding([]) is None


# --------------------------------------------------------------- perte totale
def test_total_loss_needs_several_severe_images():
    assert not suspect_total_loss(findings(("dent", 0.9, "severe", 0.9)), 3)
    assert suspect_total_loss(findings(*[("dent", 0.9, "severe", 0.9)] * 3), 3)


# --------------------------------------------------------------- confiance globale
WEIGHTS = {"view": 0.5, "damage": 0.5}


def test_confidence_is_not_capped_when_damage_model_is_missing():
    """Régression: `0.5*vue + 0.5*0.0` plafonnait le score à 0.5 même avec
    des vues parfaites, tant que le modèle de dégât n'était pas entraîné."""
    assert global_confidence([0.9, 0.9], None, WEIGHTS) == pytest.approx(0.9)


def test_confidence_blends_both_components_when_available():
    assert global_confidence([0.8, 0.8], 0.6, WEIGHTS) == pytest.approx(0.7)


def test_confidence_without_any_component():
    assert global_confidence([], None, WEIGHTS) == 0.0


def test_confidence_stays_within_bounds():
    assert 0.0 <= global_confidence([1.0], 1.0, WEIGHTS) <= 1.0
