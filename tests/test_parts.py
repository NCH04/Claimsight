"""Tests de l'agrégation par pièce — domaine pur, sans torch ni GPU."""

import pytest

from claimsight.domain.parts import (
    PartObservation,
    aggregate_parts,
    structural_parts_damaged,
)
from claimsight.domain.types import Prediction as P


def obs(part, image, det_conf, damage, dmg_conf, severity, faces=()):
    return PartObservation(part, image, det_conf, P(damage, dmg_conf),
                           P(severity, 0.8), tuple(faces))


# ------------------------------------------------------------ règle centrale
def test_worst_observation_wins_across_images():
    """Une pièce saine de face et enfoncée de trois quarts est endommagée."""
    entries = aggregate_parts([
        obs("front bumper", "a.jpg", 0.9, "none", 0.97, "none", ["front"]),
        obs("front bumper", "b.jpg", 0.8, "dent", 0.62, "moderate", ["front", "left"]),
    ])
    assert len(entries) == 1
    assert entries[0]["damage"] == "dent"
    assert entries[0]["severity"] == "moderate"


def test_undamaged_parts_are_not_reported():
    """Un constat de sinistre ne liste pas les pièces intactes."""
    entries = aggregate_parts([
        obs("hood", "a.jpg", 0.95, "none", 0.9, "none"),
        obs("trunk", "a.jpg", 0.9, "unknown", 0.0, "unknown"),
    ])
    assert entries == []


def test_views_evidence_unions_every_sighting():
    entries = aggregate_parts([
        obs("hood", "a.jpg", 0.9, "scratch", 0.7, "minor", ["front"]),
        obs("hood", "b.jpg", 0.8, "none", 0.9, "none", ["front", "left"]),
    ])
    assert entries[0]["views_evidence"] == ["front", "left"]
    assert entries[0]["detected_on"] == ["a.jpg", "b.jpg"]


# ------------------------------------------------------------ confiance
def test_confidence_combines_detection_and_damage():
    """Une pièce mal détectée rend son diagnostic peu crédible."""
    entries = aggregate_parts([obs("hood", "a.jpg", 0.5, "dent", 0.8, "moderate")])
    assert entries[0]["confidence"] == pytest.approx(0.40)


def test_unknown_damage_yields_zero_confidence():
    o = obs("hood", "a.jpg", 0.99, "unknown", 0.0, "unknown")
    assert o.confidence() == 0.0


# ------------------------------------------------------------ tri et seuils
def test_entries_are_sorted_worst_first():
    entries = aggregate_parts([
        obs("hood", "a.jpg", 0.9, "scratch", 0.8, "minor"),
        obs("front bumper", "a.jpg", 0.9, "crack", 0.8, "severe"),
        obs("trunk", "a.jpg", 0.9, "dent", 0.8, "moderate"),
    ])
    assert [e["name"] for e in entries] == ["front bumper", "trunk", "hood"]


def test_structural_damage_counts_only_severe_parts():
    entries = aggregate_parts([
        obs("front bumper", "a.jpg", 0.9, "crack", 0.8, "severe"),
        obs("hood", "a.jpg", 0.9, "dent", 0.8, "moderate"),
        obs("trunk", "a.jpg", 0.9, "crack", 0.8, "severe"),
    ])
    assert structural_parts_damaged(entries) == 2
    assert structural_parts_damaged(entries, "minor") == 3


def test_empty_input():
    assert aggregate_parts([]) == []
    assert structural_parts_damaged([]) == 0
