"""Tests de la taxonomie — aucune dépendance torch."""

import pytest

from claimsight.domain import taxonomy as tx


def test_mirror_map_covers_all_views():
    assert set(tx.VIEW_MIRROR_MAP) == set(tx.VIEW_CLASSES)


def test_mirror_map_is_an_involution():
    # Flipper deux fois doit rendre le label d'origine, sinon l'augmentation
    # corromprait silencieusement les labels gauche/droite.
    for view in tx.VIEW_CLASSES:
        assert tx.VIEW_MIRROR_MAP[tx.VIEW_MIRROR_MAP[view]] == view


def test_part_ids_are_contiguous_and_stable():
    # L'index dans PART_CLASSES est l'id de classe YOLO: il est contractuel.
    assert list(tx.PART_TO_ID.values()) == list(range(len(tx.PART_CLASSES)))
    assert tx.ID_TO_PART[0] == "front bumper"


def test_excluded_parts_are_not_in_v1():
    assert not set(tx.EXCLUDED_PARTS_V1) & set(tx.PART_CLASSES)


def test_severity_is_strictly_ordered():
    ranks = [tx.severity_rank(s) for s in tx.SEVERITY_CLASSES]
    assert ranks == sorted(ranks) == [0, 1, 2, 3]


def test_unknown_is_not_a_learned_class():
    # `unknown` est une sentinelle d'abstention, jamais une classe apprise.
    for classes in (tx.VIEW_CLASSES, tx.DAMAGE_CLASSES, tx.SEVERITY_CLASSES):
        assert tx.UNKNOWN not in classes


@pytest.mark.parametrize("label,expected", [("severe", 3), ("none", 0), ("unknown", -1)])
def test_severity_rank(label, expected):
    assert tx.severity_rank(label) == expected
