"""Tests de la couverture photo multi-label — sans torch."""

import numpy as np
import pytest

from claimsight.domain.taxonomy import (
    COVERAGE_FACES,
    VIEW_CLASSES,
    VIEW_TO_FACES,
    faces_of_view,
    mirror_faces,
)
from claimsight.pipeline.missing_photos import detect_missing, missing_from_faces
from claimsight.training.metrics import compute_multilabel_metrics, multilabel_report


# ---------------------------------------------------------------- taxonomie
def test_every_view_maps_to_faces():
    assert set(VIEW_TO_FACES) == set(VIEW_CLASSES)


def test_diagonal_views_cover_two_faces():
    """La raison d'être du multi-label: une diagonale documente deux faces."""
    assert set(faces_of_view("front-left")) == {"front", "left"}
    assert set(faces_of_view("rear-right")) == {"rear", "right"}


def test_non_directional_views_cover_nothing():
    assert faces_of_view("closeup") == ()
    assert faces_of_view("out_of_scope") == ()


def test_mirroring_swaps_sides_and_is_involutive():
    assert set(mirror_faces(("front", "left"))) == {"front", "right"}
    for faces in (("front",), ("left", "rear"), COVERAGE_FACES):
        assert set(mirror_faces(mirror_faces(faces))) == set(faces)


# ---------------------------------------------------------------- couverture
def test_missing_from_faces_lists_uncovered():
    assert missing_from_faces({"front", "rear"}) == ["left", "right"]
    assert missing_from_faces(COVERAGE_FACES) == []
    assert missing_from_faces([]) == list(COVERAGE_FACES)


def test_two_diagonal_photos_cover_everything():
    """Deux diagonales opposées suffisent — ce que la voie multi-label permet
    d'exprimer et que quatre classes exclusives ne permettaient pas."""
    covered = set(faces_of_view("front-left")) | set(faces_of_view("rear-right"))
    assert missing_from_faces(covered) == []


def test_both_paths_agree_on_the_same_photos():
    """La voie couverture et l'ancienne voie `vues` doivent conclure pareil."""
    views = ["front", "rear-left"]
    covered = {f for v in views for f in faces_of_view(v)}
    assert missing_from_faces(covered) == detect_missing(views)


# ---------------------------------------------------------------- métriques
def test_exact_match_is_stricter_than_hamming():
    y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    y_pred = np.array([[1, 0, 1, 0], [0, 1, 0, 0]])  # une face ratée sur deux images
    m = compute_multilabel_metrics(y_true, y_pred, COVERAGE_FACES)
    assert m["exact_match"] == 0.5
    assert m["hamming_accuracy"] == pytest.approx(7 / 8)


def test_per_face_f1_is_reported():
    y = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    m = compute_multilabel_metrics(y, y, COVERAGE_FACES)
    for face in COVERAGE_FACES:
        assert f"f1_{face}" in m
    assert m["exact_match"] == 1.0


def test_report_counts_support_per_face():
    y_true = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]])
    rep = multilabel_report(y_true, y_true, COVERAGE_FACES)["per_face"]
    assert rep["front"]["support"] == 2
    assert rep["right"]["support"] == 0
    assert rep["front"]["f1"] == 1.0
