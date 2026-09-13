"""Tests de la calibration — numpy et torch CPU seulement, pas de dataset."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from claimsight.training.calibrate import (  # noqa: E402
    best_thresholds,
    expected_calibration_error,
    fit_temperature,
)


# ------------------------------------------------------------- température
def test_temperature_above_one_for_an_overconfident_model():
    """Des logits trop écartés pour la vérité observée => T > 1."""
    torch.manual_seed(0)
    logits = torch.tensor([[6.0, -6.0], [6.0, -6.0], [6.0, -6.0], [-6.0, 6.0]])
    targets = torch.tensor([0, 0, 1, 1])  # une erreur franche malgré la confiance
    assert fit_temperature(logits, targets, multilabel=False) > 1.0


def test_temperature_is_positive_for_multilabel():
    logits = torch.tensor([[4.0, -4.0], [3.0, -3.0], [-2.0, 2.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    assert fit_temperature(logits, targets, multilabel=True) > 0.0


def test_temperature_never_changes_the_ranking():
    """Diviser par T > 0 ne réordonne rien: l'accuracy est inchangée."""
    logits = torch.randn(32, 4, generator=torch.Generator().manual_seed(1))
    for t in (0.5, 1.0, 2.5):
        assert torch.equal(logits.argmax(1), (logits / t).argmax(1))


# --------------------------------------------------------------------- ECE
def test_ece_is_zero_for_a_perfectly_calibrated_model():
    probs = np.full(100, 0.7)
    correct = np.zeros(100, dtype=bool)
    correct[:70] = True  # annonce 70 %, a raison 70 % du temps
    assert expected_calibration_error(probs, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_grows_with_overconfidence():
    probs = np.full(100, 0.99)
    correct = np.zeros(100, dtype=bool)
    correct[:50] = True  # annonce 99 %, a raison une fois sur deux
    assert expected_calibration_error(probs, correct) == pytest.approx(0.49, abs=0.01)


# ------------------------------------------------------------------ seuils
def test_threshold_lands_between_the_two_populations():
    probs = np.array([[0.1], [0.2], [0.8], [0.9]])
    targets = np.array([[0], [0], [1], [1]])
    thr = best_thresholds(probs, targets, ["face"])["face"]
    assert 0.2 < thr <= 0.8


def test_rare_class_gets_a_lower_threshold_than_uniform():
    """Une classe rare gagne à être déclarée présente plus tôt."""
    rng = np.random.default_rng(0)
    probs = np.concatenate([rng.uniform(0.0, 0.25, 95), rng.uniform(0.3, 0.6, 5)]).reshape(-1, 1)
    targets = np.concatenate([np.zeros(95), np.ones(5)]).astype(int).reshape(-1, 1)
    assert best_thresholds(probs, targets, ["rare"])["rare"] < 0.5


def test_one_threshold_per_class():
    probs = np.random.default_rng(2).random((50, 4))
    targets = (probs > 0.5).astype(int)
    thresholds = best_thresholds(probs, targets, ["a", "b", "c", "d"])
    assert set(thresholds) == {"a", "b", "c", "d"}
    assert all(0.0 < v < 1.0 for v in thresholds.values())
